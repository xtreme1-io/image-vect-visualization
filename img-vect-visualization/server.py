# coding='utf-8'
"""
t-SNE对图片向量进行可视化
可视化的数据量不能超过10W，否则可能永远看不到结果
处理过程中异常的样本将会被丢弃，所以，返回的向量数、点集数可能小于数据集中的图片总数
"""
import os
from datetime import datetime
import numpy as np
import minio
import json
import codecs
import cv2
import argparse
import pickle
import flask
from flask import Flask
from flask_cors import CORS
import tornado.wsgi
import tornado.httpserver
from concurrent.futures import ThreadPoolExecutor
from openTSNE import TSNE

import torch
from torchvision.models import *
app = Flask(__name__)
CORS(app, supports_credentials=True)
  
  
def _image_to_vect(img_list, dataset_id, task_id):
    '''
    convert a list of images into vect tensor. 
    :param img_list:list of images in dataset_id
    :type img_list:list({u"img_id":image_id, u"img_path":image_path})
    :param dataset_id:dataset where image_list belong to 
    :type dataset_id:long
    :param task_id:id for current task
    :type task_id:string
    '''
    file_vec = []
    imgid_list = []
    failed_imgids = []
    with torch.no_grad():
        for idx, item in enumerate(img_list):
            print (u"task :", task_id, u"  dataset:", dataset_id, u"  id :", idx, 
                   u"   file :", item[u"img_id"], u"  ", datetime.now().isoformat())
            
            try:
                resp = client.get_object(bucket_name=u"x1-community", 
                                    object_name=item[u"img_path"])
                img_bgr = cv2.imdecode(np.fromstring(resp.data, np.uint8), 1) 
            except:
                failed_imgids.append(item[u"img_id"])
                continue

            h, w, _ = img_bgr.shape
            ratio = max(h,w)/1280
            if ratio > 1.:
                img_bgr = cv2.resize(img_bgr, (int(w/ratio), int(h/ratio)))
      
            img_rgb = img_bgr[:,:,::-1]
            image = img_rgb.transpose((2, 0, 1)).astype(np.float32)#[H,W,C]-->[C,H,W]
            # normalize image
            image /= 255.
            image = torch.from_numpy(image).sub_(mean).div_(std)[None, ...]
            
            pred = model(image.to(torch.device(opt.device)))
            
            if pred.size()[-1] > opt.vect_dim:
                pred = pred[:,:opt.vect_dim]
      
            pred = pred.view(-1).cpu()
            file_vec.append(pred.numpy())
            imgid_list.append(item[u"img_id"])
      
    np_img_vec = np.array(file_vec)
    return np_img_vec, imgid_list, failed_imgids

 
CORS(app, resources=r'/*')
@app.route('/api/v1/calcSimilarity', methods=['POST'])
def imgset_2_points():
    request_data = json.loads(flask.request.get_data().decode('utf-8'))
    dataset_id = request_data[u"datasetId"]
    task_id = request_data[u"serialNumber"]
    data_info_file = request_data[u"filePath"]
    cal_type = request_data[u"type"]
    
    executor.submit(cvt_imgset_2_points, data_info_file, cal_type, dataset_id, task_id)
    return json.dumps({"code":"OK", "data":"", "message":""})
    
    
def _del_img_points(data_info, local_img_vect_path, local_point_set_path, local_point_set_json_path):
    old_imgvect = np.load(local_img_vect_path)
    old_pointset = np.load(local_point_set_path)
    flag = np.ones((old_pointset.shape[0]), dtype=np.bool)#默认所有点都要保留
    
    with codecs.open(local_point_set_json_path, u"rb", u"utf-8") as f:
        img_point_dic = json.load(f)
        imgid_id = {str(item[u"id"]).strip():idx for idx, item in enumerate(img_point_dic)}
    
    for item in data_info[u"deletedIds"]:
        img_id_str = str(item).strip()
        if img_id_str in imgid_id:
            flag[imgid_id[img_id_str]] = False
            del imgid_id[img_id_str]
    
    imgvect = old_imgvect[flag]
    pointset = old_pointset[flag]
    
    #如果图片已经存在于当前点集中，那么，将从本次新增的图片列表中被排除
    add_img = [item for item in data_info[u"addData"] if str(item[u"id"]).strip() not in imgid_id]
    
    #按照向量中的顺序，排序imgid
    sorted_imgid_id = sorted(imgid_id.items(), key=lambda item:item[1], reverse=False)
    sorted_imgid = [item[0] for item in sorted_imgid_id]
    
    #样本数应该相等
    print (u"img vect :", imgvect.shape, u"   point set :", pointset.shape, u"  imgid size :", len(imgid_id))
    return imgvect, pointset, sorted_imgid, add_img


def fput_objects(objects):
    for obj in objects:
        client.fput_object(bucket_name=u"x1-community", 
                           object_name=obj[u"minio"],
                           file_path=obj[u"local"],
                           content_type='application/text'
                           )
    return


def fget_objects(objects):
    for obj in objects:
        client.fget_object(bucket_name=u"x1-community", 
                            object_name=obj[u"minio"],
                            file_path=obj[u"local"])
    return    
        
    
def cvt_imgset_2_points(data_info_file, cal_type, dataset_id, task_id):
    
    minio_img_vect_path = opt.img_vect_bucket_name+u"/img_vect_"+str(dataset_id)+".npy"
    minio_point_set_path = opt.point_bucket_name+u"/pointset_"+str(dataset_id)+".npy"
    minio_point_set_json_path = opt.point_bucket_name+u"/pointset_"+str(dataset_id)+".json"
    minio_tsne_embedding = opt.embedding_bucket_name+u"/tsne_"+str(dataset_id)+".pkl"
    
    local_img_vect_path = u"./imgvect_"+str(dataset_id)+".npy"
    local_point_set_path = u"./pointset_"+str(dataset_id)+".npy"
    local_point_set_json_path = u"./pointset_"+str(dataset_id)+".json"
    local_tsne_embedding = u"./tsne_"+str(dataset_id)+".pkl"
    
    task_data_info = u"./data_info_"+str(task_id)+".json"

    fget_objects([{u"minio":data_info_file, u"local":task_data_info}])
    with codecs.open(task_data_info, u"rb", u"utf-8") as f:
        data_info = json.load(f)
        
    mode = cal_type
    embedding = None
    if cal_type == u"INCREMENT":#之前已经全量计算过该数据集的图片向量，本次只增量更新部分
        
        objects = [
                   {u"minio":minio_img_vect_path, u"local":local_img_vect_path},
                   {u"minio":minio_tsne_embedding, u"local":local_tsne_embedding},
                   {u"minio":minio_point_set_path, u"local":local_point_set_path},
                   {u"minio":minio_point_set_json_path, u"local":local_point_set_json_path},
                ]
        fget_objects(objects)
        
        if not os.path.exists(local_img_vect_path) or not os.path.exists(local_point_set_path)\
            or not os.path.exists(local_point_set_json_path):
            print (u"image vector or image pointset loss, please recalculate the whole image set !!!")
            return 
        
        if not os.path.exists(local_tsne_embedding):
            print (u"visualize model loss, will switch to FULL mode !")
            mode = u"FULL"
        else:
            with open(local_tsne_embedding, u"rb") as f:
                embedding = pickle.load(f)
        
    
    #1. 执行图片点删除
    filtered_img_vect = None
    filtered_pointset = None
    sorted_imgid = []
    add_img = []
    if cal_type == u"INCREMENT":
        if len(data_info[u"deletedIds"]) > 0 or len(data_info[u"addData"]) > 0:
            filtered_img_vect, filtered_pointset, sorted_imgid, add_img = _del_img_points(data_info, 
                                                                      local_img_vect_path, 
                                                                      local_point_set_path, 
                                                                      local_point_set_json_path)
            
    #2. 合并向量和更新
    images = [{u"img_id":item[u"id"], u"img_path":item[u"path"]} for item in data_info[u"fullData"]] \
            + [{u"img_id":item[u"id"], u"img_path":item[u"path"]} for item in add_img]
            
    np_img_vec, imgid_list, failed_imgids = _image_to_vect(images, dataset_id, task_id)   
            
    points = []
    #若本次新增的图片数，比数据集中原有的图片向量总数还多，则重新计算该数据集的可视化模型
    if filtered_img_vect is not None and filtered_img_vect.shape[0] <= np_img_vec.shape[0]:
        mode = u"FULL"
        
    if mode == u"FULL":#用全量数据计算可视化模型
        if filtered_img_vect is not None:#合成出整个图片集的向量数据
            np_img_vec = np.concatenate((filtered_img_vect, np_img_vec), axis=0)
            imgid_list = sorted_imgid + imgid_list  #全量的图片id
            print (u"image id list length :", len(imgid_list))
        
        #重新适配可视化模型
        tsne = TSNE(perplexity=30, n_jobs=4, n_components=2, metric="euclidean", random_state=0)
        points = tsne.fit(np_img_vec)#全量的图片点
        
        #保存模型到minio
        with open(local_tsne_embedding, u"wb") as w:
            pickle.dump(points, w)
        fput_objects([{u"minio":minio_tsne_embedding, u"local":local_tsne_embedding}])
        os.remove(local_tsne_embedding)
        
    elif mode == u"INCREMENT":
        points_added = embedding.transform(np_img_vec)
        if filtered_img_vect is not None:
            np_img_vec = np.concatenate((filtered_img_vect, np_img_vec), axis=0)
            points = np.concatenate((filtered_pointset, points_added), axis=0)
            imgid_list = sorted_imgid + imgid_list
    else:
        print (u"ERROR : not implemented yet !")
        return 
    
    #结果经由本地文件，上传至minio
    np.save(local_img_vect_path, np_img_vec)
    np.save(local_point_set_path, points)
    with codecs.open(local_point_set_json_path, u"wb", u"utf-8") as w:
        json.dump([{u"id":int(idx), u"x":item[0], u"y":item[1]} \
                            for idx, item in zip(imgid_list, points.tolist())], w)
    print (u"result saved to local disk")
    
    objects = [
               {u"minio":minio_img_vect_path, u"local":local_img_vect_path},
               {u"minio":minio_point_set_path, u"local":local_point_set_path},
               {u"minio":minio_point_set_json_path, u"local":local_point_set_json_path},
               {u"minio":u"/datasetSimilarity/result/"+str(task_id)+u".json", u"local":local_point_set_json_path},
            ]
    fput_objects(objects)
    
    os.remove(local_img_vect_path)
    os.remove(local_point_set_path)
    os.remove(local_point_set_json_path)
    os.remove(task_data_info)
    print (u"data been sent to minio")
       
      
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=18881, help='service port')
    parser.add_argument('--threads-in-pool', type=int, default=2, help='线程池的规模')
    parser.add_argument('--vect-dim', type=int, default=256, help='图片向量的维度大小')
    parser.add_argument('--minio-endpoint', type=str, default='basicai-dataset-tmp-minio-endpoint.alidev.beisai.com', 
                        help='minio地址')
    parser.add_argument('--minio-access-key', type=str, default='admin', help='')
    parser.add_argument('--minio-secret-key', type=str, default='QeL71XTy', help='')
    parser.add_argument('--point-bucket-name', type=str, default='point-set', help='')
    parser.add_argument('--img-vect-bucket-name', type=str, default='img-vect', help='')
    parser.add_argument('--embedding-bucket-name', type=str, default='embedding', help='')
    
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. cuda:0 or cpu or cuda')
    
    opt = parser.parse_args()
    print(u"args :", opt)

    model = mobilenet_v3_small(pretrained=True)#出于速度和效果权衡，可以替换为你自己的模型
    model.to(torch.device(opt.device))
    model.eval()
    
    # ImageNet mean & std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    
    executor = ThreadPoolExecutor(opt.threads_in_pool)
    
    MINIO_CONF = {
        'endpoint': opt.minio_endpoint,
        'access_key': opt.minio_access_key,
        'secret_key': opt.minio_secret_key,
        'secure': True
    }
    client = minio.Minio(**MINIO_CONF)
    print (u"client created")

    server = tornado.httpserver.HTTPServer(tornado.wsgi.WSGIContainer(app), 
                                           max_buffer_size=10485760, 
                                           body_timeout=1000.)#body_timeout仅用于限制数据传输时间
    server.bind(opt.port)
    server.start(1)
    print('Tornado server starting on port {}'.format(opt.port), flush=True)
    tornado.ioloop.IOLoop.current().start()


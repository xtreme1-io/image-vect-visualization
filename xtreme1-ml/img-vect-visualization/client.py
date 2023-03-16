#-*- encoding:utf-8 -*-
'''
'''
import json
import requests
import traceback
   
   
   
def post_data(request_url, data):
    data_str = json.dumps(data).encode('ascii')
    headers = {'Content-Type': 'application/json'}
    resp = None
    try:
        resp = requests.post(request_url, data=data_str, headers=headers)
    except Exception as err:
        traceback.print_exc()
    return resp
   
   
def model_call(data_dic):
    request_url = u"http://61.51.222.12:18881/api/v1/calcSimilarity"  
       
    resp = post_data(request_url, data_dic)
    print (resp.text)
    return resp
   
   
if __name__ == '__main__':
    '''
    d = {
  "datasetId":10,
  "serialNumber":"lycan_test_full",
  "filePath":"/datasetSimilarity/commit/test_full.json",
  "type":"FULL"
}
    '''
    d = {
  "datasetId":10,
  "serialNumber":"lycan_test_incremental",
  "filePath":"/datasetSimilarity/commit/test_incremental.json",
  "type":"INCREMENT"
}
    #'''    

    model_call(d)
    
    

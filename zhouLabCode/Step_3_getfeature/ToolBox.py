# encoding: utf-8
import numpy as np
from dingtalkchatbot.chatbot import DingtalkChatbot

#generate a file
def ToolboxCSV(filename = 'filename.csv',list=[]):
    path = "../Note_Res/"
    file = open(path+filename,mode='w')
    for tra in list :
        if(isinstance(tra,str)):
          file.write(tra)
          file.write('\n')
        else:
            file.write(str(tra))
            file.write('\n')
#generate a experiment file
def ToolboxCSV_server(filename = 'filename.csv',list=[]):
    path = "/home/cuizaixu_lab/fanqingchen/DATA/Res/Note_Res/"
    file = open(path+filename,mode='w')
    for tra in list :
        if(isinstance(tra,str)):
          file.write(tra)
          file.write('\n')
        else:
            file.write(str(tra))
            file.write('\n')

#Define a function that takes the upper triangle.Working with Symmetric Matrices
def upper_tri_indexing(matirx):
    m = matirx.shape[0]
    r,c = np.triu_indices(m,1)
    return matirx[r,c]

def send_result_Ding(test):
    webhook = 'https://oapi.dingtalk.com/robot/send?access_token=36aecb847d993acde01b431d66e178e66b54b1aacf4013747c930aa621ec7e9e'
    xiaoding = DingtalkChatbot(webhook)
    Res = 'Result:'+'\n' + test
    xiaoding.send_text(msg=Res,is_at_all=False)
def send_warning_Ding(test):
    webhook = 'https://oapi.dingtalk.com/robot/send?access_token=03d69526107ebfbb9124815833826ad2e3967307d9783bc23d1ba05ab340e6d3'
    xiaoding = DingtalkChatbot(webhook)
    Res = 'Warning:'+'\n' + str(test)
    xiaoding.send_text(msg=Res,is_at_all=False)
'''
    curl 'https://oapi.dingtalk.com/robot/send?access_token=03d69526107ebfbb9124815833826ad2e3967307d9783bc23d1ba05ab340e6d3' \
    - H 'Content-Type: application/json' \
    - d '{"msgtype": "text",
           "text": {
            "content": "Warning: 钉钉机器人群消息测试"
             }
          }'
'''
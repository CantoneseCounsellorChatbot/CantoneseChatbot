


import gdown
import os
import zipfile
# import wget
import requests
import time
def unzip(filename,extract_path):
    zFile = zipfile.ZipFile(filename, "r")
    #ZipFile.namelist(): 获取ZIP文档内所有文件的名称列表
    for fileM in zFile.namelist(): 
        zFile.extract(fileM, extract_path)
    zFile.close()



# 进度条模块
def progressbar(url,path):
  # if not os.path.exists(path): # 看是否有该文件夹，没有则创建文件夹
  #   os.mkdir(path)
  start = time.time() #下载开始时间
  response = requests.get(url, stream=True)
  size = 0 #初始化已下载大小
  chunk_size = 1024 # 每次下载的数据大小
  content_size = int(response.headers['content-length']) # 下载文件总大小
  try:
    if response.status_code == 200: #判断是否响应成功
      print('Start download,[File size]:{size:.2f} MB'.format(size = content_size / chunk_size /1024)) #开始下载，显示下载文件大小
      filepath = path 
      with open(filepath,'wb') as file: #显示进度条
        for data in response.iter_content(chunk_size = chunk_size):
          file.write(data)
          size +=len(data)
          print('\r'+'[Download progress]:%s%.2f%%' % ('>'*int(size*50/ content_size), float(size / content_size * 100)) ,end=' ')
    end = time.time() #下载结束时间
    print('Download completed!,times: %.2f秒' % (end - start)) #输出下载用时时间
  except:
    print('Error!')


id_list=['1YehWJ4BTa_kp5WuZwir5UmnOQGoLvWAn',
        '1ahb97zlnBM57BxyfleaRx-MnsPKCYdOU',
        '1FBCvnKpFoW6XZuE9VMPMEnuMvcAQOLdE',
        '1unEqtLPCXzntIev2NZVJ52AouYQp_YpP',
        '1u3cJ3u7Lss15fiS4JAp9Zckt1lG_h3b9']
url_list= ['https://drive.google.com/uc?id=1YehWJ4BTa_kp5WuZwir5UmnOQGoLvWAn',
            'https://drive.google.com/uc?id=1ahb97zlnBM57BxyfleaRx-MnsPKCYdOU',
            'https://drive.google.com/uc?id=1FBCvnKpFoW6XZuE9VMPMEnuMvcAQOLdE',
            'https://drive.google.com/uc?id=1unEqtLPCXzntIev2NZVJ52AouYQp_YpP',
            'https://drive.google.com/uc?id=1u3cJ3u7Lss15fiS4JAp9Zckt1lG_h3b9']
mega_list=["http://mega.lt.cityu.edu.hk:8081/regression_advice.zip",
            "http://mega.lt.cityu.edu.hk:8081/regression_question.zip",
            "http://mega.lt.cityu.edu.hk:8081/regression_restatement.zip",
            "http://mega.lt.cityu.edu.hk:8081/GPT_restatement.zip",
            "http://mega.lt.cityu.edu.hk:8081/restatement_mmi.zip"]
output_list = ['regression_advice.zip',
                "regression_question.zip",
                "regression_restatement.zip",
                "dialogpt_restatement.zip",
                "dialogpt_mmi.zip"]
for i in range(len(id_list)):
    if os.path.exists(output_list[i]):
        continue
    tmp = "wget --load-cookies /tmp/cookies.txt \"https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={}\" -O {}".format(id_list[i],id_list[i],output_list[i])
    os.system(tmp)
    if os.path.getsize(output_list[i]) < 31230:
        os.remove(output_list[i])
        gdown.download(url_list[i], output_list[i], quiet=False)
    try:
        unzip(output_list[i],"/content/CantoneseChatbot/")
    except:
        print("download from mega ...")
        progressbar(mega_list[i], output_list[i])
        unzip(output_list[i],"/content/CantoneseChatbot/")
print("Dowload finished")

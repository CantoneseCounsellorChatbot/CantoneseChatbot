


import gdown
import os
import zipfile

def unzip(filename,extract_path):
    zFile = zipfile.ZipFile(filename, "r")
    #ZipFile.namelist(): 获取ZIP文档内所有文件的名称列表
    for fileM in zFile.namelist(): 
        zFile.extract(fileM, extract_path)
    zFile.close()
url_list= ['https://drive.google.com/uc?id=1YehWJ4BTa_kp5WuZwir5UmnOQGoLvWAn',
            'https://drive.google.com/uc?id=1ahb97zlnBM57BxyfleaRx-MnsPKCYdOU',
            'https://drive.google.com/uc?id=1FBCvnKpFoW6XZuE9VMPMEnuMvcAQOLdE',
            'https://drive.google.com/uc?id=1unEqtLPCXzntIev2NZVJ52AouYQp_YpP',
            'https://drive.google.com/uc?id=1u3cJ3u7Lss15fiS4JAp9Zckt1lG_h3b9']
output_list = ['regression_advice.zip',
                "regression_question.zip",
                "regression_restatement.zip",
                "dialogpt_restatement.zip",
                "dialogpt_mmi.zip"]
for url,output in zip(url_list,output_list):
    gdown.download(url, output, quiet=False)
    try:
        unzip(output,"/content/CantoneseChatbot/")
    except:
        pass
    os.remove(output)
# url= 'https://drive.google.com/uc?id=1YehWJ4BTa_kp5WuZwir5UmnOQGoLvWAn'
# output = 'pretrained-model.zip'
# for i in range(100):
#   print(i)
#   gdown.download(url, output, quiet=False)
#   os.remove('pretrained-model.zip')

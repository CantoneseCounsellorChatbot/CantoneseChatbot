


import gdown
import os
import zipfile
import wget

def unzip(filename,extract_path):
    zFile = zipfile.ZipFile(filename, "r")
    #ZipFile.namelist(): 获取ZIP文档内所有文件的名称列表
    for fileM in zFile.namelist(): 
        zFile.extract(fileM, extract_path)
    zFile.close()
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
    tmp = "wget --load-cookies /tmp/cookies.txt \"https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={}}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={}\" -O {}".format(id_list[i],id_list[i],output_list[i])
    os.system(tmp)
    if os.path.getsize(output_list[i]) < 31230:
        os.remove(output_list[i])
        gdown.download(url_list[i], output_list[i], quiet=False)
    try:
        unzip(output_list[i],"/content/CantoneseChatbot/")
    except:
        print("download from mega ...")
        wget.download(mega_list[i], output_list[i])
        unzip(output_list[i],"/content/CantoneseChatbot/")
print("Dowload finished")
    # os.remove(output_list[i])
# url= 'https://drive.google.com/uc?id=1YehWJ4BTa_kp5WuZwir5UmnOQGoLvWAn'
# output = 'pretrained-model.zip'
# for i in range(100):
#   print(i)
#   gdown.download(url, output, quiet=False)
#   os.remove('pretrained-model.zip')

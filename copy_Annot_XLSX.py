import os 
from os import path
import shutil

"""
Auto Maneuver 코드가 실행되고나서 
xlsx 파일이 각 폴더에 저장이 되는데 그 파일을 하나의 폴더에 
옮겨서 저장해주는 코드임

1. 코드가 실행되는 현재경로에서 Output_data라는 폴더에 들어가서
RG3으로 시작하는 폴더의 이름을 리스트로 받아옴
그 크기만큼 for 문 반복하여서 xlsx를 처음부터 하나씩 복사하여
지정해준 abs 경로에 순서대로 저장함
그리고 그 경로 (DIR) 과 파일명 리스트를 리턴함
"""

def copy_xlsx(DATE):
    saveDir = os.getcwd() + '\\Output_xlsx'
    # if os.isdir(saveDir) == False:
    #     os.makedir(saveDir)
    try :
        os.path.isdir(saveDir)
    except:
        os.makedirs(saveDir)


    folder_dir = os.getcwd() + '\\Output_data\\' + DATE
    folders = os.listdir(folder_dir)
    folders = [file for file in folders if file.startswith("RG3")]  # RG3 으로 시작하는 파일만 리스트로 사용
    abs_XlsxPaths=[]
    for file in folders:
        if os.path.isfile(folder_dir + '\\' + file + '\\xlsx\\Annotation_' + file +'.xlsx'):
            # abs_XlsxPaths.append(folder_dir + '\\' + file + '\\xlsx\\' + file +'.xlsx')
            origin = folder_dir + '\\' + file + '\\xlsx\\Annotation_' + file +'.xlsx'
            copy = saveDir + '\\Maneuver_' + file +'.xlsx'
            shutil.copy(origin, copy)

    xlsxFiles = os.listdir(os.getcwd() + '\\Output_xlsx')
    xlsxFiles = [file for file in xlsxFiles if file.startswith("RG3_" + DATE)] 
    return xlsxFiles


DATE = "030423"
copy_xlsx(DATE)

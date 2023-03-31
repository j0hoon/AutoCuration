'''
AutoManeuver v1.1 을 위한 코드
'''
# Utils for raw


import json
import pandas as pd
import numpy as np
import scipy.io
import string
import shutil
import scipy.io 
import glob
import os
import pyrosbag as rb
import json
from pathlib import Path
import mat73


def convert_size(size_bytes):
    import math
    if size_bytes == 0:
        return "0B"

    # size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    
    p = math.pow(1024, i)
    s = round(size_bytes /p, 8)
    # s = round(size_bytes / p, 10)
    # return "%s %s" % (s, size_name[i])
    return s/1024

def cal_curve_event(Matsf):
    CURVATURE = (np.array(Matsf['SF_PP']['FRONT_VISION_LANE'][0,0]['PREPROCESSING'][0,0]['CURVATURE'][0,0][0,0]))
    LEFT_LANE = (np.array(Matsf['SF_PP']['FRONT_VISION_LANE'][0,0]['LEFT_LANE'][0,0][0,0]))
    Front_Vision_Lane_sim = np.array((Matsf['SF_PP']['Front_Vision_Lane_sim'][0,0]))
    sim_time = pd.DataFrame(np.array(Matsf['SF_PP']['sim_time']))[0][0]
    
# def cal_curve_event( sim_time ):
    curve_event = np.zeros(np.size(sim_time))
    time = np.size(curve_event) # 2399 정도 
    curve_flag =np.zeros(np.size(sim_time))
    curve_event_array = np.zeros([np.size(sim_time)])
    
    CURVATURE_REFERENCE_VALUE = 1/700
    curve_event = pd.DataFrame(curve_event)
    curve_flag =pd.DataFrame(curve_flag)
    curve_flag_size = 0
    # curve_event.loc[1000] = 1 ## test index
    # curve_event.loc[1001] = 0
    # curve_event.loc[2000] = 1
    # curve_event.loc[2001] = 0
    # curve_event.loc[2100] = 1
    # curve_event.loc[2101] = 0

    for i in range(time):
        if (np.abs(Front_Vision_Lane_sim[CURVATURE-1,LEFT_LANE-1,i]) >CURVATURE_REFERENCE_VALUE):
            curve_event.iloc[i] = 1 # curve event occur
            # print("curve event occur!\n")#test
    
    for k in range(time):
        if (k == 0):
            curve_flag.loc[0] = curve_event.loc[0]
        
        elif (k>0):     
            if (float(curve_event.loc[k]) ==1 and float(curve_event.loc[k-1]) ==0): #Curve->Straight
                # print('Curve->Straight\n')
                curve_flag.loc[k] = 1
            elif (float(curve_event.loc[k]) == 0 and float(curve_event.loc[k-1]) == 1):#Straight -> Curve
                # print('Straight -> Curve\n')
                curve_flag.loc[k] = 2
    for i in range(time):
        if float(curve_flag.loc[i] ) >  0 :                      
            curve_flag_size +=1 
    
    return curve_flag , curve_flag_size


def get_path(file_date,file_area,file_num):
    rosbag2mat_path = "D:\\Shares\\FOT_Avante Data_2\\Rosbag2Mat\\"
    rosbag_path = "D:\\Shares\\FOT_Avante Data_2\\Rosbag\\"
    mat_dir_sf = '\\\\192.168.75.252\\FOT_Avante Data_2\\Rosbag2Mat\\data_' + file_date + '\\Perception//SF'
    mat_data_sf = '\\\\' + file_area + '_' + file_date + '_' + file_num + '_SF_PP'
    mat_dir_origin = '\\\\192.168.75.252\\FOT_Avante Data_2\\Rosbag2Mat\\data_' + file_date
    mat_data_origin = '\\\\'+ file_area + '_' + file_date + '_' + file_num + '.mat'
    bag_file_dir = '\\\\192.168.75.252\\FOT_Avante Data_2\\Rosbag\\data_'+ file_date +'\\'
    bag_file_data = file_area + '_' + file_date + '_' +file_num +'.bag'
    annotation_file_dir = r'\\192.168.75.252\FOT_Avante Data_2\Rosbag2Mat\data_030822\Registration\Annotation\Annotation_'+ file_area +'_'  +file_date+'_'+file_num +'.xlsx'
    # annotation_file_dir = 'C:/Users/ACL_SIM2/Annotation_'+file_area+'_'+file_date+'_'+file_num+'.xlsx'
    return  rosbag2mat_path,rosbag_path,mat_dir_sf,mat_data_sf,mat_dir_origin,mat_data_origin,bag_file_dir,bag_file_data,annotation_file_dir


def get_VEHICLE_SENSOR_SIZE(mat_sf):
    G80 = 20
    CN7 = 19
    CHASSIS = 0  
    mat = mat_sf # 사실 mat 파일임 rAW 에서는
    # lane_data = np.array(mat_sf['SF_PP']['In_Vehicle_Sensor_sim'])
    # temp = pd.DataFrame(lane_data)
    # VEHICLE_SENSOR_SIZE = int(np.size(temp[0][0])/G80) ### G80 20 CN7 19
    try :
        VEHICLE_SENSOR_SIZE = np.size(mat['WHL_SpdFLVal'])
    except:
        VEHICLE_SENSOR_SIZE = np.size(mat['CAM_F'])
        CHASSIS = 1
    return VEHICLE_SENSOR_SIZE , CHASSIS


def get_lane_info(Mat):
    MOBILEYE_STATUS =0
    try:
        FR_CMR_Ln_LftDptDstVal = Mat['FR_CMR_Ln_LftDptDstVal']

    except:
        MOBILEYE_STATUS =1
        lane_width = 0
        curvature=0
        return lane_width, curvature ,  MOBILEYE_STATUS
    
    FR_CMR_Ln_LftDptDstVal = Mat['FR_CMR_Ln_LftDptDstVal']
    FR_CMR_Ln_LftCurveVal = Mat['FR_CMR_Ln_LftCurveVal']
    FR_CMR_Ln_QualLvlLft01Sta = Mat['FR_CMR_Ln_QualLvlLft01Sta']
    FR_CMR_Ln_QualLvlRt01Sta = Mat['FR_CMR_Ln_QualLvlRt01Sta']
    FR_CMR_Ln_RtDptDstVal = Mat['FR_CMR_Ln_RtDptDstVal']
    FR_CMR_Ln_RtCurveVal = Mat['FR_CMR_Ln_RtCurveVal']
    
    lane_width = []
    curvature = []

    for me_idx in range(np.size(FR_CMR_Ln_LftDptDstVal)):
        if (FR_CMR_Ln_QualLvlLft01Sta[me_idx] ==3 and FR_CMR_Ln_QualLvlRt01Sta[me_idx] == 3):
            lane_width.append(-FR_CMR_Ln_LftDptDstVal[me_idx]+FR_CMR_Ln_RtDptDstVal[me_idx])
            curvature.append((FR_CMR_Ln_LftCurveVal[me_idx]+FR_CMR_Ln_RtCurveVal[me_idx])/2)
        elif FR_CMR_Ln_QualLvlLft01Sta[me_idx] ==3:
            lane_width.append(-FR_CMR_Ln_LftDptDstVal[me_idx]*2)
            curvature.append(FR_CMR_Ln_LftCurveVal[me_idx])
        elif FR_CMR_Ln_QualLvlRt01Sta[me_idx] == 3:
            lane_width.append(FR_CMR_Ln_RtDptDstVal[me_idx]*2)
            curvature.append(FR_CMR_Ln_RtCurveVal[me_idx])
        elif (FR_CMR_Ln_QualLvlLft01Sta[me_idx] ==2 and FR_CMR_Ln_QualLvlRt01Sta[me_idx] == 2):
            lane_width.append(-FR_CMR_Ln_LftDptDstVal[me_idx]+FR_CMR_Ln_RtDptDstVal[me_idx])
            curvature.append((FR_CMR_Ln_LftCurveVal[me_idx]+FR_CMR_Ln_RtCurveVal[me_idx])/2)
        elif FR_CMR_Ln_QualLvlLft01Sta[me_idx] ==2:
            lane_width.append(-FR_CMR_Ln_LftDptDstVal[me_idx]*2)
            curvature.append(FR_CMR_Ln_LftCurveVal[me_idx])
        elif FR_CMR_Ln_QualLvlRt01Sta[me_idx] == 2:
            lane_width.append(FR_CMR_Ln_RtDptDstVal[me_idx]*2)
            curvature.append(FR_CMR_Ln_RtCurveVal[me_idx])
    return lane_width, curvature,MOBILEYE_STATUS   



def get_scenery_event(curve_flag , scenery_event):
    for i in range(np.size(curve_flag)):
        tmp_roadGeometry =''
        tmp_frameIndex = 0  
        if float(curve_flag.loc[i]) == 1:
            tmp_roadGeometry = 'CU'
            tmp_frameIndex = i
            scenery_event = pd.concat([scenery_event, pd.DataFrame([[tmp_frameIndex,tmp_roadGeometry]], columns=['frameIndex', 'roadGeometry'])], axis=0)
        elif float(curve_flag.loc[i]) == 2:
            tmp_roadGeometry = 'ST'
            tmp_frameIndex = i     
            scenery_event = pd.concat([scenery_event, pd.DataFrame([[tmp_frameIndex,tmp_roadGeometry]], columns=['frameIndex', 'roadGeometry'])], axis=0)  
    return scenery_event


def get_participant_objframe(index , participants,annotation_label_dynamic):
    participant_objframe = {
        "frameIndex":int(annotation_label_dynamic['FrameIndex'].iloc[index]),
        "ID":int(annotation_label_dynamic['ID'].iloc[index]),
        "participants":participants
    }
    # print("test",index)
    return participant_objframe

def find_maneuver(num):
    maneuver = " "
    if num == 1:
        maneuver = 'FVL'
    if num == 2:
        maneuver = 'FVI'
    if num == 3:
        maneuver = 'FVR'
    if num == 4:
        maneuver = 'AVL'#AVR
    if num == 5:
        maneuver = 'AVR'#AVL
    if num == 6:
        maneuver = 'RVL'
    if num == 7:
        maneuver = 'RVI'
    if num == 8:
        maneuver = 'RVR'
    return maneuver


def get_Fusion_Track_Maneuver(mat_sf,id,index,frame):
    # if id < 1000 and id >0:
    id -= 1
    frame -= 1
    Fusion_Track_Maneuver =  np.array((mat_sf['SF_PP']['Fusion_Track_Maneuver'][0,0][id,index,frame])) 
    return Fusion_Track_Maneuver


# def get_participants(index,recognition,participantID,annotation_label_dynamic):



# def get_participants_frame(tmp_arr_recog,tmp_arr_ID,participants,index):
    
#     participants.append()


        
#     # participants={
#     #     "recognition":recognition,
#     #     "maneuver":"LK",
#     #     "category":2,
#     #     "participantID":participantID
#     # }
#     return participants

def get_participants(mat_sf,index,annotation_label_dynamic):
    SF_PP_FUSION_TRACK_TRACKING_ID = int(np.array((mat_sf['SF_PP']['FUSION_TRACK'][0,0]['TRACKING'][0,0]['ID'][0,0]))) # 22
    SF_PP_FUSION_TRACK_VEHICLE_RECOGNITION_RECOGNITION = int(np.array((mat_sf['SF_PP']['FUSION_TRACK'][0,0]['VEHICLE_RECOGNITION'][0,0]['RECOGNITION'][0,0]))) #50
    ID = int(annotation_label_dynamic['ID'].iloc[index])
    frameIndex = int(annotation_label_dynamic['FrameIndex'].iloc[index])
    participants =[]
    tmp_arr_ID = []
    tmp_arr_recog = []
    for i in range(64):
        tmp_ID = get_Fusion_Track_Maneuver(mat_sf,SF_PP_FUSION_TRACK_TRACKING_ID,i,frameIndex)
        tmp_recog = find_maneuver(get_Fusion_Track_Maneuver(mat_sf,SF_PP_FUSION_TRACK_VEHICLE_RECOGNITION_RECOGNITION,i,frameIndex))
        if ((tmp_ID > 0) and (tmp_ID) != ID):
            if tmp_recog == 'FVL' or tmp_recog == 'FVI' or tmp_recog == 'FVR' or tmp_recog == 'AVL' or tmp_recog == 'AVR' or tmp_recog == 'RVL' or tmp_recog == 'RVI' or tmp_recog == 'RVR':
                tmp_arr_ID.append(tmp_ID)
                tmp_arr_recog.append(tmp_recog)
    
    size = np.size(tmp_arr_recog)
    for index in range(size):
        tmp = {
            "recognition":str(tmp_arr_recog[index]),
            "maneuver":"LK",
            "category":2,
            "participantID":int(tmp_arr_ID[index])
            }      
        participants.append(tmp)
        
    return participants


    



# def get_recognition(mat_sf,frame,ID):
    
#     FVL = float(np.array((mat_sf['SF_PP']['VEHICLE_RECOGNITION'][0,0]['FVL'][0,0])))#1
#     FVI = float(np.array((mat_sf['SF_PP']['VEHICLE_RECOGNITION'][0,0]['FVI'][0,0])))#2
#     FVR = float(np.array((mat_sf['SF_PP']['VEHICLE_RECOGNITION'][0,0]['FVR'][0,0])))#3
#     AVR = float(np.array((mat_sf['SF_PP']['VEHICLE_RECOGNITION'][0,0]['AVR'][0,0])))#4
#     AVL = float(np.array((mat_sf['SF_PP']['VEHICLE_RECOGNITION'][0,0]['AVL'][0,0])))#5
#     RVL = float(np.array((mat_sf['SF_PP']['VEHICLE_RECOGNITION'][0,0]['RVL'][0,0])))#6
#     RVI = float(np.array((mat_sf['SF_PP']['VEHICLE_RECOGNITION'][0,0]['RVI'][0,0])))#7
#     RVR = float(np.array((mat_sf['SF_PP']['VEHICLE_RECOGNITION'][0,0]['RVR'][0,0])))#8
#     SF_PP_FUSION_TRACK_TRACKING_ID = int(np.array((mat_sf['SF_PP']['FUSION_TRACK'][0,0]['TRACKING'][0,0]['ID'][0,0])))
#     SF_PP_FUSION_TRACK_VEHICLE_RECOGNITION_RECOGNITION = int(np.array((mat_sf['SF_PP']['FUSION_TRACK'][0,0]['VEHICLE_RECOGNITION'][0,0]['RECOGNITION'][0,0])))
#     for index in range(64):
#         tmp_participants = {}
#         Fusion_Track_Maneuver = get_Fusion_Track_Maneuver(mat_sf,SF_PP_FUSION_TRACK_TRACKING_ID,index,frame) #np.array((mat_sf['SF_PP']['Fusion_Track_Maneuver'][0,0][SF_PP_FUSION_TRACK_TRACKING_ID,index,frame]))
#         if (Fusion_Track_Maneuver != 0 and Fusion_Track_Maneuver != ID):
#             if(get_Fusion_Track_Maneuver(mat_sf,SF_PP_FUSION_TRACK_VEHICLE_RECOGNITION_RECOGNITION,index,frame) == FVL):
#                 tmp_participants = get_participants(index, 'FVL',get_Fusion_Track_Maneuver(mat_sf,SF_PP_FUSION_TRACK_TRACKING_ID,index,frame))
            
#             elif (get_Fusion_Track_Maneuver(mat_sf,SF_PP_FUSION_TRACK_VEHICLE_RECOGNITION_RECOGNITION,index,frame) == FVI):
#                tmp_participants = get_participants(index, 'FVI',get_Fusion_Track_Maneuver(mat_sf,SF_PP_FUSION_TRACK_TRACKING_ID,index,frame))
            
#             elif (get_Fusion_Track_Maneuver(mat_sf,SF_PP_FUSION_TRACK_VEHICLE_RECOGNITION_RECOGNITION,index,frame) == FVR):
#                tmp_participants = get_participants(index, 'FVR',get_Fusion_Track_Maneuver(mat_sf,SF_PP_FUSION_TRACK_TRACKING_ID,index,frame))
            
#             elif (get_Fusion_Track_Maneuver(mat_sf,SF_PP_FUSION_TRACK_VEHICLE_RECOGNITION_RECOGNITION,index,frame) == AVR):
#                tmp_participants = get_participants(index, 'AVR',get_Fusion_Track_Maneuver(mat_sf,SF_PP_FUSION_TRACK_TRACKING_ID,index,frame))
            
#             elif (get_Fusion_Track_Maneuver(mat_sf,SF_PP_FUSION_TRACK_VEHICLE_RECOGNITION_RECOGNITION,index,frame) == AVL):
#                tmp_participants = get_participants(index, 'AVL',get_Fusion_Track_Maneuver(mat_sf,SF_PP_FUSION_TRACK_TRACKING_ID,index,frame))
            
#             elif (get_Fusion_Track_Maneuver(mat_sf,SF_PP_FUSION_TRACK_VEHICLE_RECOGNITION_RECOGNITION,index,frame) == RVL):
#                tmp_participants = get_participants(index, 'RVL',get_Fusion_Track_Maneuver(mat_sf,SF_PP_FUSION_TRACK_TRACKING_ID,index,frame))
            
#             elif (get_Fusion_Track_Maneuver(mat_sf,SF_PP_FUSION_TRACK_VEHICLE_RECOGNITION_RECOGNITION,index,frame) == RVI):
#                tmp_participants = get_participants(index, 'RVI',get_Fusion_Track_Maneuver(mat_sf,SF_PP_FUSION_TRACK_TRACKING_ID,index,frame))                                                                           
            
#             elif (get_Fusion_Track_Maneuver(mat_sf,SF_PP_FUSION_TRACK_VEHICLE_RECOGNITION_RECOGNITION,index,frame) == RVR):
#                tmp_participants = get_participants(index, 'RVR',get_Fusion_Track_Maneuver(mat_sf,SF_PP_FUSION_TRACK_TRACKING_ID,index,frame))     
                        
                             
    
    return tmp_participants

def get_CSS(admin_dataType,admin_sampleTime,admin_version,admin_projectName,directory,admin_date,admin_travelTime,admin_fileSize,admin_georeference_type,admin_georeference_coordinates,admin_CMGT,\
    admin_AESGT,admin_Stauts,admin_travelDistance,admin_annotationType,scenery,environment,dynamic,participant):

    CSS = [{
        "dataType":admin_dataType,
        "sampleTime":admin_sampleTime,
        "version":admin_version,
        "projectName":admin_projectName, 
        "directory":directory,
        "date":admin_date,
        "travelTime":admin_travelTime,
        "fileSize": admin_fileSize,
        "georeference":{
            "type":admin_georeference_type,
            "coordinates":admin_georeference_coordinates # 줄이 안맞음 []꼴로 나와야 하는데 줄이동이 생김
        },
        "parameter":{
            "stationaryCondition":" ",
            "trigger":" "
        },
        "CMGT":admin_CMGT,
        "AESGT":admin_AESGT,
        "Status":admin_Stauts,
        "travelDistance":admin_travelDistance, #확인 필요
        "annotationType":admin_annotationType,
        "scenery":scenery,
        "environment":environment,
        "dynamic":dynamic,
        "participant":participant,
        }]    
    
    return CSS


def check_CSS_Status(GPS_STATUS,CHASSIS_STATUS,MOBILEYE_STATUS,FRONT_RADAR_STATUS,CORNER_RADAR_STATUS,LIDAR_STATUS,ODD_STATUS):
    CSS_STATUS =[]
    if GPS_STATUS:
        GPS_STATUS = 1
        CSS_STATUS.append(GPS_STATUS)
    if CHASSIS_STATUS:
        CHASSIS_STATUS =2
        CSS_STATUS.append(CHASSIS_STATUS)
    if MOBILEYE_STATUS:
        MOBILEYE_STATUS =3
        CSS_STATUS.append(MOBILEYE_STATUS)
    if FRONT_RADAR_STATUS:
        FRONT_RADAR_STATUS = 4
        CSS_STATUS.append(FRONT_RADAR_STATUS)
    if CORNER_RADAR_STATUS:
        CORNER_RADAR_STATUS = 5
        CSS_STATUS.append(CORNER_RADAR_STATUS)                
    if LIDAR_STATUS:
        LIDAR_STATUS = 6
        CSS_STATUS.append(LIDAR_STATUS)
        
    if ODD_STATUS:
        ODD_STATUS = 7
        CSS_STATUS.append(ODD_STATUS)    
        
    if np.size(CSS_STATUS) == 0:
        CSS_STATUS.append(0)
        
    return CSS_STATUS
        
def check_data(GPS_STATUS,CHASSIS_STATUS,MOBILEYE_STATUS,FRONT_RADAR_STATUS,CORNER_RADAR_STATUS,LIDAR_STATUS,ODD_STATUS,Mat)  :
    try: ## check GPS
        Mat['Latitude_decimal_Xsens']
    except:
        GPS_STATUS = 1
    
    try: ## check CHASSIS_STATUS
        Mat['WHL_SpdFLVal']
    except:
        CHASSIS_STATUS = 1

    try: ## check MOBILEYE_STATUS
        Mat['FR_CMR_Ln_LftDptDstVal']
    except:
        MOBILEYE_STATUS = 1

    try: ## check FRONT_RADAR_STATUS
        Mat['FR_RDR_Genrl_AANormAngl']
    except:
        FRONT_RADAR_STATUS = 1

    try: ## check CORNER_RADAR_STATUS
        Mat['FR_C_RDR_LH_Genrl_AlvCnt01Val']
    except:
        CORNER_RADAR_STATUS = 1       
                        
    # try: ## check Lidar_Staus # RG3 no Lidar  
    #     Mat['FR_C_RDR_LH_Genrl_AlvCnt01Val']
    # except:
    #     Lidar_Staus = 1               
    
    return GPS_STATUS,CHASSIS_STATUS,MOBILEYE_STATUS,FRONT_RADAR_STATUS,CORNER_RADAR_STATUS,LIDAR_STATUS,ODD_STATUS

def copy_xlsx(DATE):
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
    saveDir = os.getcwd() + '\\Output_xlsx'
    # if os.isdir(saveDir) == False:
    #     os.makedir(saveDir)
    if os.path.isdir(saveDir) == False:
        os.makedirs(saveDir)
    # try :
    #     os.path.isdir(saveDir)
    # except:
    #     os.makedirs(saveDir)


    folder_dir = os.getcwd() + '\\Output_data'
    folders = os.listdir(folder_dir)
    folders = [file for file in folders if file.startswith("RG3")]  # RG3 으로 시작하는 파일만 리스트로 사용
    abs_XlsxPaths=[]
    for file in folders:
        if os.path.isfile(folder_dir + '\\' + file + '\\xlsx\\' + file +'.xlsx'):
            # abs_XlsxPaths.append(folder_dir + '\\' + file + '\\xlsx\\' + file +'.xlsx')
            origin = folder_dir + '\\' + file + '\\xlsx\\' + file +'.xlsx'
            copy = saveDir + '\\' + file +'.xlsx'
            if os.path.isfile(copy) == False:
                shutil.copy(origin, copy)

    xlsxFiles = os.listdir(os.getcwd() + '\\Output_xlsx')
    xlsxFiles = [file for file in xlsxFiles if file.startswith("RG3_" + DATE)] 
    return xlsxFiles    

def get_roadName(Regi_xlsx, num):
    dataNum = list(Regi_xlsx['dataNum'])
    roadNamelist = list(Regi_xlsx['roadName'])
    for i in range(0,np.size(dataNum),2):
        if int(dataNum[i]) <= num+1 and int(dataNum[i+1]) >= num+1 :
            roadName = roadNamelist[i]
    return roadName 
            
    
def get_Ant(Ant_dir, fnum ,DATE,TYPE):
    label = pd.read_excel(Ant_dir + '\\'+ TYPE +'_'+ DATE + '_' + fnum + '.xlsx') ### auto mode 
    
    
    # label = pd.read_excel(Ant_dir + '\\Annotation_'+ TYPE +'_'+ DATE + '_' + fnum + '.xlsx', sheet_name = 1) # manual mode 
    # registration = pd.read_excel(Ant_dir + '\\Annotation_'+ TYPE +'_' + DATE + '_' + fnum+ '.xlsx', sheet_name = 0)
    
    # registration = pd.read_excel(Ant_dir + '\\'+ TYPE +'_' + DATE + '_' + fnum+ '.xlsx', sheet_name = 0)
    return label #, registration # auto mode 에는 없음
        
        
def dynamic_Story_Action(Matsf,num,longitudinalAction_acceleration,lateralAction_acceleration,lateralAction_velocity):
    
    longitudinalAction_velocity = " " # 수정해야함
    
    dynamic_init_ego_velocity = (np.array((Matsf['SF_PP']['In_Vehicle_Sensor_sim'][num,0][0,7])) +np.array((Matsf['SF_PP']['In_Vehicle_Sensor_sim'][num,0][0,6])))/2
    
    
    dynamic_story_action = { # for story
        "longitudinalAction":{     
            "velocity":longitudinalAction_velocity,
            "acceleration":longitudinalAction_acceleration
                        },
        "lateralAction":{
            "acceleration":lateralAction_acceleration,
            "velocity":lateralAction_velocity
        } 
    }      
          
def check_ODD(Regi_wrong_xlsx,num):
    ODD_STATUS = 0
    dataNum = list(Regi_wrong_xlsx['dataNum'])
    roadNamelist = list(Regi_wrong_xlsx['Description'])
    for i in range(0,np.size(dataNum)):
        if int(dataNum[i] == num+1):
            if int(roadNamelist[i]) == 7:
                ODD_STATUS = 1
    return ODD_STATUS    
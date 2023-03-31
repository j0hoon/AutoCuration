# [Change log]

# [차종현]
# 080922 Version 1 - 초안 작성
# 101122 Version 2 - Label 생성 추가
# 102222 Version 3 - 좌우 next lane 추가 및 차선 그리는 거리 수정

# [박인철]
# 121422 Ver 4   - 주석 추가, 코드 정리 및 G80 대응 추가, 자차 급감속 급가속 시점 추출함수 수정 (급가속에 대한 정의 추가적인 조사 필요)
#                - 모빌아이 차선 그리는 거리 및 함수 수정, Ego 파라미터 인덱스 Preprocessing값 사용하도록 수정, Yolov7 내부 argparser 수정
#                - Mat list로 받아 처리할 수 있도록 수정, BBOX 그리기 옵션 추가 및 BBOX 크기 변경 -> [BBOX 크기에 따라 정확도 테스트해볼것]
# 
# 010623 Ver 0.1 - xlsx 모든 프레임에 대해 출력하도록 수정 [전처리 필요], output data 경로 수정(날짜별로)
# 011623 Ver 0.2 - 
#------------------------------------------------------------------------------#

# [라이브러리 불러오기]
from tqdm import tqdm
from shapely.geometry import Polygon
import copy

import re
import cv2

import os
import sys
sys.path.append(".\\yolov7\\")
sys.path.append(".\\util\\")
from yolov7.detect import Detect

import glob
import scipy
import scipy.io
import numpy as np
import pandas as pd

import natsort

# [RE]
Site_finder = re.compile('[a-z_A-Z]*')
Filter      = re.compile('[0-9]+')

# [데이터 input path 설정]
Input_path = ""

# [Yolo 학습용 데이터 output path 설정]
Output_image_path = ""
Output_label_path = ""

# [학습용 데이터 검증을 위한 GT path 설정]
Output_GT_path    = ""

# [파라미터 설정]
# [Trajectory length 설정]
Trajectory_length = 5

# [Y축 변환을 위한 COORDINATE PARAM 설정]
Coordinate_parm = 25

# [SF_PP 범위를 고려한 Bev image 크기 설정]
BEV_height = 30 # [40]
BEV_length = 80 # [80]
 
# [Meter to pixel 설정]
M2P = 10

# [M/s to Km/h 설정]
M2K = 3.6

# [Steering ratio 설정]
Steering_ratio = 12

# [Sample time 설정]
Sample_time = 0.05

# [Cv2 color 설정]
Color_dict = {'Red'   : (0, 0, 255),
              'Red2'  : (0, 0, 170),
              'Green' : (0, 255, 0),
              'White' : (255, 255, 255),
              'BBOX'  : (255, 102, 51 )}

# [모빌아이 파라미터 설정 - PREPROCESSING값 사용]
Lane_data_dict = {'DISTANCE'            : 17, # DISTANCE            [18]
                  'ROAD_SLOPE'          : 18, # ROAD_SLOPE          [19]
                  'CURVATURE'           : 19, # CURVATURE           [20]
                  'CURVATURE_RATE'      : 20, # CURVATURE_RATE      [21]
                  'NEXT_DISTANCE'       : 21, # NEXT_DISTANCE       [22]
                  'NEXT_ROAD_SLOPE'     : 22, # NEXT_ROAD_SLOPE     [23]
                  'NEXT_CURVATURE'      : 23, # NEXT_CURVATURE      [24]
                  'NEXT_CURVATURE_RATE' : 24, # NEXT_CURVATURE_RATE [25]
                  'CONFIDENCE'          : 13} # CONFIDENCE          [14] 

def change(x):
    x = int(x)
    if x == 1: 
        return 'LC'
    else: 
        return 'LK'

# [이미지 생성에 사용되는 행렬 계산 함수]
def Get_vehicle_points(Vehicle_x, Vehicle_y, Vehicle_length, Vehicle_width, Vehicle_angle):
    # [Surrounding vehicle의 original point 설정]
    opt = np.array( [[ -0.5*Vehicle_length, -0.5*Vehicle_width],
                    [   0.5*Vehicle_length, -0.5*Vehicle_width],
                    [   0.5*Vehicle_length,  0.5*Vehicle_width],
                    [  -0.5*Vehicle_length,  0.5*Vehicle_width]] )
    
    # [Surrounding vehicle의 angle에 따라 point 회전]
    COS    = np.cos(Vehicle_angle)
    SIN    = np.sin(Vehicle_angle)
    Rotate = [[opt_in[0]*COS - opt_in[1]*SIN, opt_in[0]*SIN + opt_in[1]*COS] for opt_in in opt]
    
    # [Surrounding vehicle의 회전된 point들을 translation]
    Trans = np.array([[(Rotate_in[0]+Vehicle_x+BEV_height)*M2P, (Rotate_in[1]-Vehicle_y+(BEV_height/2))*M2P] for Rotate_in in Rotate], np.int32)

    return Trans

# [자차 이미지 생성 함수]
def Draw_vehicle_ego(Image, Ego_width, Ego_length, Steering_angle): 
    # [자차 이미지 생성을 위한 좌표 생성]
    Vehicle_x      = -4.995/2 # REL_POS_X     [28]
    Vehicle_y      = 0        # REL_POS_Y     [27]
    Vehicle_length = 4.995    # LENGTH        [11]
    Vehicle_width  = 1.925    # WIDTH         [10]
    Vehicle_angle  = Steering_angle # HEADING_ANGLE [12]

    # [rotating 및 translation 행렬 return]
    Trans = Get_vehicle_points(Vehicle_x, Vehicle_y, Vehicle_length, Vehicle_width, Vehicle_angle)

    # [자차 이미지 생성]
    cv2.fillPoly(Image, [Trans], Color_dict['Red'])

# [자차 BBOX 생성 함수]
def Draw_ego_bbox(Image, Ego_width, Ego_length):
    Ego_x_center = 30 - (Ego_length/2)
    Ego_y_center = BEV_height/2
    
    bbox_x_center = float(Ego_x_center/BEV_length)
    bbox_y_center = float(Ego_y_center/BEV_height)
    bbox_width    = float((Ego_length + Ego_width)/(BEV_length*1.5))
    bbox_height   = float((Ego_width)/BEV_height)
    
    opt1 = [(bbox_x_center-bbox_width)*BEV_length*M2P, (bbox_y_center-bbox_height)*BEV_height*M2P]
    opt2 = [(bbox_x_center+bbox_width)*BEV_length*M2P, (bbox_y_center-bbox_height)*BEV_height*M2P]
    opt3 = [(bbox_x_center+bbox_width)*BEV_length*M2P, (bbox_y_center+bbox_height)*BEV_height*M2P]
    opt4 = [(bbox_x_center-bbox_width)*BEV_length*M2P, (bbox_y_center+bbox_height)*BEV_height*M2P]

    Translated_opt = np.array([opt1,opt2,opt3,opt4],np.int32)

    cv2.polylines(Image, [Translated_opt], True, Color_dict['BBOX'], 1)

# [자차 차선 생성 함수]
def Make_lane(Image, Lane_data, Confidence_param):
    Lane_list      = []
    X_array        = []

    # [모빌아이의 Lane curvature range에 따라 Lane index 조절 필요]
    # [Lane curvature range 내부 값들은 보정 없이 3차 접합, 외부 값들은 보정 필요]
    for Lane_index in range(-30, 55):
        if 0 <= Lane_index <= 15:
            Lane_list.append(Lane_data[0] + Lane_data[1]*Lane_index + Lane_data[2]*(Lane_index**2) + Lane_data[3]*(Lane_index**3))
            X_array.append(Lane_index)
        elif -10 <= Lane_index < 0 or 15 < Lane_index <= 25:
            Lane_list.append(Lane_data[0] + Lane_data[1]*(Lane_index / Confidence_param) + Lane_data[2]*(Lane_index**2) + Lane_data[3]*(Lane_index**3 / Confidence_param))
            X_array.append(Lane_index)
        elif -20 <= Lane_index < -10 or 25 <= Lane_index < 35:
            Lane_list.append(Lane_data[0] + Lane_data[1]*(Lane_index / Confidence_param) + Lane_data[2]*(Lane_index**2) + Lane_data[3]*(Lane_index**3 / (Confidence_param+1)))
            X_array.append(Lane_index)
        else:
            Lane_list.append(Lane_data[0] + Lane_data[1]*(Lane_index / Confidence_param) + Lane_data[2]*(Lane_index**2) + Lane_data[3]*(Lane_index**3 / (Confidence_param+3)))
            X_array.append(Lane_index)
    
    Lane_list      = np.array(Lane_list)
    X_array        = np.array(X_array)

    Fitted_lane = np.array([(X_array+27)*M2P, ((BEV_height/2)-Lane_list)*M2P], np.int32).transpose()
    
    cv2.polylines(Image ,[Fitted_lane], False, Color_dict['Green'], 1)

# [사이드 차선 생성 함수]
def Make_side_lane(Image, Lane_data, Confidence_param):
    Lane_list      = []
    X_array        = []

    # [모빌아이의 Lane curvature range에 따라 Lane index 조절 필요]
    # [Lane curvature range 내부 값들은 보정 없이 3차 접합, 외부 값들은 보정 필요]
    for Lane_index in range(-30, 55):
        if 0 <= Lane_index <= 15:
            Lane_list.append(Lane_data[0] + Lane_data[1]*Lane_index + Lane_data[2]*(Lane_index**2) + Lane_data[3]*(Lane_index**3))
            X_array.append(Lane_index)
        elif -10 <= Lane_index < 0 or 15 < Lane_index <= 25:
            Lane_list.append(Lane_data[0] + Lane_data[1]*(Lane_index / Confidence_param) + Lane_data[2]*(Lane_index**2) + Lane_data[3]*(Lane_index**3 / (Confidence_param + 1)))
            X_array.append(Lane_index)
        elif -20 <= Lane_index < -10 or 25 <= Lane_index < 35:
            Lane_list.append(Lane_data[0] + Lane_data[1]*(Lane_index / Confidence_param) + Lane_data[2]*(Lane_index**2) + Lane_data[3]*(Lane_index**3 / (Confidence_param + 2)))
            X_array.append(Lane_index)
        elif -30 <= Lane_index < -20 or 35 <= Lane_index < 25:
            Lane_list.append(Lane_data[0] + Lane_data[1]*(Lane_index / Confidence_param) + Lane_data[2]*(Lane_index**2) + Lane_data[3]*(Lane_index**3 / (Confidence_param + 3)))
            X_array.append(Lane_index)
        else:
            Lane_list.append(Lane_data[0] + Lane_data[1]*(Lane_index / Confidence_param) + Lane_data[2]*(Lane_index**2) + Lane_data[3]*(Lane_index**3 / (Confidence_param + 4)))
            X_array.append(Lane_index)
    
    Lane_list      = np.array(Lane_list)
    X_array        = np.array(X_array)

    Fitted_lane = np.array([(X_array+27)*M2P, ((BEV_height/2)-Lane_list)*M2P], np.int32).transpose()
    
    cv2.polylines(Image ,[Fitted_lane], False, Color_dict['Green'], 1)

# [주변차량 이미지 생성 함수]
def Draw_surrounding_vehicle(Image, Mat, Frame):
    # [Surrounding vehicle의 데이터 가져오기]
    for Target in ['FVL', 'FVI', 'FVR', 'AVL', 'AVR', 'RVL', 'RVI', 'RVR']:
        Track_data     = Mat['SF_PP'][Target][0, 0][Frame]
        Vehicle_x      = Track_data[8]  # REL_POS_X     [9]
        Vehicle_y      = Track_data[7]  # REL_POS_Y     [8]
        Vehicle_length = Track_data[15] # LENGTH        [16]
        Vehicle_width  = Track_data[14] # WIDTH         [15]
        Vehicle_angle  = -Track_data[16] # HEADING_ANGLE [17]
        Vehicle_id     = Track_data[6]  # ID            [7]

        if Vehicle_id != 0 and Vehicle_width > 1:
            if Vehicle_x <= 50 and Vehicle_x >= -25:     # REL_POS_X     [28]
                if Vehicle_y < (BEV_height/2) and Vehicle_y > -(BEV_height/2): # REL_POS_Y     [27]
                    if abs(Vehicle_angle) < 0.5:
                        # [Surrounding vehicle의 original point 설정]
                        opt = np.array( [[ -0.5*Vehicle_length, -0.5*Vehicle_width],
                                        [   0.5*Vehicle_length, -0.5*Vehicle_width],
                                        [   0.5*Vehicle_length,  0.5*Vehicle_width],
                                        [  -0.5*Vehicle_length,  0.5*Vehicle_width]] )
                        
                        # [Surrounding vehicle의 angle에 따라 point 회전]
                        COS    = np.cos(Vehicle_angle)
                        SIN    = np.sin(Vehicle_angle)
                        Rotate = [[opt_in[0]*COS - opt_in[1]*SIN, opt_in[0]*SIN + opt_in[1]*COS] for opt_in in opt]
                        
                        # [Surrounding vehicle의 회전된 point들을 translation]
                        Trans = np.array([[(Rotate_in[0]+Vehicle_x+32)*M2P, (Rotate_in[1]-Vehicle_y+(BEV_height/2))*M2P] for Rotate_in in Rotate], np.int32)
                        
                        # [Surrounding vehicle 이미지 생성]
                        cv2.fillPoly(Image, [Trans], Color_dict['Red'])

# [주변차량 궤적 생성 함수]
def Draw_surrounding_vehicle_trajectory(Image, Mat, Frame, Trajectory):
    # [Surrounding vehicle 궤적 생성]
    if Trajectory == True:
        if Frame < 6:
            Surrounding_trajectory = np.arange(1, Frame)
        else: 
            Surrounding_trajectory = np.arange(Frame - Trajectory_length, Frame)
        
        # [과거 궤적을 그리기 위해 프레임 리스트 생성]    
        Reverse_surrounding_trajectory = Surrounding_trajectory[::-1]
        Draw_frame = []
        
        # [해당 프레임에 Surrounding vehicle의 데이터가 존재하는지 확인]
        for Frame_index in Reverse_surrounding_trajectory[:-1]:
            # [데이터가 존재한다면 과거 궤적을 그리기 위해 해당 프레임을 Draw frame 리스트에 추가]
            for Target in ['FVL', 'FVI', 'FVR', 'AVL', 'AVR', 'RVL', 'RVI', 'RVR']:
                Track_data = Mat['SF_PP'][Target][0, 0][Frame_index]
                Vehicle_angle  = Track_data[16]
                if Track_data[6] != 0 and abs(Vehicle_angle) < 0.5: # ID [7]
                    Draw_frame.append(Frame_index)

        for Frame_index2 in Draw_frame:
            for Target in ['FVL', 'FVI', 'FVR', 'AVL', 'AVR', 'RVL', 'RVI', 'RVR']:
                Track_data     = Mat['SF_PP'][Target][0, 0][Frame_index2]
                Vehicle_x      = Track_data[8]  # REL_POS_X     [9]
                Vehicle_y      = Track_data[7]  # REL_POS_Y     [8]
                Vehicle_length = Track_data[15] # LENGTH        [16]
                Vehicle_width  = Track_data[14] # WIDTH         [15]
                Vehicle_angle  = -Track_data[16] # HEADING_ANGLE [17]

                if Vehicle_x <= 50 and Vehicle_x >= -25:     # REL_POS_X     [28]
                    if Vehicle_y < (BEV_height/2) and Vehicle_y > -(BEV_height/2): # REL_POS_Y     [27]
                        if abs(Vehicle_angle) < 0.5:
                            # [Surrounding vehicle의 original point 설정]
                            opt = np.array( [[ -0.5*Vehicle_length, -0.5*Vehicle_width],
                                            [  0.5*Vehicle_length, -0.5*Vehicle_width],
                                            [  0.5*Vehicle_length,  0.5*Vehicle_width],
                                            [ -0.5*Vehicle_length,  0.5*Vehicle_width]] )
                            
                            # [Surrounding vehicle의 angle에 따라 point 회전]
                            COS    = np.cos(Vehicle_angle)
                            SIN    = np.sin(Vehicle_angle)
                            Rotate = [[opt_in[0]*COS - opt_in[1]*SIN, opt_in[0]*SIN + opt_in[1]*COS] for opt_in in opt]
                            
                            # [Surrounding vehicle의 회전된 point들을 translation]
                            Trans = np.array([[(Rotate_in[0]+Vehicle_x+32)*M2P, (Rotate_in[1]-Vehicle_y+(BEV_height/2))*M2P] for Rotate_in in Rotate], np.int32)

                            # [Surrounding vehicle 이미지 생성]
                            cv2.polylines(Image, [Trans], True, Color_dict['Red2'], 1)

# [주변차량 BBOX 생성 함수]
def Draw_surrounding_vehicle_bbox(Image, Mat, Frame, BBOX):
    # [Surrounding vehicle의 데이터 가져오기]
    for Target in ['FVL', 'FVI', 'FVR', 'AVL', 'AVR', 'RVL', 'RVI', 'RVR']:
        Track_data     = Mat['SF_PP'][Target][0, 0][Frame]
        Vehicle_x      = Track_data[8]  # REL_POS_X     [9]
        Vehicle_y      = Track_data[7]  # REL_POS_Y     [8]
        Vehicle_length = Track_data[15] # LENGTH        [16]
        Vehicle_width  = Track_data[14] # WIDTH         [15]
        Vehicle_angle  = Track_data[16] # HEADING_ANGLE [17]
        Vehicle_id     = Track_data[6]  # ID            [7]
        
        # [GT생성용 BBOX]
        if BBOX == True:
            if Vehicle_id != 0 and Vehicle_width > 1:
                if Vehicle_x <= 50 and Vehicle_x >= -25:     # REL_POS_X     [28]
                    if Vehicle_y < (BEV_height/2) and Vehicle_y > -(BEV_height/2): # REL_POS_Y     [27]
                        if abs(Vehicle_angle) < 0.5:
                            Vehicle_x_bbox = Vehicle_x + 32
                            Vehicle_y_bbox = (BEV_height/2) - Vehicle_y

                            bbox_x_center = float(Vehicle_x_bbox/BEV_length)
                            bbox_y_center = float(Vehicle_y_bbox/BEV_height)
                            bbox_width    = float((Vehicle_length + Vehicle_width)/(BEV_length*1.5))
                            bbox_height   = float((Vehicle_width)/BEV_height)
                            
                            if bbox_x_center+bbox_width > 1:
                                bbox_width    = ( 1 - bbox_x_center + bbox_width )/2 
                                bbox_x_center = 1 - bbox_width

                            if bbox_x_center-bbox_width < 0:
                                bbox_width    = ( bbox_x_center + bbox_width )/2
                                bbox_x_center = bbox_width
                                
                            if bbox_y_center+bbox_height > 1:
                                bbox_height   = ( 1 - bbox_y_center + bbox_height )/2
                                bbox_y_center = 1 - bbox_height

                            if bbox_y_center-bbox_height < 0:
                                bbox_height   = ( bbox_y_center + bbox_height )/2
                                bbox_y_center = bbox_height

                            opt1 = [(bbox_x_center-bbox_width)*BEV_length*M2P, (bbox_y_center-bbox_height)*BEV_height*M2P]
                            opt2 = [(bbox_x_center+bbox_width)*BEV_length*M2P, (bbox_y_center-bbox_height)*BEV_height*M2P]
                            opt3 = [(bbox_x_center+bbox_width)*BEV_length*M2P, (bbox_y_center+bbox_height)*BEV_height*M2P]
                            opt4 = [(bbox_x_center-bbox_width)*BEV_length*M2P, (bbox_y_center+bbox_height)*BEV_height*M2P]
                    
                            Translated_opt = np.array([opt1,opt2,opt3,opt4],np.int32)

                            #cv2.polylines(Image, [Translated_opt], True, Color_dict['BBOX'], 1)

                            # [디버그용 ID 생성]
                            cv2.putText(Image, str(int(Vehicle_id)), Translated_opt[3], fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(255, 0, 0))
                            cv2.putText(Image, Target, Translated_opt[0], fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(255, 0, 0))

# [통합 이미지 생성 함수]
def Make_semantic_maps(Mat, Time_data, Lane_data, Ego_width, Ego_length, Mat_name, Mat_date, Steering_angle, BBOX, Trajectory):
    print('Make semantic maps:')

    # [프레임에 따라 이미지 생성]
    for Frame_num in tqdm(range(len(Time_data))):
        # [이미지 Height, Width 설정]
        Image_height = BEV_height * M2P
        Image_width  = BEV_length * M2P
        Image        = np.ones((Image_height, Image_width, 3)) * 255

        # [모빌아이 차선 데이터 가져오기 및 차선 생성]
        ## [* 리스트 인덱싱시 속도저하 이슈 있음 -> 비교 필요]
        Left_lane_data  = Lane_data[Lane_data_dict['DISTANCE']:Lane_data_dict['CURVATURE_RATE']+1, 0, Frame_num]
        Right_lane_data = Lane_data[Lane_data_dict['DISTANCE']:Lane_data_dict['CURVATURE_RATE']+1, 1, Frame_num]
        Make_lane(Image, Left_lane_data, 5)
        Make_lane(Image, Right_lane_data, 5)

        # [모빌아이 주변 차선 데이터 가져오기 및 차선 생성]
        Left_side_lane_data  = Lane_data[Lane_data_dict['NEXT_DISTANCE']:Lane_data_dict['NEXT_CURVATURE_RATE']+1, 0, Frame_num]
        Right_side_lane_data = Lane_data[Lane_data_dict['NEXT_DISTANCE']:Lane_data_dict['NEXT_CURVATURE_RATE']+1, 1, Frame_num]
        Make_side_lane(Image, Left_side_lane_data, 5)
        Make_side_lane(Image, Right_side_lane_data, 5)

        # [자차량 이미지 생성]
        Draw_vehicle_ego(Image, Ego_width, Ego_length, 0)

        # [주변차량 이미지 생성]
        Draw_surrounding_vehicle_trajectory(Image, Mat, Frame_num, Trajectory)
        Draw_surrounding_vehicle(Image, Mat, Frame_num)

        # [이미지 생성 - 디버깅용]
        Image_for_debug = copy.deepcopy(Image)
        #Draw_ego_bbox(Image_for_debug, Ego_width, Ego_length)
        Draw_surrounding_vehicle_bbox(Image_for_debug, Mat, Frame_num, True)
        
        if os.path.isdir('.\\Input_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\Images') == False:
            os.makedirs('.\\Input_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\Images')

        if os.path.isdir('.\\Input_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\Images_bbox') == False:
            os.makedirs('.\\Input_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\Images_bbox')

        cv2.imwrite('.\\Input_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\Images\\' + str(Mat_name) + '_' + str(Frame_num+1) + '.png', Image)
        cv2.imwrite('.\\Input_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\Images_bbox\\' + str(Mat_name) + '_' + str(Frame_num+1) + '.png', Image_for_debug)

# [라벨 생성 함수]
def Make_label(Mat, Time_data, Ego_width, Ego_length, Mat_name, Mat_date):
    print('Make labels:')
    # [Track 딕셔너리 생성]
    Track_dict   = {}

    # [Bounding box 좌표 생성]
    for Frame_num in range(len(Time_data)):
        Track_list   = []

        # [Surrounding vehicle의 데이터 가져오기]
        for Target in ['FVL', 'FVI', 'FVR', 'AVL', 'AVR', 'RVL', 'RVI', 'RVR']:
            Track_data     = Mat['SF_PP'][Target][0, 0][Frame_num]
            Vehicle_x      = Track_data[8]  # REL_POS_X     [9]
            Vehicle_y      = Track_data[7]  # REL_POS_Y     [8]
            Vehicle_length = Track_data[15] # LENGTH        [16]
            Vehicle_width  = Track_data[14] # WIDTH         [15]
            Vehicle_id     = Track_data[6]  # ID            [7]
            Vehicle_angle  = Track_data[16] # HEADING_ANGLE [17]

            # [해당 Recognition을 가지는 차량이 존재한다면 라벨 txt에 추가]
            if Vehicle_id != 0 and Vehicle_width > 1:
                if Vehicle_x <= 50 and Vehicle_x >= -25:     # REL_POS_X     [28]
                    if Vehicle_y < (BEV_height/2) and Vehicle_y > -(BEV_height/2): # REL_POS_Y     [27]
                        if abs(Vehicle_angle) < 0.5:
                            Vehicle_x_bbox = Vehicle_x + 32
                            Vehicle_y_bbox = (BEV_height/2) - Vehicle_y

                            bbox_x_center = float(Vehicle_x_bbox/BEV_length)
                            bbox_y_center = float(Vehicle_y_bbox/BEV_height)
                            bbox_width    = float((Vehicle_length + Vehicle_width)/(BEV_length*1.5))
                            bbox_height   = float((Vehicle_width)/BEV_height)
                            
                            if bbox_x_center+bbox_width > 1:
                                bbox_width    = ( 1 - bbox_x_center + bbox_width )/2 
                                bbox_x_center = 1 - bbox_width

                            if bbox_x_center-bbox_width < 0:
                                bbox_width    = ( bbox_x_center + bbox_width )/2
                                bbox_x_center = bbox_width
                                
                            if bbox_y_center+bbox_height > 1:
                                bbox_height   = ( 1 - bbox_y_center + bbox_height )/2
                                bbox_y_center = 1 - bbox_height

                            if bbox_y_center-bbox_height < 0:
                                bbox_height   = ( bbox_y_center + bbox_height )/2
                                bbox_y_center = bbox_height

                            bbox_x_center = np.round(bbox_x_center, 4)
                            bbox_y_center = np.round(bbox_y_center, 4)
                            bbox_width  = np.round(bbox_width, 4)
                            bbox_height = np.round(bbox_height, 4)

                            Track_list.append([int(Vehicle_id), bbox_x_center, bbox_y_center, bbox_width*2, bbox_height*2])
                                    
        Track_dict[Frame_num] = Track_list

    # [생성된 Bounding box 좌표를 텍스트 파일로 생성]
    if os.path.isdir('.\\Input_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\Labels') == False:
        os.makedirs('.\\Input_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\Labels')

    for keys in tqdm(Track_dict.keys()):
        f = open('.\\Input_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\Labels\\' + str(Mat_name) + '_' + str(keys+1) + '.txt', 'w')
        f.write(f"0 {str(np.round(float(30-Ego_length/2)/BEV_length, 4))} {str(np.round(float((BEV_height/2)/BEV_height), 4))} {str(np.round(float(((Ego_length + Ego_width)/(BEV_length*1.5)))*2, 4))} {str(np.round(float(((Ego_width)/BEV_height))*2, 4))}" + "\n")
        for i in range(len(Track_dict[keys])):
            f.write(str(Track_dict[keys][i]).replace(",", "").replace("[", "").replace("]", "").replace("'",'') + "\n")
        f.close()

# [문자열로 이루어진 리스트를 IOU 계산을 위한 float 리스트로 반환하는 함수]
def Convert_str_list_to_float_list(Str_list):
    float_list = []
    
    if len(Str_list) != 0:
        for i in range(0, len(Str_list)):
            left_x  = abs(float(Str_list[i][1]) - float(Str_list[i][3]) / 2)
            right_x = abs(float(Str_list[i][1]) + float(Str_list[i][3]) / 2)
            
            y_bot   = abs(float(Str_list[i][2]) - float(Str_list[i][4]) / 2)
            y_top   = abs(float(Str_list[i][2]) + float(Str_list[i][4]) / 2)
            
            float_list.append([[Str_list[i][0]], [left_x, y_top], [right_x, y_top], [right_x, y_bot], [left_x, y_bot]])
    else:
        for i in [0]:
            left_x  = abs(float(Str_list[i][1]) - float(Str_list[i][3]) / 2)
            right_x = abs(float(Str_list[i][1]) + float(Str_list[i][3]) / 2)
            
            y_bot   = abs(float(Str_list[i][2]) - float(Str_list[i][4]) / 2)
            y_top   = abs(float(Str_list[i][2]) + float(Str_list[i][4]) / 2)
            
    return float_list

# [IOU 계산 함수]
def Calculate_iou(box_1, box_2):
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou

# [Yolo detect maneuver와 label id매칭 함수]
def Matching_id(Label_dir, Detection_dir, Mat_name, Mat_date):
    print('Matching id:')
    # [CSS maneuver 딕셔너리 생성]
    CSS_maneuver = {}

    # [Label and detection 리스트 불러오기 및 sorting]
    Label_data_list     = natsort.natsorted(glob.glob(Label_dir + '\\**'))
    Detection_data_list = natsort.natsorted(glob.glob(Detection_dir + '\\**'))

    # [개수 매칭 확인]
    if len(Label_data_list) != len(Detection_data_list): print('Not matching txt')

    for Label_index in range(0, len(Detection_data_list)): # [Label list]
        # [Frame and data 불러오기]
        label_txt_name = Label_data_list[Label_index]

        # [현 프레임 추출]
        Temp_frame = label_txt_name.split('_')[-1]
        Temp_frame = Filter.findall(Temp_frame)[0]
        
        # [txt파일 읽기]
        label_fileID     = Label_data_list[Label_index]
        detection_fileID = Detection_data_list[Label_index] 

        # [IOU 계산을 위한 리스트 구조 변경]
        Temp_label     = list(map(lambda x:x.replace('\n', ''), list(open(label_fileID, 'r'))))
        Temp_label     = list(map(lambda x:x.split(' '), Temp_label))
        Temp_label     = Convert_str_list_to_float_list(Temp_label)

        Temp_detection = list(map(lambda x:x.replace('\n', ''), list(open(detection_fileID, 'r'))))
        Temp_detection = list(map(lambda x:x.split(' '), Temp_detection))
        Temp_detection = Convert_str_list_to_float_list(Temp_detection)

        # [현 프레임 리스트 개수 저장 = 현 프레임에 존재하는 차량 대수]
        Label_size     = len(Temp_label)
        Detection_size = len(Temp_detection)
        Match_array    = {}

        # [현 프레임에 존재하는 차량 만큼 IOU 계산하여 maneuver 매칭]
        for Label_count in range(0, Label_size):
            # [순서대로 차량의 bbox 좌표 추출]
            Temp_bbox       = Temp_label[Label_count][1:]

            Match_array[Temp_label[Label_count][0][0]] = Temp_label[Label_count][0] # [?]
            
            for det in range(0, Detection_size):
                # [순서대로 Detection에서 bbox 좌표 추출]
                Temp_detection_bbox = Temp_detection[det][1:]
                
                # [위에서 추출된 차량의 bbox와 Detection 내부 차량들의 bbox IOU 계산]
                IOU = Calculate_iou(Temp_bbox, Temp_detection_bbox)
                
                # [IOU 0.8 이상이라면 해당 ID와 Detection maneuver 매칭]
                if IOU >= 0.8:
                    Match_array[Temp_label[Label_count][0][0]] = Temp_detection[det][0][0] + ' ' + Temp_label[Label_count][0][0]
            # [현재 프레임을 CSS maneuver 딕셔너리에 저장]
            CSS_maneuver[Temp_frame] = Match_array

    # [ID 변경]
    for Frame in tqdm(CSS_maneuver.keys()):
        # [ID 매칭 후 텍스트 저장을 위한 리스트 생성]
        text_list = []
        
        # [txt파일 불러오기]
        Label_maneuver_txt = open(Label_dir + '\\' +str(Mat_name) + '_' + str(Frame) + '.txt', 'r')
        Label_maneuver_txt = Label_maneuver_txt.read()
        
        # [줄 순서대로 ID 매칭]
        for Text in Label_maneuver_txt.split('\n'):
            ID = Text.split(' ')
            
            # [공백이 아니라면]
            if len(ID) != 1:
                ID_     = ID[0]  # id
                points  = ID[1:] # bbox points
                
                ID_ = CSS_maneuver[f'{Frame}'][ID_] # find GT
                # [리스트로 변환을 위해 타입 가공]
                if type(ID_) == str:
                    text_list.append([str(int(ID_.split(' ')[0]))] + points + [str(int(ID_.split(' ')[-1]))])
                    
                # [땅찍은거 or 매칭 안되는거 0으로 처리]
                else:
                    ID_ = str(0)
                    text_list.append([str(int(ID_.split(' ')[0]))] + points + ['Unknown'])
        
        # [ID 매칭 후 텍스트 생성]
        if os.path.isdir('.\\Output_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\Images\\labels_match') == False:
            os.makedirs('.\\Output_data\\'  + str(Mat_date) + '\\' + str(Mat_name) + '\\Images\\labels_match')

        f = open('.\\Output_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\Images\\labels_match\\' + str(Mat_name) + '_' + str(Frame) + ".txt", 'w')
        for j in text_list:
            f.write(' '.join(j)+'\n')
        f.close()

# [xlsx 생성]
def Make_xlsx(Mat, Matched_label_dir, Mat_name, Mat_date):
    Text_list = natsort.natsorted(glob.glob(Matched_label_dir + '**'))
    
    DF = pd.DataFrame()
    for frame, index in enumerate(Text_list):
        Text = open(index, 'r')
        Temp_list = []
        for lines in Text.readlines():
            Temp_list.append([str(frame+1)] + lines.replace('\n', '').split(' '))
        DF = pd.concat([DF, pd.DataFrame(Temp_list)], axis=0)
    DF = DF.reset_index(drop=True)
    DF.columns = ['FrameIndex', 'Maneuver' ,'X_center', 'Y_center', 'Width', 'Height', 'ID']
    DF['FrameIndex'] = DF['FrameIndex'].astype(int)

    # [Recognition 목록]
    Recog_list = ['FVL', 'FVI', 'FVR', 'AVL' ,'AVR' ,'RVL' ,'RVI', 'RVR']
    Recog_dict = {}

    # [특정 Frame의 특정 Recognition에 존재하는 Vehicle의 ID 추출]
    for recog in Recog_list: 
        Temp_data = np.array(Mat['SF_PP'][recog])
        Temp_data = Temp_data[0, 0]
        Temp_list = []
        for i in range(len(Temp_data)):
            Temp_list.append(int(Temp_data[i, 6])) # ID [7]
        
        Recog_dict[recog] = Temp_list

        
    # [Xlsx용 임시 데이터프레임 생성]
    df = pd.DataFrame()

    # [주변차량 매칭]
    for frame in tqdm(DF['FrameIndex'].unique()):
        # [자차를 제외한 차량과 특정 프레임 데이터만 인덱싱]
        Data = DF[(DF['ID']!='0') & (DF['FrameIndex'] == frame)]
        
        # [Recognition FVL부터 순회]
        for j in Recog_dict.keys():
            recog_data = Recog_dict[j][frame - 1]

            for k in range(len(Data)):
                Data2 = Data.iloc[k]
                # [해당 프레임에 특정 recognition을 가진 vehicle의 ID 매칭여부 확인]
                if Data2.ID != 'Unknown':
                    if int(Data2.ID) == int(recog_data):
                        temp_df = pd.DataFrame([[frame, j, Data2.Maneuver, Data2.ID]])
                        df = pd.concat([df, temp_df], axis=0)
    df.columns = ['FrameIndex', 'Recognition', 'Maneuver', 'ID']

    Ego_df = DF[DF['ID']=='0'].copy()
    Ego_df['Recognition'] = 'Ego'
    Ego_df['REL_Y'] = 0

    df = pd.concat([df, Ego_df.loc[:, ['FrameIndex', 'Recognition', 'Maneuver', 'ID']].copy()], axis=0)
    df = df.sort_values('FrameIndex').reset_index(drop=True)
    df['Category'] = 2

    if os.path.isdir('.\\Output_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\xlsx') == False:
        os.makedirs('.\\Output_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\xlsx')

    df_dir = '.\\Output_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\xlsx\\'

    df['Maneuver'] = df['Maneuver'].apply(change)

    df.loc[:, ['FrameIndex', 'Recognition', 'Maneuver', 'Category', 'ID']].to_excel(f'{df_dir}\\Annotation_{Mat_name}.xlsx', index=False)


# [메인 함수]
def Main(Mat_dir):
    # [Mat list 생성]
    Mat_list = natsort.natsorted(glob.glob(Mat_dir + '\\*.mat'))
    print(f'Number of mats: {len(Mat_list)}' + '\n')

    # [Mat list 내 mat 개수만큼 반복]
    for Mat_ in Mat_list:
        print(f'Current mat: {Mat_}')

        # [Mat data 불러오기]
        Mat = scipy.io.loadmat(Mat_)

        # [Mat 이름 추출]    
        Mat_name = Mat_.split('_SF_PP.mat')[0].split('\\')[-1]

        # [Mat date 추출]
        Mat_date = Filter.findall(Mat_name)[1]

        # [Sim time 가져오기]
        Time_data = Mat['SF_PP']['sim_time'][0, 0]
        
        # [Ego width, length 가져오기]
        Ego_width  = float(Mat['SF_PP']['EGO_VEHICLE'][0, 0][0, 0]['WIDTH'])
        Ego_length = float(Mat['SF_PP']['EGO_VEHICLE'][0, 0][0, 0]['LENGTH'])

        # [모빌아이 차선 데이터 가져오기]
        Lane_data = Mat['SF_PP']['Front_Vision_Lane_sim'][0, 0]

        # [이미지 생성]
        Make_semantic_maps(Mat, Time_data, Lane_data, Ego_width, Ego_length, Mat_name, Mat_date, 0, BBOX=True, Trajectory=True)

        # [라벨 생성]
        Make_label(Mat, Time_data, Ego_width, Ego_length, Mat_name, Mat_date)

        # [Yolov7 Detect]
        Source_dir  = '.\\Input_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\Images\\'
        Weights_dir = '.\\tiny_20230123v6.pt'
        Detect(Source_dir, Mat_name, Mat_date, Weights_dir)

        # [ID 매칭]
        Label_dir     = '.\\Input_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\Labels\\'
        Detection_dir = '.\\Output_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\Images\\labels\\'
        Matching_id(Label_dir, Detection_dir, Mat_name, Mat_date)

        # [Xlsx 생성]
        Matched_label_dir = '.\\Output_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\Images\\labels_match\\'
        Make_xlsx(Mat, Matched_label_dir, Mat_name, Mat_date)

Main(r'C:\\Users\\ACL_SIM_12\\Documents\\까마귀\\test')
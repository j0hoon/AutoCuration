from tqdm import tqdm
from shapely.geometry import Polygon
import copy

import re
import cv2

import os
import sys

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
Trajectory_length = 15

# [Y축 변환을 위한 COORDINATE PARAM 설정]
Coordinate_parm = 25

# [SF_PP 범위를 고려한 Bev image 크기 설정]
BEV_height = 40 # [40]
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

# [자차 이미지 생성 함수]
def Draw_vehicle_ego(Image, Ego_width, Ego_length): 
    # [자차 이미지 생성을 위한 좌표 생성]
    Ego_left_top  = [((BEV_length/2)-Ego_length)*M2P, ((BEV_height/2)-(Ego_width/2))*M2P]
    Ego_left_bot  = [((BEV_length/2)-Ego_length)*M2P, ((BEV_height/2)+(Ego_width/2))*M2P]
    Ego_right_top = [(BEV_length/2)*M2P, ((BEV_height/2)-(Ego_width/2))*M2P]
    Ego_right_bot = [(BEV_length/2)*M2P, ((BEV_height/2)+(Ego_width/2))*M2P]

    # [Cv2 fillPoly 사용을 위해 리스트 -> numpy array 변환]
    Ego_opt = np.array([Ego_left_top, Ego_right_top, Ego_right_bot, Ego_left_bot], int)

    # [자차 이미지 생성]
    cv2.fillPoly(Image, [Ego_opt], Color_dict['Red'])

def Draw_vehicle_ego2(Image, Ego_width, Ego_length, Steering_angle): 
    # [자차 이미지 생성을 위한 좌표 생성]
    Vehicle_x      = -4.995/2 # REL_POS_X     [28]
    Vehicle_y      = 0 # REL_POS_Y     [27]
    Vehicle_length = 4.995 # LENGTH        [11]
    Vehicle_width  = 1.925 # WIDTH         [10]
    Vehicle_angle  = Steering_angle # HEADING_ANGLE [12]
    # print(Vehicle_angle)
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
    Trans = np.array([[(Rotate_in[0]+Vehicle_x+BEV_height)*M2P, (Rotate_in[1]-Vehicle_y+(BEV_height/2))*M2P] for Rotate_in in Rotate], np.int32)
    
    # [자차 이미지 생성]
    cv2.fillPoly(Image, [Trans], Color_dict['Red'])


# [자차 BBOX 생성 함수]
def Draw_ego_bbox(Image, Ego_width, Ego_length):
    Ego_x_center = (BEV_length/2) - (Ego_length/2)
    Ego_y_center = BEV_height/2
    
    bbox_x_center = float(Ego_x_center/BEV_length)
    bbox_y_center = float(Ego_y_center/BEV_height)
    bbox_width    = float((Ego_length + Ego_width)/(BEV_length*1.5))
    bbox_height   = float((Ego_width * 2)/BEV_height)
    
    opt1 = [(bbox_x_center-bbox_width)*BEV_length*M2P, (bbox_y_center-bbox_height)*BEV_height*M2P]
    opt2 = [(bbox_x_center+bbox_width)*BEV_length*M2P, (bbox_y_center-bbox_height)*BEV_height*M2P]
    opt3 = [(bbox_x_center+bbox_width)*BEV_length*M2P, (bbox_y_center+bbox_height)*BEV_height*M2P]
    opt4 = [(bbox_x_center-bbox_width)*BEV_length*M2P, (bbox_y_center+bbox_height)*BEV_height*M2P]

    Translated_opt = np.array([opt1,opt2,opt3,opt4],np.int32)

    cv2.polylines(Image, [Translated_opt], True, Color_dict['BBOX'], 1)

# [주변차량 이미지 생성 함수]
def Draw_surrounding_vehicle(Image, Track_number, Track_data, Frame):
    # [Matlab 인덱스와 Python 인덱스 맞추기]
    Track_number = int(Track_number) - 1
    
    # [Surrounding vehicle 궤적 생성]
    if Frame < 16:
        Surrounding_trajectory = np.arange(1, Frame)
    else: 
        Surrounding_trajectory = np.arange(Frame - Trajectory_length, Frame)
    
    # [과거 궤적을 그리기 위해 프레임 리스트 생성]    
    Reverse_surrounding_trajectory = Surrounding_trajectory[::-1]
    Draw_frame = []
    
    # [해당 프레임에 Surrounding vehicle의 데이터가 존재하는지 확인]
    for Frame_index in Reverse_surrounding_trajectory:
        # [데이터가 존재한다면 과거 궤적을 그리기 위해 해당 프레임을 Draw frame 리스트에 추가]
        if Track_data[25, Track_number, Frame_index] != 0: # ID [26]
            Draw_frame.append(Frame_index)

    for Frame_index2 in Draw_frame:
        # [Surrounding vehicle의 데이터 가져오기]
        Vehicle_x      = Track_data[27, Track_number, Frame_index2] # REL_POS_X     [28]
        Vehicle_y      = Track_data[26, Track_number, Frame_index2] # REL_POS_Y     [27]
        Vehicle_length = Track_data[10, Track_number, Frame_index2] # LENGTH        [11]
        Vehicle_width  = Track_data[9,  Track_number, Frame_index2] # WIDTH         [10]
        Vehicle_angle  = Track_data[11, Track_number, Frame_index2] # HEADING_ANGLE [12]

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
        Trans = np.array([[(Rotate_in[0]+Vehicle_x+BEV_height)*M2P, (Rotate_in[1]-Vehicle_y+(BEV_height/2))*M2P] for Rotate_in in Rotate], np.int32)

        # [Surrounding vehicle 이미지 생성]
        cv2.polylines(Image, [Trans], True, Color_dict['Red2'], 1)

    # [Surrounding vehicle의 데이터 가져오기]
    Vehicle_x      = Track_data[27, Track_number, Frame] # REL_POS_X     [28]
    Vehicle_y      = Track_data[26, Track_number, Frame] # REL_POS_Y     [27]
    Vehicle_length = Track_data[10, Track_number, Frame] # LENGTH        [11]
    Vehicle_width  = Track_data[9,  Track_number, Frame] # WIDTH         [10]
    Vehicle_angle  = Track_data[11, Track_number, Frame] # HEADING_ANGLE [12]
    
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
    Trans = np.array([[(Rotate_in[0]+Vehicle_x+(BEV_length/2))*M2P, (Rotate_in[1]-Vehicle_y+(BEV_height/2))*M2P] for Rotate_in in Rotate], np.int32)

    # [Surrounding vehicle 이미지 생성]
    cv2.fillPoly(Image, [Trans], Color_dict['Red'])
        
# [주변 FVL FVI FVR AVL AVR RVL RVI RVR 이미지 생성 함수]
def Draw_surrounding_vehicle_2_bbox(Image, Mat, Frame, BBOX):
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
                if Vehicle_x < (BEV_length/2) and Vehicle_x > -(BEV_length/2):     # REL_POS_X     [28]
                    if Vehicle_y < (BEV_height/2) and Vehicle_y > -(BEV_height/2): # REL_POS_Y     [27]
                        if abs(Vehicle_angle) < 0.5:
                            Vehicle_x_bbox = Vehicle_x + (BEV_length/2)
                            Vehicle_y_bbox = (BEV_height/2) - Vehicle_y

                            bbox_x_center = float(Vehicle_x_bbox/BEV_length)
                            bbox_y_center = float(Vehicle_y_bbox/BEV_height)
                            bbox_width    = float((Vehicle_length + Vehicle_width)/(BEV_length*1.5))
                            bbox_height   = float((Vehicle_width * 2)/BEV_height)
                            
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

                            cv2.polylines(Image, [Translated_opt], True, Color_dict['BBOX'], 1)

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
                            Trans = np.array([[(Rotate_in[0]+Vehicle_x+(BEV_length/2))*M2P, (Rotate_in[1]-Vehicle_y+(BEV_height/2))*M2P] for Rotate_in in Rotate], np.int32)

                            # [Surrounding vehicle 이미지 생성]
                            cv2.fillPoly(Image, [Trans], Color_dict['Red'])

                            # [디버그용 ID 생성]
                            cv2.putText(Image, str(int(Vehicle_id)), Trans[3], fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(255, 0, 0))
                            cv2.putText(Image, Target, Trans[1], fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(255, 0, 0))

# [주변 FVL FVI FVR AVL AVR RVL RVI RVR 이미지 생성 함수]
def Draw_surrounding_vehicle_2(Image, Mat, Frame, Trajectory):
    # [Surrounding vehicle 궤적 생성]
    if Trajectory == True:
        if Frame < 16:
            Surrounding_trajectory = np.arange(1, Frame)
        else: 
            Surrounding_trajectory = np.arange(Frame - Trajectory_length, Frame)
        
        # [과거 궤적을 그리기 위해 프레임 리스트 생성]    
        Reverse_surrounding_trajectory = Surrounding_trajectory[::-1]
        Draw_frame = []
        
        # [해당 프레임에 Surrounding vehicle의 데이터가 존재하는지 확인]
        for Frame_index in Reverse_surrounding_trajectory:
            # [데이터가 존재한다면 과거 궤적을 그리기 위해 해당 프레임을 Draw frame 리스트에 추가]
            for Target in ['FVL', 'FVI', 'FVR', 'AVL', 'AVR', 'RVL', 'RVI', 'RVR']:
                Track_data     = Mat['SF_PP'][Target][0, 0][Frame_index]
                if Track_data[6] != 0: # ID [7]
                    Draw_frame.append(Frame_index)

        for Frame_index2 in Draw_frame:
            for Target in ['FVL', 'FVI', 'FVR', 'AVL', 'AVR', 'RVL', 'RVI', 'RVR']:
                Track_data     = Mat['SF_PP'][Target][0, 0][Frame_index2]
                Vehicle_x      = Track_data[8]  # REL_POS_X     [9]
                Vehicle_y      = Track_data[7]  # REL_POS_Y     [8]
                Vehicle_length = Track_data[15] # LENGTH        [16]
                Vehicle_width  = Track_data[14] # WIDTH         [15]
                Vehicle_angle  = Track_data[16] # HEADING_ANGLE [17]

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
                Trans = np.array([[(Rotate_in[0]+Vehicle_x+BEV_height)*M2P, (Rotate_in[1]-Vehicle_y+(BEV_height/2))*M2P] for Rotate_in in Rotate], np.int32)

                # [Surrounding vehicle 이미지 생성]
                cv2.polylines(Image, [Trans], True, Color_dict['Red2'], 1)

    # [Surrounding vehicle의 데이터 가져오기]
    for Target in ['FVL', 'FVI', 'FVR', 'AVL', 'AVR', 'RVL', 'RVI', 'RVR']:
        Track_data     = Mat['SF_PP'][Target][0, 0][Frame]
        Vehicle_x      = Track_data[8]  # REL_POS_X     [9]
        Vehicle_y      = Track_data[7]  # REL_POS_Y     [8]
        Vehicle_length = Track_data[15] # LENGTH        [16]
        Vehicle_width  = Track_data[14] # WIDTH         [15]
        Vehicle_angle  = Track_data[16] # HEADING_ANGLE [17]
        Vehicle_id     = Track_data[6]  # ID            [7]
        
        if Vehicle_id != 0 and Vehicle_width > 1:
            if Vehicle_x < (BEV_length/2) and Vehicle_x > -(BEV_length/2):     # REL_POS_X     [28]
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
                        Trans = np.array([[(Rotate_in[0]+Vehicle_x+(BEV_length/2))*M2P, (Rotate_in[1]-Vehicle_y+(BEV_height/2))*M2P] for Rotate_in in Rotate], np.int32)

                        # [Surrounding vehicle 이미지 생성]
                        cv2.fillPoly(Image, [Trans], Color_dict['Red'])

# [자차 차선 생성 함수]
def Make_lane_array(Image, Lane_data, Confidence_param):
    Lane_list      = []
    X_array        = []

    # [모빌아이의 Lane curvature range에 따라 Lane index 조절 필요]
    # [Lane curvature range 내부 값들은 보정 없이 3차 접합, 외부 값들은 보정 필요]
    for Lane_index in range(-40, 41):
        if -10 <= Lane_index <= 10:
            Lane_list.append(Lane_data[0] + Lane_data[1]*Lane_index + Lane_data[2]*(Lane_index**2) + Lane_data[3]*(Lane_index**3))
            X_array.append(Lane_index)
        elif -20 <= Lane_index < -10 or 10 < Lane_index <= 20:
            Lane_list.append(Lane_data[0] + Lane_data[1]*(Lane_index / Confidence_param) + Lane_data[2]*(Lane_index**2) + Lane_data[3]*(Lane_index**3 / Confidence_param))
            X_array.append(Lane_index)
        elif -30 <= Lane_index < -20 or 30 <= Lane_index < 30:
            Lane_list.append(Lane_data[0] + Lane_data[1]*(Lane_index / Confidence_param) + Lane_data[2]*(Lane_index**2) + Lane_data[3]*(Lane_index**3 / (Confidence_param+1)))
            X_array.append(Lane_index)
        else:
            Lane_list.append(Lane_data[0] + Lane_data[1]*(Lane_index / Confidence_param) + Lane_data[2]*(Lane_index**2) + Lane_data[3]*(Lane_index**3 / (Confidence_param+3)))
            X_array.append(Lane_index)
    
    Lane_list      = np.array(Lane_list)
    X_array        = np.array(X_array)

    Fitted_lane = np.array([(X_array+(BEV_length/2))*M2P, ((BEV_height/2)-Lane_list)*M2P], np.int32).transpose()
    
    cv2.polylines(Image ,[Fitted_lane], False, Color_dict['Green'], 1)

# [차선 생성 함수 - Test 필요]
def Make_lane_array_2(Image, Lane_data):
    Lane_list      = []
    X_array        = []

    for Lane_index in range(-40, 41):
        Lane_list.append(Lane_data[0] + Lane_data[1]*Lane_index + Lane_data[2]*(Lane_index**2) + Lane_data[3]*(Lane_index**3))
        X_array.append(Lane_index)

    Lane_list      = np.array(Lane_list)
    X_array        = np.array(X_array)

    Fitted_lane = np.array([(X_array+(BEV_length/2))*M2P, ((BEV_height/2)-Lane_list)*M2P], np.int32).transpose()
    
    cv2.polylines(Image ,[Fitted_lane], False, Color_dict['Green'], 1)

# [사이드 차선 생성 함수]
def Make_side_lane_array(Image, Lane_data, Confidence_param):
    Lane_list      = []
    X_array        = []

    # [모빌아이의 Lane curvature range에 따라 Lane index 조절 필요]
    # [Lane curvature range 내부 값들은 보정 없이 3차 접합, 외부 값들은 보정 필요]
    for Lane_index in range(-40, 41):
        if -5 <= Lane_index <= 5:
            Lane_list.append(Lane_data[0] + Lane_data[1]*Lane_index + Lane_data[2]*(Lane_index**2) + Lane_data[3]*(Lane_index**3))
            X_array.append(Lane_index)
        elif -15 <= Lane_index < -5 or 5 < Lane_index <= 15:
            Lane_list.append(Lane_data[0] + Lane_data[1]*(Lane_index / Confidence_param) + Lane_data[2]*(Lane_index**2) + Lane_data[3]*(Lane_index**3 / (Confidence_param + 1)))
            X_array.append(Lane_index)
        elif -25 <= Lane_index < -15 or 25 <= Lane_index < 15:
            Lane_list.append(Lane_data[0] + Lane_data[1]*(Lane_index / Confidence_param) + Lane_data[2]*(Lane_index**2) + Lane_data[3]*(Lane_index**3 / (Confidence_param + 2)))
            X_array.append(Lane_index)
        elif -35 <= Lane_index < -25 or 35 <= Lane_index < 25:
            Lane_list.append(Lane_data[0] + Lane_data[1]*(Lane_index / Confidence_param) + Lane_data[2]*(Lane_index**2) + Lane_data[3]*(Lane_index**3 / (Confidence_param + 3)))
            X_array.append(Lane_index)
        else:
            Lane_list.append(Lane_data[0] + Lane_data[1]*(Lane_index / Confidence_param) + Lane_data[2]*(Lane_index**2) + Lane_data[3]*(Lane_index**3 / (Confidence_param + 4)))
            X_array.append(Lane_index)
    
    Lane_list      = np.array(Lane_list)
    X_array        = np.array(X_array)

    Fitted_lane = np.array([(X_array+(BEV_length/2))*M2P, ((BEV_height/2)-Lane_list)*M2P], np.int32).transpose()
    
    cv2.polylines(Image ,[Fitted_lane], False, Color_dict['Green'], 1)

# [자차 및 주변차량 라벨 생성 함수]
def Make_label(Track_data, Time_data, Ego_width, Ego_length, Mat_name):
    print('Make labels:')
    # [Track 딕셔너리 생성]
    Track_dict   = {}

    # [Bounding box 좌표 생성]
    for Frame_num in range(len(Time_data)):
        Track_list   = []

        for Track_num in range(len(Track_data)):
            if Track_data[25, Track_num, Frame_num] != 0 and Track_data[9, Track_num, Frame_num]>1:             # ID [26], WIDTH [10]
                if Track_data[27, Track_num, Frame_num] < (BEV_length/2) and Track_data[27, Track_num, Frame_num] > -(BEV_length/2):      # REL_POS_X     [28]
                    if Track_data[26, Track_num, Frame_num] < (BEV_height/2) and Track_data[26, Track_num, Frame_num] > -(BEV_height/2): # REL_POS_Y     [27]
                        if abs(Track_data[30, Track_num, Frame_num]) < 0.7:                                     # HEADING_ANGLE [31]
                            Vehicle_x = Track_data[27, Track_num, Frame_num] + (BEV_length/2)
                            Vehicle_y = (BEV_height/2) - Track_data[26, Track_num, Frame_num]
                            Length    = Track_data[10, Track_num, Frame_num]
                            Width     = Track_data[9, Track_num, Frame_num]
        
                            bbox_x_center = float(Vehicle_x/BEV_length)
                            bbox_y_center = float(Vehicle_y/BEV_height)
                            bbox_width    = float((Length + Width * 2)/BEV_length)
                            bbox_height   = float((Width* 3)/BEV_height)
                            
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

                            Track_list.append([int(Track_num+1), bbox_x_center, bbox_y_center, bbox_width, bbox_height])
                
        Track_dict[Frame_num] = Track_list

    # [생성된 Bounding box 좌표를 텍스트 파일로 생성]
    if os.path.isdir('.\\Input_data\\' + str(Mat_name) + '\\Labels') == False:
        os.makedirs('.\\Input_data\\' + str(Mat_name) + '\\Labels')

    for keys in tqdm(Track_dict.keys()):
        f = open('.\\Input_data\\' + str(Mat_name) + '\\Labels\\' + str(1) + '_' + str(keys+1) + '.txt', 'w')
        f.write(f"0 {str(float((BEV_length/2)-Ego_width/2)/BEV_length)} {str(float((BEV_height/2)/BEV_height))} {str(float((Ego_width*2+Ego_length)/BEV_length))} {str(float(Ego_width*3)/BEV_height)}" + "\n")
        for i in range(len(Track_dict[keys])):
            f.write(str(Track_dict[keys][i]).replace(",", "").replace("[", "").replace("]", "") + "\n")
        f.close()

# [자차 및 FVL FVI FVR AVL AVR RVL RVI RVR 라벨 생성 함수]
def Make_label_2(Mat, Time_data, Ego_width, Ego_length, Mat_name, Mat_date):
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
                if Vehicle_x < (BEV_length/2) and Vehicle_x > -(BEV_length/2):     # REL_POS_X     [28]
                    if Vehicle_y < (BEV_height/2) and Vehicle_y > -(BEV_height/2): # REL_POS_Y     [27]
                        if abs(Vehicle_angle) < 0.5:
                            Vehicle_x_bbox = Vehicle_x + (BEV_length/2)
                            Vehicle_y_bbox = (BEV_height/2) - Vehicle_y

                            bbox_x_center = float(Vehicle_x_bbox/BEV_length)
                            bbox_y_center = float(Vehicle_y_bbox/BEV_height)
                            bbox_width    = float((Vehicle_length + Vehicle_width)/(BEV_length*1.5))
                            bbox_height   = float((Vehicle_width * 2)/BEV_height)
                            
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

                            Track_list.append([int(Vehicle_id), bbox_x_center, bbox_y_center, bbox_width*2, bbox_height*2])
                                    
        Track_dict[Frame_num] = Track_list

    # [생성된 Bounding box 좌표를 텍스트 파일로 생성]
    if os.path.isdir('.\\Input_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\Labels') == False:
        os.makedirs('.\\Input_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\Labels')

    for keys in tqdm(Track_dict.keys()):
        f = open('.\\Input_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\Labels\\' + str(Mat_name) + '_' + str(keys+1) + '.txt', 'w')
        f.write(f"0 {str(float((BEV_length/2)-Ego_length/2)/BEV_length)} {str(float((BEV_height/2)/BEV_height))} {str(float(((Ego_length + Ego_width)/(BEV_length*1.5)))*2)} {str(float(((Ego_width * 2)/BEV_height))*2)}" + "\n")
        for i in range(len(Track_dict[keys])):
            f.write(str(Track_dict[keys][i]).replace(",", "").replace("[", "").replace("]", "") + "\n")
        f.close()

# [통합 이미지 생성 함수]
def Make_semantic_maps2(Track_data, Time_data, Lane_data, Ego_width, Ego_length, Mat_name, Mat_date):
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
        Make_lane_array_2(Image, Left_lane_data)
        Make_lane_array_2(Image, Right_lane_data)

        # [모빌아이 주변 차선 데이터 가져오기 및 차선 생성]
        Left_side_lane_data  = Lane_data[Lane_data_dict['NEXT_DISTANCE']:Lane_data_dict['NEXT_CURVATURE_RATE']+1, 0, Frame_num]
        Right_side_lane_data = Lane_data[Lane_data_dict['NEXT_DISTANCE']:Lane_data_dict['NEXT_CURVATURE_RATE']+1, 1, Frame_num]
        # Make_side_lane_array(Image, Left_side_lane_data, 3)
        # Make_side_lane_array(Image, Right_side_lane_data, 3)
        Make_lane_array_2(Image, Left_side_lane_data, 3)
        Make_lane_array_2(Image, Right_side_lane_data, 3)

        # [자차량 이미지 생성]
        Draw_vehicle_ego(Image, Ego_width, Ego_length)
                
        # [주변차량 이미지 생성]
        Track_number = []
        # [0~63번 트랙에 대해]
        for Track_num in range(0,64):
            # [해당 인덱스에 차량이 존재한다면 그리고 WIDTH가 1보다 크다면]
            if Track_data[25, Track_num, Frame_num] != 0 and Track_data[9, Track_num, Frame_num] > 1:             # ID [26], WIDTH [10]
                # [해당 인덱스 차량의 상대 X좌표가 -40m, 40m 사이에 있다면]
                if Track_data[27, Track_num, Frame_num] < (BEV_length/2) and Track_data[27, Track_num, Frame_num] > -(BEV_length/2):      # REL_POS_X     [28]
                    # [해당 인덱스 차량의 상대 Y좌표가 -20m, 20m 사이에 있다면]
                    if Track_data[26, Track_num, Frame_num] < (BEV_height/2) and Track_data[26, Track_num, Frame_num] > -(BEV_height/2): # REL_POS_Y     [27]
                        # [그리고 마지막으로 해당 인덱스 차량의 절대 헤딩각이 0.7rad 미만이라면 이미지를 생성한다]
                        if abs(Track_data[30, Track_num, Frame_num]) < 0.7:                                     # HEADING_ANGLE [31]
                            Track_number.append(Track_num+1)

        for Vehicle_num in range(0,len(Track_number)):
            Draw_surrounding_vehicle(Image, Track_number[Vehicle_num], Track_data, Frame_num)

        if os.path.isdir('.\\Input_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\Images_bbox') == False:
            os.makedirs('.\\Input_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\Images_bbox')

        if os.path.isdir('.\\Input_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\Images') == False:
            os.makedirs('.\\Input_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\Images')

        cv2.imwrite('.\\Input_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\Images_bbox\\' + str(Mat_name) + '_' + str(Frame_num+1) + '.png', Image)
        cv2.imwrite('.\\Input_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\Images\\' + str(Mat_name) + '_' + str(Frame_num+1) + '.png', Image2)

# [통합 이미지 생성 함수2]
def Make_semantic_maps_2(Mat, Time_data, Lane_data, Ego_width, Ego_length, Mat_name, Mat_date, Steering_angle, BBOX, Trajectory):
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
        Make_lane_array(Image, Left_lane_data, 3)
        Make_lane_array(Image, Right_lane_data, 3)

        # [모빌아이 주변 차선 데이터 가져오기 및 차선 생성]
        Left_side_lane_data  = Lane_data[Lane_data_dict['NEXT_DISTANCE']:Lane_data_dict['NEXT_CURVATURE_RATE']+1, 0, Frame_num]
        Right_side_lane_data = Lane_data[Lane_data_dict['NEXT_DISTANCE']:Lane_data_dict['NEXT_CURVATURE_RATE']+1, 1, Frame_num]
        Make_side_lane_array(Image, Left_side_lane_data, 3)
        Make_side_lane_array(Image, Right_side_lane_data, 3)

        # [자차량 이미지 생성]
        Draw_vehicle_ego2(Image, Ego_width, Ego_length, 0)
                
        # [주변 FVL FVI FVR AVL AVR RVL RVI RVR 이미지 생성]
        Image2 = copy.deepcopy(Image)
        
        Draw_surrounding_vehicle_2(Image2, Mat, Frame_num, Trajectory)
        Draw_surrounding_vehicle_2(Image, Mat, Frame_num, Trajectory)

        Draw_surrounding_vehicle_2_bbox(Image, Mat, Frame_num, BBOX)
        Draw_ego_bbox(Image, Ego_width, Ego_length)
        
        if os.path.isdir('.\\Input_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\Images_bbox') == False:
            os.makedirs('.\\Input_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\Images_bbox')

        if os.path.isdir('.\\Input_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\Images') == False:
            os.makedirs('.\\Input_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\Images')

        cv2.imwrite('.\\Input_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\Images_bbox\\' + str(Mat_name) + '_' + str(Frame_num+1) + '.png', Image)
        cv2.imwrite('.\\Input_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\Images\\' + str(Mat_name) + '_' + str(Frame_num+1) + '.png', Image2)


# [자차 급가속, 급감속시점 추출 함수]
def Find_ego_ac_dc(Ego_velocty, Ego_acceleration, Mat_name, Mat_date):
    print('Find Deceleration:')
    # [자차 속도와 가속도 m/s에서 km/h로 단위 변환, 판다스 데이터프레임에 넣기 위해 행 -> 열 변환]
    Ego_concat = np.concatenate((Ego_velocty.reshape(-1, 1), Ego_acceleration.reshape(-1, 1)), axis=1)

    # [자차 속도와 가속도에 대한 데이터프레임 생성]
    Ego_data_frame = pd.DataFrame(Ego_concat, columns=['Velocity', 'Acceleration'])

    # [공식에 따라 급감속 여부 확인]
    # [Incident Detection Based on Vehicle CAN-Data Within the Large Scale Field Operational Test “euroFOT”]
    # [Thresholds of longitudinal acceleration]

    # [V < 50 km/h 인덱스 추출]
    Velocity_under_50_index   = Ego_data_frame[Ego_data_frame['Velocity'] < 50].index
    
    # [50 km/h <= V <= 150 km/h 인덱스 추출]
    Velocity_50_bet_150_index = Ego_data_frame[(50 <= Ego_data_frame['Velocity']) & (Ego_data_frame['Velocity'] <= 150)].index

    # [150 km/h < V 인덱스 추출]
    Velocity_over_150_index   =  Ego_data_frame[Ego_data_frame['Velocity'] > 150].index

    # [공식에 따라 기준점 생성]
    Ego_data_frame.loc[Velocity_under_50_index,   'DCC_thresholds'] = -6
    Ego_data_frame.loc[Velocity_50_bet_150_index, 'DCC_thresholds'] = 2 * (Ego_data_frame.loc[Velocity_50_bet_150_index, 'Velocity'] / 100) - 6
    Ego_data_frame.loc[Velocity_over_150_index,   'DCC_thresholds'] = -4
    ## [High thresholds: -8]

    # [감속여부 추출]
    DCC_index = Ego_data_frame[Ego_data_frame['Acceleration'] < Ego_data_frame['DCC_thresholds']].index
    Ego_data_frame.loc[DCC_index, 'DCC_check'] = 1
    Ego_data_frame['DCC_check'] = Ego_data_frame['DCC_check'].fillna(0)
    
    # [바로 직전 프레임과 비교하여 DCC_check_3값이 달라진다면 DCC start or DCC end 판단]
    Ego_data_frame['DCC_check_2'] = Ego_data_frame['DCC_check'].shift(1)
    Ego_data_frame['DCC_check_3'] = Ego_data_frame['DCC_check'] - Ego_data_frame['DCC_check_2']
    Ego_data_frame = Ego_data_frame.fillna(0)

    DCC_start_index = Ego_data_frame.query('DCC_check_3 == 1').index + 1
    DCC_end_index   = Ego_data_frame.query('DCC_check_3 == -1').index + 1

    # [3프레임 미만으로 끊기는 데이터 제외]
    if len(DCC_start_index) == len(DCC_end_index):
        Except_index = DCC_start_index - DCC_end_index < -3
        DCC_start_index = list(DCC_start_index[Except_index])
        DCC_end_index   = list(DCC_end_index[Except_index])

        DCC_start_index.extend(DCC_end_index)
        DCC_start_index.sort()

    elif len(DCC_start_index) > len(DCC_end_index):
        Except_index = DCC_start_index[:len(DCC_end_index)] - DCC_end_index < -3
        DCC_start_index = list(DCC_start_index[:len(DCC_end_index)][Except_index])
        DCC_end_index   = list(DCC_end_index[Except_index])

        DCC_start_index.extend(DCC_end_index)
        DCC_start_index.sort()

    elif len(DCC_start_index) < len(DCC_end_index):
        Except_index = DCC_start_index - DCC_end_index[:len(DCC_start_index)] < -3
        DCC_start_index = list(DCC_start_index[Except_index])
        DCC_end_index   = list(DCC_end_index[:len(DCC_start_index)])

        DCC_start_index.extend(DCC_end_index)
        DCC_start_index.sort()

    # [csv파일 저장]
    if os.path.isdir('.\\Input_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\Deceleration') == False:
        os.makedirs('.\\Input_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\Deceleration')

    # [FrameIndex 추가]
    Frame_df = pd.DataFrame(np.arange(1, len(Ego_velocty)+1), columns=['FrameIndex'])
    Frame_df = pd.concat([Frame_df, Ego_data_frame], axis=1)

    # [디버그용 모든시점 저장]
    Frame_df.to_csv('.\\Input_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\Deceleration\\' + str(Mat_name) + '_DCC_ALL.csv', index=False)

    # [DCC시점 추출]
    Frame_df = Frame_df[Frame_df['FrameIndex'].isin(DCC_start_index)]

    # [DCC시점만 저장]
    Frame_df.to_csv('.\\Input_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\Deceleration\\' + str(Mat_name) + '_DCC.csv', index=False)

    print(f'Deceleration count: {sum(Frame_df["DCC_check"])}' + '\n')

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

    for Label_index in range(0, len(Label_data_list)):
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
        Match_id_array = {}

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
                
                # [IOU 0.7 이상이라면 해당 ID와 Detection maneuver 매칭]
                if IOU >= 0.7:
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

# [Maneuver check]
def Maneuver_check(input):
    if input == '0':
        return 'LK'
    elif input == '1':
        return 'LC'
    elif input == '2':
        return 'LT' #LT
    elif input == '3':
        return 'RT' #RT
    else:
        return input
        
# [Ego check]
def Ego_check(input):
    if input == '0':
        return 'Ego'
    else:
        return input

# [매칭된 label txt를 통해 maneuver의 변화를 찾는 함수]
def Finding_maneuver_change(Label_dir, Label_maneuver_dir, Mat_name, Mat_date):
    print('Finding change:')
    # [임시 데이터프레임 및 리스트 생성]
    DF    = pd.DataFrame(columns=['FrameIndex', 'Recognition', 'Maneuver', 'Category', 'ID'])
    List_ = []
    
    # [프레임 리스트 추출 및 sorting]
    Label_list    = natsort.natsorted(glob.glob(Label_dir + '\\*'))
    Maneuver_list = natsort.natsorted(glob.glob(Label_maneuver_dir + '\\*'))
    
    # [프레임 리스트 만큼 반복]
    for Frame_index in tqdm(range(0, len(Label_list)-1)):
        # [Maneuver 매칭된 txt파일 불러오기 - Maneuver]
        Maneuver_label = open(Maneuver_list[Frame_index], 'r')
        Maneuver_label = Maneuver_label.read()

        # [원본 label txt파일 불러오기 - ID]
        ID_label = open(Label_list[Frame_index], 'r')
        ID_label = ID_label.read()

        # [현 프레임의 ID와 Maneuver를 저장하기 위한 딕셔너리 생성]
        Maneuver_dict = {}

        # [현 프레임의 ID와 Maneuver를 딕셔너리에 저장]
        for Maneuver, ID in zip(Maneuver_label.split('\n')[:-1], ID_label.split('\n')[:-1]):
            Maneuver_dict[ID.split(' ')[0]] = Maneuver.split(' ')[0] # [ID = Maneuver의 형식]
            
        # [현 프레임 기준 바로 앞 프레임의 ID와 Maneuver를 저장하기 위한 딕셔너리 생성 - (Frame_index + 1)]
        Maneuver_future_dict = {}

        # [Maneuver 매칭된 txt파일 불러오기 - Maneuver]
        Maneuver_future_label = open(Maneuver_list[Frame_index+1], 'r')
        Maneuver_future_label = Maneuver_future_label.read()

        # [원본 label txt파일 불러오기 - ID]
        ID_future_label = open(Label_list[Frame_index+1], 'r')
        ID_future_label = ID_future_label.read()

        # [현 프레임 기준 바로 앞 프레임의 ID와 Maneuver를 딕셔너리에 저장]
        for Future_manuver, Future_id in zip(Maneuver_future_label.split('\n')[:-1], ID_future_label.split('\n')[:-1]):
            Maneuver_future_dict[Future_id.split(' ')[0]] = Future_manuver.split(' ')[0] # [ID = Maneuver의 형식]

        # [사라지거나 새로 나타나는 차량 ID 추출]          
        Disappearance = list(Maneuver_dict.keys() - Maneuver_future_dict.keys())[:]
        Appearance    = list(Maneuver_future_dict.keys() - Maneuver_dict.keys())[:]
        
        # [사라지거나 새로 나타나는 차량은 딕셔너리에서 제거]
        if len(Disappearance) != 0:
            for index_ in range(0, len(Disappearance)):
                List_.append(f'ID: {Disappearance[index_]}, Start_frame: {Frame_index+1}, End_frame: {Frame_index+2}, Start_maneuver: None, Change_maneuver: None')
                Maneuver_dict.pop(list(Disappearance)[index_], None)

        if len(Appearance) != 0:
            for index_ in range(0, len(Appearance)):
                List_.append(f'ID: {Appearance[index_]}, Start_frame: {Frame_index+1}, End_frame: {Frame_index+2}, Start_maneuver: None, Change_maneuver: None')
                Maneuver_future_dict.pop(list(Appearance)[index_], None)
        
        for i, j in zip(Maneuver_dict.keys(), Maneuver_future_dict.keys()):
            if Maneuver_dict[i] != Maneuver_future_dict[j]:
                #list_.append(f'maneuver_change_id: {i}, change_start_frame: {frame_list[frame]}, change_end_frame: {frame_list[frame+1]}, before_manuever: {maneuver_dict[i]}, after_manuever: {maneuver_dict_future[j]}')
                List_.append(f'ID: {i}, Start_frame: {Frame_index+1}, End_frame: {Frame_index+2}, Start_maneuver: {Maneuver_dict[i]}, Change_maneuver: {Maneuver_future_dict[j]}')
                
                #temp_df = pd.DataFrame([[frame_list[frame+1], check2(i), check(Maneuver_future_dict[j]), 2, i]], columns=[['FrameIndex', 'Recognition', 'Maneuver', 'Category', 'ID']])
                temp_df = pd.DataFrame([[Frame_index+2, Ego_check(i), Maneuver_check(Maneuver_future_dict[j]), 2, i]], columns=['FrameIndex', 'Recognition', 'Maneuver', 'Category', 'ID'])
                DF = pd.concat([DF, temp_df], axis=0)

    if os.path.isdir('.\\Output_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\csv') == False:
        os.makedirs('.\\Output_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\csv')

    DF.to_csv('.\\Output_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\csv\\' + str(Mat_name) + '.csv', index=False)

# [Track 매칭 함수]
def Track_matching(Mat, Csv_dir, Mat_name, Mat_date):
    print('Track matching:')
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

    # [Csv 리스트 생성 및 불러오기]
    Csv_list = natsort.natsorted(glob.glob(Csv_dir + '\\*'))
    CSV = pd.read_csv(Csv_list[-1])

    # [Xlsx용 임시 데이터프레임 생성]
    df = pd.DataFrame()
    
    # [주변차량 매칭]
    for Index in tqdm(range(len(CSV[CSV['Recognition']!='Ego']))):
        Data = CSV[CSV['Recognition']!='Ego'].iloc[Index]
        
        for j in Recog_dict.keys():
            # [Recognition FVL부터 순회]
            #print(i, j, data.FrameIndex)
            recog_data = Recog_dict[j][Data.FrameIndex]

            # [해당 프레임에 특정 recognition을 가진 vehicle의 ID 매칭여부 확인]
            if recog_data != 0 and Data.ID == recog_data:
                REL_Y = np.array(Mat['SF_PP'][j])[0, 0][Data.FrameIndex][7] # REL_POS_Y [8]
                temp_df = pd.DataFrame([[Data.FrameIndex, j, Data.Maneuver, Data.Category, Data.ID, REL_Y]])
                df = pd.concat([df, temp_df], axis=0)
    if len(df) == 0:
        pass
    else:
        df.columns = ['FrameIndex', 'Recognition', 'Maneuver', 'Category', 'ID', 'REL_Y']

        df = pd.concat([CSV[CSV['Recognition'] =='Ego'], df], axis=0).sort_values('FrameIndex')
        df.reset_index(drop=True, inplace=True)

        # # [주변차량 LCL LCR 처리부분]
        # df['REL_Y_before'] = df['REL_Y'].shift(1)
        # df['REL_sub'] = df['REL_Y'] - df['REL_Y_before']

        # df.fillna(0, inplace=True)

        # df.loc[df[(df['Maneuver']=='LC') & (df['REL_sub'] > 0) & (df['Recognition'] != 'Ego')].index, 'Maneuver'] = 'LCL'
        # df.loc[df[(df['Maneuver']=='LC') & (df['REL_sub'] < 0) & (df['Recognition'] != 'Ego')].index, 'Maneuver'] = 'LCR'

        # # [자차 처리]
        # df.loc[df[(df['Maneuver']=='LC') & (df['REL_sub'] > 0) & (df['Recognition'] == 'Ego')].index, 'Maneuver'] = 'LCL'
        # df.loc[df[(df['Maneuver']=='LC') & (df['REL_sub'] < 0) & (df['Recognition'] == 'Ego')].index, 'Maneuver'] = 'LCR'

        if os.path.isdir('.\\Output_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\xlsx') == False:
            os.makedirs('.\\Output_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\xlsx')

        df[['FrameIndex', 'Recognition', 'Maneuver', 'Category', 'ID', 'REL_Y']].to_excel('.\\Output_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\xlsx\\' + str(Mat_name) + '.xlsx', index=False)

    print()

# [Track 매칭 함수 - 2]
def Track_matching_2(Mat, Csv_dir, Mat_name, Mat_date):
    # [CSV를 받아 주변차량의 경우 해당 프레임에서의 REL_Y, 자차의 경우 ~를 가져와 방향처리]
    print('Track matching:')
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

    # [Csv 리스트 생성 및 불러오기]
    Csv_list = natsort.natsorted(glob.glob(Csv_dir + '\\*'))
    CSV = pd.read_csv(Csv_list[-1])

    # [Xlsx용 임시 데이터프레임 생성]
    df = pd.DataFrame()

    # [주변차량 매칭]
    for Index in tqdm(range(len(CSV[CSV['Recognition']!='Ego']))):
        Data = CSV[CSV['Recognition']!='Ego'].iloc[Index]
        
        for j in Recog_dict.keys():
            # [Recognition FVL부터 순회]
            #print(i, j, Data.FrameIndex)
            recog_data = Recog_dict[j][Data.FrameIndex - 1]

            # [해당 프레임에 특정 recognition을 가진 vehicle의 ID 매칭여부 확인]
            if recog_data != 0 and Data.ID == recog_data:
                REL_Y = np.array(Mat['SF_PP'][j])[0, 0][Data.FrameIndex-1][7] # REL_POS_Y [8]
                temp_df = pd.DataFrame([[Data.FrameIndex, j, Data.Maneuver, Data.Category, Data.ID, REL_Y]])
                df = pd.concat([df, temp_df], axis=0)

    if len(df) == 0:
        pass
    else:
        df.columns = ['FrameIndex', 'Recognition', 'Maneuver', 'Category', 'ID', 'REL_Y']

        df = pd.concat([CSV[CSV['Recognition'] =='Ego'], df], axis=0).sort_values('FrameIndex')

        # [자차량 매칭]
        for Index in tqdm(range(len(CSV[CSV['Recognition']=='Ego']))):
            Ego_Data = CSV[CSV['Recognition']=='Ego'].iloc[Index]

            Ego_Y = np.array(Mat['SF_PP']['In_Vehicle_Sensor_sim'][0, 0][Ego_Data.FrameIndex-1][19])

            temp_index = df[(df['FrameIndex'] == Ego_Data.FrameIndex) & (df['Recognition'] == "Ego")].index
            df.loc[temp_index, 'REL_Y'] = Ego_Y

        df.reset_index(drop=True, inplace=True)

        # [LCL LCR 판단]
        for ID in df[df['ID']!=0].ID.unique():
            Temp = df[df['ID'] == ID].copy()
            df.loc[Temp.index, 'BEFORE_REL_Y'] = Temp['REL_Y'] - Temp['REL_Y'].shift(-1)

        for ID in df[df['ID']==0].ID.unique():
            Temp = df[df['ID'] == ID].copy()
            df.loc[Temp.index, 'BEFORE_REL_Y'] = Temp['REL_Y'] - Temp['REL_Y'].shift(1)


        LCR_index = df[(df['Recognition'] != 'Ego') & (df['Maneuver'] == 'LC') & (df['BEFORE_REL_Y'] > 0.4)].index
        df.loc[LCR_index, 'Maneuver'] = 'LCR'

        LCL_index = df[(df['Recognition'] != 'Ego') & (df['Maneuver'] == 'LC') & (df['BEFORE_REL_Y'] < -0.4)].index
        df.loc[LCL_index, 'Maneuver'] = 'LCL'

        Ego_LCL_index = df[(df['Recognition'] == 'Ego') & (df['Maneuver'] == 'LC') & (df['BEFORE_REL_Y'] <= 0.05)].index
        df.loc[Ego_LCL_index, 'Maneuver'] = 'LCL'

        Ego_LCR_index = df[(df['Recognition'] == 'Ego') & (df['Maneuver'] == 'LC') & (df['BEFORE_REL_Y'] > 0.05)].index
        df.loc[Ego_LCR_index, 'Maneuver'] = 'LCR'

        if os.path.isdir('.\\Output_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\xlsx') == False:
            os.makedirs('.\\Output_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\xlsx')

        df[['FrameIndex', 'Recognition', 'Maneuver', 'Category', 'ID']].to_excel('.\\Output_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\xlsx\\' + str(Mat_name) + '.xlsx', index=False)
        df.to_csv('.\\Output_data\\' + str(Mat_date) + '\\' + str(Mat_name) + '\\xlsx\\' + str(Mat_name) + 'for_debug.csv', index=False)

######################################################################################################################

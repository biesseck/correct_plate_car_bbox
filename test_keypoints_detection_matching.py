import os, sys
import numpy as np
import json
import sklearn
import argparse
import glob
from datetime import datetime
import cv2
import copy
import torch
from torch.utils.tensorboard import SummaryWriter
from screeninfo import get_monitors


monitor = None
for m in get_monitors():
    if m.is_primary:
        monitor = m
        break


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--show-imgs', action='store_true')
    args = parser.parse_args()
    return args


#                                     width=-1,   height=768
def resize_img_keep_aspect_ratio(img, width=1366, height=-1, diff=100):
    assert width < 0 or height < 0, f'Error, width ({width}) or height ({height}) must be less than 0'
    if width > 0:
        scale = (width-diff) / img.shape[1]
    elif height > 0:
        scale = (height-diff)  / img.shape[0]
    img_rescaled = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    return img_rescaled


def make_groundtruth_tracks_paths(original_tracks, groundtruth_tracks_folder):
    groundtruth_tracks = [None] * len(original_tracks)
    for idx, original_track in enumerate(original_tracks):
        assert os.path.isdir(original_track), f'Error, could not find original track path: \'{original_track}\''
        original_track = original_track.rstrip('/')
        parent_path = '/'.join(original_track.split('/')[:-1])
        track_name = original_track.split('/')[-1]
        groundtruth_tracks[idx] = os.path.join(parent_path, groundtruth_tracks_folder, track_name)
        assert os.path.isdir(groundtruth_tracks[idx]), f'Error, groundtruth track path is not valid: \'{groundtruth_tracks[idx]}\''
    #     print('groundtruth_tracks[idx]:', groundtruth_tracks[idx])
    # sys.exit(0)
    return groundtruth_tracks


def find_track_files(path_track, ext='.json'):
    assert os.path.isdir(path_track), f'Error, could not find path \'{path_track}\''
    pattern_json = os.path.join(path_track, f"*.{ext.lstrip('*').lstrip('.')}")
    paths_jsons = glob.glob(pattern_json)
    paths_jsons.sort()
    # for path_json in paths_jsons:
    #     print('path_json:', path_json)
    # sys.exit(0)
    return paths_jsons


def filter_orig_track_files_by_groundtruth(orig_track_files, corr_track_files):
    filtered_orig_track_files = []
    ignored_orig_track_files = []
    corr_track_filesnames = [file.split('/')[-1] for file in corr_track_files]
    for idx, orig_track_file in enumerate(orig_track_files):
        orig_track_filename = orig_track_file.split('/')[-1]
        if orig_track_filename in corr_track_filesnames:
            filtered_orig_track_files.append(orig_track_files[idx])
        else:
            ignored_orig_track_files.append(orig_track_files[idx])
    # print('ignored_orig_track_files:', ignored_orig_track_files)
    # print('filtered_orig_track_files:', filtered_orig_track_files)
    assert len(filtered_orig_track_files) == len(corr_track_files), f'Error, len(filtered_orig_track_files) ({len(filtered_orig_track_files)}) != len(corr_track_files) ({len(corr_track_files)}). They must be equal.'
    return filtered_orig_track_files


def load_track_json(path_json):
    f = open(path_json)
    data = json.load(f)
    
    # EXAMPLE OF DATA:
    # shape: {'label': 'car', 'points': [[540.0, 691.7021276595744], [860.0, 958.5]], 'group_id': None, 'description': '', 'shape_type': 'rectangle', 'flags': {}}
    #        {'label': 'lp', 'points': [[740.79526555414, 909.3849246886093], [788.3703789130918, 917.9192058893386], [786.3027406432444, 940.4366770528723], [738.7276272842927, 931.902395852143]], 'group_id': None, 'description': '', 'shape_type': 'polygon', 'flags': {}}
    # shapes = data['shapes']
    return data


def load_img(path_img):
    img_bgr = cv2.imread(path_img)
    return img_bgr


def load_track_data(paths_files_track):
    track_data = [None] * len(paths_files_track)
    for idx_file, json_path in enumerate(paths_files_track):
        frame_data = load_track_json(json_path)
        path_img = os.path.join(os.path.dirname(json_path), frame_data['imagePath'])
        img_bgr = load_img(path_img)
        frame_data['imageData'] = img_bgr
        track_data[idx_file] = frame_data
    return track_data


def get_coords_shape_labelme_format(shape):
    points = shape
    if type(shape) is dict:
        points = shape['points']
    if len(points) == 2:
        upper_left, bottom_right = points
        x1, y1 = upper_left
        x2, y2 = bottom_right[0], x1, 
        x3, y3 = bottom_right
        x4, y4 = x1, bottom_right[1]
    elif len(points) == 4:
        x1, y1 = points[0]
        x2, y2 = points[1]
        x3, y3 = points[2]
        x4, y4 = points[3]
    return {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'x3': x3, 'y3': y3, 'x4': x4, 'y4': y4}


def draw_bbox_car_lp(image, shapes_list, color_bgr=(0,255,0)):
    for idx_shape, shape in enumerate(shapes_list):
        points = shape['points']
        points_int = [[round(coord) for coord in point] for point in points]
        # print('points_int:', points_int)
        if shape['shape_type'] == 'rectangle':
            image = cv2.rectangle(image,points_int[0],points_int[1],color_bgr,2)
            image = cv2.circle(image, center=(int(points_int[0][0]), int(points_int[0][1])), radius=5, color=color_bgr, thickness=-1)
            image = cv2.circle(image, center=(int(points_int[1][0]), int(points_int[1][1])), radius=5, color=color_bgr, thickness=-1)
            image = cv2.putText(image, f'({int(points_int[0][0])},{int(points_int[0][1])})', org=(int(points_int[0][0]), int(points_int[0][1])-15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=color_bgr, thickness=1, lineType=cv2.LINE_AA)
            image = cv2.putText(image, f'({int(points_int[1][0])},{int(points_int[1][1])})', org=(int(points_int[1][0]), int(points_int[1][1])+15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=color_bgr, thickness=1, lineType=cv2.LINE_AA)
        elif shape['shape_type'] == 'polygon':
            image = cv2.polylines(image,[np.array(points_int,dtype=np.int32)],True,color_bgr,1)
            image = cv2.circle(image, center=(int(points_int[0][0]), int(points_int[0][1])), radius=5, color=color_bgr, thickness=-1)
            image = cv2.circle(image, center=(int(points_int[1][0]), int(points_int[1][1])), radius=5, color=color_bgr, thickness=-1)
            image = cv2.circle(image, center=(int(points_int[2][0]), int(points_int[2][1])), radius=5, color=color_bgr, thickness=-1)
            image = cv2.circle(image, center=(int(points_int[3][0]), int(points_int[3][1])), radius=5, color=color_bgr, thickness=-1)
            image = cv2.putText(image, f'({int(points_int[0][0])},{int(points_int[0][1])})', org=(int(points_int[0][0]), int(points_int[0][1])-15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=color_bgr, thickness=1, lineType=cv2.LINE_AA)
            image = cv2.putText(image, f'({int(points_int[1][0])},{int(points_int[1][1])})', org=(int(points_int[1][0]), int(points_int[1][1])-15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=color_bgr, thickness=1, lineType=cv2.LINE_AA)
            image = cv2.putText(image, f'({int(points_int[2][0])},{int(points_int[2][1])})', org=(int(points_int[2][0]), int(points_int[2][1])+15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=color_bgr, thickness=1, lineType=cv2.LINE_AA)
            image = cv2.putText(image, f'({int(points_int[3][0])},{int(points_int[3][1])})', org=(int(points_int[3][0]), int(points_int[3][1])+15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=color_bgr, thickness=1, lineType=cv2.LINE_AA)
        
    return image

'''
def get_patterns_shapes(car_coords, lp_coords):
    car_width = car_coords['x3'] - car_coords['x1']
    car_height = car_coords['y3'] - car_coords['y1']
    # lp_width = lp_coords['x3'] - lp_coords['x1']
    # lp_height = lp_coords['y3'] - lp_coords['y1']
    lp_width = np.linalg.norm(np.array((lp_coords['x2'],lp_coords['y2'])) - np.array((lp_coords['x1'],lp_coords['y1'])))
    lp_height = np.linalg.norm(np.array((lp_coords['x3'],lp_coords['y3'])) - np.array((lp_coords['x2'],lp_coords['y2'])))

    lp_x1_into_bbox_car = lp_coords['x1'] - car_coords['x1']
    lp_y1_into_bbox_car = lp_coords['y1'] - car_coords['y1']
    lp_x2_into_bbox_car = lp_coords['x2'] - car_coords['x1']
    lp_y2_into_bbox_car = lp_coords['y2'] - car_coords['y1']
    lp_x3_into_bbox_car = lp_coords['x3'] - car_coords['x1']
    lp_y3_into_bbox_car = lp_coords['y3'] - car_coords['y1']
    lp_x4_into_bbox_car = lp_coords['x4'] - car_coords['x1']
    lp_y4_into_bbox_car = lp_coords['y4'] - car_coords['y1']

    frame_patterns = {}
    frame_patterns['00_car_width'] = car_width
    frame_patterns['01_car_height'] = car_height
    frame_patterns['02_car_aspect_ratio'] = car_height / car_width
    frame_patterns['03_lp_width'] = lp_width
    frame_patterns['04_lp_height'] = lp_height
    frame_patterns['05_width_ratio'] = car_width / lp_width
    frame_patterns['06_height_ratio'] = car_height / lp_height
    frame_patterns['07_horiz_posit_ratio'] = car_width / lp_x1_into_bbox_car
    frame_patterns['08_verti_posit_ratio'] = car_height / lp_y1_into_bbox_car
    frame_patterns['09_lp_x1_into_bbox_car'] = lp_x1_into_bbox_car
    frame_patterns['10_lp_y1_into_bbox_car'] = lp_y1_into_bbox_car
    frame_patterns['11_lp_x2_into_bbox_car'] = lp_x2_into_bbox_car
    frame_patterns['12_lp_y2_into_bbox_car'] = lp_y2_into_bbox_car
    frame_patterns['13_lp_x3_into_bbox_car'] = lp_x3_into_bbox_car
    frame_patterns['14_lp_y3_into_bbox_car'] = lp_y3_into_bbox_car
    frame_patterns['15_lp_x4_into_bbox_car'] = lp_x4_into_bbox_car
    frame_patterns['16_lp_y4_into_bbox_car'] = lp_y4_into_bbox_car

    return frame_patterns
'''


# def get_patterns_shapes(car_coords, lp_coords):
def get_patterns_shapes(bbox, shapes):
    # print('shapes:', shapes)
    # sys.exit(0)
    car_points = None
    lp_points = None
    for shape in shapes:
        if shape['label'] == 'car':
            car_points = shape['points']
        elif shape['label'] == 'lp':
            lp_points = shape['points']
    
    car_width = car_points[1][0] - car_points[0][0]
    car_height = car_points[1][1] - car_points[0][1]
    # lp_width = np.linalg.norm(np.array((lp_coords['x2'],lp_coords['y2'])) - np.array((lp_coords['x1'],lp_coords['y1'])))
    # lp_height = np.linalg.norm(np.array((lp_coords['x3'],lp_coords['y3'])) - np.array((lp_coords['x2'],lp_coords['y2'])))
    lp_width = np.linalg.norm(np.array(lp_points[1]) - np.array(lp_points[0]))
    lp_height = np.linalg.norm(np.array(lp_points[2]) - np.array(lp_points[1]))

    lp_x1_into_bbox_car = lp_points[0][0] - car_points[0][0]  # lp_coords['x1'] - car_coords['x1']
    lp_y1_into_bbox_car = lp_points[0][1] - car_points[0][1]  # lp_coords['y1'] - car_coords['y1']
    lp_x2_into_bbox_car = lp_points[1][0] - car_points[0][0]  # lp_coords['x2'] - car_coords['x1']
    lp_y2_into_bbox_car = lp_points[1][1] - car_points[0][1]  # lp_coords['y2'] - car_coords['y1']
    lp_x3_into_bbox_car = lp_points[2][0] - car_points[0][0]  # lp_coords['x3'] - car_coords['x1']
    lp_y3_into_bbox_car = lp_points[2][1] - car_points[0][1]  # lp_coords['y3'] - car_coords['y1']
    lp_x4_into_bbox_car = lp_points[3][0] - car_points[0][0]  # lp_coords['x4'] - car_coords['x1']
    lp_y4_into_bbox_car = lp_points[3][1] - car_points[0][1]  # lp_coords['y4'] - car_coords['y1']

    frame_patterns = {}
    frame_patterns['00_car_width'] = car_width
    frame_patterns['01_car_height'] = car_height
    frame_patterns['02_car_aspect_ratio'] = car_height / car_width
    frame_patterns['03_lp_width'] = lp_width
    frame_patterns['04_lp_height'] = lp_height
    frame_patterns['05_width_ratio'] = car_width / lp_width
    frame_patterns['06_height_ratio'] = car_height / lp_height
    frame_patterns['07_horiz_posit_ratio'] = car_width / lp_x1_into_bbox_car
    frame_patterns['08_verti_posit_ratio'] = car_height / lp_y1_into_bbox_car
    frame_patterns['09_lp_x1_into_bbox_car'] = lp_x1_into_bbox_car
    frame_patterns['10_lp_y1_into_bbox_car'] = lp_y1_into_bbox_car
    frame_patterns['11_lp_x2_into_bbox_car'] = lp_x2_into_bbox_car
    frame_patterns['12_lp_y2_into_bbox_car'] = lp_y2_into_bbox_car
    frame_patterns['13_lp_x3_into_bbox_car'] = lp_x3_into_bbox_car
    frame_patterns['14_lp_y3_into_bbox_car'] = lp_y3_into_bbox_car
    frame_patterns['15_lp_x4_into_bbox_car'] = lp_x4_into_bbox_car
    frame_patterns['16_lp_y4_into_bbox_car'] = lp_y4_into_bbox_car

    return frame_patterns


'''
def get_patterns_shapes(bbox, shapes):
    all_props_points_to_bbox = [None] * len(shapes)
    for idx_shape, shape in enumerate(shapes):
        points = shape['points']
        props_points_to_bbox = get_patterns_bbox_points(bbox, points)
        
        # print('shape:', shape)
        all_props_points_to_bbox[idx_shape] = {shape['label']: props_points_to_bbox}
        # sys.exit(0)

    return all_props_points_to_bbox
'''


def get_patterns_bbox_points(bbox, points):
    # print('bounding_rect:', bounding_rect)
    # print('points_rect:', points_rect)
    bbox_width, bbox_height = np.array(bbox[1]) - np.array(bbox[0])
    # print('bbox_width, bbox_height:', bbox_width, bbox_height)
    props_points_to_bbox = [None] * len(points)
    for idx_point, point in enumerate(points):
        reference = bbox[0]   # top left point of bbox
        x_prop, y_prop = (np.array(reference)-np.array(point)) / (bbox_width, bbox_height)
        props_points_to_bbox[idx_point] = (x_prop, y_prop)
        # print('proportions[idx_point]:', props_points_to_bbox[idx_point])
    # sys.exit(0)
    return props_points_to_bbox


def predict_points_from_props(bbox, curr_points, right_props):
    bbox_width, bbox_height = np.array(bbox[1]) - np.array(bbox[0])
    pred_points_to_bbox = [None] * len(curr_points)
    for idx_prop, right_prop in enumerate(right_props):
        reference = bbox[0]   # top left point of bbox
        x_pred, y_pred = np.array(reference) - (np.array([bbox_width, bbox_height]) * right_prop)
        pred_points_to_bbox[idx_prop] = (x_pred, y_pred)
    return pred_points_to_bbox


def predict_shapes_from_bbox_kps(bounding_rect_good_matches_frame0, frame0_curr_shapes,
                                 bounding_rect_good_matches_frame1, frame1_curr_shapes):
    # print('frame1_curr_shapes:', frame1_curr_shapes)
    # sys.exit(0)
    frame1_pred_shapes = copy.deepcopy(frame1_curr_shapes)
    for idx_shape, (frame0_curr_shape, frame1_curr_shape, frame1_pred_shape) in enumerate(zip(frame0_curr_shapes, frame1_curr_shapes, frame1_pred_shapes)):
        frame0_curr_points = frame0_curr_shape['points']
        frame0_right_props = get_patterns_bbox_points(bounding_rect_good_matches_frame0, frame0_curr_points)

        frame1_curr_points = frame1_curr_shape['points']
        frame0_pred_points = predict_points_from_props(bounding_rect_good_matches_frame1, frame1_curr_points, frame0_right_props)
    
        frame1_pred_shape['points'] = frame0_pred_points
    return frame1_pred_shapes


def predict_track_compute_patterns_bbox_car_lp(args,track_data):
    frames_patterns = [None] * len(track_data)
    for idx_frame in range(len(track_data)-1):
        frame0_data = track_data[idx_frame]
        frame1_data = track_data[idx_frame+1]
        # print('frame0_data.keys():', frame0_data.keys())

        # frame0_descs = frame0_data['descs_sift']
        # frame1_descs = frame1_data['descs_sift']
        centroid_good_matches_frame0 = frame0_data['centroid_good_matches_frame']
        centroid_good_matches_frame1 = frame0_data['centroid_good_matches_frame+1']
        bounding_rect_good_matches_frame0 = frame0_data['bounding_rect_good_matches_frame']
        bounding_rect_good_matches_frame1 = frame0_data['bounding_rect_good_matches_frame+1']
        # frame0_data['bounding_rect_good_matches_frame'] = frame0_bounding_rect_good_matches
        # frame0_data['bounding_rect_good_matches_frame+1'] = frame1_bounding_rect_good_matches

        frame0_curr_shapes = frame0_data['shapes']
        frame1_curr_shapes = frame1_data['shapes']
        frame1_pred_shapes = predict_shapes_from_bbox_kps(bounding_rect_good_matches_frame0, frame0_curr_shapes,
                                                               bounding_rect_good_matches_frame1, frame1_curr_shapes)
        frame1_data['shapes'] = frame1_pred_shapes

        frame0_patterns_curr_shapes = get_patterns_shapes(bounding_rect_good_matches_frame0, frame0_curr_shapes)
        frame1_patterns_curr_shapes = get_patterns_shapes(bounding_rect_good_matches_frame1, frame1_curr_shapes)
        frame1_patterns_pred_shapes = get_patterns_shapes(bounding_rect_good_matches_frame1, frame1_pred_shapes)
        frames_patterns[idx_frame] = {'frame0_patterns_curr_shapes': frame0_patterns_curr_shapes,
                                      'frame1_patterns_curr_shapes': frame1_patterns_curr_shapes,
                                      'frame1_patterns_pred_shapes': frame1_patterns_pred_shapes}
        
        if args.show_imgs:
            # ONLY FOR VISUALIZATION (TESTS)
            image0 = frame0_data['imageData'].copy()
            image1 = frame1_data['imageData'].copy()

            image0 = draw_bbox_car_lp(image0, frame0_curr_shapes)
            image1 = draw_bbox_car_lp(image1, frame1_curr_shapes)
            image1 = draw_bbox_car_lp(image1, frame1_pred_shapes, color_bgr=(255,255,100))

            image0 = cv2.putText(image0, frame0_data['imagePath'], org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness = 2, lineType=cv2.LINE_AA)
            image1 = cv2.putText(image1, frame1_data['imagePath'], org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness = 2, lineType=cv2.LINE_AA)
            image0 = cv2.circle(image0, center=(int(centroid_good_matches_frame0[0]), int(centroid_good_matches_frame0[1])), radius=7, color=(255,0,0), thickness=4)
            image1 = cv2.circle(image1, center=(int(centroid_good_matches_frame1[0]), int(centroid_good_matches_frame1[1])), radius=7, color=(255,0,0), thickness=4)
            
            color_bbox_kps = (255,0,0)
            image0 = cv2.rectangle(image0,bounding_rect_good_matches_frame0[0],bounding_rect_good_matches_frame0[1],color_bbox_kps,2)
            image1 = cv2.rectangle(image1,bounding_rect_good_matches_frame1[0],bounding_rect_good_matches_frame1[1],color_bbox_kps,2)
            image0 = cv2.circle(image0, center=(int(bounding_rect_good_matches_frame0[0][0]), int(bounding_rect_good_matches_frame0[0][1])), radius=7, color=color_bbox_kps, thickness=-1)
            image0 = cv2.circle(image0, center=(int(bounding_rect_good_matches_frame0[1][0]), int(bounding_rect_good_matches_frame0[1][1])), radius=7, color=color_bbox_kps, thickness=-1)
            image0 = cv2.putText(image0, f'({int(bounding_rect_good_matches_frame0[0][0])},{int(bounding_rect_good_matches_frame0[0][1])})', org=(int(bounding_rect_good_matches_frame0[0][0]), int(bounding_rect_good_matches_frame0[0][1])-15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,0,0), thickness=1, lineType=cv2.LINE_AA)
            image0 = cv2.putText(image0, f'({int(bounding_rect_good_matches_frame0[1][0])},{int(bounding_rect_good_matches_frame0[1][1])})', org=(int(bounding_rect_good_matches_frame0[1][0]), int(bounding_rect_good_matches_frame0[1][1])+15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,0,0), thickness=1, lineType=cv2.LINE_AA)
            
            image0_1 = np.hstack((image0, image1))
            # image0_1 = cv2.resize(image0_1, (0, 0), fx=0.7, fy=0.7)
            image0_1 = resize_img_keep_aspect_ratio(image0_1, width=monitor.width, height=-1)
            cv2.imshow('image0_1', image0_1)
            cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    # sys.exit(0)
    return frames_patterns


def detect_describe_keypoints(tracks_data):
    sift = cv2.SIFT_create()

    for idx_frame, frame_data in enumerate(tracks_data):
        print(f"frame {idx_frame}/{len(tracks_data)} - \'{frame_data['imagePath']}\'", end='\r')
        img_color = frame_data['imageData']
        img_gray= cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        # kps = sift.detect(img_gray, None)
        kps, descs = sift.detectAndCompute(img_gray, None)
        frame_data['kps_sift'] = kps
        frame_data['descs_sift'] = descs

        # img_with_kp = cv2.drawKeypoints(img_gray, kps, 0, (0, 255, 0))
        # cv2.imshow('img_with_kp', img_with_kp)
        # cv2.waitKey(0)
    print('')
    return tracks_data


def is_keypoint_inside_bbox(keypoint, bbox):
    x, y = keypoint.pt
    x1, y1 = bbox[0]
    x2, y2 = bbox[1]
    return x1 <= x <= x2 and y1 <= y <= y2


def draw_bbox_car(img_bgr, bbox, color=(0, 0, 255), thickness=6):
    img_bgr_copy = img_bgr.copy()
    pt1 = (int(bbox[0][0]), int(bbox[0][1]))
    pt2 = (int(bbox[1][0]), int(bbox[1][1]))
    img_bgr_copy = cv2.rectangle(img_bgr_copy, pt1, pt2, color, thickness)
    return img_bgr_copy


def filter_keypoints(tracks_data):
    for idx_frame, frame_data in enumerate(tracks_data):
        bbox_car = frame_data['shapes'][0]['points']
        kps = frame_data['kps_sift']
        descs = frame_data['descs_sift']

        idx_kps_to_keep = []
        for idx_kp in range(len(kps)):
            if is_keypoint_inside_bbox(kps[idx_kp], bbox_car):
                idx_kps_to_keep.append(idx_kp)
        
        kps_to_keep = [None] * len(idx_kps_to_keep)
        descs_to_keep = np.zeros((len(idx_kps_to_keep),descs.shape[1]), dtype=np.float32)
        for idx_kp in range(len(idx_kps_to_keep)):
            kps_to_keep[idx_kp] = kps[idx_kps_to_keep[idx_kp]]
            descs_to_keep[idx_kp] = descs[idx_kps_to_keep[idx_kp]]

        frame_data['kps_sift'] = kps_to_keep
        frame_data['descs_sift'] = descs_to_keep

        # img_color = frame_data['imageData']
        # img_color = draw_bbox_car(img_color, bbox_car)
        # img_with_kp = cv2.drawKeypoints(img_color, kps_to_keep, 0, (0, 255, 0))
        # cv2.imshow('img_with_kp', img_with_kp)
        # cv2.waitKey(0)

    # sys.exit(0)
    return tracks_data


def match_keypoints(track_data):
    # bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    index_params = dict(algorithm=0, trees=20) 
    search_params = dict(checks=150)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    for idx_frame in range(len(track_data)-1):
        # frame0_data = tracks_data[0]
        frame0_data = track_data[idx_frame]
        frame1_data = track_data[idx_frame+1]

        frame0_descs = frame0_data['descs_sift']
        frame1_descs = frame1_data['descs_sift']

        all_matches = flann.knnMatch(frame0_descs, frame1_descs, k=2)
        frame0_data[f"all_matches_sift_imagePath={frame0_data['imagePath']}_imagePath={frame1_data['imagePath']}"] = all_matches

        good_matches_idx = []
        good_matches = []
        good_matches_mask = [[0, 0] for i in range(len(all_matches))]
        for i, (m, n) in enumerate(all_matches): 
            if m.distance < 0.3*n.distance:
                good_matches_idx.append(i)
                good_matches.append(all_matches[i])
                good_matches_mask[i] = [1, 0]

        frame0_data[f"good_matches_sift_imagePath={frame0_data['imagePath']}_imagePath={frame1_data['imagePath']}"] = good_matches

        '''
        image0 = frame0_data['imageData']
        image1 = frame1_data['imageData']
        image0_cpy = image0.copy()
        image1_cpy = image1.copy()
        frame0_kps = frame0_data['kps_sift']
        frame1_kps = frame1_data['kps_sift']

        # COMPUTE CENTROIDS
        centroid_kps_good_matches_frame0 = np.zeros((2,), dtype=np.float32)
        centroid_kps_good_matches_frame1 = np.zeros((2,), dtype=np.float32)
        for (good_match, _) in good_matches:
            kp_good_match_frame0 = frame0_kps[good_match.queryIdx].pt
            kp_good_match_frame1 = frame1_kps[good_match.trainIdx].pt
            centroid_kps_good_matches_frame0 += kp_good_match_frame0
            centroid_kps_good_matches_frame1 += kp_good_match_frame1
        centroid_kps_good_matches_frame0 /= len(good_matches)
        centroid_kps_good_matches_frame1 /= len(good_matches)
        print('centroid_kps_good_matches_frame0:', centroid_kps_good_matches_frame0)
        print('centroid_kps_good_matches_frame1:', centroid_kps_good_matches_frame1)

        # ONLY FOR VISUALIZATION (TESTS)
        print('len(good_matches_idx):', len(good_matches_idx), '    good_matches_idx:', good_matches_idx)
        image0_cpy = cv2.putText(image0_cpy, frame0_data['imagePath'], org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness = 2, lineType=cv2.LINE_AA)
        image1_cpy = cv2.putText(image1_cpy, frame1_data['imagePath'], org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness = 2, lineType=cv2.LINE_AA)

        for (good_match, _) in good_matches:
            good_match_frame0_kp = (int(frame0_kps[good_match.queryIdx].pt[0]), int(frame0_kps[good_match.queryIdx].pt[1]))
            good_match_frame1_kp = (int(frame1_kps[good_match.trainIdx].pt[0]), int(frame1_kps[good_match.trainIdx].pt[1]))
            image0_cpy = cv2.circle(image0_cpy, center=good_match_frame0_kp, radius=5, color=(0,0,255), thickness=2)
            image1_cpy = cv2.circle(image1_cpy, center=good_match_frame1_kp, radius=5, color=(127,127,255), thickness=2)
        # sys.exit(0)

        image0_cpy = cv2.circle(image0_cpy, center=(int(centroid_kps_good_matches_frame0[0]), int(centroid_kps_good_matches_frame0[1])), radius=7, color=(255,0,0), thickness=4)
        image1_cpy = cv2.circle(image1_cpy, center=(int(centroid_kps_good_matches_frame1[0]), int(centroid_kps_good_matches_frame1[1])), radius=7, color=(255,0,0), thickness=4)

        Matched = cv2.drawMatchesKnn(image0_cpy, 
                                     frame0_kps, 
                                     image1_cpy, 
                                     frame1_kps, 
                                     all_matches, 
                                     outImg=None, 
                                     matchColor=(0, 155, 0), 
                                     singlePointColor=(0, 255, 255), 
                                     matchesMask=good_matches_mask, 
                                     flags=0
                                     )
        Matched = cv2.resize(Matched, (0, 0), fx=0.7, fy=0.7)
        cv2.imshow('Matched', Matched)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    return track_data
        

def compute_bounding_rectangle(points):
    x, y, w, h = cv2.boundingRect(points)
    bbox = [[x, y], [x+w, y+h]]
    return bbox


def compute_bbox_centroid_from_good_matches(track_data):
    for idx_frame in range(len(track_data)-1):
        frame0_data = track_data[idx_frame]
        frame1_data = track_data[idx_frame+1]

        frame0_kps = frame0_data['kps_sift']
        frame1_kps = frame1_data['kps_sift']

        # frame0_descs = frame0_data['descs_sift']
        # frame1_descs = frame1_data['descs_sift']

        # frame0_all_matches = frame0_data[f"all_matches_sift_imagePath={frame0_data['imagePath']}_imagePath={frame1_data['imagePath']}"]
        frame0_good_matches = frame0_data[f"good_matches_sift_imagePath={frame0_data['imagePath']}_imagePath={frame1_data['imagePath']}"]

        # frame0_centroid_good_matches = np.zeros((2,), dtype=np.float32)
        # frame1_centroid_good_matches = np.zeros((2,), dtype=np.float32)
        frame0_kps_good_matches = np.zeros((len(frame0_good_matches),2), dtype=np.float32)
        frame1_kps_good_matches = np.zeros((len(frame0_good_matches),2), dtype=np.float32)
        for idx_good_match, (good_match, _) in enumerate(frame0_good_matches):
            kp_good_match_frame0 = frame0_kps[good_match.queryIdx].pt
            kp_good_match_frame1 = frame1_kps[good_match.trainIdx].pt
            # frame0_centroid_good_matches += kp_good_match_frame0
            # frame1_centroid_good_matches += kp_good_match_frame1
            frame0_kps_good_matches[idx_good_match] = kp_good_match_frame0
            frame1_kps_good_matches[idx_good_match] = kp_good_match_frame1
        # frame0_centroid_good_matches /= len(frame0_good_matches)
        # frame1_centroid_good_matches /= len(frame0_good_matches)
        frame0_centroid_good_matches = frame0_kps_good_matches.mean(axis=0)
        frame1_centroid_good_matches = frame1_kps_good_matches.mean(axis=0)
        # print('frame0_kps_good_matches:', frame0_kps_good_matches)
        # print('frame1_kps_good_matches:', frame1_kps_good_matches)
        # print('frame0_centroid_good_matches:', frame0_centroid_good_matches)
        # print('frame1_centroid_good_matches:', frame1_centroid_good_matches)
        # sys.exit(0)
        frame0_data['centroid_good_matches_frame'] = frame0_centroid_good_matches
        frame0_data['centroid_good_matches_frame+1'] = frame1_centroid_good_matches

        frame0_bounding_rect_good_matches = compute_bounding_rectangle(frame0_kps_good_matches)
        frame1_bounding_rect_good_matches = compute_bounding_rectangle(frame1_kps_good_matches)

        frame0_data['bounding_rect_good_matches_frame'] = frame0_bounding_rect_good_matches
        frame0_data['bounding_rect_good_matches_frame+1'] = frame1_bounding_rect_good_matches
    return track_data


def plot_track_patterns(writer, track_name, orig_track_patterns, corr_track_patterns):
    # print('orig_track_patterns:', orig_track_patterns)
    # sys.exit(0)
    for idx_frame in range(len(orig_track_patterns)):
        if not orig_track_patterns[idx_frame] is None:
            orig_frame_patterns = orig_track_patterns[idx_frame]
            corr_frame_patterns = corr_track_patterns[idx_frame]
            # print('orig_track_pattern:', orig_track_pattern)
            # sys.exit(0)
            for key_frame_type in orig_frame_patterns.keys():
                for key_pattern in orig_frame_patterns[key_frame_type].keys():
                    # print('orig_frame_patterns[key_frame_type]:', orig_frame_patterns[key_frame_type])
                    # sys.exit(0)
                    writer.add_scalars(f'{track_name}_{key_frame_type}/{key_pattern}',
                            {'orig': orig_frame_patterns[key_frame_type][key_pattern],
                             'pred': corr_frame_patterns[key_frame_type][key_pattern]},
                            idx_frame)
                    writer.flush()

                    # for idx_frame, (orig_track_pattern, corr_track_pattern) in enumerate(zip(orig_track_patterns, corr_track_patterns)):
                    #     writer.add_scalars(f'{track_name}/{key_pattern}',
                    #         {'orig': orig_track_pattern[key_pattern],
                    #         'corr': corr_track_pattern[key_pattern]},
                    #         idx_frame)
    # sys.exit(0) 
    writer.close()


def main_detect_match_keypoints(args, original_tracks, groundtruth_tracks_folder, path_save_plots):
    date_time = datetime.now().strftime("%m%d%Y_%H%M%S")
    path_save_plots = path_save_plots + '/' + date_time
    os.makedirs(path_save_plots)
    writer = SummaryWriter(path_save_plots)

    groundtruth_tracks = make_groundtruth_tracks_paths(original_tracks, groundtruth_tracks_folder)

    for orig_track, gt_track in zip(original_tracks, groundtruth_tracks):
        print('orig_track:', orig_track)
        print('gt_track:', gt_track)
        track_name = orig_track.split('/')[-1]
        print(f'Searching files of track \'{orig_track}\'')
        orig_track_files = find_track_files(orig_track, '.json')
        print(f'Searching files of track \'{gt_track}\'')
        gt_track_files = find_track_files(gt_track, '.json')
        # print('len(orig_track_files):', len(orig_track_files))
        # print('len(gt_track_files):', len(gt_track_files))
        # sys.exit(0)

        print(f'Filtering track files...')
        filter_orig_track_files = filter_orig_track_files_by_groundtruth(orig_track_files, gt_track_files)

        print(f'Loading files of track \'{orig_track}\'')
        orig_track_data = load_track_data(filter_orig_track_files)
        print(f'Loading files of track \'{gt_track}\'')
        gt_track_data = load_track_data(gt_track_files)
        # print('orig_track_shapes:', orig_track_shapes)
        # for orig_track in orig_track_shapes:
        #     print('orig_track:', orig_track)
        # sys.exit(0)

        print('Detecting and describing keypoints...')
        orig_track_data_with_kp_desc = detect_describe_keypoints(orig_track_data)
        gt_track_data_with_kp_desc = detect_describe_keypoints(gt_track_data)
        # sys.exit(0)

        print('Filtering keypoints inside car bbox...')
        orig_track_data_with_kp_desc = filter_keypoints(orig_track_data)
        gt_track_data_with_kp_desc = filter_keypoints(gt_track_data)

        print('Matching keypoints....')
        orig_track_data_with_kp_desc_match = match_keypoints(orig_track_data_with_kp_desc)
        gt_track_data_with_kp_desc_match = match_keypoints(gt_track_data_with_kp_desc)

        print('Computing centroids....')
        orig_track_data_with_kp_desc_matches_bbox_centroid = compute_bbox_centroid_from_good_matches(orig_track_data_with_kp_desc_match)
        gt_track_data_with_kp_desc_matches_bbox_centroid = compute_bbox_centroid_from_good_matches(gt_track_data_with_kp_desc_match)
        # sys.exit(0)

        print('Predicting shapes points and computing patterns...')
        orig_track_patterns = predict_track_compute_patterns_bbox_car_lp(args, orig_track_data_with_kp_desc_matches_bbox_centroid)
        gt_track_patterns = predict_track_compute_patterns_bbox_car_lp(args, gt_track_data_with_kp_desc_matches_bbox_centroid)

        print('Saving charts...')
        plot_track_patterns(writer, track_name, orig_track_patterns, gt_track_patterns)
        # plot_track_patterns(writer, track_name, gt_track_patterns)
        print('----------')

    # print(f'Logs saved in \'{path_save_plots}\'')
    print('----------')



if __name__ == '__main__':
    args = parse_args()

    # original_tracks = [
    #     './vehicle_tracks_valfride/vehicle_000010_NoFrameRepetition',
    #     './vehicle_tracks_valfride/vehicle_000004',
    #     './vehicle_tracks_valfride/vehicle_000006',
    #     './vehicle_tracks_valfride/vehicle_000010',
    # ]

    original_tracks = [
        './vehicle_tracks_valfride/vehicle_000010_NoFrameRepetition'
    ]

    groundtruth_tracks_folder = 'groundtruth'

    path_save_plots = './logs_analysis/keypoints'

    main_detect_match_keypoints(args, original_tracks, groundtruth_tracks_folder, path_save_plots)

    print('\nFinished!')
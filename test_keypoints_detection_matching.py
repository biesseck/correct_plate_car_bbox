import os, sys
import numpy as np
import json
import sklearn
import argparse
import glob
from datetime import datetime
import cv2

import torch
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    return None


def make_corrected_tracks_paths(original_tracks, corrected_tracks_folder):
    corrected_tracks = [None] * len(original_tracks)
    for idx, original_track in enumerate(original_tracks):
        assert os.path.isdir(original_track), f'Error, could not find original track path: \'{original_track}\''
        original_track = original_track.rstrip('/')
        parent_path = '/'.join(original_track.split('/')[:-1])
        track_name = original_track.split('/')[-1]
        corrected_tracks[idx] = os.path.join(parent_path, corrected_tracks_folder, track_name)
        assert os.path.isdir(corrected_tracks[idx]), f'Error, corrected track path is not valid: \'{corrected_tracks[idx]}\''
    #     print('corrected_tracks[idx]:', corrected_tracks[idx])
    # sys.exit(0)
    return corrected_tracks


def find_track_files(path_track, ext='.json'):
    assert os.path.isdir(path_track), f'Error, could not find path \'{path_track}\''
    pattern_json = os.path.join(path_track, f"*.{ext.lstrip('*').lstrip('.')}")
    paths_jsons = glob.glob(pattern_json)
    paths_jsons.sort()
    # for path_json in paths_jsons:
    #     print('path_json:', path_json)
    # sys.exit(0)
    return paths_jsons


def filter_orig_track_files_by_corrected(orig_track_files, corr_track_files):
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


def compute_patterns_car_lp_shapes(track_shapes):
    frames_patterns = []
    for idx_frame, frame_shape in enumerate(track_shapes):
        car_shape, lp_shape = frame_shape
        assert car_shape['label'] == 'car', f'Error, car_shape[\'label\'] != \'car\'. It should be \'car\'.'
        assert lp_shape['label'] == 'lp', f'Error, lp_shape[\'label\'] != \'lp\'. It should be \'lp\'.'
        # print('idx_frame:', idx_frame, '    car_shape:', car_shape, '    lp_shape:', lp_shape)
    
        car_coords = get_coords_shape_labelme_format(car_shape['points'])
        lp_coords = get_coords_shape_labelme_format(lp_shape['points'])
        frame_patterns = get_patterns_shapes(car_coords, lp_coords)
        
        frames_patterns.append(frame_patterns)
        # print('car_coords:', car_coords)
        # print('lp_coords:', lp_coords)
        # print('frame_patterns:', frame_patterns)
        # sys.exit(0)
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


def match_keypoints(tracks_data):
    # bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    index_params = dict(algorithm=0, trees=20) 
    search_params = dict(checks=150)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    for idx_frame in range(len(tracks_data)-1):
        # frame0_data = tracks_data[0]
        frame0_data = tracks_data[idx_frame]
        frame1_data = tracks_data[idx_frame+1]

        frame0_descs = frame0_data['descs_sift']
        frame1_descs = frame1_data['descs_sift']

        all_matches = flann.knnMatch(frame0_descs, frame1_descs, k=2)
        frame0_data[f"match_sift_imagePath={frame0_data['imagePath']}_imagePath={frame1_data['imagePath']}"] = all_matches

        good_matches_idx = []
        good_matches_mask = [[0, 0] for i in range(len(all_matches))]
        for i, (m, n) in enumerate(all_matches): 
            if m.distance < 0.5*n.distance: 
                good_matches_mask[i] = [1, 0]
                good_matches_idx.append(i)

        # ONLY FOR VISUALIZATION (TESTS)
        print('len(good_matches_idx):', len(good_matches_idx), '    good_matches_idx:', good_matches_idx)
        image0 = frame0_data['imageData']
        image1 = frame1_data['imageData']
        image0_cpy = image0.copy()
        image1_cpy = image1.copy()
        image0_cpy = cv2.putText(image0_cpy, frame0_data['imagePath'], org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness = 2, lineType=cv2.LINE_AA)
        image1_cpy = cv2.putText(image1_cpy, frame1_data['imagePath'], org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness = 2, lineType=cv2.LINE_AA)
        frame0_kps = frame0_data['kps_sift']
        frame1_kps = frame1_data['kps_sift']
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

    return tracks_data
        

def plot_track_patterns(writer, track_name, orig_track_patterns, corr_track_patterns):
    for key_pattern in orig_track_patterns[0].keys():
        for idx_frame, (orig_track_pattern, corr_track_pattern) in enumerate(zip(orig_track_patterns, corr_track_patterns)):
            writer.add_scalars(f'{track_name}/{key_pattern}',
                {'orig': orig_track_pattern[key_pattern],
                 'corr': corr_track_pattern[key_pattern]},
                 idx_frame)
    # sys.exit(0) 


def main_detect_match_keypoints(args, original_tracks, corrected_tracks_folder, path_save_plots):
    date_time = datetime.now().strftime("%m%d%Y_%H%M%S")
    path_save_plots = path_save_plots + '/' + date_time
    os.makedirs(path_save_plots)
    writer = SummaryWriter(path_save_plots)

    corrected_tracks = make_corrected_tracks_paths(original_tracks, corrected_tracks_folder)

    for orig_track, corr_track in zip(original_tracks, corrected_tracks):
        print('orig_track:', orig_track)
        print('corr_track:', corr_track)
        track_name = orig_track.split('/')[-1]
        print(f'Searching files of track \'{orig_track}\'')
        orig_track_files = find_track_files(orig_track, '.json')
        print(f'Searching files of track \'{corr_track}\'')
        corr_track_files = find_track_files(corr_track, '.json')
        # print('len(orig_track_files):', len(orig_track_files))
        # print('len(corr_track_files):', len(corr_track_files))
        # sys.exit(0)

        print(f'Filtering track files...')
        filter_orig_track_files = filter_orig_track_files_by_corrected(orig_track_files, corr_track_files)

        print(f'Loading files of track \'{orig_track}\'')
        orig_track_data = load_track_data(filter_orig_track_files)
        print(f'Loading files of track \'{corr_track}\'')
        corr_track_data = load_track_data(corr_track_files)
        # print('orig_track_shapes:', orig_track_shapes)
        # for orig_track in orig_track_shapes:
        #     print('orig_track:', orig_track)
        # sys.exit(0)

        print('Detecting and describing keypoints...')
        orig_track_data_with_kp_desc = detect_describe_keypoints(orig_track_data)
        # sys.exit(0)

        print('Filtering keypoints inside car bbox...')
        orig_track_data_with_kp_desc = filter_keypoints(orig_track_data)

        print('Matching keypoints....')
        orig_track_data_with_kp_desc_match = match_keypoints(orig_track_data_with_kp_desc)

        # sys.exit(0)

        # orig_track_patterns = compute_patterns_car_lp_shapes(orig_track_data)
        # corr_track_patterns = compute_patterns_car_lp_shapes(corr_track_data)
        # for orig_track_pattern in orig_track_patterns:
        #     print('orig_track_pattern:', orig_track_pattern)

        # print('Saving charts...')
        # plot_track_patterns(writer, track_name, orig_track_patterns, corr_track_patterns)
        print('----------')

    # print(f'Logs saved in \'{path_save_plots}\'')
    print('----------')



if __name__ == '__main__':
    args = parse_args()

    # original_tracks = [
    #     './vehicle_tracks_valfride/vehicle_000004',
    #     './vehicle_tracks_valfride/vehicle_000006',
    #     './vehicle_tracks_valfride/vehicle_000010',
    #     './vehicle_tracks_valfride/vehicle_000010_NoFrameRepetition',
    # ]

    original_tracks = [
        './vehicle_tracks_valfride/vehicle_000010_NoFrameRepetition'
    ]

    corrected_tracks_folder = 'corrected'

    # corrected_tracks = [
    #     './vehicle_tracks_valfride/corrected/vehicle_000004',
    #     './vehicle_tracks_valfride/corrected/vehicle_000006',
    #     './vehicle_tracks_valfride/corrected/vehicle_000010',
    # ]

    path_save_plots = './logs_analysis/keypoints'

    main_detect_match_keypoints(args, original_tracks, corrected_tracks_folder, path_save_plots)

    print('\nFinished!')
import os, sys
import numpy as np
import json
import sklearn
import argparse
import glob
from datetime import datetime

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


def find_track_files(path_track):
    assert os.path.isdir(path_track), f'Error, could not find path \'{path_track}\''
    pattern_json = os.path.join(path_track, '*.json')
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
    shapes = data['shapes']
    return shapes


def load_track_shapes(paths_files_track):
    track_shapes = [None] * len(paths_files_track)
    for idx_file, file_path in enumerate(paths_files_track):
        frame_shapes = load_track_json(file_path)
        track_shapes[idx_file] = frame_shapes
    return track_shapes


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

    frame_patterns = {}
    frame_patterns['00_car_width'] = car_width
    frame_patterns['01_car_height'] = car_height
    frame_patterns['02_lp_width'] = lp_width
    frame_patterns['03_lp_height'] = lp_height
    frame_patterns['04_width_ratio'] = car_width / lp_width
    frame_patterns['05_height_ratio'] = car_height / lp_height
    frame_patterns['06_horiz_posit_ratio'] = car_width / lp_coords['x1']
    frame_patterns['07_verti_posit_ratio'] = car_height / lp_coords['y1']

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
        

def plot_track_patterns(writer, track_name, orig_track_patterns, corr_track_patterns):
    for key_pattern in orig_track_patterns[0].keys():
        for idx_frame, (orig_track_pattern, corr_track_pattern) in enumerate(zip(orig_track_patterns, corr_track_patterns)):
            writer.add_scalars(f'{track_name}/{key_pattern}',
                {'orig': orig_track_pattern[key_pattern],
                'corr': corr_track_pattern[key_pattern],},
                idx_frame)
    # sys.exit(0) 


def main_analyse(args, original_tracks, corrected_tracks_folder, path_save_plots):
    date_time = datetime.now().strftime("%m%d%Y_%H%M%S")
    path_save_plots = path_save_plots + '/' + date_time
    os.makedirs(path_save_plots)
    writer = SummaryWriter(path_save_plots)

    corrected_tracks = make_corrected_tracks_paths(original_tracks, corrected_tracks_folder)

    for orig_track, corr_track in zip(original_tracks, corrected_tracks):
        print('orig_track:', orig_track)
        print('corr_track:', corr_track)
        track_name = orig_track.split('/')[-1]
        orig_track_files = find_track_files(orig_track)
        corr_track_files = find_track_files(corr_track)
        # print('len(orig_track_files):', len(orig_track_files))
        # print('len(corr_track_files):', len(corr_track_files))
        # sys.exit(0)

        filter_orig_track_files = filter_orig_track_files_by_corrected(orig_track_files, corr_track_files)

        orig_track_shapes = load_track_shapes(filter_orig_track_files)
        corr_track_shapes = load_track_shapes(corr_track_files)

        orig_track_patterns = compute_patterns_car_lp_shapes(orig_track_shapes)
        corr_track_patterns = compute_patterns_car_lp_shapes(corr_track_shapes)
        # for orig_track_pattern in orig_track_patterns:
        #     print('orig_track_pattern:', orig_track_pattern)

        print('Saving charts...')
        plot_track_patterns(writer, track_name, orig_track_patterns, corr_track_patterns)

        print('----------')

if __name__ == '__main__':
    args = parse_args()

    original_tracks = [
        '/home/biesseck/GitHub/bjgbiesseck_correct_plate_car_bbox/vehicle_tracks_valfride/vehicle_000004',
        '/home/biesseck/GitHub/bjgbiesseck_correct_plate_car_bbox/vehicle_tracks_valfride/vehicle_000006',
        '/home/biesseck/GitHub/bjgbiesseck_correct_plate_car_bbox/vehicle_tracks_valfride/vehicle_000010',
    ]

    corrected_tracks_folder = 'corrected'

    # corrected_tracks = [
    #     '/home/biesseck/GitHub/bjgbiesseck_correct_plate_car_bbox/vehicle_tracks_valfride/corrected/vehicle_000004',
    #     '/home/biesseck/GitHub/bjgbiesseck_correct_plate_car_bbox/vehicle_tracks_valfride/corrected/vehicle_000006',
    #     '/home/biesseck/GitHub/bjgbiesseck_correct_plate_car_bbox/vehicle_tracks_valfride/corrected/vehicle_000010',
    # ]

    path_save_plots = './logs_analysis'

    main_analyse(args, original_tracks, corrected_tracks_folder, path_save_plots)

    print('\nFinished!')
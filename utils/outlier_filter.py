from utils import DataReader
import os
import json
from tqdm import tqdm

target_folder = 'data/train_annotation'
file_name = 'annotation.txt'


def landmark_num_checker(point_clouds):
    valid_number = {'yellow': 15, 'red': 17, 'green': 10, 'blue': 9}
    for c in valid_number:
        if valid_number[c] != len(point_clouds[c]):
            print('---'*10)
            print('[ERR NUM NOT AGREE] {} should be {}, but found {}'.format(c, valid_number[c], len(point_clouds[c])))
            return False
    return True
    
def location_filter(point_clouds):
    results = {}
    for color in point_clouds:
        results[color] = []
        for x, y in point_clouds[color]:
            if x > 300 and y > 300:
                results[color].append([x, y])
    return results


if __name__ == "__main__":
    target_case = os.listdir(target_folder)
    failure_count = 0
    cannot_fix = 0
    with tqdm(target_case) as t:
        for case_name in t:
            point_clouds = DataReader.parse_annotation(os.path.join(target_folder, case_name, file_name))
            stats = {c: len(point_clouds[c]) for c in point_clouds}
            if not landmark_num_checker(point_clouds):
                print("case '{}' is not legal\n{}".format(case_name, json.dumps(point_clouds)))
                new_point_clouds = location_filter(point_clouds)
                if not landmark_num_checker(new_point_clouds):
                    print("location filtered case '{}' is not legal\n{}".format(case_name, json.dumps(new_point_clouds)))
                    cannot_fix += 1
                failure_count += 1
            t.set_postfix(stats)
    print("Summary")
    print("In folder {}, {} failures found, {} cannot be auto-fixed".format(target_folder, failure_count, cannot_fix))
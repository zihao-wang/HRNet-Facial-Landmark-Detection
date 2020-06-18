import os
from collections import defaultdict

import numpy as np
import ot
from tqdm import tqdm

from utils import DataReader

np.set_printoptions(precision=2)

golden_train_dir = "data/golden_train_annotation"
train_dir = "data/test/non-adult_annotation"
sorted_train_dir = "data/test/sorted_non-adult_annotation"

fail_list = open("data/fail_point_cloud_align.txt", 'wt')

golden_cases = [f for f in os.listdir(golden_train_dir) 
                if os.path.isdir(os.path.join(golden_train_dir, f))]
train_cases = [f for f in os.listdir(train_dir)
                if os.path.isdir(os.path.join(train_dir, f))]
train_cases = list(set(train_cases).difference(list(golden_cases)))

print(golden_cases, train_cases)

def save_refined_annotation(dir, case, annotation):
    os.makedirs(os.path.join(sorted_train_dir, case), exist_ok=True)
    with open(os.path.join(sorted_train_dir, case, 'annotation.txt'), 'wt') as f:
        for color in annotation:
            f.write(color + "\n")
            for xy in annotation[color]:
                f.write("{}, {}\n".format(xy[0], xy[1]))

def align_annotation(golden_annotations, annotation, align_method='W'):
    re_ranking = {}
    for k, ref_pclist in golden_annotations.items():
        if not k in annotation:
            return None
        pc = annotation[k]
        P = 0
        L = len(pc)
        for rpc in ref_pclist:
            if not len(rpc) == len(pc):
                return None
            err = np.sum(np.abs(pc - rpc))
            if align_method == 'W':
                dP = ot.bregman.empirical_sinkhorn(
                    pc, rpc, 0.1, method='sinkhorn_stabilized')
            elif align_method == 'GW':
                C_pc = ot.utils.dist(pc, pc)
                C_rpc = ot.utils.dist(rpc, rpc)
                p = ot.utils.unif(len(pc))
                q = ot.utils.unif(len(rpc))
                dP = ot.gromov.gromov_wasserstein(
                    C_pc, C_rpc, p, q, loss_fun='square_loss')
            elif align_method == 'FGW':
                C_pc = ot.utils.dist(pc, pc)
                C_rpc = ot.utils.dist(rpc, rpc)
                M = ot.utils.dist(pc, rpc)
                p = ot.utils.unif(len(pc))
                q = ot.utils.unif(len(rpc))
                dP = ot.gromov.fused_gromov_wasserstein(
                    M, C_pc, C_rpc, p, q, 
                    loss_fun='square_loss', alpha=0.2)
            H = - np.sum(dP * np.log(dP + 1e-5))
            # print(k, H)
            P = P + dP
            # print(k, np.argmax(dP, axis=1))
        re_ranking[k] = np.argmax(P, axis=1).tolist()

    return re_ranking


if __name__ == '__main__':
    golden_annotations = defaultdict(list)
    for case in golden_cases:
        file = os.path.join(golden_train_dir, case, "annotation.txt")
        annotation = DataReader.parse_annotation(file)
        for k, v in annotation.items():
            golden_annotations[k].append(np.asarray(v))
    t = tqdm(train_cases)
    for case in t:
        raw_annotation = os.path.join(train_dir, case, "annotation.txt")
        annotation = {k: np.asarray(v)
            for k, v in DataReader.parse_annotation(raw_annotation).items()}
        re_ranking = align_annotation(
            golden_annotations, annotation, align_method='FGW')
        if re_ranking:
            re_annotation = {k: np.asarray([v[i] for i in re_ranking[k]])
                            for k, v in annotation.items()}
            save_refined_annotation(sorted_train_dir, case, re_annotation)
        else:
            fail_list.write(case + "\n")
        t.set_postfix({"case": case})
    t.close()
    fail_list.close()

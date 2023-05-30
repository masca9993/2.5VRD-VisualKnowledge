import pandas as pd
from utils import prepare_dataset, prepare_dataset_across2_0
from evaluate_kenn import run_within_kenn
from evaluate_MLPwithin import run_MLPwithin
from evaluate_singleMLPwithin import run_singleMLPwithin
from scipy.stats import ttest_ind
import argparse
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('-nruns', help='number of runs of the test')
args = parser.parse_args()


objects_within_train = pd.read_csv("visual_relationship/subset_data/within_image_objects_train.csv")
vrd_within_train = pd.read_csv("visual_relationship/subset_data/within_image_vrd_train.csv")
objects_within_validation = pd.read_csv("visual_relationship/subset_data/within_image_objects_validation.csv")
vrd_within_validation = pd.read_csv("visual_relationship/subset_data/within_image_vrd_validation.csv")
objects_within_test = pd.read_csv("visual_relationship/subset_data/within_image_objects_test.csv")
vrd_within_test = pd.read_csv("visual_relationship/subset_data/within_image_vrd_test.csv")

train_within, y_train_within, train_ids_within = prepare_dataset(vrd_within_train, objects_within_train)
val_within, y_val_within, val_ids_within = prepare_dataset(vrd_within_validation, objects_within_validation)
test_within, y_test_within, test_ids_within = prepare_dataset(vrd_within_test, objects_within_test)


n_runs = args.nruns
f1_distanceMLP = []
f1_occlusionMLP = []

f1_distance_kenn = []
f1_occlusion_kenn = []

for i in range(int(n_runs)):
    print("\n -Starting run: ", i)
    k_dis, k_occ = run_within_kenn(train_within, y_train_within, val_within, y_val_within, test_within, y_test_within)
    mlp_dis, mlp_occ = run_singleMLPwithin(train_within, y_train_within, val_within,
                                           y_val_within, test_within, y_test_within)

    f1_distance_kenn.append(k_dis)
    f1_occlusion_kenn.append(k_occ)
    f1_distanceMLP.append(mlp_dis)
    f1_occlusionMLP.append(mlp_occ)


print("Kenn average f1 score Distance:", sum(f1_distance_kenn)/len(f1_distance_kenn))
print("MLP average f1 score Distance", sum(f1_distanceMLP)/len(f1_distanceMLP), "\n")
print("Kenn average f1 score Occlusion:", sum(f1_occlusion_kenn)/len(f1_occlusion_kenn))
print("MLP average f1 score Occlusion", sum(f1_occlusionMLP)/len(f1_occlusionMLP))

_, pvalue_dis = ttest_ind(f1_distance_kenn, f1_distanceMLP, equal_var=True)
_, pvalue_occ = ttest_ind(f1_occlusion_kenn, f1_occlusionMLP, equal_var=True)
print("p value Distance: ", pvalue_dis)
print("p value Occlusion: ", pvalue_occ)
#value, pvalue = ttest_ind(f1_scores_kenn_groundtruth, f1_scores_kenn, equal_var=True)
#print("p value among two kenns: ", pvalue)


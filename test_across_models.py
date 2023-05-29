import pandas as pd
from utils import prepare_dataset, prepare_dataset_across2_0, prepare_dataset_across, transitive_triplets_across, predict_across
from evaluate_kenn_across import run_across_kenn
from evaluate_MLPacross import run_acrossMLP
from scipy.stats import ttest_ind
import argparse
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('-nruns', help='number of runs of the test')
parser.add_argument('-computerel', action='store_true', help='Compute transitive relations of the objects compared in the test dataset')
parser.add_argument('-c_indexes', action='store_true', help='Compute the preactivations indexes corresponding to transitive relations')
args = parser.parse_args()


objects_across_train = pd.read_csv("visual_relationship/subset_data/across_images_objects_train.csv")
vrd_across_train = pd.read_csv("visual_relationship/subset_data/across_images_vrd_train.csv")
objects_across_validation = pd.read_csv("visual_relationship/subset_data/across_images_objects_validation.csv")
vrd_across_validation = pd.read_csv("visual_relationship/subset_data/across_images_vrd_validation.csv")
objects_across_test = pd.read_csv("visual_relationship/subset_data/across_images_objects_test.csv")
vrd_across_test = pd.read_csv("visual_relationship/subset_data/across_images_vrd_test.csv")

objects_within_train = pd.read_csv("visual_relationship/subset_data/within_image_objects_train.csv")
vrd_within_train = pd.read_csv("visual_relationship/subset_data/within_image_vrd_train.csv")
objects_within_validation = pd.read_csv("visual_relationship/subset_data/within_image_objects_validation.csv")
vrd_within_validation = pd.read_csv("visual_relationship/subset_data/within_image_vrd_validation.csv")
objects_within_test = pd.read_csv("visual_relationship/subset_data/within_image_objects_test.csv")
vrd_within_test = pd.read_csv("visual_relationship/subset_data/within_image_vrd_test.csv")

train_within, y_train_within, train_ids_within = prepare_dataset(vrd_within_train, objects_within_train)
val_within, y_val_within, val_ids_within = prepare_dataset(vrd_within_validation, objects_within_validation)
test_within, y_test_within, test_ids_within = prepare_dataset(vrd_within_test, objects_within_test)

trainMLP, y_trainMLP, train_idsMLP = prepare_dataset_across(vrd_across_train, objects_across_train)
valMLP, y_valMLP, val_idsMLP = prepare_dataset_across(vrd_across_validation, objects_across_validation)
testMLP, y_testMLP, test_idsMLP = prepare_dataset_across(vrd_across_test, objects_across_test)

train, y_train, train_dict, train_ids, across_ids_train = prepare_dataset_across2_0(vrd_across_train, objects_across_train)
val, y_val, val_dict, val_ids, across_ids_val = prepare_dataset_across2_0(vrd_across_validation, objects_across_validation)
test, y_test, test_dict, test_ids, across_ids_test = prepare_dataset_across2_0(vrd_across_test, objects_across_test)

if args.computerel:
    train_triplets = transitive_triplets_across(vrd_across_train, train_dict, train_ids_within)
    with open('train_transitive_triplets.pkl', 'wb') as file:
        pickle.dump(train_triplets, file)

    val_triplets = transitive_triplets_across(vrd_across_validation, val_dict, val_ids_within)
    with open('val_transitive_triplets.pkl', 'wb') as file:
        pickle.dump(val_triplets, file)

    test_triplets = transitive_triplets_across(vrd_across_test, test_dict, test_ids_within)
    with open('test_transitive_triplets.pkl', 'wb') as file:
        pickle.dump(test_triplets, file)
else:
    with open('train_transitive_triplets.pkl', 'rb') as file:
        train_triplets = pickle.load(file)
    with open('val_transitive_triplets.pkl', 'rb') as file:
        val_triplets = pickle.load(file)
    with open('test_transitive_triplets.pkl', 'rb') as file:
        test_triplets = pickle.load(file)

val_within_rel = [(val_dict[ids[0]][ids[1]], val_dict[ids[2]][ids[3]]) for ids in val_ids_within]
val_across_rel = [(val_dict[ids[0]][ids[1]], val_dict[ids[2]][ids[3]]) for ids in val_ids]

test_within_rel = [(test_dict[ids[0]][ids[1]], test_dict[ids[2]][ids[3]]) for ids in test_ids_within]
test_across_rel = [(test_dict[ids[0]][ids[1]], test_dict[ids[2]][ids[3]]) for ids in test_ids]

val_relations = val_within_rel + val_across_rel
test_relations = test_within_rel + test_across_rel

index_xy_train = []
index_yz_train = []
index_xz_train = []

if args.c_indexes:
    index_xy_val = []
    index_yz_val = []
    index_xz_val = []
    index_xy_test = []
    index_yz_test = []
    index_xz_test = []

    for rel in val_triplets:
        index_xy_val.append(val_relations.index(tuple(rel[:2])))
        index_yz_val.append(val_relations.index(tuple(rel[-2:])))
        index_xz_val.append(val_relations.index(tuple(rel[:1] + rel[-1:])))

    for rel in test_triplets:
        index_xy_test.append(test_relations.index(tuple(rel[:2])))
        index_yz_test.append(test_relations.index(tuple(rel[-2:])))
        index_xz_test.append(test_relations.index(tuple(rel[:1] + rel[-1:])))

    with open('indexes_lists.pickle', 'wb') as file:
        pickle.dump(index_xy_val, file)
        pickle.dump(index_yz_val, file)
        pickle.dump(index_xz_val, file)
        pickle.dump(index_xy_test, file)
        pickle.dump(index_yz_test, file)
        pickle.dump(index_xz_test, file)
else:
    with open('indexes_lists.pickle', 'rb') as file:
        index_xy_val = pickle.load(file)
        index_yz_val = pickle.load(file)
        index_xz_val = pickle.load(file)
        index_xy_test = pickle.load(file)
        index_yz_test = pickle.load(file)
        index_xz_test = pickle.load(file)


n_runs = args.nruns
f1_scoresMLP = []
f1_scores_kenn = []
#f1_scores_kenn_groundtruth = []

for i in range(int(n_runs)):
    #f1_scores_kenn_groundtruth.append(run_across_kenn(train, y_train, y_train_within, val, y_val, y_val_within, test, y_test, y_test_within, index_xy_val, index_yz_val, index_xz_val,
    #                    index_xy_test, index_yz_test, index_xz_test))
    f1_scores_kenn.append(run_across_kenn(train, y_train, train_within, val, y_val, val_within, test, y_test, test_within, index_xy_val, index_yz_val, index_xz_val,
                        index_xy_test, index_yz_test, index_xz_test))
    f1_scoresMLP.append(run_acrossMLP(train, y_train, val, y_val, test, y_test))


#print("Kenn with 'within' groundtruth average f1 score:", sum(f1_scores_kenn_groundtruth)/len(f1_scores_kenn_groundtruth))
print("Kenn average f1 score:", sum(f1_scores_kenn)/len(f1_scores_kenn))
print("MLP average f1 score", sum(f1_scoresMLP)/len(f1_scoresMLP))
_, pvalue = ttest_ind(f1_scores_kenn, f1_scoresMLP, equal_var=True)
print("p value: ", pvalue)
#value, pvalue = ttest_ind(f1_scores_kenn_groundtruth, f1_scores_kenn, equal_var=True)
#print("p value among two kenns: ", pvalue)


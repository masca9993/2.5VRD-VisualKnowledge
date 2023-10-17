import numpy as np
import pandas as pd
import torch


def img_obj_dict(array):
    unique_rows = np.unique([tuple(row) for row in array], axis=0)
    numeric_ids = np.arange(len(unique_rows))
    dictionary = dict(zip(map(tuple, unique_rows), numeric_ids))

    new_dict = {}

    for key, value in dictionary.items():
        a = key[0]
        if a not in new_dict:
            new_dict[a] = {}
        new_dict[a][key[1]] = value

    return new_dict


def within_relations_dict(comparison_ids, rel_dict):
    within_rel_dict = {}
    for comparison in comparison_ids:
        if comparison[0] in rel_dict and comparison[2] in rel_dict and comparison[1] in rel_dict[comparison[0]] and comparison[3] in rel_dict[comparison[2]]:
            if comparison[0] not in within_rel_dict:
                within_rel_dict[comparison[0]] = [[rel_dict[comparison[0]][comparison[1]], rel_dict[comparison[2]][comparison[3]]]]
            else:
                within_rel_dict[comparison[0]].append([rel_dict[comparison[0]][comparison[1]], rel_dict[comparison[2]][comparison[3]]])

    return within_rel_dict


def prepare_dataset(vrd: pd.DataFrame, objects: pd.DataFrame, pct=1):

    vrd_1 = vrd.merge(objects, how="left",
                      left_on=["image_id_1", "object_id_1"],
                      right_on=["image_id", "object_id"],
                      ).rename(
        columns={"xmin": "xmin1", "xmax": "xmax1", "ymin": "ymin1", "ymax": "ymax1"}).drop(
        columns=['image_id', 'object_id', 'entity'])

    data = vrd_1.merge(objects, how="left",
                       left_on=["image_id_2", "object_id_2"],
                       right_on=["image_id", "object_id"],
                       ).rename(
        columns={"xmin": "xmin2", "xmax": "xmax2", "ymin": "ymin2", "ymax": "ymax2"}).drop(
        columns=['image_id', 'object_id', 'entity'])

    data = data[data["distance"] != -1]
    data = data[data["occlusion"] != -1]

    sym_data = pd.DataFrame()
    sym_data["image_id_1"] = data["image_id_2"]
    sym_data["object_id_1"] = data["object_id_2"]
    sym_data["image_id_2"] = data["image_id_1"]
    sym_data["object_id_2"] = data["object_id_1"]
    t = np.where(data["distance"] == 2, 1, data["distance"])
    sym_data["distance"] = t
    sym_data["distance"] = np.where(data["distance"] == 1, 2, t)
    t = np.where(data["occlusion"] == 2, 1, data["occlusion"])
    sym_data["occlusion"] = t
    sym_data["occlusion"] = np.where(data["occlusion"] == 1, 2, t)

    vrd_1 = sym_data.merge(objects, how="left",
                      left_on=["image_id_1", "object_id_1"],
                      right_on=["image_id", "object_id"],
                      ).rename(
        columns={"xmin": "xmin1", "xmax": "xmax1", "ymin": "ymin1", "ymax": "ymax1"}).drop(
        columns=['image_id', 'object_id', 'entity'])

    sym_data = vrd_1.merge(objects, how="left",
                       left_on=["image_id_2", "object_id_2"],
                       right_on=["image_id", "object_id"],
                       ).rename(
        columns={"xmin": "xmin2", "xmax": "xmax2", "ymin": "ymin2", "ymax": "ymax2"}).drop(
        columns=['image_id', 'object_id', 'entity'])

    data = pd.concat([data, sym_data]).sample(frac = pct)

    y = data[["distance", "occlusion"]]
    data.drop(columns=["raw_distance", "raw_occlusion", "distance", "occlusion"], inplace=True)
    data_ids = data[["image_id_1", "object_id_1", "image_id_2", "object_id_2"]].astype(str)
    data.drop(columns=["image_id_1", "object_id_1", "image_id_2", "object_id_2"], inplace=True)

    return torch.tensor(data.values, dtype=torch.float32), torch.tensor(y.values), data_ids.values


def prepare_dataset_across(vrd: pd.DataFrame, objects: pd.DataFrame):

    vrd_1 = vrd.merge(objects, how="left",
                      left_on=["image_id_1", "object_id_1"],
                      right_on=["image_id", "object_id"],
                      ).rename(
        columns={"xmin": "xmin1", "xmax": "xmax1", "ymin": "ymin1", "ymax": "ymax1"}).drop(
        columns=['image_id', 'object_id', 'entity'])

    data = vrd_1.merge(objects, how="left",
                       left_on=["image_id_2", "object_id_2"],
                       right_on=["image_id", "object_id"],
                       ).rename(
        columns={"xmin": "xmin2", "xmax": "xmax2", "ymin": "ymin2", "ymax": "ymax2"}).drop(
        columns=['image_id', 'object_id', 'entity'])

    data = data[data["distance"] != -1]

    y = data[["distance"]]
    data.drop(columns=["raw_distance", "raw_occlusion", "distance", "occlusion"], inplace=True)
    data_ids = data[["image_id_1", "object_id_1", "image_id_2", "object_id_2"]].astype(str)
    data.drop(columns=["image_id_1", "object_id_1", "image_id_2", "object_id_2"], inplace=True)

    return torch.tensor(data.values, dtype=torch.float32), torch.squeeze(torch.tensor(y.values)), data_ids.values


def prepare_dataset_across2_0(vrd: pd.DataFrame, objects: pd.DataFrame):

    vrd_1 = vrd.merge(objects, how="left",
                      left_on=["image_id_1", "object_id_1"],
                      right_on=["image_id", "object_id"],
                      ).rename(
        columns={"xmin": "xmin1", "xmax": "xmax1", "ymin": "ymin1", "ymax": "ymax1"}).drop(
        columns=['image_id', 'object_id', 'entity'])

    data = vrd_1.merge(objects, how="left",
                       left_on=["image_id_2", "object_id_2"],
                       right_on=["image_id", "object_id"],
                       ).rename(
        columns={"xmin": "xmin2", "xmax": "xmax2", "ymin": "ymin2", "ymax": "ymax2"}).drop(
        columns=['image_id', 'object_id', 'entity'])

    vrd_ids = data[["image_id_1", "object_id_1", "image_id_2", "object_id_2"]].astype(str)
    object_newdict = img_obj_dict(np.concatenate((vrd_ids.values[:, :2], vrd_ids.values[:, 2:])))

    data = data[data["distance"] != -1]

    y = data[["distance"]]

    #id_to_obj = {value: key for key, value in object_dict.items()}

    data.drop(columns=["raw_distance", "raw_occlusion", "distance", "occlusion"], inplace=True)
    data_ids = data[["image_id_1", "object_id_1", "image_id_2", "object_id_2"]].astype(str)
    comparison_ids = data[["image_id_1", "image_id_2"]].astype(str).itertuples(index=False, name=None)
    data.drop(columns=["image_id_1", "object_id_1", "image_id_2", "object_id_2"], inplace=True)

    return torch.tensor(data.values, dtype=torch.float32), torch.squeeze(torch.tensor(y.values)), object_newdict, data_ids.values, list(set([i for i in comparison_ids]))

def find_transitive_triplets(relations):
    all_transitive_triplets = []

    for rel1 in relations:
        for rel2 in relations:
            if rel1[1] == rel2[0]:
                all_transitive_triplets.append([rel1[0], rel1[1], rel2[1]])

    triplets = all_transitive_triplets.copy()

    for tr in all_transitive_triplets:
        if [tr[0], tr[2]] not in relations:
            triplets.remove(tr)

    return triplets

'''
def transitive_triplets(vrd_across_test: pd.DataFrame, test_dict, within_relations):

    image_id_1_prev = vrd_across_test["image_id_1"].iloc[0]
    image_id_2_prev = vrd_across_test["image_id_2"].iloc[0]
    transitive_triplets = []
    across_relations = []

    for index, row in vrd_across_test.iterrows():
        if row["image_id_1"] == image_id_1_prev and row["image_id_2"] == image_id_2_prev:
            across_relations.append([test_dict[image_id_1_prev][str(row["object_id_1"])],
                                     test_dict[image_id_2_prev][str(row["object_id_2"])]])
        else:
            if image_id_1_prev in within_relations:
                within_relations = within_relations[image_id_1_prev]
            else:
                within_relations = []
            if image_id_2_prev in within_relations:
                within_relations.append(within_relations[image_id_2_prev][0])
            images_relations = across_relations + within_relations
            transitive_triplets += find_transitive_triplets(images_relations)

            across_relations = []
            image_id_1_prev = row["image_id_1"]
            image_id_2_prev = row["image_id_2"]
            across_relations.append([test_dict[image_id_1_prev][str(row["object_id_1"])],
                                     test_dict[image_id_2_prev][str(row["object_id_2"])]])

    with open('transitive_triplets.pkl', 'wb') as file:
        pickle.dump(transitive_triplets, file) '''

def find_value(lst, target):
    for i, sublist in enumerate(lst):
        if isinstance(sublist, list):
            for j, subsublist in enumerate(sublist):
                if isinstance(subsublist, list):
                    for k, value in enumerate(subsublist):
                        if value == target:
                            return [i, j, k]
                elif subsublist == target:
                    return [i, j]
        elif sublist == target:
            return [i]
    return None


def remove_within_triplets(triplets, within_dict):
    triplets = np.asarray(triplets)
    x = triplets[:, 0]
    z = triplets[:, 2]
    values = list(within_dict.values())
    mask = []
    for i in range(x.shape[0]):
        if find_value(values, x[i]) is not None and find_value(values, z[i]) is not None:
            index_x = find_value(values, x[i])[0]
            index_z = find_value(values, z[i])[0]
            mask.append(list(within_dict.keys())[index_x] == list(within_dict.keys())[index_z])
        else:
            mask.append(False)

    mask = np.array(mask)
    return triplets[~mask, :].tolist()  #, triplets[mask, :]


def transitive_triplets_across(vrd_across: pd.DataFrame, ids_dict, ids_within):
    all_within_relations = within_relations_dict(ids_within, ids_dict)

    vrd_across = vrd_across[vrd_across["distance"] != -1]

    image_id_1_prev = vrd_across["image_id_1"].iloc[0]
    image_id_2_prev = vrd_across["image_id_2"].iloc[0]
    transitive_triplets = []
    across_relations = []

    for index, row in vrd_across.iterrows():
        if row["image_id_1"] == image_id_1_prev and row["image_id_2"] == image_id_2_prev:
            across_relations.append([ids_dict[image_id_1_prev][str(row["object_id_1"])],
                                     ids_dict[image_id_2_prev][str(row["object_id_2"])]])
        else:
            if image_id_1_prev in all_within_relations:
                within_relations = all_within_relations[image_id_1_prev]
            else:
                within_relations = []
            if image_id_2_prev in all_within_relations:
                within_relations.append(all_within_relations[image_id_2_prev][0])
            images_relations = across_relations + within_relations
            prov_triplets = find_transitive_triplets(images_relations)
            if len(prov_triplets) > 0:
                transitive_triplets += remove_within_triplets(prov_triplets, all_within_relations)

            across_relations = []
            image_id_1_prev = row["image_id_1"]
            image_id_2_prev = row["image_id_2"]
            across_relations.append([ids_dict[image_id_1_prev][str(row["object_id_1"])],
                                     ids_dict[image_id_2_prev][str(row["object_id_2"])]])

    return transitive_triplets


def predict_across(activations, threshold=0.01):
    predictions = []

    for activation in activations:

        if activation < 0.5 + threshold and activation > 0.5 - threshold:
            prediction = 2
        elif activation >= 0.5 + threshold:
            prediction = 1
        else:
            prediction = 0

        predictions.append(prediction)

    return predictions

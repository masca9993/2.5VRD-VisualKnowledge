import torch
import pandas as pd

def prepare_dataset(vrd: pd.DataFrame, objects: pd.DataFrame):

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

    y = data[["distance", "occlusion"]]
    data.drop(columns=["raw_distance", "raw_occlusion", "distance", "occlusion"], inplace=True)
    data_ids = data[["image_id_1", "object_id_1", "image_id_2", "object_id_2"]]
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
    data_ids = data[["image_id_1", "object_id_1", "image_id_2", "object_id_2"]]
    data.drop(columns=["image_id_1", "object_id_1", "image_id_2", "object_id_2"], inplace=True)

    return torch.tensor(data.values, dtype=torch.float32), torch.tensor(y.values), data_ids.values

import os
import csv
import shutil


def csv_to_dictionary(csv_file):
    result = {}

    # Open the CSV file
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)

        # Iterate over each row in the CSV file
        for row in csv_reader:
            key = row[0]
            value = row[1]

            result[key] = value

    return result

def move_last_items_to_beginning(csv_file):
    # Create a new directory to store modified files
    new_directory = os.path.dirname(csv_file)
    new_directory = os.path.join('yolov5-master/datasets/coco128/', 'labels')
    os.makedirs(new_directory, exist_ok=True)

    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)

        # Create a dictionary to store rows with the same ID
        id_rows = {}

        class_names_dict = csv_to_dictionary("visual_relationship/subset_data/class-descriptions-boxable.csv")
        new_dict = {}
        for i, value in enumerate(class_names_dict.values()):
            new_dict[value] = i

        # Iterate over each row in the CSV file
        for row in csv_reader:
            id_value = row[0]  # Get the ID value from the first element of the row

            if id_value not in id_rows:
                id_rows[id_value] = []

            id_rows[id_value].append(row)

        id_rows.pop("image_id")
        for id_value, rows in id_rows.items():
            for row in rows:
                row = [str(new_dict[class_names_dict[row[2]]])] + row[3:]
                print(row)
                # Create a new file with modified row data
                filename = f"{id_value}.txt"
                file_path = os.path.join(new_directory, filename)

                with open(file_path, 'a') as new_file:
                    new_file.write(' '.join(row) + '\n')

                print(f"Created file: {filename} in {new_directory}")

# Specify the CSV file path
csv_file_path = 'visual_relationship/subset_data/within_image_objects_train.csv'

move_last_items_to_beginning(csv_file_path)
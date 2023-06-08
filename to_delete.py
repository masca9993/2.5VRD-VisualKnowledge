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

dict = csv_to_dictionary("visual_relationship/subset_data/class-descriptions-boxable.csv")
new_dict = {}
for i, value in enumerate(dict.values()):
        new_dict[i] = value

for i, value in enumerate(new_dict.values()):
        print(i,":", value)
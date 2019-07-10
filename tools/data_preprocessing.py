import numpy as np
from shutil import copyfile
import os
import re

# min_ratio = 0.406
# min_width = 64
# min_height = 64
#
# max_ratio = 2.714
# max_width = 272
# max_height = 272


def unify_file_number(number):
    number = int(number[1:len(number) - 4])
    if 0 <= number < 10:
        return "000" + str(number)
    elif 10 <= number < 100:
        return "00" + str(number)
    elif 100 <= number < 1000:
        return "0" + str(number)
    else:
        return str(number)


# def temp_func():
#     i = 0
#     for class_folder in os.listdir("../../dataset5/F_"):
#         input_class_path = os.path.join("../../dataset5/F_", class_folder)
#         output_class_path = os.path.join("../../dataset5/F", class_folder)
#         if not os.path.exists(output_class_path):
#             os.makedirs(output_class_path)
#         file_names = np.array(os.listdir(input_class_path))
#         file_names.sort()
#         for file_name in file_names:
#             input_full_path = os.path.join(input_class_path, file_name)
#             output_full_path = os.path.join(output_class_path,
#                                       "color_" + unify_file_number(file_name))
#             copyfile(input_full_path, output_full_path)
#             i += 1
#             print(i)


def main():
    train_ratio = 0.8
    i = 0
    data_folders = os.listdir("../../dataset5")
    for data_folder in data_folders:
        input_path = os.path.join("../../dataset5", data_folder)
        output_path_test = os.path.join("../../prepared_data/test",
                                        data_folder)
        output_path_train = os.path.join("../../prepared_data/train",
                                         data_folder)
        if not os.path.exists(output_path_test):
            os.makedirs(output_path_test)
        if not os.path.exists(output_path_train):
            os.makedirs(output_path_train)

        class_folders = os.listdir(input_path)
        for class_folder in class_folders:
            input_class_path = os.path.join(input_path, class_folder)
            output_class_path_test = os.path.join(output_path_test,
                                                  class_folder)
            output_class_path_train = os.path.join(output_path_train,
                                                   class_folder)
            if not os.path.exists(output_class_path_test):
                os.makedirs(output_class_path_test)
            if not os.path.exists(output_class_path_train):
                os.makedirs(output_class_path_train)
            file_names = np.array(os.listdir(input_class_path))
            r = re.compile("color")
            matching = np.vectorize(lambda x: bool(r.search(x)))
            color_files = file_names[matching(file_names)]
            # np.random.shuffle(color_files)
            color_files.sort()
            training_color_files = \
                color_files[:int(color_files.shape[0] * train_ratio)]
            testing_color_files = \
                color_files[int(color_files.shape[0] * train_ratio):]
            for file_name in training_color_files:
                input_full_path = os.path.join(input_class_path, file_name)
                output_full_path = os.path.join(output_class_path_train,
                                                file_name)
                copyfile(input_full_path, output_full_path)
                file_name = file_name.replace("color", "depth")
                i += 1
                print(i)
                # input_full_path = os.path.join(input_class_path, file_name)
                # output_full_path = os.path.join(output_class_path_train,
                #                                 file_name)
                # copyfile(input_full_path, output_full_path)
                # i += 1
                # print(i)
            for file_name in testing_color_files:
                input_full_path = os.path.join(input_class_path, file_name)
                output_full_path = os.path.join(output_class_path_test,
                                                file_name)
                copyfile(input_full_path, output_full_path)
                file_name = file_name.replace("color", "depth")
                i += 1
                print(i)
                # input_full_path = os.path.join(input_class_path, file_name)
                # output_full_path = os.path.join(output_class_path_test,
                #                                 file_name)
                # copyfile(input_full_path, output_full_path)
                # i += 1
                # print(i)


if __name__ == '__main__':
    # temp_func()
    main()

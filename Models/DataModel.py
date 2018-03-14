import os
import shutil
from Models import SoundModel, ImageModel
from Definitions import DATA_RAW_DIR, DATA_TRAIN_DIR, DATA_VALIDATION_DIR, DATA_DIR


class DataModel:

    def prepare_data(self):
        sound_model = SoundModel.SoundModel()
        image_model = ImageModel.ImageModel()
        split_dir = os.path.join(DATA_DIR, 'raw_split')

        self.empty_dir_if_exist(split_dir)
        self.create_dir_if_not_exist(split_dir)

        if (os.path.isdir(DATA_RAW_DIR)):
            cars_list = [entry for entry in os.listdir(DATA_RAW_DIR) if
                         os.path.isdir(os.path.join(DATA_RAW_DIR, entry))]
            for car in cars_list:
                train_dir = os.path.join(DATA_TRAIN_DIR, car)
                self.empty_dir_if_exist(train_dir)
                self.create_dir_if_not_exist(train_dir)

                validation_dir = os.path.join(DATA_VALIDATION_DIR, car)
                self.empty_dir_if_exist(validation_dir)
                self.create_dir_if_not_exist(validation_dir)

                car_dir = os.path.join(DATA_RAW_DIR, car)
                files = [os.path.join(car_dir, entry) for entry in os.listdir(car_dir) if
                         os.path.isfile(os.path.join(car_dir, entry))]

                split_car_dir = os.path.join(split_dir, car)
                self.empty_dir_if_exist(split_car_dir)
                self.create_dir_if_not_exist(split_car_dir)

                for file in files:
                    file_without_ext = os.path.splitext(os.path.basename(file))[0]

                    sound_model.split_file(file, os.path.join(split_car_dir, file_without_ext + "%04d.wav"), 0.2)

                files_for_images = [os.path.join(split_car_dir, entry) for entry in os.listdir(split_car_dir) if
                                    os.path.isfile(os.path.join(split_car_dir, entry))]
                count = len(files_for_images)
                # we need to have same amout images on train and validation dirs
                image_in_dir = count // 2
                i = 0

                while i < image_in_dir:
                    print("Generating image for " + files_for_images[i])
                    data, sr = sound_model.load_file(files_for_images[i])
                    image_model.generate_morlet_scalogram(data, os.path.join(train_dir, str(i) + ".png"))
                    i += 1

                i = image_in_dir
                while i < (image_in_dir * 2):
                    print("Generating image for " + files_for_images[i])
                    data, sr = sound_model.load_file(files_for_images[i])
                    image_model.generate_morlet_scalogram(data, os.path.join(validation_dir, str(i) + ".png"))
                    i += 1
                print("Done generating data")
        else:
            print("Error")

    def create_dir_if_not_exist(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def empty_dir_if_exist(self, path):
        if os.path.exists(path):
            shutil.rmtree(path)

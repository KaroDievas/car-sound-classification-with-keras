import threading
import os
from Models import SoundModel, ImageModel


class ThreadedImageModel(threading.Thread):
    def __init__(self, thread_id, start_from_index, end_at_index, audio_files, image_path):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.start_from_index = start_from_index
        self.end_at_index = end_at_index
        self.audio_files = audio_files
        self.image_path = image_path
        self.image_model = ImageModel.ImageModel()
        self.sound_model = SoundModel.SoundModel()

    def run(self):
        print("Starting thread with ID =" + str(self.thread_id))
        i = self.start_from_index

        while i < self.end_at_index:
            print("Generating image for " + self.audio_files[i])
            data, sr = self.sound_model.load_file(self.audio_files[i])
            self.image_model.generate_morlet_scalogram(data, os.path.join(self.image_path, str(i) + ".png"))
            i += 1
        print("Exiting thread with ID =" + str(self.thread_id))

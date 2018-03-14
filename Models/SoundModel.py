import librosa
import os


class SoundModel:

    def split_file(self, file, output_file, duration=0.4):
        if os.path.isfile(file):
            os.system(
                "ffmpeg -i " + file + " -f segment -segment_time " + str(
                    duration) + " -c copy " + output_file)
        else:
            print("     *** File", file, "does not exist.  Skipping.")
        return

    def load_file(self, file):
        # loading as mono
        data, sr = librosa.load(file, sr=None)
        return data, sr

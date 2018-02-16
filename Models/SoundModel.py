from __future__ import print_function

import numpy as np
import librosa
import os


class SoundModel:
    def split_file(self, file, duration=0.4):
        if os.path.isfile(file):
            print("     Splitting file: ", file, " into ", duration, "-second files... ", end="", sep="")
            signal, sr = librosa.load(file, sr=None, mono=False)  # don't assume sr or mono
            if (1 == signal.ndim):
                print("       this is a mono file.  signal.shape = ", signal.shape)
            else:
                print("       this is a multi-channel file: signal.shape = ", signal.shape)
            axis = signal.ndim - 1
            signal_length = signal.shape[axis]
            stride = duration * sr  # length of clip in samples
            print("stride= ", stride, ", signal_length = ", signal_length)
            indices = np.arange(stride, signal_length, stride)  # where to split
            clip_list = np.split(signal, indices, axis=axis)  # do the splitting
            intended_length = stride
            clips = self.fix_last_element(clip_list, intended_length, axis)  # what to do with last clip

            sections = int(np.ceil(signal.shape[axis] / stride))  # just to check
            if (sections != clips.shape[0]):  # just in case
                print("              **** Warning: sections = " + str(sections) + ", but clips.shape[0] = " + str(
                    clips.shape[0]))
            ndigits = len(str(sections))  # find out # digits needed to print section #s
            for i in range(sections):
                clip = clips[i]
                filename_no_ext = os.path.splitext(file)[0]
                ext = os.path.splitext(file)[1]
                outfile = filename_no_ext + "_s" + '{num:{fill}{width}}'.format(num=i + 1, fill='0',
                                                                                width=ndigits) + ext
                print("        Saving file", outfile)
                librosa.output.write_wav(outfile, clip, sr)
        else:
            print("     *** File", file, "does not exist.  Skipping.")
        return

    def split_audio_files(self, file_list):
        for file in file_list:
            self.split_file(file)
        return

    def fix_last_element(self, clip_list, intended_length, axis):
        last_length = clip_list[-1].shape[axis]
        num_zeros = intended_length - last_length
        if (num_zeros > 0):
            ndims = clip_list[-1].ndim
            pad_list = []
            for i in range(ndims):
                if (axis == i):
                    pad_list.append((0, num_zeros))
                else:
                    pad_list.append((0, 0))
            clip_list[-1] = np.pad(clip_list[-1], pad_list, mode='constant')

        clips = np.asarray(clip_list)
        return clips

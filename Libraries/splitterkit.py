# Authored by Andrew Jiang
# BCG digital Ventures
import os, wave, math, collections

# define named tuples
metatuple = collections.namedtuple('metatuple', ['nchannels', 'sampwidth', 'framerate', 'nframes', 'comptype', 'compname'])
datatuple = collections.namedtuple('datatuple', ['meta', 'data'])

# opens a wav file and returns the data as a tuple
def readwave(src):
    read = wave.open(src, 'rb')
    meta = read.getparams()
    # turn params into a metatuple
    meta = metatuple(meta[0], meta[1], meta[2], meta[3], meta[4], meta[5])
    data = read.readframes(meta.nframes)
    read.close()
    return datatuple(meta, [data])

# writes to a directory
def writewave(dest, data):
    files = []
    data = separate(data)
    for i in range(len(data)):
        destfile = dest + `i` + '.wav'
        makedir(destfile) # make sure dir exists
        write = wave.open(destfile, 'wb')
        write.setparams(data[i].meta)
        write.writeframes(data[i].data)
        write.close()
        files.append(destfile)
    return files

# helper function that creates dir if it doesn't exist
def makedir(dest):
    if(os.path.isdir(os.path.dirname(dest)) != True):
        os.makedirs(os.path.dirname(dest))

# slices audio data at given start, end: frame#
def slicewave(data, start, end):
    if(len(data.data) > 1):
        data = merge(data) # insurance
    meta = data.meta
    start *= meta.sampwidth # deal with sample width
    end *= meta.sampwidth
    spliced = data.data[0][start:end]
    nf = len(spliced) / meta.sampwidth
    meta = meta._replace(nframes=nf)
    return datatuple(meta, [spliced])

# slices audio data at given start, end: seconds
def slicewave_s(data, start, end):
    fr = float(data.meta.framerate)
    newdata = slicewave(data, int(float(start) * fr), int(float(end) * fr))
    return newdata

# splits audio data into equal intervals: # of frames
def split(data, interval=None, overlap=None):
    if(interval == None):
        interval = data.meta.framerate # =1s
    if(overlap == None):
        overlap = interval
    if(interval < 1 or overlap < 1):
        raise ValueError('cannot iterate')
    iterations = int(math.ceil(1.0 * data.meta.nframes / interval))
    canned = []
    for i in range(iterations):
        start = i * interval
        end = start + overlap
        canned.append(slicewave(data, start, end))
    newdata = combine(canned)
    return newdata

# splits audio data into equal intervals: seconds
def split_s(data, interval=None, overlap=None):
    fr = float(data.meta.framerate)
    if(interval != None):
        interval = int(float(interval) * fr)
    if(overlap != None):
        overlap = int(float(overlap) * fr)
    newdata = split(data, interval, overlap)
    return newdata

# separate a data tuple containing multiple audio tracks
# into an array of data tuples containing single audio tracks
def separate(data):
    newdata = []
    nframes = data.meta.nframes
    ndata = len(data.data)
    for i in range(ndata):
        nf = len(data.data[i]) / data.meta.sampwidth
        meta = data.meta._replace(nframes=nf)
        newdata.append(datatuple(meta, data.data[i]))
    return newdata

# combine an array of data tuples containing single audio tracks
# into a single data tuple containing multiple audio tracks
def combine(data):
    newdata = []
    meta = data[0].meta
    for i in range(len(data)):
        newdata += data[i].data
    nf = len(''.join(newdata)) / meta.sampwidth
    meta = meta._replace(nframes=nf)
    return datatuple(meta, newdata)

# merge multiple audio tracks into one
def merge(data):
    meta = data.meta
    newdata = ''.join(data.data)
    return datatuple(meta, [newdata])

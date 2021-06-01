import numpy as np
import pyaudio
import time
import librosa
import matplotlib.pyplot as plt
import librosa.display
from tensorflow.keras.models import load_model


import cv2


def preprocessDataForNN(img_arr):
    img_arr = img_arr*(1/.255)
    f = np.resize(img_arr, (150, 150, 3))
    res = f.reshape(-1, 150, 150, 3)
    return res


def create_spectrogram(y, sample_rate):
    
    plt.interactive(False)
        
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max))

    plt.close()    
    fig.clf()
    plt.close(fig)
    plt.close('all')

    del sample_rate,fig,ax
    
    return spectrogram
    


# In[7]:


import pyaudio
import wave

CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 1

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 1.5


p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

frames = []

model = load_model(r'/home/timur/Documents/Projects/sound_classif/save_1.h5')

# while(True):
#     frames = []
#     for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
#         data = stream.read(CHUNK)
#         frames.extend(data)
    
    #print("* done recording")

    frames = bytes(frames)

    numpy_array = np.frombuffer(frames, dtype=np.float32)
    sp = create_spectrogram(numpy_array, RATE)

    plt.imshow(sp, interpolation='nearest')
    # plt.show()
    librosa.display.specshow(librosa.power_to_db(sp, ref=np.max))
    print(sp.shape)
    # img_compl = preprocessDataForNN(sp)
    final_res = model.predict(img_compl)
    print(final_res)
    break

# numpy_array = np.frombuffer(pull, dtype=np.float32)
# sp = create_spectrogram(numpy_array, RATE)
# print(sp)
p.terminate()
# In[5]:


from matplotlib import pyplot as plt 
plt.imshow(sp, interpolation='nearest') 
plt.show() 
librosa.display.specshow(librosa.power_to_db(sp, ref=np.max))


from sys import argv
import cv2 as cv
import numpy as np

img = sp

ret, threshold = cv.threshold(img, 120, 255, cv.THRESH_BINARY)

# cv.imshow('threshold', threshold)
# cv.imshow('orig', img)


normalizedImg = np.zeros((150, 150))
normalizedImg = cv.normalize(img,  normalizedImg, 0, 255, cv.NORM_MINMAX)
# cv.imshow('dst_rt', normalizedImg)

cv.waitKey(0)
cv.destroyAllWindows()


# In[14]:


import imageio
import numpy
img = imageio.imread(r'/home/timur/Documents/Projects/sound_classif/git/Save_Transport/dataset/train/one_sec_cut.jpg/0_1_01.jpg')
array = numpy.asarray(img)
print(array.shape)


# In[ ]:





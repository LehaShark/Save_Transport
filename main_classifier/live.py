<<<<<<< HEAD
=======
#!/usr/bin/env python
# coding: utf-8

# In[2]:


>>>>>>> 7914ce3b17be6df36a221624f105edcfb8c33c16
import numpy as np
import pyaudio
import time
import librosa
import matplotlib.pyplot as plt
import librosa.display
from tensorflow.keras.models import load_model
<<<<<<< HEAD


import cv2


def preprocessDataForNN(img_arr):
    img_arr = img_arr*(1/.255)
    f = np.resize(img_arr, (150, 150, 3))
=======
import keyboard
import cv2
import io
import wave


# In[3]:


def get_img_from_fig(fig, dpi=400):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


# In[15]:


def preprocessDataForNN(img_arr): 
    
    img_arr = img_arr/255.0
    f = cv2.resize(img_arr, (150, 150))
>>>>>>> 7914ce3b17be6df36a221624f105edcfb8c33c16
    res = f.reshape(-1, 150, 150, 3)
    
    return res


<<<<<<< HEAD
def create_spectrogram(y, sample_rate):
=======
# In[5]:


def createSpectrogramToBuffer(y, sample_rate):
>>>>>>> 7914ce3b17be6df36a221624f105edcfb8c33c16
    
    plt.interactive(False)
        
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max))

    arr = get_img_from_fig(fig)
    plt.close()    
    fig.clf()
    plt.close(fig)
    plt.close('all')
  
    return arr


# In[6]:



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

    return spectrogram


# In[16]:


CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100
<<<<<<< HEAD
RECORD_SECONDS = 1.5

=======
RECORD_SECONDS = 1

#WAVE_OUTPUT_FILENAME = "output.wav"
>>>>>>> 7914ce3b17be6df36a221624f105edcfb8c33c16

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index = 1) # change this parameter to good work

print("* recording")

frames = []

<<<<<<< HEAD
model = load_model(r'/home/timur/Documents/Projects/sound_classif/save_1.h5')
=======
model = load_model(r'D:\save_1.h5')
#model = load_model('/home/aigaf/Downloads/Telegram Desktop/save_1.h5')
>>>>>>> 7914ce3b17be6df36a221624f105edcfb8c33c16

# while(True):
#     frames = []
#     for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
#         data = stream.read(CHUNK)
#         frames.extend(data)
    
    #print("* done recording")

    frames = bytes(frames)

    numpy_array = np.frombuffer(frames, dtype=np.float32)
<<<<<<< HEAD
    sp = create_spectrogram(numpy_array, RATE)

    plt.imshow(sp, interpolation='nearest')
    # plt.show()
    librosa.display.specshow(librosa.power_to_db(sp, ref=np.max))
    print(sp.shape)
    # img_compl = preprocessDataForNN(sp)
    final_res = model.predict(img_compl)
    print(final_res)
    break
=======
    
    sp = create_spectrogram(numpy_array, RATE)     
    plt.imshow(sp, interpolation='nearest') 
    plt.show() 
    librosa.display.specshow(librosa.power_to_db(sp, ref=np.max))
    
    img_array = createSpectrogramToBuffer(numpy_array,RATE)    
    img_compl = preprocessDataForNN(img_array)
    final_res = model.predict(img_compl)
    print(final_res)
    
    #try:  # used try so that if user pressed other than the given key error will not be shown
        #if keyboard.is_pressed('q'):  # if key 'q' is pressed 
            #print('Exit')
            #break  # finishing the loop
    #except:
        #break  # if user pressed a key other than the given key the loop will break
    #print(sp)
>>>>>>> 7914ce3b17be6df36a221624f105edcfb8c33c16

# numpy_array = np.frombuffer(pull, dtype=np.float32)
# sp = create_spectrogram(numpy_array, RATE)
# print(sp)
p.terminate()
<<<<<<< HEAD
# In[5]:
=======


# In[7]:
>>>>>>> 7914ce3b17be6df36a221624f105edcfb8c33c16


from matplotlib import pyplot as plt 
plt.imshow(sp, interpolation='nearest') 
plt.show() 
librosa.display.specshow(librosa.power_to_db(sp, ref=np.max))


<<<<<<< HEAD
=======
# In[1]:


#1
#img - numpy array spectrogram

>>>>>>> 7914ce3b17be6df36a221624f105edcfb8c33c16
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


# In[8]:


<<<<<<< HEAD
import imageio
import numpy
img = imageio.imread(r'/home/timur/Documents/Projects/sound_classif/git/Save_Transport/dataset/train/one_sec_cut.jpg/0_1_01.jpg')
array = numpy.asarray(img)
print(array.shape)
=======
# get information about audio devices
p = pyaudio.PyAudio()
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i))
>>>>>>> 7914ce3b17be6df36a221624f105edcfb8c33c16


# In[ ]:





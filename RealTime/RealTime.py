from memory_profiler import memory_usage
from tensorflow import keras
import librosa
import librosa.display
import pylab
import matplotlib.pyplot as plt
from matplotlib import figure
import gc
from path import Path
import numpy as np
from keras_preprocessing.image import ImageDataGenerator


model = keras.models.load_model('/home/aigaf/Desktop/Repositories/Save_Transport/main_classifier/save_1.h5')

def create_spectrogram(filename,name):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    
    filename  = '/home/aigaf/Desktop/Repositories/Save_Transport/RealTime/jpg/Untitled/' + name + '.jpg'
    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()    
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,S
    
import sounddevice as sd
from scipy.io.wavfile import write

import sounddevice as sd
from scipy.io.wavfile import write

fs = 44100  # Sample rate
seconds = 1  # Duration of recording

#while(True): add if you wanna make it in realtime

for i in range(5):
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    
    path = '/home/aigaf/Desktop/Repositories/Save_Transport/RealTime/wav/'
    name = 'output' + str(i)
    
    write(path + name + '.wav', fs, myrecording) # конверт в wav и сохранение
    create_spectrogram(path + name + '.wav',name)
    
liveMode = ImageDataGenerator(rescale=1./255)
live_generator = liveMode.flow_from_directory('/home/aigaf/Desktop/Repositories/Save_Transport/RealTime/jpg',
                                               target_size=(150, 150),batch_size=1)
filenames = live_generator.filenames
nb_samples = len(filenames)
predict = model.predict_generator(live_generator,steps = nb_samples)

print(predict)

from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from data_prep import valik
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import cv2
import math
import random
random.seed(228)
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model


model = load_model(r'D:\save_115001.h5')

n = 30
generator = ImageDataGenerator(rescale=1./255)
predict_generator = generator.flow_from_directory(r'D:\reposetory\Save_Transport\datasets\new_dataset\test',
                                               target_size=(150, 150),  shuffle = True, batch_size = n)


def getPredictOrLabels(predictOrLabels, generator):
        '''true == labels, false == predict'''

        amountOfBatches = predict_generator.samples / n
        amountOfBatches = round(amountOfBatches)
        array = []

        for i in range(amountOfBatches):
                if predictOrLabels == False:
                        tmp = model.predict(predict_generator[i][0])
                else:
                        tmp = predict_generator[i][1]

                array.extend(tmp)
        return array


def getDataForMetrics(generator):
        y_true = []
        y_pred = []

        y_true = getPredictOrLabels(True, generator)
        y_pred = getPredictOrLabels(False, generator)

        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_true, axis=1)

        return y_true, y_pred


def getPredictOrLabels(predictOrLabels, generator):
        '''true == labels, false == predict'''

        amountOfBatches = predict_generator.samples / n
        amountOfBatches = round(amountOfBatches)
        array = []

        for i in range(amountOfBatches):
                if predictOrLabels == False:
                        tmp = model.predict(predict_generator[i][0])
                else:
                        tmp = predict_generator[i][1]

                array.extend(tmp)
        return array

y_true,y_pred = getDataForMetrics(predict_generator)
confusion_matrix(y_true, y_pred)

accuracy_score(y_true,y_pred)

print(confusion_matrix(y_true, y_pred))

# #a = Image.open('cut.jpg')
# #a = cv2.imread('cut1.jpg')
# img_arr = np.asarray(a)
# # print(im/g_arr.shape)
# img_arr = img_arr*(1/.255)
# print(img_arr.shape)
# f = cv2.resize(img_arr, (150, 150))
# res = f.reshape(-150, 150, 1)
#
# results = model.evaluate(valik)
# print('loss, accuracy =', results)
#
#
# final_res = model.predict(valik[1][0])
# print(final_res)
# data = {'y_Actual':    [ tuple(i) for i in valik[1][1]],
#         'y_Predicted': [ tuple(i) for i in final_res]
#         }
#
# df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
# print (df)

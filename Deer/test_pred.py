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


model = load_model('/home/timur/Documents/Projects/sound_classif/save_115001.h5')
# new_data = ImageDataGenerator(rescale=1.0/255)
# new_data1 = ImageDataGenerator(rescale=1.0/255)
#
# new_train_data = new_data.flow_from_directory(
#         '/home/timur/Documents/Projects/sound_classif/git/Save_Transport/new_dataset/train/',
#         target_size=(150, 150),
#         batch_size=20)
#
# new_test_data = new_data1.flow_from_directory('/home/timur/Documents/Projects/sound_classif/git/Save_Transport/new_dataset/train/',
#         target_size=(150, 150),
#         batch_size=25)

# history = model.fit(new_train_data,
#                       validation_data=new_test_data,
#                       steps_per_epoch=3,
#                       epochs=50,
#                       validation_steps=3,
#                       verbose=2)



# model.save('new_model.h5')
# print(data[0].shape)

# a = Image.open('cut.jpg')
# a = cv2.imread('cut1.jpg')
# img_arr = np.asarray(data)
# print(im/g_arr.shape)
# img_arr = img_arr*(1/.255)
# print(img_arr.shape)
# f = cv2.resize(img_arr, (150, 150))
# res = f.reshape(-150, 150, 1)

results = model.evaluate(valik)
print('loss, accuracy =', results)


final_res = model.predict(valik[1][0])
print(final_res)
data = {'y_Actual':    [ tuple(i) for i in valik[1][1]],
        'y_Predicted': [ tuple(i) for i in final_res]
        }

df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
print (df)

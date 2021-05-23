from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from data_prep import valik
import pandas as pd
from sklearn.metrics import confusion_matrix
from PIL import Image, ImageFont, ImageDraw


model = load_model('git/Save_Transport/AiGaf/save_1.h5')


results = model.evaluate(valik)
print('loss, accuracy =', results)


final_res = model.predict(valik[1][0])

data = {'y_Actual':    [ tuple(i) for i in valik[1][1]],
        'y_Predicted': [ tuple(i) for i in final_res]
        }

df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
print (df)

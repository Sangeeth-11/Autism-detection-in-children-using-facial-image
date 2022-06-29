# from model import model
import os

import numpy as np
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator, load_img
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf
from preprocessing import validation_generator, total_validate, batch_size, test_df, test_generator, nb_samples, \
    test_gen

def predict():
    model=load_model('vgg16.h5')
    loss, accuracy = model.evaluate_generator(validation_generator, total_validate // batch_size, workers=12)
    print("Test: accuracy = %f  ;  loss = %f " % (accuracy*100, loss))

    test_filenames2 = os.listdir("static/files")
    test_df2 = pd.DataFrame({
        'filename': test_filenames2
    })
    print(test_df2)

    nb_samples2 = test_df2.shape[0]
    image_size = 224

    test_gen2 = ImageDataGenerator(rescale=1./255)
    test_generator2 = test_gen.flow_from_dataframe(
        test_df2,
        "static/files/",
        x_col='filename',
        y_col=None,
        class_mode=None,
        batch_size=batch_size,
        target_size=(image_size, image_size),
        shuffle=False
    )

    threshold = 0.5
    predict2 = model.predict_generator(test_generator2, steps=np.ceil(nb_samples2 / batch_size))
    test_df2['category'] = np.where(predict2 > threshold, 1,0)

    sample_test2 = test_df2.sample(n=1).reset_index()
    sample_test2.head()
    plt.figure(figsize=(12, 12))
    for index, row in sample_test2.iterrows():
        filename = row['filename']
        category = row['category']
        img = load_img("static/files/"+filename, target_size=(256, 256))
        plt.subplot(3, 2, index+1)
        plt.get_current_fig_manager().window.state('zoomed')


        frame1=plt.imshow(img)
        if category==1:
            x='Autistic'
        else:
            x='Non_Autistic'
        return x
#     plt.xlabel('(' + x + ')')
#     plt.tight_layout()
#     #frame1.axes.get_xaxis().set_visible(False)
#     frame1.axes.get_xaxis().set_ticks([])
#     frame1.axes.get_yaxis().set_visible(False)
#     plt.show()
#
# predict()

import csv

import numpy as np
import pandas as pd
import sns as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model

from preprocessing import test_generator, test_df, nb_samples, batch_size

model=load_model('vgg16.h5')
predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))
threshold = 0.5
test_df['category'] = np.where(predict > threshold, 1,0)

# print(test_df['category'])

sample_test = test_df.sample(n=9).reset_index()
#print(sample_test.head())
plt.figure(figsize=(12, 12))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    img = tf.keras.utils.load_img("AutismDataset/test/"+filename, target_size=(256, 256))
    plt.subplot(3, 3, index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')')
plt.tight_layout()
# plt.show()

submission_df = test_df.copy()
submission_df['id'] = submission_df['filename'].str.split('.').str[0]
submission_df['label'] = submission_df['category']
submission_df.drop(['filename', 'category'], axis=1, inplace=True)
submission_df.to_csv('submission_13010030.csv', index=False)

pd.read_csv('submission_13010030.csv')

# plt.figure(figsize=(10,5))
# sns.countplot(submission_df['label'])
# plt.title("(Predicted data)")

submission_df = test_df.copy()
submission_df['id'] = submission_df['filename'].str.split('.').str[0]
submission_df['label'] = submission_df['category']
submission_df.drop(['filename', 'category'], axis=1, inplace=True)
submission_df.to_csv('submission_13010030.csv', index=False)


# plt.figure(figsize=(10,5))
# sns.countplot(submission_df['id'])
# plt.title("(Test data)")

my_reader = csv.reader(open('submission_13010030.csv'))
predicted_autistic = 0
for record in my_reader:
    if record[1] == '1':
        predicted_autistic += 1
print("predicted Autistic : " ,predicted_autistic)

my_reader = csv.reader(open('submission_13010030.csv'))
predicted_non_autistic = 0
for record1 in my_reader:
    if record1[1] == '0':
        predicted_non_autistic += 1
print("predicted Non Autistic : " ,predicted_non_autistic)

my_reader = csv.reader(open('submission_13010030.csv'))
autistic = 0
for record1 in my_reader:
    if record1[0] == 'Autistic':
        autistic += 1
print("Actual Autistic : " ,autistic)

my_reader = csv.reader(open('submission_13010030.csv'))
non_autistic = 0
for record1 in my_reader:
    if record1[0] == 'Non_Autistic':
        non_autistic += 1
print("Actual Non Autistic : " ,non_autistic)


#accuracy for predicting
print("Actual Non Autistic percentage in total test data: " ,(non_autistic/300)*100,"%")
print("Predicted Non Autistic percentage in total test data: " ,(predicted_non_autistic/300)*100,"%")
print("Actual Autistic percentage in total test data: " ,(autistic/300)*100,"%")

print("Predicted Autistic percentage in total test data: " ,(predicted_autistic/300)*100,"%")

my_reader = csv.reader(open('submission_13010030.csv'))
true_pos = 0 #autistic,1
for record1 in my_reader:
    if record1[0] == 'Autistic' and record1[1]=='1':
        true_pos += 1
print("True positive : " ,true_pos)

my_reader = csv.reader(open('submission_13010030.csv'))
true_neg = 0 #non_autistic,0
for record1 in my_reader:
    if record1[0] == 'Non_Autistic' and record1[1]=='0':
        true_neg += 1
print("True Negative : " ,true_neg)

my_reader = csv.reader(open('submission_13010030.csv'))
false_pos = 0 #autistic,0
for record1 in my_reader:
    if record1[0] == 'Autistic' and record1[1]=='0':
       false_pos += 1
print("false Positive : " ,false_pos)

my_reader = csv.reader(open('submission_13010030.csv'))
false_neg = 0 #non_autistic,1
for record1 in my_reader:
    if record1[0] == 'Non_Autistic' and record1[1]=='1':
       false_neg += 1
print("false Negative : " ,false_neg)

accuracy = (true_pos + true_neg)/(true_pos + true_neg + false_pos + false_neg)
print("Accuracy is: ",accuracy*100,"%")

precision = true_pos / ( true_pos + false_pos)
print("Precision is: ",precision*100,"%")

sensitivity = true_pos / (true_pos + false_neg)
print("Sensitivity is: ",sensitivity*100,"%")

Specificity =true_neg / (true_neg + false_pos)
print("Specificity is: ",Specificity*100,"%")



# Import Modules
import os
import pandas as pd
import numpy as np
from itertools import chain
import cv2
import splitfolders

# Plotting
#%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import matplotlib.image as mpimg
import seaborn as sns

# Metrics
from sklearn.metrics import confusion_matrix, roc_curve,auc, classification_report

# Deep Learning
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img

base_dir = "E:\\Major Project"
input_ = os.path.join(base_dir,"CRC-VAL-HE-7K")

# split data into training, vlaidation and testing sets
#splitfolders.ratio(input_, output="output", seed = 101, ratio=(0.8, 0.1, 0.1))

data_dir = os.path.join('E:\\Major Project','output')

# Define train, valid and test directories
train_dir = os.path.join(data_dir, 'train')
valid_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')
os.listdir('output')

# Data Augmentation
train_datagen = ImageDataGenerator(rescale=1./255.,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   shear_range=0.4,
                                   width_shift_range=0.25,
                                   height_shift_range=0.25,
                                   rotation_range=45,
                                   fill_mode='nearest')
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Train ImageDataGenerator
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size = 64,
                                                    target_size = (224,224),
                                                    class_mode = 'categorical',
                                                    shuffle=True,
                                                    seed=42,
                                                    color_mode='rgb')

valid_generator = valid_datagen.flow_from_directory(valid_dir,
                                                    batch_size=64,
                                                    target_size=(224,224),
                                                    class_mode='categorical',
                                                    shuffle=True,
                                                    seed=42,
                                                    color_mode='rgb')

test_generator = test_datagen.flow_from_directory(test_dir,
                                                batch_size=1,
                                                target_size=(224,224),
                                                class_mode='categorical',
                                                shuffle=False,
                                                seed=42,
                                                color_mode='rgb')

base_model_resnet50 = ResNet50(input_shape=(224,224,3),
                               include_top=False, 
                               weights='imagenet')

x = base_model_resnet50.output
flat = Flatten()(x)
hidden_1 = Dense(1024, activation='relu')(flat)
drop_1 = Dropout(0.2)(hidden_1)
hidden_2 = Dense(512, activation='relu')(drop_1)
drop_2 = Dropout(0.3)(hidden_2)
output = Dense(9, activation= 'softmax')(drop_2)

model_resnet50_01 = Model(base_model_resnet50.inputs, output)

# Call Backs
filepath = 'resent50_base_model_wt.keras'
es = EarlyStopping(monitor='loss', verbose=1, mode='min', patience=4)
cp = ModelCheckpoint(filepath, monitor='val_loss',verbose=1, save_best_only=True,
                     save_weights_only=False, mode='auto', save_freq='epoch')
# compile the model
optimizer = Adam(learning_rate=0.0004)

# Compile the model using the optimizer
model_resnet50_01.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
"""
resnet50_history_01 = model_resnet50_01.fit(train_generator,
                                            steps_per_epoch= int(len(input_)/64),
                                            epochs=10,
                                            callbacks = [es, cp],
                                        validation_data = valid_generator)

# save model
if not os.path.isdir('model_weights/'):
    os.mkdir('model_weights/')
model_resnet50_01.save_weights(filepath='model_weights/resent50_base_model_wt.weights.h5', overwrite=True)

"""

# Load the saved model
model_resnet50_01.load_weights('resent50_base_model_wt.keras')
# Evaluate the model on the hold out validation and test datasets

res_val_eval_01 = model_resnet50_01.evaluate(valid_generator)
res_test_eval_01 = model_resnet50_01.evaluate(test_generator)

print('Validation loss:      {}'.format(res_val_eval_01[0]))
print('Validation accuracy:  {}'.format(res_val_eval_01[1]))
print('Test loss:            {}'.format(res_test_eval_01[0]))
print('Test accuracy:        {}'.format(res_test_eval_01[1]))

# Predict probabilities
nb_samples = len(test_generator)
res_predictions_01 = model_resnet50_01.predict(test_generator,steps = nb_samples,verbose=1)
# Predict labels
res_pred_labels_01 = np.argmax(res_predictions_01, axis=1)

# Classification Report
print('|'+'-'*67+'|')
print('|-------Classification Report: ReseNet50 Training Cycle #1----------|')
print('|'+'-'*67+'|')
print(classification_report(test_generator.classes, res_pred_labels_01, 
                            target_names=['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']))

# Traning cycle 1
base_model_resnet50_02 = ResNet50(input_shape=(224, 224, 3),
                                               include_top = False)
x = base_model_resnet50_02.output
flat = Flatten()(x)
hidden_1 = Dense(1024, activation='relu')(flat)
drop_1 = Dropout(0.2)(hidden_1)
hidden_2 = Dense(512, activation='relu')(drop_1)
drop_2 = Dropout(0.3)(hidden_2)
output = Dense(9, activation= 'softmax')(drop_2)

model_resnet50_02 = Model(base_model_resnet50_02.inputs, output)
model_resnet50_02.load_weights('resent50_base_model_wt.keras')

for layer in model_resnet50_02.layers[:160]:
    layer.trainable= False
    
# Call Backs
filepath_02 = 'resent50_model_02_wt.keras'
es_02 = EarlyStopping(monitor='loss', verbose=1, mode='min', patience=4)
cp_02 = ModelCheckpoint(filepath_02, monitor='val_loss',verbose=1, save_best_only=True,
                     save_weights_only=False, mode='auto', save_freq='epoch')

optimizer = Adam(learning_rate=0.0004)

# Compile the model using the optimizer
model_resnet50_02.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

resnet50_history_02 = model_resnet50_02.fit(train_generator,
                                                      steps_per_epoch= int(len(input_)/64,
                                                      epochs=10,
                                                      callbacks = [es_02, cp_02],
                                                      validation_data = valid_generator)

# save model
if not os.path.isdir('model_weights/'):
    os.mkdir('model_weights/')
model_resnet50_02.save_weights(filepath='resent50_base_model_wt.weights.h5', overwrite=True)


# Load the saved model
model_resnet50_02.load_weights('resent50_model_02_wt.keras')
# Evaluate the model on the hold out validation and test datasets

res_val_eval_02 = model_resnet50_02.evaluate(valid_generator)
res_test_eval_02 = model_resnet50_02.evaluate(test_generator)

print('Validation loss:     {}'.format(res_val_eval_02[0]))
print('Validation accuracy: {}'.format(res_val_eval_02[1]))
print('Test loss:           {}'.format(res_test_eval_02[0]))
print('Test accuracy:       {}'.format(res_test_eval_02[1]))
print('*'*75)

# Predict probabilities
nb_samples = len(test_generator)
res_predictions_02 = model_resnet50_02.predict(test_generator, steps = nb_samples,verbose=1)
# Predict labels
res_pred_labels_02 = np.argmax(res_predictions_02, axis=1)

# Classification Report
print('|'+'-'*67+'|')
print('|-------Classification Report: ReseNet50 Training Cycle #2----------|')
print('|'+'-'*67+'|')
print(classification_report(test_generator.classes, res_pred_labels_02, 
                            target_names=['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']))

# Traning cycle 2
base_model_resnet50_03 = ResNet50(input_shape=(224, 224, 3),
                                               include_top = False)
x = base_model_resnet50_03.output
flat = Flatten()(x)
hidden_1 = Dense(1024, activation='relu')(flat)
drop_1 = Dropout(0.2)(hidden_1)
hidden_2 = Dense(512, activation='relu')(drop_1)
drop_2 = Dropout(0.3)(hidden_2)
output = Dense(9, activation= 'softmax')(drop_2)

model_resnet50_03 = Model(base_model_resnet50_03.inputs, output)
model_resnet50_03.load_weights('resent50_model_02_wt.keras')

for layer in model_resnet50_03.layers[:118]:
    layer.trainable= False
    
# Call Backs
filepath_03 = 'resent50_model_03_wt.keras'
es_03 = EarlyStopping(monitor='loss', verbose=1, mode='min', patience=4)
cp_03 = ModelCheckpoint(filepath_03, monitor='val_loss',verbose=1, save_best_only=True,
                     save_weights_only=False, mode='auto', save_freq='epoch')

optimizer = Adam(learning_rate=0.0004)

# Compile the model using the optimizer
model_resnet50_03.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

resnet50_history_03 = model_resnet50_03.fit(train_generator,
                                                      steps_per_epoch= int(len(input_)/64,
                                                      epochs=10,
                                                      callbacks = [es_02, cp_02],
                                                      validation_data = valid_generator)
# save model
if not os.path.isdir('model_weights/'):
    os.mkdir('model_weights/')
model_resnet50_03.save_weights(filepath='resent50_model_03_wt.weights.h5', overwrite=True)

# Load the saved model
model_resnet50_03.load_weights('resent50_model_03_wt.keras')
# Evaluate the model on the hold out validation and test datasets

res_val_eval_03 = model_resnet50_03.evaluate(valid_generator)
res_test_eval_03 = model_resnet50_03.evaluate(test_generator)

print('Validation loss:     {0:.3f}'.format(res_val_eval_03[0]))
print('Validation accuracy: {0:.3f}'.format(res_val_eval_03[1]))
print('Test loss:           {0:.3f}'.format(res_test_eval_03[0]))
print('Test accuracy:       {0:.3f}'.format(res_test_eval_03[1]))

# Predict probabilities
nb_samples = len(test_generator)
res_predictions_03 = model_resnet50_03.predict(test_generator,steps = nb_samples,verbose=1)
# Predict labels
res_pred_labels_03 = np.argmax(res_predictions_03, axis=1)

# Classification Report
print('|'+'-'*67+'|')
print('|-------Classification Report: ReseNet50 Training Cycle #3----------|')
print('|'+'-'*67+'|')
print(classification_report(test_generator.classes, res_pred_labels_03, 
                            target_names=['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']))

#Fine tuning Resnet50 model training cycle 4

base_model_resnet50_04 = ResNet50(input_shape=(224, 224, 3),
                                               include_top = False)
x = base_model_resnet50_04.output
flat = Flatten()(x)
hidden_1 = Dense(1024, activation='relu')(flat)
drop_1 = Dropout(0.2)(hidden_1)
hidden_2 = Dense(512, activation='relu')(drop_1)
drop_2 = Dropout(0.3)(hidden_2)
output = Dense(9, activation= 'softmax')(drop_2)

model_resnet50_04 = Model(base_model_resnet50_04.inputs, output)
model_resnet50_04.load_weights('resent50_model_03_wt.h5')

# Call Backs
filepath_04 = 'resent50_model_04_wt.h5'
es_04 = EarlyStopping(monitor='loss', verbose=1, mode='min', patience=4)
cp_04 = ModelCheckpoint(filepath_04, monitor='val_loss',verbose=1, save_best_only=True,
                     save_weights_only=False, mode='auto', save_freq='epoch')

optimizer = Adam(learning_rate=0.0004)
model_resnet50_04.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
resnet50_history_04 = model_resnet50_04.fit_generator(train_generator,
                                                      steps_per_epoch= int(len(input_)/64,
                                                      epochs=30,
                                                      callbacks = [es_04, cp_04],
                                                      validation_data = valid_generator)

# Confusion Matrices
res_conf_mat_01 = pd.DataFrame(confusion_matrix(test_generator.classes, res_pred_labels_01), 
                        index=['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM'], 
                        columns=['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM'])

res_conf_mat_02 = pd.DataFrame(confusion_matrix(test_generator.classes, res_pred_labels_02), 
                        index=['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM'], 
                        columns=['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM'])

res_conf_mat_03 = pd.DataFrame(confusion_matrix(test_generator.classes, res_pred_labels_03), 
                        index=['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM'], 
                        columns=['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM'])

res_conf_mat_04 = pd.DataFrame(confusion_matrix(test_generator.classes, res_pred_labels_04), 
                        index=['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM'], 
                        columns=['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM'])

# Plot Confusion Matrices
sns.set(font_scale=1.8)
fig, ([ax1,ax2],[ax3,ax4]) = plt.subplots(nrows=2, ncols=2, figsize=(28,28))

#ax1
sns.heatmap(res_conf_mat_01, ax=ax1, annot=True, fmt=".1f", linewidths=0.5, square=True, cmap='Blues_r', annot_kws={"size": 18})
ax1.set_ylabel("Actual Label", fontsize=24)
ax1.set_xlabel("Predicted Label", fontsize=24)
all_sample_title="ResNet50 Training Round #1 \nAccuracy Score: {0:.3f}".format(res_test_eval_01[1])
ax1.set_title(all_sample_title, size=32)
ax1.set_ylim(len(res_conf_mat_01)-0.1, -0.1)

#ax2
sns.heatmap(res_conf_mat_02, ax=ax2, annot=True, fmt=".1f", linewidths=0.5, square=True, cmap='Blues_r', annot_kws={"size": 18})
ax2.set_ylabel("Actual Label", fontsize=24)
ax2.set_xlabel("Predicted Label", fontsize=24)
all_sample_title="ResNet50 Training Round #2 \nAccuracy Score: {0:.3f}".format(res_test_eval_02[1])
ax2.set_title(all_sample_title, size=32)
ax2.set_ylim(len(res_conf_mat_02)-0.1, -0.1)

#ax3
sns.heatmap(res_conf_mat_03, ax=ax3, annot=True, fmt=".1f", linewidths=0.5, square=True, cmap='Blues_r', annot_kws={"size": 18})
ax3.set_ylabel("Actual Label", fontsize=24)
ax3.set_xlabel("Predicted Label", fontsize=24)
all_sample_title="ResNet50 Training Round#3 \nAccuracy Score: {0:.3f}".format(res_test_eval_03[1])
ax3.set_title(all_sample_title, size=32)
ax3.set_ylim(len(res_conf_mat_03)-0.1, -0.1)

#ax4
sns.heatmap(res_conf_mat_04, ax=ax4, annot=True, fmt=".1f", linewidths=0.5, square=True, cmap='Blues_r', annot_kws={"size": 18})
ax4.set_ylabel("Actual Label", fontsize=24)
ax4.set_xlabel("Predicted Label", fontsize=24)
all_sample_title="ResNet50 Training Round#4 \nAccuracy Score: {0:.3f}".format(res_test_eval_04[1])
ax4.set_title(all_sample_title, size=32)
ax4.set_ylim(len(res_conf_mat_04)-0.1, -0.1)

plt.tight_layout()

# Test data
filenames = test_generator.filenames
nb_samples = len(filenames)

class_labels = test_generator.class_indices
class_names = {value:key for key,value in class_labels.items()}

labels = (ind_test_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in res_pred_labels_04]

filenames = test_generator.filenames
results = pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})

random_files = results.sample(36)
filenames = random_files['Filename'].tolist()
predicted_labels = random_files['Predictions'].tolist()
test_file_paths = ['E:\\Major Project/output/test/'+ filename for filename in filenames]
# Tissue types dictionary mapping class names with full names
tissue_types = {"ADI": "Adipose tissue",
               "BACK": "Background",
               "DEB": "Debris",
               "LYM": "Lymphocyte aggregates",
               "MUC": "Mucus", 
               "MUS": "Muscle",
               "NORM": "Normal mucosa", 
               "STR": "Stroma",
               "TUM": "Tumor epithelium"}

fig = plt.figure(figsize=(45,60))
fig.subplots_adjust(top=0.88)
columns = 4
rows = 9

for i in range(1, columns*rows+1):
    fig.add_subplot(rows, columns, i)
    plt.imshow(mpimg.imread(test_file_paths[i-1]))
    plt.axis('off')
    true_label = test_file_paths[i-1].split('/')[-2]
    predicted_label = predicted_labels[i-1]
    plt.title("True Label: {}\nPredicted Label: {}".format(tissue_types[true_label], tissue_types[predicted_label]), fontsize=28)
plt.tight_layout()
    
plt.show()

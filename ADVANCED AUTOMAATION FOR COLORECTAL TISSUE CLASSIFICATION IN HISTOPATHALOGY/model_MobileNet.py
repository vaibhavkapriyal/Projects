import os
import pandas as pd
import numpy as np
from itertools import chain
import cv2
import splitfolders

# Plotting
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#%matplotlib inline
plt.style.use('ggplot')
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

# Create the base model from the pre-trained model MobileNet V2
base_model_mobilenetv2_01 = MobileNet(input_shape=(224,224,3),
                                      include_top=False)

for layer in base_model_mobilenetv2_01.layers:
    layer.trainable=False

# Add the top classification block
x = base_model_mobilenetv2_01.output
flat = GlobalAveragePooling2D()(x)
hidden_1 = Dense(1024, activation='relu')(flat)
drop_1 = Dropout(0.2)(hidden_1)
hidden_2 = Dense(512, activation='relu')(drop_1)
drop_2 = Dropout(0.3)(hidden_2)
output = Dense(9, activation= 'softmax')(drop_2)

model_mobilenet_01 = Model(base_model_mobilenetv2_01.inputs, output)
model_mobilenet_01.summary()

# Call Backs
filepath = 'mobilenet_base_model_wt.keras'
es_01 = EarlyStopping(monitor='loss', verbose=1, mode='min', patience=4)
cp_01 = ModelCheckpoint(filepath, monitor='val_loss',verbose=1, save_best_only=True,
                     save_weights_only=False, mode='auto', save_freq='epoch')

optimizer = Adam(learning_rate=0.0004)
model_mobilenet_01.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

mobilenetv2_history_01 = model_mobilenet_01.fit(train_generator,
                                                          steps_per_epoch=225,
                                                          epochs=10,
                                                          callbacks = [es_01, cp_01],
                                                          validation_data = valid_generator)
# save model
if not os.path.isdir('model_weights/'):
    os.mkdir('model_weights/')
model_mobilenet_01.save_weights(filepath='model_weights/mobilenet_base_model_wt.weights.h5', 
                               overwrite=True)
# Load the saved model
model_mobilenet_01.load_weights('mobilenet_base_model_wt.keras')

# Evaluate the model on the hold out validation and test datasets
mn_val_eval_01 = model_mobilenet_01.evaluate(valid_generator)
mn_test_eval_01 = model_mobilenet_01.evaluate(test_generator)

print('Validation loss:      {0:.3f}'.format(mn_val_eval_01[0]))
print('Validation accuracy:  {0:.3f}'.format(mn_val_eval_01[1]))
print('Test loss:            {0:.3f}'.format(mn_test_eval_01[0]))
print('Test accuracy:        {0:.3f}'.format(mn_test_eval_01[1]))

# Prediction Probabilities
nb_samples = len(test_generator)
mn_predictions_01 = model_mobilenet_01.predict_generator(test_generator,steps = nb_samples, verbose=1)
# Prediction Labels
mn_pred_labels_01 = np.argmax(mn_predictions_01, axis=1)

# Classification Report
print('|'+'-'*67+'|')
print('|-------Classification Report: MobileNet Training Cycle #1-----------|')
print('|'+'-'*67+'|')
print(classification_report(test_generator.classes, mn_pred_labels_01, 
                            target_names=['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']))

# Build a model
base_model_mobilenet_02 = MobileNet(input_shape=(224, 224, 3),
                                    include_top = False)
x = base_model_mobilenet_02.output
flat = GlobalAveragePooling2D()(x)
hidden_1 = Dense(1024, activation='relu')(flat)
drop_1 = Dropout(0.2)(hidden_1)
hidden_2 = Dense(512, activation='relu')(drop_1)
drop_2 = Dropout(0.3)(hidden_2)
output = Dense(9, activation= 'softmax')(drop_2)
model_mobilenet_02= Model(base_model_mobilenet_02.inputs, output)

# Load weights from the previous traning session
model_mobilenet_02.load_weights('model_weights/mobilenet_base_model_wt.keras')

# Freeze layers
for layer in model_mobilenet_02.layers[:82]:
    layer.trainable=False
    
# Call Backs
filepath = 'mobilenet_base_model_wt.keras'
es_02 = EarlyStopping(monitor='loss', verbose=1, mode='min', patience=4)
cp_02 = ModelCheckpoint(filepath, monitor='val_loss',verbose=1, save_best_only=True,
                     save_weights_only=False, mode='auto', save_freq='epoch')

# compile a model
optimizer = Adam(learning_rate=0.0004)
model_mobilenet_02.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
mobilenetv2_history_02 = model_mobilenet_02.fit(train_generator,
                                                          steps_per_epoch=225,
                                                          epochs=25,
                                                          callbacks = [es_02, cp_02],
                                                          validation_data = valid_generator)
# save model
if not os.path.isdir('model_weights/'):
    os.mkdir('model_weights/')
model_mobilenet_02.save_weights(filepath='mobilenet_model_02.weights.h5', 
                               overwrite=True)
# Load the saved model
model_mobilenet_02.load_weights('mobilenet_base_model_wt.keras')
# Evaluate the model on the hold out validation and test datasets

mn_val_eval_02 = model_mobilenet_02.evaluate(valid_generator)
mn_test_eval_02 = model_mobilenet_02.evaluate(test_generator)

print('Validation loss:     {}'.format(mn_val_eval_02[0]))
print('Validation accuracy: {}'.format(mn_val_eval_02[1]))
print('Test loss:           {}'.format(mn_test_eval_02[0]))
print('Test accuracy:       {}'.format(mn_test_eval_02[1]))

# Prediction Probabilities
nb_samples = len(test_generator)
mn_predictions_02 = model_mobilenet_02.predict_generator(test_generator,steps = nb_samples, verbose=1)
# Prediction Labels
mn_pred_labels_02 = np.argmax(mn_predictions_02, axis=1)

# Classification Report
print('|'+'-'*67+'|')
print('|-------Classification Report: MobileNet Training Cycle #2-----------|')
print('|'+'-'*67+'|')
print(classification_report(test_generator.classes, mn_pred_labels_02, 
                            target_names=['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']))
base_model_mobilenet_03 = MobileNet(input_shape=(224, 224, 3),
                                    include_top = False)
x = base_model_mobilenet_03.output
flat = GlobalAveragePooling2D()(x)
hidden_1 = Dense(1024, activation='relu')(flat)
drop_1 = Dropout(0.2)(hidden_1)
hidden_2 = Dense(512, activation='relu')(drop_1)
drop_2 = Dropout(0.3)(hidden_2)
output = Dense(9, activation= 'softmax')(drop_2)

model_mobilenet_03= Model(base_model_mobilenet_03.inputs, output)
model_mobilenet_03.load_weights('mobilenet_base_model_wt.keras')

# Call Backs
filepath = 'mobilenet_base_model_wt.keras'
es_03 = EarlyStopping(monitor='loss', verbose=1, mode='min', patience=4)
cp_03 = ModelCheckpoint(filepath, monitor='val_loss',verbose=1, save_best_only=True,
                     save_weights_only=False, mode='auto', save_freq='epoch')

for layer in model_mobilenet_03.layers[:55]:
    layer.trainable=False

optimizer = Adam(learning_rate=0.0004)   
model_mobilenet_03.compile(optimizer = optimizer,
                          loss = 'categorical_crossentropy',
                          metrics = ['accuracy'])
mobilenetv2_history_03 = model_mobilenet_03.fit(train_generator,
                                 steps_per_epoch=225,
                                 epochs=10,
                                 callbacks = [es_03, cp_03],
                                 validation_data = valid_generator)
# save model
if not os.path.isdir('model_weights/'):
    os.mkdir('model_weights/')
model_mobilenet_03.save_weights(filepath='model_weights/mobilenet_base_model_wt.weights.h5', overwrite=True)

# Load the saved model
model_mobilenet_03.load_weights('mobilenet_base_model_wt.keras')
# Evaluate the model on the hold out validation and test datasets

mn_val_eval_03 = model_mobilenet_03.evaluate(valid_generator)
mn_test_eval_03 = model_mobilenet_03.evaluate(test_generator)

print('Validation loss:       {}'.format(mn_val_eval_03[0]))
print('Validation accuracy:   {}'.format(mn_val_eval_03[1]))
print('Test loss:             {}'.format(mn_test_eval_03[0]))
print('Test accuracy:         {}'.format(mn_test_eval_03[1]))

# Predict probability
nb_samples = len(test_generator)
mn_predictions_03 = model_mobilenet_03.predict_generator(test_generator,steps = nb_samples,verbose=1)
# Prediction labels                                                  
mn_pred_labels_03 = np.argmax(mn_predictions_03, axis=1)

# Classification Report
print('|'+'-'*67+'|')
print('|-------Classification Report: MobileNet Training Cycle #3----------|')
print('|'+'-'*67+'|')
print(classification_report(test_generator.classes, mn_pred_labels_03, 
                            target_names=['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']))
base_model_mobilenet_04 = MobileNet(input_shape=(224, 224, 3),
                                    include_top = False)
x = base_model_mobilenet_04.output
flat = GlobalAveragePooling2D()(x)
hidden_1 = Dense(1024, activation='relu')(flat)
drop_1 = Dropout(0.2)(hidden_1)
hidden_2 = Dense(512, activation='relu')(drop_1)
drop_2 = Dropout(0.3)(hidden_2)
output = Dense(9, activation= 'softmax')(drop_2)

model_mobilenet_final= Model(base_model_mobilenet_04.inputs, output)

# Load weights from the previous traning session
model_mobilenet_final.load_weights('mobilenet_base_model_wt.keras')

# Call Backs
filepath = 'mobilenet_base_model_wt.keras'
es_04 = EarlyStopping(monitor='loss', verbose=1, mode='min', patience=4)
cp_04 = ModelCheckpoint(filepath, monitor='val_loss',verbose=1, save_best_only=True,
                     save_weights_only=False, mode='auto', save_freq='epoch')

# For the final tuning of the entire let's use Stochastic Gradient Descent and slow tuning
sgd = SGD(lr=.00001, decay=1e-6, momentum=0.9, nesterov=True)
model_mobilenet_final.compile(optimizer = sgd,
                          loss = 'categorical_crossentropy',
                          metrics = ['accuracy'])
mobilenetv2_history_final = model_mobilenet_final.fit(train_generator,
                                 steps_per_epoch=225,
                                 epochs=50,
                                 callbacks = [es_04, cp_04],
                                 validation_data = valid_generator)
# Load the saved model
model_mobilenet_final.load_weights('mobilenet_base_model_wt.keras')
# Evaluate the model on the hold out validation and test datasets

mn_val_eval_final = model_mobilenet_final.evaluate(valid_generator)
mn_test_eval_final = model_mobilenet_final.evaluate(test_generator)

print('Validation loss:      {}'.format(mn_val_eval_final[0]))
print('Validation accuracy:  {}'.format(mn_val_eval_final[1]))
print('Test loss:            {}'.format(mn_test_eval_final[0]))
print('Test accuracy:        {}'.format(mn_test_eval_final[1]))

# Prediction probability
nb_samples = len(test_generator)
mn_predictions_final= model_mobilenet_final.predict_generator(test_generator,steps = nb_samples,verbose=1)
# Predict labels
mn_pred_labels_final = np.argmax(mn_predictions_final, axis=1)

# Classification Report
print('|'+'-'*67+'|')
print('|-------Classification Report: MobileNet Training Cycle #3----------|')
print('|'+'-'*67+'|')

print(classification_report(test_generator.classes, mn_pred_labels_final, 
                            target_names=['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']))
mn_conf_mat_01 = pd.DataFrame(confusion_matrix(test_generator.classes, mn_pred_labels_01), 
                        index=['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM'], 
                        columns=['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM'])


mn_conf_mat_02 = pd.DataFrame(confusion_matrix(test_generator.classes, mn_pred_labels_02), 
                        index=['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM'], 
                        columns=['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM'])



mn_conf_mat_03 = pd.DataFrame(confusion_matrix(test_generator.classes, mn_pred_labels_03), 
                        index=['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM'], 
                        columns=['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM'])


mn_conf_mat_final = pd.DataFrame(confusion_matrix(test_generator.classes, mn_pred_labels_final), 
                        index=['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM'], 
                        columns=['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM'])
# Plot confusion matrices
sns.set(font_scale=1.8)
fig, ([ax1,ax2],[ax3,ax4]) = plt.subplots(nrows=2, ncols=2, figsize=(28,28))

#ax1
sns.heatmap(mn_conf_mat_01, ax=ax1, annot=True, fmt=".1f", linewidths=0.5, square=True, cmap='Blues_r', annot_kws={"size": 18})
ax1.set_ylabel("Actual Label", fontsize=24)
ax1.set_xlabel("Predicted Label", fontsize=24)
all_sample_title="MobileNet Training Round #1 \nAccuracy Score: {0:.3f}".format(mn_test_eval_01[1])
ax1.set_title(all_sample_title, size=32)
ax1.set_ylim(len(mn_conf_mat_01)-0.1, -0.1)

#ax2
sns.heatmap(mn_conf_mat_02, ax=ax2, annot=True, fmt=".1f", linewidths=0.5, square=True, cmap='Blues_r', annot_kws={"size": 18})
ax2.set_ylabel("Actual Label", fontsize=24)
ax2.set_xlabel("Predicted Label", fontsize=24)
all_sample_title="MobileNet Training Round #2 \nAccuracy Score: {0:.3f}".format(mn_test_eval_02[1])
ax2.set_title(all_sample_title, size=32)
ax2.set_ylim(len(mn_conf_mat_02)-0.1, -0.1)

#ax3
sns.heatmap(mn_conf_mat_03, ax=ax3, annot=True, fmt=".1f", linewidths=0.5, square=True, cmap='Blues_r', annot_kws={"size": 18})
ax3.set_ylabel("Actual Label", fontsize=24)
ax3.set_xlabel("Predicted Label", fontsize=24)
all_sample_title="MobileNet Training Round#3 \nAccuracy Score: {0:.3f}".format(mn_test_eval_03[1])
ax3.set_title(all_sample_title, size=32)
ax3.set_ylim(len(mn_conf_mat_03)-0.1, -0.1)

#ax4
sns.heatmap(mn_conf_mat_final, ax=ax4, annot=True, fmt=".1f", linewidths=0.5, square=True, cmap='Blues_r', annot_kws={"size": 18})
ax4.set_ylabel("Actual Label", fontsize=24)
ax4.set_xlabel("Predicted Label", fontsize=24)
all_sample_title="MobileNet Training Round#4 \nAccuracy Score: {0:.3f}".format(mn_test_eval_final[1])
ax4.set_title(all_sample_title, size=32)
ax4.set_ylim(len(mn_conf_mat_final)-0.1, -0.1)


plt.tight_layout()







# Import Modules
import os
import pandas as pd
import numpy as np
from itertools import chain
import cv2
import splitfolders

# Plotting
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
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
#splitfolders.ratio(input_, 'output="output"', seed = 101, ratio=(0.8, 0.1, 0.1))

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


# Instantiate the vgg19 model without the top classifier. 
base_model = VGG19(input_shape=(224, 224, 3), weights='imagenet', include_top = False)

# Add a classifier to the convolution block classifier
x = base_model.output
flat = Flatten()(x)
hidden_1 = Dense(1024, activation='relu')(flat)
drop_1 = Dropout(0.2)(hidden_1)
hidden_2 = Dense(512, activation='relu')(drop_1)
drop_2 = Dropout(0.3)(hidden_2)
output = Dense(9, activation= 'softmax')(drop_2)

# Define the model
model_vgg19_01 = Model(base_model.inputs, output)

# Freeze all layers in the convolution block. We don't want to train these weights yet.
for layer in base_model.layers:
        layer.trainable=False
        
#model_vgg19_01.summary()

# Call Backs
filepath = 'vgg19_base_model_wt.keras'
es = EarlyStopping(monitor='loss', verbose=1, mode='min', patience=4)
cp = ModelCheckpoint(filepath, monitor='val_loss',verbose=1, save_best_only=True,
                     save_weights_only=False, mode='auto', save_freq='epoch')


# Compile a model
optimizer = Adam(learning_rate=0.0004)
model_vgg19_01.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Model fitting
history_01 = model_vgg19_01.fit(
            train_generator,
            steps_per_epoch=int(len(input_)/64),
            epochs=10,
            callbacks = [es, cp],
            validation_data = valid_generator)


# save model
if not os.path.isdir('model_weights/'):
    os.mkdir('model_weights/')
model_vgg19_01.save_weights(filepath='model_weights/vgg19_base_model_wt.weights.h5', overwrite=True)

print(history_01.history.keys())
# Load the saved model
model_vgg19_01.load_weights('vgg19_base_model_wt.keras')
# Evaluate the model on the hold out validation and test datasets

vgg_val_eval_01 = model_vgg19_01.evaluate(valid_generator)
vgg_test_eval_01 = model_vgg19_01.evaluate(test_generator)

print('Validation loss:     {}'.format(vgg_val_eval_01[0]))
print('Validation accuracy: {}'.format(vgg_val_eval_01[1]))
print('Test loss:           {}'.format(vgg_test_eval_01[0]))
print('Test accuracy:       {}'.format(vgg_test_eval_01[1]))

# Predict probabilities
nb_samples = len(test_generator)
vgg_predictions_01 = model_vgg19_01.predict(test_generator,steps = nb_samples,verbose=1)

# Predict labels
vgg_pred_labels_01 = np.argmax(vgg_predictions_01, axis=1)

# Classification Report
print('|'+'-'*75+'|')
print('|-------------------Classification Report: Training Cycle #1----------------|')
print('|'+'-'*75+'|')
print(classification_report(test_generator.classes, vgg_pred_labels_01, 
                            target_names=['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']))

# Plot performance of vgg19 base model
fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
fig.suptitle("VGG19 Base Model Training", fontsize=20)
max_epoch = len(history_01.history['accuracy'])+1
epochs_list = list(range(1, max_epoch))

ax1.plot(epochs_list, history_01.history['accuracy'], color='b', linestyle='-', label='Training Data')
ax1.plot(epochs_list, history_01.history['val_accuracy'], color='r', linestyle='-', label ='Validation Data')
ax1.set_title('Training Accuracy', fontsize=14)
ax1.set_xlabel('Epochs', fontsize=14)
ax1.set_ylabel('Accuracy', fontsize=14)
ax1.legend(frameon=False, loc='lower center', ncol=2)

ax2.plot(epochs_list, history_01.history['loss'], color='b', linestyle='-', label='Training Data')
ax2.plot(epochs_list, history_01.history['val_loss'], color='r', linestyle='-', label ='Validation Data')
ax2.set_title('Training Loss', fontsize=14)
ax2.set_xlabel('Epochs', fontsize=14)
ax2.set_ylabel('Loss', fontsize=14)
ax2.legend(frameon=False, loc='upper center', ncol=2)


# Construct VGG19 model without the classifer and weights trained on imagenet data
base_model_02 = VGG19(input_shape=(224, 224, 3),
                 include_top = False)
x = base_model_02.output
flat = Flatten()(x)
hidden_1 = Dense(1024, activation='relu')(flat)
drop_1 = Dropout(0.2)(hidden_1)
hidden_2 = Dense(512, activation='relu')(drop_1)
drop_2 = Dropout(0.3)(hidden_2)
output = Dense(9, activation= 'softmax')(drop_2)

model_vgg19_02 = Model(base_model_02.inputs, output)
# Load weights from the trained base model
model_vgg19_02.load_weights('vgg19_base_model_wt.keras')


# Freeze layers upto the 19th layer.
for layer in model_vgg19_02.layers[:19]:
    layer.trainable =  False
    
# Call Backs
filepath = 'vgg19_model_wt_ft_01.keras'
es = EarlyStopping(monitor='loss', verbose=1, mode='min', patience=4)
cp = ModelCheckpoint(filepath, monitor='val_loss',verbose=1, save_best_only=True,
                     save_weights_only=False, mode='auto', save_freq='epoch')

#print(model_vgg19_02.summary())
optimizer = Adam(learning_rate=0.0004)
model_vgg19_02.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])


# Model fitting
history_02 = model_vgg19_02.fit(
            train_generator,
            steps_per_epoch=int(len(input_)/64),
            epochs=10,
            callbacks = [es, cp],
            validation_data = valid_generator)


# save model
if not os.path.isdir('model_weights/'):
    os.mkdir('model_weights/')
model_vgg19_02.save_weights(filepath='model_weights/vgg19_model_wt_ft_01.weights.h5', overwrite=True)

# Plot performance of vgg19 finetune model
fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
fig.suptitle("VGG19 Fine Tuning (Unfreeze 2 Conv Layers)", fontsize=20)
max_epoch = len(history_02.history['accuracy'])+1
epochs_list = list(range(1, max_epoch))

ax1.plot(epochs_list, history_02.history['accuracy'], color='b', linestyle='-', label='Training Data')
ax1.plot(epochs_list, history_02.history['val_accuracy'], color='r', linestyle='-', label ='Validation Data')
ax1.set_title('Training Accuracy', fontsize=14)
ax1.set_xlabel('Epochs', fontsize=14)
ax1.set_ylabel('Accuracy', fontsize=14)
ax1.legend(frameon=False, loc='lower center', ncol=2)

ax2.plot(epochs_list, history_02.history['loss'], color='b', linestyle='-', label='Training Data')
ax2.plot(epochs_list, history_02.history['val_loss'], color='r', linestyle='-', label ='Validation Data')
ax2.set_title('Training Loss', fontsize=14)
ax2.set_xlabel('Epochs', fontsize=14)
ax2.set_ylabel('Loss', fontsize=14)
ax2.legend(frameon=False, loc='upper center', ncol=2)


# Load the saved model
model_vgg19_02.load_weights('vgg19_model_wt_ft_01.keras')
# Evaluate the model on the hold out validation and test datasets

vgg_val_eval_02 = model_vgg19_02.evaluate(valid_generator)
vgg_test_eval_02 = model_vgg19_02.evaluate(test_generator)

print('Validation loss:     {0:.3f}'.format(vgg_val_eval_02[0]))
print('Validation accuracy: {0:.3f}'.format(vgg_val_eval_02[1]))
print('Test loss:           {0:.3f}'.format(vgg_test_eval_02[0]))
print('Test accuracy:       {0:.3f}'.format(vgg_test_eval_02[1]))

# Predict probabilities
nb_samples = len(test_generator)
vgg_predictions_02 = model_vgg19_02.predict(test_generator,steps = nb_samples,verbose=1)

# Predict labels
vgg_pred_labels_02 = np.argmax(vgg_predictions_02, axis=1)

# Classification Report
print('|'+'-'*67+'|')
print('|------------Classification Report: VGG19 Training Cycle #2-----------|')
print('|'+'-'*67+'|')
print(classification_report(test_generator.classes, vgg_pred_labels_02, 
                            target_names=['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']))


# Build VGG19
base_model_03 = VGG19(input_shape=(224, 224, 3),
                 include_top = False)
x = base_model_03.output
flat = Flatten()(x)
hidden_1 = Dense(1024, activation='relu')(flat)
drop_1 = Dropout(0.2)(hidden_1)
hidden_2 = Dense(512, activation='relu')(drop_1)
drop_2 = Dropout(0.3)(hidden_2)
output = Dense(9, activation= 'softmax')(drop_2)

model_vgg19_03 = Model(base_model_03.inputs, output)
# Load the weights saved after the first round of fine tuning
model_vgg19_03.load_weights('vgg19_model_wt_ft_01.keras')

# Callbacks
filepath = 'vgg19_model_finetuned.keras'

cp = ModelCheckpoint(filepath, monitor='val_loss',verbose=1, save_best_only=True,
                     save_weights_only=False, mode='auto', save_freq='epoch')

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.0001)

optimizer = Adam(learning_rate=0.0004)
model_vgg19_03.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])



# Model Fitting
history_03 = model_vgg19_03.fit(
            train_generator,
            steps_per_epoch=int(len(input_)/64),
            epochs=25,
            callbacks = [learning_rate_reduction, cp],
            validation_data = valid_generator)


# save model
if not os.path.isdir('model_weights/'):
    os.mkdir('model_weights/')
model_vgg19_03.save_weights(filepath='model_weights/vgg19_model_finetuned.weights.h5', overwrite=True)
model_vgg19_03.save("vgg19_model_finetuned.keras")


# Load the saved model
model_vgg19_03.load_weights('vgg19_model_finetuned.keras')
# Evaluate the model on the hold out validation and test datasets

vgg_val_eval_03 = model_vgg19_03.evaluate(valid_generator)
vgg_test_eval_03 = model_vgg19_03.evaluate(test_generator)

print('Validation loss:       {0:.3f}'.format(vgg_val_eval_03[0]))
print('Validation accuracy:   {0:.3f}'.format(vgg_val_eval_03[1]))
print('Test loss:             {0:.3f}'.format(vgg_test_eval_03[0]))
print('Test accuracy:         {0:.3f}'.format(vgg_test_eval_03[1]))

# Predict probabilities
nb_samples = len(test_generator)
vgg_predictions_03 = model_vgg19_03.predict(test_generator,steps = nb_samples, verbose=1)
# Predict labels
vgg_pred_labels_03 = np.argmax(vgg_predictions_03, axis=1)

# Classification Report
print('|'+'-'*67+'|')
print('|------------Classification Report: VGG19 Training Cycle #3-----------|')
print('|'+'-'*67+'|')
print(classification_report(test_generator.classes, vgg_pred_labels_03, 
                            target_names=['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']))



# Confusion Matrices
vgg_conf_mat_01 = pd.DataFrame(confusion_matrix(test_generator.classes, vgg_pred_labels_01), 
                        index=['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM'], 
                        columns=['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM'])


vgg_conf_mat_02 = pd.DataFrame(confusion_matrix(test_generator.classes, vgg_pred_labels_02), 
                        index=['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM'], 
                        columns=['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM'])


vgg_conf_mat_03 = pd.DataFrame(confusion_matrix(test_generator.classes, vgg_pred_labels_03), 
                        index=['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM'], 
                        columns=['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM'])

# Plotting Confusion Matrices
sns.set(font_scale=1.2)
fig, (ax1,ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(7,35))

#ax1
sns.heatmap(vgg_conf_mat_01, ax=ax1, annot=True, fmt=".1f", linewidths=0.5, square=True, cmap='Blues_r', annot_kws={"size": 10})
ax1.set_ylabel("Actual Label", fontsize=20)
ax1.set_xlabel("Predicted Label", fontsize=20)
all_sample_title="VGG19 Training Round #1 \nAccuracy Score: {0:.3f}".format(vgg_test_eval_01[1])
ax1.set_title(all_sample_title, size=24)
ax1.set_ylim(len(vgg_conf_mat_01)-0.1, -0.1)


#ax2
sns.heatmap(vgg_conf_mat_02, ax=ax2, annot=True, fmt=".1f", linewidths=0.5, square=True, cmap='Blues_r', annot_kws={"size": 10})
ax2.set_ylabel("Actual Label", fontsize=20)
ax2.set_xlabel("Predicted Label", fontsize=20)
all_sample_title="VGG19 Training Round #2 \nAccuracy Score: {0:.3f}".format(vgg_test_eval_02[1])
ax2.set_title(all_sample_title, size=24)
ax2.set_ylim(len(vgg_conf_mat_02)-0.1, -0.1)

#ax3
sns.heatmap(vgg_conf_mat_03, ax=ax3, annot=True, fmt=".1f", linewidths=0.5, square=True, cmap='Blues_r', annot_kws={"size": 10})
ax3.set_ylabel("Actual Label", fontsize=20)
ax3.set_xlabel("Predicted Label", fontsize=20)
all_sample_title="VGG19 Training Round#3 \nAccuracy Score: {0:.3f}".format(vgg_test_eval_03[1])
ax3.set_title(all_sample_title, size=24)
ax3.set_ylim(len(vgg_conf_mat_03)-0.1, -0.1)



from itertools import chain

# training_accuracy
training_accuracy = []
training_accuracy.append(history_01.history['accuracy'])
training_accuracy.append(history_02.history['accuracy'])
training_accuracy.append(history_03.history['accuracy'])
training_accuracy_ = list(itertools.chain(*training_accuracy))

# training_loss
training_loss = []
training_loss.append(history_01.history['loss'])
training_loss.append(history_02.history['loss'])
training_loss.append(history_03.history['loss'])
training_loss_ = list(itertools.chain(*training_loss))

# validation_accuracy
validation_accuracy = []
validation_accuracy.append(history_01.history['val_accuracy'])
validation_accuracy.append(history_02.history['val_accuracy'])
validation_accuracy.append(history_03.history['val_accuracy'])
validation_accuracy_ = list(itertools.chain(*validation_accuracy))

# validation_loss
validation_loss = []
validation_loss.append(history_01.history['val_loss'])
validation_loss.append(history_02.history['val_loss'])
validation_loss.append(history_03.history['val_loss'])
validation_loss_ = list(itertools.chain(*validation_loss))


training_metrics_df = pd.DataFrame({'training_accuracy': training_accuracy_,
                                   'training_loss': training_loss_,
                                   'validation_accuracy': validation_accuracy_,
                                   'validation_loss': validation_loss_})
training_metrics_df.to_csv('training_metrics_03.csv', index=False)
# Plot performance of vgg19 finetune model
fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
fig.suptitle("VGG19 Fine Tuning (Unfreeze All Layers)", fontsize=20)
max_epoch = len(history_03.history['accuracy'])+1
epochs_list = list(range(1, max_epoch))

ax1.plot(epochs_list, history_03.history['accuracy'], color='b', linestyle='-', label='Training Data')
ax1.plot(epochs_list, history_03.history['val_accuracy'], color='r', linestyle='-', label ='Validation Data')
ax1.set_title('Training Accuracy', fontsize=14)
ax1.set_xlabel('Epochs', fontsize=14)
ax1.set_ylabel('Accuracy', fontsize=14)
ax1.legend(frameon=False, loc='lower center', ncol=2)

ax2.plot(epochs_list, history_03.history['loss'], color='b', linestyle='-', label='Training Data')
ax2.plot(epochs_list, history_03.history['val_loss'], color='r', linestyle='-', label ='Validation Data')
ax2.set_title('Training Loss', fontsize=14)
ax2.set_xlabel('Epochs', fontsize=14)
ax2.set_ylabel('Loss', fontsize=14)
ax2.legend(frameon=False, loc='upper center', ncol=2)



"""
Dog vs Cat Convolutional Neural Network Classifier
Problem Statement and Dataset: https://www.kaggle.com/c/dogs-vs-cats
"""

import time
from warnings import filterwarnings
filterwarnings('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# For Jupyter Notebook environments (Optional)
# %matplotlib inline

# ==========================================
# 1. MODEL ARCHITECTURE
# ==========================================
def build_classifier():
    classifier = Sequential()

    # First Convolutional Block
    classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    # Second Convolutional Block
    classifier.add(Conv2D(32, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    # Flattening & Fully Connected Layer
    classifier.add(Flatten())
    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(Dense(units=1, activation='sigmoid'))

    # Optimizer Config
    adam = tf.keras.optimizers.Adam(
        learning_rate=0.001, 
        beta_1=0.9, 
        beta_2=0.999, 
        epsilon=1e-07, 
        amsgrad=False
    )

    classifier.compile(
        optimizer=adam, 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    
    return classifier

# ==========================================
# 2. DATA PIPELINE (AUGMENTATION)
# ==========================================
def load_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    # Training Split (Expected ~19,998 images)
    train_set = train_datagen.flow_from_directory(
        'train',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary'
    )

    # Validation Split (Expected ~5,000 images)
    test_set = test_datagen.flow_from_directory(
        'test',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )

    # Unseen Test Split (Expected ~12,500 images)
    test_set1 = test_datagen.flow_from_directory(
        'test1',
        target_size=(64, 64),
        batch_size=32,
        shuffle=False
    )
    
    return train_set, test_set, test_set1

# ==========================================
# 3. TRAINING AND EVALUATION PIPELINE
# ==========================================
if __name__ == '__main__':
    # Initialize Architecture
    classifier = build_classifier()
    classifier.summary()

    # Load Data Stream
    train_set, test_set, test_set1 = load_data_generators()

    # TensorBoard setup
    tensorboard = TensorBoard(log_dir="logs/{}".format(int(time.time())))

    # Fit Model
    print("--- Training Started ---")
    classifier.fit(
        train_set,
        steps_per_epoch=800,
        epochs=200,
        validation_data=test_set,
        validation_steps=20,
        callbacks=[tensorboard]
    )

    # Save Model Weights
    classifier.save('resources/dogcat_model_bak.h5')
    print("Model saved to resources/dogcat_model_bak.h5")

    # Evaluate Accuracy/Loss Metrics
    x1 = classifier.evaluate(train_set)
    x2 = classifier.evaluate(test_set)

    print('\n--- Performance Metrics ---')
    print('Training Accuracy : %1.2f%%' % (x1[1] * 100))
    print('Training Loss     : %1.6f' % (x1[0]))
    print('Validation Accuracy: %1.2f%%' % (x2[1] * 100))
    print('Validation Loss   : %1.6f' % (x2[0]))

    # ==========================================
    # 4. CONFUSION MATRIX & ERROR ANALYSIS
    # ==========================================
    test_set.reset()
    ytesthat = classifier.predict(test_set)

    df = pd.DataFrame({
        'filename': test_set.filenames,
        'predict': ytesthat[:, 0],
        'y': test_set.classes
    })

    pd.set_option('display.float_format', lambda x: '%.5f' % x)
    df['y_pred'] = (df['predict'] > 0.5).astype(int)

    misclassified = df[df['y'] != df['y_pred']]
    print(f'Total misclassified validation images: {misclassified["y"].count()}')

    # Confusion matrix generation
    conf_matrix = confusion_matrix(df.y, df.y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, cmap="YlGnBu", annot=True, fmt='g')
    plt.xlabel('Predicted Value')
    plt.ylabel('True Value')
    plt.title('Validation Set Confusion Matrix')
    plt.savefig('resources/confusion_matrix.png')
    plt.show()

    # ==========================================
    # 5. ERROR VISUALIZATION
    # ==========================================
    # Plotting Cat images misclassified as Dogs
    CatasDog = df['filename'][(df.y == 0) & (df.y_pred == 1)]
    if len(CatasDog) > 0:
        fig = plt.figure(figsize=(15, 6))
        columns, rows = 7, 3
        for i in range(min(columns * rows, len(CatasDog))):
            img = image.load_img('test/' + CatasDog.iloc[i], target_size=(64, 64))
            fig.add_subplot(rows, columns, i + 1)
            plt.axis('off')
            plt.imshow(img)
        plt.suptitle('Cats Misclassified as Dogs')
        plt.show()

    # Plotting Dog images misclassified as Cats
    DogasCat = df['filename'][(df.y == 1) & (df.y_pred == 0)]
    if len(DogasCat) > 0:
        fig = plt.figure(figsize=(15, 6))
        columns, rows = 7, 3
        for i in range(min(columns * rows, len(DogasCat))):
            img = image.load_img('test/' + DogasCat.iloc[i], target_size=(64, 64))
            fig.add_subplot(rows, columns, i + 1)
            plt.axis('off')
            plt.imshow(img)
        plt.suptitle('Dogs Misclassified as Cats')
        plt.show()

    # ==========================================
    # 6. FEATURE MAP VISUALIZATIONS
    # ==========================================
    # Pick a baseline image for visualization
    img_path = 'test/Cat/14.jpg'
    img1 = image.load_img(img_path, target_size=(64, 64))
    img_arr = image.img_to_array(img1) / 255.0
    img_input = np.expand_dims(img_arr, axis=0)

    # Submodels mapping input -> layer outputs
    conv2d_6_output = Model(inputs=classifier.input, outputs=classifier.get_layer('conv2d_6').output)
    conv2d_7_output = Model(inputs=classifier.input, outputs=classifier.get_layer('conv2d_7').output)

    conv2d_6_features = conv2d_6_output.predict(img_input)
    conv2d_7_features = conv2d_7_output.predict(img_input)

    # Plot Grid Matrix of 1st Layer Activations (32 Filter features)
    fig = plt.figure(figsize=(14, 7))
    columns, rows = 8, 4
    for i in range(columns * rows):
        fig.add_subplot(rows, columns, i + 1)
        plt.axis('off')
        plt.title(f'filter {i}', fontsize=8)
        plt.imshow(conv2d_6_features[0, :, :, i], cmap='gray')
    plt.suptitle('First Convolutional Layer Features')
    plt.show()

    # ==========================================
    # 7. UNSEEN INFERENCE ENGINE DEMO
    # ==========================================
    fig = plt.figure(figsize=(15, 6))
    columns, rows = 7, 3

    for i in range(columns * rows):
        fig.add_subplot(rows, columns, i + 1)
        random_filename = np.random.choice(test_set1.filenames)
        
        img_unseen = image.load_img('test1/' + random_filename, target_size=(64, 64))
        img_arr_unseen = image.img_to_array(img_unseen) / 255.0
        img_ready = np.expand_dims(img_arr_unseen, axis=0)
        
        prediction = classifier.predict(img_ready, verbose=0)
        
        if prediction[0, 0] > 0.5:
            label = 'Dog: %1.2f' % (prediction[0, 0])
        else:
            label = 'Cat: %1.2f' % (1.0 - prediction[0, 0])
            
        plt.text(5, 10, label, color='red', fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
        plt.imshow(img_unseen)
        plt.axis('off')
        
    plt.suptitle('Model Performance Matrix on Unseen Data')
    plt.show()
  

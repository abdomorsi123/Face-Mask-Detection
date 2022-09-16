import pandas as pd
import numpy as np
import seaborn as sns
import os
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import cv2
from scipy.spatial import distance
import glob
from warnings import filterwarnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model, load_model
from sklearn.metrics import classification_report , confusion_matrix
from PIL import Image                                                                                
from skimage import io
filterwarnings("ignore")



class MaskPrediction:
    def __init__(self):
        self.model = Sequential()


        
    def get_data(self):
    
        self.path  = 'C:/Users/Abdalrhman Morsi/Desktop/face musk detectioon/Face Mask Dataset/'
        self.dataset = {"image_path":[],"mask_status":[],"where":[]}
        for where in os.listdir(self.path):
            for status in os.listdir(self.path+"/"+where):
                for image in glob.glob(self.path+where+"/"+status+"/"+"*.png"):
                    self.dataset["image_path"].append(image)
                    self.dataset["mask_status"].append(status)
                    self.dataset["where"].append(where)
        self.dataset = pd.DataFrame(self.dataset)
        
        print("Data Scoring Done")
    
    def preprocess(self):
        self.get_data()
        train_df = self.dataset[self.dataset["where"] == "Train"]
        test_df = self.dataset[self.dataset["where"] == "Test"]
        valid_df = self.dataset[self.dataset["where"] == "Validation"]
        
        train_df = train_df.sample(frac=1)
        test_df = test_df.sample(frac=1)
        valid_df = valid_df.sample(frac=1)
        
        datagen = ImageDataGenerator(rescale = 1./255)
        
        self.__train_generator=datagen.flow_from_dataframe( 
                                                    dataframe=train_df,
                                                    directory="../input",
                                                    x_col="image_path",
                                                    y_col="mask_status",
                                                    batch_size=80,
                                                    seed=42,
                                                    shuffle=False,
                                                    class_mode="binary",
                                                    target_size=(150,150))
        
        self.__valid_generator=datagen.flow_from_dataframe(
                                                    dataframe=valid_df,
                                                    directory="../input",
                                                    x_col="image_path",
                                                    y_col="mask_status",
                                                    batch_size=80,
                                                    seed=42,
                                                    shuffle=False,
                                                    class_mode="binary",
                                                    target_size=(150,150))
        
        self.__test_generator=datagen.flow_from_dataframe(
                                                    dataframe=test_df,
                                                    directory="../input",
                                                    x_col="image_path",
                                                    y_col="mask_status",
                                                    batch_size=80,
                                                    seed=42,
                                                    shuffle=False,
                                                    class_mode="binary",
                                                    target_size=(150,150))
      
        
        print("Data Preprocessing Done")


    def classify(self):
        self.preprocess()
        self.__model = load_model('C:/Users/Abdalrhman Morsi/Desktop/face musk detectioon/face_musk.h5')
        acc= (self.__model.evaluate_generator(self.__test_generator, verbose=1))[1]
        return (round(acc*100,2))
    
    
    def predict(self,image):
        face_model = cv2.CascadeClassifier('C:/Users/Abdalrhman Morsi/Desktop/face musk detectioon/faces XML/haarcascade_frontalface_alt.xml')
        mask_label = {0:'Has Mask!',1:'No Mask'}
        dist_label = {0:(0,255,0),1:(255,0,0)}
        MIN_DISTANCE = 0
        
        img = io.imread(image)
        image1= ''.join(char for char in image if char.isalnum())
        faces = face_model.detectMultiScale(img,scaleFactor=1.1, minNeighbors=4)


        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
        if len(faces)>=1:
            label = [0 for i in range(len(faces))]
            for i in range(len(faces)-1):
                for j in range(i+1, len(faces)):
                    dist = distance.euclidean(faces[i][:2],faces[j][:2])
                    if dist<MIN_DISTANCE:
                        label[i] = 1
                        label[j] = 1
            for i in range(len(faces)):
                (x,y,w,h) = faces[i]
                crop = img[y:y+h,x:x+w]
                crop = cv2.resize(crop,(150,150))
                crop = np.reshape(crop,[1,150,150,3])/255.0
                mask_result = self.__model.predict(crop)
                cv2.putText(img,mask_label[round(mask_result[0][0])],(x, y), cv2.FONT_HERSHEY_SIMPLEX,1,dist_label[label[i]],2)
                cv2.rectangle(img,(x,y),(x+w,y+h),dist_label[label[i]],1)
                plt.figure(figsize=(40,40))
                plt.imshow(img)
                plt.savefig(f"{image1}.jpeg",bbox_inches="tight")
            return (image1)

        else:
            print("No Face!")
                  
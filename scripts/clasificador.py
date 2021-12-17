from keras.models import load_model
import cv2
import numpy as np

class Clasificador():
    def __init__(self):
        self.model = load_model('modelo/best_model.h5')
        self.parser = {"1":"Banana", "5":"Ciruela", "2":"Kiwi", "3":"Mango", "0":"Manzana", "4":"Naranja"}

    def pred(self,img):
        self.im_pred = img.copy()
        self.im_pred = cv2.resize(self.im_pred,(150,150))
        self.im_pred = self.im_pred[np.newaxis,:,:,:]
        y_pred = self.model.predict(self.im_pred)
        
        max_pred = np.max(y_pred)
        return self.parser[str(y_pred.argmax())]
  
    
    
   
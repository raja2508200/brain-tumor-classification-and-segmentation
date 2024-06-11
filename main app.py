import os
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
import cv2
from PIL import Image 
from skimage import transform
from flask import Flask, redirect, url_for, request, render_template
from glioma_seg import gliomaseg
from meningioma_seg import meningioma
from pitiutary import pituitary
app = Flask(__name__,static_url_path='/',
            static_folder='./',template_folder='./')

MODEL_VGG16_PATH = 'brain_tumor.h5'


MODEL_VGG16 = load_model(MODEL_VGG16_PATH)
MODEL_VGG16.make_predict_function()          # Necessary



def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (224, 224, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

def model_predict(img_path, model):
    image = load(img_path)
    preds = model.predict(image,batch_size=32)
    return preds

@app.route('/')
@app.route('/index.html')
def index():
   return render_template('templates/index.html')

@app.route('/upload.html')
def upload():
   return render_template("templates/upload.html")

@app.route('/upload_chest.html')
def upload_chest():
   return render_template('templates/upload_chest.html')
a=[]
@app.route('/uploaded_chest', methods = ['POST', 'GET'])
def uploaded_chest():
   if request.method == 'POST':
           f = request.files['file']
           basepath = os.path.dirname(__file__)
           file_path = os.path.join(basepath,"uploads/upload_img.png")
           f.save("uploads/upload_img.png")
           lst = ["Glioma","Meningioma","Normal","Pituitary"]
           preds_VGG16 = model_predict(file_path, MODEL_VGG16)
           pred_class_VGG16 = np.argmax(preds_VGG16)            
           result_VGG16 = str(lst[pred_class_VGG16])
           a.append(result_VGG16)
           return render_template('templates/results_chest.html',result=result_VGG16)
   return render_template('templates/index.html')


@app.route('/segment', methods=['GET','POST'])
def segment():
         if a[-1] == "Pituitary" :
            stage=pituitary()
            return render_template('templates/segment.html',stage=stage)       
         elif a[-1]=="Meningioma":
           stage=meningioma()
           return render_template('templates/segment.html',stage=stage)
         elif a[-1] == "Glioma":
            stage=gliomaseg()
            return render_template('templates/segment.html',stage=stage)
         else:
            return "cannot segment this image because this normal brain"
      



if __name__ == '__main__':
   app.run(port="3000", debug=False)

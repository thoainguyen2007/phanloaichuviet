from tensorflow import keras
model = keras.models.load_model('something(1).hdf5')
#!pip install -qq gradio
import gradio as gr
import numpy as np

def recognize_digit(input):
  input = input.reshape((1,28,28))
  
  prediction = np.squeeze(model.predict(input))
  label = [0,1,2,3,4,5,6,7,8,9]
  output = dict(zip(label, prediction.tolist()))
  return output
  
gr.Interface(fn=recognize_digit, inputs="sketchpad", outputs="label").launch(debug = True)

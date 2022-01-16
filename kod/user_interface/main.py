import kivy
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.config import Config 
from kivy.uix.widget import Widget
from kivy.uix.stencilview import StencilView
from kivy.graphics import Line, Rectangle, Color
from kivy.uix.label import Label
from PIL import Image

global width, height
width = 715
height = 500
    
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', width) 
Config.set('graphics', 'height', height)


class DrawInput(StencilView):
    global color
    color = (1,1,1)
    
    def on_touch_down(self, touch):
        with self.canvas:
            Color(*color)
            if not self.collide_point(*touch.pos): return
            touch.ud["line"] = Line(points=(touch.x, touch.y), width=15)
        
    def on_touch_move(self, touch):
        Color(*color)
        if not self.collide_point(*touch.pos): return
        touch.ud["line"].points += (touch.x, touch.y)
  
# creating the App class
class MyApp(App):
  
    def build(self):
        Fl = FloatLayout()
  
        btnPredict = Button(text ='Predict',
                    size_hint =(.3, .2),
                    pos_hint ={'x': 0.7, 'y':0.8 })
        
        btnReset = Button(text = 'Reset',
                          size_hint = (.3, .2),
                          pos_hint ={'x': 0.7, 'y':0.6 })
        self.label = Label( text = 'Label',
                            size_hint = (.3, .6),
                            pos_hint={'x':0.7, 'y':0.2})
        btnReset.bind( on_release=self.clear_canvas)
        btnPredict.bind (on_release=self.prediction)

        self.painter = DrawInput(size = [width*0.7, height],
                                 size_hint = (.7, 1),
                                pos_hint = {'x': 0.0, 'y': 0.0})
        
        with self.painter.canvas:
            Color(0,0,0)
            Rectangle(pos=self.painter.pos, size=self.painter.size)
        
        
        # adding widget i.e button
        Fl.add_widget(btnPredict)
        Fl.add_widget(btnReset)
        Fl.add_widget(self.painter)
        Fl.add_widget(self.label)
        
        # return the layout
        return Fl
    def clear_canvas( self, obj):
        self.painter.canvas.clear()
    
    def prediction (self, event):
        self.painter.export_to_png("draw.png")
        img = Image.open('draw.png')
        resized_img = img.resize((28, 28))
        sample = np.array(resized_img)
        sample = sample.astype('float32')
        sample /= 255.0
        predictions = my_model.predict(sample)
        #sortedPredictions = np.sort(predictions)
        #print(sortedPredictions)
        #self.label.text = str(sorted[0]) + '\n' + str(sorted[1]) + str(sorted[2]) + '\n' + str(sorted[3]) + str(sorted[4])
        #os.remove('draw.png')
  
# run the App
if __name__ == "__main__":
    my_model = tf.keras.models.load_model('final_simple_v1.0/')
    MyApp().run()

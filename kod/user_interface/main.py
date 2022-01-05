import kivy
import os
import tensorflow as tf
import numpy as np
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.config import Config 
from kivy.uix.widget import Widget
from kivy.graphics import Line
from kivy.uix.label import Label
from PIL import Image
    
Config.set('graphics', 'resizable', True) 

model = tf.keras.models.load_model('final_simple_v1.0')

class DrawInput(Widget):
    
    def on_touch_down(self, touch):
        with self.canvas:
           if not self.collide_point(*touch.pos): return
           touch.ud["line"] = Line(points=(touch.x, touch.y), width=10)
        
    def on_touch_move(self, touch):
        if not self.collide_point(*touch.pos): return
        touch.ud["line"].points += (touch.x, touch.y)
  
# creating the App class
class MyApp(App):
  
    def build(self):
        Fl = FloatLayout()
  
        btnPredict = Button(text ='Predict',
                    size_hint =(.3, .2),
                    pos_hint ={'x': 0.7, 'y':0.8 })
        
        btnReset = Button(text = 'Reset', size_hint = (.3, .2), pos_hint ={'x': 0.7, 'y':0.6 })
        self.label = Label( text = 'Label', size_hint = (.3, .6), pos_hint={'x':0.7, 'y':0.2})
        btnReset.bind( on_release=self.clear_canvas)
        btnPredict.bind (on_release=self.prediction)
        
        self.painter = DrawInput(size_hint = (.7, 1), pos_hint = {'x': 0.0, 'y': 0.0})
        
        
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
        predictions = model.predict(resized_img)
        sorted = np.sort(predictions)
        self.label.text = str(sorted[0]) + '\n' + str(sorted[1]) + str(sorted[2]) + '\n' + str(sorted[3]) + str(sorted[4])
        os.remove('draw.png')
  
# run the App
if __name__ == "__main__":
    MyApp().run()
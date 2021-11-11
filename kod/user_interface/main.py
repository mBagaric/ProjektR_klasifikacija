from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.graphics import Line

from kivy.config import Config
Config.set('graphics', 'width', '900')
Config.set('graphics', 'height', '600')

class DrawInput(Widget):
    
    def on_touch_down(self, touch):
        with self.canvas:
            touch.ud["line"] = Line(points=(touch.x, touch.y), width=10)
        
    def on_touch_move(self, touch):
        touch.ud["line"].points += (touch.x, touch.y)


class SimpleKivy4(App):
    
    def build(self):
        #mainLayout = RelativeLayout()
        #self.painter = DrawInput()
        #self.painter.height = 400
        #self.painter.width = 400
        #btnReset = Button( text="reset")
        #btnReset.height = 200
        #btnReset.bind( on_release=self.clear_canvas)
        #mainLayout.add_widget( self.painter)
        #mainLayout.add_widget( btnReset)
        #return mainLayout

        parent = Widget();
        
        btnReset = Button( text="reset")
        btnReset.height = 400
        btnReset.bind( on_release=self.clear_canvas)

        self.painter = DrawInput()
        self.painter.height = 400
        self.painter.width = 400

        parent.add_widget( btnReset)
        parent.add_widget( self.painter)
        return parent

    def clear_canvas( self, obj):
        self.painter.canvas.clear()
SimpleKivy4().run()
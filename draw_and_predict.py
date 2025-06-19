import tkinter as tk
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from tkinter import messagebox


model = tf.keras.models.load_model('saved_model.keras')  # We'll save the model in a sec

# Create a blank canvas
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Draw a Digit")
        self.canvas = tk.Canvas(self, width=280, height=280, bg='white')
        self.canvas.pack()
        self.canvas.bind('<B1-Motion>', self.paint)
        self.button = tk.Button(self, text='Predict', command=self.predict)
        self.button.pack()
        self.image = Image.new("L", (280, 280), 'white')

    def paint(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='black')
        for i in range(x-r, x+r):
            for j in range(y-r, y+r):
                if 0 <= i < 280 and 0 <= j < 280:
                    self.image.putpixel((i, j), 0)

    def predict(self):
        img = self.image.resize((28, 28))
        img = ImageOps.invert(img)
        img = np.array(img) / 255.0
        img = img.reshape(1, 28, 28)
        pred = model.predict(img)
        digit = np.argmax(pred)
        tk.messagebox.showinfo("Prediction", f"The model thinks it's a: {digit}")

app = App()
app.mainloop()

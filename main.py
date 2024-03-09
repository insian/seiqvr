import tkinter as tk
import matplotlib
import random
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from running import node
from PIL import Image,ImageTk
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasAgg
from matplotlib.figure import Figure
import sys

def close_window():
    root.destroy()
    sys.exit()

def win_deleted():
    print("closed")
    close_window()

def your_simulator_model():
# Extract relevant columns from the DataFrame
  cols = node.df.columns.difference(['sus'])
  plt.figure(figsize=(8, 8))
# Plot each column against the index
  for col in cols:
      plt.plot(node.df.index, node.df[col])

# Add legend and show plot
  plt.legend(cols)
  return plt.gcf().canvas

# Function to update the graph (not needed anymore)
# def update_graph(figure):  # Removed since gcf is directly used
#     canvas.draw()
#     figure.canvas.draw()  # Removed since gcf is directly used

# Function to handle the run simulation button click
def run_simulation():
    figure = your_simulator_model()  # Get the figure object from the model
    ag = figure.switch_backends(FigureCanvasAgg)
    ag.draw()
    A = np.asarray(ag.buffer_rgba())
    img=Image.fromarray(A)
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()
    ratio = min(canvas_width / img.width, canvas_height / img.height)
    new_width = int(img.width * ratio)
    new_height = int(img.height * ratio)
    new_image = img.resize((new_width, new_height), Image.ANTIALIAS)
    new_photo = ImageTk.PhotoImage(new_image)
    canvas.create_image(0,0,anchor=tk.NW, image=new_photo)
    canvas.image = new_photo

    # img = Image.fromarray(A)
    # img.show()

# Create the main window
root = tk.Tk()
root.geometry("600x640")
root.title("Simulator Model")

# Create frames for layout
#map_frame = tk.Frame(root, width=400, height=200, bd=1, relief=tk.SUNKEN)
graph_frame = tk.Frame(root, width=600, height=600, bd=1, relief=tk.SUNKEN)
button_frame = tk.Frame(root)

# Place the frames in the window
#map_frame.grid(row=0, column=0, rowspan=2, padx=5, pady=5)
# graph_frame.grid(row=0, column=1, rowspan=2, padx=5, pady=5)
# button_frame.grid(row=2, columnspan=2, padx=5, pady=5)
graph_frame.pack(side="top", padx=5, pady=5)
button_frame.pack(side="bottom", fill="x", padx=5, pady=5)

# Add map image (replace with your image path)
#map_image = tk.PhotoImage(file="D:/final yr project/Our work/Code/maharashtra_map.png")
#map_label = tk.Label(map_frame, image=map_image)
#map_label.pack()

# Create a canvas for the graph
canvas = tk.Canvas(graph_frame,width=600, height=600)
canvas.pack()

# Create the run simulation button
run_button = tk.Button(button_frame, text="Run Simulation", command=run_simulation)
run_button.pack()
root.protocol("WM_DELETE_WINDOW", win_deleted)
# Run the main loop
root.mainloop()

import tkinter as tk
from tkinter import filedialog

def get_path_to_trained_model(initialdir='runs'):

    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(initialdir=initialdir, title='Select trained model')

    return file_path
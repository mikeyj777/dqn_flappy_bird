import tkinter as tk
from tkinter import filedialog

def get_path_to_trained_model():

    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(initialdir='runs', title='Select trained model')

    return file_path
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
import threading
import time
import sys

class GraficUserInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network Training Manager")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
     # Frame dei controlli
        control_frame = tk.Frame(root)
        control_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Selezione optimizer
        tk.Label(control_frame, text="Optimizer").pack()
        self.optimizer_var = tk.StringVar()
        self.optimizer_menu = ttk.Combobox(control_frame, textvariable=self.optimizer_var, state='readonly')
        self.optimizer_menu['values'] = ("SGD", "Adam", "RMSprop")
        self.optimizer_menu.pack()
        self.optimizer_menu.current(0)
        
         # Selezione variazione learning rate
        tk.Label(control_frame, text="Variazione Learning Rate").pack()
        self.lr_variation_var = tk.StringVar()
        self.lr_variation_menu = ttk.Combobox(control_frame, textvariable=self.lr_variation_var, state='readonly')
        self.lr_variation_menu['values'] = ("Esponenziale", "Sinusoide", "Lineare")
        self.lr_variation_menu.pack()
        self.lr_variation_menu.current(0)
        self.lr_variation_menu.bind("<<ComboboxSelected>>", self.update_variation_controls)
        
        self.variation_frame = tk.Frame(control_frame)
        self.variation_frame.pack()
        
        # Pulsanti
        self.start_button = tk.Button(control_frame, text="Start Training", command=self.start_training)
        self.start_button.pack(pady=5)
        
        self.stop_button = tk.Button(control_frame, text="Stop Training", command=self.stop_training, state=tk.DISABLED)
        self.stop_button.pack(pady=5)
        
        # Grafici dinamici
        self.graphs = []
        self.graph_frame = tk.Frame(root)
        self.graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.training = False
        self.epoch = 0
        
    def on_close(self):
        self.stop_training()
        self.root.destroy()
        sys.exit()
      
        
    def start_training(self):
        self.training = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        threading.Thread(target=self.train, daemon=True).start()
    
    def stop_training(self):
        self.training = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
    
    def train(self):
        while self.training:
            time.sleep(1)
            loss = random.uniform(0.1, 1.0)  # Simulazione della loss
            self.update_event_graph("loss", loss)
    
    def update_variation_controls(self, event):
        for widget in self.variation_frame.winfo_children():
            widget.destroy()
        
        variation_type = self.lr_variation_var.get()
        if variation_type == "Esponenziale":
            tk.Label(self.variation_frame, text="Fattore di Decadimento").pack()
            tk.Entry(self.variation_frame).pack()
        elif variation_type == "Sinusoide":
            tk.Label(self.variation_frame, text="Frequenza").pack()
            tk.Entry(self.variation_frame).pack()
        elif variation_type == "Lineare":
            tk.Label(self.variation_frame, text="Pendenza").pack()
            tk.Entry(self.variation_frame).pack()
    
    def add_graph(self, event_name):
        fig, ax = plt.subplots()
        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.graphs.append({"event": event_name, "fig": fig, "ax": ax, "canvas": canvas, "data": []})
    
    def update_event_graph(self, event_name, value):
        for graph in self.graphs:
            if graph["event"] == event_name:
                graph["data"].append(value)
                graph["ax"].clear()
                graph["ax"].plot(graph["data"], marker='o', linestyle='-')
                graph["canvas"].draw()
                
        
if __name__ == "__main__":
    root = tk.Tk()
    app = GraficUserInterface(root)
    app.add_graph("loss")  # Aggiunge un grafico per la loss
    app.add_graph("loss")  # Aggiunge un grafico per la loss
    app.add_graph("loss")  # Aggiunge un grafico per la loss
    root.mainloop()
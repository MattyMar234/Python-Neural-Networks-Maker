import os
import random
import math

# Percorso alla cartella del dataset
dataset_path = "/app/Data/Datasets/munich480/data16"

# Percentuali di divisione
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# File per salvare i nomi delle cartelle
train_file = "train_folders.txt"
val_file = "val_folders.txt"
test_file = "test_folders.txt"

# Ottieni una lista delle cartelle nel dataset
folders = [folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]
folders.sort()  # Ordina le cartelle in modo che la divisione sia sempre la stessa
random.shuffle(folders)  # Mischia le cartelle per una divisione casuale

# Calcola il numero di cartelle per ogni set
num_total = len(folders)
num_train = math.floor(num_total * train_ratio)
num_val = math.floor(num_total * val_ratio)
num_test = num_total - num_train - num_val  # Assicura che la somma sia il totale

# Divide le cartelle
train_folders = folders[:num_train]
val_folders = folders[num_train:num_train + num_val]
test_folders = folders[num_train + num_val:]

# Funzione per salvare la lista delle cartelle in un file
def save_to_file(folders, filename):
    with open(filename, "w") as f:
        for folder in folders:
            f.write(f"{folder}\n")

# Salva i nomi delle cartelle nei rispettivi file
save_to_file(train_folders, train_file)
save_to_file(val_folders, val_file)
save_to_file(test_folders, test_file)

print(f"Cartelle di training salvate in '{train_file}'")
print(f"Cartelle di validation salvate in '{val_file}'")
print(f"Cartelle di test salvate in '{test_file}'")
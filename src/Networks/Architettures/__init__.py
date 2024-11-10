# src/Networks/Architetture/__init__.py
import pkgutil
import inspect
import importlib
from typing import Final
import pytorch_lightning as pl
from pathlib import Path

# Trova il percorso del pacchetto 'Architetture'
_package_path = Path(__file__).parent
_package_name = 'Networks.Architettures'  # Nome completo del pacchetto
AvailableArchitetture: dict | None = None


def _find_subclasses(package_path, base_class, package_name):
    subclasses_dict = {}  # Dizionario per memorizzare il nome della classe e il riferimento

    if isinstance(package_path, Path):
        package_path = [str(package_path)]  # Converti Path in lista di stringhe
    
    for loader, module_name, is_pkg in pkgutil.walk_packages(package_path, package_name + "."):
        if module_name == package_name:
            continue
        
        try:
            module = importlib.import_module(module_name)
            
            # Per ogni classe nel modulo, controlliamo se è definita nel modulo stesso
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Verifica se la classe è una sottoclasse diretta di base_class
                if issubclass(obj, base_class) and obj is not base_class:
                    # Verifica se la classe è definita nel modulo corrente
                    if obj.__module__ == module_name:
                        subclasses_dict[obj.__name__] = obj  # Aggiungi la classe al dizionario
        except Exception as e:
            print(f"Error importing {module_name}: {e}")
    
    return subclasses_dict

AvailableArchitetture: Final[dict] = _find_subclasses(_package_path, pl.LightningModule, _package_name)

# Funzione per creare un'istanza di una classe trovata
def create_instance(class_name, **params):
    # Trova tutte le sottoclassi di LightningModule come un dizionario
    
    
    # Trova la classe con il nome specificato nel dizionario
    if class_name in AvailableArchitetture:
        # Crea l'istanza della classe con i parametri forniti
        class_ref = AvailableArchitetture[class_name]
        instance = class_ref(**params)  # Passa i parametri come keyword arguments
        return instance
    else:
        print(f"Class {class_name} not found.")
        return None



    



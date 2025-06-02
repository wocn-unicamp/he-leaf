import random
import warnings
from phe import paillier
import numpy as np
import json


class Client:
    
    def __init__(self, client_id, group=None, train_data={'x' : [],'y' : []}, eval_data={'x' : [],'y' : []}, model=None):
        self._model = model
        self.id = client_id
        self.group = group
        self.train_data = train_data
        self.eval_data = eval_data


    def train(self, num_epochs=1, batch_size=10, minibatch=None):
        """Trains on self.model using the client's train_data.

        Args:
            num_epochs: Number of epochs to train. Unsupported if minibatch is provided (minibatch has only 1 epoch)
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
        Return:
            comp: number of FLOPs executed in training process
            num_samples: number of samples used in training
            update: set of weights
            update_size: number of bytes in update
        """
        if minibatch is None:
            data = self.train_data
            comp, update = self.model.train(data, num_epochs, batch_size)
        else:
            frac = min(1.0, minibatch)
            num_data = max(1, int(frac*len(self.train_data["x"])))
            xs, ys = zip(*random.sample(list(zip(self.train_data["x"], self.train_data["y"])), num_data))
            data = {'x': xs, 'y': ys}

            # Minibatch trains for only 1 epoch - multiple local epochs don't make sense!
            num_epochs = 1
            comp, update = self.model.train(data, num_epochs, num_data)
        num_train_samples = len(data['y'])
        

        # --- CÁLCULO DEL PORCENTAJE DE PESOS DE LA ÚLTIMA CAPA ---
        total_pesos = 0
        pesos_ultima_capa = 0

        for idx, arr in enumerate(update):
            arr_np = np.array(arr)
            n_pesos = arr_np.size
            # print(f"Capa {idx}: {n_pesos} pesos, shape: {arr_np.shape}")
            total_pesos += n_pesos
            if idx == len(update) - 1:
                pesos_ultima_capa = n_pesos

        porcentaje = 100 * pesos_ultima_capa / total_pesos if total_pesos > 0 else 0

        # print(f"Total pesos: {total_pesos}")
        # print(f"Pesos última capa: {pesos_ultima_capa}")
        # print(f"La última capa representa {porcentaje:.6f}% de todos los parámetros del modelo.")

        # --- CIFRADO DE UPDATE ---

        # Leer la llave pública Paillier desde archivo (formato JSON)
        public_key_name = 'public_key.json'  # Cambia la ruta si es necesario
        with open(public_key_name, 'r') as f:
            pub_data = json.load(f)
        public_key = paillier.PaillierPublicKey(pub_data['n'])

        # Definir un factor de escala para convertir floats a enteros (Paillier cifra solo enteros)
        scale = 1e6  # Aumenta o reduce para ajustar la precisión según lo que necesites

        # Prepara las listas para almacenar los parámetros y las formas originales
        enc_update = []  # Aquí se guardarán los parámetros (solo la última capa cifrada)
        shapes = []      # Aquí se guardarán las formas originales de cada capa
        num_layers = len(update)  # Número total de capas del modelo
        # print("Número de capas del modelo:", num_layers)
        # print("Tamano ultima capa:", len(update[-1]))
        # print("valor ultima capa:", update[-1])

        # Recorrer todas las capas del modelo
        for idx, arr in enumerate(update):
            arr_np = np.array(arr)     # Convierte a array numpy
            shapes.append(arr_np.shape) # Guarda la forma original
            if idx == num_layers - 1:  # Si es la última capa...
                flat_arr = arr_np.flatten()  # Aplana el array
                # Cifra cada elemento: multiplica por scale, convierte a int, y cifra
                enc_arr = [public_key.encrypt(int(x * scale)) for x in flat_arr]
                enc_update.append(enc_arr)   # Guarda la capa cifrada
            else:
                # Para el resto de capas, simplemente aplana y guarda los pesos en claro
                enc_update.append(arr_np.flatten().tolist())

        ## Sumar operaciones de cifrado
        # Guardar el valor original
        flops_entrenamiento = comp
        # Ponderar el costo del cifrado (cada cifrado Paillier cuenta como 1000 FLOPs):
        ops_cifrado = pesos_ultima_capa * 1000
        comp += ops_cifrado

        print(f"FLOPs entrenamiento: {flops_entrenamiento}, operaciones de cifrado: {ops_cifrado}, total: {comp}")




        # Devuelve los resultados:
        #   - comp: operaciones realizadas,
        #   - num_train_samples: muestras usadas,
        #   - (enc_update, shapes): parámetros del modelo (última capa cifrada) y sus formas.
        return comp, num_train_samples, (enc_update, shapes)
           
        #return comp, num_train_samples, update

    def test(self, set_to_use='test'):
        """Tests self.model on self.test_data.
        
        Args:
            set_to_use. Set to test on. Should be in ['train', 'test'].
        Return:
            dict of metrics returned by the model.
        """
        assert set_to_use in ['train', 'test', 'val']
        if set_to_use == 'train':
            data = self.train_data
        elif set_to_use == 'test' or set_to_use == 'val':
            data = self.eval_data
        return self.model.test(data)

    @property
    def num_test_samples(self):
        """Number of test samples for this client.

        Return:
            int: Number of test samples for this client
        """
        if self.eval_data is None:
            return 0
        return len(self.eval_data['y'])

    @property
    def num_train_samples(self):
        """Number of train samples for this client.

        Return:
            int: Number of train samples for this client
        """
        if self.train_data is None:
            return 0
        return len(self.train_data['y'])

    @property
    def num_samples(self):
        """Number samples for this client.

        Return:
            int: Number of samples for this client
        """
        train_size = 0
        if self.train_data is not None:
            train_size = len(self.train_data['y'])

        test_size = 0 
        if self.eval_data is not  None:
            test_size = len(self.eval_data['y'])
        return train_size + test_size

    @property
    def model(self):
        """Returns this client reference to model being trained"""
        return self._model

    @model.setter
    def model(self, model):
        warnings.warn('The current implementation shares the model among all clients.'
                      'Setting it on one client will effectively modify all clients.')
        self._model = model

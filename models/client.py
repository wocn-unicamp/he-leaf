import random
import warnings
from phe import paillier
import numpy as np
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed


def cifrar_valor(val, public_key_n, scale):
    public_key = paillier.PaillierPublicKey(public_key_n)
    return public_key.encrypt(int(val * scale))


def calcular_sensibilidad_param(layer_idx, idx, param_shape, parametros_originales, X_sample, y_sample, epsilon, model):
    param_flat = parametros_originales[layer_idx].flatten()
    valor_original = param_flat[idx]

    param_flat[idx] = valor_original + epsilon
    parametros_modificados_plus = [p.copy() for p in parametros_originales]
    parametros_modificados_plus[layer_idx] = param_flat.reshape(param_shape)
    model.set_params(parametros_modificados_plus)
    loss_plus = model.evaluate_loss(X_sample, y_sample)

    param_flat[idx] = valor_original - epsilon
    parametros_modificados_minus = [p.copy() for p in parametros_originales]
    parametros_modificados_minus[layer_idx] = param_flat.reshape(param_shape)
    model.set_params(parametros_modificados_minus)
    loss_minus = model.evaluate_loss(X_sample, y_sample)

    grad_wm = (loss_plus - loss_minus) / (2 * epsilon)

    deriv_yk = np.zeros(len(y_sample))
    for k in range(len(y_sample)):
        yk_original = y_sample[k]

        y_sample[k] = yk_original + epsilon
        loss_plus_yk = model.evaluate_loss(X_sample, y_sample)

        y_sample[k] = yk_original - epsilon
        loss_minus_yk = model.evaluate_loss(X_sample, y_sample)

        y_sample[k] = yk_original

        grad_yk = (loss_plus_yk - loss_minus_yk) / (2 * epsilon)
        deriv_yk[k] = (grad_wm - grad_yk) / epsilon

    sensibilidad = np.mean(np.abs(deriv_yk))
    param_flat[idx] = valor_original  # restaurar valor original

    return idx, sensibilidad




class Client:
    
    def __init__(self, client_id, group=None, train_data={'x' : [],'y' : []}, eval_data={'x' : [],'y' : []}, model=None):
        self._model = model
        self.id = client_id
        self.group = group
        self.train_data = train_data
        self.eval_data = eval_data

    def cargar_mascara(ruta_mascara='mascara.json'):
        try:
            with open(ruta_mascara, 'r') as f:
                mascara = json.load(f)
            return np.array(mascara, dtype=bool)
        except Exception as e:
            print(f"Error al cargar la máscara: {e}")
            return None

    def train_without_he(self, num_epochs=1, batch_size=10, minibatch=None):
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
        
        return comp, num_train_samples, update


    def train_lastLayer_he(self, num_epochs=1, batch_size=10, minibatch=None):
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
            print(f"Capa {idx}: {n_pesos} pesos, shape: {arr_np.shape}")
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
        scale = 1e3  # Aumenta o reduce para ajustar la precisión según lo que necesites

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

        # print(f"FLOPs entrenamiento: {flops_entrenamiento}, operaciones de cifrado: {ops_cifrado}, total: {comp}")

        # Devuelve los resultados:
        #   - comp: operaciones realizadas,
        #   - num_train_samples: muestras usadas,
        #   - (enc_update, shapes): parámetros del modelo (última capa cifrada) y sus formas.
        return comp, num_train_samples, (enc_update, shapes)
    
    # Entrenamiento con cifrado serial
    # Utiliza un único hilo para cifrar los parámetros del modelo según una máscara
    def train_serial(self, num_epochs=1, batch_size=10, minibatch=None):
        print("="*40)
        print(f"Cliente {self.id} entrenando con {self.num_train_samples} muestras.")

        if minibatch is None:
            data = self.train_data
            comp, update = self.model.train(data, num_epochs, batch_size)
        else:
            frac = min(1.0, minibatch)
            num_data = max(1, int(frac*len(self.train_data["x"])))
            xs, ys = zip(*random.sample(list(zip(self.train_data["x"], self.train_data["y"])), num_data))
            data = {'x': xs, 'y': ys}
            num_epochs = 1
            comp, update = self.model.train(data, num_epochs, num_data)
        
        num_train_samples = len(data['y'])

        # --- CIFRADO SEGÚN MÁSCARA ---
        print("Cargando máscara de parámetros a cifrar...")
        with open('mascara.json', 'r') as f:
            mascara = np.array(json.load(f), dtype=bool)

        print("Aplanando parámetros del modelo...")
        flat_update = np.concatenate([np.array(arr).flatten() for arr in update])
        shapes = [np.array(arr).shape for arr in update]

        if mascara.size != flat_update.size:
            raise ValueError(f"Tamaño de la máscara ({mascara.size}) y los parámetros ({flat_update.size}) no coincide.")

        print("Cargando llave pública Paillier...")
        with open('public_key.json', 'r') as f:
            pub_data = json.load(f)
        public_key = paillier.PaillierPublicKey(pub_data['n'])
        scale = 1e3

        # --- CIFRADO SECUENCIAL (SINGLE THREAD) CON BARRA DE PROGRESO ---
        indices_a_cifrar = [i for i, cifrar in enumerate(mascara) if cifrar]
        print(f"Iniciando cifrado SECUENCIAL de {len(indices_a_cifrar)} parámetros (de {len(flat_update)})...")

        start_time = time.time()
        enc_update = flat_update.copy().tolist()
        ops_cifrado = 0

        for i in tqdm(indices_a_cifrar, desc="Cifrando (HE secuencial)", unit="param"):
            enc_update[i] = public_key.encrypt(int(flat_update[i] * scale))
            ops_cifrado += 1000

        elapsed_time = time.time() - start_time
        print(f"Tiempo de cifrado SECUENCIAL: {elapsed_time:.2f} segundos.")
        print("="*40)

        comp += ops_cifrado
        return comp, num_train_samples, (enc_update, shapes)


    # Entrenamiento con cifrado paralelo optimizado com mascara
    # Utiliza múltiples procesos para cifrar los parámetros del modelo según una máscara
    def train(self, num_epochs=1, batch_size=10, minibatch=None):
        print("="*40)
        print(f"Cliente {self.id} entrenando con {self.num_train_samples} muestras.")

        if minibatch is None:
            data = self.train_data
            comp, update = self.model.train(data, num_epochs, batch_size)
        else:
            frac = min(1.0, minibatch)
            num_data = max(1, int(frac*len(self.train_data["x"])))
            xs, ys = zip(*random.sample(list(zip(self.train_data["x"], self.train_data["y"])), num_data))
            data = {'x': xs, 'y': ys}
            num_epochs = 1
            comp, update = self.model.train(data, num_epochs, num_data)

        num_train_samples = len(data['y'])

        # Carga máscara
        print("Cargando máscara...")
        with open('mascara.json', 'r') as f:
            mascara = np.array(json.load(f), dtype=bool)

        # Aplanar parámetros
        print("Aplanando parámetros...")
        flat_update = np.concatenate([np.array(arr).flatten() for arr in update])
        shapes = [np.array(arr).shape for arr in update]

        if mascara.size != flat_update.size:
            raise ValueError("Tamaño máscara y parámetros no coinciden.")

        # Carga llave pública
        print("Cargando llave pública...")
        with open('public_key.json', 'r') as f:
            pub_data = json.load(f)
        public_key_n = pub_data['n']
        scale = 1e3

        # Cifrado paralelo real (con procesos)
        print(f"Iniciando cifrado paralelo de {np.sum(mascara)} parámetros con procesos...")
        start_time = time.time()

        enc_update = flat_update.copy().tolist()
        indices_a_cifrar = [i for i, cifrar in enumerate(mascara) if cifrar]

        ops_cifrado = 0

        # Máximo uso de CPU (todos los núcleos disponibles)
        max_workers = multiprocessing.cpu_count()

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(cifrar_valor, flat_update[i], public_key_n, scale): i 
                for i in indices_a_cifrar
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Cifrando (HE)", unit="param"):
                idx = futures[future]
                enc_update[idx] = future.result()
                ops_cifrado += 1000

        elapsed_time = time.time() - start_time
        print(f"Tiempo cifrado paralelo optimizado: {elapsed_time:.2f} segundos.")
        print("="*40)

        comp += ops_cifrado
        return comp, num_train_samples, (enc_update, shapes)

    # **** Mapa de sensibilidad ****
    # Calcula la sensibilidad de cada parámetro del modelo respecto a los datos de entrenamiento.
    # Utiliza derivadas numéricas para estimar cómo cambia la pérdida al modificar cada parámetro y etiqueta.
    # El resultado es un array de sensibilidad por parámetro, que luego se puede usar para construir una máscara de cifrado en el servidor.
    # Este método es útil para identificar qué parámetros son más sensibles a los cambios en los datos,
    # lo que puede ayudar a optimizar el cifrado y la privacidad en el entrenamiento federado.    # Derivadas numéricas:
    # Usa pequeñas perturbaciones (epsilon) para estimar las derivadas necesarias.
    # Sensibilidad:
    # Calcula la segunda derivada cruzada respecto a cada parámetro y etiqueta del dato, luego toma el valor absoluto y promedia.
    # Salida:
    # Retorna un array plano de sensibilidad por parámetro, que luego podrás usar para construir tu máscara de cifrado en el servidor.  



    def calcular_mapa_sensibilidad(self, sample_size=10, param_subsample_rate=0.1):
        X = np.array(self.train_data['x'])
        y_true = np.array(self.train_data['y'])
        num_samples = X.shape[0]

        if num_samples > sample_size:
            sample_indices = np.random.choice(num_samples, size=sample_size, replace=False)
            X_sample = X[sample_indices]
            y_sample = y_true[sample_indices]
        else:
            X_sample = X
            y_sample = y_true

        parametros_originales = self.model.get_params()
        sensibilidad = []
        epsilon = 1e-5

        print(f"Calculando mapa de sensibilidad con {len(y_sample)} ejemplos...")

        for layer_idx, param in enumerate(parametros_originales):
            param_shape = param.shape
            param_flat = param.flatten()

            sensibilidad_param = np.zeros(param_flat.shape)
            args = [(layer_idx, idx, param_shape, parametros_originales, X_sample, y_sample.copy(), epsilon, self.model)
                    for idx in range(len(param_flat))]

            with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                results = list(tqdm(executor.map(lambda p: calcular_sensibilidad_param(*p), args),
                                    total=len(param_flat),
                                    desc=f"Sensibilidad capa {layer_idx}"))

            for idx_result, sensibilidad_result in results:
                sensibilidad_param[idx_result] = sensibilidad_result

            sensibilidad.extend(sensibilidad_param.tolist())

        self.model.set_params(parametros_originales)
        print("Mapa de sensibilidad calculado.")

        return np.array(sensibilidad)



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

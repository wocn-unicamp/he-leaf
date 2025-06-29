import numpy as np
from phe import paillier
import json
from baseline_constants import BYTES_WRITTEN_KEY, BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY

class Server:
    
    def __init__(self, client_model):
        self.client_model = client_model
        self.model = client_model.get_params()
        self.selected_clients = []
        self.updates = []
    


    def select_clients(self, my_round, possible_clients, num_clients=20):
        """Selects num_clients clients randomly from possible_clients.
        
        Note that within function, num_clients is set to
            min(num_clients, len(possible_clients)).

        Args:
            possible_clients: Clients from which the server can select.
            num_clients: Number of clients to select; default 20
        Return:
            list of (num_train_samples, num_test_samples)
        """
        num_clients = min(num_clients, len(possible_clients))
        np.random.seed(my_round)
        self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)

        return [(c.num_train_samples, c.num_test_samples) for c in self.selected_clients]



    def generar_mascara(self, porcentaje=1, ruta_archivo='mascara.json'):

        model_params = self.model
        total_params = sum(param.size for param in model_params)
        num_params_cifrados = int(total_params * porcentaje / 100)

        mascara = np.zeros(total_params, dtype=bool)
        mascara[:num_params_cifrados] = True
        np.random.shuffle(mascara)

        try:
            with open(ruta_archivo, 'w') as f:
                json.dump(mascara.tolist(), f)
            # (Opcional) guarda la máscara como atributo para futuro uso
            self.mascara = mascara
            return True
        except Exception as e:
            print(f"Error al guardar la máscara: {e}")
            return False



    def train_model(self, num_epochs=1, batch_size=10, minibatch=None, clients=None):
        """Trains self.model on given clients.
        
        Trains model on self.selected_clients if clients=None;
        each client's data is trained with the given number of epochs
        and batches.

        Args:
            clients: list of Client objects.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
        Return:
            bytes_written: number of bytes written by each client to server 
                dictionary with client ids as keys and integer values.
            client computations: number of FLOPs computed by each client
                dictionary with client ids as keys and integer values.
            bytes_read: number of bytes read by each client from server
                dictionary with client ids as keys and integer values.
        """
        if clients is None:
            clients = self.selected_clients
        sys_metrics = {
            c.id: {BYTES_WRITTEN_KEY: 0,
                   BYTES_READ_KEY: 0,
                   LOCAL_COMPUTATIONS_KEY: 0} for c in clients}
        for c in clients:
            c.model.set_params(self.model)
            comp, num_samples, update = c.train(num_epochs, batch_size, minibatch)

            sys_metrics[c.id][BYTES_READ_KEY] += c.model.size
            sys_metrics[c.id][BYTES_WRITTEN_KEY] += c.model.size
            sys_metrics[c.id][LOCAL_COMPUTATIONS_KEY] = comp

            self.updates.append((num_samples, update))

        return sys_metrics

    def update_model_without_he(self):
        total_weight = 0.
        base = [0] * len(self.updates[0][1])
        for (client_samples, client_model) in self.updates:
            total_weight += client_samples
            for i, v in enumerate(client_model):
                base[i] += (client_samples * v.astype(np.float64))
        averaged_soln = [v / total_weight for v in base]
       
        self.model = averaged_soln
        self.updates = []

    def update_model_he_last_layer(self):

         # ----- Cargar la llave privada Paillier -----
        private_key_name = 'private_key.json'
        with open(private_key_name, 'r') as f:
            priv_data = json.load(f)
        public_key = paillier.PaillierPublicKey(priv_data['public_key_n'])
        private_key = paillier.PaillierPrivateKey(public_key, priv_data['p'], priv_data['q'])
        scale = 1e3  # Debe ser el mismo usado en el cliente

        total_weight = 0.
        # `base` será una lista de arrays para acumular la suma ponderada de cada capa
        # Toma la estructura del primer update para inicializar base
        _, (first_update, shapes) = self.updates[0]
        base = []
        for idx, arr in enumerate(first_update):
            # Si la capa es cifrada, crea array de ceros del tamaño adecuado
            if idx == len(first_update) - 1:
                base.append(np.zeros(len(arr)))
            else:
                base.append(np.zeros(len(arr)))

        # ----------- Agregación ponderada cliente por cliente -----------
        for (client_samples, (client_model, shapes)) in self.updates:
            total_weight += client_samples
            for idx, arr in enumerate(client_model):
                if idx == len(client_model) - 1:
                    # Es la última capa, hay que descifrar antes de sumar
                    # Descifra cada peso y convierte a float
                    dec_layer = [private_key.decrypt(w) / scale for w in arr]
                    base[idx] += client_samples * np.array(dec_layer, dtype=np.float64)
                else:
                    # Capas en claro, solo suma
                    base[idx] += client_samples * np.array(arr, dtype=np.float64)

        # --------- Promediar y reconstruir las capas con su forma original -----------
        averaged_soln = []
        for idx, arr in enumerate(base):
            arr_avg = arr / total_weight
            shape = shapes[idx]
            arr_avg = arr_avg.reshape(shape)
            averaged_soln.append(arr_avg)

        # Actualiza el modelo global con los nuevos pesos promediados
        self.model = averaged_soln
        self.updates = []

    def update_model(self):
        #Imprime que comienza la actualización del modelo con HE y el número de clientes
        print('>> Comenzando la actualización del modelo con HE, número de clientes:', len(self.updates))


        # ----- Cargar la llave privada Paillier -----
        private_key_name = 'private_key.json'
        with open(private_key_name, 'r') as f:
            priv_data = json.load(f)
        public_key = paillier.PaillierPublicKey(priv_data['public_key_n'])
        private_key = paillier.PaillierPrivateKey(public_key, priv_data['p'], priv_data['q'])
        scale = 1e2  # Debe ser el mismo usado en el cliente

        # ----- Cargar la máscara -----
        with open('mascara.json', 'r') as f:
            mascara = np.array(json.load(f), dtype=bool)

        # ----- Obtener shapes y cantidad de parámetros -----
        _, (first_update, shapes) = self.updates[0]
        total_params = mascara.size

        # Inicializa la suma ponderada de cada parámetro (vector plano)
        base = np.zeros(total_params, dtype=np.float64)
        total_weight = 0.

        # ----------- Agregación ponderada cliente por cliente -----------
        for (client_samples, (client_model, shapes)) in self.updates:
            total_weight += client_samples
            flat_update = np.array(client_model, dtype=object)  # puede haber objetos Paillier y floats
            suma_cliente = np.zeros(total_params, dtype=np.float64)

            # Descifra solo donde mascara==True
            for i in range(total_params):
                if mascara[i]:
                    suma_cliente[i] = private_key.decrypt(flat_update[i]) / scale
                else:
                    suma_cliente[i] = float(flat_update[i])
            base += client_samples * suma_cliente

        # Promediar
        base /= total_weight

        # --------- Reconstruir las capas con sus shapes -----------
        averaged_soln = []
        idx = 0
        for shape in shapes:
            size = np.prod(shape)
            arr = base[idx:idx+size].reshape(shape)
            averaged_soln.append(arr)
            idx += size

        self.model = averaged_soln
        self.updates = []


    def test_model(self, clients_to_test, set_to_use='test'):
        """Tests self.model on given clients.

        Tests model on self.selected_clients if clients_to_test=None.

        Args:
            clients_to_test: list of Client objects.
            set_to_use: dataset to test on. Should be in ['train', 'test'].
        """
        metrics = {}

        if clients_to_test is None:
            clients_to_test = self.selected_clients

        for client in clients_to_test:
            client.model.set_params(self.model)
            c_metrics = client.test(set_to_use)
            metrics[client.id] = c_metrics
        
        return metrics

    def get_clients_info(self, clients):
        """Returns the ids, hierarchies and num_samples for the given clients.

        Returns info about self.selected_clients if clients=None;

        Args:
            clients: list of Client objects.
        """
        if clients is None:
            clients = self.selected_clients

        ids = [c.id for c in clients]
        groups = {c.id: c.group for c in clients}
        num_samples = {c.id: c.num_samples for c in clients}
        return ids, groups, num_samples

    def save_model(self, path):
        """Saves the server model on checkpoints/dataset/model.ckpt."""
        # Save server model
        self.client_model.set_params(self.model)
        model_sess =  self.client_model.sess
        return self.client_model.saver.save(model_sess, path)

    def close_model(self):
        self.client_model.close()


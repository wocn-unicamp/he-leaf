"""Script to run the baselines."""
import argparse
import importlib
import numpy as np
import os
import sys
import random
import tensorflow as tf
import metrics.writer as metrics_writer

from baseline_constants import MAIN_PARAMS, MODEL_PARAMS
from client import Client
from server import Server
from model import ServerModel

from utils.args import parse_args
from utils.model_utils import read_data

from tqdm import tqdm  # Importa tqdm para mostrar la barra de progreso

STAT_METRICS_PATH = 'metrics/stat_metrics.csv'
SYS_METRICS_PATH = 'metrics/sys_metrics.csv'



def main():

    args = parse_args()

    # Set the random seed if provided (affects client sampling, and batching)
    # Semilla para el generador de números aleatorios estándar de Python
    random.seed(1 + args.seed)

    # Semilla para el generador de números aleatorios de NumPy
    np.random.seed(12 + args.seed)

    # Semilla para el generador de números aleatorios de TensorFlow (TF 1.x)
    tf.set_random_seed(123 + args.seed)


    # Verifica que el archivo del modelo existe; si no, muestra un error y detiene el programa.
    model_path = '%s/%s.py' % (args.dataset, args.model)

    # Verifica que el archivo del modelo existe; si no, muestra un error y detiene el programa.
    if not os.path.exists(model_path):
        print('ERROR: Por favor, especifica un dataset y un modelo válidos. Archivo no encontrado:', model_path)
        exit(1)  # Detiene la ejecución si el archivo no existe

    # Para la importación dinámica, convierte la ruta al formato de módulo Python.
    # Ejemplo: 'femnist/cnn.py' -> 'femnist.cnn'
    model_path = '%s.%s' % (args.dataset, args.model)
    
    # -----------------------------------------------------------------------------
    # Importa dinámicamente el módulo del modelo y obtiene la clase ClientModel.
    # -----------------------------------------------------------------------------

    # Imprime un encabezado indicando el modelo que se está utilizando.
    print('############################## %s ##############################' % model_path)

    # Almacena el valor de HE en una variable para uso posterior.
    he_percentage = args.he
    print('>> Porcentaje de parámetros cifrados (HE): %.2f' % he_percentage)  # Mostrando el valor de HE

    # Importa dinámicamente el módulo del modelo, por ejemplo: import femnist.cnn
    mod = importlib.import_module(model_path)

    # Obtiene la clase ClientModel del módulo importado.
    ClientModel = getattr(mod, 'ClientModel')

    # -----------------------------------------------------------------------------
    # Obtiene los parámetros principales de la simulación desde MAIN_PARAMS,
    # usando el dataset y el tiempo de simulación seleccionados.
    # Si el usuario especifica un valor por línea de comando, lo respeta; de lo contrario, usa el valor por defecto.
    # -----------------------------------------------------------------------------
    tup = MAIN_PARAMS[args.dataset][args.t]  # (num_rounds, eval_every, clients_per_round)

    # Número total de rondas de entrenamiento federado
    num_rounds = args.num_rounds if args.num_rounds != -1 else tup[0]

    # Frecuencia de evaluación (cada cuántas rondas se evalúa el modelo)
    eval_every = args.eval_every if args.eval_every != -1 else tup[1]

    # Número de clientes seleccionados por ronda
    clients_per_round = args.clients_per_round if args.clients_per_round != -1 else tup[2]

    # Suppress tf warnings
    tf.logging.set_verbosity(tf.logging.WARN)

    # Create 2 models
    model_params = MODEL_PARAMS[model_path]
    if args.lr != -1:
        model_params_list = list(model_params)
        model_params_list[0] = args.lr
        model_params = tuple(model_params_list)

    # Create client model, and share params with server model
    tf.reset_default_graph()
    client_model = ClientModel(args.seed, *model_params)

    # Create server
    server = Server(client_model)

    # Create clients
    clients = setup_clients(args.dataset, client_model, args.use_val_set)
    client_ids, client_groups, client_num_samples = server.get_clients_info(clients)
    print('Clients in Total: %d' % len(clients))


    # Initial status
    print('--- Random Initialization ---')
    stat_writer_fn = get_stat_writer_function(client_ids, client_groups, client_num_samples, args)
    sys_writer_fn = get_sys_writer_function(args)
    print_stats(0, server, clients, client_num_samples, args, stat_writer_fn, args.use_val_set)

    # Generate the mask for HE
    exito = server.generar_mascara(he_percentage)


    # Simulate training
    for i in range(num_rounds):
        print('--- Round %d of %d: Training %d Clients ---' % (i + 1, num_rounds, clients_per_round))

        # Select clients to train this round
        server.select_clients(i, online(clients), num_clients=clients_per_round)
        c_ids, c_groups, c_num_samples = server.get_clients_info(server.selected_clients)


        # Paso 1: Cada cliente seleccionado calcula su mapa de sensibilidad
        print(f'--- Calculando mapas de sensibilidad para los {clients_per_round} clientes seleccionados en la ronda {i+1} ---')
        sens_maps = []
        for client in tqdm(server.selected_clients, desc=f"Mapas de sensibilidad (round {i+1})"):
            # sample_size: número de ejemplos aleatorios locales usados para estimar la sensibilidad.
            #              Valores típicos: 5-20. Menor = más rápido, pero menos preciso.
            # param_subsample_rate: fracción de los parámetros del modelo a considerar (entre 0 y 1).
            #                       Por ejemplo, 0.1 significa que solo se calcula la sensibilidad para el 10% de los parámetros.
            #                       Menor = más rápido, pero menos preciso.
            max_sample_size = len(client.train_data['y'])
            sens_map = client.calcular_mapa_sensibilidad(
                sample_size=1,            # Usa 10 ejemplos de entrenamiento locales para el cálculo de sensibilidad,  max=max_sample_size
                param_subsample_rate=1   # Calcula la sensibilidad solo para el 10% de los parámetros del modelo, max = 1.0
            )
            sens_maps.append(sens_map)



        # Simulate server model training on selected clients' data
        sys_metrics = server.train_model(num_epochs=args.num_epochs, batch_size=args.batch_size, minibatch=args.minibatch)
        sys_writer_fn(i + 1, c_ids, sys_metrics, c_groups, c_num_samples)
        
        # Update server model
        server.update_model()

        # Test model
        if (i + 1) % eval_every == 0 or (i + 1) == num_rounds:
            print_stats(i + 1, server, clients, client_num_samples, args, stat_writer_fn, args.use_val_set)
    
    # Save server model
    ckpt_path = os.path.join('checkpoints', args.dataset)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    save_path = server.save_model(os.path.join(ckpt_path, '{}.ckpt'.format(args.model)))
    print('Model saved in path: %s' % save_path)

    # Close models
    server.close_model()

def online(clients):
    """We assume all users are always online."""
    return clients


def create_clients(users, groups, train_data, test_data, model):
    if len(groups) == 0:
        groups = [[] for _ in users]
    clients = [Client(u, g, train_data[u], test_data[u], model) for u, g in zip(users, groups)]
    return clients


def setup_clients(dataset, model=None, use_val_set=False):
    """Instantiates clients based on given train and test data directories.

    Return:
        all_clients: list of Client objects.
    """
    eval_set = 'test' if not use_val_set else 'val'
    train_data_dir = os.path.join('..', 'data', dataset, 'data', 'train')
    test_data_dir = os.path.join('..', 'data', dataset, 'data', eval_set)

    users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)

    clients = create_clients(users, groups, train_data, test_data, model)

    return clients


def get_stat_writer_function(ids, groups, num_samples, args):

    def writer_fn(num_round, metrics, partition):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples, partition, args.metrics_dir, '{}_{}'.format(args.metrics_name, 'stat'))

    return writer_fn


def get_sys_writer_function(args):

    def writer_fn(num_round, ids, metrics, groups, num_samples):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples, 'train', args.metrics_dir, '{}_{}'.format(args.metrics_name, 'sys'))

    return writer_fn


def print_stats(
    num_round, server, clients, num_samples, args, writer, use_val_set):
    
    train_stat_metrics = server.test_model(clients, set_to_use='train')
    print_metrics(train_stat_metrics, num_samples, prefix='train_')
    writer(num_round, train_stat_metrics, 'train')

    eval_set = 'test' if not use_val_set else 'val'
    test_stat_metrics = server.test_model(clients, set_to_use=eval_set)
    print_metrics(test_stat_metrics, num_samples, prefix='{}_'.format(eval_set))
    writer(num_round, test_stat_metrics, eval_set)


def print_metrics(metrics, weights, prefix=''):
    """Prints weighted averages of the given metrics.

    Args:
        metrics: dict with client ids as keys. Each entry is a dict
            with the metrics of that client.
        weights: dict with client ids as keys. Each entry is the weight
            for that client.
    """
    ordered_weights = [weights[c] for c in sorted(weights)]
    metric_names = metrics_writer.get_metrics_names(metrics)
    to_ret = None
    for metric in metric_names:
        ordered_metric = [metrics[c][metric] for c in sorted(metrics)]
        print('%s: %g, 10th percentile: %g, 50th percentile: %g, 90th percentile %g' \
              % (prefix + metric,
                 np.average(ordered_metric, weights=ordered_weights),
                 np.percentile(ordered_metric, 10),
                 np.percentile(ordered_metric, 50),
                 np.percentile(ordered_metric, 90)))


if __name__ == '__main__':
    main()

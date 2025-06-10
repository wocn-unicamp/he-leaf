import argparse
import importlib
import numpy as np
import os
import random
import tensorflow as tf
import metrics.writer as metrics_writer
from baseline_constants import MAIN_PARAMS, MODEL_PARAMS
from client import Client
from server import Server
from utils.args import parse_args
from utils.model_utils import read_data

STAT_METRICS_PATH = 'metrics/stat_metrics.csv'
SYS_METRICS_PATH = 'metrics/sys_metrics.csv'

# Establece las semillas para reproducibilidad
def set_seeds(seed):
    random.seed(1 + seed)
    np.random.seed(12 + seed)
    tf.set_random_seed(123 + seed)

# Verifica existencia del archivo del modelo
def verify_model_path(dataset, model):
    model_path = f'{dataset}/{model}.py'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'ERROR: Archivo no encontrado: {model_path}')
    return f'{dataset}.{model}'

# Inicializa modelos cliente y servidor
def init_models(model_path, seed, lr):
    model_params = MODEL_PARAMS[model_path]
    if lr != -1:
        model_params = (lr, *model_params[1:])
    tf.reset_default_graph()
    mod = importlib.import_module(model_path)
    ClientModel = getattr(mod, 'ClientModel')
    client_model = ClientModel(seed, *model_params)
    return client_model, Server(client_model)

# Configura clientes para la simulación
def setup_clients(dataset, model, use_val_set):
    eval_set = 'test' if not use_val_set else 'val'
    train_data_dir = os.path.join('..', 'data', dataset, 'data', 'train')
    test_data_dir = os.path.join('..', 'data', dataset, 'data', eval_set)
    users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)
    groups = groups or [[] for _ in users]
    return [Client(u, g, train_data[u], test_data[u], model) for u, g in zip(users, groups)]

# Función de escritura de métricas estadísticas
def get_stat_writer(client_ids, groups, client_num_samples, args):
    def writer(num_round, metrics, partition):
        metrics_writer.print_metrics(
            num_round, client_ids, metrics, groups, client_num_samples, partition,
            args.metrics_dir, f'{args.metrics_name}_stat')
    return writer

# Función de escritura de métricas del sistema
def get_sys_writer(args):
    def writer(num_round, client_ids, metrics, groups, client_num_samples):
        metrics_writer.print_metrics(
            num_round, client_ids, metrics, groups, client_num_samples, 'train',
            args.metrics_dir, f'{args.metrics_name}_sys')
    return writer

# Imprime métricas detalladas
def print_stats(num_round, server, clients, num_samples, args, writer, use_val_set):
    for set_to_use in ('train', 'val' if use_val_set else 'test'):
        metrics = server.test_model(clients, set_to_use=set_to_use)
        print_metrics(metrics, num_samples, prefix=f'{set_to_use}_')
        writer(num_round, metrics, set_to_use)

# Función auxiliar para impresión de métricas
def print_metrics(metrics, weights, prefix=''):
    ordered_weights = [weights[c] for c in sorted(weights)]
    metric_names = metrics_writer.get_metrics_names(metrics)
    for metric in metric_names:
        ordered_metric = [metrics[c][metric] for c in sorted(metrics)]
        print(f'{prefix}{metric}: {np.average(ordered_metric, weights=ordered_weights):g}, '
              f'10th percentile: {np.percentile(ordered_metric, 10):g}, '
              f'50th percentile: {np.percentile(ordered_metric, 50):g}, '
              f'90th percentile: {np.percentile(ordered_metric, 90):g}')

# Ejecuta la simulación de entrenamiento federado
def run_simulation(args):
    set_seeds(args.seed)
    model_path = verify_model_path(args.dataset, args.model)
    client_model, server = init_models(model_path, args.seed, args.lr)

    clients = setup_clients(args.dataset, client_model, args.use_val_set)
    client_ids, client_groups, client_num_samples = server.get_clients_info(clients)

    num_rounds, eval_every, clients_per_round = (
        args.num_rounds if args.num_rounds != -1 else MAIN_PARAMS[args.dataset][args.t][0],
        args.eval_every if args.eval_every != -1 else MAIN_PARAMS[args.dataset][args.t][1],
        args.clients_per_round if args.clients_per_round != -1 else MAIN_PARAMS[args.dataset][args.t][2]
    )

    stat_writer_fn = get_stat_writer(client_ids, client_groups, client_num_samples, args)
    sys_writer_fn = get_sys_writer(args)

    print_stats(0, server, clients, client_num_samples, args, stat_writer_fn, args.use_val_set)

    for i in range(num_rounds):
        server.select_clients(i, clients, clients_per_round)
        sys_metrics = server.train_model(num_epochs=args.num_epochs, batch_size=args.batch_size, minibatch=args.minibatch)
        sys_writer_fn(i + 1, client_ids, sys_metrics, client_groups, client_num_samples)
        server.update_model()

        if (i + 1) % eval_every == 0 or (i + 1) == num_rounds:
            print_stats(i + 1, server, clients, client_num_samples, args, stat_writer_fn, args.use_val_set)

    save_path = server.save_model(os.path.join('checkpoints', args.dataset, f'{args.model}.ckpt'))
    print(f'Model saved in path: {save_path}')
    server.close_model()

# Ejecución principal
if __name__ == '__main__':
    args = parse_args()
    run_simulation(args)

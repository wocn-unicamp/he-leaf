import json
from phe import paillier

# 1. Generar llaves
public_key, private_key = paillier.generate_paillier_keypair()

# 2. Serializar llaves
def save_paillier_keys(pub_key, priv_key, pub_filename, priv_filename):
    # Guardar la llave pública
    pub_data = {
        'n': pub_key.n
    }
    with open(pub_filename, 'w') as f:
        json.dump(pub_data, f)

    # Guardar la llave privada
    priv_data = {
        'p': priv_key.p,
        'q': priv_key.q,
        'public_key_n': priv_key.public_key.n
    }
    with open(priv_filename, 'w') as f:
        json.dump(priv_data, f)

save_paillier_keys(public_key, private_key, 'public_key.json', 'private_key.json')

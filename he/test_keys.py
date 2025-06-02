import json
from phe import paillier

def load_paillier_public_key(pub_filename):
    with open(pub_filename, 'r') as f:
        pub_data = json.load(f)
    return paillier.PaillierPublicKey(pub_data['n'])

def load_paillier_private_key(priv_filename):
    with open(priv_filename, 'r') as f:
        priv_data = json.load(f)
    public_key = paillier.PaillierPublicKey(priv_data['public_key_n'])
    return paillier.PaillierPrivateKey(public_key, priv_data['p'], priv_data['q'])

# Cargar llaves
public_key = load_paillier_public_key('public_key.json')
private_key = load_paillier_private_key('private_key.json')

# ¡Ahora puedes usar las llaves como antes!
dato1 = 5
dato2 = 7
cifrado1 = public_key.encrypt(dato1)
cifrado2 = public_key.encrypt(dato2)
suma_cifrada = cifrado1 + cifrado2
producto_cifrado = cifrado1 * 10

print("Suma:", private_key.decrypt(suma_cifrada))           # 12
print("Producto:", private_key.decrypt(producto_cifrado))   # 50

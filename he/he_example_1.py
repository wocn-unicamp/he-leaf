from phe import paillier

# 1. Generar llaves
public_key, private_key = paillier.generate_paillier_keypair()

# 2. Cifrar datos
dato1 = 5
dato2 = 7
cifrado1 = public_key.encrypt(dato1)
cifrado2 = public_key.encrypt(dato2)

# 3. Operaciones homomórficas
suma_cifrada = cifrado1 + cifrado2     # 5 + 7
producto_cifrado = cifrado1 * 10       # 5 * 10

# 4. Descifrar resultados
print("Suma:", private_key.decrypt(suma_cifrada))           # 12
print("Producto:", private_key.decrypt(producto_cifrado))   # 50

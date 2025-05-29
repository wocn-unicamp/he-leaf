
# HE-LEAF: Benchmark para Cenários Federados com Criptografia Homomórfica

Este repositório é uma versão do benchmark **LEAF** que incorpora criptografia homomórfica utilizando a biblioteca **Pyfhel**, proporcionando privacidade aprimorada em cenários de aprendizado federado.

## Recursos

- **Página Inicial Original:** [leaf.cmu.edu](https://leaf.cmu.edu)
- **Artigo Original:** ["LEAF: A Benchmark for Federated Settings"](https://arxiv.org/abs/1812.01097)
- **Biblioteca Utilizada:** [Pyfhel](https://github.com/ibarrond/Pyfhel)

## Conjunto de Dados

1. **FEMNIST**
   - Imagens de 28x28 pixels (ajustáveis para 128x128 pixels)
   - 62 classes (10 dígitos, 26 letras minúsculas, 26 letras maiúsculas)
   - 3500 usuários
   - Tarefa: Classificação de imagens

2. **Sentiment140**
   - Tweets com 660120 usuários
   - Tarefa: Análise de sentimentos

3. **Shakespeare**
   - Diálogos das obras de Shakespeare
   - 1129 usuários (reduzidos a 660 conforme escolha de comprimento de sequência)
   - Tarefa: Predição do próximo caractere

4. **Celeba**
   - Baseado em [Large-scale CelebFaces Attributes Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
   - 9343 usuários
   - Tarefa: Classificação de imagens (Sorrindo vs. Não sorrindo)

5. **Synthetic Dataset**
   - Geração personalizada de dados sintéticos desafiadores para aprendizado federado
   - Número de dispositivos, classes e dimensões personalizáveis
   - Tarefa: Classificação

6. **Reddit**
   - Comentários preprocessados do Reddit (dados de dezembro de 2017)
   - 1,660,820 usuários, 56,587,343 comentários
   - Tarefa: Predição da próxima palavra


## Instalação e Requisitos

⚠️ **Atenção:**  
Devido a problemas de incompatibilidade e indisponibilidade do Python 3.6 nos repositórios oficiais das versões mais recentes do Ubuntu (como 24.04 "Noble"), **recomenda-se fortemente** utilizar o [Miniconda](https://docs.conda.io/en/latest/miniconda.html) ou [Anaconda](https://www.anaconda.com/) para criar um ambiente virtual com Python 3.6.

### Passos para criar o ambiente com Conda:

1. **Instale o Miniconda** (caso ainda não tenha):

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Siga as instruções na tela e, ao final, reinicie seu terminal ou rode:
source ~/.bashrc
```

2. **Crie e ative o ambiente com Python 3.6**:

```bash
conda create -n leaf_env python=3.6
conda activate leaf_env
```



3. **Atualize o pip e instale as dependências necessárias**:

```bash
pip install --upgrade pip
pip install numpy==1.16.4 scipy==1.2.1 tensorflow==1.13.1 Pillow==6.2.1 matplotlib==3.0.3 jupyter==1.0.0 pandas==0.24.2 grpcio==1.16.1 protobuf==3.19.6 pyfhel
```

- Certifique-se de que `wget` está instalado e funcionando (especialmente em macOS).
- Para gerar os conjuntos de dados específicos, consulte as instruções dentro das respectivas pastas.
- A pasta `models` contém instruções para executar as implementações básicas de referência com a adição da criptografia homomórfica.


**Apagar ambiente (Se for necessario)**:
```bash
conda env remove -n leaf-env
```
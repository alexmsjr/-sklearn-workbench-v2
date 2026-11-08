# ##############################################################
#
#   É NESTE ARQUIVO QUE VOCÊ ESCREVE OS MÉTODOS DE MACHINE LEARNING.
#
#   A interface (main.py) nunca precisa ser alterada para acrescentar
#   um método. Ela descobre tudo pelo dicionário METODOS, lá no fim
#   deste arquivo — é o único ponto de ligação entre os dois.
#
#   Duas seções importam para você:
#
#     1) "SEUS MÉTODOS DE ML"  -> onde escrever o código do método
#     2) "LIGAÇÃO COM A TELA"  -> onde registrar o método que escreveu
#
#   Este arquivo continua rodando sozinho no terminal
#   (python ml_clas_sk.py), igual a antes.
#
# ##############################################################

import pandas as pd
import numpy as np
import configparser
import ast
import os

from sklearn.model_selection import train_test_split
from sklearn import metrics as me
from sklearn.preprocessing import StandardScaler

# METODOS DE MACHINE LEARNING
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

def metricas(texto,y_teste,y_pred):
    cm = me.confusion_matrix(y_teste, y_pred)
    ac = me.accuracy_score(y_teste, y_pred)
    pr = me.precision_score(y_teste, y_pred, average='macro')
    re = me.recall_score(y_teste, y_pred, average='macro')
    f1 = me.f1_score(y_teste, y_pred, average='macro')
    print("\n------- ",texto) 
    print("Matriz de Confusão:\n", cm)
    print(f"Acurácia...........: {ac*100:.2f}%")    
    print(f"Precisão...........: {pr*100:.2f}%")
    print(f"Recall.............: {re*100:.2f}%")
    print(f"F1-measure.........: {f1*100:.2f}%")

def ler_arquivo(arquivo): 
    df = pd.read_csv(arquivo, header=None) 
    
    atributos = df.iloc[:, :-1].astype(float).values.tolist() 
    classes = df.iloc[:, -1].tolist() 
    
    return atributos, classes


def converter(valor):
    try:
        return ast.literal_eval(valor)
    except (ValueError, SyntaxError):   
        return valor


def ler_config(arquivo="config_cl.txt"):
    cfg = configparser.ConfigParser()
    cfg.read(arquivo, encoding='utf-8')

    parametros = {}
    for secao in cfg.sections():
        parametros[secao] = {}
        for chave, valor in cfg[secao].items():
            parametros[secao][chave] = converter(valor)
    return parametros


# ==============================================================
#        DE QUAL ARQUIVO OS PARÂMETROS SÃO LIDOS
# ==============================================================
# São dois arquivos:
#
#   config_cl_padrao.txt  -> os valores padrão do scikit-learn.
#                            Gerado automaticamente, NÃO deve ser editado.
#   config_cl.txt         -> os seus valores. É o que o botão
#                            "Modificar Parâmetros" abre no Bloco de Notas.
#
# O botão "Parâmetros Padrão" da interface liga e desliga a variável
# usar_padrao abaixo, escolhendo entre os dois.
ARQUIVO_PADRAO = "config_cl_padrao.txt"
ARQUIVO_PERSONALIZADO = "config_cl.txt"

usar_padrao = False


# Pasta deste arquivo. Os config são procurados aqui, e não na pasta de
# onde o programa foi chamado: sem isto, rodar de outro diretório faz o
# configparser não achar nada, em silêncio, e o erro só aparece depois
# como um KeyError sem explicação.
PASTA = os.path.dirname(os.path.abspath(__file__))


def parametros_de(metodo):
    """Os parâmetros do método, vindos do arquivo que estiver ativo.

    A leitura acontece a cada execução, e não uma vez no início: assim o
    arquivo pode ser editado com o programa aberto que o valor novo já
    vale no próximo Executar, sem reiniciar.
    """
    arquivo = ARQUIVO_PADRAO if usar_padrao else ARQUIVO_PERSONALIZADO
    caminho = os.path.join(PASTA, arquivo)
    config = ler_config(caminho)

    if metodo not in config:
        raise ValueError(
            f"Não achei a seção [{metodo}] em {arquivo}.\n\n"
            f"Seções encontradas: {', '.join(config) or 'nenhuma'}\n"
            f"Arquivo procurado: {caminho}"
        )
    return config[metodo]
# ==============================================================
#                      SEUS MÉTODOS DE ML
# ==============================================================
# Cada método é um PAR de funções com estes nomes e assinaturas:
#
#   def <nome>_treino(x_train, y_train):
#       ...cria e treina o modelo...
#       return modelo              <- precisa RETORNAR o modelo
#
#   def <nome>_teste(x_teste, y_teste, modelo):
#       y_pred = modelo.predict(x_teste)
#       metricas("TEXTO NA TELA", y_teste, y_pred)
#
# O que o metricas() imprime é exatamente o que aparece no painel
# direito da interface — ela captura esses print(). Então, para mudar
# o que é exibido, mexa no metricas() ou chame outra coisa aqui.
#
# Quem faz o train_test_split e o StandardScaler é o runner.py; as
# funções abaixo recebem os dados já divididos e prontos.
#
# Escreva os métodos novos nesta seção, e depois registre cada um no
# METODOS no fim do arquivo.
# ==============================================================

# MLP
def mlp_treino(x_train, y_train):
    mlp_model = MLPClassifier(**parametros_de("MLP"))
    mlp_model.fit(x_train, y_train)
    return mlp_model
def mlp_teste(x_teste, y_teste, mlp_model): 
    y_pred = mlp_model.predict(x_teste)
    metricas("MLP",y_teste,y_pred)

# RANDOM FOREST
def rf_treino(x_train, y_train):
    rf_model = RandomForestClassifier(**parametros_de("Random Forest"))
    rf_model.fit(x_train, y_train)
    return rf_model
def rf_teste(x_teste, y_teste, rf_model):
    y_pred = rf_model.predict(x_teste)
    metricas("RANDOM FOREST", y_teste, y_pred)

# DECISION TREE
def dt_treino(x_train, y_train):
    dt_model = DecisionTreeClassifier(**parametros_de("Decision Tree"))
    dt_model.fit(x_train, y_train)
    return dt_model
def dt_teste(x_teste, y_teste, dt_model):
    y_pred = dt_model.predict(x_teste)
    metricas("DECISION TREE", y_teste, y_pred)

# KNEIGHBORS CLASSIFIER
def kn_treino(x_train, y_train):
    kn_model = KNeighborsClassifier(**parametros_de("KNeighbors Classifier"))
    print("PARAMETROS USADOS: ", kn_model.get_params())
    kn_model.fit(x_train, y_train)
    return kn_model

def kn_teste(x_teste,y_teste,kn_model):
    y_pred = kn_model.predict(x_teste)
    metricas("KNEIGHBORS CLASSIFIER", y_teste, y_pred)

# EXTRA TREE CLASSIFIER

def et_treino(x_train, y_train):
    et_model = ExtraTreeClassifier(**parametros_de("Extra Tree Classifier"))
    print("PARAMETROS USADOS: ", et_model.get_params())
    et_model.fit(x_train, y_train)
    return et_model

def et_teste(x_teste, y_teste, et_model):
    y_pred = et_model.predict(x_teste)
    metricas("EXTRA TREE CLASSIFIER", y_teste, y_pred)



# ==============================================================
#            LIGAÇÃO COM A TELA  (registre aqui)
# ==============================================================
# Esta é a ÚNICA ligação entre este arquivo e a interface. Tudo que
# estiver neste dicionário aparece no dropdown "Método Selecionado"
# e entra no "Executar Todos". Nada no main.py precisa mudar.
#
#   "nome que aparece na tela": (função_treino, função_teste, normalizar?)
#                                                             ^^^^^^^^^^
#      True  -> o método recebe os dados passados pelo StandardScaler
#      False -> o método recebe os dados crus
#
# Esse terceiro campo existe porque as duas funções de treino têm a
# mesma assinatura: olhando só para elas não dá para saber qual espera
# dado normalizado. Ele reproduz o que o módulo principal abaixo já
# fazia — a MLP recebia x_treino_sc e a Random Forest recebia x_treino.
#
# PARA ACRESCENTAR UM MÉTODO NOVO:
#   1. escreva o par *_treino / *_teste na seção "SEUS MÉTODOS DE ML"
#   2. acrescente uma linha aqui, como no exemplo comentado abaixo
METODOS = {
    "MLP":           (mlp_treino, mlp_teste, True),
    "Random Forest": (rf_treino,  rf_teste,  False),
    "Decision Tree": (dt_treino,  dt_teste,  False),
    "KNeighbors Classifier": (kn_treino,  kn_teste,  True),
    "Extra Tree Classifier": (et_treino, et_teste,  False),

    # exemplo de um método novo (descomente depois de escrever o par):
    # "SVM":         (svm_treino, svm_teste, True),
}


# **************************************************************
#                       MÓDULO PRINCIPAL
# **************************************************************
if __name__ == "__main__":
    arquivo = "dados.txt"
    feat, cl_str = ler_arquivo(arquivo)
    classes = list(set(cl_str))
    cl = np.array([classes.index(c) for c in cl_str])
    x_treino, x_teste, y_treino, y_teste = train_test_split(
        feat,
        cl,
        test_size=0.2,
        random_state=42,
        stratify=cl
    )

    # NORMALIZAÇÃO PARA MLP
    scaler = StandardScaler()
    x_treino_sc = scaler.fit_transform(x_treino)
    x_teste_sc = scaler.transform(x_teste)

    # MLP
    modelo_mlp = mlp_treino(x_treino_sc,y_treino)
    mlp_teste(x_teste_sc,y_teste,modelo_mlp)

    # RANDOM FOREST
    modelo_rf = rf_treino(x_treino,y_treino)
    rf_teste(x_teste,y_teste,modelo_rf)

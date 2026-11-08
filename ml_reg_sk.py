# ##############################################################
#
#   É NESTE ARQUIVO QUE VOCÊ ESCREVE OS MÉTODOS DE REGRESSÃO.
#
#   Mesma estrutura do ml_clas_sk.py: os hiperparâmetros vêm de
#   arquivo de texto e os métodos são registrados no METODOS, no fim
#   do arquivo, que é o único ponto de ligação com a interface.
#
#   Continua rodando sozinho no terminal (python ml_reg_sk.py).
#
# ##############################################################

import pandas as pd
import numpy as np
import configparser
import ast
import os

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics as me
from sklearn.preprocessing import StandardScaler

def metricas(texto,y_teste,y_pred):
    mae = me.mean_absolute_error(y_teste, y_pred)
    mse = me.mean_squared_error(y_teste, y_pred)
    rmse = np.sqrt(mse)
    r2 = me.r2_score(y_teste, y_pred)
    print("\n------------------------------------",texto)
    print("------------------------------------")
    print(f"MAE ...............: {mae:.4f}")
    print(f"MSE ...............: {mse:.4f}")
    print(f"RMSE ..............: {rmse:.4f}")
    print(f"R² ................: {r2:.4f}")


# ==============================================================
# LEITURA DO ARQUIVO
# ==============================================================
def ler_arquivo(arquivo):
    df = pd.read_csv(arquivo, header=None)
    atributos = df.iloc[:, :-1].astype(float).values
    target = df.iloc[:, -1].astype(float).values
    return atributos, target

# ==============================================================
#        DE QUAL ARQUIVO OS PARÂMETROS SÃO LIDOS
# ==============================================================
# São dois arquivos:
#
#   config_reg_padrao.txt -> os valores padrão do scikit-learn.
#                            Gerado automaticamente, NÃO deve ser editado.
#   config_reg.txt        -> os seus valores. É o que o botão
#                            "Modificar Parâmetros" abre no Bloco de Notas.
#
# O botão "Parâmetros Padrão" da interface liga e desliga a variável
# usar_padrao abaixo, escolhendo entre os dois.
ARQUIVO_PADRAO = "config_reg_padrao.txt"
ARQUIVO_PERSONALIZADO = "config_reg.txt"

usar_padrao = False


def converter(valor):
    """Texto do arquivo -> valor Python do tipo certo.

    O literal_eval só entende literais (números, tuplas, None). Quando o
    valor é uma palavra solta como 'relu', ele falha — e o except devolve
    o texto puro, que é o que o scikit-learn espera nesses parâmetros.
    """
    try:
        return ast.literal_eval(valor)
    except (ValueError, SyntaxError):
        return valor


def ler_config(arquivo):
    cfg = configparser.ConfigParser()
    cfg.read(arquivo, encoding='utf-8')

    parametros = {}
    for secao in cfg.sections():
        parametros[secao] = {}
        for chave, valor in cfg[secao].items():
            parametros[secao][chave] = converter(valor)
    return parametros


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
#                   SEUS MÉTODOS DE REGRESSÃO
# ==============================================================
def mlp_treino(x_train, y_train):
    mlp_model = MLPRegressor(**parametros_de("MLP"))
    mlp_model.fit(x_train, y_train)
    return mlp_model

def mlp_teste(x_teste, y_teste, mlp_model):
    y_pred = mlp_model.predict(x_teste)
    metricas("MLP",y_teste,y_pred)
    
def random_forest_treino(x_train, y_train):
    rf_model = RandomForestRegressor(**parametros_de("Random Forest"))
    rf_model.fit(x_train, y_train)
    return rf_model

def random_forest_teste(x_teste, y_teste, rf_model):
    y_pred = rf_model.predict(x_teste)
    metricas("RANDOM FOREST",y_teste,y_pred)


# ==============================================================
#            LIGAÇÃO COM A TELA  (registre aqui)
# ==============================================================
# Esta é a ÚNICA ligação entre este arquivo e a interface. Tudo que
# estiver neste dicionário aparece no dropdown "Método Selecionado"
# da aba Regressão e entra no "Executar Todos".
#
#   "nome na tela": (função_treino, função_teste, normalizar?)
#
#      True  -> o método recebe os dados passados pelo StandardScaler
#      False -> o método recebe os dados crus
METODOS = {
    "MLP":           (mlp_treino, mlp_teste, True),
    "Random Forest": (random_forest_treino, random_forest_teste, False),
}
# ==============================================================
#                       MÓDULO PRINCIPAL
# ==============================================================
# Só roda quando este arquivo é executado direto no terminal.
# Quando a interface importa o módulo, este bloco é pulado.
if __name__ == "__main__":
    arquivo = "dados_reg.txt"

    feat, target = ler_arquivo(arquivo)

    x_treino, x_teste, y_treino, y_teste = train_test_split(
        feat, target, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    x_treino_sc = scaler.fit_transform(x_treino)
    x_teste_sc = scaler.transform(x_teste)

    modelo = mlp_treino(x_treino_sc, y_treino)
    mlp_teste(x_teste_sc, y_teste, modelo)

    modelo = random_forest_treino(x_treino, y_treino)
    random_forest_teste(x_teste, y_teste, modelo)

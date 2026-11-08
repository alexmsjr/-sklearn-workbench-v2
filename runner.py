"""
Ponte entre a interface e os scripts de Machine Learning.

Esta é a única parte do projeto que encosta no arquivo do professor, e ela
roda sem abrir janela nenhuma:

    python runner.py <arquivo> [pct_teste]

Isso serve para conferir que a saída bate com a do terminal antes de culpar
a interface por algum número diferente.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import os
import warnings
from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------
# Carregamento do módulo de ML
#
# O import é adiado e protegido de propósito, por dois motivos:
#
# 1. O ml_clas_sk.py é o arquivo que você edita o tempo todo. Um erro nele
#    (um SyntaxError no meio de uma alteração, por exemplo) derrubaria a
#    interface antes da janela aparecer, sem explicação nenhuma.
#
# 2. O Smart App Control do Windows às vezes bloqueia os .pyd do scipy e do
#    scikit-learn enquanto consulta a reputação deles na nuvem. É um bloqueio
#    transitório: na tentativa seguinte costuma funcionar. Mas se o import
#    estivesse no topo deste arquivo, ele derrubaria o app na inicialização.
#
# Com o import aqui dentro, a janela sempre abre e o erro aparece no painel
# de saída, onde dá para ler e tentar de novo.
# --------------------------------------------------------------------------
CLASSIFICACAO = "Classificação"
REGRESSAO = "Regressão"

# Qual arquivo de ML atende cada aba da interface.
MODULOS = {
    CLASSIFICACAO: "ml_clas_sk",
    REGRESSAO: "ml_reg_sk",
}

_tipo = CLASSIFICACAO
_modulo = None
_erro_import = None


def tipo_atual() -> str:
    return _tipo


def definir_tipo(tipo: str) -> None:
    """Troca entre classificação e regressão.

    Descarta o módulo em cache para o próximo acesso importar o outro
    arquivo de ML.
    """
    global _tipo, _modulo, _erro_import
    if tipo not in MODULOS:
        raise ValueError(f"Tipo desconhecido: {tipo}")
    if tipo != _tipo:
        _tipo = tipo
        _modulo = None
        _erro_import = None


def modulo_ml():
    """O módulo de ML do tipo atual, ou None se não pôde ser importado."""
    global _modulo, _erro_import
    if _modulo is None and _erro_import is None:
        try:
            _modulo = importlib.import_module(MODULOS[_tipo])
        except Exception as erro:
            _erro_import = erro
    return _modulo


def esquecer_erro() -> None:
    """Descarta um erro de import anterior para permitir nova tentativa.

    Serve para o bloqueio transitório do Smart App Control: basta o usuário
    clicar de novo em vez de reabrir o programa.
    """
    global _erro_import
    _erro_import = None


def erro_import():
    """A exceção que impediu o carregamento do módulo de ML, se houve."""
    modulo_ml()
    return _erro_import


# --------------------------------------------------------------------------
# Dados carregados
# --------------------------------------------------------------------------
@dataclass
class Dados:
    """O arquivo do usuário já lido e pronto para treinar."""

    caminho: str
    feat: list          # atributos (lista de listas de float)
    cl: np.ndarray      # classificação: classes em inteiro / regressão: alvo float
    classes: list       # nomes das classes; vazio na regressão
    tipo: str = CLASSIFICACAO

    @property
    def nome(self) -> str:
        return os.path.basename(self.caminho)

    @property
    def n_instancias(self) -> int:
        return len(self.feat)

    @property
    def n_atributos(self) -> int:
        return len(self.feat[0]) if len(self.feat) else 0

    def resumo(self) -> str:
        """Texto curto para o card do arquivo na interface."""
        cabeca = f"{self.n_instancias} instâncias · {self.n_atributos} atributos"
        if self.tipo == REGRESSAO:
            # Na regressão não há classes: o que importa é a faixa do alvo.
            return (
                f"{cabeca}\n"
                f"alvo de {self.cl.min():.4g} a {self.cl.max():.4g} "
                f"(média {self.cl.mean():.4g})"
            )
        nomes = ", ".join(str(c) for c in self.classes)
        return f"{cabeca}\n{len(self.classes)} classes: {nomes}"


# --------------------------------------------------------------------------
# API usada pela interface
# --------------------------------------------------------------------------
def arquivo_personalizado() -> str:
    """O arquivo de parâmetros que o botão 'Modificar Parâmetros' abre.

    Vem do próprio módulo de ML, então acompanha a aba ativa sem a
    interface precisar saber os nomes dos arquivos.
    """
    mod = modulo_ml()
    return getattr(mod, "ARQUIVO_PERSONALIZADO", "") if mod else ""


def nomes_metodos() -> list:
    """Os métodos declarados no METODOS do arquivo de ML."""
    mod = modulo_ml()
    return list(mod.METODOS.keys()) if mod else []


def carregar(caminho: str) -> Dados:
    """Lê o arquivo apontado pelo usuário usando o ler_arquivo do professor."""
    if modulo_ml() is None:
        raise ValueError(
            "O módulo de ML não pôde ser carregado:\n\n" + str(erro_import())
        )
    try:
        feat, cl_str = modulo_ml().ler_arquivo(caminho)
    except Exception as erro:
        raise ValueError(
            f"Não consegui ler '{os.path.basename(caminho)}'.\n\n"
            f"O formato esperado é CSV sem cabeçalho, com os atributos "
            f"numéricos e a última coluna sendo "
            + ("o alvo numérico" if _tipo == REGRESSAO else "a classe")
            + ".\n\n"
            f"Detalhe: {erro}"
        ) from erro

    if not len(feat):
        raise ValueError("O arquivo está vazio.")

    if _tipo == REGRESSAO:
        # Na regressão o alvo já é numérico: não há classes para codificar.
        return Dados(caminho, feat, np.asarray(cl_str, dtype=float), [], REGRESSAO)

    # O professor usa list(set(...)) aqui. Trocado por sorted() para a ordem
    # das classes ser sempre a mesma: set de strings muda de ordem a cada
    # execução do Python, o que embaralhava as linhas da matriz de confusão.
    # As métricas não mudam, só a ordem dos rótulos fica estável.
    classes = sorted(set(cl_str))
    cl = np.array([classes.index(c) for c in cl_str])
    return Dados(caminho, feat, cl, classes, CLASSIFICACAO)


def executar(nome: str, dados: Dados, pct_teste: int, usar_padrao: bool = False) -> str:
    """Treina e testa um método, devolvendo o que ele imprimiu."""
    # Importado aqui, e não no topo do arquivo, para um bloqueio transitório
    # do Smart App Control não derrubar o app na inicialização.
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    modulo = modulo_ml()
    # Diz ao módulo de ML de qual arquivo ler os hiperparâmetros. Ele relê
    # a cada execução, então a troca vale já nesta.
    modulo.usar_padrao = usar_padrao

    treino_fn, teste_fn, normalizar = modulo.METODOS[nome]

    x_treino, x_teste, y_treino, y_teste = train_test_split(
        dados.feat,
        dados.cl,
        test_size=pct_teste / 100,
        random_state=42,
        # Estratificar só faz sentido com classes; na regressão o alvo é
        # contínuo e quase todo valor seria único.
        stratify=dados.cl if dados.tipo == CLASSIFICACAO else None,
    )

    # Só os métodos marcados no METODOS recebem os dados normalizados,
    # igual ao que o módulo principal do professor faz.
    if normalizar:
        scaler = StandardScaler()
        x_treino = scaler.fit_transform(x_treino)
        x_teste = scaler.transform(x_teste)

    saida = io.StringIO()
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        with contextlib.redirect_stdout(saida), contextlib.redirect_stderr(saida):
            modelo = treino_fn(x_treino, y_treino)
            teste_fn(x_teste, y_teste, modelo)

    texto = saida.getvalue()

    # Avisos do sklearn (convergência, por exemplo) não passam pelo print(),
    # então seriam perdidos sem isto.
    vistos = []
    for aviso in avisos:
        msg = f"[aviso] {aviso.category.__name__}: {aviso.message}"
        if msg not in vistos:
            vistos.append(msg)
    if vistos:
        texto += "\n" + "\n".join(vistos) + "\n"

    return texto


def executar_todos(dados: Dados, pct_teste: int, usar_padrao: bool = False):
    """Roda todos os métodos em sequência, um resultado por vez."""
    for nome in nomes_metodos():
        yield nome, executar(nome, dados, pct_teste, usar_padrao)


# --------------------------------------------------------------------------
# Execução pelo terminal, sem interface
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("uso: python runner.py <arquivo> [pct_teste]")
        raise SystemExit(1)

    pct = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    dados = carregar(sys.argv[1])

    print(f"Arquivo: {dados.nome}")
    print(dados.resumo())
    print(f"Teste: {pct}%")

    for _nome, texto in executar_todos(dados, pct):
        print(texto, end="")

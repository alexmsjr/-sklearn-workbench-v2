# sklearn-workbench

Trabalho de Inteligência Artificial II. A proposta era montar um Weka próprio:
em vez de escrever um script novo a cada experimento, ter uma tela onde eu aponto
a base, escolho o método, defino a divisão de treino e teste e leio as métricas.
Por baixo é tudo scikit-learn; o que eu construí foi a bancada em volta dele.

Feito por Alexandre Marciano e Davi de Castro.

## O que ele faz

- Classificação e regressão, cada uma na sua aba.
- Carrega a base pelo seletor de arquivos do sistema.
- Executa um método ou todos de uma vez, e guarda o histórico das execuções —
  clicar numa linha antiga traz o resultado de volta sem treinar de novo.
- Mostra a base carregada numa aba separada, para conferir o que entrou.
- Os hiperparâmetros vêm de arquivo de texto, não do código.

## Os parâmetros ficam em arquivo

Essa foi a parte que mais mudou como eu trabalho. Cada método lê os
hiperparâmetros de um `.txt` no formato INI, lido de novo a cada execução:

```ini
[MLP]
hidden_layer_sizes = (125,)
activation = logistic
alpha = 0.01
max_iter = 100000
```

Dá para editar com o programa aberto e o valor novo já vale no próximo Executar.
São dois arquivos por tipo de problema:

| arquivo | o que é |
|---|---|
| `config_cl.txt` / `config_reg.txt` | os meus valores. É o que o botão **Modificar Parâmetros** abre no Bloco de Notas |
| `config_cl_padrao.txt` / `config_reg_padrao.txt` | os valores padrão do scikit-learn, gerados do `get_params()`. Não edito |

O botão **Parâmetros Padrão** alterna entre os dois, então dá para comparar o meu
ajuste com a biblioteca crua sem apagar nada.

## Como rodar

```
pip install -r requirements.txt
python main.py
```

O `scikit-learn` está travado na 1.7.2 de propósito — o motivo está comentado no
`requirements.txt`.

Os métodos também rodam sozinhos no terminal, sem abrir janela:

```
python ml_clas_sk.py
python runner.py <base> [pct_teste]
```

## A base de dados

CSV sem cabeçalho: os atributos numéricos nas primeiras colunas e a última coluna
com a classe (ou, na regressão, com o valor do alvo).

```
5.1,3.5,1.4,0.2,setosa
4.9,3.0,1.4,0.2,setosa
```

A base que eu usei não está no repositório — aponte o programa para a sua.

## Os arquivos

| arquivo | o que é |
|---|---|
| `ml_clas_sk.py` | os métodos de classificação |
| `ml_reg_sk.py` | os métodos de regressão |
| `main.py` | a interface, em PySide6 |
| `runner.py` | a ponte entre as duas; é a única parte que encosta nos arquivos de ML |
| `style.qss` | o visual — mexer aqui não exige tocar no `main.py` |
| `fontes/` | Source Serif 4 embutida (licença SIL OFL), carregada por código, para a interface sair igual em qualquer máquina |

## Acrescentar um método

Escrever o par de funções e registrar no dicionário `METODOS`, no fim do arquivo:

```python
def svm_treino(x_train, y_train):
    modelo = SVC(**parametros_de("SVM"))
    modelo.fit(x_train, y_train)
    return modelo

def svm_teste(x_teste, y_teste, modelo):
    metricas("SVM", y_teste, modelo.predict(x_teste))

METODOS = {
    "MLP":           (mlp_treino, mlp_teste, True),    # True = dado normalizado
    "Random Forest": (rf_treino,  rf_teste,  False),   # False = dado cru
    "SVM":           (svm_treino, svm_teste, True),    # <- o novo
}
```

Ele aparece sozinho no dropdown e entra no "Executar Todos". Não precisa mexer na
interface — o `METODOS` é o único ponto de contato entre os dois lados.

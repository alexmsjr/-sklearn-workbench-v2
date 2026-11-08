"""
myWeka - interface para os scripts de Machine Learning.

Reproduz o desenho feito no Photoshop. Os valores do bloco MEDIDAS foram
extraidos pixel a pixel dos PNGs do mock, nao estimados no olho.

    python main.py

A logica de ML nao mora aqui: esta interface so conversa com runner.py, que
por sua vez chama o ml_clas_sk.py.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QIntValidator,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import runner

# ==========================================================================
# MEDIDAS - extraidas do mock
# ==========================================================================
# Todos os valores abaixo foram medidos pixel a pixel no mock, nao
# estimados. As cores correspondentes estao no style.qss.
JANELA_L, JANELA_A = 940, 740
# Minimo medido testando altura por altura com arquivo carregado (o
# pior caso). Nao use o minimumSizeHint() do Qt: ele ignora a altura
# real de rotulos com quebra de linha e ja errou neste projeto.
JANELA_MIN_L, JANELA_MIN_A = 780, 640
RAIO_JANELA = 11        # canto arredondado da janela, medido no mock

PAINEL_L = 278          # largura escolhida p/ o rodape caber numa linha
MARGEM = 14             # recuo dos controles dentro do painel
CONTROLE_A = 38         # altura de cada caixa de controle
ESP = 9                 # espaco vertical entre as caixas
BARRA_ABAS_A = 40       # altura da barra de abas do topo
ESP_ABAS_CONTEUDO = 11  # faixa do fundo entre a barra e o conteudo

CELULA_ICONE_L = 38     # celula quadrada do icone, a esquerda da caixa
CELULA_TESTE_L = 82     # celula do rotulo "% Teste:"
ALTURA_MIN_HISTORICO = 110

# A logo saiu do mock novo, mas o codigo que a desenha continua aqui.
MOSTRAR_LOGO = False
LOGO_L = 150

COR_ICONE = "#F2F2F2"
COR_ICONE_OFF = "#6A6B6D"

# Texto do tooltip do botao que liga/desliga os parametros padrao.
AJUDA_PADRAO = """<b>Parâmetros Padrão ligado</b><br>
Os métodos rodam com os valores que o scikit-learn usa por omissão, lidos de
um arquivo gerado automaticamente que não deve ser editado. Serve de linha de
base para comparação.<br><br>
<b>Parâmetros Padrão desligado</b><br>
Os métodos usam os valores do seu arquivo personalizado — o mesmo que o botão
<i>Modificar Parâmetros</i> abre no Bloco de Notas."""

PASTA = os.path.dirname(os.path.abspath(__file__))


# ==========================================================================
# Icones desenhados a mao
#
# Pintados com QPainter em vez de usar caracteres de fonte: o traco fino do
# mock depende de fonte de simbolos que nem toda maquina tem, e um glifo
# faltando viraria um quadrado vazio na tela.
# ==========================================================================
def _caneta(p: QPainter, cor: str, largura: float = 1.4) -> None:
    pen = QPen(QColor(cor))
    pen.setWidthF(largura)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)


def icone_documento(p: QPainter, r: QRectF, cor: str) -> None:
    larg, alt, dobra = 13.0, 16.0, 4.5
    x = r.center().x() - larg / 2
    y = r.center().y() - alt / 2
    _caneta(p, cor)

    folha = QPainterPath()
    folha.moveTo(x, y)
    folha.lineTo(x + larg - dobra, y)
    folha.lineTo(x + larg, y + dobra)
    folha.lineTo(x + larg, y + alt)
    folha.lineTo(x, y + alt)
    folha.closeSubpath()
    p.drawPath(folha)

    canto = QPainterPath()
    canto.moveTo(x + larg - dobra, y)
    canto.lineTo(x + larg - dobra, y + dobra)
    canto.lineTo(x + larg, y + dobra)
    p.drawPath(canto)


def icone_seta(p: QPainter, r: QRectF, cor: str) -> None:
    cx, cy = r.center().x(), r.center().y()
    meio, ponta = 8.0, 4.5
    _caneta(p, cor)
    p.drawLine(QPointF(cx - meio, cy), QPointF(cx + meio, cy))
    cabeca = QPainterPath()
    cabeca.moveTo(cx + meio - ponta, cy - ponta)
    cabeca.lineTo(cx + meio, cy)
    cabeca.lineTo(cx + meio - ponta, cy + ponta)
    p.drawPath(cabeca)


def icone_chevron(p, r, cor, baixo=True, tam=5.0, largura=1.4) -> None:
    cx, cy = r.center().x(), r.center().y()
    dy = tam * 0.55
    topo = cy - dy if baixo else cy + dy
    base = cy + dy if baixo else cy - dy
    _caneta(p, cor, largura)
    cam = QPainterPath()
    cam.moveTo(cx - tam, topo)
    cam.lineTo(cx, base)
    cam.lineTo(cx + tam, topo)
    p.drawPath(cam)


def icone_lista(p: QPainter, r: QRectF, cor: str) -> None:
    """Tres tracos horizontais, o 'hamburguer' do mock."""
    cx, cy = r.center().x(), r.center().y()
    meio = 7.5
    _caneta(p, cor, 1.5)
    for dy in (-5.0, 0.0, 5.0):
        p.drawLine(QPointF(cx - meio, cy + dy), QPointF(cx + meio, cy + dy))


def icone_alvo(p: QPainter, r: QRectF, cor: str, ligado: bool = False) -> None:
    """Circulo vazio quando desligado, com miolo cheio quando ligado."""
    centro = QPointF(r.center().x(), r.center().y())
    _caneta(p, cor, 1.4)
    p.drawEllipse(centro, 7.0, 7.0)
    if ligado:
        p.setBrush(QColor(cor))
        p.drawEllipse(centro, 3.6, 3.6)


def icone_abrir(p: QPainter, r: QRectF, cor: str) -> None:
    """Caixa com seta saindo: abrir em outro programa."""
    cx, cy = r.center().x(), r.center().y()
    lado = 6.0
    _caneta(p, cor, 1.4)

    # caixa aberta no canto superior direito
    cam = QPainterPath()
    cam.moveTo(cx + lado - 4, cy - lado)
    cam.lineTo(cx - lado, cy - lado)
    cam.lineTo(cx - lado, cy + lado)
    cam.lineTo(cx + lado, cy + lado)
    cam.lineTo(cx + lado, cy - lado + 4)
    p.drawPath(cam)

    # a seta saindo
    p.drawLine(QPointF(cx, cy), QPointF(cx + lado + 3, cy - lado - 3))
    ponta = QPainterPath()
    ponta.moveTo(cx + lado - 1, cy - lado - 3)
    ponta.lineTo(cx + lado + 3, cy - lado - 3)
    ponta.lineTo(cx + lado + 3, cy - lado + 1)
    p.drawPath(ponta)


def icone_engrenagem(p: QPainter, r: QRectF, cor: str) -> None:
    # Circulo central mais oito dentes radiais. Desenhada com traco fino
    # para combinar com os outros icones do painel.
    cx, cy = r.center().x(), r.center().y()
    raio_int, raio_ext = 4.2, 7.0
    _caneta(p, cor, 1.3)
    p.drawEllipse(QPointF(cx, cy), raio_int, raio_int)
    for i in range(8):
        ang = math.pi * 2 * i / 8
        dx, dy = math.cos(ang), math.sin(ang)
        p.drawLine(
            QPointF(cx + dx * raio_int, cy + dy * raio_int),
            QPointF(cx + dx * raio_ext, cy + dy * raio_ext),
        )


def icone_x(p, r, cor, tam=4.5, largura=1.3) -> None:
    cx, cy = r.center().x(), r.center().y()
    _caneta(p, cor, largura)
    p.drawLine(QPointF(cx - tam, cy - tam), QPointF(cx + tam, cy + tam))
    p.drawLine(QPointF(cx + tam, cy - tam), QPointF(cx - tam, cy + tam))


def icone_menos(p, r, cor, tam=5.0, largura=1.3) -> None:
    cx, cy = r.center().x(), r.center().y()
    _caneta(p, cor, largura)
    p.drawLine(QPointF(cx - tam, cy), QPointF(cx + tam, cy))


# ==========================================================================
# Widgets
# ==========================================================================
class BotaoControle(QPushButton):
    """Botao de 50px do painel, com o icone alinhado a direita."""

    def __init__(self, texto, icone, parent=None):
        super().__init__(texto, parent)
        self.setObjectName("controle")
        self.setFixedHeight(CONTROLE_A)
        self.setCursor(Qt.PointingHandCursor)
        self._icone = icone

    def paintEvent(self, ev):
        super().paintEvent(ev)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        cor = COR_ICONE if self.isEnabled() else COR_ICONE_OFF
        # Centro em width-22. No mock media width-27, mas os rotulos de
        # la eram curtos; "Modificar Parametros" tem 165px e encostava no
        # icone. Os 5px de folga nao sao perceptiveis nos outros botoes.
        self._icone(p, QRectF(self.width() - 36, 0, 28, self.height()), cor)
        p.end()


class CelulaIcone(QWidget):
    """Celula quadrada a esquerda da caixa, so com o icone."""

    def __init__(self, icone, parent=None):
        super().__init__(parent)
        self.setObjectName("celulaIcone")
        self.setFixedWidth(CELULA_ICONE_L - 2)  # desconta as bordas da caixa
        self._icone = icone

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        cor = COR_ICONE if self.isEnabled() else COR_ICONE_OFF
        self._icone(p, QRectF(0, 0, self.width(), self.height()), cor)
        p.end()


class Caixa(QPushButton):
    """Caixa do mock: borda fina com celulas separadas por divisorias.

    O truque: a propria caixa tem o fundo na cor da borda (#555555) e um
    recuo de 1px; as celulas sao opacas por cima. O que sobra desenha a
    moldura e as divisorias internas, sem precisar de widget separado
    para cada linha.

    Herda de QPushButton para ganhar clicked, setCheckable e setEnabled
    de graca; o texto dela fica vazio, quem mostra texto sao as celulas.
    """

    def __init__(self, destaque=False, parent=None):
        super().__init__(parent)
        self.setObjectName("caixaForte" if destaque else "caixa")
        self.setFixedHeight(CONTROLE_A)
        self.setCursor(Qt.PointingHandCursor)

        self._linha = QHBoxLayout(self)
        self._linha.setContentsMargins(1, 1, 1, 1)
        self._linha.setSpacing(1)  # a divisoria entre celulas

    def celula(self, widget, largura=None, expande=False):
        if largura is not None:
            widget.setFixedWidth(largura)
        # Sem isto um QLineEdit fica com a altura natural dele (26px) e
        # sobra faixa vazia dentro da celula.
        widget.setMinimumHeight(CONTROLE_A - 2)
        self._linha.addWidget(widget, 1 if expande else 0)
        return widget

    # O Qt nao propaga o :hover do pai para os filhos no seletor, como o
    # CSS faria. Entao o realce e marcado numa propriedade das celulas e
    # o estilo e reaplicado na mao.
    def _marcar(self, sob_o_mouse: bool) -> None:
        for i in range(self._linha.count()):
            w = self._linha.itemAt(i).widget()
            if w is None:
                continue
            w.setProperty("sobOMouse", sob_o_mouse)
            w.style().unpolish(w)
            w.style().polish(w)

    def enterEvent(self, ev):
        super().enterEvent(ev)
        if self.isEnabled():
            self._marcar(True)

    def leaveEvent(self, ev):
        super().leaveEvent(ev)
        self._marcar(False)

    def rotulo(self, texto, nome="celulaRotulo"):
        lbl = QLabel(texto)
        lbl.setObjectName(nome)
        lbl.setAlignment(Qt.AlignVCenter)
        return self.celula(lbl, expande=True)


class BotaoToggle(BotaoControle):
    """Botao liga/desliga, com indicador circular no lugar do icone.

    Vazio = desligado, preenchido = ligado.
    """

    def __init__(self, texto, parent=None):
        super().__init__(texto, self._desenhar, parent)
        self.setCheckable(True)

    def _desenhar(self, p, r, cor):
        centro = QPointF(r.center().x(), r.center().y())
        _caneta(p, cor, 1.4)
        p.drawEllipse(centro, 6.5, 6.5)
        if self.isChecked():
            p.setBrush(QColor(cor))
            p.drawEllipse(centro, 3.4, 3.4)


class BarraAbas(QFrame):
    """As abas do topo: Classificacao e Regressao."""

    mudou = Signal(str)

    def __init__(self, nomes, nome_aba="aba", parent=None):
        super().__init__(parent)
        self.setObjectName("barraAbas")

        linha = QHBoxLayout(self)
        linha.setContentsMargins(0, 0, 0, 0)
        linha.setSpacing(0)

        self._botoes = []
        for i, nome in enumerate(nomes):
            b = QPushButton(nome)
            b.setObjectName(nome_aba)
            b.setCheckable(True)
            b.setChecked(i == 0)
            b.setCursor(Qt.PointingHandCursor)
            # Sem isto a aba fica uma pastilha curta boiando no meio da
            # barra: o QPushButton nasce com altura fixa, entao o layout
            # da a ele so o sizeHint e centra o resto. Esticada, ela
            # desce ate encostar no conteudo, que e o efeito do desenho.
            b.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
            b.clicked.connect(lambda _=False, n=nome: self._escolher(n))
            self._botoes.append(b)
            linha.addWidget(b)
        linha.addStretch(1)

    def _escolher(self, nome: str) -> None:
        for b in self._botoes:
            b.setChecked(b.text() == nome)
        self.mudou.emit(nome)

    def atual(self) -> str:
        for b in self._botoes:
            if b.isChecked():
                return b.text()
        return self._botoes[0].text()

    def paintEvent(self, ev):
        """Desenha a divisoria de baixo pulando a aba ativa.

        E isso que cria o efeito de pasta: a aba ativa fica aberta para o
        conteudo, as outras ficam fechadas por uma linha. Feito a mao
        porque o QSS nao consegue interromper uma borda no meio.
        """
        super().paintEvent(ev)
        if not self.property("linhaInferior"):
            return
        ativo = next((b for b in self._botoes if b.isChecked()), None)
        p = QPainter(self)
        p.setPen(QPen(QColor("#555555"), 1))
        y = self.height() - 1
        if ativo is None:
            p.drawLine(0, y, self.width(), y)
        else:
            p.drawLine(0, y, ativo.x() - 1, y)
            p.drawLine(ativo.x() + ativo.width(), y, self.width(), y)
        p.end()


class ComboMetodo(QComboBox):
    """Dropdown 'Metodo Selecionado', com o chevron pintado."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("controle")
        self.setFixedHeight(CONTROLE_A)
        self.setCursor(Qt.PointingHandCursor)

    def paintEvent(self, ev):
        super().paintEvent(ev)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        cor = COR_ICONE if self.isEnabled() else COR_ICONE_OFF
        # O chevron do combo fica mais a direita que os icones dos botoes:
        # centro em width-19, tambem medido no mock.
        icone_chevron(p, QRectF(self.width() - 33, 0, 28, self.height()), cor, True)
        p.end()


class BotaoChevron(QPushButton):
    """Uma das setas do spinner do %Teste."""

    def __init__(self, baixo, parent=None):
        super().__init__(parent)
        self.setObjectName("chevron")
        self.setCursor(Qt.PointingHandCursor)
        self.setAutoRepeat(True)
        self.setAutoRepeatDelay(400)
        self.setAutoRepeatInterval(90)
        self._baixo = baixo

    def paintEvent(self, ev):
        super().paintEvent(ev)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        cor = COR_ICONE if self.isEnabled() else COR_ICONE_OFF
        icone_chevron(p, QRectF(0, 0, self.width(), self.height()), cor, self._baixo, 5.0, 1.3)
        p.end()


class CampoNumero(QLineEdit):
    """Campo do %Teste. Seleciona o conteudo inteiro ao receber o foco.

    Sem isto o campo parece travado assim que tem um valor: com "35"
    escrito, clicar no fim e digitar tentaria formar "357", que o
    QIntValidator (1 a 95) recusa. Selecionando tudo ao focar, a tecla
    digitada substitui o valor em vez de tentar acrescentar.
    """

    def focusInEvent(self, ev):
        super().focusInEvent(ev)
        # Adiado um ciclo do loop de eventos de proposito: o clique do
        # mouse posiciona o cursor DEPOIS do focusInEvent e desfaria a
        # selecao se ela fosse feita aqui direto.
        QTimer.singleShot(0, self.selectAll)


class BotaoIcone(QPushButton):
    """Botao pequeno so com icone (fechar card, minimizar, fechar janela)."""

    def __init__(self, icone, tamanho=28, nome="botaoJanela", parent=None):
        super().__init__(parent)
        self.setObjectName(nome)
        self.setFixedSize(tamanho, tamanho)
        self.setCursor(Qt.PointingHandCursor)
        self._icone = icone

    def paintEvent(self, ev):
        super().paintEvent(ev)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        self._icone(p, QRectF(0, 0, self.width(), self.height()), COR_ICONE)
        p.end()


class CardArquivo(QFrame):
    """Substitui o botao 'Carregar Arquivo' quando ha um arquivo carregado."""

    fechado = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("cardArquivo")

        col = QVBoxLayout(self)
        col.setContentsMargins(13, 8, 8, 10)
        col.setSpacing(8)

        topo = QHBoxLayout()
        topo.setContentsMargins(0, 0, 0, 0)
        topo.setSpacing(4)
        self.lbl_nome = QLabel()
        self.lbl_nome.setObjectName("nomeArquivo")
        botao = BotaoIcone(icone_x, 22, "botaoFecharCard")
        botao.clicked.connect(self.fechado.emit)
        topo.addWidget(self.lbl_nome, 1)
        topo.addWidget(botao, 0, Qt.AlignTop)
        col.addLayout(topo)

        self.lbl_info = QLabel()
        self.lbl_info.setObjectName("infoArquivo")
        self.lbl_info.setWordWrap(True)
        col.addWidget(self.lbl_info)

    def preencher(self, nome: str, info: str) -> None:
        metrica = self.lbl_nome.fontMetrics()
        self.lbl_nome.setText(metrica.elidedText(nome, Qt.ElideMiddle, 200))
        self.lbl_nome.setToolTip(nome)
        self.lbl_info.setText(info)


# ==========================================================================
# Execucao em thread separada
#
# A MLP esta com max_iter=10000 e o "Executar Todos" roda tudo em sequencia.
# Na thread da interface a janela congelaria e pareceria travada.
# ==========================================================================
class Trabalhador(QObject):
    # Emite o nome junto do texto para a interface conseguir rotular a
    # linha do historico.
    saiu = Signal(str, str)
    terminou = Signal()

    def __init__(self, dados, pct, nomes, usar_padrao):
        super().__init__()
        self._dados = dados
        self._pct = pct
        self._nomes = nomes
        self._usar_padrao = usar_padrao

    def rodar(self) -> None:
        for nome in self._nomes:
            try:
                texto = runner.executar(nome, self._dados, self._pct, self._usar_padrao)
            except Exception as erro:  # nao derruba os metodos seguintes
                texto = "\n---------------- " + nome + "\nFalhou: " + str(erro) + "\n"
            self.saiu.emit(nome, texto)
        self.terminou.emit()


# ==========================================================================
# Janela
# ==========================================================================
class Janela(QWidget):
    BORDA = 6  # faixa sensivel para redimensionar

    def __init__(self):
        super().__init__()
        self.setWindowTitle("myWeka")
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setMouseTracking(True)
        self.resize(JANELA_L, JANELA_A)
        self.setMinimumSize(JANELA_MIN_L, JANELA_MIN_A)

        self._dados = None
        self._thread = None
        self._trab = None
        self._rotulo_lote = None
        self._textos_lote = []
        self._mover_de = None
        self._redim_de = None

        self._montar()
        self._estado_sem_arquivo()

        # Se o arquivo de ML nem importou, a janela abre mesmo assim e
        # explica o motivo, em vez de o app sumir com um traceback.
        erro = runner.erro_import()
        if erro is not None:
            self.saida.setPlainText(
                "Não consegui carregar o ml_clas_sk.py:\n\n"
                + str(erro)
                + "\n\nSe a mensagem acima fala em 'Controle de Aplicativo', é o\n"
                "Windows bloqueando temporariamente as bibliotecas enquanto\n"
                "consulta a reputação delas. Clique em Carregar Arquivo para\n"
                "tentar de novo."
            )

    # ---------------------------------------------------------------- montagem
    def _montar(self) -> None:
        fora = QVBoxLayout(self)
        fora.setContentsMargins(0, 0, 0, 0)

        raiz = QFrame()
        raiz.setObjectName("raiz")
        fora.addWidget(raiz)

        # Em cima a barra de titulo (abas + minimizar/fechar); embaixo,
        # painel da esquerda e caixa de saida.
        pilha = QVBoxLayout(raiz)
        pilha.setContentsMargins(0, 0, 0, 0)
        pilha.setSpacing(0)
        pilha.addWidget(self._montar_barra_titulo())

        corpo = QFrame()
        corpo.setObjectName("corpo")
        linha = QHBoxLayout(corpo)
        linha.setContentsMargins(0, ESP_ABAS_CONTEUDO, MARGEM, MARGEM)
        linha.setSpacing(0)
        linha.addWidget(self._montar_painel())
        linha.addWidget(self._montar_saida(), 1)
        pilha.addWidget(corpo, 1)

    def _montar_barra_titulo(self) -> QFrame:
        """As abas Classificacao/Regressao e os botoes da janela.

        A aba ativa recebe a cor do fundo da janela, entao ela se funde
        com o conteudo de baixo, como no mock.
        """
        barra = QFrame()
        barra.setObjectName("barraTitulo")
        barra.setFixedHeight(BARRA_ABAS_A)

        linha = QHBoxLayout(barra)
        linha.setContentsMargins(4, 0, MARGEM, 0)
        linha.setSpacing(0)

        self.abas = BarraAbas([runner.CLASSIFICACAO, runner.REGRESSAO])
        self.abas.mudou.connect(self._trocar_tipo)
        linha.addWidget(self.abas)
        linha.addStretch(1)

        b_min = BotaoIcone(icone_menos, 28)
        b_min.clicked.connect(self.showMinimized)
        b_fechar = BotaoIcone(icone_x, 28, "botaoFechar")
        b_fechar.clicked.connect(self.close)
        linha.addWidget(b_min)
        linha.addSpacing(18)
        linha.addWidget(b_fechar)
        return barra

    def _montar_painel(self) -> QFrame:
        painel = QFrame()
        painel.setObjectName("painel")
        painel.setFixedWidth(PAINEL_L)

        col = QVBoxLayout(painel)
        col.setContentsMargins(MARGEM, 0, MARGEM - 2, 0)
        col.setSpacing(ESP)

        if MOSTRAR_LOGO:
            col.addWidget(self._montar_logo(), 0, Qt.AlignHCenter)

        # --- carregar arquivo: [icone][rotulo] -------------------------
        self.btn_carregar = Caixa()
        self.btn_carregar.celula(CelulaIcone(icone_documento))
        self.btn_carregar.rotulo("Carregar Arquivo")
        self.btn_carregar.clicked.connect(self._abrir_arquivo)
        col.addWidget(self.btn_carregar)

        self.card = CardArquivo()
        self.card.fechado.connect(self._descarregar)
        col.addWidget(self.card)

        # --- selecionar metodo: [icone][combo] -------------------------
        caixa_metodo = Caixa()
        caixa_metodo.celula(CelulaIcone(icone_lista))
        self.combo = ComboMetodo()
        self.combo.addItems(runner.nomes_metodos())
        self.combo.setPlaceholderText("Selecionar Método")
        self.combo.setCurrentIndex(-1)
        self.combo.currentTextChanged.connect(self._atualizar_rotulo_executar)
        caixa_metodo.celula(self.combo, expande=True)
        # Clicar na celula do icone tambem abre a lista.
        caixa_metodo.clicked.connect(self.combo.showPopup)
        col.addWidget(caixa_metodo)

        # --- parametros padrao: [indicador][rotulo] --------------------
        self.btn_padrao = Caixa()
        self.btn_padrao.setCheckable(True)
        self._luz_padrao = self.btn_padrao.celula(
            CelulaIcone(lambda p, r, c: icone_alvo(p, r, c, self.btn_padrao.isChecked()))
        )
        self.btn_padrao.rotulo("Parâmetros Padrão")
        self.btn_padrao.setToolTip(AJUDA_PADRAO)
        self.btn_padrao.toggled.connect(self._alternar_padrao)
        col.addWidget(self.btn_padrao)

        # --- modificar parametros: [rotulo][icone] ---------------------
        # Unico com o icone a direita, como no desenho.
        self.btn_config = Caixa()
        self.btn_config.rotulo("Modificar Parâmetros")
        self.btn_config.celula(CelulaIcone(icone_abrir))
        self.btn_config.clicked.connect(self._abrir_config)
        col.addWidget(self.btn_config)

        # --- %Teste: [rotulo][campo] -----------------------------------
        col.addWidget(self._montar_teste())

        # --- executar: caixas em destaque, sem icone -------------------
        self.btn_metodo = Caixa(destaque=True)
        self._rotulo_metodo = self.btn_metodo.rotulo(
            'Executar Método', "celulaRotuloCentro"
        )
        self.btn_metodo.clicked.connect(self._executar_um)
        col.addWidget(self.btn_metodo)

        self.btn_todos = Caixa(destaque=True)
        self.btn_todos.rotulo("Executar Todos Métodos", "celulaRotuloCentro")
        self.btn_todos.clicked.connect(self._executar_todos)
        col.addWidget(self.btn_todos)

        # O historico ocupa o espaco que sobra e empurra a identificacao
        # para o rodape.
        col.addWidget(self._montar_historico(), 1)
        col.addWidget(self._montar_card_info())
        return painel

    def _montar_historico(self) -> QFrame:
        """Lista das execucoes desta sessao.

        Mesma construcao da Caixa: fundo na cor da borda e celulas
        opacas por cima, entao a faixa do titulo fica separada da lista
        por uma divisoria de 1px.
        """
        caixa = QFrame()
        caixa.setObjectName("caixaHistorico")

        col = QVBoxLayout(caixa)
        col.setContentsMargins(1, 1, 1, 1)
        col.setSpacing(1)

        titulo = QLabel("Histórico")
        titulo.setObjectName("tituloHistorico")
        titulo.setFixedHeight(CONTROLE_A - 2)
        col.addWidget(titulo)

        self.historico = QListWidget()
        self.historico.setObjectName("historico")
        # O texto de cada execucao fica guardado no proprio item, entao
        # clicar reexibe o resultado sem treinar de novo.
        self.historico.itemClicked.connect(self._mostrar_do_historico)
        self.historico.setMinimumHeight(ALTURA_MIN_HISTORICO)
        col.addWidget(self.historico, 1)
        return caixa

    def _montar_card_info(self) -> QFrame:
        """Card de rodape com a identificacao do trabalho.

        Usa o mesmo estilo do card de arquivo: no style.qss as duas
        regras sao compartilhadas (#cardArquivo, #cardInfo).
        """
        card = QFrame()
        card.setObjectName("cardInfo")

        col = QVBoxLayout(card)
        col.setContentsMargins(13, 9, 13, 11)
        col.setSpacing(6)

        titulo = QLabel("Trabalho de Inteligência Artificial II")
        titulo.setObjectName("tituloInfo")
        titulo.setWordWrap(True)
        col.addWidget(titulo)

        autores = QLabel("Desenvolvido por Alexandre Marciano e Davi de Castro")
        autores.setObjectName("textoInfo")
        autores.setWordWrap(True)
        col.addWidget(autores)
        return card

    def _separador(self) -> QFrame:
        """A linha fina de 1px que divide os blocos do painel."""
        sep = QFrame()
        sep.setObjectName("separador")
        sep.setFixedHeight(1)
        return sep

    def _montar_logo(self) -> QLabel:
        """A logo no espaco livre acima do Carregar Arquivo."""
        rotulo = QLabel()
        rotulo.setObjectName("logo")
        rotulo.setAlignment(Qt.AlignCenter)

        caminho = os.path.join(PASTA, "logo.png")
        imagem = QPixmap(caminho)
        if not imagem.isNull():
            # scaledToWidth preserva a proporcao sozinho; sem o
            # SmoothTransformation a reducao fica serrilhada.
            rotulo.setPixmap(imagem.scaledToWidth(LOGO_L, Qt.SmoothTransformation))
        # Se o logo.png nao existir, o rotulo fica vazio em vez de quebrar.
        return rotulo

    def _montar_teste(self) -> Caixa:
        """Caixa do %Teste: [rotulo][campo editavel]."""
        caixa = Caixa()
        caixa.setCursor(Qt.ArrowCursor)

        rotulo = QLabel("% Teste:")
        rotulo.setObjectName("celulaRotulo")
        caixa.celula(rotulo, largura=CELULA_TESTE_L - 2)

        # Nasce vazio: a porcentagem e uma escolha do usuario, nao um
        # valor que a interface chuta. Sem setMaxLength, que descartava a
        # tecla antes mesmo de o validador ser consultado.
        self.campo_teste = CampoNumero()
        self.campo_teste.setObjectName("valorTeste")
        self.campo_teste.setValidator(QIntValidator(1, 95, self))
        self.campo_teste.editingFinished.connect(self._ajustar_teste)
        caixa.celula(self.campo_teste, expande=True)

        # Clicar em qualquer ponto da caixa leva o foco para o campo.
        caixa.clicked.connect(self.campo_teste.setFocus)
        return caixa

    def _montar_saida(self) -> QFrame:
        """Caixa da direita: abas Dados/Resultados sobre o conteudo."""
        area = QFrame()
        area.setObjectName("areaSaida")

        col = QVBoxLayout(area)
        col.setContentsMargins(1, 1, 1, 1)
        col.setSpacing(0)

        self.abas_saida = BarraAbas(["Dados", "Resultados"], "abaSaida")
        # Nome proprio porque esta faixa tem fundo (#252525), ao contrario
        # da barra do topo, que e transparente sobre a barra de titulo.
        self.abas_saida.setObjectName("barraAbasSaida")
        self.abas_saida.setProperty("linhaInferior", True)
        self.abas_saida.setFixedHeight(CONTROLE_A)
        col.addWidget(self.abas_saida)

        self.pilha_saida = QStackedWidget()
        self.pilha_saida.setObjectName("pilhaSaida")

        # pagina 0: Dados -- o conteudo do arquivo carregado
        self.tabela = QTextEdit()
        self.tabela.setObjectName("saida")
        self.tabela.setReadOnly(True)
        self.tabela.setLineWrapMode(QTextEdit.NoWrap)
        self.tabela.setPlaceholderText("Carregue um arquivo para ver os dados.")
        self.pilha_saida.addWidget(self.tabela)

        # pagina 1: Resultados
        self.saida = QTextEdit()
        self.saida.setObjectName("saida")
        self.saida.setReadOnly(True)
        # Quebra na largura da janela e rola so na vertical. O minimo da
        # janela (JANELA_MIN_L) garante colunas de sobra, entao a matriz
        # de confusao nao chega a ser quebrada na pratica.
        self.saida.setLineWrapMode(QTextEdit.WidgetWidth)
        self.saida.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # Curto de proposito: o placeholder do QTextEdit nao quebra linha,
        # entao um texto longo apareceria cortado na borda.
        self.saida.setPlaceholderText("Carregue um arquivo e execute um método.")
        self.pilha_saida.addWidget(self.saida)

        col.addWidget(self.pilha_saida, 1)

        # Só agora que a pilha existe: a aba começa em Resultados.
        self.abas_saida.mudou.connect(self._trocar_painel_saida)
        self.abas_saida._escolher("Resultados")
        return area

    def _trocar_painel_saida(self, nome: str) -> None:
        self.pilha_saida.setCurrentIndex(0 if nome == "Dados" else 1)

    def _atualizar_rotulo_executar(self, *_) -> None:
        """O botao mostra o metodo escolhido, como no desenho."""
        nome = self.combo.currentText()
        self._rotulo_metodo.setText(
            f'Executar "{nome}"' if nome else "Executar Método"
        )

    def _mostrar_dados(self) -> None:
        """Preenche a aba Dados com o conteudo do arquivo carregado.

        Lista o arquivo inteiro, sem cortar: quem abre esta aba quer
        conferir a base, e uma previa cortada esconde justamente as
        linhas do fim, que sao as que costumam estar erradas.
        """
        if self._dados is None:
            self.tabela.clear()
            return

        d = self._dados
        rotulo_alvo = "Alvo" if d.tipo == runner.REGRESSAO else "Classe"
        colunas = [f"Atr {i + 1}" for i in range(d.n_atributos)] + [rotulo_alvo]

        linhas = [d.resumo(), ""]
        larg = 12
        linhas.append("".join(c.rjust(larg) for c in colunas))
        linhas.append("-" * (larg * len(colunas)))

        for i in range(d.n_instancias):
            valores = [f"{v:g}" for v in d.feat[i]]
            if d.tipo == runner.REGRESSAO:
                valores.append(f"{d.cl[i]:g}")
            else:
                valores.append(str(d.classes[d.cl[i]]))
            linhas.append("".join(v.rjust(larg) for v in valores))

        self.tabela.setPlainText("\n".join(linhas))

    # ---------------------------------------------------------------- estados
    def _estado_sem_arquivo(self) -> None:
        self._dados = None
        if hasattr(self, "tabela"):
            self.tabela.clear()
        self.btn_carregar.setVisible(True)
        self.card.setVisible(False)
        self._habilitar(False)

    def _habilitar(self, ok: bool) -> None:
        """Liga/desliga o que depende de ter uma base carregada.

        Só os dois botoes de executar entram aqui. O %Teste e o dropdown
        de metodo sao configuracao, nao acao: ficam sempre editaveis,
        senao o usuario nem consegue preenche-los antes de abrir a base.
        """
        for w in (self.btn_metodo, self.btn_todos):
            w.setEnabled(ok)

    # ---------------------------------------------------------------- arquivo
    def _abrir_arquivo(self) -> None:
        # Sem DontUseNativeDialog: abre o seletor nativo do Windows.
        caminho, _ = QFileDialog.getOpenFileName(
            self,
            "Selecionar base de dados",
            "",
            "Dados (*.txt *.csv *.data);;Todos os arquivos (*)",
        )
        if not caminho:
            return

        # Descarta um erro de import anterior: se foi o bloqueio transitorio
        # do Windows, esta tentativa pode funcionar.
        runner.esquecer_erro()
        try:
            self._dados = runner.carregar(caminho)
        except Exception as erro:
            self._estado_sem_arquivo()
            self.saida.setPlainText(str(erro))
            return

        self._atualizar_metodos()
        self._mostrar_dados()
        self.card.preencher(self._dados.nome, self._dados.resumo())
        self.btn_carregar.setVisible(False)
        self.card.setVisible(True)
        self._habilitar(True)
        self.saida.clear()

    def _atualizar_metodos(self) -> None:
        """Repovoa o dropdown se ele nasceu vazio.

        Acontece quando o ml_clas_sk.py nao pode ser importado no inicio
        (um erro no arquivo, ou o bloqueio transitorio do Windows). Depois
        que o import funciona, a lista precisa aparecer.
        """
        if self.combo.count() == 0:
            self.combo.addItems(runner.nomes_metodos())
            self.combo.setCurrentIndex(-1)

    def _descarregar(self) -> None:
        self._estado_sem_arquivo()
        self.saida.clear()

    # ---------------------------------------------------------------- abas
    def _trocar_tipo(self, tipo: str) -> None:
        """Alterna entre classificacao e regressao.

        A base carregada e descartada de proposito: uma base de
        classificacao tem a classe em texto na ultima coluna e nao serve
        para regressao, e vice-versa.
        """
        if self._thread is not None:
            return
        runner.definir_tipo(tipo)
        runner.esquecer_erro()

        self.combo.clear()
        self.combo.addItems(runner.nomes_metodos())
        self.combo.setCurrentIndex(-1)
        self._atualizar_rotulo_executar()

        self._estado_sem_arquivo()
        self.saida.clear()
        erro = runner.erro_import()
        if erro is not None:
            self.saida.setPlainText(
                f"Não consegui carregar o módulo de {tipo}:\n\n{erro}"
            )

    # ---------------------------------------------------------------- padrao
    def _alternar_padrao(self, ligado: bool) -> None:
        """O botao so guarda a escolha; quem aplica e o runner, na execucao."""
        self.btn_config.setEnabled(not ligado)
        self.btn_config.setToolTip(
            "Desligue os Parâmetros Padrão para editar os seus valores."
            if ligado
            else ""
        )

    # ---------------------------------------------------------------- historico
    def _registrar(self, metodo: str, texto: str) -> None:
        """Guarda uma execucao no historico, com o resultado junto."""
        if self._dados is None:
            return
        item = QListWidgetItem(f"{metodo}, {self._dados.nome}")
        item.setData(Qt.UserRole, texto)
        self.historico.addItem(item)
        self.historico.setCurrentItem(item)

    def _mostrar_do_historico(self, item: QListWidgetItem) -> None:
        """Reexibe o resultado guardado, sem treinar de novo."""
        self.saida.setPlainText(item.data(Qt.UserRole) or "")

    # ---------------------------------------------------------------- config
    def _abrir_config(self) -> None:
        """Abre o arquivo de hiperparametros no Bloco de Notas.

        O arquivo e relido a cada execucao pelo modulo de ML, entao basta
        salvar no Bloco de Notas e clicar em Executar de novo.
        """
        # O nome vem do proprio modulo de ML, entao acompanha a aba ativa
        # (config_cl.txt na classificacao, config_reg.txt na regressao).
        arquivo = runner.arquivo_personalizado()
        caminho = os.path.join(PASTA, arquivo) if arquivo else ""
        if not arquivo or not os.path.exists(caminho):
            self.saida.setPlainText(
                "Não encontrei o arquivo de configuração:\n\n"
                + (caminho or "(o módulo de ML não informou o nome)")
                + "\n\nEle precisa estar na mesma pasta do programa."
            )
            return
        try:
            # notepad.exe explicito, e nao os.startfile: este ultimo abriria
            # o editor padrao do sistema, que pode nao ser o Bloco de Notas.
            subprocess.Popen(["notepad.exe", caminho])
        except OSError as erro:
            self.saida.setPlainText(
                "Não consegui abrir o Bloco de Notas:\n\n"
                + str(erro)
                + "\n\nAbra manualmente: "
                + caminho
            )

    # ---------------------------------------------------------------- %Teste
    def _pct(self):
        """A % digitada, ja limitada a 1-95, ou None se o campo esta vazio.

        Devolver None em vez de um padrao e proposital: como o campo
        nasce vazio, chutar 20 faria o usuario receber um resultado de
        20% sem nunca ter escolhido isso.
        """
        texto = self.campo_teste.text().strip()
        if not texto:
            return None
        try:
            return min(max(int(texto), 1), 95)
        except ValueError:
            return None

    def _somar_teste(self, delta: int) -> None:
        atual = self._pct()
        if atual is None:
            # Campo vazio nao tem de onde somar: a primeira seta semeia
            # 20, a mesma proporcao do test_size=0.2 do ml_clas_sk.py.
            self.campo_teste.setText("20")
            return
        self.campo_teste.setText(str(min(max(atual + delta, 1), 95)))

    def _ajustar_teste(self) -> None:
        """Normaliza o valor quando o usuario sai do campo.

        O validador deixa passar estados intermediarios ("96", "0"), que
        precisam ser ajustados. Campo vazio continua vazio.
        """
        pct = self._pct()
        if pct is not None:
            self.campo_teste.setText(str(pct))

    # ---------------------------------------------------------------- execucao
    def _executar_um(self) -> None:
        if self.combo.currentIndex() < 0:
            self.saida.setPlainText("Escolha um método no menu acima.")
            return
        self._executar([self.combo.currentText()])

    def _executar_todos(self) -> None:
        # Um rotulo de lote: os tres metodos viram UMA entrada no
        # historico, com os resultados juntos, em vez de tres separadas.
        self._executar(runner.nomes_metodos(), "Todos os métodos")

    def _executar(self, nomes, rotulo_lote=None) -> None:
        if self._dados is None or self._thread is not None:
            return

        pct = self._pct()
        if pct is None:
            self.saida.setPlainText("Informe a porcentagem de teste no campo %Teste.")
            return

        self.saida.clear()
        self._habilitar(False)
        self.btn_carregar.setEnabled(False)
        self.card.setEnabled(False)

        # Com rotulo de lote os resultados sao acumulados e viram uma
        # entrada so no historico, registrada no fim.
        self._rotulo_lote = rotulo_lote
        self._textos_lote = []

        self._thread = QThread(self)
        self._trab = Trabalhador(self._dados, pct, nomes, self.btn_padrao.isChecked())
        self._trab.moveToThread(self._thread)
        self._thread.started.connect(self._trab.rodar)
        self._trab.saiu.connect(self._receber)
        self._trab.terminou.connect(self._encerrar)
        self._thread.start()

    def _receber(self, metodo: str, texto: str) -> None:
        """Mostra o resultado e alimenta o historico."""
        self._anexar(texto)
        if self._rotulo_lote is None:
            self._registrar(metodo, texto)
        else:
            self._textos_lote.append(texto)   # vira uma entrada so no fim

    def _anexar(self, texto: str) -> None:
        cursor = self.saida.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(texto)
        self.saida.setTextCursor(cursor)

    def _encerrar(self) -> None:
        # Fim de um "Executar Todos": agora que todos os metodos
        # terminaram, o lote inteiro vira uma unica linha no historico.
        if self._rotulo_lote is not None and self._textos_lote:
            self._registrar(self._rotulo_lote, "".join(self._textos_lote))
        self._rotulo_lote = None
        self._textos_lote = []

        self._thread.quit()
        self._thread.wait()
        self._thread.deleteLater()
        self._thread = None
        self._trab = None
        self._habilitar(True)
        self.btn_carregar.setEnabled(True)
        self.card.setEnabled(True)

    def closeEvent(self, ev):
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)
        super().closeEvent(ev)

    # ---------------------------------------------------------------- mouse
    # Janela sem moldura: arrastar e redimensionar viram responsabilidade nossa.
    def _zona(self, pos) -> str:
        direita = pos.x() >= self.width() - self.BORDA
        baixo = pos.y() >= self.height() - self.BORDA
        if direita and baixo:
            return "canto"
        if direita:
            return "direita"
        if baixo:
            return "baixo"
        return ""

    def mousePressEvent(self, ev):
        if ev.button() != Qt.LeftButton:
            return
        zona = self._zona(ev.position().toPoint())
        if zona:
            self._redim_de = (zona, ev.globalPosition().toPoint(), self.size())
        else:
            self._mover_de = ev.globalPosition().toPoint() - self.frameGeometry().topLeft()
        ev.accept()

    def mouseMoveEvent(self, ev):
        if self._redim_de is not None:
            zona, inicio, tam = self._redim_de
            d = ev.globalPosition().toPoint() - inicio
            larg = tam.width() + (d.x() if zona in ("direita", "canto") else 0)
            alt = tam.height() + (d.y() if zona in ("baixo", "canto") else 0)
            self.resize(max(larg, self.minimumWidth()), max(alt, self.minimumHeight()))
        elif self._mover_de is not None and ev.buttons() & Qt.LeftButton:
            self.move(ev.globalPosition().toPoint() - self._mover_de)
        else:
            zona = self._zona(ev.position().toPoint())
            self.setCursor(
                {
                    "canto": Qt.SizeFDiagCursor,
                    "direita": Qt.SizeHorCursor,
                    "baixo": Qt.SizeVerCursor,
                }.get(zona, Qt.ArrowCursor)
            )
        ev.accept()

    def mouseReleaseEvent(self, ev):
        self._mover_de = None
        self._redim_de = None
        ev.accept()


# ==========================================================================
def carregar_fontes() -> None:
    """Registra a Source Serif 4 que vem embutida na pasta fontes/.

    A fonte vai junto do projeto em vez de depender de instalacao, entao
    a interface sai igual em qualquer maquina. Se os arquivos sumirem, o
    style.qss cai na Georgia sozinho e nada quebra.
    """
    for arquivo in ("SourceSerif4-Regular.otf", "SourceSerif4-Semibold.otf"):
        QFontDatabase.addApplicationFont(os.path.join(PASTA, "fontes", arquivo))


def main() -> int:
    app = QApplication(sys.argv)
    carregar_fontes()

    # Sem hinting: o Qt deixa de encaixar as hastes das letras na grade de
    # pixels. Em fonte serifada o encaixe engrossa hastes finas e remates
    # de forma desigual, e e isso que deixa o texto com aspecto granulado.
    # Este e o mesmo modo de renderizacao que o Photoshop usa, entao a tela
    # fica parecida com o mock.
    fonte = app.font()
    fonte.setHintingPreference(QFont.PreferNoHinting)
    app.setFont(fonte)

    folha = os.path.join(PASTA, "style.qss")
    if os.path.exists(folha):
        with open(folha, encoding="utf-8") as f:
            app.setStyleSheet(f.read())

    janela = Janela()
    janela.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

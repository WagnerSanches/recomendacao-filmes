"""Configurações do projeto.

A conexão com o Neo4j vem de variáveis de ambiente ou do arquivo .env (não versionado).
Copie .env.example para .env e preencha a senha.
"""
import os

_RAIZ = os.path.dirname(os.path.abspath(__file__))


def _carregar_env(caminho):
    """Lê um .env simples (CHAVE=valor). Variáveis já definidas no ambiente têm prioridade."""
    if not os.path.exists(caminho):
        return
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if linha and not linha.startswith("#") and "=" in linha:
                chave, valor = linha.split("=", 1)
                os.environ.setdefault(chave.strip(), valor.strip().strip('"').strip("'"))


_carregar_env(os.path.join(_RAIZ, ".env"))

# Conexão com o Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Onde ficam ratings.csv/movies.csv (ou rating.csv/movie.csv do Kaggle)
DATA_DIR = os.path.join(_RAIZ, "data")

# Subconjunto de usuários (None = todos). O dataset de 20M (138 mil usuários) é grande demais
# para Python puro; 2000 usuários dão ~290 mil notas.
MAX_USERS = 2000

# Divisão treino/teste (feita por usuário: 80% das notas de cada usuário vão para treino)
TEST_RATIO = 0.2
SEED = 42

# Similaridade (correlação de Pearson entre filmes)
MIN_RATINGS_MOVIE = 20    # filmes com menos notas de treino não recebem similaridade
MAX_MOVIES = 2000         # usa só os N filmes mais avaliados (None = sem limite)
MIN_COMMON_USERS = 10     # mínimo de usuários em comum para calcular a correlação
MIN_CORR = 0.0            # só grava arestas SIMILAR com correlação acima deste valor
SHRINKAGE_BETA = 100      # corr = pearson · n / (n + β); 0 desliga o shrinkage

# KNN
K = 20                    # número de vizinhos usados na predição
MIN_NEIGHBORS = 2         # mínimo de vizinhos para fazer uma predição

# Recomendação / avaliação
RELEVANCE_THRESHOLD = 4.0  # nota a partir da qual consideramos que o usuário "gostou"
TOP_N = 10

BATCH_SIZE = 10000

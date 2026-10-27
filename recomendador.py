"""Funções compartilhadas: leitura dos CSVs, divisão treino/teste, Pearson e predição KNN."""
import csv
import math
import os
import random
import re
from collections import defaultdict
from datetime import datetime, timezone

from neo4j import GraphDatabase

import config


def conectar():
    if not config.NEO4J_PASSWORD:
        raise SystemExit("Defina NEO4J_PASSWORD no arquivo .env (veja .env.example) ou como variável de ambiente.")
    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
    driver.verify_connectivity()
    return driver


def consultar(driver, query, **params):
    registros, _, _ = driver.execute_query(query, params, database_=config.NEO4J_DATABASE)
    return registros


def em_lotes(itens, tamanho=config.BATCH_SIZE):
    for inicio in range(0, len(itens), tamanho):
        yield itens[inicio:inicio + tamanho]


# ---------------------------------------------------------------- leitura dos CSVs

_ANO = re.compile(r"\s*\((\d{4})\)\s*$")


def caminho_csv(*nomes):
    """Aceita tanto ratings.csv/movies.csv (GroupLens) quanto rating.csv/movie.csv (Kaggle)."""
    for nome in nomes:
        caminho = os.path.join(config.DATA_DIR, nome)
        if os.path.exists(caminho):
            return caminho
    raise FileNotFoundError(f"Nenhum de {nomes} encontrado em {config.DATA_DIR}")


def converter_timestamp(valor):
    """GroupLens usa epoch (1112486027); Kaggle usa data ('2005-04-02 23:53:47')."""
    if valor.isdigit():
        return int(valor)
    return int(datetime.strptime(valor, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())


def ler_filmes():
    filmes = []
    with open(caminho_csv("movies.csv", "movie.csv"), encoding="utf-8", newline="") as f:
        for linha in csv.DictReader(f):
            titulo = linha["title"].strip()
            ano = None
            achou = _ANO.search(titulo)
            if achou:
                ano = int(achou.group(1))
                titulo = titulo[:achou.start()]
            generos = [g for g in linha["genres"].split("|") if g and g != "(no genres listed)"]
            filmes.append({"movieId": int(linha["movieId"]), "title": titulo, "year": ano, "genres": generos})
    return filmes


def ler_notas():
    """Lê as notas. Com MAX_USERS, mantém só os primeiros N usuários (o arquivo vem ordenado
    por userId), filtrando durante a leitura para não carregar 20M de linhas na memória."""
    notas = []
    usuarios = set()
    with open(caminho_csv("ratings.csv", "rating.csv"), encoding="utf-8", newline="") as f:
        for linha in csv.DictReader(f):
            usuario = int(linha["userId"])
            if config.MAX_USERS and usuario not in usuarios:
                if len(usuarios) >= config.MAX_USERS:
                    break
                usuarios.add(usuario)
            notas.append({
                "userId": usuario,
                "movieId": int(linha["movieId"]),
                "rating": float(linha["rating"]),
                "timestamp": converter_timestamp(linha["timestamp"]),
            })
    return notas


def dividir_treino_teste(notas):
    """Marca cada nota com split='train' ou 'test'. A divisão é por usuário, para que todo
    usuário do teste também tenha histórico no treino."""
    rng = random.Random(config.SEED)
    por_usuario = defaultdict(list)
    for n in notas:
        por_usuario[n["userId"]].append(n)
    for lista in por_usuario.values():
        rng.shuffle(lista)
        qtd_teste = int(len(lista) * config.TEST_RATIO)
        for i, n in enumerate(lista):
            n["split"] = "test" if i < qtd_teste else "train"
    return notas


# ---------------------------------------------------------------- correlação de Pearson

def calcular_pearson(notas_treino):
    """notas_treino: lista de (userId, movieId, rating).
    Retorna lista de (movie_i, movie_j, corr, usuarios_em_comum, pearson) com i < j, onde
    corr = pearson · n / (n + β) é a correlação com shrinkage usada no KNN."""
    por_filme = defaultdict(list)
    for _, m, r in notas_treino:
        por_filme[m].append(r)

    elegiveis = [m for m, rs in por_filme.items() if len(rs) >= config.MIN_RATINGS_MOVIE]
    elegiveis.sort(key=lambda m: len(por_filme[m]), reverse=True)
    if config.MAX_MOVIES:
        elegiveis = elegiveis[:config.MAX_MOVIES]
    elegiveis = set(elegiveis)

    # r̄_i: média de todas as notas de treino do filme i
    media = {m: sum(por_filme[m]) / len(por_filme[m]) for m in elegiveis}

    # desvios (r_i^k - r̄_i) agrupados por usuário
    desvios = defaultdict(list)
    for u, m, r in notas_treino:
        if m in elegiveis:
            desvios[u].append((m, r - media[m]))

    # acumuladores por par: [soma d_i*d_j, soma d_i², soma d_j², n]
    acumulado = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    for lista in desvios.values():
        lista.sort()
        for a in range(len(lista)):
            mi, di = lista[a]
            for b in range(a + 1, len(lista)):
                mj, dj = lista[b]
                acc = acumulado[(mi, mj)]
                acc[0] += di * dj
                acc[1] += di * di
                acc[2] += dj * dj
                acc[3] += 1

    resultado = []
    for (mi, mj), (soma_prod, soma_i, soma_j, n) in acumulado.items():
        if n < config.MIN_COMMON_USERS or soma_i == 0 or soma_j == 0:
            continue
        c = soma_prod / (math.sqrt(soma_i) * math.sqrt(soma_j))
        if c > config.MIN_CORR:
            # shrinkage: correlações com poucos usuários em comum perdem peso
            ajustada = c * n / (n + config.SHRINKAGE_BETA)
            resultado.append((mi, mj, ajustada, n, c))
    return resultado


# ---------------------------------------------------------------- predição KNN

def prever(filme, notas_usuario, vizinhos):
    """r̂_i^u = Σ c(i,j)·r_j^u / Σ |c(i,j)|, sobre os K filmes j mais similares a i
    que o usuário u avaliou no treino.

    notas_usuario: {movieId: nota} do usuário no treino
    vizinhos: {movieId: {movieId: correlacao}}
    """
    candidatos = [(c, notas_usuario[j]) for j, c in vizinhos.get(filme, {}).items() if j in notas_usuario]
    if len(candidatos) < config.MIN_NEIGHBORS:
        return None
    candidatos.sort(key=lambda x: abs(x[0]), reverse=True)
    knn = candidatos[:config.K]
    denominador = sum(abs(c) for c, _ in knn)
    if denominador == 0:
        return None
    return sum(c * r for c, r in knn) / denominador

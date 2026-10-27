"""Avalia o modelo nos 20% de teste.

Para cada nota de teste (u, i, nota real), prevemos r̂_i^u usando só as notas de treino
e as arestas SIMILAR do grafo. Depois calculamos:

1. Precisão e Recall (classificação): o filme é "recomendado" se r̂ >= limiar e é
   "relevante" se a nota real >= limiar.
       Precisão = recomendados e relevantes / recomendados
       Recall   = recomendados e relevantes / relevantes
   Pares sem predição (sem vizinhos suficientes) contam como não recomendados.

2. Precisão@N e Recall@N (ranking): para cada usuário, ordena seus filmes de teste pela
   nota prevista e pega os N primeiros como recomendação.

3. MAE e RMSE das notas previstas (extra).
"""
import math
from collections import defaultdict

import config
from recomendador import conectar, consultar, prever


def carregar(driver):
    treino = defaultdict(dict)
    teste = []
    for x in consultar(driver, """
        MATCH (u:User)-[r:RATED]->(m:Movie)
        RETURN u.userId AS u, m.movieId AS m, r.rating AS r, r.split AS split
    """):
        if x["split"] == "train":
            treino[x["u"]][x["m"]] = x["r"]
        else:
            teste.append((x["u"], x["m"], x["r"]))

    vizinhos = defaultdict(dict)
    for x in consultar(driver, "MATCH (a:Movie)-[s:SIMILAR]->(b:Movie) RETURN a.movieId AS a, b.movieId AS b, s.corr AS c"):
        vizinhos[x["a"]][x["b"]] = x["c"]
        vizinhos[x["b"]][x["a"]] = x["c"]
    return treino, teste, vizinhos


def main():
    limiar = config.RELEVANCE_THRESHOLD
    with conectar() as driver:
        print("Carregando dados do Neo4j...")
        treino, teste, vizinhos = carregar(driver)
    print(f"{len(teste)} notas de teste")

    vp = fp = fn = 0
    erros_abs = erros_quad = 0.0
    previstos = 0
    por_usuario = defaultdict(list)  # u -> [(previsto, real)]

    for u, m, real in teste:
        previsto = prever(m, treino.get(u, {}), vizinhos)
        relevante = real >= limiar
        recomendado = previsto is not None and previsto >= limiar

        if recomendado and relevante:
            vp += 1
        elif recomendado:
            fp += 1
        elif relevante:
            fn += 1

        if previsto is not None:
            previstos += 1
            erros_abs += abs(previsto - real)
            erros_quad += (previsto - real) ** 2
            por_usuario[u].append((previsto, real))

    precisao = vp / (vp + fp) if vp + fp else 0.0
    recall = vp / (vp + fn) if vp + fn else 0.0
    f1 = 2 * precisao * recall / (precisao + recall) if precisao + recall else 0.0

    # Precisão@N / Recall@N por usuário
    total_relevantes = defaultdict(int)
    for u, _, real in teste:
        if real >= limiar:
            total_relevantes[u] += 1
    precisoes_n, recalls_n = [], []
    for u, pares in por_usuario.items():
        if total_relevantes[u] == 0:
            continue
        top = sorted(pares, key=lambda p: p[0], reverse=True)[:config.TOP_N]
        acertos = sum(1 for _, real in top if real >= limiar)
        precisoes_n.append(acertos / len(top))
        recalls_n.append(acertos / total_relevantes[u])

    print(f"\nCobertura: {previstos}/{len(teste)} notas de teste com predição ({previstos / len(teste):.1%})")
    print(f"\nClassificação (limiar = {limiar})")
    print(f"  Precisão: {precisao:.4f}")
    print(f"  Recall:   {recall:.4f}")
    print(f"  F1:       {f1:.4f}")
    if precisoes_n:
        print(f"\nRanking (N = {config.TOP_N}, média de {len(precisoes_n)} usuários)")
        print(f"  Precisão@{config.TOP_N}: {sum(precisoes_n) / len(precisoes_n):.4f}")
        print(f"  Recall@{config.TOP_N}:   {sum(recalls_n) / len(recalls_n):.4f}")
    if previstos:
        print("\nErro da nota prevista")
        print(f"  MAE:  {erros_abs / previstos:.4f}")
        print(f"  RMSE: {math.sqrt(erros_quad / previstos):.4f}")


if __name__ == "__main__":
    main()

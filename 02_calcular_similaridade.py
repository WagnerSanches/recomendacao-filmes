"""Calcula a correlação de Pearson entre filmes (usando só as notas de treino)
e grava como arestas (:Movie)-[:SIMILAR {corr, pearson, common}]-(:Movie) no Neo4j."""
import time

import config
from recomendador import calcular_pearson, conectar, consultar, em_lotes


def main():
    with conectar() as driver:
        print("Lendo notas de treino do Neo4j...")
        registros = consultar(driver, """
            MATCH (u:User)-[r:RATED {split: 'train'}]->(m:Movie)
            RETURN u.userId AS u, m.movieId AS m, r.rating AS r
        """)
        notas = [(x["u"], x["m"], x["r"]) for x in registros]
        print(f"{len(notas)} notas de treino")

        print("Calculando correlações de Pearson...")
        inicio = time.time()
        pares = calcular_pearson(notas)
        print(f"{len(pares)} pares com correlação > {config.MIN_CORR} ({time.time() - inicio:.0f}s)")

        print("Gravando arestas SIMILAR...")
        while consultar(driver, "MATCH ()-[s:SIMILAR]->() WITH s LIMIT 10000 DELETE s RETURN count(*) AS qtd")[0]["qtd"]:
            pass
        linhas = [{"i": i, "j": j, "c": c, "n": n, "p": p} for i, j, c, n, p in pares]
        for lote in em_lotes(linhas):
            consultar(driver, """
                UNWIND $lote AS p
                MATCH (a:Movie {movieId: p.i})
                MATCH (b:Movie {movieId: p.j})
                CREATE (a)-[:SIMILAR {corr: p.c, pearson: p.p, common: p.n}]->(b)
            """, lote=lote)
    print("Pronto.")


if __name__ == "__main__":
    main()

"""Consultas no grafo.

Uso:
    python 03_consultas.py --usuario 1
    python 03_consultas.py --usuario 1 --genero Comedy --nota 4 --filme 1
"""
import argparse

import config
from recomendador import conectar, consultar


def filmes_por_genero_e_nota(driver, usuario, genero, nota_min):
    """Quantos filmes do gênero X o usuário assistiu e deu nota >= nota_min."""
    return consultar(driver, """
        MATCH (u:User {userId: $usuario})-[r:RATED]->(m:Movie)-[:IN_GENRE]->(:Genre {name: $genero})
        WHERE r.rating >= $nota_min
        RETURN count(m) AS qtd, collect(m.title)[0..10] AS exemplos
    """, usuario=usuario, genero=genero, nota_min=nota_min)[0]


def generos_favoritos(driver, usuario):
    return consultar(driver, """
        MATCH (u:User {userId: $usuario})-[r:RATED]->(:Movie)-[:IN_GENRE]->(g:Genre)
        RETURN g.name AS genero, count(*) AS qtd, round(avg(r.rating), 2) AS media
        ORDER BY qtd DESC LIMIT 5
    """, usuario=usuario)


def filmes_similares(driver, filme):
    return consultar(driver, """
        MATCH (a:Movie {movieId: $filme})-[s:SIMILAR]-(b:Movie)
        RETURN a.title AS filme, b.title AS similar, round(s.corr, 3) AS corr, s.common AS em_comum
        ORDER BY s.corr DESC LIMIT 10
    """, filme=filme)


def top_recomendacoes(driver, usuario, n=config.TOP_N):
    """KNN item-item: para cada filme i que o usuário não viu, pega os K filmes j mais
    similares a i que ele avaliou no treino e calcula Σ c(i,j)·r_j / Σ |c(i,j)|."""
    return consultar(driver, """
        MATCH (u:User {userId: $usuario})-[r:RATED {split: 'train'}]->(j:Movie)-[s:SIMILAR]-(i:Movie)
        WHERE NOT EXISTS { (u)-[:RATED]->(i) }
        WITH i, s.corr AS c, r.rating AS nota
        ORDER BY abs(c) DESC
        WITH i, collect([c, nota])[0..$k] AS knn
        WHERE size(knn) >= $min_vizinhos
        WITH i, size(knn) AS vizinhos,
             reduce(num = 0.0, x IN knn | num + x[0] * x[1]) AS num,
             reduce(den = 0.0, x IN knn | den + abs(x[0])) AS den
        WHERE den > 0
        RETURN i.movieId AS movieId, i.title AS titulo, i.year AS ano,
               round(num / den, 3) AS nota_prevista, vizinhos
        ORDER BY nota_prevista DESC, vizinhos DESC
        LIMIT $n
    """, usuario=usuario, k=config.K, min_vizinhos=config.MIN_NEIGHBORS, n=n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--usuario", type=int, default=1)
    parser.add_argument("--genero", default="Comedy")
    parser.add_argument("--nota", type=float, default=4.0)
    parser.add_argument("--filme", type=int, default=1, help="movieId para listar similares")
    args = parser.parse_args()

    with conectar() as driver:
        r = filmes_por_genero_e_nota(driver, args.usuario, args.genero, args.nota)
        print(f"\nUsuário {args.usuario} assistiu {r['qtd']} filmes de {args.genero} com nota >= {args.nota}")
        for titulo in r["exemplos"]:
            print(f"  - {titulo}")

        print(f"\nGêneros mais assistidos pelo usuário {args.usuario}:")
        for g in generos_favoritos(driver, args.usuario):
            print(f"  {g['genero']:<20} {g['qtd']:>4} filmes  média {g['media']}")

        print(f"\nFilmes mais similares ao filme {args.filme}:")
        for s in filmes_similares(driver, args.filme):
            print(f"  {s['similar']:<50} corr={s['corr']}  em comum={s['em_comum']}")

        print(f"\nTop {config.TOP_N} recomendações para o usuário {args.usuario}:")
        for i, rec in enumerate(top_recomendacoes(driver, args.usuario), start=1):
            print(f"  {i:>2}. {rec['titulo']} ({rec['ano']})  nota prevista {rec['nota_prevista']}  [{rec['vizinhos']} vizinhos]")


if __name__ == "__main__":
    main()

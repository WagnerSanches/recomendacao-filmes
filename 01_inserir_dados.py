"""Lê movies.csv e ratings.csv, separa treino/teste e insere tudo no Neo4j.

Modelo:
    (:User {userId})
    (:Movie {movieId, title, year})
    (:Genre {name})
    (:User)-[:RATED {rating, timestamp, split}]->(:Movie)
    (:Movie)-[:IN_GENRE]->(:Genre)
    (:Movie)-[:SIMILAR {corr, common}]-(:Movie)   <- criada em 02_calcular_similaridade.py

ATENÇÃO: apaga todo o conteúdo do banco antes de inserir.
"""
import config
from recomendador import conectar, consultar, dividir_treino_teste, em_lotes, ler_filmes, ler_notas


def limpar_banco(driver):
    while True:
        apagados = consultar(driver, "MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(*) AS qtd")[0]["qtd"]
        if apagados == 0:
            break


def criar_restricoes(driver):
    consultar(driver, "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.userId IS UNIQUE")
    consultar(driver, "CREATE CONSTRAINT movie_id IF NOT EXISTS FOR (m:Movie) REQUIRE m.movieId IS UNIQUE")
    consultar(driver, "CREATE CONSTRAINT genre_name IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE")


def inserir_filmes(driver, filmes):
    for lote in em_lotes(filmes):
        consultar(driver, """
            UNWIND $lote AS f
            MERGE (m:Movie {movieId: f.movieId})
            SET m.title = f.title, m.year = f.year
            WITH m, f
            UNWIND f.genres AS nome
            MERGE (g:Genre {name: nome})
            MERGE (m)-[:IN_GENRE]->(g)
        """, lote=lote)


def inserir_usuarios(driver, notas):
    usuarios = sorted({n["userId"] for n in notas})
    for lote in em_lotes(usuarios):
        consultar(driver, "UNWIND $lote AS id MERGE (:User {userId: id})", lote=lote)


def inserir_notas(driver, notas):
    total = len(notas)
    for i, lote in enumerate(em_lotes(notas), start=1):
        consultar(driver, """
            UNWIND $lote AS n
            MATCH (u:User {userId: n.userId})
            MATCH (m:Movie {movieId: n.movieId})
            CREATE (u)-[:RATED {rating: n.rating, timestamp: n.timestamp, split: n.split}]->(m)
        """, lote=lote)
        print(f"  notas: {min(i * config.BATCH_SIZE, total)}/{total}", end="\r")
    print()


def main():
    print("Lendo CSVs...")
    filmes = ler_filmes()
    notas = dividir_treino_teste(ler_notas())
    treino = sum(1 for n in notas if n["split"] == "train")
    print(f"{len(filmes)} filmes, {len(notas)} notas ({treino} treino / {len(notas) - treino} teste)")

    with conectar() as driver:
        print("Limpando banco...")
        limpar_banco(driver)
        criar_restricoes(driver)
        print("Inserindo filmes e gêneros...")
        inserir_filmes(driver, filmes)
        print("Inserindo usuários...")
        inserir_usuarios(driver, notas)
        print("Inserindo notas...")
        inserir_notas(driver, notas)
    print("Pronto.")


if __name__ == "__main__":
    main()

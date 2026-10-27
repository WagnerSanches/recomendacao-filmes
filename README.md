# Sistema de Recomendação de Filmes com Grafos (MovieLens + Neo4j)

Filtragem colaborativa item-item: correlação de Pearson entre filmes, gravada como arestas no grafo,
e regressão com KNN para prever a nota que um usuário daria a um filme.

Usa só a biblioteca padrão do Python mais o driver oficial `neo4j`.

## 1) Modelagem do grafo

```
(:User {userId})
    -[:RATED {rating, timestamp, split}]->
(:Movie {movieId, title, year})
    -[:IN_GENRE]->
(:Genre {name})

(:Movie)-[:SIMILAR {corr, common}]-(:Movie)
```

| Elemento | Origem | Atributos |
|---|---|---|
| `User` (nó) | `ratings.csv` | `userId` |
| `Movie` (nó) | `movies.csv` | `movieId`, `title`, `year` (o ano é extraído do título) |
| `Genre` (nó) | coluna `genres` de `movies.csv` | `name` |
| `RATED` (aresta) | `ratings.csv` | `rating`, `timestamp`, `split` (`train`/`test`) |
| `IN_GENRE` (aresta) | coluna `genres` | (nenhum) |
| `SIMILAR` (aresta) | calculada | `corr` (Pearson), `common` (nº de usuários em comum) |

Os gêneros viraram nós, e não uma lista dentro do filme, para permitir consultas como
"comédias que o usuário X avaliou com nota ≥ 4" percorrendo o grafo.

## 2) Modelo

**Correlação de Pearson** entre os filmes *i* e *j*, somando sobre os usuários *k* que avaliaram os dois
(usando só as notas de treino):

$$c(i,j) = \frac{\sum_k (r_i^k - \bar{r}_i)(r_j^k - \bar{r}_j)}{\sqrt{\sum_k (r_i^k - \bar{r}_i)^2}\sqrt{\sum_k (r_j^k - \bar{r}_j)^2}}$$

**KNN:** para prever a nota do usuário *u* no filme *i*, pegamos os *K* filmes *j* mais similares a *i*
que *u* avaliou:

$$\hat{r}_i^u = \frac{\sum_{j \in N_K} c(i,j)\, r_j^u}{\sum_{j \in N_K} |c(i,j)|}$$

Decisões (todas ajustáveis em `config.py`):
- Só são gravadas correlações positivas (`MIN_CORR = 0`). Com pesos negativos e notas brutas,
  a fórmula pode gerar notas fora da escala 0,5–5.
- Um par precisa de pelo menos `MIN_COMMON_USERS` usuários em comum, para evitar correlações
  de 1,0 calculadas com 2 ou 3 pessoas.
- Só os `MAX_MOVIES` filmes mais avaliados (com pelo menos `MIN_RATINGS_MOVIE` notas) entram no cálculo.
  O número de pares cresce com o quadrado da quantidade de filmes.

## 3) Avaliação

Divisão 80/20 **por usuário** (seed fixa): 80% das notas de cada usuário vão para treino. Assim, todo
usuário do teste tem histórico para o KNN usar. O `04_avaliacao.py` mostra:
- **Precisão e Recall:** recomendado = nota prevista ≥ 4; relevante = nota real ≥ 4.
- **Precisão@10 e Recall@10:** para cada usuário, os 10 filmes de teste com maior nota prevista.
- MAE / RMSE e cobertura (% das notas de teste em que foi possível prever).

## Como rodar

1. Instale o [Neo4j Desktop](https://neo4j.com/download/) (ou o Community Server), crie um banco e
   defina a senha. Ajuste em `config.py` ou via variáveis de ambiente
   `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`.
2. Coloque os CSVs em `data/`: `movies.csv`/`ratings.csv` (GroupLens) ou `movie.csv`/`rating.csv` (Kaggle 20M).
   Por padrão, `MAX_USERS = 2000` em `config.py` usa só os 2000 primeiros usuários (~290 mil notas).
   Aumente com cuidado: o tempo do cálculo de Pearson cresce rápido.
3. Instale e execute:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

python 01_inserir_dados.py         # cria nós e arestas (APAGA o banco antes)
python 02_calcular_similaridade.py # Pearson -> arestas SIMILAR
python 03_consultas.py --usuario 1 --genero Comedy --nota 4
python 04_avaliacao.py
```

## O que cada critério de avaliação cobre

| Critério | Onde está |
|---|---|
| Modelagem: nós e arestas | Seção 1 deste README e docstring de `01_inserir_dados.py` |
| Script de inserção no Neo4j em Python | `01_inserir_dados.py` (dados) e `02_calcular_similaridade.py` (arestas `SIMILAR`) |
| Script de consulta em Python | `03_consultas.py` |
| Consulta "top recomendações para o usuário X" | `03_consultas.py --usuario X`, função `top_recomendacoes` |
| Avaliação do modelo (Precisão e Recall, 80/20) | `04_avaliacao.py` |

## Resultados

Configuração padrão: dataset 20M do Kaggle, primeiros 2000 usuários (290.944 notas: 233.503 de treino
e 57.441 de teste), 27.278 filmes e 569.670 arestas `SIMILAR`.

Tempos: inserção ~30s, cálculo de Pearson ~16s, script de similaridade completo ~60s.

### Avaliação (`python 04_avaliacao.py`)

```
Cobertura: 46003/57441 notas de teste com predição (80.1%)

Classificação (limiar = 4.0)
  Precisão: 0.8076
  Recall:   0.2508
  F1:       0.3828

Ranking (N = 10, média de 1962 usuários)
  Precisão@10: 0.6261
  Recall@10:   0.6604

Erro da nota prevista
  MAE:  0.7069
  RMSE: 0.9105
```

**Interpretação:**
- **Precisão alta (0,81), recall baixo (0,25):** quando o modelo prevê nota ≥ 4, acerta 81% das vezes.
  Mas a predição é uma média ponderada de notas, então tende a ficar perto da média do usuário e
  raramente passa de 4. Por isso, muitos filmes que o usuário gostou ficam abaixo do limiar.
- **No ranking, o recall sobe (0,66):** quando ordenamos os filmes pela nota prevista, em vez de usar
  um limiar fixo, o modelo coloca bem os filmes relevantes no topo da lista.
- **Erro médio (MAE):** cerca de 0,7 ponto na escala de 0,5 a 5.
- **Cobertura de 80%:** os 20% sem predição são filmes fora dos 2000 mais avaliados, ou filmes com
  menos de 2 vizinhos avaliados pelo usuário.

### Consultas (`python 03_consultas.py --usuario 1`)

```
Usuário 1 assistiu 21 filmes de Comedy com nota >= 4.0

Top 10 recomendações para o usuário 1:
   1. Princess Diaries, The (2001)  nota prevista 4.298  [7 vizinhos]
   2. 102 Dalmatians (2000)  nota prevista 4.288  [16 vizinhos]
   3. Hidalgo (2004)  nota prevista 4.216  [20 vizinhos]
   4. Thor (2011)  nota prevista 4.182  [15 vizinhos]
   5. Intouchables (2011)  nota prevista 4.148  [8 vizinhos]
   6. Hunger Games: Catching Fire, The (2013)  nota prevista 4.143  [20 vizinhos]
   7. Mighty Wind, A (2003)  nota prevista 4.142  [20 vizinhos]
   8. Hunger Games, The (2012)  nota prevista 4.134  [20 vizinhos]
   9. Bourne Identity, The (1988)  nota prevista 4.128  [20 vizinhos]
  10. Persepolis (2007)  nota prevista 4.126  [19 vizinhos]
```

**Limitação observada:** com `MIN_COMMON_USERS = 10`, aparecem correlações altas baseadas em poucos
usuários (por exemplo, Toy Story × Dead Calm: 0,89 com 14 usuários em comum). Aumentar esse valor
deixa as similaridades mais confiáveis, mas reduz a cobertura.

## Consultas Cypher úteis

```cypher
// Quantas comédias o usuário 1 avaliou com nota >= 4
MATCH (u:User {userId: 1})-[r:RATED]->(m:Movie)-[:IN_GENRE]->(:Genre {name: 'Comedy'})
WHERE r.rating >= 4
RETURN count(m);

// Filmes mais parecidos com Toy Story
MATCH (:Movie {movieId: 1})-[s:SIMILAR]-(b:Movie)
RETURN b.title, s.corr ORDER BY s.corr DESC LIMIT 10;
```

A consulta de top recomendações (KNN feito em Cypher) está em `03_consultas.py`, na função `top_recomendacoes`.

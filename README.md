# 🎥 Sistema de Recomendação de Filmes com Grafos

Sistema de recomendação por **filtragem colaborativa item-item** sobre o dataset **MovieLens**.
Os dados ficam num grafo **Neo4j**. A similaridade entre filmes (correlação de Pearson) é gravada como
arestas do grafo, e a nota que um usuário daria a um filme é prevista com **regressão KNN**.

**Autor:** Wagner Sanches

Restrição do enunciado: usar só as bibliotecas solicitadas. O projeto usa apenas a **biblioteca padrão do
Python** e o driver oficial **`neo4j`**, sem pandas, numpy ou scikit-learn. Pearson, KNN e as métricas
foram implementados à mão.

---

## Sumário

1. [Estrutura do projeto](#estrutura-do-projeto)
2. [Modelagem do grafo](#1-modelagem-do-grafo)
3. [Modelo de recomendação](#2-modelo-de-recomendação)
4. [Avaliação](#3-avaliação)
5. [Resultados](#resultados)
6. [Ajuste da similaridade](#ajuste-da-similaridade)
7. [Como executar](#como-executar)
8. [Critérios de avaliação × arquivos](#critérios-de-avaliação--arquivos)

---

## Estrutura do projeto

```
recomendacao-filmes/
├── config.py                    # parâmetros (K, β, divisão 80/20...) e conexão via .env
├── recomendador.py              # leitura dos CSVs, divisão treino/teste, Pearson, KNN
├── 01_inserir_dados.py          # insere usuários, filmes, gêneros e notas no Neo4j
├── 02_calcular_similaridade.py  # calcula Pearson e grava as arestas SIMILAR
├── 03_consultas.py              # consultas Cypher, incluindo as top recomendações
├── 04_avaliacao.py              # precisão, recall, precisão@N, recall@N, MAE, RMSE
├── .env.example                 # modelo do arquivo de conexão
├── requirements.txt
└── data/                        # CSVs do MovieLens (não versionados)
```

---

## 1) Modelagem do grafo

```
                 RATED {rating, timestamp, split}
  (:User) ─────────────────────────────────────────▶ (:Movie) ──IN_GENRE──▶ (:Genre)
  {userId}                                           {movieId,              {name}
                                                      title, year}
                                                        │   ▲
                                                        └───┘
                                       SIMILAR {corr, pearson, common}
```

### Nós

| Rótulo | Origem | Atributos |
|---|---|---|
| `User` | `ratings.csv` | `userId` |
| `Movie` | `movies.csv` | `movieId`, `title`, `year` (o ano é extraído do título: `"Toy Story (1995)"` → `"Toy Story"`, `1995`) |
| `Genre` | coluna `genres` de `movies.csv` | `name` |

### Arestas

| Tipo | Ligação | Atributos |
|---|---|---|
| `RATED` | `User → Movie` | `rating` (0,5 a 5), `timestamp` (epoch), `split` (`train` ou `test`) |
| `IN_GENRE` | `Movie → Genre` | (nenhum) |
| `SIMILAR` | `Movie — Movie` | `pearson` (correlação bruta), `corr` (correlação com shrinkage, usada no KNN), `common` (nº de usuários em comum) |

**Decisões de modelagem:**
- **Gênero como nó, não como lista dentro do filme:** consultas por gênero viram travessias no grafo
  (`(u)-[:RATED]->(m)-[:IN_GENRE]->(g)`) e usam índice.
- **A divisão treino/teste fica no próprio grafo:** a aresta `RATED` guarda o `split`. Assim, Pearson e
  KNN usam só o treino, e a avaliação compara com o teste sem duplicar dados.
- **`SIMILAR` gravada uma vez por par:** a correlação é simétrica, então cada par fica com uma aresta
  só, consultada sem direção (`-[:SIMILAR]-`).
- **Restrições de unicidade** em `User.userId`, `Movie.movieId` e `Genre.name`. Elas também criam os
  índices usados pelos `MATCH`.

---

## 2) Modelo de recomendação

### Correlação de Pearson entre filmes

Os usuários têm poucas avaliações cada, então a similaridade é calculada entre **filmes**. Para os filmes
*i* e *j*, somando sobre os usuários *k* que avaliaram os dois (só notas de treino):

$$c(i,j) = \frac{\sum_k (r_i^k - \bar{r}_i)(r_j^k - \bar{r}_j)}{\sqrt{\sum_k (r_i^k - \bar{r}_i)^2}\sqrt{\sum_k (r_j^k - \bar{r}_j)^2}}$$

- $r_i^k$: nota do usuário *k* para o filme *i*
- $\bar{r}_i$: nota média do filme *i* no treino

Implementação (`recomendador.calcular_pearson`): para cada usuário, percorre os pares de filmes que ele
avaliou e acumula $\sum d_i d_j$, $\sum d_i^2$, $\sum d_j^2$ e *n*. É uma passada só sobre os dados,
sem montar a matriz usuário × filme.

### Shrinkage

Correlações calculadas com poucos usuários em comum são ruidosas: Toy Story × *Dead Calm* tinha Pearson
0,89 com só 14 usuários. Por isso a correlação usada no KNN é reduzida proporcionalmente a *n*
(usuários em comum):

$$c'(i,j) = c(i,j) \cdot \frac{n}{n + \beta}, \qquad \beta = 100$$

Com 14 usuários em comum, a correlação cai a 12% do valor. Com 236, mantém 70%. A aresta guarda as
duas versões (`pearson` e `corr`).

### Regressão com KNN

Para prever a nota do usuário *u* no filme *i*, pegamos os *K* filmes *j* mais similares a *i* que *u*
avaliou no treino:

$$\hat{r}_i^u = \frac{\sum_{j \in N_K(i;u)} c'(i,j)\, r_j^u}{\sum_{j \in N_K(i;u)} |c'(i,j)|}$$

> O enunciado escreve $r_i^k$ no numerador, mas pela lógica do KNN item-item o termo é $r_j^u$:
> a nota que o próprio usuário deu a cada vizinho *j*.

A **top recomendação** de um usuário (`03_consultas.top_recomendacoes`) faz esse mesmo cálculo
inteiro em Cypher. Ela percorre `(u)-[:RATED]->(j)-[:SIMILAR]-(i)` para todo filme *i* que o usuário
não avaliou e ordena pela nota prevista.

### Parâmetros (`config.py`)

| Parâmetro | Valor | Motivo |
|---|---|---|
| `MAX_USERS` | 2000 | O dataset de 20M é grande demais para Python puro. 2000 usuários dão ~290 mil notas |
| `MIN_RATINGS_MOVIE` | 20 | Filmes com poucas notas não têm média confiável |
| `MAX_MOVIES` | 2000 | O número de pares cresce com o quadrado da quantidade de filmes |
| `MIN_COMMON_USERS` | 10 | Descarta pares com pouquíssimo suporte |
| `MIN_CORR` | 0 | Só correlações positivas. Com pesos negativos e notas brutas, a fórmula pode sair da escala |
| `SHRINKAGE_BETA` | 100 | Ver [ajuste da similaridade](#ajuste-da-similaridade) |
| `K` | 20 | Vizinhos usados na predição |
| `MIN_NEIGHBORS` | 2 | Não prevê com um vizinho só |
| `RELEVANCE_THRESHOLD` | 4,0 | Nota a partir da qual o filme é considerado relevante |

---

## 3) Avaliação

**Divisão 80/20 por usuário** (seed 42): as notas de cada usuário são embaralhadas, e 80% vão para
treino e 20% para teste. Assim, todo usuário do teste tem histórico para o KNN. Pearson e KNN usam
**apenas** o treino.

Para cada nota de teste (*u*, *i*, nota real), prevemos $\hat{r}_i^u$ e calculamos:

- **Precisão e Recall (classificação):** o filme é *recomendado* se $\hat{r} \ge 4$ e *relevante* se a nota real é ≥ 4.
  - Precisão = recomendados ∩ relevantes / recomendados
  - Recall = recomendados ∩ relevantes / relevantes
  - Notas sem predição contam como não recomendadas, ou seja, penalizam o recall.
- **Precisão@10 e Recall@10 (ranking):** para cada usuário, os 10 filmes de teste com maior nota prevista
  formam a lista de recomendação. O resultado é a média entre os usuários.
- **MAE e RMSE:** erro da nota prevista.
- **Cobertura:** fração das notas de teste em que foi possível prever.

---

## Resultados

Dataset **MovieLens 20M** (versão do Kaggle), primeiros 2000 usuários:

| | |
|---|---|
| Notas | 290.944 (233.503 de treino, 57.441 de teste) |
| Filmes | 27.278 |
| Arestas `SIMILAR` | 569.670 |
| Tempo de inserção | ~30 s |
| Tempo de similaridade | ~60 s |

### Métricas (`python 04_avaliacao.py`)

| Métrica | Valor |
|---|---|
| Cobertura | 80,1% |
| **Precisão** | **0,7935** |
| **Recall** | **0,3781** |
| F1 | 0,5121 |
| Precisão@10 | 0,6465 |
| Recall@10 | 0,6720 |
| MAE | 0,6781 |
| RMSE | 0,8911 |

**Interpretação:**
- **Precisão alta:** quando o modelo prevê nota ≥ 4, acerta 79% das vezes.
- **Recall menor:** a predição é uma média ponderada de notas e tende a ficar perto da média do usuário,
  então muitos filmes que ele gostou ficam pouco abaixo de 4. Os 20% de notas sem predição também
  entram como não recomendados.
- **No ranking, o recall sobe para 0,67:** ordenando pela nota prevista em vez de usar um limiar fixo,
  dois terços dos filmes relevantes de cada usuário aparecem no seu top 10.
- **Erro médio (MAE):** cerca de 0,68 ponto na escala de 0,5 a 5.

### Consultas (`python 03_consultas.py --usuario 1 --filme 1`)

```
Usuário 1 assistiu 21 filmes de Comedy com nota >= 4.0

Filmes mais similares ao filme 1:
  Toy Story 2                    corr=0.5    pearson=0.784  em comum=176
  Shrek                          corr=0.309  pearson=0.478  em comum=183
  Saving Private Ryan            corr=0.302  pearson=0.454  em comum=198
  Jumanji                        corr=0.297  pearson=0.502  em comum=145
  Back to the Future             corr=0.297  pearson=0.423  em comum=236
  ...

Top 10 recomendações para o usuário 1:
   1. Princess Diaries, The (2001)  nota prevista 4.348  [7 vizinhos]
   2. 102 Dalmatians (2000)  nota prevista 4.308  [16 vizinhos]
   3. Hidalgo (2004)  nota prevista 4.266  [20 vizinhos]
   4. Thor (2011)  nota prevista 4.253  [15 vizinhos]
   5. Hunger Games, The (2012)  nota prevista 4.241  [20 vizinhos]
   6. Intouchables (2011)  nota prevista 4.22  [8 vizinhos]
   7. Hunger Games: Catching Fire, The (2013)  nota prevista 4.216  [20 vizinhos]
   8. Bourne Identity, The (1988)  nota prevista 4.214  [20 vizinhos]
   9. Persepolis (2007)  nota prevista 4.208  [19 vizinhos]
  10. Mighty Wind, A (2003)  nota prevista 4.206  [20 vizinhos]
```

---

## Ajuste da similaridade

A primeira versão usava o Pearson bruto. Os filmes "mais similares" a Toy Story eram *Dead Calm* (0,89),
*Waiting to Exhale* (0,77) e *Summer of Sam* (0,76), todos com 11 a 14 usuários em comum.
Testamos duas formas de corrigir isso:

**a) Exigir mais usuários em comum** (sem shrinkage, K = 20):

| `MIN_COMMON_USERS` | Arestas | Cobertura | Precisão | Recall | F1 | P@10 | R@10 | MAE |
|---|---|---|---|---|---|---|---|---|
| 10 | 569.670 | 80,1% | 0,8076 | 0,2508 | 0,3828 | 0,6261 | 0,6604 | 0,7069 |
| 20 | 232.604 | 77,0% | 0,7641 | 0,3038 | 0,4348 | 0,6187 | 0,6357 | 0,7100 |
| 30 | 118.965 | 68,7% | 0,7510 | 0,3101 | 0,4390 | 0,6277 | 0,6143 | 0,7068 |
| 50 | 44.706 | 54,5% | 0,7548 | 0,2948 | 0,4240 | 0,6423 | 0,5640 | 0,6960 |

Cortar pares derruba a cobertura e o Recall@10.

**b) Shrinkage** (`MIN_COMMON_USERS = 10`):

| β | K | Precisão | Recall | F1 | P@10 | R@10 | MAE |
|---|---|---|---|---|---|---|---|
| 0 (sem) | 20 | 0,8076 | 0,2508 | 0,3828 | 0,6261 | 0,6604 | 0,7069 |
| 25 | 20 | 0,8045 | 0,3270 | 0,4650 | 0,6404 | 0,6682 | 0,6801 |
| 50 | 20 | 0,7982 | 0,3544 | 0,4909 | 0,6433 | 0,6699 | 0,6780 |
| **100** | **20** | **0,7935** | **0,3781** | **0,5121** | **0,6465** | **0,6720** | **0,6781** |
| 200 | 20 | 0,7884 | 0,3944 | 0,5258 | 0,6479 | 0,6722 | 0,6787 |
| 400 | 20 | 0,7851 | 0,4058 | 0,5350 | 0,6495 | 0,6735 | 0,6793 |
| 100 | 10 | 0,7751 | 0,4220 | 0,5464 | 0,6422 | 0,6690 | 0,6820 |
| 50 | 30 | 0,8067 | 0,3327 | 0,4711 | 0,6438 | 0,6700 | 0,6806 |
| 50 | 50 | 0,8095 | 0,3094 | 0,4476 | 0,6443 | 0,6705 | 0,6854 |

O shrinkage mantém todas as arestas (cobertura de 80%) e melhora quase tudo em relação ao Pearson bruto:
recall de 0,25 para 0,38, Precisão@10 de 0,626 para 0,647 e MAE de 0,707 para 0,678. A precisão
cai só 1,4 ponto. A partir de β ≈ 100, as métricas de ranking e o MAE se estabilizam. Escolhemos
**β = 100 e K = 20** em vez de buscar o máximo em cada métrica, porque esses valores foram escolhidos
olhando o próprio conjunto de teste e um ajuste fino demais seria sobreajuste a ele.

---

## Como executar

**Pré-requisitos:** Python 3.10+ e Neo4j 5.x (Neo4j Desktop ou Community Server) com uma instância iniciada.

1. **Dados:** baixe o MovieLens e coloque os CSVs em `data/`. São aceitos `movies.csv`/`ratings.csv`
   ([GroupLens](https://grouplens.org/datasets/movielens/)) ou `movie.csv`/`rating.csv`
   ([Kaggle 20M](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset)).

2. **Conexão:** copie `.env.example` para `.env` e preencha a senha do Neo4j:
   ```
   NEO4J_URI=neo4j://127.0.0.1:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=sua_senha
   ```
   Variáveis de ambiente com os mesmos nomes têm prioridade sobre o `.env`.

3. **Ambiente:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate            # Windows  (Linux/Mac: source .venv/bin/activate)
   pip install -r requirements.txt
   ```

4. **Executar em ordem:**
   ```bash
   python 01_inserir_dados.py          # ⚠️ apaga o banco e insere tudo (~30 s)
   python 02_calcular_similaridade.py  # Pearson → arestas SIMILAR (~60 s)
   python 03_consultas.py --usuario 1 --genero Comedy --nota 4 --filme 1
   python 04_avaliacao.py
   ```

### Consultas Cypher para o Neo4j Browser

```cypher
// Quantas comédias o usuário 1 avaliou com nota >= 4
MATCH (u:User {userId: 1})-[r:RATED]->(m:Movie)-[:IN_GENRE]->(:Genre {name: 'Comedy'})
WHERE r.rating >= 4
RETURN count(m);

// Filmes mais parecidos com Toy Story
MATCH (:Movie {movieId: 1})-[s:SIMILAR]-(b:Movie)
RETURN b.title, s.corr, s.pearson, s.common ORDER BY s.corr DESC LIMIT 10;

// Visualizar a vizinhança de um usuário no grafo
MATCH p = (:User {userId: 1})-[:RATED]->(:Movie)-[:IN_GENRE]->(:Genre)
RETURN p LIMIT 50;
```

---

## Critérios de avaliação × arquivos

| Critério | Onde está |
|---|---|
| Modelagem: o que são nós e arestas | [Modelagem do grafo](#1-modelagem-do-grafo) e docstring de `01_inserir_dados.py` |
| Script de inserção no Neo4j em Python | `01_inserir_dados.py` (dados) e `02_calcular_similaridade.py` (arestas `SIMILAR`) |
| Script de consulta em Python | `03_consultas.py` |
| Consulta "top recomendações para o usuário X" | `python 03_consultas.py --usuario X`, função `top_recomendacoes` |
| Avaliação do modelo (80/20, Precisão e Recall) | `04_avaliacao.py` e [Resultados](#resultados) |

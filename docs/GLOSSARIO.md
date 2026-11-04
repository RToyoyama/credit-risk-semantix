# Glossário

Termos técnicos do projeto explicados em linguagem simples. Cada termo indica a etapa em que apareceu pela primeira vez.

| Termo | Nome em inglês | Explicação simples | Etapa |
|---|---|---|---|
| Inadimplência / calote | *default* | quando o cliente não paga a fatura no prazo. Aqui, "não pagar o próximo mês" | 1 |
| Probabilidade de default (PD) | *probability of default* | a chance, de 0% a 100%, de um cliente não pagar. "PD de 20%" quer dizer que, de cada 100 clientes parecidos, cerca de 20 não pagam | 1 |
| Carteira | *portfolio* | o conjunto de todos os clientes da emissora, vistos juntos | 1 |
| Perda esperada (EL) | *expected loss* | quanto se espera perder com um cliente, em dinheiro, considerando a chance de calote e quanto ele deve | 1 |
| Score de crédito | *credit score* | uma nota (aqui de 0 a 1000) que resume o risco: quanto maior, menor o risco | 1 |
| Faixa de risco / rating | *rating* | grupos de clientes com risco parecido (A = menor risco, E = maior), cada um com uma ação | 1 |
| Motivos da decisão | *reason codes* | os principais fatores que levaram um cliente a ter o risco que tem, explicados em frases | 1 |
| Falso negativo | *false negative* | o modelo diz "vai pagar", mas o cliente dá calote. É o erro que deixa o prejuízo passar | 1 |
| Falso positivo | *false positive* | o modelo diz "vai dar calote", mas o cliente pagaria. É o erro que incomoda um bom cliente | 1 |
| Custo assimétrico | *asymmetric cost* | quando um tipo de erro custa muito mais que o outro, por isso não dá para tratar os dois igual | 1 |
| Limiar de decisão | *decision threshold* | o ponto de corte da probabilidade a partir do qual se age. Ex.: "agir se PD ≥ 15%" | 1 |
| Holdout | *holdout set* | um pedaço dos dados (aqui 20%) guardado "na gaveta" e usado só uma vez, no fim, como prova final | 1 |
| Acurácia | *accuracy* | a porcentagem de acertos. Engana quando uma classe é rara: dizer "ninguém dá calote" acerta cerca de 78% aqui | 1 |
| Data card | *data card* | a "bula" de um conjunto de dados: fonte, licença, significado das colunas, limitações | 3 |
| Schema | *schema* | o formato esperado de uma tabela: quais colunas, de que tipo e com que valores permitidos | 3 |
| Validação de schema | *schema validation* | conferir, antes de usar, se os dados chegaram no formato esperado, como conferir uma entrega pela nota fiscal | 3 |
| SHA-256 / hash | *hash* | uma "impressão digital" do arquivo: se um único byte mudar, o código muda totalmente | 3 |
| Parquet | *Parquet* | formato de arquivo para tabelas, compacto e rápido, que guarda o tipo de cada coluna | 3 |
| Camada raw | *raw layer* | os dados exatamente como vieram da fonte, sem nenhuma alteração | 3 |
| Crédito rotativo | *revolving credit* | quando o cliente paga só parte da fatura e o resto vira dívida com juros para o mês seguinte | 3 |
| Vazamento de dados | *data leakage* | quando o modelo, sem querer, usa uma informação que não teria no mundo real, como "colar na prova". Resultado: nota ótima no teste e fracasso na prática | 4 |
| Estratificação | *stratified split* | separar os dados mantendo a mesma proporção de cada classe dos dois lados (aqui, 22,12% de inadimplentes no treino e no holdout) | 4 |
| Seed (semente) | *random seed* | número que fixa o "sorteio" do computador, para que ele saia igual toda vez | 4 |
| Flag | *flag / indicator* | coluna 0/1 que marca um caso especial (ex.: "o valor original não era documentado") | 4 |
| Camada interim | *interim layer* | os dados já limpos, mas ainda sem features novas | 4 |
| Análise exploratória (EDA) | *exploratory data analysis* | olhar e testar os dados antes de modelar, para entender o que está associado ao alvo | 5 |
| Intervalo de confiança (IC) | *confidence interval* | a margem de erro de uma estimativa. "22,12% (IC 95%: 21,60%–22,66%)" quer dizer que a taxa verdadeira muito provavelmente está nessa faixa | 5 |
| Bootstrap | *bootstrap* | calcular a margem de erro sorteando milhares de "amostras parecidas" (com reposição) a partir dos próprios dados | 5 |
| Intervalo de Wilson | *Wilson interval* | fórmula para a margem de erro de uma proporção que funciona bem até em grupos pequenos | 5 |
| p-valor | *p-value* | a chance de ver uma diferença tão grande quanto a observada **se, na verdade, não houvesse diferença nenhuma**. Responde "existe diferença?", não "ela importa?" | 5 |
| Significância estatística | *statistical significance* | quando o p-valor fica abaixo de um limite combinado antes (aqui, 5%). Com amostras grandes, até diferenças mínimas ficam significativas | 5 |
| Tamanho de efeito | *effect size* | mede **quanto** uma variável se relaciona com o alvo, independente do tamanho da amostra. Responde "a diferença importa?" | 5 |
| Teste qui-quadrado | *chi-squared test* | verifica se a taxa de default muda entre as categorias de uma variável. Só é confiável se cada célula esperar ≥ 5 casos | 5 |
| V de Cramér | *Cramér's V* | tamanho de efeito para categorias: 0 = sem relação, 1 = a categoria determina o resultado | 5 |
| Teste de Mann-Whitney | *Mann-Whitney U test* | compara dois grupos numa variável numérica sem supor curva em forma de sino. Só detecta "um grupo tende a ter valores maiores" (não vê relações em U) | 5 |
| Correlação rank-biserial | *rank-biserial correlation* | tamanho de efeito do Mann-Whitney, de -1 a 1. `(r + 1) / 2` é a chance de um inadimplente sorteado ter valor maior que um adimplente sorteado | 5 |
| Correção de Holm | *Holm-Bonferroni correction* | ajuste para quando se fazem muitos testes: deixa cada um mais exigente, para não "achar" coisas por puro acaso | 5 |
| Associação × causalidade | *correlation vs. causation* | duas coisas andarem juntas não quer dizer que uma causa a outra | 5 |
| Segmentação | *customer segmentation* | dividir os clientes em grupos com comportamento parecido, para tratar cada grupo de um jeito | 6 |
| Utilização do limite | *credit utilization* | quanto do limite a fatura ocupa. Fatura de 3 mil num limite de 10 mil = 30% | 6 |
| Padronização | *standardization* | colocar todas as colunas na mesma régua: (valor − média) ÷ desvio-padrão. Evita que uma coluna "grite mais alto" só por causa da escala | 6 |
| PCA | *principal component analysis* | técnica que resume muitas colunas parecidas em poucas "direções" que guardam a maior parte da informação | 6 |
| Componente principal | *principal component* | cada uma dessas "direções": uma combinação das colunas originais | 6 |
| Variância explicada | *explained variance* | quanto da variação total dos dados um componente consegue representar | 6 |
| Autovetor e autovalor | *eigenvector / eigenvalue* | na álgebra linear do PCA: o autovetor é a direção do componente, e o autovalor diz quanta variância ele carrega | 6 |
| Regra de Kaiser | *Kaiser criterion* | manter só componentes com autovalor > 1, ou seja, que "valem" mais que uma coluna original sozinha | 6 |
| Carga | *loading* | o peso de cada coluna original num componente. Positiva: andam juntos; negativa: andam em sentidos opostos | 6 |

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
| K-means | *k-means clustering* | algoritmo que escolhe k centros e coloca cada cliente no grupo do centro mais próximo, repetindo até estabilizar | 6 |
| Centroide | *centroid* | o "centro" de um grupo: a média dos clientes que pertencem a ele | 6 |
| Inércia | *inertia* | soma das distâncias (ao quadrado) de cada cliente até o centro do seu grupo. Sempre cai quando se aumenta o número de grupos | 6 |
| Método do cotovelo | *elbow method* | escolher k no ponto em que acrescentar mais um grupo passa a reduzir pouco a inércia | 6 |
| Silhueta | *silhouette score* | de -1 a 1: compara a distância de um cliente até o próprio grupo com a distância até o grupo vizinho. Perto de 1 = grupos bem separados | 6 |
| ARI | *Adjusted Rand Index* | nota de 0 a 1 de quão iguais são duas divisões dos mesmos clientes em grupos (1 = idênticas; 0 = como um sorteio) | 6 |
| Validação cruzada | *cross-validation* | dividir o treino em partes, treinar em algumas e testar na que ficou de fora, repetindo para cada parte. Dá uma nota mais confiável que um único teste | 6 |
| Fold | *fold* | cada uma das partes da validação cruzada. Aqui: 5 partes × 3 repetições = 15 rodadas | 6 |
| Comparação pareada | *paired comparison* | comparar duas versões de um modelo **nos mesmos folds**, para que a diferença venha da mudança e não da sorte da divisão | 6 |
| Win rate | *win rate* | em quantos folds a versão nova venceu a antiga. Win rate de 60% = venceu em 9 de 15 | 6 |
| PR AUC | *precision-recall AUC / average precision* | mede o quanto o modelo acha os inadimplentes sem acusar bons pagadores. Um chute tem PR AUC igual à taxa de default (0,22) | 6 |
| Criação de variáveis | *feature engineering* | transformar os dados brutos em colunas que falam a "língua" do problema. Ex.: em vez de "fatura de NT$ 50 mil", "fatura = 90% do limite" | 7 |
| Função pura | *pure function* | função que, para a mesma entrada, sempre dá a mesma saída e não depende de mais nada. Aqui: cada feature usa só os dados daquele cliente | 7 |
| Tendência | *trend* | diferença entre o comportamento recente e o antigo. Positiva = o valor está subindo | 7 |
| Volatilidade | *volatility* | o quanto um valor oscila de um mês para o outro (aqui, o desvio-padrão da utilização) | 7 |
| Escala logarítmica | *log transform* | trocar o valor pelo seu logaritmo, o que "aproxima" números muito grandes. De 10 mil para 1 milhão há 100×, mas no log a distância fica parecida com a de 100 para 10 mil | 7 |
| Pré-processamento | *preprocessing* | preparar as colunas para o modelo: pôr números na mesma régua, transformar categorias em colunas 0/1 etc. | 8 |
| Pipeline | *pipeline* | uma "linha de montagem": passos de preparação e o modelo encadeados, que são ajustados e aplicados sempre juntos e na mesma ordem | 8 |
| ColumnTransformer | *ColumnTransformer* | um "despachante" que manda cada grupo de colunas para a sua transformação e junta tudo no final | 8 |
| One-hot | *one-hot encoding* | transformar uma categoria em colunas 0/1. "Escolaridade = universidade" vira `education_2 = 1` e as demais 0 | 8 |
| Categoria infrequente | *infrequent category* | categoria rara demais no treino (aqui, < 50 casos), agrupada numa coluna única para não virar ruído | 8 |
| Clonagem | *cloning* | copiar um pipeline **sem** o que ele aprendeu, como a validação cruzada faz a cada fold para não vazar informação | 8 |
| Regressão logística | *logistic regression* | modelo que soma as variáveis com pesos e converte o resultado numa probabilidade entre 0 e 1 | 9 |
| Árvore de Decisão | *decision tree* | modelo que faz uma sequência de perguntas sim/não ("atrasou 2 meses?") até chegar a uma resposta | 9 |
| Random Forest | *random forest* | centenas de árvores, cada uma treinada numa amostra diferente dos dados; a resposta é a média delas | 9 |
| XGBoost | *extreme gradient boosting* | árvores construídas em sequência, cada uma corrigindo os erros das anteriores | 9 |
| Hiperparâmetro | *hyperparameter* | uma configuração do modelo escolhida antes do treino (ex.: profundidade máxima da árvore), e não aprendida dos dados | 9 |
| ROC AUC | *area under the ROC curve* | a chance de o modelo dar risco maior a um inadimplente do que a um bom pagador, sorteando um de cada. 0,5 = chute; 1 = perfeito | 9 |
| KS | *Kolmogorov-Smirnov statistic* | a maior distância entre a fração de maus e a de bons pagadores acumulada ao longo do score. Métrica padrão do mercado de crédito | 9 |
| Brier score | *Brier score* | erro médio ao quadrado entre a probabilidade prevista e o que aconteceu (0 ou 1). Mede se as probabilidades são confiáveis; menor é melhor | 9 |
| Razão de chances | *odds ratio* | o número pelo qual a chance de calote é multiplicada quando uma variável aumenta 1 unidade, mantidas as outras. 2 = dobra; 0,5 = cai pela metade | 9 |
| Multicolinearidade | *multicollinearity* | quando variáveis carregam quase a mesma informação; o modelo divide o peso entre elas de forma arbitrária e os coeficientes perdem o sentido individual | 9 |
| VIF | *variance inflation factor* | mede a multicolinearidade: 1 ÷ (1 − R²), em que R² é o quanto as outras variáveis explicam aquela. Acima de 5 é alerta | 9 |
| Pseudo-R² | *pseudo R-squared (McFadden)* | o quanto o modelo melhora em relação a não saber nada, numa escala de 0 a 1. Em crédito, 0,1 a 0,3 é comum | 9 |
| Confusão | *confounding* | quando o efeito de uma variável é, na verdade, o efeito de outra que anda junto com ela (ex.: pagar pouco "parecia" arriscado, mas era o atraso) | 9 |
| Desbalanceamento de classes | *class imbalance* | quando uma das respostas é bem mais rara que a outra (aqui, 22% de inadimplentes contra 78% de bons pagadores) | 10 |
| Pesos por classe | *class weight* | dar mais peso aos erros na classe rara durante o treino. Aqui, cada inadimplente "vale" cerca de 3,5 bons pagadores | 10 |
| SMOTE | *Synthetic Minority Over-sampling Technique* | criar exemplos sintéticos da classe rara, interpolando entre exemplos parecidos, até as classes ficarem do mesmo tamanho | 10 |
| Calibração | *calibration* | quando as probabilidades "falam a verdade": entre os clientes que receberam 30%, cerca de 30% dão calote | 10 |
| Curva de confiabilidade | *reliability curve / calibration curve* | gráfico da probabilidade prevista contra a taxa real, por faixas. Num modelo calibrado, os pontos ficam sobre a diagonal | 10 |
| Previsão fora do fold (OOF) | *out-of-fold prediction* | a previsão de cada cliente feita por um modelo que **não** o viu no treino. Permite avaliar o treino inteiro sem tocar no holdout | 10 |
| Ajuste de hiperparâmetros | *hyperparameter tuning* | testar várias configurações do modelo e ficar com a que dá a melhor nota na validação cruzada | 11 |
| Busca em grade | *grid search* | testar **todas** as combinações de uma lista de valores. Bom quando há poucos hiperparâmetros | 11 |
| Busca aleatória | *randomized search* | sortear combinações num espaço amplo. Com muitos hiperparâmetros, acha soluções tão boas quanto a grade, em bem menos tempo | 11 |
| Otimismo da busca | *selection bias / optimistic bias* | a nota do melhor candidato de uma busca é inflada pela sorte, como ficar com o maior de 60 dados. Mede-se reavaliando o escolhido em folds que a busca não viu | 11 |
| Regularização | *regularization* | um "freio" que impede o modelo de dar pesos exagerados às variáveis. Na logística, `C` pequeno = freio forte | 11 |
| Taxa de aprendizado | *learning rate* | no XGBoost, o tamanho do passo de cada nova árvore ao corrigir as anteriores. Passos menores pedem mais árvores, mas costumam generalizar melhor | 11 |
| Combinação de modelos | *ensemble* | juntar as previsões de vários modelos numa só, na esperança de que os erros de um sejam compensados pelos acertos de outro | 11 |
| Diversidade | *diversity* | o quanto os modelos erram em clientes diferentes. Sem diversidade, combinar não ajuda | 11 |
| Votação soft | *soft voting* | combinação que tira a média das probabilidades dos modelos | 11 |
| Stacking | *stacking* | combinação em que um modelo "chefe" aprende quanto confiar em cada modelo, usando previsões feitas em dados que eles não viram | 11 |
| Calibração sigmoid (Platt) | *Platt scaling* | corrige as probabilidades com uma curva em "S" aprendida em dados que o modelo não viu | 12 |
| Calibração isotônica | *isotonic regression* | corrige as probabilidades com uma "escada" livre e sempre crescente; mais flexível que a sigmoid, mas pode se ajustar ao ruído | 12 |
| LGD | *Loss Given Default* | a fração da dívida que se perde quando o cliente dá calote. LGD de 60% = de cada NT$ 100 devidos, NT$ 60 não voltam | 12 |
| EAD | *Exposure at Default* | quanto o cliente deve no momento do calote; aqui, a fatura mais recente | 12 |
| Valor esperado | *expected value* | o resultado médio de uma decisão, pesando cada desfecho pela sua probabilidade. Ex.: 20% de chance de ganhar 1.000 e 80% de perder 100 = 0,2 × 1.000 − 0,8 × 100 = +120 | 12 |
| Restrição de capacidade | *capacity constraint* | o limite de quantos clientes a equipe consegue acionar por mês; com ele, a pergunta vira "quem acionar primeiro?" | 12 |
| Análise de sensibilidade | *sensitivity analysis* | refazer a conta trocando uma premissa de cada vez, para ver se a conclusão muda quando ela está errada | 12 |
| Robustez | *robustness check* | conferir se uma conclusão se mantém quando se muda um detalhe do método | 12 |
| Atributo sensível | *protected / sensitive attribute* | uma característica da pessoa que não deveria influenciar a decisão (ex.: sexo, estado civil), por motivos éticos ou legais | 13 |
| Não inferioridade | *non-inferiority* | critério que aceita uma mudança se ela não piorar o resultado além de uma margem combinada antes. Útil quando a mudança tem outro benefício (aqui, ético) | 13 |
| Equidade | *fairness* | verificar se o modelo funciona e decide de forma parecida para grupos diferentes (ex.: homens e mulheres) | 13 |
| Fonte única de verdade | *single source of truth* | um único lugar no código define a regra (aqui, o modelo oficial), e todas as partes o usam, sem cópias que possam divergir | 13 |
| Importância por permutação | *permutation importance* | embaralhar uma variável entre os clientes e medir quanto o modelo piora. Se piora muito, ele dependia dela | 13 |
| Dependência parcial | *partial dependence plot (PDP)* | mudar uma variável para o mesmo valor em todos os clientes e ver a PD média; mostra o "formato" do efeito | 13 |
| Matriz de confusão | *confusion matrix* | tabela que cruza a decisão (acionado ou não) com o que aconteceu (calote ou não) | 13 |
| Variável proxy | *proxy variable* | uma variável que, sem ser sensível, anda junto com um atributo sensível e pode reproduzir uma discriminação de forma indireta | 13 |

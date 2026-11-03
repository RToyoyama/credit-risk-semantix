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

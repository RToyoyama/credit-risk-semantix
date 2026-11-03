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

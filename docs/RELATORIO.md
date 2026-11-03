# Relatório — Risco de Crédito Orientado a Custo

> Documento vivo: cada etapa do projeto acrescenta uma parte. Os três tópicos obrigatórios da entrega (**Coleta de dados**, **Modelagem** e **Conclusões**) são preenchidos ao longo das etapas. Termos técnicos estão explicados no [Glossário](GLOSSARIO.md).

---

## 1. O problema de negócio

### 1.1 Contexto

Uma emissora de cartão de crédito empresta dinheiro todo mês: o cliente compra agora e paga depois. A maioria paga em dia, mas uma parte deixa de pagar. Isso se chama **inadimplência**, que o mercado chama de **default**.

Quando o cliente para de pagar, a emissora costuma recuperar só uma parte da dívida, e tarde. Por isso, agir **antes** do calote vale mais do que cobrar depois. Exemplos de ação: oferecer renegociação, reduzir o limite ou fazer um contato preventivo.

### 1.2 A decisão que o projeto apoia

> **Todo mês, para cada cliente: agir preventivamente ou não?**

A resposta não é um "sim/não" isolado. O projeto entrega uma **política para a carteira** (o conjunto de todos os clientes). Ela tem quatro partes:

1. **Quem tem risco?** A probabilidade de cada cliente não pagar no próximo mês, chamada de **PD** (*probability of default*).
2. **Quanto isso custa?** A **perda esperada** em dinheiro de cada cliente.
3. **O que fazer?** Um **score de 0 a 1000** e **faixas de risco de A a E** (também chamadas de *rating*), cada uma com uma ação.
4. **Por quê?** Os **motivos da decisão** de cada cliente, em linguagem de negócio.

### 1.3 Por que os erros não custam igual

O modelo pode errar de dois jeitos, e cada erro tem um preço diferente:

| Erro | O que acontece | Custo para a emissora |
|---|---|---|
| **Falso negativo**: o modelo diz "vai pagar", mas o cliente dá calote | ninguém age | perde-se a parte da dívida que não for recuperada |
| **Falso positivo**: o modelo diz "vai dar calote", mas o cliente pagaria | age-se à toa | custo da ação + atrito com um bom cliente (que pode reduzir o uso do cartão ou cancelar) |

**Exemplo ilustrativo** (valores hipotéticos, só para mostrar a ideia; as premissas reais ficam em `config.py` e são justificadas na Etapa 12):

- Um cliente deve NT$ 10.000. Se ele der calote e ninguém agir, a emissora perde 60% disso: **NT$ 6.000**.
- Uma ação preventiva custa, por exemplo, **NT$ 200** entre contato e margem perdida.

Nesse cenário, deixar passar **um** caloteiro custa o mesmo que incomodar **30** bons clientes. Por isso, usar o limiar padrão de 50% ("acima de 50% de chance, eu ajo") é arbitrário. O ponto de corte certo é o que **minimiza o custo total**, e ele depende do valor que cada cliente deve.

### 1.4 Por que usar dados e Machine Learning, e não uma regra fixa

Uma regra fixa, como "agir em quem atrasou 2 meses ou mais", é simples, mas tem três limites:

| Limite da regra fixa | O que o modelo faz de diferente |
|---|---|
| Olha **uma** variável por vez | combina histórico de pagamento, uso do limite, valores das faturas e perfil |
| Dá um "sim/não", sem grau de risco | dá uma **probabilidade**, que permite ordenar clientes e calcular o custo esperado |
| O corte é escolhido por intuição | o corte é escolhido **pelo custo**, com análise de sensibilidade às premissas |

A regra fixa não é descartada: ela entra como **ponto de comparação**. O modelo só se justifica se superar a regra (ver 1.5).

**Quando ML não seria a escolha certa:** se houvesse pouquíssimos dados, se a regulação exigisse uma regra 100% explicável sem exceções, ou se o ganho sobre a regra simples fosse menor que o custo de manter um modelo. Esses pontos são reavaliados nas Conclusões.

### 1.5 Métrica de sucesso (definida antes de ver os resultados)

O sucesso é medido em **dinheiro**, não em acurácia:

> **Métrica principal:** resultado financeiro esperado da política (perdas evitadas − custos das ações), comparado com dois cenários de referência:
> 1. **Não agir em ninguém** (o status quo);
> 2. **A melhor regra fixa simples** baseada no histórico de atraso.

**Critério de decisão, fixado agora:** o modelo é recomendado se tiver resultado financeiro **maior que os dois cenários de referência** no holdout (os 20% de dados separados e só usados no fim), com o intervalo de confiança da diferença acima de zero.

**Métricas técnicas de apoio:** ROC AUC, PR AUC, KS e Brier score (explicadas no glossário e detalhadas na Etapa 9). A acurácia é mostrada só para explicar por que ela engana: com cerca de 22% de inadimplentes, um modelo que diz "ninguém dá calote" acerta cerca de 78% e não tem utilidade nenhuma.

### 1.6 Dados utilizados (resumo)

*Default of Credit Card Clients*, de Yeh & Lien (2009), disponível no UCI Machine Learning Repository (licença CC BY 4.0): 30.000 clientes de um banco de Taiwan, de abril a setembro de 2005. Detalhes completos na seção **Coleta de dados** (Etapa 3) e em [`data_card.md`](data_card.md).

> TODO (Etapa 3): confirmar com os dados baixados o número de linhas e a taxa de inadimplência.

---

## 2. Coleta de dados

> TODO (Etapas 3–4).

## 3. Modelagem

> TODO (Etapas 5–17).

## 4. Conclusões

> TODO (Etapa 20).

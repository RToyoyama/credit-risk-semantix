# Data Card — Default of Credit Card Clients

Um *data card* é a "bula" do conjunto de dados: de onde veio, o que cada coluna significa, como pode ser usado e quais são os cuidados.

## Fonte

| Item | Valor |
|---|---|
| Nome | *Default of Credit Card Clients* |
| Autores | I-Cheng Yeh e Che-hui Lien |
| Repositório | [UCI Machine Learning Repository, dataset 350](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) |
| DOI | [10.24432/C55S3H](https://doi.org/10.24432/C55S3H) |
| Licença | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): uso livre, inclusive comercial, com citação obrigatória |
| Arquivo | `default+of+credit+card+clients.zip` → `default of credit card clients.xls` |
| SHA-256 do zip | `56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602` (conferido em 2026-11-03) |
| Dados pessoais | anonimizados: não há nome, documento ou endereço; `ID` é só um número sequencial |

**Citação:**
> Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications, 36*(2), 2473–2480. https://doi.org/10.1016/j.eswa.2007.12.020

## Como obter

```bash
make data
```

O comando baixa o zip direto da UCI (sem login), confere o SHA-256 (se o arquivo mudar na fonte, o processo para em vez de seguir com dados diferentes), lê a planilha, valida o schema e salva `data/raw/credit_default_raw.parquet`. Se o zip já existir e estiver íntegro, o download é pulado.

**Plano B (manual):** se o link da UCI sair do ar, há um espelho no Kaggle (`uciml/default-of-credit-card-clients-dataset`, arquivo `UCI_Credit_Card.csv`). Ele **não** é usado pelo código porque exige conta e token, e o projeto deve rodar sem credenciais. Diferenças conhecidas do espelho: formato CSV e alvo chamado `default.payment.next.month`.

## Visão geral

| Item | Valor |
|---|---|
| Clientes (linhas) | 30.000 |
| Colunas | 25 (ID + 23 variáveis + alvo) |
| Período | abril a setembro de 2005 |
| Local | Taiwan (moeda: NT$, dólar taiwanês) |
| Valores nulos | 0 |
| IDs duplicados | 0 |
| Inadimplentes no mês seguinte | **6.636 (22,12%)** |

## Dicionário de variáveis

Os nomes são os originais da UCI (camada raw). A renomeação para um padrão consistente acontece na limpeza (Etapa 4).

### Perfil do cliente

| Coluna | Significado | Valores |
|---|---|---|
| `ID` | identificador do cliente | 1 a 30.000 |
| `LIMIT_BAL` | limite de crédito concedido (NT$), incluindo cartões adicionais da família | 10.000 a 1.000.000; mediana 140.000 |
| `SEX` | sexo | 1 = masculino (11.888), 2 = feminino (18.112) |
| `EDUCATION` | escolaridade | 1 = pós-graduação, 2 = universidade, 3 = ensino médio, 4 = outros; **0, 5, 6 não documentados** |
| `MARRIAGE` | estado civil | 1 = casado, 2 = solteiro, 3 = outros; **0 não documentado** |
| `AGE` | idade em anos | 21 a 79; mediana 34 |

### Histórico de pagamento (6 meses)

| Colunas | Significado |
|---|---|
| `PAY_0`, `PAY_2` … `PAY_6` | situação do pagamento em set, ago, jul, jun, mai e abr/2005 (`PAY_0` = mais recente). **Não existe `PAY_1`** na fonte |
| `BILL_AMT1` … `BILL_AMT6` | valor da fatura em set … abr/2005 (NT$). Pode ser **negativo**: crédito a favor do cliente |
| `PAY_AMT1` … `PAY_AMT6` | valor pago em set … abr/2005 (NT$) |

**Escala do `PAY_*`:**

| Valor | Significado | Origem |
|---|---|---|
| -2 | **quitou a fatura integralmente** (saldo baixo). A interpretação usual "sem consumo" é contrariada pelos dados: esses clientes têm fatura positiva e pagaram 100% da anterior (mediana) — ver EDA, seção 6 | não documentado; leitura baseada nos dados |
| -1 | pagou em dia (valor integral) | documentação |
| 0 | pagou o mínimo, usando crédito rotativo | não documentado; **confirmado pelos dados** (pagam ~6% da fatura anterior) |
| 1 a 9 | meses de atraso (9 = nove ou mais) | documentação. Na prática o máximo observado é 8 |

### Alvo

| Coluna | Significado | Valores |
|---|---|---|
| `default payment next month` | o cliente ficou inadimplente em out/2005? | 0 = não (23.364), 1 = sim (6.636) |

## Peculiaridades encontradas na validação

| Achado | Quantidade | Tratamento |
|---|---|---|
| `EDUCATION` = 5 / 6 / 0 | 280 / 51 / 14 clientes (1,15% no total) | agrupar em "outros" (Etapa 4) |
| `MARRIAGE` = 0 | 54 clientes (0,18%) | agrupar em "outros" (Etapa 4) |
| `BILL_AMT` negativo em pelo menos um mês | 1.930 clientes | válido (crédito a favor); EAD = 0 nesses casos (Etapa 14) |
| Sequência `PAY_0, PAY_2…` sem `PAY_1` | — | renomear (Etapa 4) |
| Significado de -2 e 0 em `PAY_*` não documentado | — | investigado na EDA: -2 = quitou a fatura; 0 = rotativo. As "contradições" status × fatura (4.594 clientes) são, em sua maioria, artefato da interpretação "sem consumo" |
| Status 1 (1 mês de atraso) quase ausente de agosto para trás | 25 (ago), 3 (jul), 2 (jun), 0 (mai, abr) no treino | categorias raras agrupadas nos testes estatísticos |

## Limitações

- **Tempo:** dados de 2005. Comportamento de crédito, juros e regras mudaram; o modelo não deve ser usado em produção hoje sem recalibração.
- **Geografia:** um único banco de Taiwan. Os resultados não se transferem diretamente para o Brasil.
- **Janela curta:** só 6 meses de histórico e um único mês de desfecho (out/2005), período próximo à crise de cartões de Taiwan (2005–2006), o que pode inflar a taxa de default.
- **Sem variáveis de renda, dívida total ou bureau de crédito**, que um modelo real usaria.
- **Atributos sensíveis** (sexo, idade, estado civil): usados na EDA e na checagem de equidade (Etapa 13). O uso deles como variável do modelo é decidido e justificado na modelagem.
- **Interpretação de `PAY_*` = -2 e 0** não é oficial; a leitura adotada vem do comportamento observado nos dados (EDA).

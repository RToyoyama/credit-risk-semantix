# Model card — PD de cartão de crédito (XGBoost)

> Um *model card* (ficha do modelo) é a "bula" de um modelo: para que ele serve, para que **não** serve, como foi avaliado e quais são os riscos. Termos técnicos estão no [Glossário](GLOSSARIO.md). Todos os números vêm de arquivos em [`reports/`](../reports/).

## 1. Identificação

| | |
|---|---|
| Nome | PD de cartão de crédito — risco de calote no próximo mês |
| Algoritmo | XGBoost (árvores de decisão em sequência, cada uma corrigindo os erros das anteriores), hiperparâmetros em [`best_params.json`](../reports/best_params.json) |
| Versão | a do pacote `credit-risk` (`pyproject.toml`); a API devolve a versão em toda resposta e a data de treino em `/model-info` |
| Código | `models.final_pipeline()` → treino em `final.run_final()` → uso em `scoring.score_clients()` |
| Autor | Renan Toyoyama (projeto EBAC × Semantix) |
| Licença | código MIT; dados CC BY 4.0 |

## 2. Uso pretendido

- **O que faz:** estima a **PD** (probabilidade de o cliente não pagar a próxima fatura) de clientes de cartão de crédito, a partir de 6 meses de histórico de pagamento, faturas, pagamentos, limite e escolaridade.
- **Para que serve:** priorizar, todo mês, **ações preventivas** (contato, renegociação, revisão de limite) numa carteira com capacidade limitada. A PD é combinada com a exposição (perda esperada e valor esperado da ação) em `scoring.py`, que também gera score 0–1000, faixa A–E e motivos.
- **Quem usa:** analistas de risco e gestores de carteira, via API (`/score`) ou dashboard.

## 3. Uso NÃO pretendido

- **Aprovar ou negar crédito a novos clientes:** o modelo só viu clientes que já tinham cartão e histórico. Não serve para quem pede crédito pela primeira vez.
- **Decisão automática sem revisão humana** que prejudique o cliente (corte de limite, cobrança): o modelo prioriza; a decisão final fica com uma pessoa e uma política aprovada.
- **Outro país, outra época ou outro produto** sem retreinar e revalidar: os dados são de Taiwan, 2005.
- **Explicar causas:** o modelo encontra associações. "Limite baixo" está ligado a mais risco porque o banco deu limite baixo a quem já achava arriscado; aumentar o limite **não** reduz o risco.

## 4. Dados

| | |
|---|---|
| Fonte | *Default of Credit Card Clients* (Yeh & Lien, 2009), UCI, dataset 350 — ficha em [`data_card.md`](data_card.md) |
| Treino | 24.000 clientes (22,12% de calote) |
| Holdout (prova lacrada) | 6.000 clientes (22,12%), IDs em [`holdout_ids.csv`](../reports/holdout_ids.csv), usado **uma única vez** |
| Entradas | limite, escolaridade, status de pagamento (6 meses), faturas (6), pagamentos (6) e 13 variáveis de resumo calculadas por regras fixas |
| **Não usa** | **sexo, estado civil e idade** (retirados sem perda de desempenho; ver Seção 6). A API recusa esses campos (erro 422) |

## 5. Desempenho

**Validação cruzada** (5 partes × 3 repetições, só no treino) × **holdout** (avaliado uma vez):

| Métrica | O que mede | CV (média) | Holdout (IC 95%) |
|---|---|---|---|
| PR AUC | quão bem acha os caloteiros (referência sem modelo: 0,22) | 0,566 | **0,560** (0,531–0,589) |
| ROC AUC | quão bem ordena caloteiro × bom pagador (0,5 = sorte) | 0,788 | 0,782 (0,767–0,796) |
| KS | maior separação entre as duas curvas, padrão em crédito | 0,441 | 0,427 (0,402–0,457) |
| Brier | erro das probabilidades (menor = melhor) | 0,133 | 0,134 (0,129–0,140) |

As quatro métricas do holdout ficaram dentro da faixa prevista pela CV.

**Resultado de negócio no holdout** (acionando 10% dos clientes, 600 de 6.000):

| Política | Valor líquido |
|---|---|
| Maior valor esperado (modelo: PD × exposição) | **NT$ 7,52 mi** |
| Regra de atraso (sem modelo) | NT$ 5,17 mi |
| Diferença | **+NT$ 2,36 mi (+46%)**, IC 95%: +1,59 a +3,15 mi |

**Calibração:** sem correção (sigmoid e isotônica foram testadas e rejeitadas na Etapa 12). A perda esperada prevista no holdout (NT$ 39,4 mi) ficou 2,3% abaixo da real (NT$ 40,4 mi).

## 6. Equidade

Sexo e idade **não entram** no modelo, mas são usados para **auditar** os resultados no holdout. Critérios fixados antes: diferença entre a PD média e a taxa real ≤ 3 p.p. em cada grupo; ROC AUC variando ≤ 0,05 entre grupos.

| Grupo | Taxa real | PD média | Diferença | ROC AUC | % acionados |
|---|---|---|---|---|---|
| Feminino | 21,3% | 21,6% | +0,4 p.p. | 0,786 | 9,4% |
| Masculino | 23,4% | 22,7% | −0,6 p.p. | 0,774 | 10,8% |
| 21–30 anos | 22,0% | 23,2% | +1,2 p.p. | 0,790 | 9,1% |
| 31–45 anos | 21,2% | 20,8% | −0,3 p.p. | 0,776 | 10,7% |
| 46+ anos | 25,3% | 23,2% | −2,0 p.p. | 0,781 | 9,9% |

**Aprovado** (maior diferença 2,0 p.p.; maior variação de ROC AUC dentro de um atributo: 0,014). **Limite:** a auditoria olha resultados; não descarta discriminação indireta por variáveis ligadas ao grupo (*proxies*), como o limite de crédito.

## 7. Riscos e limitações conhecidas

| Risco | Efeito | Mitigação |
|---|---|---|
| Dados antigos e de outro país (Taiwan, 2005, perto de uma crise de cartões) | o modelo treinado não se transfere; o método sim | retreinar com dados locais antes de qualquer uso |
| Premissas de negócio não medidas (LGD 60%, eficácia da ação 30%, custos NT$ 100 + 300) | o valor em dinheiro depende delas; a eficácia é a que mais pesa | sensibilidade: o modelo vence a regra em 12 de 12 cenários; medir a eficácia real com um piloto com grupo de controle |
| Pior desempenho **dentro** do atraso crônico (ROC AUC 0,68 × 0,73–0,74) | ordena mal os clientes já muito arriscados e subestima o risco do grupo em 3,8 p.p. | esses clientes já caem quase todos na faixa E e são acionados |
| Mudança do público (crise, nova campanha) | o modelo "envelhece" | monitoramento mensal: PSI + PD média + participação da faixa E (`monitoring.py`) |
| O PSI sozinho não percebeu uma crise simulada | falso "estável" | indicadores complementares de negócio (PD média > 1,2×, faixa E > 1,5×) |
| Só 6 meses de histórico, sem renda nem dados de birô | teto de desempenho mais baixo que o de um modelo real | limitação de fonte; documentada |

## 8. Explicabilidade

Cada cliente recebe até **3 motivos** em português (`reasons.py`), calculados por TreeSHAP (a contribuição exata de cada variável para a PD daquele cliente) e agrupados em 7 conceitos de negócio. Um fator só vira motivo se aumentar a chance de calote em pelo menos 5%. Na faixa E, o 1º motivo é sempre atraso.

## 9. Como reproduzir

`make up` (Docker) ou `make data split final` (local). `make docker-reproduce` reconstrói tudo do zero e compara as métricas do holdout com as publicadas (diferença obtida: 0,0).

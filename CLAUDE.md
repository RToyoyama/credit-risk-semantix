# PROJETO: Risco de Crédito Orientado a Custo — Entrega Semantix

Você é meu **tutor e par de programação** neste projeto. Leia este arquivo inteiro antes de qualquer ação e siga-o em todas as sessões.

═══════════════════════════════════════
## 0. CONTEXTO
═══════════════════════════════════════

**Quem sou eu:** Renan, estudante de Ciência de Dados (EBAC), GitHub `RToyoyama`. Já concluí um projeto completo (Titanic/Kaggle) com esta mesma arquitetura: pacote `src/`, testes, ruff, pre-commit, Makefile, Docker, Conventional Commits, um commit por etapa.

**O que é esta entrega:** projeto de parceria EBAC × Semantix ("Projeto para Cientista de Dados"). Os melhores trabalhos entram no processo seletivo da Semantix. Clientes da Semantix incluem PicPay, Boa Vista SCPC, next e Vindi, então um problema de **crédito/fintech** conversa diretamente com o negócio deles.

**Briefing oficial (requisitos obrigatórios):**
1. Dissertar sobre um problema real e justificar por que dados e ML ajudam a resolvê-lo.
2. Usar fontes de dados **públicas e não confidenciais**, devidamente referenciadas.
3. Análise exploratória (EDA) com os pontos relevantes para o problema.
4. Modelo de ML ou estatístico adequado ao problema, avaliado com métricas adequadas.
5. Visualização de dados que comunique os resultados (Data Storytelling).
6. Repositório no GitHub com todo o projeto.
7. Documentação em Markdown ou PDF com os tópicos **Coleta de dados, Modelagem e Conclusões**.
8. Entrega na plataforma: **o link do repositório no GitHub**. O repositório precisa se sustentar sozinho: quem abrir o link deve entender o problema, os resultados e como rodar, sem nenhum material extra.

**Critérios de excelência:** profundidade na justificativa das escolhas; rigor metodológico (comparação de modelos, validação cruzada, tuning, **sem vazamento de dados**); Data Storytelling; engenharia e reprodutibilidade em padrão de portfólio profissional.

**Conteúdo do curso que o projeto deve evidenciar:** combinação de modelos, álgebra linear, **PCA**, **K-means**, **probabilidade**, **inferência estatística**, **regressão**, validação cruzada, **XGBoost × Árvore de Decisão**.

═══════════════════════════════════════
## 1. O PROJETO (abordagem escolhida)
═══════════════════════════════════════

**Problema de negócio:** uma emissora de cartão de crédito precisa decidir, a cada mês, quais clientes têm alto risco de inadimplência no próximo pagamento, para agir antes (renegociação, redução de limite, contato preventivo). Errar tem **custos assimétricos**:
- **Falso negativo** (não prever um inadimplente): perda do saldo devido.
- **Falso positivo** (acionar um bom pagador): custo da ação + atrito com o cliente / receita perdida.

**Diferencial do projeto:** a entrega final **não é uma acurácia**. É uma **política de crédito para a carteira**, com a narrativa **"do risco individual à política de carteira"**:

| Pergunta de negócio | Componente | Etapa |
|---|---|---|
| Quem tem risco? | Probabilidade de default (**PD**) calibrada | 9–13 |
| Quanto isso custa? | **Perda esperada** em dinheiro: EL = PD × LGD × EAD | 14 |
| O que fazer com cada cliente? | **Score 0–1000** e **faixas de rating A–E**, cada uma com uma ação | 14 |
| Por que essa decisão? | **Motivos da decisão** (*reason codes*) por cliente | 15 |
| Como usar em produção? | **API de scoring** (FastAPI) servindo score, faixa e motivos | 17 |
| Quando o modelo envelhece? | **Monitoramento de drift** com PSI | 16 |
| Como o gestor decide? | **Dashboard** com simulação de custos, faixas e carteira | 18 |

Os segmentos de clientes (PCA + K-means) dão o contexto interpretável que atravessa todas essas peças.

**Dados:** *Default of Credit Card Clients* (Yeh & Lien, 2009). 30.000 clientes de Taiwan (abr–set/2005), 23 variáveis, alvo binário `default payment next month` na UCI (com espaços; ~22% positivos), renomeado para `default` na limpeza.
- Fonte oficial: UCI Machine Learning Repository, dataset 350: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients (licença CC BY 4.0).
- **Download direto da UCI, sem Kaggle:** `https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip` (zip com `default of credit card clients.xls`). Validar o SHA-256 do zip (`56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602`, conferido em 2026-11-03). Ler o `.xls` com `engine="calamine"` (`python-calamine`; o `xlrd` está depreciado no pandas 3.1); a planilha tem **duas linhas de cabeçalho** (`X1..X23` e os nomes reais), usar `header=1`. Motivo: o avaliador roda `make data` sem precisar de conta ou token.
- **Plano B (só documentado no `data_card.md`, fora do código):** espelho no Kaggle `uciml/default-of-credit-card-clients-dataset`, para o caso de o link da UCI sair do ar.
- Citação: Yeh, I. C., & Lien, C. H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients.* Expert Systems with Applications, 36(2), 2473–2480.

**Peculiaridades conhecidas dos dados (tratar e documentar):**
- `EDUCATION` tem valores não documentados (0, 5, 6) → agrupar em "outros" e justificar.
- `MARRIAGE` tem o valor 0, não documentado → agrupar em "outros".
- `PAY_0..PAY_6` usam -2 e -1 (pagou; a EDA mostrou que -2 = **quitou a fatura**, não "sem consumo"), 0 (crédito rotativo) e 1–9 (meses de atraso). Repare que a sequência pula `PAY_1`: renomeie as colunas para um padrão consistente.
- Valores monetários em NT$ (dólar taiwanês); `BILL_AMT` pode ser negativo (crédito a favor do cliente).
- Os dados são de 2005, de outro país: registrar como limitação (distribuição temporal e geográfica).

**Como cada conteúdo do curso aparece:**

| Conteúdo | Onde aparece no projeto |
|---|---|
| Probabilidade e inferência | Intervalos de confiança (bootstrap) da taxa de default; testes qui-quadrado e Mann-Whitney com tamanho de efeito (Cramér's V, rank-biserial) na EDA |
| Álgebra linear e PCA | Redução das séries de fatura/pagamento; variância explicada; visualização dos segmentos |
| K-means | Segmentação comportamental **sem o alvo**, com perfil de risco de cada segmento |
| Regressão | Regressão logística como baseline interpretável (coeficientes → razão de chances) |
| XGBoost × Árvore de Decisão | Duelo central dos modelos, mais Random Forest |
| Validação cruzada e tuning | `RepeatedStratifiedKFold`, comparação pareada, seed separada na busca |
| Combinação de modelos | Ensemble (votação soft / stacking), avaliado contra o melhor modelo individual |

═══════════════════════════════════════
## 2. PROTOCOLO DO TUTOR (seguir SEMPRE)
═══════════════════════════════════════

- Antes do código: **conceito + por quê + boas práticas + quando usar vs quando NÃO usar**. Teoria: **máx. 10 linhas por bloco**.
- Projeto dividido em **ETAPAS numeradas**. Antes de cada etapa, exiba um **mini-roadmap**.
- Ao fim de cada etapa: código + validação + **commit**. Etapa grande → Parte A e Parte B, cada uma com seu commit.
- **Só avance para a próxima etapa após minha confirmação.** Etapas marcadas `[trivial]` podem ser executadas sem pedir.
- Nível padrão: **intermediário** (eu altero com `NÍVEL: iniciante | intermediário | avançado`).
- Respostas diretas, sem floreios (direto não significa difícil: veja a regra de linguagem abaixo); não repita o que já foi dito; código limpo sem comentários óbvios; o trivial apenas cite.

**Linguagem: simples primeiro, técnica só quando necessário.**
Eu aprendo melhor quando entendo a ideia antes do nome dela. Em todas as explicações (no chat, nos commits, nos notebooks, no README e no relatório):
- **Comece pela ideia em palavras do dia a dia**, de preferência com uma analogia ou um exemplo concreto com números pequenos. Só depois apresente o nome técnico.
- **Use termos técnicos apenas quando forem realmente necessários** (porque é assim que o mercado chama, ou porque vou precisar pesquisar o termo depois).
- **Todo termo técnico, sigla ou nome alternativo é explicado na primeira vez que aparece**, no formato: *nome (o que é, em uma frase simples)*. Exemplo: "vazamento de dados (*data leakage*): quando o modelo, sem querer, tem acesso a uma informação que não teria no mundo real, como 'colar na prova'".
- **Quando a mesma coisa tiver nomes diferentes**, diga isso explicitamente. Exemplos: "probabilidade de default, que o mercado chama de PD"; "faixa de risco, também chamada de *rating*".
- **Uma ideia por vez.** Se um conceito depende de outro, explique o de base primeiro.
- **Explique o "para quê" antes do "como".** Antes de mostrar uma fórmula ou um trecho de código, diga que problema ele resolve.
- **Fórmulas sempre acompanhadas de um exemplo numérico.** Exemplo: "EL = PD × LGD × EAD. Se um cliente tem 20% de chance de não pagar, deve NT$ 10.000 e perderíamos 60% disso: 0,20 × 0,60 × 10.000 = NT$ 1.200 de perda esperada."
- **Ao interpretar resultados, traduza para impacto prático**: o que o número significa para o negócio e o que muda na decisão, não só o valor.
- **Mantenha um glossário vivo** em `docs/GLOSSARIO.md`: cada termo técnico novo entra com a explicação simples, o nome em inglês (quando houver) e a etapa onde apareceu. O glossário é linkado no README.
- Se eu disser **"não entendi"** ou **"explica de novo"**, reexplique por outro caminho (outra analogia, outro exemplo), sem repetir o mesmo texto.

**Como trabalhar no Claude Code:**
- Você **cria os arquivos e roda os comandos**, mas explica as decisões não óbvias de cada arquivo (tabela "escolha → motivo" quando houver várias).
- Depois de rodar um experimento, **mostre os resultados reais e interprete comigo** antes do commit: ganho, variância, win rate, o que muda nas decisões.
- **Nunca invente números.** Todo número em Markdown, notebook, README ou dashboard vem de uma saída real. Se ainda não rodou, deixe um marcador `TODO` e avise.
- Antes de commitar, mostre a mensagem e **peça minha confirmação**. Eu posso querer revisar o diff.
- Se um resultado contrariar a hipótese, diga isso com clareza. Resultado negativo também é resultado e vai documentado.

**Comandos rápidos:**
- `/e` → pula a explicação, entrega só o código
- `/skip` → etapa trivial, executa sem confirmação
- `/readme` → gera o README.md consolidado até aqui
- `/log` → exibe todos os commits do projeto em ordem cronológica

**Formato de commit (Conventional Commits):**
```
tipo(escopo)[!]: descrição curta em inglês

Descrição em português.
[Resultado: números relevantes da etapa, quando houver]
```
Tipos: feat | fix | refactor | docs | chore | test | ci | perf. `!` = breaking change.
Sempre: `git add -A && pre-commit run --all-files; git add -A` antes do commit (os hooks podem corrigir arquivos).
**Nunca** adicionar `Co-Authored-By` (nem outra atribuição a IA) em commits ou PRs: o Claude não deve aparecer como contribuidor do repositório.

═══════════════════════════════════════
## 3. STACK E PADRÕES
═══════════════════════════════════════

- **Python 3.11** (via `uv`), pandas, numpy, scikit-learn, xgboost, imbalanced-learn, scipy, statsmodels, matplotlib, seaborn, plotly, streamlit, joblib, shap, fastapi, uvicorn, pydantic (v2), python-calamine (ler o `.xls` da UCI; `xlrd` está depreciado no pandas 3.1).
- Dev: ruff, pytest, pre-commit, ipykernel, httpx2 (para o `TestClient` do FastAPI; o Starlette depreciou o `httpx` nesse uso), nbformat e nbclient (execução headless dos notebooks).
- `snake_case`, **type hints obrigatórios**, docstrings no **padrão Google**.
- Arquitetura de dados em camadas: `raw → interim (limpo) → processed (features)`, mais `reports/` para resultados.
- Não usar PySpark/dbt/Airflow: 30 mil linhas cabem em memória. Justifique isso no README como decisão consciente, não como lacuna.
- **Orquestração (decidido na Etapa 16):** sem Airflow no projeto. `monitoring.run_monitoring(lote)` é independente de orquestrador; o agendamento real é um workflow `schedule` (cron mensal) no GitHub Actions (Etapa 19); o relatório traz o desenho (Mermaid) de como seria uma DAG Airflow em produção (Etapa 20).

**Configuração do ruff (aprendida no projeto anterior):**
```toml
[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "N", "D", "ANN"]
ignore = ["D104", "D107", "N803", "N806"]  # N803/N806: convenção sklearn (X maiúsculo)
unfixable = ["F401"]                        # não apagar imports sozinho

[tool.ruff.lint.pydocstyle]
convention = "google"

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["D", "ANN"]
"*.ipynb" = ["D", "ANN", "E402"]
"app/*" = ["D"]
"api/*" = ["B008"]                         # Depends()/Query() em defaults é o padrão do FastAPI
```

═══════════════════════════════════════
## 4. MEU AMBIENTE E ARMADILHAS JÁ CONHECIDAS
═══════════════════════════════════════

**Ambiente:** Windows + **WSL 2 (Ubuntu)**, VS Code, projetos em `~/projetos/`. Docker Desktop com integração WSL ativa. `gh` disponível.

Previna estes problemas, que já aconteceram comigo:

| Armadilha | Prevenção |
|---|---|
| `python3.11` não existe no apt da minha distro | Criar o venv com `uv venv --python 3.11 --seed .venv` |
| Commit falha porque o hook modificou arquivos | Sempre `pre-commit run --all-files; git add -A` antes do commit |
| Hook do ruff com ID antigo (`ruff`) | Usar `ruff-check` e `ruff-format` no `.pre-commit-config.yaml` |
| `ruff --fix` apagou import que seria usado depois | `unfixable = ["F401"]` |
| Notebook aberto no VS Code sobrescreve a formatação do hook | Fechar a aba do notebook antes de rodar os hooks |
| Makefile com espaços no lugar de TAB | Gerar com TAB real e validar com `make -n` |
| `SVC(probability=True)` depreciado no scikit-learn 1.9 | Usar `CalibratedClassifierCV(SVC(), ensemble=False)` |
| `roc_auc` quebra em modelos sem `predict_proba`/`decision_function` | Avaliação deve pular a métrica e registrar `NaN` |
| `train_score` inflado quando há encoders que usam o alvo | Documentar; não usar gap de overfitting como medida absoluta |
| `fit_time` em CSV versionado gera diff falso a cada execução | Salvar tempos em arquivo separado, fora do Git |
| Docker no WSL: comando não encontrado | Ativar *Settings → Resources → WSL Integration* |
| `nvidia-nccl-cu12` não existe em ARM | Documentar `--platform linux/amd64` para Mac Apple Silicon |
| Commit B "vazio" porque o notebook não estava salvo | Conferir `git status` antes de commitar |

═══════════════════════════════════════
## 5. REGRAS METODOLÓGICAS (inegociáveis)
═══════════════════════════════════════

1. **Holdout intocado.** Separe 20% estratificado (seed fixa) logo após a limpeza e salve os IDs. Ele só é usado **uma vez**, na avaliação final (Etapa 13). Toda escolha (features, modelo, tuning, limiar) é feita com CV no treino. Isso resolve a lição do projeto anterior: lá a CV dizia 0.85 e o leaderboard 0.78, sem um teste local confiável.
2. **Toda estatística aprendida dos dados fica dentro do pipeline** (imputação, escala, PCA, K-means quando usado como feature, SMOTE, encoders).
3. **Métricas adequadas ao desbalanceamento (~22%):** ROC AUC, **PR AUC (average precision)**, **KS** (padrão em crédito), **Brier score** (calibração) e, na decisão, **lucro/custo esperado**. Mostre explicitamente por que a acurácia engana aqui (o modelo "ninguém dá default" acerta ~78%).
4. **CV:** `RepeatedStratifiedKFold` (5×3, seed 42) para avaliar; seed diferente para a busca de hiperparâmetros; **comparação pareada fold a fold** com win rate.
5. **Critério de decisão definido antes de ver o resultado** em todo experimento (ex.: "adoto SMOTE só se PR AUC subir com win rate > 60%").
6. **O limiar de decisão, os cortes das faixas de rating e a referência do PSI são definidos com o treino** (previsões out-of-fold), nunca no holdout. O holdout só **confirma**.
7. **Uma única fonte de verdade para o scoring.** O módulo `scoring.py` (PD → EL → score → faixa → motivos) é usado igualmente pela API, pelo dashboard e pela avaliação. Nenhuma regra de negócio duplicada.
8. **Premissas de negócio explícitas.** LGD, custos de ação e cortes das faixas ficam em `config.py`, com justificativa no relatório e análise de sensibilidade. Nada de número mágico.
9. **Reprodutibilidade:** seeds em `config.py`, `requirements.lock`, Docker validado reproduzindo as métricas.

═══════════════════════════════════════
## 6. ESTRUTURA DO PROJETO
═══════════════════════════════════════

```
credit-risk-semantix/
├── .github/workflows/ci.yml     # lint + testes a cada push
├── api/
│   ├── main.py                  # FastAPI: /health, /score, /score/batch, /model-info
│   └── schemas.py               # contratos de entrada e saída (pydantic)
├── app/
│   └── dashboard.py             # Streamlit (storytelling + simulador)
├── data/
│   ├── raw/                     # CSV original (não versionado)
│   ├── interim/                 # dados limpos (não versionado)
│   └── processed/               # features + split (não versionado)
├── docs/
│   ├── RELATORIO.md             # Coleta de dados | Modelagem | Conclusões
│   ├── data_card.md             # fonte, licença, dicionário, limitações
│   ├── model_card.md            # uso pretendido, métricas, riscos, vieses
│   ├── GLOSSARIO.md             # termos técnicos explicados de forma simples
│   └── images/                  # capturas do dashboard exibidas no README e no relatório
├── models/                      # .joblib (não versionado)
├── notebooks/
│   ├── 01_eda_inferencia.ipynb
│   ├── 02_segmentacao.ipynb
│   └── 03_modelagem_e_decisao.ipynb   # narrativa final, importa o pacote
├── reports/
│   ├── figures/
│   └── *.csv / *.json           # resultados de cada experimento
├── src/credit_risk/
│   ├── __init__.py
│   ├── config.py                # caminhos, seeds, colunas, custos padrão
│   ├── data.py                  # download, leitura, validação de schema
│   ├── cleaning.py              # categorias não documentadas, renomeação
│   ├── split.py                 # holdout estratificado reprodutível
│   ├── stats.py                 # IC bootstrap, testes, tamanhos de efeito
│   ├── features.py              # features determinísticas
│   ├── segmentation.py          # PCA + K-means, perfis de segmento
│   ├── preprocessing.py         # ColumnTransformer
│   ├── models.py                # catálogo de modelos + build_pipeline
│   ├── evaluate.py              # CV repetida, métricas, comparação pareada
│   ├── balancing.py             # nenhum × class_weight × SMOTE
│   ├── tuning.py                # Grid/RandomizedSearchCV, best_params.json
│   ├── decision.py              # calibração, curva de custo, limiar ótimo
│   ├── explain.py               # permutation importance, PDP, odds ratios
│   ├── expected_loss.py         # EAD, LGD, EL por cliente e por carteira
│   ├── scoring.py               # PD → score 0–1000 → faixa A–E → ação
│   ├── reasons.py               # motivos da decisão por cliente (SHAP)
│   ├── monitoring.py            # PSI por feature e do score, relatório de drift
│   └── final.py                 # treino final + avaliação única no holdout
├── tests/
├── .dockerignore
├── .gitignore
├── .pre-commit-config.yaml
├── Dockerfile
├── docker-compose.yml           # api + dashboard
├── Makefile
├── pyproject.toml
├── requirements.lock
└── README.md
```

═══════════════════════════════════════
## 7. ROADMAP DAS ETAPAS
═══════════════════════════════════════

**Antes da Etapa 1, entregue:** (1) a estrutura de pastas acima, (2) o checklist de pré-requisitos com versões, (3) o `.gitignore` contextual (dados, modelos, `.venv`, caches, `.env`, com exceções `!**/.gitkeep`). Não há material de módulos do curso: usar o conhecimento geral de ciência de dados.

| Etapa | Entrega | Pontos-chave |
|---|---|---|
| **1** | Repositório + problema de negócio | `git init -b main`, estrutura, `gh repo create`. `docs/RELATORIO.md` já nasce com a seção do problema: contexto, decisão a apoiar, custos assimétricos, por que ML (vs. regra fixa), métrica de sucesso de negócio |
| **2** | Ambiente | Parte A: uv, `pyproject.toml`, `pip install -e ".[dev]"`, lock. Parte B: ruff, pre-commit, Makefile, `.vscode/settings.json`, teste de fumaça |
| **3** | Ingestão + data card | `make data` baixando direto da UCI (urllib + checagem SHA-256); `data.py` com validação de schema (colunas, tipos, faixas, nulos, IDs únicos) e testes com dados sintéticos; `docs/data_card.md` com fonte, licença, citação e dicionário de variáveis |
| **4** | Limpeza + holdout | `cleaning.py` (categorias não documentadas, renomeação `PAY_0→PAY_1`, flags de inconsistência); `split.py` com holdout 20% estratificado salvo em disco; testes garantindo que treino e holdout não se cruzam |
| **5** | EDA com inferência | Notebook `01`. Taxa de default com IC bootstrap 95%; default por sexo, educação, estado civil, idade e limite com testes (qui-quadrado, Mann-Whitney) **e tamanho de efeito**; correção para múltiplas comparações (Holm); análise do histórico `PAY_*` (o sinal mais forte esperado). Cada análise termina em **Hipótese → Evidência → Decisão**. Discutir que significância ≠ relevância com n=24 mil. **Só no treino** |
| **6** | Segmentação (PCA + K-means) | Parte A: features comportamentais **sem alvo** (utilização, atraso, padrão de pagamento), padronização, PCA (variância explicada, cargas). Parte B: K-means com k por cotovelo + silhueta, estabilidade entre seeds, **perfil e nome de negócio** para cada segmento, taxa de default por segmento (descritiva). Experimento: o segmento como feature melhora o modelo? (ajustado dentro do pipeline) |
| **7** | Feature engineering | Utilização do limite, máximo e média de atraso, meses em atraso, tendência da fatura, razão pagamento/fatura, volatilidade. Funções puras + testes. Separar o que é determinístico do que aprende dos dados |
| **8** | Pré-processamento | `ColumnTransformer` com saída pandas; escala só para modelos sensíveis; one-hot com `handle_unknown="ignore"`; testes de clonagem e de dados não vistos |
| **9** | Duelo baseline | Regressão logística × Árvore de Decisão × Random Forest × XGBoost. ROC AUC, PR AUC, KS, Brier, acurácia (para mostrar por que engana). Comparação pareada e boxplot. Interpretação dos coeficientes da logística como razão de chances |
| **10** | Balanceamento | Nenhum × `class_weight` × SMOTE (imblearn, só no fit). Mostrar o efeito na **calibração** (Brier, curva de confiabilidade): balancear distorce as probabilidades, o que importa para a decisão por custo |
| **11** | Tuning + ensemble | Parte A: Grid (logística, árvore) e Randomized (RF, XGBoost) com seed de busca separada; medir o otimismo da busca. Parte B: diversidade dos erros OOF e ensemble (votação soft / stacking) contra o melhor individual |
| **12** | Calibração + política de decisão | `CalibratedClassifierCV` (sigmoid × isotônica, Brier, curva de confiabilidade). Custo **dependente do cliente**: o custo de um falso negativo é a perda daquele cliente (LGD × EAD), e o de um falso positivo é o custo da ação + margem perdida (premissas em `config.py`). Curva de **resultado esperado × limiar** com previsões OOF; limiar ótimo comparado com 0.5 e com "não agir em ninguém"; **análise de sensibilidade** à LGD e aos custos |
| **13** | Avaliação final no holdout | Treino final no treino completo, **uma única avaliação no holdout**: métricas com IC bootstrap, matriz de confusão no limiar escolhido, resultado financeiro, comparação CV × holdout. Interpretabilidade global: permutation importance, PDP, desempenho por segmento e **checagem de equidade** por sexo/idade. Salvar `models/` e `reports/` |
| **14** | Perda esperada + score e faixas | **Parte A — Perda esperada:** EAD = fatura atual (tratando saldo negativo como zero), LGD como premissa justificada (ex.: 60%) com sensibilidade, EL = PD × LGD × EAD por cliente; ranking da carteira por EL × por PD (mostrar quantos clientes mudam de prioridade e quanto de perda o top-10% concentra). **Parte B — Score e rating:** transformação monotônica PD → score 0–1000 (escala log-odds, com *points to double the odds* documentado); 5 faixas A–E com cortes definidos no OOF do treino; validar **monotonicidade** da taxa de default entre faixas e estabilidade no holdout; tabela de política (faixa → ação → volume → default → EL) |
| **15** | Motivos da decisão | `reasons.py`: os 3 fatores que mais aumentam o risco de cada cliente, via SHAP (`TreeExplainer` se o modelo final for de árvores; contribuição coef × valor padronizado se for a logística). Traduzir nomes técnicos para frases de negócio ("2 meses de atraso no último pagamento", "uso do limite acima de 90%"). Testes: motivos estáveis, apontando na direção do risco, e coerentes com a importância global |
| **16** | Monitoramento (PSI) | `monitoring.py`: PSI do score e das principais features, com a distribuição de treino como referência e faixas de leitura padrão do mercado (< 0.10 estável, 0.10–0.25 atenção, > 0.25 investigar). Validar no holdout (PSI deve ser baixo) e **simular drift** com uma subpopulação deslocada (ex.: só clientes jovens ou de limite baixo) para mostrar o alarme disparando. `make monitor` gera `reports/monitoring.json`. Discutir como isso viraria rotina mensal em produção |
| **17** | API de scoring (FastAPI) | Parte A: `api/schemas.py` (pydantic, validação de faixas e categorias, exemplos no OpenAPI) e `api/main.py` com `/health`, `/model-info` (versão, data de treino, métricas do holdout), `/score` (PD, EL, score, faixa, ação, motivos) e `/score/batch`; modelo carregado uma vez no startup; respostas versionadas. Parte B: testes com `TestClient` (contrato, erro 422 em entrada inválida, consistência com `scoring.py`), `make api`, documentação Swagger em `/docs` |
| **18** | Dashboard Streamlit (storytelling) | Páginas: (1) o problema e quanto a inadimplência custa; (2) o que os dados revelam; (3) os segmentos; (4) o modelo e a decisão, com **sliders de LGD, custos e limiar** atualizando o resultado em tempo real; (5) a carteira por faixa A–E com EL; (6) monitoramento (PSI); (7) simulador de cliente com score, faixa e **motivos**. Usa o mesmo `scoring.py` da API. Títulos que **afirmam o insight**, cor de destaque só no que importa, números sempre com contexto. `make app`. Publicar no Streamlit Community Cloud com o link no topo do README; capturas em `docs/images/` |
| **19** | Docker + CI | Dockerfile (python:3.11-slim, usuário não-root, lock, modo editável, `libgomp1`); `docker-compose.yml` subindo **api (8000) e dashboard (8501)** com healthcheck; `.dockerignore`; `make docker-<alvo>` e `make up`; validar reprodução das métricas no container. GitHub Actions rodando lint + testes (incluindo os da API) |
| **20** | Documentação | `docs/RELATORIO.md` com os três tópicos obrigatórios (**Coleta de dados**, **Modelagem**, **Conclusões**), incluindo as premissas de negócio, a política por faixa, o monitoramento e limitações; `docs/model_card.md`; notebook `03` narrativo em **primeira pessoa** (como se eu tivesse escrito), lendo resultados salvos com flag `RERUN_EXPERIMENTS`; README com **resumo executivo no topo** (problema, resultado em dinheiro, recomendação), badges, como rodar (Docker e local), exemplo de chamada da API, arquitetura com diagrama Mermaid, resultados e lições. Exportar o relatório em PDF (opcional) |
| **21** | Entrega | Revisão final do repositório como se eu fosse o avaliador (README, links, figuras renderizando, CI verde, dashboard e `docker compose up` funcionando), release/tag `v1.0.0`, post para o LinkedIn |

**Autorização LGPD (opcional, do material da Semantix):** se eu quiser que o trabalho seja apresentado à Semantix, o README deve ter uma seção com a declaração de cessão. Deixe **marcadores** para nome, RG e CPF; **nunca preencha nem versione dados pessoais**. Lembre-me de completar só no texto enviado pela plataforma, se eu optar por isso.

═══════════════════════════════════════
## 8. COMO COMEÇAR
═══════════════════════════════════════

Estou com o VS Code aberto e o terminal em `~/projetos/`. Comece pela entrega pré-Etapa 1 (estrutura, checklist, `.gitignore`), depois apresente a **Etapa 1** com o mini-roadmap e aguarde minha confirmação.

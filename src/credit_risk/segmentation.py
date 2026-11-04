"""Segmentação comportamental: padronização + PCA (+ K-means na Parte B).

Tudo que aprende com os dados (médias, desvios, direções do PCA) fica
dentro de um ``Pipeline``, para poder ser ajustado só no treino de cada fold.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from credit_risk import config
from credit_risk.features import behavior_features


def make_pca_pipeline(n_components: int | float | None = None) -> Pipeline:
    """Monta o pipeline dados limpos → features comportamentais → escala → PCA.

    Args:
        n_components: Componentes a manter (None mantém todos).

    Returns:
        Pipeline não ajustado.
    """
    return Pipeline(
        [
            ("behavior", FunctionTransformer(behavior_features)),
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=config.SEED)),
        ]
    )


def explained_variance_table(pca: PCA) -> pd.DataFrame:
    """Variância explicada por componente, acumulada e autovalor.

    Args:
        pca: PCA já ajustado sobre dados padronizados.

    Returns:
        Tabela indexada por ``PC1``, ``PC2``...
    """
    ratio = pca.explained_variance_ratio_
    return pd.DataFrame(
        {
            "variancia": ratio,
            "acumulada": np.cumsum(ratio),
            "autovalor": pca.explained_variance_,
        },
        index=[f"PC{i}" for i in range(1, len(ratio) + 1)],
    )


def choose_n_components(cumulative: pd.Series, target: float = config.PCA_VARIANCE_TARGET) -> int:
    """Menor número de componentes cuja variância acumulada atinge a meta.

    Args:
        cumulative: Variância explicada acumulada (crescente).
        target: Meta, entre 0 e 1.

    Returns:
        Número de componentes.
    """
    return int(np.searchsorted(cumulative.to_numpy(), target - 1e-12) + 1)


def loadings(pca: PCA, feature_names: list[str], n_components: int | None = None) -> pd.DataFrame:
    """Cargas: quanto cada feature original pesa em cada componente.

    Args:
        pca: PCA ajustado.
        feature_names: Nomes das features de entrada, na ordem.
        n_components: Quantos componentes mostrar (None = todos).

    Returns:
        Tabela features × componentes.
    """
    comps = pca.components_[:n_components]
    return pd.DataFrame(
        comps.T, index=feature_names, columns=[f"PC{i}" for i in range(1, len(comps) + 1)]
    )


class SegmentEncoder(BaseEstimator, TransformerMixin):
    """Atribui cada cliente a um segmento comportamental (PCA + K-means).

    Os rótulos do K-means são arbitrários; por isso, após o ajuste, os grupos
    são renomeados por regra: o de maior média de meses em atraso recebe o
    último código e os demais são ordenados pela utilização média. Com k = 3,
    isso dá 0 = quitadores, 1 = rotativo em dia, 2 = atraso crônico.

    Args:
        n_segments: Número de grupos.
        n_components: Componentes do PCA usados no agrupamento.
        one_hot: Se True, ``transform`` devolve colunas 0/1 por segmento.
        random_state: Semente do K-means.
    """

    def __init__(
        self,
        n_segments: int = config.N_SEGMENTS,
        n_components: int = config.PCA_N_COMPONENTS,
        one_hot: bool = False,
        random_state: int = config.SEED,
    ) -> None:
        self.n_segments = n_segments
        self.n_components = n_components
        self.one_hot = one_hot
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: object = None) -> "SegmentEncoder":
        """Ajusta padronização, PCA e K-means e fixa a ordem dos grupos.

        Args:
            X: Dados limpos (precisa das colunas de limite, status, fatura e pagamento).
            y: Ignorado (a segmentação não usa o alvo).

        Returns:
            O próprio objeto ajustado.
        """
        self.pca_ = make_pca_pipeline(self.n_components).fit(X)
        scores = self.pca_.transform(X)
        self.kmeans_ = KMeans(
            self.n_segments, n_init=config.KMEANS_N_INIT, random_state=self.random_state
        ).fit(scores)
        behavior = behavior_features(X)
        raw = self.kmeans_.labels_
        delay = (behavior[config.DELAY] > 0).sum(axis=1).groupby(raw).mean()
        util = behavior[config.UTILIZATION].mean(axis=1).groupby(raw).mean()
        chronic = int(delay.idxmax())
        order = [int(g) for g in util.drop(chronic).sort_values().index] + [chronic]
        self.mapping_ = {raw_label: new for new, raw_label in enumerate(order)}
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Segmento de cada cliente, já com os códigos padronizados.

        Args:
            X: Dados limpos.

        Returns:
            Vetor de inteiros entre 0 e ``n_segments - 1``.
        """
        raw = self.kmeans_.predict(self.pca_.transform(X))
        return np.vectorize(self.mapping_.get)(raw)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Segmento como coluna única ou como colunas 0/1.

        Args:
            X: Dados limpos.

        Returns:
            Matriz ``(n, 1)`` com o código, ou ``(n, n_segments)`` em one-hot.
        """
        seg = self.predict(X)
        if self.one_hot:
            return np.eye(self.n_segments)[seg]
        return seg.reshape(-1, 1)

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Nomes das colunas de saída.

        Args:
            input_features: Ignorado.

        Returns:
            Nomes das colunas geradas.
        """
        if self.one_hot:
            return np.array([f"segment_{i}" for i in range(self.n_segments)])
        return np.array(["segment"])


def kmeans_selection(
    scores: np.ndarray, ks: range = config.K_RANGE, seed: int = config.SEED
) -> pd.DataFrame:
    """Inércia, silhueta e tamanho do menor grupo para cada k.

    Args:
        scores: Componentes do PCA de cada cliente.
        ks: Valores de k a testar.
        seed: Semente do K-means e da amostra da silhueta.

    Returns:
        Uma linha por k.
    """
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(scores), min(config.SILHOUETTE_SAMPLE, len(scores)), replace=False)
    rows = []
    for k in ks:
        km = KMeans(k, n_init=config.KMEANS_N_INIT, random_state=seed).fit(scores)
        rows.append(
            {
                "k": k,
                "inercia": km.inertia_,
                "silhueta": silhouette_score(scores[sample], km.labels_[sample]),
                "menor_grupo": np.bincount(km.labels_).min() / len(scores),
            }
        )
    return pd.DataFrame(rows)


def choose_k(selection: pd.DataFrame, min_share: float = config.MIN_SEGMENT_SHARE) -> int:
    """Melhor silhueta entre os k em que todo grupo tem pelo menos ``min_share``.

    Args:
        selection: Saída de ``kmeans_selection``.
        min_share: Fração mínima de clientes no menor grupo.

    Returns:
        O k escolhido.

    Raises:
        ValueError: Se nenhum k respeitar o tamanho mínimo.
    """
    valid = selection[selection["menor_grupo"] >= min_share]
    if valid.empty:
        raise ValueError("Nenhum k respeita o tamanho mínimo de grupo.")
    return int(valid.loc[valid["silhueta"].idxmax(), "k"])


def stability(
    scores: np.ndarray,
    k: int,
    n_seeds: int = 5,
    n_subsamples: int = 10,
    frac: float = 0.8,
    seed: int = config.SEED,
) -> dict[str, float]:
    """Estabilidade do agrupamento medida pelo ARI.

    Compara (a) ajustes com sementes diferentes e (b) ajustes em subamostras
    de ``frac`` dos clientes, aplicados a todos, contra a referência.

    Args:
        scores: Componentes do PCA.
        k: Número de grupos.
        n_seeds: Sementes comparadas duas a duas.
        n_subsamples: Quantidade de subamostras.
        frac: Fração de clientes em cada subamostra.
        seed: Semente da referência e do sorteio das subamostras.

    Returns:
        ARI mínimo entre sementes e ARI mínimo e médio das subamostras.
    """
    labels = [
        KMeans(k, n_init=config.KMEANS_N_INIT, random_state=seed + s).fit_predict(scores)
        for s in range(n_seeds)
    ]
    seeds_ari = [
        adjusted_rand_score(labels[i], labels[j])
        for i in range(n_seeds)
        for j in range(i + 1, n_seeds)
    ]
    rng = np.random.default_rng(seed)
    sub_ari = []
    for _ in range(n_subsamples):
        idx = rng.choice(len(scores), int(frac * len(scores)), replace=False)
        km = KMeans(k, n_init=config.KMEANS_N_INIT, random_state=seed).fit(scores[idx])
        sub_ari.append(adjusted_rand_score(labels[0], km.predict(scores)))
    return {
        "ari_min_sementes": float(min(seeds_ari)),
        "ari_min_subamostras": float(min(sub_ari)),
        "ari_medio_subamostras": float(np.mean(sub_ari)),
    }


def segment_profile(df: pd.DataFrame, segments: np.ndarray) -> pd.DataFrame:
    """Perfil comportamental médio de cada segmento (sem o alvo).

    Args:
        df: Dados limpos.
        segments: Segmento de cada cliente (códigos padronizados).

    Returns:
        Uma linha por segmento, com nome, tamanho e médias de comportamento.
    """
    b = behavior_features(df)
    g = pd.Series(segments, index=df.index, name="segment")
    prof = pd.DataFrame(
        {
            "clientes": g.value_counts().sort_index(),
            "participacao": g.value_counts(normalize=True).sort_index(),
            "utilizacao_set": b["utilization_1"].groupby(g).mean(),
            "utilizacao_abr": b["utilization_6"].groupby(g).mean(),
            "paga_da_fatura": b[config.PAYMENT_RATIO].mean(axis=1).groupby(g).mean(),
            "quita_todo_mes": (b[config.PAYMENT_RATIO] >= 1).all(axis=1).groupby(g).mean(),
            "meses_com_atraso": (b[config.DELAY] > 0).sum(axis=1).groupby(g).mean(),
            "limite_mediano": df["limit_bal"].groupby(g).median(),
        }
    )
    prof.insert(0, "nome", prof.index.map(config.SEGMENT_NAMES))
    return prof

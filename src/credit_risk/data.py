"""Ingestão dos dados da UCI: download verificado, leitura e validação de schema.

Uso: ``python -m credit_risk.data`` (ou ``make data``).
"""

import hashlib
import io
import logging
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from credit_risk import config

logger = logging.getLogger(__name__)


class SchemaError(ValueError):
    """Os dados violam o schema esperado."""


@dataclass
class ValidationReport:
    """Resultado da validação: erros bloqueiam o uso; avisos só informam.

    Attributes:
        errors: Violações que tornam os dados inutilizáveis.
        warnings: Peculiaridades conhecidas, tratadas em etapas posteriores.
    """

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Indica se não há erros."""
        return not self.errors


def sha256sum(path: Path) -> str:
    """Calcula o SHA-256 de um arquivo, lendo em blocos.

    Args:
        path: Arquivo a verificar.

    Returns:
        O hash em hexadecimal.
    """
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_raw(
    url: str = config.RAW_URL,
    dest: Path = config.RAW_ZIP,
    expected_sha256: str = config.RAW_SHA256,
    force: bool = False,
) -> Path:
    """Baixa o zip da UCI e confere a integridade pelo SHA-256.

    Se o arquivo já existir com o hash correto, o download é pulado.

    Args:
        url: Endereço do arquivo.
        dest: Onde salvar o zip.
        expected_sha256: Hash esperado do arquivo.
        force: Baixa de novo mesmo que o arquivo exista.

    Returns:
        O caminho do zip verificado.

    Raises:
        ValueError: Se o hash do arquivo baixado não bater com o esperado.
    """
    if dest.exists() and not force and sha256sum(dest) == expected_sha256:
        logger.info("Zip já presente e íntegro: %s", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    logger.info("Baixando %s", url)
    with urllib.request.urlopen(url, timeout=config.DOWNLOAD_TIMEOUT_S) as resp:
        tmp.write_bytes(resp.read())

    actual = sha256sum(tmp)
    if actual != expected_sha256:
        tmp.unlink()
        raise ValueError(
            f"SHA-256 inesperado: {actual} (esperado {expected_sha256}). "
            "O arquivo da UCI pode ter mudado; veja o plano B em docs/data_card.md."
        )
    tmp.replace(dest)
    return dest


def read_raw_xls(zip_path: Path, member: str = config.RAW_XLS_NAME) -> pd.DataFrame:
    """Lê a planilha de dentro do zip, sem extraí-la para o disco.

    A planilha tem duas linhas de cabeçalho (``X1..X23`` e os nomes reais);
    usa-se a segunda.

    Args:
        zip_path: Caminho do zip da UCI.
        member: Nome do arquivo ``.xls`` dentro do zip.

    Returns:
        Os dados com os nomes de coluna originais.
    """
    with zipfile.ZipFile(zip_path) as zf:
        content = zf.read(member)
    return pd.read_excel(io.BytesIO(content), header=1, engine="calamine")


def validate_raw(df: pd.DataFrame) -> ValidationReport:
    """Confere colunas, tipos, nulos, unicidade do ID, categorias e faixas.

    Args:
        df: Dados com os nomes de coluna originais da UCI.

    Returns:
        Relatório com erros e avisos.
    """
    report = ValidationReport()

    missing = [c for c in config.RAW_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in config.RAW_COLUMNS]
    if missing:
        report.errors.append(f"Colunas ausentes: {missing}")
    if extra:
        report.errors.append(f"Colunas inesperadas: {extra}")
    if missing:
        return report

    non_int = [c for c in config.RAW_COLUMNS if not pd.api.types.is_integer_dtype(df[c])]
    if non_int:
        report.errors.append(f"Colunas não inteiras: {non_int}")

    n_null = int(df[config.RAW_COLUMNS].isna().sum().sum())
    if n_null:
        report.errors.append(f"{n_null} valores nulos")

    n_dup = int(df[config.ID_COL].duplicated().sum())
    if n_dup:
        report.errors.append(f"{n_dup} IDs duplicados")

    for col, documented in config.DOCUMENTED_CATEGORIES.items():
        known = config.KNOWN_UNDOCUMENTED.get(col, set())
        counts = df[col].value_counts()
        pairs = zip(counts.index.to_numpy(dtype=int), counts.to_numpy(), strict=True)
        undocumented = {int(v): int(n) for v, n in pairs if v not in documented}
        unknown = {v: n for v, n in undocumented.items() if v not in known}
        tolerated = {v: n for v, n in undocumented.items() if v in known}
        if unknown:
            report.errors.append(f"{col}: valores desconhecidos {unknown}")
        if tolerated:
            report.warnings.append(f"{col}: valores não documentados {tolerated}")

    for col, (low, high) in config.VALID_RANGES.items():
        below = int((df[col] < low).sum()) if low is not None else 0
        above = int((df[col] > high).sum()) if high is not None else 0
        if below or above:
            report.errors.append(f"{col}: {below} abaixo de {low}, {above} acima de {high}")

    return report


def ensure_valid(df: pd.DataFrame) -> ValidationReport:
    """Valida e interrompe se houver erro.

    Args:
        df: Dados com os nomes de coluna originais da UCI.

    Returns:
        O relatório (só avisos, se chegou aqui).

    Raises:
        SchemaError: Se houver qualquer erro de validação.
    """
    report = validate_raw(df)
    for w in report.warnings:
        logger.warning(w)
    if not report.ok:
        raise SchemaError("; ".join(report.errors))
    return report


def build_raw(force: bool = False) -> Path:
    """Executa a ingestão completa: download → leitura → validação → Parquet.

    Args:
        force: Baixa o zip de novo mesmo que já exista.

    Returns:
        O caminho do Parquet da camada raw.
    """
    zip_path = download_raw(force=force)
    df = read_raw_xls(zip_path)
    ensure_valid(df)
    df.to_parquet(config.RAW_PARQUET, index=False)
    logger.info("Raw salvo: %s (%d linhas)", config.RAW_PARQUET, len(df))
    return config.RAW_PARQUET


def load_raw() -> pd.DataFrame:
    """Lê a camada raw já validada.

    Returns:
        Os dados com os nomes de coluna originais.

    Raises:
        FileNotFoundError: Se ``make data`` ainda não foi executado.
    """
    if not config.RAW_PARQUET.exists():
        raise FileNotFoundError(f"{config.RAW_PARQUET} não existe. Rode `make data`.")
    return pd.read_parquet(config.RAW_PARQUET)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    build_raw()

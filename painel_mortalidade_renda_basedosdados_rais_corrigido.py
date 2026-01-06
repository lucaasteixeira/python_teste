# Painel Mortalidade × **Renda (IPEA)** — Pooling, FE e RE (one-way)

Versão do painel trocando **PIB municipal** por **renda (IPEAData)**, mantendo:

- taxa de mortalidade (óbitos/população) com zeros tratados;
- especificação em log;
- alternativa cíclica via HP por município (λ anual = 6,25);
- Pooling, FE (individual), FE (tempo) e RE (individual);
- erros‑padrão cluster por município.


# Pacotes
!pip install basedosdados linearmodels
import basedosdados as bd
import pandas as pd
import numpy as np

from IPython.display import display

from linearmodels.panel import PooledOLS, PanelOLS, RandomEffects, compare
from statsmodels.tsa.filters.hp_filter import hpfilter
from scipy import stats

## 1) Parâmetros


# Parâmetros
BILLING_PROJECT_ID = "dadoscagedartigo"   # BigQuery billing project (obrigatório p/ rodar no BD)
UFS = ["PR","SC","RS"]                    # ajuste aqui
ANO_INI = 2010
ANO_FIM = 2019

# Proxy de renda (substitui IPEAData):
# Usaremos RAIS (formal) via Base dos Dados, agregando município-ano.
# A renda final usada no painel será: renda_formal_pc = soma_remuneracao_formal / populacao.
RAIS_TABLE = "basedosdados.br_me_rais.microdados_vinculos"

# HP filter (anual): lambda padrão usado em aplicações anuais costuma ser 6.25 (Ravn & Uhlig, 2002)
USE_HP = True
HP_LAMBDA_ANUAL = 6.25

# Filtro de causas no SIM: ICD-10 começando com X ou Y (causas externas).
# Se quiser todas as causas, mude para False.
FILTRAR_CAUSAS_XY = True

print("UFs:", UFS, "| anos:", ANO_INI, "-", ANO_FIM)


## 2) Diretório de municípios (filtros e chaves)


ufs_sql = ",".join([f"'{uf}'" for uf in UFS])

query_munis = f"""
SELECT
  CAST(id_municipio AS STRING) AS id_municipio,
  sigla_uf
FROM `basedosdados.br_bd_diretorios_brasil.municipio`
WHERE sigla_uf IN ({ufs_sql})
"""

df_munis = bd.read_sql(query_munis, billing_project_id=BILLING_PROJECT_ID)
df_munis["id_municipio"] = df_munis["id_municipio"].astype(str).str.zfill(7)
df_munis.head(), df_munis.shape


## 3) RAIS (Base dos Dados): proxy de renda por município‑ano (substitui IPEAData)


# ------------------------------------------------------------
# 3.1) Descobrir automaticamente qual coluna de remuneração existe na RAIS
# (isso evita erro caso o schema mude)
# ------------------------------------------------------------
ufs_sql = ",".join([f"'{uf}'" for uf in UFS])
T = ANO_FIM - ANO_INI + 1

def pick_rais_wage_column(billing_project_id: str) -> str:
    # lista candidatos comuns (nomes variam por versão/tratamento)
    candidates = [
        "valor_remuneracao_media", "remuneracao_media",
        "valor_remuneracao_dezembro", "remuneracao_dezembro",
        "valor_remuneracao_media_sm", "remuneracao_media_sm",
        "vl_remuneracao_media", "vl_remuneracao_dezembro"
    ]
    q = """
    SELECT column_name
    FROM `basedosdados.br_me_rais.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name = 'microdados_vinculos'
    """
    cols = bd.read_sql(q, billing_project_id=billing_project_id)["column_name"].tolist()
    # match por igualdade primeiro
    for c in candidates:
        if c in cols:
            return c
    # match por aproximação (contém 'remuner' e 'media' ou 'dezembro')
    for c in cols:
        cl = c.lower()
        if "remuner" in cl and ("media" in cl or "dez" in cl):
            return c
    raise ValueError(
        "Não achei coluna de remuneração na RAIS.microdados_vinculos. "
        f"Colunas disponíveis (amostra): {cols[:50]}"
    )

WAGE_COL = pick_rais_wage_column(BILLING_PROJECT_ID)
print("Coluna de remuneração selecionada na RAIS:", WAGE_COL)

# ------------------------------------------------------------
# 3.2) Agregar RAIS para município-ano (massa salarial formal)
# - soma_remun: soma da remuneração (coluna escolhida) sobre vínculos
# - vinculos: contagem de vínculos (para diagnóstico)
# Obs.: a renda per capita será construída depois com a população.
# ------------------------------------------------------------
query_renda = f"""
SELECT
  CAST(id_municipio AS STRING) AS id_municipio,
  ano AS year,
  SUM(CAST({WAGE_COL} AS FLOAT64)) AS soma_remun,
  COUNT(1) AS vinculos
FROM `{RAIS_TABLE}`
WHERE ano BETWEEN {ANO_INI} AND {ANO_FIM}
  AND sigla_uf IN ({ufs_sql})
  AND {WAGE_COL} IS NOT NULL
  AND CAST({WAGE_COL} AS FLOAT64) > 0
GROUP BY 1,2
"""

df_renda = bd.read_sql(query_renda, billing_project_id=BILLING_PROJECT_ID)
df_renda["id_municipio"] = df_renda["id_municipio"].astype(str).str.zfill(7)
df_renda.head(), df_renda.shape


### 3.2 Preparar chaves e checagens


# Preparar chave e checagens rápidas
df_renda = df_renda.dropna(subset=["id_municipio","year","soma_remun"])
df_renda["year"] = df_renda["year"].astype(int)

# renda média formal (por vínculo) — útil p/ diagnóstico
df_renda["renda_media_formal"] = df_renda["soma_remun"] / df_renda["vinculos"].replace({0: np.nan})

print("Renda (RAIS) — anos:", df_renda["year"].min(), "-", df_renda["year"].max())
print("Municípios na renda (RAIS):", df_renda["id_municipio"].nunique())
df_renda.head()


## 4) SIM e População (Base dos Dados)


# -----------------------------
# 4) SIM (mortalidade) e População (Base dos Dados)
# -----------------------------

# Mortalidade (SIM) por município-ano
query_deaths = f"""
SELECT
  CAST(id_municipio_ocorrencia AS STRING) AS id_municipio,
  ano AS year,
  COUNT(1) AS deaths
FROM `basedosdados.br_ms_sim.microdados`
WHERE ano BETWEEN {ANO_INI} AND {ANO_FIM}
  AND sigla_uf IN ({ufs_sql})
  { "AND REGEXP_CONTAINS(causa_basica, r'^[XY]')" if FILTRAR_CAUSAS_XY else "" }
GROUP BY 1,2
"""

df_deaths = bd.read_sql(query_deaths, billing_project_id=BILLING_PROJECT_ID)
df_deaths["id_municipio"] = df_deaths["id_municipio"].astype(str).str.zfill(7)
df_deaths["year"] = df_deaths["year"].astype(int)

# População por município-ano (IBGE/Estimativas)
# Removido o filtro 'sigla_uf' daqui pois a tabela br_ms_populacao.municipio não possui essa coluna.
# O filtro será aplicado após o carregamento, usando os municípios já filtrados de df_munis.
query_pop = f"""
SELECT
  CAST(id_municipio AS STRING) AS id_municipio,
  ano AS year,
  populacao
FROM `basedosdados.br_ms_populacao.municipio`
WHERE ano BETWEEN {ANO_INI} AND {ANO_FIM}
"""

df_pop = bd.read_sql(query_pop, billing_project_id=BILLING_PROJECT_ID)
df_pop["id_municipio"] = df_pop["id_municipio"].astype(str).str.zfill(7)
df_pop["year"] = df_pop["year"].astype(int)

# Filtrar df_pop pelos municípios das UFs selecionadas
df_pop = df_pop[df_pop["id_municipio"].isin(df_munis["id_municipio"])].copy()

# Agrupar df_pop para garantir que cada id_municipio e year tenha uma única entrada de populacao
df_pop = df_pop.groupby(["id_municipio", "year"])["populacao"].sum().reset_index()

print("SIM (deaths):", df_deaths.shape, "| Pop:", df_pop.shape)
df_deaths.head(), df_pop.head()

# Checagem e deduplicação das chaves município-ano
def ensure_unique_muny_year(df, name):
    dup = df.duplicated(subset=['id_municipio','year']).sum()
    print(f'{name}: {dup} duplicatas em id_municipio-year')
    if dup:
        dup_rows = df[df.duplicated(subset=['id_municipio','year'], keep=False)]
        display(dup_rows.sort_values(['id_municipio','year']).head())
        df = df.drop_duplicates(subset=['id_municipio','year'], keep='first')
        print(f'{name}: registros únicos após remoção -> {df.shape}')
    return df

df_deaths = ensure_unique_muny_year(df_deaths, 'df_deaths')
df_pop = ensure_unique_muny_year(df_pop, 'df_pop')
df_renda = ensure_unique_muny_year(df_renda, 'df_renda')


## 5) Painel balanceado usando População × Renda (RAIS)


# -----------------------------
# 5) Construir painel (mortalidade por 100 mil + renda formal per capita)
# -----------------------------
base = (
    df_deaths.merge(df_pop, on=["id_municipio","year"], how="inner")
            .merge(df_renda[["id_municipio","year","soma_remun","vinculos"]], on=["id_municipio","year"], how="inner")
)

# taxa de mortalidade por 100k hab
base["mort_rate_100k"] = (base["deaths"] / base["populacao"]) * 100000.0

# renda formal per capita (massa salarial / população)
base["renda_pc"] = base["soma_remun"] / base["populacao"]

# filtro: valores > 0 (como você pediu)
base = base[(base["mort_rate_100k"] > 0) & (base["renda_pc"] > 0)].copy()

# checagem: municípios por ano (antes de balancear)
munis_por_ano = (base.groupby("year")["id_municipio"].nunique().reset_index(name="n_municipios"))
display(munis_por_ano)

# Painel balanceado: só municípios que aparecem em TODOS os anos do intervalo
years_full = set(range(ANO_INI, ANO_FIM + 1))
counts = base.groupby("id_municipio")["year"].nunique()
munis_full = counts[counts == len(years_full)].index

base_bal = base[base["id_municipio"].isin(munis_full)].copy()

# checagem pós-balanceamento
munis_por_ano_bal = (base_bal.groupby("year")["id_municipio"].nunique().reset_index(name="n_municipios"))
display(munis_por_ano_bal)

print("Painel (antes):", base.shape, "| municípios únicos:", base["id_municipio"].nunique())
print("Painel (balanceado+positivo):", base_bal.shape, "| municípios únicos:", base_bal["id_municipio"].nunique())

base_bal.head()


## 6) Transformações (logs e HP)


df = (
    base_bal
    .sort_values(["id_municipio", "year"])
    .drop_duplicates(subset=["id_municipio", "year"])
    .set_index(["id_municipio", "year"])
)

df["ln_mort"]  = np.log(df["mort_rate_100k"])
df["ln_renda"] = np.log(df["renda_pc"])

df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["ln_mort", "ln_renda"])

def hp_cycle_by_entity(series, lamb=HP_LAMBDA_ANUAL, min_T=5):
    # Esta função auxiliar será aplicada a cada grupo por `transform`
    def _hp_transform(x):
        x_float = x.astype(float)
        # Verifica se há dados suficientes para o hpfilter
        if x_float.notna().sum() < min_T:
            # Se não houver dados suficientes, retorna NaNs para todo o grupo, preservando o índice original
            return pd.Series(np.nan, index=x.index)
        try:
            # hpfilter retorna ciclo e tendência, precisamos do ciclo
            cycle, _ = hpfilter(x_float, lamb=lamb)
            return cycle
        except Exception:
            # Lida com possíveis erros do hpfilter retornando NaNs
            return pd.Series(np.nan, index=x.index)

    # Usa transform para aplicar a função a cada grupo, retornando uma Série com o índice original
    return series.groupby(level=0).transform(_hp_transform)

if USE_HP:
    df["cyc_ln_mort"]  = hp_cycle_by_entity(df["ln_mort"])
    df["cyc_ln_renda"] = hp_cycle_by_entity(df["ln_renda"])

df.head()


## 7) Estimações (Pooling, FE e RE) + Hausman


def add_constant(X: pd.DataFrame) -> pd.DataFrame:
    Xc = X.copy()
    if "const" not in Xc.columns:
        Xc.insert(0, "const", 1.0)
    return Xc

def hausman(fe_res, re_res):
    # Hausman chi2 = (b-B)' [Var(b)-Var(B)]^{-1} (b-B)
    b = fe_res.params
    B = re_res.params
    common = b.index.intersection(B.index)
    b = b[common]; B = B[common]
    Vb = fe_res.cov.loc[common, common]
    VB = re_res.cov.loc[common, common]
    diff = (b - B).values.reshape(-1, 1)
    Vdiff = (Vb - VB).values
    try:
        stat = float(diff.T @ np.linalg.inv(Vdiff) @ diff)
        df_h = len(common)
        pval = 1.0 - stats.chi2.cdf(stat, df_h)
        return stat, df_h, pval
    except np.linalg.LinAlgError:
        return np.nan, len(common), np.nan

# -----------------------------
# 7) Estimações (Pooling, FE_ind, FE_time, RE_ind)
# -----------------------------
y = df["ln_mort"]
X = add_constant(df[["ln_renda"]])

pooled = PooledOLS(y, X).fit(cov_type="clustered", cluster_entity=True)
fe_i   = PanelOLS(y, X, entity_effects=True).fit(cov_type="clustered", cluster_entity=True)
fe_t   = PanelOLS(y, X, time_effects=True).fit(cov_type="clustered", cluster_entity=True)
re_i   = RandomEffects(y, X).fit(cov_type="clustered", cluster_entity=True)

print(compare({"Pooled": pooled, "FE_ind": fe_i, "FE_time": fe_t, "RE_ind": re_i}))

H, df_h, p = hausman(fe_i, re_i)
print(f"Hausman (FE_ind vs RE_ind): chi2={H:.4f}, df={df_h}, p-value={p:.6f}")


### 7.1 Cíclico (HP): ciclo ln(mort) ~ ciclo ln(renda)


if USE_HP:
    df_cyc = df.dropna(subset=["cyc_ln_mort","cyc_ln_renda"]).copy()
    y_c = df_cyc["cyc_ln_mort"]
    X_c = add_constant(df_cyc[["cyc_ln_renda"]])

    pooled_c = PooledOLS(y_c, X_c).fit(cov_type="clustered", cluster_entity=True)
    fe_i_c   = PanelOLS(y_c, X_c, entity_effects=True).fit(cov_type="clustered", cluster_entity=True)
    fe_t_c   = PanelOLS(y_c, X_c, time_effects=True).fit(cov_type="clustered", cluster_entity=True)
    re_i_c   = RandomEffects(y_c, X_c).fit(cov_type="clustered", cluster_entity=True)

    print(compare({"Pooled_cyc": pooled_c, "FE_ind_cyc": fe_i_c, "FE_time_cyc": fe_t_c, "RE_ind_cyc": re_i_c}))
else:
    pooled_c = fe_i_c = fe_t_c = re_i_c = None
    print("USE_HP=False (pulando modelos cíclicos).")


## 8) Exportar resultados


import os

def coef_table(res, model_name):
    if res is None:
        return pd.DataFrame(columns=["model","var","coef","se","t","p"])
    out = pd.DataFrame({
        "model": model_name,
        "coef": res.params,
        "se": res.std_errors,
        "t": res.tstats,
        "p": res.pvalues
    })
    return out.reset_index(names="var")

results = pd.concat([
    coef_table(pooled, "Pooled_static"),
    coef_table(fe_i, "FE_ind_static"),
    coef_table(fe_t, "FE_time_static"),
    coef_table(re_i, "RE_ind_static"),
    coef_table(pooled_c, "Pooled_cyc" if USE_HP else "Pooled_cyc"),
    coef_table(fe_i_c, "FE_ind_cyc" if USE_HP else "FE_ind_cyc"),
    coef_table(fe_t_c, "FE_time_cyc" if USE_HP else "FE_time_cyc"),
    coef_table(re_i_c, "RE_ind_cyc" if USE_HP else "RE_ind_cyc"),
], ignore_index=True)

out_path = "/mnt/data/results_mortalidade_renda_rais_bd.xlsx"

# Create the directory if it does not exist
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    results.to_excel(writer, index=False, sheet_name="coeficientes")
    base_bal.to_excel(writer, index=False, sheet_name="painel_base_balanceado")
out_path
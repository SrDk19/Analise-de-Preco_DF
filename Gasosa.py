import base64
import webbrowser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing

DATA_DIR = Path("dados")
OUT_DIR = Path("saida")
UF = "DF"
PRODUTO = "GASOLINA COMUM"
CONSUMO_MENSAL_L = 100
MESES_PREVISAO = 12

OUT_DIR.mkdir(exist_ok=True)

def carregar_dados() -> pd.DataFrame:
    arquivos = sorted(DATA_DIR.glob("*.csv"))
    if not arquivos:
        raise SystemExit(f"Nenhum CSV encontrado em {DATA_DIR.resolve()}")

    partes = []
    for arq in arquivos:
        df = None
        for enc in ("utf-8", "latin-1"):
            try:
                df = pd.read_csv(arq, sep=";", decimal=",", encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            raise SystemExit(f"Não consegui ler o arquivo: {arq}")
        df.columns = df.columns.str.strip()
        df = df.rename(columns={
            "Estado - Sigla": "uf",
            "Produto": "produto",
            "Data da Coleta": "data",
            "Valor de Venda": "preco",
        })[["uf", "produto", "data", "preco"]]
        partes.append(df)

    df = pd.concat(partes, ignore_index=True)
    df["produto"] = df["produto"].str.upper().str.strip()
    df["uf"] = df["uf"].str.upper().str.strip()
    df["data"] = pd.to_datetime(df["data"], dayfirst=True, format="mixed", errors="coerce")
    if df["preco"].dtype == object:
        df["preco"] = df["preco"].str.replace(",", ".", regex=False)
    df["preco"] = pd.to_numeric(df["preco"], errors="coerce")
    df = df.dropna(subset=["data", "preco"])
    df["mes"] = df["data"].dt.to_period("M").dt.to_timestamp()
    return df

def serie_mensal(df: pd.DataFrame) -> pd.DataFrame:
    """Preço médio mensal do produto no DF e no Brasil (todos os postos)."""
    g = df[df["produto"] == PRODUTO]
    brasil = g.groupby("mes")["preco"].mean().rename("brasil")
    local = g[g["uf"] == UF].groupby("mes")["preco"].mean().rename(UF)
    m = pd.concat([local, brasil], axis=1).dropna()
    m["dif_rs"] = m[UF] - m["brasil"]
    m["dif_pct"] = m["dif_rs"] / m["brasil"] * 100
    return m

def analise_anual(m: pd.DataFrame) -> pd.DataFrame:
    a = m.groupby(pd.DatetimeIndex(m.index).year)[[UF, "brasil", "dif_rs", "dif_pct"]].mean()
    a["var_ano_" + UF + "_%"] = a[UF].pct_change() * 100
    a["var_ano_brasil_%"] = a["brasil"].pct_change() * 100
    return a

def simular_economia(m: pd.DataFrame) -> None:
    """Quanto o DF pagaria a mais/menos e quanto poderia mudar."""
    ult12 = m.tail(12)
    dif = ult12["dif_rs"].mean()
    anual = dif * CONSUMO_MENSAL_L * 12
    print("\n=== CENÁRIO: DF convergindo para a média nacional ===")
    print(f"Diferença média (últimos 12 meses): R$ {dif:+.3f}/litro "
          f"({ult12['dif_pct'].mean():+.2f}%)")
    print(f"Para {CONSUMO_MENSAL_L} L/mês: R$ {anual:+.2f} por ano "
          f"({'a mais' if anual > 0 else 'a menos'} que a média nacional)")
    s = m["dif_pct"]
    melhor = s.index[int(s.to_numpy().argmin())]
    pior = s.index[int(s.to_numpy().argmax())]
    print(f"Menor diferença histórica: {s.min():+.2f}% em {melhor:%m/%Y}")
    print(f"Maior diferença histórica: {s.max():+.2f}% em {pior:%m/%Y}")

    print("\nSensibilidade (variação do preço no DF -> impacto anual):")
    for pct in (-5, -2, 2, 5):
        delta = ult12[UF].mean() * pct / 100 * CONSUMO_MENSAL_L * 12
        print(f"  {pct:+d}% no preço -> R$ {delta:+.2f}/ano")

def prever(serie: pd.Series, meses: int) -> pd.DataFrame:
    s = serie.asfreq("MS").interpolate()
    sazonal = "add" if len(s) >= 24 else None
    modelo = ExponentialSmoothing(
        s, trend="add", damped_trend=True,
        seasonal=sazonal, seasonal_periods=12 if sazonal else None,
    ).fit()
    prev = modelo.forecast(meses)
    sigma = np.std(modelo.resid, ddof=1)
    h = np.arange(1, meses + 1)
    return pd.DataFrame({
        "previsao": prev,
        "inf": prev - 1.96 * sigma * np.sqrt(h),
        "sup": prev + 1.96 * sigma * np.sqrt(h),
    })

def panorama_geral(df: pd.DataFrame) -> pd.DataFrame:
    """Todos os combustíveis: preço médio no DF vs Brasil (últimos 12 meses)."""
    corte = df["mes"].max() - pd.DateOffset(months=11)
    r = df[df["mes"] >= corte]
    brasil = r.groupby("produto")["preco"].mean().rename("brasil")
    local = r[r["uf"] == UF].groupby("produto")["preco"].mean().rename(UF)
    t = pd.concat([local, brasil], axis=1).dropna()
    t["dif_%"] = (t[UF] / t["brasil"] - 1) * 100
    return t.sort_values("dif_%")

def graficos(df, m, prev_df, prev_br, prev_dif):
    
    #Evolução dos preços
    
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(m.index, m[UF], label=UF)
    ax.plot(m.index, m["brasil"], label="Brasil")
    ax.set(title=f"{PRODUTO.title()} - preço médio mensal (R$/L)", ylabel="R$/L")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT_DIR / "01_evolucao_precos.png", dpi=150); plt.close(fig)

    #Diferença DF - Brasil
    
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(m.index, m["dif_pct"], color="tab:red")
    ax.axhline(0, color="k", lw=.8)
    ax.set(title=f"{UF} em relação à média nacional (%)", ylabel="%")
    ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT_DIR / "02_diferenca_pct.png", dpi=150); plt.close(fig)

    #Previsão

    fig, ax = plt.subplots(figsize=(11, 5))
    hist = m.tail(36)
    ax.plot(hist.index, hist[UF], label=f"{UF} (histórico)")
    ax.plot(prev_df.index, prev_df["previsao"], "--", label=f"{UF} (previsão)")
    ax.fill_between(prev_df.index, prev_df["inf"], prev_df["sup"], alpha=.2)
    ax.plot(hist.index, hist["brasil"], label="Brasil (histórico)")
    ax.plot(prev_br.index, prev_br["previsao"], "--", label="Brasil (previsão)")
    ax.set(title=f"Previsão {MESES_PREVISAO} meses - {PRODUTO.title()}", ylabel="R$/L")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT_DIR / "03_previsao.png", dpi=150); plt.close(fig)

    #Todos os combustíveis no DF
    
    pivot = (df[df["uf"] == UF].groupby([df["mes"].dt.year, "produto"])["preco"]
             .mean().unstack())
    ax = pivot.plot(figsize=(11, 5), marker="o")
    ax.set(title=f"Preço médio anual por combustível - {UF}", ylabel="R$ (por litro/m³/kg)",
           xlabel="Ano")
    ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUT_DIR / "04_todos_combustiveis.png", dpi=150); plt.close()

def _img_b64(nome: str) -> str:
    dados = (OUT_DIR / nome).read_bytes()
    return "data:image/png;base64," + base64.b64encode(dados).decode()

def gerar_relatorio_html(m, anual, pan, prev_df, prev_br) -> Path:
    ult12 = m.tail(12)
    dif = ult12["dif_rs"].mean()
    dif_pct = ult12["dif_pct"].mean()
    economia = dif * CONSUMO_MENSAL_L * 12

    #Tabela mensal (últimos 24 meses)
    
    mensal = m.tail(24).round(3).copy()
    mensal.index = pd.DatetimeIndex(mensal.index).strftime("%m/%Y")
    mensal.columns = [UF, "Brasil", "Dif. (R$)", "Dif. (%)"]

    #Tabela de previsão

    prev = pd.DataFrame({
        f"{UF} previsão": prev_df["previsao"],
        f"{UF} mín.": prev_df["inf"],
        f"{UF} máx.": prev_df["sup"],
        "Brasil previsão": prev_br["previsao"],
    })
    prev["Dif. (R$)"] = prev[f"{UF} previsão"] - prev["Brasil previsão"]
    prev = prev.round(3)
    prev.index = pd.DatetimeIndex(prev.index).strftime("%m/%Y")

    def tabela(df_):
        return df_.to_html(classes="tabela", border=0, float_format="%.3f")

    css = """
    body{font-family:Segoe UI,Arial,sans-serif;max-width:1100px;margin:30px auto;
         padding:0 16px;color:#222;background:#fafafa}
    h1{margin-bottom:4px} h2{margin-top:40px;border-bottom:2px solid #ddd;padding-bottom:6px}
    .cards{display:flex;gap:16px;flex-wrap:wrap;margin:20px 0}
    .card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:14px 20px;flex:1;min-width:220px}
    .card b{display:block;font-size:1.6em;margin-top:4px}
    .tabela{border-collapse:collapse;background:#fff;width:100%;font-size:.9em}
    .tabela th,.tabela td{border:1px solid #e0e0e0;padding:6px 10px;text-align:right}
    .tabela th{background:#f0f0f0}
    img{max-width:100%;border:1px solid #ddd;background:#fff;margin:10px 0}
    .scroll{overflow-x:auto}
    """

    cor = "#c0392b" if dif > 0 else "#27ae60"
    html = f"""<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="utf-8">
<title>{PRODUTO.title()}: {UF} x Brasil</title><style>{css}</style></head>
<body>
<h1>{PRODUTO.title()}: {UF} x média nacional</h1>
<p>Fonte: ANP - Levantamento de Preços de Combustíveis</p>

<div class="cards">
  <div class="card">Diferença média (12 meses)<b style="color:{cor}">R$ {dif:+.3f}/L</b></div>
  <div class="card">Diferença em %<b style="color:{cor}">{dif_pct:+.2f}%</b></div>
  <div class="card">Impacto anual ({CONSUMO_MENSAL_L} L/mês)<b style="color:{cor}">R$ {economia:+.2f}</b></div>
</div>

<h2>Evolução dos preços</h2>
<img src="{_img_b64('01_evolucao_precos.png')}">
<img src="{_img_b64('02_diferenca_pct.png')}">

<h2>Resumo anual</h2>
<div class="scroll">{tabela(anual.round(3))}</div>

<h2>Últimos 24 meses</h2>
<div class="scroll">{tabela(mensal)}</div>

<h2>Previsão ({MESES_PREVISAO} meses)</h2>
<img src="{_img_b64('03_previsao.png')}">
<div class="scroll">{tabela(prev)}</div>

<h2>Panorama de todos os combustíveis (últimos 12 meses)</h2>
<div class="scroll">{tabela(pan.round(3))}</div>
<img src="{_img_b64('04_todos_combustiveis.png')}">
</body></html>"""

    caminho = OUT_DIR / "relatorio.html"
    caminho.write_text(html, encoding="utf-8")
    return caminho

def main():
    df = carregar_dados()
    print(f"{len(df):,} registros | {df['data'].min():%d/%m/%Y} a {df['data'].max():%d/%m/%Y}")

    m = serie_mensal(df)
    anual = analise_anual(m)
    m.to_csv(OUT_DIR / "mensal.csv")
    anual.to_csv(OUT_DIR / "anual.csv")

    print(f"\n--- {PRODUTO}: {UF} vs Brasil (média anual) ---")
    print(anual.round(3).to_string())
    if 2023 in anual.index:
        print(f"\nVariação nacional em 2023 nos dados: "
              f"{anual.loc[2023, 'var_ano_brasil_%']:.2f}%  (compare com os 6,1% que você citou)")

    simular_economia(m)

    prev_df = prever(m[UF], MESES_PREVISAO)
    prev_br = prever(m["brasil"], MESES_PREVISAO)
    prev_dif = prev_df["previsao"] - prev_br["previsao"]
    prev_df.assign(brasil=prev_br["previsao"], dif_rs=prev_dif).to_csv(OUT_DIR / "previsao.csv")
    print(f"\n--- Previsão ({MESES_PREVISAO} meses) ---")
    print(f"{UF}: R$ {prev_df['previsao'].iloc[-1]:.2f}/L | "
          f"Brasil: R$ {prev_br['previsao'].iloc[-1]:.2f}/L | "
          f"diferença: R$ {prev_dif.iloc[-1]:+.2f}/L")

    pan = panorama_geral(df)
    pan.to_csv(OUT_DIR / "panorama_combustiveis.csv")
    print(f"\n--- Panorama geral (últimos 12 meses) ---")
    print(pan.round(3).to_string())

    graficos(df, m, prev_df, prev_br, prev_dif)
    caminho = gerar_relatorio_html(m, anual, pan, prev_df, prev_br)
    webbrowser.open(caminho.resolve().as_uri())
    print(f"\nTabelas e gráficos salvos em {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
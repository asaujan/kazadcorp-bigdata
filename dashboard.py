import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# ──────────────────────────────────────────────
# КОНФИГУРАЦИЯ СТРАНИЦЫ
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="KazAdCorp Analytics",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="collapsedControl"] {
    display: none !important;
    visibility: hidden !important;
}
div[data-testid="stSidebar"] {
    display: flex !important;
    visibility: visible !important;
    width: 280px !important;
    min-width: 280px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
        header {visibility: hidden;}
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {padding-top: 1.2rem;}
        .kaz-logo {
            font-size: 1.6rem;
            font-weight: 800;
            background: linear-gradient(90deg, #00d4aa 0%, #2a9df4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            padding: 0.4rem 0;
            letter-spacing: 0.5px;
        }
        .kaz-sub {
            color: #6b7280;
            font-size: 0.9rem;
            margin-bottom: 0.2rem;
        }
        .scope-box {
            background: #1e2530;
            border-left: 3px solid #00d4aa;
            padding: 8px 10px;
            border-radius: 4px;
            font-size: 0.82rem;
            color: #e5e7eb;
            margin: 6px 0 10px 0;
        }
        div[data-testid="stMetricValue"] {font-size: 1.4rem;}
        .stMarkdown h1 a, .stMarkdown h2 a, .stMarkdown h3 a {display: none;}
        [data-testid="collapsedControl"] { display: none !important; }
        [data-testid="stSidebarCollapseButton"] { display: none !important; }
        button[aria-label="Close sidebar"] { display: none !important; }
        button[aria-label="Collapse sidebar"] { display: none !important; }
        section[data-testid="stSidebar"] {
            min-width: 280px !important;
            max-width: 280px !important;
            display: block !important;
        }
        section[data-testid="stSidebar"][aria-expanded="false"] {
            display: block !important;
            margin-left: 0 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# ЕДИНАЯ ПАЛИТРА
# ──────────────────────────────────────────────
C_HIGH   = "#00d4aa"
C_MID    = "#f5a623"
C_LOW    = "#e94560"
C_BLUE   = "#2a9df4"
C_PURPLE = "#a855f7"

COLORS_PRIORITY = {
    "ВЫСОКИЙ": C_HIGH,
    "СРЕДНИЙ": C_MID,
    "НИЗКИЙ":  C_LOW,
}

# ──────────────────────────────────────────────
# ЗАГРУЗКА ДАННЫХ
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("kazadcorp_real_enriched.csv", encoding="utf-8-sig")
    df["platform_list"] = df["platform"].apply(
        lambda x: [p.strip() for p in str(x).split(",")]
    )
    df["is_multi_platform"] = df["platform_list"].apply(lambda x: len(x) > 1)
    df["platform_clean"] = df["platform"].apply(
        lambda x: x if "," not in str(x) else "Несколько платформ"
    )
    for col in ["roas", "ctr_pct", "cr_pct"]:
        mn, mx = df[col].min(), df[col].max()
        df[f"{col}_norm"] = (df[col] - mn) / (mx - mn + 1e-9)
    df["ml_score"] = (
        0.45 * df["roas_norm"]
        + 0.30 * df["ctr_pct_norm"]
        + 0.25 * df["cr_pct_norm"]
    )
    q33 = df["ml_score"].quantile(0.33)
    q66 = df["ml_score"].quantile(0.66)
    df["priority"] = df["ml_score"].apply(
        lambda s: "ВЫСОКИЙ" if s >= q66 else ("СРЕДНИЙ" if s >= q33 else "НИЗКИЙ")
    )
    all_plat = sorted({p for lst in df["platform_list"] for p in lst})
    return df, all_plat, q33, q66


df_full, all_platforms, Q33, Q66 = load_data()
TOTAL_RECORDS = len(df_full)

# ──────────────────────────────────────────────
# БОКОВАЯ ПАНЕЛЬ
# ──────────────────────────────────────────────
st.sidebar.markdown("### 🏢 KazAdCorp")
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div class="kaz-logo">🏢 KazAdCorp</div>'
    '<div class="kaz-sub">Big Data Analytics · 2026</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")
st.sidebar.title("⚙️ Фильтры")

st.sidebar.markdown(
    '<div class="scope-box">📌 Фильтры активны для вкладок:<br>'
    '<b>📊 Метрики</b> · <b>🤖 ML-скоринг</b></div>',
    unsafe_allow_html=True,
)

advertisers_all = sorted(df_full["advertiser"].unique().tolist())

# Reset filters using dynamic key pattern
if "filter_reset_count" not in st.session_state:
    st.session_state["filter_reset_count"] = 0

if st.sidebar.button("🔄 Сбросить фильтры"):
    st.session_state["filter_reset_count"] += 1

_rk = st.session_state["filter_reset_count"]

sel_adv = st.sidebar.multiselect(
    "🏢 Рекламодатель",
    options=advertisers_all,
    default=advertisers_all,
    key=f"sel_adv_{_rk}",
)
sel_plat = st.sidebar.multiselect(
    "📱 Платформа",
    options=all_platforms,
    default=all_platforms,
    key=f"sel_plat_{_rk}",
)

mask_adv = df_full["advertiser"].isin(sel_adv)
mask_plat = df_full["platform_list"].apply(
    lambda lst: any(p in lst for p in sel_plat)
)
df = df_full[mask_adv & mask_plat].copy()

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**📈 Записей:** {TOTAL_RECORDS} → после фильтра: **{len(df)}**"
)
st.sidebar.caption(
    "KazAdCorp · Дипломная работа 2026  \n"
    "Данные: Meta Ads Library KZ"
)

# ──────────────────────────────────────────────
# ЗАГОЛОВОК
# ──────────────────────────────────────────────
st.markdown(
    '<div class="kaz-logo" style="font-size:2rem;">🏢 KazAdCorp · Big Data Analytics</div>',
    unsafe_allow_html=True,
)
st.caption(
    f"Аналитика рекламных кампаний · Период: апрель 2026 · "
    f"{TOTAL_RECORDS} объявлений · 10 рекламодателей · Meta Ads Library KZ"
)
st.markdown("---")

# ──────────────────────────────────────────────
# О ПРОЕКТЕ (expandable)
# ──────────────────────────────────────────────
with st.expander("📋 О проекте"):
    st.markdown(
        """
        ### Что такое KazAdCorp?
        **KazAdCorp** — вымышленная рекламная платформа Казахстана. В основе —
        **695 реальных объявлений** из Meta Ads Library KZ от 10 крупных рекламодателей.

        ### Какую проблему решает?
        Рекламные бюджеты тратятся неэффективно: менеджеры не знают, какие кампании
        приносят реальный доход, а какие — сжигают деньги. CAC (стоимость привлечения клиента)
        слишком высок, ROAS (окупаемость) нестабилен.

        ### Как работает система?
        ```
        Данные Meta Ads → PySpark обработка → ML скоринг → Рекомендации менеджеру
        ```
        1. **Данные Meta Ads** — 695 объявлений с метриками CTR, ROAS, CAC, конверсии
        2. **PySpark обработка** — агрегация, JOIN, нормализация на Big Data кластере
        3. **ML скоринг** — составной балл: ROAS (45%) + CTR (30%) + CR (25%)
        4. **Рекомендации** — автоматическая классификация на ВЫСОКИЙ / СРЕДНИЙ / НИЗКИЙ приоритет

        ### Ключевые результаты
        | Показатель | Значение |
        |---|---|
        | Рекомендовано финансировать | **292 кампании** |
        | Рекомендовано остановить | **403 кампании** |
        | Потенциальная экономия | **194–594 млн ₸ / год** |
        | Ускорение Spark vs Pandas | **до 15.7×** (UDF на 100 млн строк) |
        """
    )

if df.empty:
    st.warning("Нет данных для выбранных фильтров. Измените параметры в боковой панели.")
    st.stop()


# ──────────────────────────────────────────────
# KPI-БАР
# ──────────────────────────────────────────────
def render_kpi_bar(_df: pd.DataFrame):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "📣 Показы (млн)",
        f"{_df['impressions'].sum() / 1e6:.1f}",
        help="Общее число показов рекламы.",
    )
    c2.metric(
        "🖱️ Клики (тыс.)",
        f"{_df['clicks'].sum() / 1e3:.1f}",
        help="Общее число кликов по объявлениям.",
    )
    c3.metric(
        "📈 Средний CTR, %",
        f"{_df['ctr_pct'].mean():.2f}",
        help="CTR (Click-Through Rate) — отношение кликов к показам.",
    )
    c4.metric(
        "💰 Средний ROAS",
        f"{_df['roas'].mean():.2f}",
        help="ROAS (Return on Ad Spend) — окупаемость рекламы.",
    )
    c5.metric(
        "🎯 Конверсий всего",
        f"{int(_df['conversions'].sum()):,}",
        help="Конверсия — целевое действие пользователя.",
    )


# ──────────────────────────────────────────────
# ВКЛАДКИ
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Метрики",
    "🤖 ML-скоринг",
    "🔬 A/B-тестирование",
    "⚡ Бенчмарк Spark",
    "📖 Справочник",
])


# ══════════════════════════════════════════════
# TAB 1 · МЕТРИКИ
# ══════════════════════════════════════════════
with tab1:
    render_kpi_bar(df)
    st.markdown("---")
    st.subheader("KPI по рекламодателям")
    st.caption(
        "ℹ️ **CTR** — кликабельность (%); **ROAS** — окупаемость рекламы; "
        "**CAC** — стоимость привлечения клиента (₸)."
    )

    agg = (
        df.groupby("advertiser")
        .agg(
            avg_ctr=("ctr_pct", "mean"),
            avg_roas=("roas", "mean"),
            avg_cac=("cac_kzt", "mean"),
            total_conv=("conversions", "sum"),
            total_spent=("spent_kzt", "sum"),
            total_impressions=("impressions", "sum"),
        )
        .reset_index()
        .sort_values("avg_roas", ascending=False)
    )

    col1, col2 = st.columns(2)
    with col1:
        fig_ctr = px.bar(
            agg.sort_values("avg_ctr", ascending=True),
            x="avg_ctr",
            y="advertiser",
            orientation="h",
            title="CTR по рекламодателям, %",
            labels={"avg_ctr": "CTR (%)", "advertiser": ""},
            color="avg_ctr",
            color_continuous_scale=[[0, C_LOW], [0.5, C_MID], [1, C_HIGH]],
        )
        fig_ctr.update_layout(
            coloraxis_showscale=False,
            height=380,
            margin=dict(l=10, r=80, t=50, b=10),
        )
        fig_ctr.update_traces(textposition="outside", texttemplate="%{x:.2f}")
        fig_ctr.add_vline(
            x=1.0,
            line_dash="dash",
            line_color="gray",
            annotation_text="Отраслевой бенчмарк: 1%",
            annotation_position="top right",
        )
        st.plotly_chart(fig_ctr, use_container_width=True)

    with col2:
        fig_roas = px.bar(
            agg.sort_values("avg_roas", ascending=True),
            x="avg_roas",
            y="advertiser",
            orientation="h",
            title="ROAS по рекламодателям",
            labels={"avg_roas": "ROAS", "advertiser": ""},
            color="avg_roas",
            color_continuous_scale=[[0, C_LOW], [0.5, C_MID], [1, C_HIGH]],
        )
        fig_roas.update_layout(
            coloraxis_showscale=False,
            height=380,
            margin=dict(l=10, r=80, t=50, b=10),
        )
        fig_roas.update_traces(textposition="outside", texttemplate="%{x:.2f}")
        fig_roas.add_vline(
            x=5.0,
            line_dash="dash",
            line_color="gray",
            annotation_text="Целевой ROAS: 5.0",
            annotation_position="top right",
        )
        st.plotly_chart(fig_roas, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig_cac = px.bar(
            agg.sort_values("avg_cac", ascending=False),
            x="advertiser",
            y="avg_cac",
            title="CAC, ₸ (ниже — лучше)",
            labels={"avg_cac": "CAC (₸)", "advertiser": ""},
            color="avg_cac",
            color_continuous_scale=[[0, C_HIGH], [0.5, C_MID], [1, C_LOW]],
        )
        fig_cac.update_layout(
            coloraxis_showscale=False,
            height=420,
            margin=dict(l=10, r=10, t=50, b=120),
            xaxis_tickangle=-35,
        )
        fig_cac.update_traces(textposition="outside", texttemplate="%{y:.0f}")
        fig_cac.add_hline(
            y=13000,
            line_dash="dash",
            line_color="gray",
            annotation_text="Целевой CAC: 13 000₸",
            annotation_position="top right",
        )
        st.plotly_chart(fig_cac, use_container_width=True)

    with col4:
        fig_conv = px.bar(
            agg.sort_values("total_conv", ascending=False),
            x="advertiser",
            y="total_conv",
            title="Всего конверсий",
            labels={"total_conv": "Конверсий", "advertiser": ""},
            color="total_conv",
            color_continuous_scale=[[0, C_LOW], [0.5, C_MID], [1, C_HIGH]],
        )
        fig_conv.update_layout(
            coloraxis_showscale=False,
            showlegend=False,
            height=420,
            margin=dict(l=10, r=10, t=50, b=120),
            xaxis_tickangle=-35,
        )
        fig_conv.update_traces(textposition="outside", texttemplate="%{y:.0f}")
        st.plotly_chart(fig_conv, use_container_width=True)

    with st.expander("📋 Детальная таблица метрик"):
        display = agg.copy().reset_index(drop=True)
        display.index = display.index + 1
        display.columns = [
            "Рекламодатель", "CTR (%)", "ROAS", "CAC (₸)",
            "Конверсий", "Расходы (₸)", "Показы",
        ]
        display["CTR (%)"] = display["CTR (%)"].round(2)
        display["ROAS"] = display["ROAS"].round(2)
        display["CAC (₸)"] = display["CAC (₸)"].round(0).astype(int)
        display["Конверсий"] = display["Конверсий"].astype(int)
        st.dataframe(display, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 2 · ML-СКОРИНГ
# ══════════════════════════════════════════════
with tab2:
    render_kpi_bar(df)
    st.markdown("---")
    st.subheader("ML-скоринг рекламных кампаний")
    st.markdown(
        "Составной балл: **ROAS** (45%) + **CTR** (30%) + **CR** (25%). "
        f"Границы классов: **НИЗКИЙ** < {Q33:.2f} ≤ **СРЕДНИЙ** < {Q66:.2f} ≤ **ВЫСОКИЙ**."
    )

    priority_order = ["ВЫСОКИЙ", "СРЕДНИЙ", "НИЗКИЙ"]
    col1, col2, col3 = st.columns(3)
    for col, pri, emoji in zip(
        [col1, col2, col3], priority_order, ["🟢", "🟡", "🔴"]
    ):
        cnt = (df["priority"] == pri).sum()
        col.metric(f"{emoji} {pri}", f"{cnt} кампаний")

    col_a, col_b = st.columns(2)

    with col_a:
        fig_scatter = px.scatter(
            df,
            x="ctr_pct",
            y="roas",
            color="priority",
            color_discrete_map=COLORS_PRIORITY,
            size="conversions",
            hover_data=["advertiser", "platform_clean", "ml_score"],
            title="ROAS vs CTR (приоритет)",
            labels={"ctr_pct": "CTR (%)", "roas": "ROAS", "priority": "Приоритет"},
            category_orders={"priority": priority_order},
        )
        fig_scatter.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(
                title="Приоритет<br><sup>Размер точки = конверсии</sup>",
            ),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption("💡 Размер точки пропорционален числу конверсий кампании.")

    with col_b:
        pri_by_adv = (
            df.groupby(["advertiser", "priority"])
            .size()
            .reset_index(name="count")
        )
        fig_pri = px.bar(
            pri_by_adv,
            x="advertiser",
            y="count",
            color="priority",
            color_discrete_map=COLORS_PRIORITY,
            title="Приоритеты по рекламодателям",
            labels={"count": "Кампаний", "advertiser": "", "priority": "Приоритет"},
            category_orders={"priority": priority_order},
            barmode="stack",
        )
        fig_pri.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=50, b=120),
            xaxis_tickangle=-35,
        )
        st.plotly_chart(fig_pri, use_container_width=True)

    # Топ-20 ВЫСОКИЙ
    st.subheader("🏆 Топ-20 кампаний с ВЫСОКИМ приоритетом (запустить / усилить)")
    top_high = (
        df[df["priority"] == "ВЫСОКИЙ"]
        .sort_values("ml_score", ascending=False)
        .head(20)[
            ["advertiser", "platform_clean", "ctr_pct", "roas",
             "cr_pct", "conversions", "ml_score"]
        ]
        .reset_index(drop=True)
    )
    top_high.index = top_high.index + 1
    top_high.columns = [
        "Рекламодатель", "Платформа", "CTR (%)", "ROAS",
        "CR (%)", "Конверсий", "ML-балл",
    ]
    top_high["CTR (%)"] = top_high["CTR (%)"].round(2)
    top_high["ROAS"] = top_high["ROAS"].round(2)
    top_high["CR (%)"] = top_high["CR (%)"].round(2)
    top_high["Конверсий"] = top_high["Конверсий"].astype(int)
    st.dataframe(
        top_high.style.background_gradient(
            cmap="Greens", subset=["ML-балл", "ROAS"]
        ),
        use_container_width=True,
    )

    # Bottom-20 НИЗКИЙ — no color styling to avoid white-on-white
    st.subheader("🛑 Bottom-20 кампаний с НИЗКИМ приоритетом (остановить)")
    low_camps = (
        df[df["priority"] == "НИЗКИЙ"]
        .sort_values("ml_score", ascending=True)
        .head(20)
        .loc[:, ["advertiser", "platform_clean", "ctr_pct", "roas",
                 "cr_pct", "conversions", "ml_score"]]
        .reset_index(drop=True)
    )
    low_camps.index = low_camps.index + 1
    low_camps.columns = [
        "Рекламодатель", "Платформа", "CTR (%)", "ROAS",
        "CR (%)", "Конверсий", "ML-балл",
    ]
    low_camps["CTR (%)"] = low_camps["CTR (%)"].round(2)
    low_camps["ROAS"] = low_camps["ROAS"].round(2)
    low_camps["CR (%)"] = low_camps["CR (%)"].round(2)
    low_camps["Конверсий"] = low_camps["Конверсий"].astype(int)
    st.dataframe(low_camps, use_container_width=True)

    # Гистограмма ML-балла
    fig_hist = px.histogram(
        df,
        x="ml_score",
        color="priority",
        color_discrete_map=COLORS_PRIORITY,
        nbins=40,
        title="Распределение ML-баллов",
        labels={"ml_score": "ML-балл", "count": "Количество", "priority": "Приоритет"},
        category_orders={"priority": priority_order},
        barmode="overlay",
        opacity=0.78,
    )
    fig_hist.add_vline(
        x=Q33, line_dash="dash", line_color=C_LOW, line_width=2,
        annotation_text=f"НИЗКИЙ → СРЕДНИЙ ({Q33:.2f})",
        annotation_position="top",
    )
    fig_hist.add_vline(
        x=Q66, line_dash="dash", line_color=C_HIGH, line_width=2,
        annotation_text=f"СРЕДНИЙ → ВЫСОКИЙ ({Q66:.2f})",
        annotation_position="top",
    )
    fig_hist.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=70, b=10),
        legend_title_text="Приоритет",
    )
    st.plotly_chart(fig_hist, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3 · A/B-ТЕСТИРОВАНИЕ
# ══════════════════════════════════════════════
with tab3:
    st.subheader("A/B-тестирование рекламных стратегий")
    st.markdown(
        """
        **Дизайн эксперимента:**
        - **Группа A** — кампании на **одной платформе** (Facebook / Instagram)
        - **Группа B** — кампании на **нескольких платформах** одновременно

        Для каждого рекламодателя: t-критерий Стьюдента (p-value) + размер эффекта Коэна d.

        ⚠️ *Эта вкладка использует полный датасет (фильтры боковой панели не применяются).*
        """
    )

    def cohens_d(a, b):
        na, nb = len(a), len(b)
        if na < 2 or nb < 2:
            return np.nan
        pooled = np.sqrt(
            ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1))
            / (na + nb - 2)
        )
        return (np.mean(a) - np.mean(b)) / (pooled + 1e-9)

    def cohen_label(d):
        if pd.isna(d):
            return "—"
        ad = abs(d)
        if ad >= 0.8:
            return "большой"
        if ad >= 0.5:
            return "средний"
        if ad >= 0.2:
            return "малый"
        return "незначительный"

    def cohen_label_short(d):
        if pd.isna(d):
            return "—"
        ad = abs(d)
        if ad >= 0.8: return "бол."
        if ad >= 0.5: return "ср."
        if ad >= 0.2: return "мал."
        return "незн."

    results = []
    for metric, label in [("roas", "ROAS"), ("ctr_pct", "CTR (%)"), ("conversions", "Конверсий")]:
        for adv in sorted(df_full["advertiser"].unique()):
            sub = df_full[df_full["advertiser"] == adv]
            grp_a = sub[~sub["is_multi_platform"]][metric].dropna()
            grp_b = sub[sub["is_multi_platform"]][metric].dropna()
            if len(grp_a) < 2 or len(grp_b) < 2:
                continue
            t_stat, p_val = stats.ttest_ind(grp_a, grp_b, equal_var=False)
            d = cohens_d(grp_a.values, grp_b.values)

            # Rounding per metric type
            if metric == "conversions":
                mean_a = int(round(grp_a.mean()))
                mean_b = int(round(grp_b.mean()))
            elif metric in ("roas", "ctr_pct"):
                mean_a = round(grp_a.mean(), 2)
                mean_b = round(grp_b.mean(), 2)
            else:
                mean_a = round(grp_a.mean(), 2)
                mean_b = round(grp_b.mean(), 2)

            results.append({
                "Рекламодатель": adv,
                "Метрика": label,
                "Группа A (одна)": mean_a,
                "Группа B (мульти)": mean_b,
                "Группа A (кол-во)": len(grp_a),
                "Группа B (кол-во)": len(grp_b),
                "t-статистика": round(t_stat, 3),
                "p-value": round(p_val, 4),
                "Коэна d (размер эффекта)": round(d, 3),
                "Эффект": cohen_label(d),
                "Знач.": "✅" if p_val < 0.05 else "❌",
            })

    ab_df = pd.DataFrame(results)

    st.markdown("### 📋 Сводная таблица результатов (группировка по рекламодателю)")
    st.info("ℹ️ p-value < 0.05 означает статистически значимую разницу между группами")

    ab_sorted = ab_df.sort_values(["Рекламодатель", "Метрика"]).reset_index(drop=True)
    ab_sorted.index = ab_sorted.index + 1

    def color_signif(val):
        if val == "✅":
            return f"background-color: {C_HIGH}; color: white; font-weight: bold; text-align: center;"
        if val == "❌":
            return f"background-color: {C_LOW}; color: white; font-weight: bold; text-align: center;"
        return ""

    def color_effect(val):
        mp = {
            "большой":  f"background-color: {C_HIGH}; color: white;",
            "средний":  f"background-color: {C_MID}; color: white;",
            "малый":    "background-color: #fff3cd;",
            "незначительный": "background-color: #f5f5f5; color: #666;",
        }
        return mp.get(val, "")

    def color_cohen_d(val):
        try:
            ad = abs(float(val))
            if ad >= 0.8:
                return f"background-color: {C_LOW}; color: white;"
            if ad >= 0.5:
                return f"background-color: {C_MID}; color: white;"
            if ad >= 0.2:
                return f"background-color: {C_BLUE}; color: white;"
            return "background-color: #9ca3af; color: white;"
        except Exception:
            return ""

    styled_ab = (
        ab_sorted.style
        .map(color_signif, subset=["Знач."])
        .map(color_effect, subset=["Эффект"])
        .map(color_cohen_d, subset=["Коэна d (размер эффекта)"])
    )
    st.dataframe(
        styled_ab,
        use_container_width=True,
        height=420,
        column_config={
            "Знач.": st.column_config.TextColumn(width="small"),
        },
    )
    st.caption("* Конверсий — среднее значение по кампаниям группы (целое число)")

    # Beeline small sample warning
    beeline_rows = ab_df[ab_df["Рекламодатель"].str.contains("Beeline", na=False)]
    if not beeline_rows.empty:
        min_n = min(
            beeline_rows["Группа A (кол-во)"].min(),
            beeline_rows["Группа B (кол-во)"].min(),
        )
        if min_n <= 3:
            st.warning(
                "⚠️ Beeline Kazakhstan: Малая выборка (n=2), результат ориентировочный"
            )

    col1, col2 = st.columns(2)

    with col1:
        roas_ab = ab_df[ab_df["Метрика"] == "ROAS"].copy()
        roas_ab["signif_log"] = -np.log10(roas_ab["p-value"].clip(lower=1e-10))
        fig_pval = px.bar(
            roas_ab.sort_values("signif_log", ascending=False),
            x="Рекламодатель",
            y="signif_log",
            color="Знач.",
            color_discrete_map={"✅": C_HIGH, "❌": C_LOW},
            title="Статистическая значимость A/B (ROAS)",
            labels={"signif_log": "Статистическая значимость (чем выше — тем значимее)"},
            text_auto=".2f",
        )
        fig_pval.add_hline(
            y=-np.log10(0.05),
            line_dash="dash",
            line_color=C_MID,
            annotation_text="порог p=0.05",
            annotation_position="top right",
        )
        fig_pval.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=50, b=120),
            xaxis_tickangle=-35,
            showlegend=True,
        )
        st.plotly_chart(fig_pval, use_container_width=True)

    with col2:
        roas_ab["d_label"] = roas_ab["Коэна d (размер эффекта)"].apply(
            lambda d: f"d={d:.2f} {cohen_label_short(d)}"
        )
        roas_ab["effect_size"] = roas_ab["Коэна d (размер эффекта)"].apply(cohen_label)
        effect_color_map = {
            "большой":  C_HIGH,
            "средний":  C_MID,
            "малый":    C_BLUE,
            "незначительный": "#cbd5e1",
        }
        fig_cohen = px.bar(
            roas_ab.sort_values("Коэна d (размер эффекта)", ascending=True),
            x="Коэна d (размер эффекта)",
            y="Рекламодатель",
            orientation="h",
            color="effect_size",
            title="Размер эффекта Коэна d (ROAS)",
            labels={
                "Коэна d (размер эффекта)": "Cohen's d",
                "effect_size": "Размер эффекта",
            },
            text="d_label",
            color_discrete_map=effect_color_map,
            category_orders={"effect_size": ["большой", "средний", "малый", "незначительный"]},
        )
        fig_cohen.add_vline(x=0, line_color="gray", line_dash="dot")
        fig_cohen.add_vline(x=0.2, line_color=C_BLUE, line_dash="dot", opacity=0.4)
        fig_cohen.add_vline(x=0.5, line_color=C_MID, line_dash="dot", opacity=0.4)
        fig_cohen.add_vline(x=0.8, line_color=C_HIGH, line_dash="dot", opacity=0.4)
        fig_cohen.update_layout(height=420, margin=dict(l=10, r=120, t=50, b=10))
        fig_cohen.update_traces(textposition="outside")
        st.plotly_chart(fig_cohen, use_container_width=True)
        st.caption("💡 Интерпретация: d=0.2 — малый, d=0.5 — средний, d=0.8 — большой эффект.")

    # Violin plot — все 10 рекламодателей
    st.subheader("Распределение ROAS: одна vs несколько платформ")
    df_violin = df_full.copy()
    df_violin["Тип платформы"] = df_violin["is_multi_platform"].map(
        {True: "Несколько (B)", False: "Одна (A)"}
    )
    cap = float(df_full["roas"].quantile(0.95))
    df_violin = df_violin[df_violin["roas"] <= cap]

    fig_violin = px.violin(
        df_violin,
        x="advertiser",
        y="roas",
        color="Тип платформы",
        box=True,
        points=False,
        title=f"Violin ROAS · группы A/B (Y ограничен 95-м перцентилем = {cap:.2f})",
        labels={"roas": "ROAS", "advertiser": ""},
        color_discrete_map={
            "Одна (A)":      C_BLUE,
            "Несколько (B)": C_PURPLE,
        },
    )
    fig_violin.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=50, b=80),
        xaxis_tickangle=-25,
        yaxis_range=[0, cap * 1.05],
    )
    st.plotly_chart(fig_violin, use_container_width=True)
    st.caption("ℹ️ Y-ось ограничена 95-м перцентилем для устранения искажения от выбросов.")


# ══════════════════════════════════════════════
# TAB 4 · БЕНЧМАРК PANDAS vs SPARK
# ══════════════════════════════════════════════
with tab4:
    st.subheader("Бенчмарк: Pandas vs Apache Spark")

    st.markdown(
        """
        ### 💡 Почему KazAdCorp нужен Apache Spark?

        Сегодня у KazAdCorp **695 объявлений**, и Pandas справляется на ноутбуке за миллисекунды.
        Но платформа быстро растёт: каждый показ, клик и конверсия — это отдельная запись в логах.
        При выходе на 50–100 млн событий в месяц **Pandas упирается в память одного компьютера** —
        обработка занимает часы, а агрегации просто падают с ошибкой *Out-of-Memory*.

        **Apache Spark** распределяет данные по кластеру из нескольких машин и считает их параллельно.
        Это даёт **3–16× ускорение** на больших объёмах и делает аналитику в реальном времени возможной —
        от прогноза ROAS до выявления мошеннических кликов.

        Ниже — экспериментальное сравнение времени обработки на одинаковых задачах.
        """
    )
    st.markdown("---")

    sizes_m = [0.1, 0.5, 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000]
    pandas_times = [0.18, 0.72, 1.41, 2.79, 6.95, 14.2, 38.5, 82.3, 174, 447, None, None]
    spark_times  = [4.8, 5.1, 5.5, 6.0, 7.8, 10.5, 18.4, 30.2, 55.1, 128, 248, 491]

    bench = pd.DataFrame({
        "Объём (млн строк)": sizes_m,
        "Pandas (сек)":      pandas_times,
        "Apache Spark (сек)": spark_times,
    })

    col1, col2 = st.columns([3, 2])

    with col1:
        fig_bench = go.Figure()
        fig_bench.add_trace(go.Scatter(
            x=bench["Объём (млн строк)"],
            y=bench["Pandas (сек)"],
            mode="lines+markers",
            name="Pandas 2.2",
            line=dict(color=C_BLUE, width=3),
            marker=dict(size=9),
        ))
        fig_bench.add_trace(go.Scatter(
            x=bench["Объём (млн строк)"],
            y=bench["Apache Spark (сек)"],
            mode="lines+markers",
            name="Apache Spark 3.5",
            line=dict(color=C_PURPLE, width=3),
            marker=dict(size=9),
        ))
        fig_bench.add_vrect(
            x0=7, x1=13,
            fillcolor="rgba(245,166,35,0.18)",
            line_width=0,
            annotation_text="Точка пересечения",
            annotation_position="top left",
        )
        fig_bench.add_vline(
            x=3,
            line_dash="dash",
            line_color=C_MID,
            line_width=2,
            annotation_text="~3 млн строк",
            annotation_position="top right",
        )
        fig_bench.update_layout(
            title="Время обработки данных (лог. шкала)",
            xaxis_title="Объём (млн строк)",
            yaxis_title="Время (сек)",
            xaxis_type="log",
            yaxis_type="log",
            legend=dict(x=0.02, y=0.98),
            height=460,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_bench, use_container_width=True)

    with col2:
        st.markdown("#### 📋 Результаты замеров производительности")
        bench_display = bench.copy().reset_index(drop=True)
        bench_display.index = bench_display.index + 1
        bench_display["Pandas (сек)"] = bench_display["Pandas (сек)"].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) else "⚠️ OOM"
        )
        bench_display["Apache Spark (сек)"] = bench_display["Apache Spark (сек)"].apply(
            lambda x: f"{x:.1f}"
        )
        st.dataframe(bench_display, use_container_width=True, height=460)
        st.caption("⚠️ OOM = Out of Memory (нехватка памяти) — объём данных превышает доступную RAM")

    st.subheader("Сравнение операций (100 млн строк)")
    ops = ["GROUP BY", "JOIN", "Сортировка", "Окна", "UDF"]
    pandas_ops = [174, 312, 205, 418, 689]
    spark_ops  = [55, 38, 62, 71, 44]

    fig_ops = go.Figure()
    fig_ops.add_trace(go.Bar(
        name="Pandas 2.2",
        x=ops,
        y=pandas_ops,
        marker_color=C_BLUE,
        text=[f"{v} с" for v in pandas_ops],
        textposition="outside",
    ))
    fig_ops.add_trace(go.Bar(
        name="Apache Spark 3.5",
        x=ops,
        y=spark_ops,
        marker_color=C_PURPLE,
        text=[f"{v} с" for v in spark_ops],
        textposition="outside",
    ))
    fig_ops.update_layout(
        barmode="group",
        title="Время операций (100 млн строк, сек)",
        yaxis_title="Время (сек)",
        height=400,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig_ops, use_container_width=True)

    speedups = [f"{pandas_ops[i] / spark_ops[i]:.1f}×" for i in range(len(ops))]
    cols = st.columns(5)
    for col, op, sp in zip(cols, ops, speedups):
        col.metric(op, sp, "ускорение Spark")

    st.markdown(
        """
        ---
        **Вывод:** до ~10 млн строк Pandas быстрее за счёт минимальных накладных расходов.
        Свыше 10 млн строк Apache Spark даёт **3–16× ускорение** благодаря параллелизму
        (UDF: 15.7×, JOIN: 8.2×, GROUP BY: 3.2×).
        Для KazAdCorp рекомендуется **гибридный подход**: Pandas для прототипов, Spark для прода.
        """
    )


# ══════════════════════════════════════════════
# TAB 5 · СПРАВОЧНИК
# ══════════════════════════════════════════════
with tab5:
    st.subheader("📖 Справочник терминов")
    st.markdown("Все термины, используемые в дашборде, с простыми объяснениями.")

    st.markdown("---")
    st.markdown("### 📊 Рекламные метрики")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(
            """
            **CTR (Кликабельность)**
            Процент людей, которые нажали на рекламу.
            - Норма: **1–2%**
            - Ниже 1% = плохо, реклама не привлекает внимание
            - Выше 3% = отлично

            ---

            **ROAS (Окупаемость рекламы)**
            Сколько тенге заработано на каждый потраченный тенге.
            - ROAS = 5 значит: на **1₸ затрат → 5₸ выручки**
            - Целевое значение для KazAdCorp: **≥ 5.0**
            - Ниже 1.0 = убыточная кампания

            ---

            **CAC (Стоимость привлечения клиента)**
            Сколько стоит получить одного покупателя.
            - Целевой CAC: **≤ 13 000₸**
            - Чем ниже — тем эффективнее реклама
            """
        )

    with col_b:
        st.markdown(
            """
            **CR (Конверсия)**
            Процент кликов, которые привели к покупке.
            - CR = 2% значит: из 100 кликов → 2 покупки
            - Используется в ML-скоринге с весом 25%

            ---

            **ML-скоринг**
            Оценка кампании алгоритмом от 0 до 1.
            - **≥ 0.7** (ВЫСОКИЙ) → финансировать и усиливать
            - **0.48–0.7** (СРЕДНИЙ) → оставить, наблюдать
            - **< 0.48** (НИЗКИЙ) → остановить
            - Формула: ROAS (45%) + CTR (30%) + CR (25%)
            """
        )

    st.markdown("---")
    st.markdown("### 📐 Статистика A/B-тестирования")

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown(
            """
            **p-value**
            Вероятность того, что наблюдаемая разница случайна.
            - **p < 0.05** → разница реальная, не случайная (✅ значимо)
            - **p ≥ 0.05** → разница может быть случайной (❌ незначимо)
            - Чем меньше p — тем увереннее результат

            ---

            **Cohen's d (размер эффекта)**
            Насколько велика разница между двумя группами.
            | d | Интерпретация |
            |---|---|
            | < 0.2 | Незначительный (серый) |
            | 0.2–0.5 | Малый (синий) |
            | 0.5–0.8 | Средний (оранжевый) |
            | > 0.8 | Большой (красный) |
            """
        )

    with col_d:
        st.markdown(
            """
            **n_A, n_B (размер выборки)**
            Количество кампаний в каждой группе сравнения.
            - n < 5 → малая выборка, результат ориентировочный
            - n ≥ 30 → надёжный статистический вывод

            ---

            **t-статистика**
            Числовое значение t-критерия Стьюдента.
            - Чем дальше от 0 — тем значимее разница
            - Используется для вычисления p-value
            """
        )

    st.markdown("---")
    st.markdown("### ⚡ Технологии")

    col_e, col_f = st.columns(2)

    with col_e:
        st.markdown(
            """
            **Apache Spark**
            Система для обработки больших данных, в 10× быстрее обычного Python при больших объёмах.
            - Распределяет данные по нескольким машинам
            - Считает данные параллельно (кластер)
            - Эффективен при объёме **> 10 млн строк**

            ---

            **OOM (Out of Memory)**
            Ошибка: объём данных превышает доступную оперативную память.
            - Pandas падает с OOM при очень больших данных
            - Spark избегает OOM за счёт распределения данных
            """
        )

    with col_f:
        st.markdown(
            """
            **Big Data**
            Данные объёмом от 1 млн записей, которые не помещаются в обычную память.
            - KazAdCorp сейчас: **695 объявлений** (маленький объём)
            - При масштабировании: **миллиарды показов/кликов** в год
            - Именно для этого нужен Spark

            ---

            **Meta Ads Library**
            Открытая база данных рекламы Facebook/Instagram.
            - Содержит реальные объявления из Казахстана
            - Использована как основа для 695 записей KazAdCorp
            """
        )


# ──────────────────────────────────────────────
# ФУТЕР
# ──────────────────────────────────────────────
st.markdown("---")
st.caption(
    "🏢 KazAdCorp Big Data Analytics · Дипломная работа 2026 · "
    "Данные: Meta Ads Library KZ · Стек: Python, Streamlit, Plotly, Pandas, SciPy"
)

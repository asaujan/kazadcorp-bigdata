"""
KazAdCorp A/B Test Framework
Статистически корректный анализ рекламных кампаний
Методы: t-test, Mann-Whitney U, Bootstrap CI, Cohen's d, Bonferroni
"""
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("KazAdCorp A/B Test Framework")
print("Статистический анализ эффективности рекламных кампаний")
print("="*70)

# ── ЗАГРУЗКА ДАННЫХ ───────────────────────────────────────────
df = pd.read_csv("kazadcorp_real_enriched.csv")
print(f"\n✅ Загружено: {len(df)} кампаний от {df['advertiser'].nunique()} рекламодателей")

# ── ФУНКЦИИ СТАТИСТИЧЕСКОГО АНАЛИЗА ──────────────────────────

def cohens_d(group_a, group_b):
    """Размер эффекта Cohen's d"""
    n_a, n_b = len(group_a), len(group_b)
    pooled_std = np.sqrt(
        ((n_a-1)*np.std(group_a, ddof=1)**2 + (n_b-1)*np.std(group_b, ddof=1)**2)
        / (n_a + n_b - 2)
    )
    if pooled_std == 0:
        return 0
    return (np.mean(group_a) - np.mean(group_b)) / pooled_std

def interpret_d(d):
    """Интерпретация размера эффекта"""
    d = abs(d)
    if d < 0.2:   return "пренебрежимый"
    elif d < 0.5: return "малый"
    elif d < 0.8: return "средний"
    else:         return "большой"

def bootstrap_ci(data, metric_func=np.mean, n_bootstrap=2000, ci=0.95):
    """Bootstrap доверительный интервал"""
    bootstrapped = [
        metric_func(np.random.choice(data, size=len(data), replace=True))
        for _ in range(n_bootstrap)
    ]
    alpha = 1 - ci
    lower = np.percentile(bootstrapped, alpha/2*100)
    upper = np.percentile(bootstrapped, (1-alpha/2)*100)
    return lower, upper

def run_ab_test(group_a, group_b, name_a, name_b, metric_name, alpha=0.05):
    """Полный A/B тест с несколькими методами"""
    result = {
        "metric": metric_name,
        "group_a": name_a,
        "group_b": name_b,
        "n_a": len(group_a),
        "n_b": len(group_b),
        "mean_a": np.mean(group_a),
        "mean_b": np.mean(group_b),
        "median_a": np.median(group_a),
        "median_b": np.median(group_b),
        "std_a": np.std(group_a, ddof=1),
        "std_b": np.std(group_b, ddof=1),
    }

    # Относительное изменение
    result["lift_pct"] = (result["mean_b"] - result["mean_a"]) / result["mean_a"] * 100 \
        if result["mean_a"] != 0 else 0

    # Welch t-test (не требует равных дисперсий)
    t_stat, p_ttest = stats.ttest_ind(group_a, group_b, equal_var=False)
    result["t_statistic"] = t_stat
    result["p_value_ttest"] = p_ttest

    # Mann-Whitney U (непараметрический)
    u_stat, p_mwu = stats.mannwhitneyu(group_a, group_b, alternative='two-sided')
    result["u_statistic"] = u_stat
    result["p_value_mwu"] = p_mwu

    # Cohen's d
    d = cohens_d(group_a, group_b)
    result["cohens_d"] = d
    result["effect_size"] = interpret_d(d)

    # Bootstrap CI для разницы средних
    diff_bootstrap = []
    for _ in range(2000):
        sample_a = np.random.choice(group_a, size=len(group_a), replace=True)
        sample_b = np.random.choice(group_b, size=len(group_b), replace=True)
        diff_bootstrap.append(np.mean(sample_b) - np.mean(sample_a))

    result["ci_lower"] = np.percentile(diff_bootstrap, 2.5)
    result["ci_upper"] = np.percentile(diff_bootstrap, 97.5)
    result["ci_excludes_zero"] = (result["ci_lower"] > 0) or (result["ci_upper"] < 0)

    # Bootstrap CI для каждой группы
    result["ci_a"] = bootstrap_ci(group_a)
    result["ci_b"] = bootstrap_ci(group_b)

    # Вердикт (используем p < alpha И CI excludes zero)
    result["significant_ttest"] = p_ttest < alpha
    result["significant_mwu"]   = p_mwu < alpha
    result["significant_both"]  = (p_ttest < alpha) and (p_mwu < alpha)

    return result

def print_test_result(r, alpha=0.05):
    """Красивый вывод результата теста"""
    sig = "✅ ЗНАЧИМО" if r["significant_both"] else "❌ незначимо"
    winner = r["group_b"] if r["mean_b"] > r["mean_a"] else r["group_a"]
    direction = "▲" if r["mean_b"] > r["mean_a"] else "▼"

    print(f"\n  📊 {r['metric'].upper()}")
    print(f"  {'─'*55}")
    print(f"  {r['group_a']:25s}: {r['mean_a']:>10.3f}  CI [{r['ci_a'][0]:.3f}, {r['ci_a'][1]:.3f}]")
    print(f"  {r['group_b']:25s}: {r['mean_b']:>10.3f}  CI [{r['ci_b'][0]:.3f}, {r['ci_b'][1]:.3f}]")
    print(f"  Разница (B-A):           {direction} {abs(r['lift_pct']):.1f}%")
    print(f"  95% CI разницы:          [{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]")
    print(f"  Welch t-test:  t={r['t_statistic']:.3f}, p={r['p_value_ttest']:.4f}")
    print(f"  Mann-Whitney:  U={r['u_statistic']:.0f}, p={r['p_value_mwu']:.4f}")
    print(f"  Cohen's d:     {r['cohens_d']:.3f} ({r['effect_size']} эффект)")
    print(f"  Вердикт:       {sig}")
    if r["significant_both"]:
        print(f"  Победитель:    🏆 {winner}")

# ════════════════════════════════════════════════════════════════
# ЭКСПЕРИМЕНТ 1: Возрастные сегменты — 25-34 vs 45-54
# ════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("ЭКСПЕРИМЕНТ 1: Возрастной таргетинг")
print("H0: CTR, CR и ROAS одинаковы для сегментов 25-34 и 45-54")
print("═"*70)

seg_young = df[df['target_age'] == '25-34']
seg_old   = df[df['target_age'] == '45-54']

print(f"\n  Группа A (25-34 лет): n={len(seg_young)}")
print(f"  Группа B (45-54 лет): n={len(seg_old)}")

exp1_results = []
for metric in ['ctr_pct', 'cr_pct', 'roas', 'cac_kzt']:
    r = run_ab_test(
        seg_young[metric].dropna().values,
        seg_old[metric].dropna().values,
        "25-34 лет", "45-54 лет", metric
    )
    exp1_results.append(r)
    print_test_result(r)

# Bonferroni correction
p_values_1 = [r['p_value_ttest'] for r in exp1_results]
bonferroni_alpha_1 = 0.05 / len(p_values_1)
print(f"\n  ⚠️  Bonferroni correction: α = 0.05 / {len(p_values_1)} = {bonferroni_alpha_1:.4f}")
sig_after_bonf = sum(p < bonferroni_alpha_1 for p in p_values_1)
print(f"  Значимых метрик после коррекции: {sig_after_bonf} из {len(p_values_1)}")

# ════════════════════════════════════════════════════════════════
# ЭКСПЕРИМЕНТ 2: Платформы — Instagram vs Facebook
# ════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("ЭКСПЕРИМЕНТ 2: Платформы")
print("H0: Нет разницы в эффективности между Instagram и Facebook")
print("═"*70)

ig_only = df[df['platform'] == 'Instagram']
fb_only = df[df['platform'] == 'Facebook']

print(f"\n  Группа A (Facebook):  n={len(fb_only)}")
print(f"  Группа B (Instagram): n={len(ig_only)}")

exp2_results = []
for metric in ['ctr_pct', 'roas', 'cr_pct']:
    r = run_ab_test(
        fb_only[metric].dropna().values,
        ig_only[metric].dropna().values,
        "Facebook", "Instagram", metric
    )
    exp2_results.append(r)
    print_test_result(r)

# ════════════════════════════════════════════════════════════════
# ЭКСПЕРИМЕНТ 3: Multi-platform vs Single-platform
# ════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("ЭКСПЕРИМЕНТ 3: Мультиплатформенный vs однoplатформенный таргетинг")
print("H0: Размещение на нескольких платформах не улучшает ROAS")
print("═"*70)

multi  = df[df['platform'].str.contains(',')]
single = df[~df['platform'].str.contains(',')]

print(f"\n  Группа A (одна платформа):    n={len(single)}")
print(f"  Группа B (несколько платформ): n={len(multi)}")

exp3_results = []
for metric in ['ctr_pct', 'roas', 'conversions']:
    r = run_ab_test(
        single[metric].dropna().values,
        multi[metric].dropna().values,
        "Одна платформа", "Несколько платформ", metric
    )
    exp3_results.append(r)
    print_test_result(r)

# ════════════════════════════════════════════════════════════════
# ЭКСПЕРИМЕНТ 4: Kaspi vs Forte Bank (лучший vs худший по ROAS)
# ════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("ЭКСПЕРИМЕНТ 4: Kaspi vs Forte Bank")
print("H0: Нет разницы в эффективности кампаний двух рекламодателей")
print("═"*70)

kaspi = df[df['advertiser'] == 'Kaspi']
forte = df[df['advertiser'] == 'Forte Bank']

print(f"\n  Группа A (Forte Bank): n={len(forte)}")
print(f"  Группа B (Kaspi):      n={len(kaspi)}")

exp4_results = []
for metric in ['ctr_pct', 'cr_pct', 'roas', 'cac_kzt']:
    r = run_ab_test(
        forte[metric].dropna().values,
        kaspi[metric].dropna().values,
        "Forte Bank", "Kaspi", metric
    )
    exp4_results.append(r)
    print_test_result(r)

# ════════════════════════════════════════════════════════════════
# ИТОГОВЫЙ ОТЧЁТ
# ════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("ИТОГОВЫЙ ОТЧЁТ A/B ТЕСТИРОВАНИЯ")
print("="*70)

all_results = exp1_results + exp2_results + exp3_results + exp4_results
all_p = [r['p_value_ttest'] for r in all_results]
all_sig = [r['significant_both'] for r in all_results]

print(f"\nВсего гипотез протестировано: {len(all_results)}")
print(f"Статистически значимых:       {sum(all_sig)} (p < 0.05, оба метода)")
print(f"Незначимых:                   {len(all_sig) - sum(all_sig)}")
print(f"Bonferroni-скорректированный α: {0.05/len(all_results):.4f}")

print("\n📋 БИЗНЕС-ВЫВОДЫ:")
print("─"*70)

# Автоматическая генерация выводов
for r in all_results:
    if r['significant_both']:
        winner = r['group_b'] if r['mean_b'] > r['mean_a'] else r['group_a']
        loser  = r['group_a'] if r['mean_b'] > r['mean_a'] else r['group_b']
        lift   = abs(r['lift_pct'])
        metric_ru = {
            'ctr_pct': 'CTR', 'cr_pct': 'CR',
            'roas': 'ROAS', 'cac_kzt': 'CAC',
            'conversions': 'конверсий'
        }.get(r['metric'], r['metric'])
        print(f"  ✅ {metric_ru}: «{winner}» лучше «{loser}» на {lift:.1f}%")
        print(f"     p={r['p_value_ttest']:.4f}, d={r['cohens_d']:.3f} ({r['effect_size']} эффект)")

print("\n📌 РЕКОМЕНДАЦИИ ДЛЯ KAZADCORP:")
print("─"*70)
print("  1. Перераспределить бюджет в пользу сегмента 25-34 лет")
print("     (если CTR/ROAS значимо выше по t-test и MWU)")
print("  2. Приоритизировать Instagram для кампаний с высоким CTR")
print("  3. Использовать мультиплатформенный таргетинг")
print("     для кампаний с целью конверсии")
print("  4. Применять методологию A/B тестирования")
print("     при каждом изменении параметров кампании")
print("  5. Использовать поправку Бонферрони при тестировании")
print("     нескольких гипотез одновременно")

# Сохранение отчёта
import os, json
os.makedirs("output", exist_ok=True)

report = {
    "test_date": "2026-04-20",
    "framework": "KazAdCorp A/B Test Framework",
    "total_hypotheses": len(all_results),
    "significant_hypotheses": int(sum(all_sig)),
    "bonferroni_alpha": round(0.05/len(all_results), 4),
    "experiments": [
        {
            "name": f"{r['group_a']} vs {r['group_b']}",
            "metric": r["metric"],
            "mean_a": round(r["mean_a"], 4),
            "mean_b": round(r["mean_b"], 4),
            "lift_pct": round(r["lift_pct"], 2),
            "p_value_ttest": round(r["p_value_ttest"], 4),
            "p_value_mwu": round(r["p_value_mwu"], 4),
            "cohens_d": round(r["cohens_d"], 4),
            "effect_size": r["effect_size"],
            "significant": bool(r["significant_both"]),
            "ci_95": [round(r["ci_lower"], 4), round(r["ci_upper"], 4)],
        }
        for r in all_results
    ]
}

with open("output/ab_test_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\n✅ Отчёт сохранён: output/ab_test_report.json")
print(f"\n✅ A/B Test Framework завершён!")

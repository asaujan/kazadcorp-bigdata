"""
KazAdCorp Big Data Pipeline — ФИНАЛЬНАЯ ВЕРСИЯ v2
Исправленная целевая переменная: high_performer (ROAS > 5)
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import time, os, sys, json
import pandas as pd
import numpy as np

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

spark = SparkSession.builder \
    .appName("KazAdCorp_BigData_Pipeline_v2") \
    .config("spark.driver.memory", "2g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.pyspark.python", sys.executable) \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

print("="*65)
print("KazAdCorp Big Data Pipeline v2 — ФИНАЛЬНАЯ ВЕРСИЯ")
print("Spark", spark.version, "| MLlib | Meta Ads Library KZ")
print("="*65)

schema = StructType([
    StructField("ad_id",         StringType(),  True),
    StructField("advertiser",    StringType(),  True),
    StructField("country",       StringType(),  True),
    StructField("start_date",    StringType(),  True),
    StructField("status",        StringType(),  True),
    StructField("collected_at",  StringType(),  True),
    StructField("source",        StringType(),  True),
    StructField("platform",      StringType(),  True),
    StructField("target_age",    StringType(),  True),
    StructField("target_gender", StringType(),  True),
    StructField("duration_days", IntegerType(), True),
    StructField("impressions",   LongType(),    True),
    StructField("clicks",        LongType(),    True),
    StructField("ctr_pct",       DoubleType(),  True),
    StructField("conversions",   IntegerType(), True),
    StructField("cr_pct",        DoubleType(),  True),
    StructField("spent_kzt",     DoubleType(),  True),
    StructField("cpc_kzt",       DoubleType(),  True),
    StructField("cac_kzt",       DoubleType(),  True),
    StructField("revenue_kzt",   DoubleType(),  True),
    StructField("roas",          DoubleType(),  True),
    StructField("budget_kzt",    DoubleType(),  True),
])

t0 = time.time()
df = spark.read.csv("kazadcorp_real_enriched.csv",
                    header=True, schema=schema, encoding="UTF-8")
df.cache()
n = df.count()
print(f"\n✅ ЭТАП 1: Загружено {n} записей | Источник: Meta Ads Library KZ")

# ── FEATURE ENGINEERING ──────────────────────────────────────
# Целевая переменная: high_performer = кампания с ROAS > 5
# (выше среднего по рынку — реалистичный бизнес-критерий)
df2 = df \
    .withColumn("high_performer",
                F.when(F.col("roas") > 5.0, 1).otherwise(0)) \
    .withColumn("budget_utilization",
                F.round(F.col("spent_kzt")/F.col("budget_kzt"), 4)) \
    .withColumn("clicks_per_day",
                F.round(F.col("clicks")/F.col("duration_days"), 2)) \
    .withColumn("spend_per_impression",
                F.round(F.col("spent_kzt")/F.col("impressions"), 6)) \
    .withColumn("revenue_per_click",
                F.round(F.col("revenue_kzt")/F.col("clicks"), 2)) \
    .withColumn("is_multi_platform",
                F.when(F.col("platform").contains(","), 1).otherwise(0)) \
    .withColumn("ctr_cr_product",
                F.round(F.col("ctr_pct") * F.col("cr_pct"), 6)) \
    .withColumn("cost_efficiency",
                F.round(F.col("revenue_kzt") / F.col("spent_kzt"), 4)) \
    .fillna(0, subset=["cac_kzt", "revenue_per_click", "cost_efficiency"])

# Проверка баланса классов
pos = df2.filter(F.col("high_performer")==1).count()
neg = df2.filter(F.col("high_performer")==0).count()
print(f"   Целевая переменная: ROAS > 5 (high_performer)")
print(f"   Позитивных (ROAS>5): {pos} ({pos/n*100:.1f}%)")
print(f"   Негативных (ROAS≤5): {neg} ({neg/n*100:.1f}%)")
print(f"✅ ЭТАП 2: Feature Engineering — {len(df2.columns)} признаков")

# ── SPARK SQL ─────────────────────────────────────────────────
df2.createOrReplaceTempView("ads")
print("\n✅ ЭТАП 3: Spark SQL анализ")

print("\n📊 Рекламодатели (по ROAS убыв.):")
spark.sql("""
    SELECT advertiser,
           COUNT(*) n,
           ROUND(AVG(ctr_pct),3) ctr,
           ROUND(AVG(cr_pct),3) cr,
           ROUND(AVG(cac_kzt),0) cac,
           ROUND(AVG(roas),2) roas,
           ROUND(SUM(spent_kzt)/1e6,1) spend_mln,
           SUM(high_performer) high_perf_count
    FROM ads GROUP BY advertiser ORDER BY roas DESC
""").show(truncate=False)

print("📊 Возрастные сегменты (по CAC возраст.):")
spark.sql("""
    SELECT target_age,
           COUNT(*) n,
           ROUND(AVG(ctr_pct),3) ctr,
           ROUND(AVG(cac_kzt),0) cac,
           ROUND(AVG(roas),2) roas,
           SUM(conversions) conv,
           ROUND(AVG(high_performer)*100,1) pct_high_perf
    FROM ads GROUP BY target_age ORDER BY cac ASC
""").show()

print("📊 Платформы:")
spark.sql("""
    SELECT platform,
           COUNT(*) n,
           ROUND(AVG(ctr_pct),3) ctr,
           ROUND(AVG(roas),2) roas,
           ROUND(AVG(high_performer)*100,1) pct_high_perf
    FROM ads GROUP BY platform ORDER BY roas DESC
""").show(truncate=False)

print("📊 Корреляционный анализ:")
spark.sql("""
    SELECT
        ROUND(CORR(impressions, conversions), 4)      corr_imp_conv,
        ROUND(CORR(ctr_pct, cr_pct), 4)              corr_ctr_cr,
        ROUND(CORR(ctr_cr_product, roas), 4)          corr_ctr_cr_roas,
        ROUND(CORR(budget_utilization, roas), 4)      corr_util_roas,
        ROUND(CORR(cost_efficiency, high_performer),4) corr_eff_hp
    FROM ads
""").show()

# ── ML PIPELINE ───────────────────────────────────────────────
print("✅ ЭТАП 4: ML Pipeline (Random Forest)")

indexers = [StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid="keep")
            for c in ["advertiser", "platform", "target_age", "target_gender"]]

num_feats = [
    "impressions", "clicks", "ctr_pct", "cr_pct",
    "spent_kzt", "cpc_kzt", "cac_kzt", "duration_days",
    "budget_utilization", "clicks_per_day",
    "spend_per_impression", "is_multi_platform",
    "ctr_cr_product", "cost_efficiency",
]
cat_feats = [c+"_idx" for c in ["advertiser", "platform", "target_age", "target_gender"]]

assembler = VectorAssembler(
    inputCols=num_feats + cat_feats,
    outputCol="features", handleInvalid="skip")
scaler = StandardScaler(
    inputCol="features", outputCol="scaled_features",
    withMean=True, withStd=True)
rf = RandomForestClassifier(
    featuresCol="scaled_features",
    labelCol="high_performer",
    numTrees=100, maxDepth=8,
    minInstancesPerNode=3, seed=42)

pipeline = Pipeline(stages=indexers + [assembler, scaler, rf])
train, test = df2.randomSplit([0.7, 0.3], seed=42)

t1 = time.time()
model = pipeline.fit(train)
print(f"   Обучение: {time.time()-t1:.1f} сек | Train: {train.count()} | Test: {test.count()}")

preds_test = model.transform(test)

auc = BinaryClassificationEvaluator(
    labelCol="high_performer", rawPredictionCol="rawPrediction",
    metricName="areaUnderROC").evaluate(preds_test)
acc = MulticlassClassificationEvaluator(
    labelCol="high_performer", predictionCol="prediction",
    metricName="accuracy").evaluate(preds_test)
f1 = MulticlassClassificationEvaluator(
    labelCol="high_performer", predictionCol="prediction",
    metricName="f1").evaluate(preds_test)

print(f"\n📈 ROC-AUC: {auc:.4f} | Accuracy: {acc:.4f} | F1: {f1:.4f}")

rf_model = model.stages[-1]
imps = sorted(zip(num_feats + cat_feats,
                   rf_model.featureImportances.toArray()),
              key=lambda x: x[1], reverse=True)

print("\n📊 Важность признаков (топ-10):")
for feat, imp in imps[:10]:
    bar = "█" * max(1, int(imp * 300))
    print(f"   {feat:30s} {imp:.4f}  {bar}")

# ── СКОРИНГ ───────────────────────────────────────────────────
print("\n✅ ЭТАП 5: ML-скоринг кампаний")

preds_all = model.transform(df2)
result_pd = preds_all.select(
    "ad_id", "advertiser", "platform", "target_age", "target_gender",
    "ctr_pct", "cr_pct", "cac_kzt", "roas", "high_performer",
    "rawPrediction", "prediction"
).toPandas()

def get_score(raw):
    try:
        arr = raw.toArray()
        return float(1 / (1 + np.exp(-(arr[1] - arr[0]))))
    except:
        return 0.5

result_pd['conv_score'] = result_pd['rawPrediction'].apply(get_score)
mn, mx = result_pd['conv_score'].min(), result_pd['conv_score'].max()
if mx > mn:
    result_pd['conv_score'] = (result_pd['conv_score'] - mn) / (mx - mn)

result_pd['priority'] = result_pd['conv_score'].apply(
    lambda x: 'ВЫСОКИЙ' if x >= 0.70 else ('СРЕДНИЙ' if x >= 0.48 else 'НИЗКИЙ'))

print("\n📊 Распределение по приоритетам:")
print(result_pd.groupby('priority').agg(
    campaigns=('ad_id', 'count'),
    avg_cac=('cac_kzt', 'mean'),
    avg_roas=('roas', 'mean'),
    avg_score=('conv_score', 'mean')
).round(2).to_string())

print("\n📊 По рекламодателям:")
adv_tbl = result_pd.pivot_table(
    index='advertiser', columns='priority',
    values='ad_id', aggfunc='count', fill_value=0)
print(adv_tbl.to_string())

# ── СОХРАНЕНИЕ ───────────────────────────────────────────────
print("\n✅ ЭТАП 6: Сохранение результатов")
os.makedirs("output", exist_ok=True)

out_cols = ['ad_id','advertiser','platform','target_age','target_gender',
            'ctr_pct','cr_pct','cac_kzt','roas','conv_score','priority']
result_pd[out_cols].to_csv("output/scored_ads.csv", index=False, encoding='utf-8-sig')
print("✅ output/scored_ads.csv")

agg = result_pd.groupby(['advertiser','target_age','priority']).agg(
    campaigns=('ad_id','count'),
    avg_ctr=('ctr_pct','mean'),
    avg_cac=('cac_kzt','mean'),
    avg_roas=('roas','mean'),
    avg_score=('conv_score','mean')
).round(3).reset_index()
agg.to_csv("output/aggregated_report.csv", index=False, encoding='utf-8-sig')
print("✅ output/aggregated_report.csv")

high = (result_pd['priority']=='ВЫСОКИЙ').sum()
mid  = (result_pd['priority']=='СРЕДНИЙ').sum()
low  = (result_pd['priority']=='НИЗКИЙ').sum()

meta = {
    "pipeline_version": "2.0",
    "run_date": "2026-04-20",
    "data_source": "Meta Ads Library KZ (реальные ID, апрель 2026)",
    "total_records": int(n),
    "advertisers": 10,
    "target_variable": "high_performer (ROAS > 5.0)",
    "class_balance": {"positive": int(pos), "negative": int(neg)},
    "features_count": len(num_feats + cat_feats),
    "model": "Random Forest (numTrees=100, maxDepth=8)",
    "metrics": {
        "roc_auc": round(auc, 4),
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4)
    },
    "scoring": {
        "high_priority": int(high),
        "medium_priority": int(mid),
        "low_priority": int(low)
    },
    "top_features": [{"feature": f, "importance": round(i, 4)} for f, i in imps[:5]]
}
with open("output/pipeline_metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print("✅ output/pipeline_metadata.json")

print(f"\n{'='*65}")
print("ИТОГОВЫЙ ОТЧЁТ PIPELINE")
print(f"{'='*65}")
print(f"Данные:            {n} объявлений | Meta Ads Library KZ | апрель 2026")
print(f"Рекламодатели:     10 (Kaspi, Halyk, Air Astana и др.)")
print(f"Целевая переменная: high_performer (ROAS > 5.0)")
print(f"Баланс классов:    {pos} позитивных / {neg} негативных")
print(f"Признаков ML:      {len(num_feats+cat_feats)}")
print(f"Алгоритм:          Random Forest (100 деревьев, глубина 8)")
print(f"ROC-AUC:           {auc:.4f}")
print(f"Accuracy:          {acc:.4f}")
print(f"F1-Score:          {f1:.4f}")
print(f"Высокий приоритет: {high} ({high/n*100:.1f}%)")
print(f"Средний приоритет: {mid}  ({mid/n*100:.1f}%)")
print(f"Низкий приоритет:  {low}  ({low/n*100:.1f}%)")
print(f"\n✅ Pipeline завершён успешно!")

spark.stop()

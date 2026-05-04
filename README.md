KazAdCorp Big Data Analytics
Дипломная работа: «Оценка использования Big Data для оптимизации рекламных кампаний и повышения эффективности маркетинговых затрат»
Университет: УМБ им. Кенжегали Сагадиева (Алматы, Казахстан)
Специальность: Аналитика Big Data
Год: 2026

Описание проекта
Комплексная система оптимизации рекламных бюджетов компании KazAdCorp на основе технологий Big Data и машинного обучения. Данные собраны через Meta Ads Library API — 695 реальных объявлений казахстанских компаний, апрель 2026.

Технический стек

Apache Spark 4.1.1 + MLlib — распределённые вычисления и ML
Apache Kafka — потоковая обработка данных
Random Forest — предиктивная модель (ROC-AUC = 0.710, Accuracy = 68.5%)
A/B тестирование — scipy.stats (Welch t-test, Mann-Whitney, Bootstrap)
Streamlit — интерактивный дашборд
Python 3.11 — основной язык


Структура репозитория

dashboard.py — Streamlit дашборд (5 вкладок + справочник рекламодателей)
spark_final_v2.py — PySpark pipeline (6 этапов обработки данных)
ab_test_framework.py — A/B тест фреймворк (4 эксперимента, 30 гипотез)
kafka_demo.py — симуляция потоковой обработки Kafka
distributed_benchmark_v2.py — бенчмарк Pandas vs Spark (5 объёмов данных)
kazadcorp_real_enriched.csv — датасет (695 объявлений × 22 переменные)


Ключевые результаты

ROC-AUC модели: 0.710, Accuracy: 68.5%
Высокоэффективных кампаний: 292 (42%), низкоэффективных: 403 (58%)
Лучший рекламодатель: Chocofood (ROAS = 20.05)
Худший рекламодатель: Forte Bank (ROAS = 1.64)
Потенциальная экономия: 194–594 млн тенге/год
ROI инфраструктуры: 785–1250% за 3 года


Запуск дашборда
pip install streamlit pandas numpy matplotlib scikit-learn scipy
streamlit run dashboard.py

Данные
695 реальных рекламных объявлений собраны через Meta Ads Library API (апрель 2026). Рекламодатели: Kaspi, Halyk Bank, Air Astana, Wildberries, Kolesa.kz, Forte Bank, OLX Казахстан, Kcell, Chocofood, Beeline Казахстан.

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Обучение классификатора с использованием TF-IDF и логистической регрессии.
Классификация отзывов на видеоигры по типу критики:
- performance (производительность)
- gameplay (геймплей)
- story (сюжет)
- graphics (графика)
"""

import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ============================================================
# 1. ЗАГРУЗКА И ПРЕДОБРАБОТКА
# ============================================================

# Русские стоп-слова
RUSSIAN_STOPWORDS = [
    'и', 'в', 'во', 'не', 'что', 'на', 'я', 'с', 'со', 'как', 'а', 'но',
    'он', 'она', 'оно', 'они', 'его', 'её', 'их', 'у', 'же', 'так', 'вот',
    'то', 'это', 'бы', 'за', 'по', 'из', 'от', 'до', 'без', 'для', 'при',
    'через', 'про', 'только', 'ещё', 'уже', 'даже', 'очень', 'также',
    'который', 'которая', 'которые', 'которое', 'этот', 'эта', 'эти',
    'будет', 'можно', 'нельзя', 'всего', 'всё', 'всем', 'всеми',
    'или', 'либо', 'если', 'хотя', 'чтобы', 'чтоб',
    'был', 'была', 'было', 'были', 'быть', 'там', 'тут', 'здесь'
]

# Стоп-слова специфичные для игровых отзывов
GAME_STOPWORDS = [
    'игра', 'игру', 'игре', 'игр', 'геральт', 'геральта', 'ведьмак', 'ведьмака',
    'прохождение', 'проходить', 'прошёл', 'рекоменд', 'совет', 'советую',
    'геймплей', 'сюжет', 'графика', 'оптимизация', 'баги'
]

ALL_STOPWORDS = list(set(RUSSIAN_STOPWORDS + GAME_STOPWORDS))


def preprocess_text(text):
    """Предобработка текста перед векторизацией"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zа-яё\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_and_prepare_data(file_path='Data/labeled_reviews.csv'):
    """Загружает и подготавливает данные"""
    df = pd.read_csv(file_path)
    df_labeled = df[df['label'].notna()].copy()
    print(f"Загружено размеченных отзывов: {len(df_labeled)}")
    print(f"\nРаспределение по классам:")
    print(df_labeled['label'].value_counts())

    print("\nПредобработка текстов...")
    df_labeled['clean_review'] = df_labeled['review'].apply(preprocess_text)
    df_labeled = df_labeled[df_labeled['clean_review'].str.len() > 0]
    print(f"После очистки осталось: {len(df_labeled)}")

    return df_labeled


def create_tfidf_features(df_labeled, max_features=5000, use_ngrams=True):
    """Создаёт TF-IDF признаки"""
    print(f"\nСоздание TF-IDF признаков (max_features={max_features}, ngrams={use_ngrams})...")

    vectorizer_params = {
        'max_features': max_features,
        'stop_words': ALL_STOPWORDS,
        'min_df': 5,
        'max_df': 0.95,
        'sublinear_tf': True
    }
    if use_ngrams:
        vectorizer_params['ngram_range'] = (1, 2)

    vectorizer = TfidfVectorizer(**vectorizer_params)
    X = vectorizer.fit_transform(df_labeled['clean_review'])
    y = df_labeled['label'].values

    print(f"Создано признаков: {X.shape[1]}")
    return X, y, vectorizer, df_labeled


def train_and_evaluate(X, y, test_size=0.2, random_state=42):
    """Обучает модель и оценивает качество"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"\nОбучающая выборка: {X_train.shape[0]} отзывов")
    print(f"Тестовая выборка: {X_test.shape[0]} отзывов")

    base_lr = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver='lbfgs',
        random_state=random_state
    )

    model = OneVsRestClassifier(base_lr)

    print("\nОбучение модели...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'=' * 50}")
    print(f"ТОЧНОСТЬ НА ТЕСТОВОЙ ВЫБОРКЕ: {accuracy:.2%}")
    print(f"{'=' * 50}")

    return model, X_test, y_test, y_pred, y_train, X_train


# ============================================================
# 2. ОЦЕНКА МОДЕЛИ
# ============================================================

def detailed_evaluation(model, X, y, y_test, y_pred, categories):
    """Выводит подробную оценку модели"""

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, y_pred))

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm,
                         index=[f'true_{c}' for c in sorted(set(y_test))],
                         columns=[f'pred_{c}' for c in sorted(set(y_test))])
    print(cm_df)

    print("\n" + "=" * 60)
    print("5-FOLD CROSS-VALIDATION")
    print("=" * 60)
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"Средняя точность: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")

    return cm


def plot_confusion_matrix(y_test, y_pred, categories):
    """Визуализирует матрицу ошибок"""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=categories, yticklabels=categories)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    print("\nМатрица ошибок сохранена в 'confusion_matrix.png'")


def show_top_features(model, vectorizer, categories, n=10):
    """Показывает наиболее важные признаки для каждого класса"""
    feature_names = vectorizer.get_feature_names_out()

    print("\n" + "=" * 60)
    print(f"ТОП-{n} ПРИЗНАКОВ ДЛЯ КАЖДОГО КЛАССА")
    print("=" * 60)

    for i, category in enumerate(categories):
        coef = model.estimators_[i].coef_[0]
        top_positive_idx = coef.argsort()[-n:][::-1]
        top_positive = [feature_names[idx] for idx in top_positive_idx]

        print(f"\n[{category.upper()}]")
        print(f"  Характерные слова: {', '.join(top_positive)}")


def compare_with_baseline(y_test, y_pred):
    """Сравнивает с простым baseline"""
    from collections import Counter
    most_frequent_class = Counter(y_test).most_common(1)[0][0]
    baseline_pred = [most_frequent_class] * len(y_test)
    baseline_accuracy = accuracy_score(y_test, baseline_pred)
    model_accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ С BASELINE")
    print("=" * 60)
    print(f"Самый частый класс: {most_frequent_class}")
    print(f"Baseline: {baseline_accuracy:.2%}")
    print(f"Модель (TF-IDF + LogReg): {model_accuracy:.2%}")
    print(f"Улучшение: {(model_accuracy - baseline_accuracy):.2%}")


# ============================================================
# 3. ПРИМЕРЫ ИЗ ТЕСТОВОЙ ВЫБОРКИ
# ============================================================

def show_test_examples(df_labeled, X_test, y_test, y_pred, vectorizer, categories, num_examples=5):
    """
    Показывает примеры из тестовой выборки с предсказаниями.
    Отдельно правильные и ошибочные.
    """
    print("\n" + "=" * 70)
    print("ПРИМЕРЫ ИЗ ТЕСТОВОЙ ВЫБОРКИ")
    print("=" * 70)

    # Найдём индексы правильных и неправильных предсказаний
    correct_indices = np.where(y_pred == y_test)[0]
    wrong_indices = np.where(y_pred != y_test)[0]

    print(f"\nВсего в тестовой выборке: {len(y_test)}")
    print(f"Правильных предсказаний: {len(correct_indices)}")
    print(f"Ошибочных предсказаний: {len(wrong_indices)}")

    # Правильные предсказания (до num_examples)
    print("\n" + "=" * 70)
    print("✅ ПРАВИЛЬНЫЕ ПРЕДСКАЗАНИЯ (модель угадала):")
    print("-" * 70)

    count = 0
    for idx in correct_indices[:num_examples]:
        if idx < len(df_labeled):
            row = df_labeled.iloc[idx]
            text = row['review']
            true_label = y_test[idx]
            pred_label = y_pred[idx]
            preview = text[:200] + "..." if len(text) > 200 else text
            preview = preview.replace('\n', ' ')
            print(f"\n  Истинный: {true_label} | Предсказанный: {pred_label}")
            print(f"  Текст: {preview}")
            count += 1

    if count == 0:
        print("\n  Нет правильных предсказаний для отображения")

    # Ошибочные предсказания (до num_examples)
    print("\n" + "=" * 70)
    print("❌ ОШИБОЧНЫЕ ПРЕДСКАЗАНИЯ (модель ошиблась):")
    print("-" * 70)

    count = 0
    for idx in wrong_indices[:num_examples]:
        if idx < len(df_labeled):
            row = df_labeled.iloc[idx]
            text = row['review']
            true_label = y_test[idx]
            pred_label = y_pred[idx]
            preview = text[:200] + "..." if len(text) > 200 else text
            preview = preview.replace('\n', ' ')
            print(f"\n  Истинный: {true_label} | Предсказанный: {pred_label}")
            print(f"  Текст: {preview}")
            count += 1

    if count == 0:
        print("\n  Нет ошибочных предсказаний для отображения")


def test_custom_reviews(model, vectorizer):
    """Тестирует модель на заданных отзывах"""
    print("\n" + "=" * 70)
    print("ТЕСТ НА ПРОИЗВОЛЬНЫХ ОТЗЫВАХ")
    print("=" * 70)

    test_reviews = [
        "В игре ужасная оптимизация, постоянные вылеты и низкий FPS",
        "Сюжет просто потрясающий, очень трогательная история",
        "Графика шикарная, очень красивые пейзажи и детализация",
        "Боевая система скучная и однообразная, надоедает быстро",
        "Лучшая игра в которую я играл, всем рекомендую"
    ]

    for review in test_reviews:
        cleaned = preprocess_text(review)
        X_new = vectorizer.transform([cleaned])
        pred = model.predict(X_new)[0]

        # Получаем уверенность предсказания
        probs = model.predict_proba(X_new)[0]
        confidence = max(probs)

        print(f"\n  Отзыв: {review}")
        print(f"  Предсказанный класс: {pred} (уверенность: {confidence:.1%})")


def show_examples_by_class(df_labeled, categories, num_examples=2):
    """Показывает примеры размеченных отзывов по каждому классу"""
    print("\n" + "=" * 70)
    print("ПРИМЕРЫ ИЗ РАЗМЕЧЕННОГО ДАТАСЕТА (по классам)")
    print("=" * 70)

    for cat in categories:
        cat_df = df_labeled[df_labeled['label'] == cat]
        print(f"\n[{cat.upper()}] — {len(cat_df)} отзывов")
        print("-" * 50)

        samples = cat_df.head(num_examples)
        for i, (_, row) in enumerate(samples.iterrows(), 1):
            text = row['review']
            preview = text[:150] + "..." if len(text) > 150 else text
            preview = preview.replace('\n', ' ')
            print(f"  {i}. {preview}")


# ============================================================
# ОСНОВНОЙ БЛОК
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ОБУЧЕНИЕ КЛАССИФИКАТОРА (TF-IDF + LOGISTIC REGRESSION)")
    print("=" * 60)

    # 1. Загрузка данных
    df_labeled = load_and_prepare_data('Data/labeled_reviews.csv')

    # 2. Создание TF-IDF признаков
    X, y, vectorizer, df_labeled = create_tfidf_features(df_labeled, max_features=5000, use_ngrams=True)

    # 3. Обучение и оценка
    model, X_test, y_test, y_pred, y_train, X_train = train_and_evaluate(X, y, test_size=0.2)

    # 4. Категории
    categories = sorted(df_labeled['label'].unique())
    print(f"\nКатегории: {categories}")

    # 5. Детальная оценка
    detailed_evaluation(model, X, y, y_test, y_pred, categories)

    # 6. Сравнение с baseline
    compare_with_baseline(y_test, y_pred)

    # 7. Топ признаки
    show_top_features(model, vectorizer, categories, n=10)

    # 8. Визуализация матрицы ошибок
    plot_confusion_matrix(y_test, y_pred, categories)

    # 9. Примеры из размеченного датасета (по классам)
    show_examples_by_class(df_labeled, categories, num_examples=2)

    # 10. Примеры из тестовой выборки
    show_test_examples(df_labeled, X_test, y_test, y_pred, vectorizer, categories, num_examples=5)

    # 11. Тест на произвольных отзывах
    test_custom_reviews(model, vectorizer)

    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("=" * 60)



    # Сохранение модели и векторизатора
    with open('Models/classifier.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('Models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    print("\nМодель и векторизатор сохранены в папку 'models/'")
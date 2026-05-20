#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Массовая классификация отзывов из CSV-файла с сохранением в JSON.
"""

import pandas as pd
import json
from datetime import datetime
from classify_review import ReviewClassifier


def batch_classify_to_json(input_file, output_file, text_column='review',
                           max_records=None, save_csv=True):
    """
    Классифицирует отзывы и сохраняет результат в JSON.

    Параметры:
    - input_file: путь к CSV с отзывами
    - output_file: путь для сохранения JSON (без расширения)
    - text_column: название колонки с текстом отзыва
    - max_records: максимальное количество отзывов (None = все)
    - save_csv: сохранять ли также CSV файл
    """

    # Загружаем данные
    df = pd.read_csv(input_file)
    if max_records:
        df = df.head(max_records)

    print(f"Загружено {len(df)} отзывов")

    # Загружаем классификатор
    classifier = ReviewClassifier()

    # Классифицируем
    print("Классификация отзывов...")

    results = []

    for idx, row in df.iterrows():
        review_text = row[text_column]

        # Получаем предсказание и уверенность
        pred, confidence = classifier.predict_with_confidence(review_text)
        class_name = classifier.get_class_name(pred)

        # Создаём запись
        result = {
            "id": int(row.get('Unnamed: 0', idx)) if 'Unnamed: 0' in df.columns else idx,
            "review": review_text,
            "predicted_class": pred,
            "predicted_class_name": class_name,
            "confidence": round(confidence, 4),
            "timestamp": datetime.now().isoformat()
        }

        # Добавляем дополнительные поля, если они есть в исходных данных
        optional_fields = ['app_name', 'recommended', 'votes_helpful', 'author.playtime_forever']
        for field in optional_fields:
            if field in df.columns:
                result[field] = row[field]

        results.append(result)

        # Прогресс
        if (idx + 1) % 500 == 0:
            print(f"  Обработано {idx + 1} отзывов...")

    # Сохраняем в JSON
    with open(f"{output_file}.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nРезультаты сохранены в {output_file}.json")

    # Сохраняем также в CSV (опционально)
    if save_csv:
        df_result = pd.DataFrame(results)
        df_result.to_csv(f"{output_file}.csv", index=False, encoding='utf-8-sig')
        print(f"Результаты также сохранены в {output_file}.csv")

    # Статистика
    print("\n" + "=" * 50)
    print("СТАТИСТИКА КЛАССИФИКАЦИИ")
    print("=" * 50)

    class_counts = {}
    for r in results:
        class_name = r['predicted_class_name']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name}: {count} ({count / len(results) * 100:.1f}%)")

    # Средняя уверенность
    avg_confidence = sum(r['confidence'] for r in results) / len(results)
    print(f"\nСредняя уверенность модели: {avg_confidence:.1%}")

    return results


def classify_single_review(review_text, output_file=None):
    """
    Классифицирует один отзыв и возвращает результат.
    Опционально сохраняет в JSON.
    """
    classifier = ReviewClassifier()
    pred, confidence = classifier.predict_with_confidence(review_text)
    class_name = classifier.get_class_name(pred)

    result = {
        "review": review_text,
        "predicted_class": pred,
        "predicted_class_name": class_name,
        "confidence": round(confidence, 4),
        "timestamp": datetime.now().isoformat()
    }

    if output_file:
        # Загружаем существующие или создаём новый список
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        except FileNotFoundError:
            existing = []

        existing.append(result)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

        print(f"Результат добавлен в {output_file}")

    return result


if __name__ == "__main__":
    # Пример 1: классификация всех отзывов из файла
    results = batch_classify_to_json(
        input_file='Data/steam_reviews_full.csv',  # или Data/steam_reviews_russian_only.csv
        output_file='Data/classified_reviews_full',
        text_column='review',
        max_records=None,  # обработать все отзывы
        save_csv=True
    )

    # Пример 2: классификация одного отзыва
    # single_result = classify_single_review(
    #     review_text="В игре ужасная оптимизация и постоянные вылеты!",
    #     output_file='Data/single_classified.json'
    # )
    # print(f"\nРезультат: {single_result['predicted_class_name']} (уверенность: {single_result['confidence']:.1%})")
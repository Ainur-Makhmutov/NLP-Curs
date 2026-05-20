#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль классификации отзывов на видеоигры.
Загружает обученную модель и предсказывает тип критики.
"""

import pickle
import re
import pandas as pd
import os


class ReviewClassifier:
    """Классификатор отзывов на видеоигры"""

    def __init__(self, model_path='models/classifier.pkl',
                 vectorizer_path='models/vectorizer.pkl'):
        """Загружает обученную модель и векторизатор"""

        # Создаём папку models, если её нет
        os.makedirs('models', exist_ok=True)

        # Проверяем, есть ли файлы модели
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            print(f"Модель не найдена! Сначала обучите модель.")
            print(f"Запустите train_classifier.py")
            self.model = None
            self.vectorizer = None
        else:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)

        # Карта для красивых названий классов
        self.class_names = {
            'performance': '⚙️ Производительность',
            'gameplay': '🎮 Геймплей',
            'story': '📖 Сюжет',
            'graphics': '🎨 Графика'
        }

    def _preprocess(self, text):
        """Предобработка текста (должна совпадать с обучением)"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zа-яё\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def predict(self, text):
        """Предсказывает тип критики для одного отзыва"""
        if self.model is None:
            return None
        cleaned = self._preprocess(text)
        X = self.vectorizer.transform([cleaned])
        prediction = self.model.predict(X)[0]
        return prediction

    def predict_with_confidence(self, text):
        """Предсказывает тип критики с уверенностью"""
        if self.model is None:
            return None, 0.0
        cleaned = self._preprocess(text)
        X = self.vectorizer.transform([cleaned])
        prediction = self.model.predict(X)[0]
        probs = self.model.predict_proba(X)[0]
        confidence = max(probs)
        return prediction, confidence

    def predict_batch(self, texts):
        """Предсказывает для нескольких отзывов"""
        if self.model is None:
            return [None] * len(texts)
        cleaned = [self._preprocess(t) for t in texts]
        X = self.vectorizer.transform(cleaned)
        predictions = self.model.predict(X)
        return predictions

    def get_class_name(self, class_id):
        """Возвращает красивое название класса"""
        return self.class_names.get(class_id, class_id)

    def is_ready(self):
        """Проверяет, загружена ли модель"""
        return self.model is not None and self.vectorizer is not None


# Пример использования
if __name__ == "__main__":
    # Загружаем классификатор
    classifier = ReviewClassifier()

    if not classifier.is_ready():
        print("Модель не загружена. Запустите train_classifier.py для обучения.")
        exit(1)

    # Тестовые отзывы
    test_reviews = [
        "В игре ужасная оптимизация, постоянные вылеты и низкий FPS",
        "Сюжет просто потрясающий, очень трогательная история",
        "Графика шикарная, очень красивые пейзажи",
        "Боевая система скучная и однообразная",
        "Лучшая игра в которую я играл, всем рекомендую"
    ]

    print("=" * 60)
    print("КЛАССИФИКАЦИЯ ОТЗЫВОВ")
    print("=" * 60)

    for review in test_reviews:
        pred, conf = classifier.predict_with_confidence(review)
        class_name = classifier.get_class_name(pred)
        print(f"\nОтзыв: {review}")
        print(f"Тип критики: {class_name} (уверенность: {conf:.1%})")
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль разметки отзывов на видеоигры по типу критики.
Метод: правила + словари (rule-based labeling)

Обновления:
- Добавлены синонимы и общие слова для увеличения покрытия
- Добавлены стоп-слова для исключения ложных срабатываний
- Добавлены правила с низким приоритетом (общая похвала/критика)
"""

import pandas as pd
import re
from typing import List, Tuple, Optional


# ============================================================
# 1. БАЗОВЫЕ УСЛОВИЯ ДЛЯ ПРАВИЛ
# ============================================================

class Condition:
    def check(self, text: str) -> bool:
        raise NotImplementedError


class Contains(Condition):
    def __init__(self, word: str, ignore_case: bool = True):
        self.word = word.lower() if ignore_case else word
        self.ignore_case = ignore_case

    def check(self, text: str) -> bool:
        if self.ignore_case:
            text = text.lower()
        return self.word in text

    def __repr__(self):
        return f"contains('{self.word}')"


class ContainsAny(Condition):
    def __init__(self, words: List[str], ignore_case: bool = True):
        self.words = [w.lower() if ignore_case else w for w in words]
        self.ignore_case = ignore_case

    def check(self, text: str) -> bool:
        if self.ignore_case:
            text = text.lower()
        return any(word in text for word in self.words)

    def __repr__(self):
        return f"contains_any({self.words})"


class ContainsAll(Condition):
    def __init__(self, words: List[str], ignore_case: bool = True):
        self.words = [w.lower() if ignore_case else w for w in words]
        self.ignore_case = ignore_case

    def check(self, text: str) -> bool:
        if self.ignore_case:
            text = text.lower()
        return all(word in text for word in self.words)

    def __repr__(self):
        return f"contains_all({self.words})"


class NotContains(Condition):
    def __init__(self, word: str, ignore_case: bool = True):
        self.word = word.lower() if ignore_case else word
        self.ignore_case = ignore_case

    def check(self, text: str) -> bool:
        if self.ignore_case:
            text = text.lower()
        return self.word not in text

    def __repr__(self):
        return f"not_contains('{self.word}')"


class And(Condition):
    def __init__(self, *conditions: Condition):
        self.conditions = conditions

    def check(self, text: str) -> bool:
        return all(cond.check(text) for cond in self.conditions)

    def __repr__(self):
        return f"and({', '.join(repr(c) for c in self.conditions)})"


class Or(Condition):
    def __init__(self, *conditions: Condition):
        self.conditions = conditions

    def check(self, text: str) -> bool:
        return any(cond.check(text) for cond in self.conditions)

    def __repr__(self):
        return f"or({', '.join(repr(c) for c in self.conditions)})"


# ============================================================
# 2. ПРАВИЛА КЛАССИФИКАЦИИ (С ДОБАВЛЕННЫМИ СИНОНИМАМИ)
# ============================================================

class Rule:
    def __init__(self, condition: Condition, category: str, priority: int = 0, name: str = ""):
        self.condition = condition
        self.category = category
        self.priority = priority
        self.name = name

    def applies(self, text: str) -> bool:
        return self.condition.check(text)

    def __repr__(self):
        return f"Rule('{self.name}', {self.category}, priority={self.priority})"


# ============================================================
# 3. ОПРЕДЕЛЕНИЕ ПРАВИЛ (РАСШИРЕННЫЕ СЛОВАРИ + СТОП-СЛОВА)
# ============================================================

# СТОП-СЛОВА ДЛЯ ИСКЛЮЧЕНИЯ (контексты, где слово НЕ про аспект)
STOP_CONTEXTS = {
    'графика': ['настройк', 'видеокарт', 'драйвер', 'процессор'],  # техническая графика
    'баг': ['смешн', 'забавн', 'прикольн', 'юмор'],  # баги как фича
    'фпс': ['низк', 'средн', 'высок', 'ultra'],  # обсуждение настроек, не критика
}


def get_rules() -> List[Rule]:
    rules = []

    # ========================================================
    # ВЫСОКИЙ ПРИОРИТЕТ (точные технические термины)
    # ========================================================

    # --- ПРОИЗВОДИТЕЛЬНОСТЬ (приоритет 100-90) ---
    rules.append(Rule(
        ContainsAny(['фпс', 'fps', 'лага', 'тормоз', 'фриз', 'просадк', 'микрофриз']),
        'performance', 100, 'FPS/лаги'
    ))
    rules.append(Rule(
        ContainsAny(['оптимизац', 'оптимизации', 'требовани', 'системные', 'производительност']),
        'performance', 99, 'оптимизация'
    ))
    rules.append(Rule(
        And(ContainsAny(['баг', 'баги', 'багов', 'глюк', 'глюки']),
            NotContains('смешн'), NotContains('забавн'), NotContains('прикольн')),
        'performance', 98, 'баги/глюки (критические)'
    ))
    rules.append(Rule(
        ContainsAny(['вылета', 'краш', 'crash', 'падает', 'зависа', 'завис', 'ошибк', 'error']),
        'performance', 97, 'вылеты/краши/ошибки'
    ))
    rules.append(Rule(
        ContainsAny(['катсцен', 'долг', 'загрузк', 'лоад', 'скипн', 'пропуст']),
        'performance', 96, 'долгие загрузки'
    ))

    # --- ГЕЙМПЛЕЙ (приоритет 89-80) ---
    rules.append(Rule(
        ContainsAny(['геймплей', 'gameplay', 'геймплейная', 'геймплейные', 'геймплейно']),
        'gameplay', 89, 'прямое упоминание gameplay'
    ))
    rules.append(Rule(
        ContainsAny(['боевк', 'сражен', 'битв', 'драк', 'combat', 'бой', 'бить']),
        'gameplay', 88, 'боевая система'
    ))
    rules.append(Rule(
        ContainsAny(['квест', 'задани', 'мисси', 'миссии', 'контракт', 'заказ']),
        'gameplay', 87, 'квесты/задания'
    ))
    rules.append(Rule(
        ContainsAny(['механик', 'управлени', 'прокачк', 'скилл', 'баланс', 'инвентарь']),
        'gameplay', 86, 'механики/управление'
    ))
    rules.append(Rule(
        ContainsAny(['скучн', 'однообразн', 'надоед', 'монотонн']),
        'gameplay', 85, 'скучная критика'
    ))
    rules.append(Rule(
        ContainsAny(['затягива', 'увлека', 'залип', 'залипат', 'втягив']),
        'gameplay', 84, 'положительный отклик геймплей'
    ))

    # --- СЮЖЕТ (приоритет 79-70) ---
    rules.append(Rule(
        ContainsAny(['сюжет', 'story', 'plot', 'сюжета', 'сюжетная', 'сюжетные']),
        'story', 79, 'прямое упоминание сюжета'
    ))
    rules.append(Rule(
        ContainsAny(['история', 'повествовани', 'narration', 'рассказ']),
        'story', 78, 'история/повествование'
    ))
    rules.append(Rule(
        ContainsAny(['концовк', 'финал', 'эпилог', 'конец', 'развязк']),
        'story', 77, 'концовка'
    ))
    rules.append(Rule(
        ContainsAny(['диалог', 'персонаж', 'герои', 'character', 'герой', 'протагонист']),
        'story', 76, 'диалоги/персонажи'
    ))
    rules.append(Rule(
        ContainsAny(['трогательн', 'плакал', 'слез', 'душ', 'сопережив']),
        'story', 75, 'эмоциональная реакция на сюжет'
    ))
    rules.append(Rule(
        ContainsAny(['книг', 'сапковск', 'анджей', 'оригинал', 'первоисточник']),
        'story', 74, 'отсылка к книгам'
    ))

    # --- ГРАФИКА (приоритет 69-60) ---
    rules.append(Rule(
        ContainsAny(['график', 'графон', 'graphics', 'visual', 'визуал']),
        'graphics', 69, 'прямое упоминание графики'
    ))
    rules.append(Rule(
        ContainsAny(['текстур', 'texture', 'модел', 'анимац', 'шейдер']),
        'graphics', 68, 'текстуры/модели/анимации'
    ))
    rules.append(Rule(
        ContainsAny(['красив', 'пейзаж', 'локации', 'атмосфер', 'виды', 'красота']),
        'graphics', 67, 'визуальная эстетика'
    ))
    rules.append(Rule(
        ContainsAny(['устарел', 'мыл', 'картинк', 'мультяшн', 'стилистик']),
        'graphics', 66, 'критика графики'
    ))
    rules.append(Rule(
        ContainsAny(['свет', 'тень', 'освещен', 'детализац', 'реалистичн']),
        'graphics', 65, 'детали графики'
    ))

    # ========================================================
    # СРЕДНИЙ ПРИОРИТЕТ (комбинированные и уточняющие правила)
    # ========================================================

    rules.append(Rule(
        And(Contains('график'), Contains('хорош'), NotContains('настройк')),
        'graphics', 55, 'хорошая графика'
    ))
    rules.append(Rule(
        And(Contains('сюжет'), Contains('интересн')),
        'story', 54, 'интересный сюжет'
    ))
    rules.append(Rule(
        And(Contains('боевк'), Contains('скучн')),
        'gameplay', 53, 'скучная боевка'
    ))
    rules.append(Rule(
        And(ContainsAny(['оптимизац', 'фпс']), ContainsAny(['плох', 'ужасн', 'никак'])),
        'performance', 52, 'плохая оптимизация'
    ))

    # ========================================================
    # НИЗКИЙ ПРИОРИТЕТ (общие слова — "последняя надежда")
    # Общие положительные отзывы → gameplay
    # Общие критические отзывы → performance
    # Общие восторженные → story
    # ========================================================

    # Общая похвала (без конкретики) — чаще относится к геймплею
    rules.append(Rule(
        ContainsAny(['классн', 'отличн', 'хорош', 'супер', 'крут', 'топ',
                     'заебись', 'афигенно', 'бомба', 'пушка', 'шикарн',
                     'великолепн', 'прекрасн', 'замечательн', 'прикольн']),
        'gameplay', 30, 'общая положительная оценка'
    ))

    # Общая критика (без конкретики) — чаще относится к производительности/багам
    rules.append(Rule(
        ContainsAny(['говно', 'мусор', 'ужасн', 'отстой', 'херня', 'разочарован',
                     'кал', 'дерьмо', 'шлак', 'позор', 'уныл', 'скучн']),
        'performance', 29, 'общая негативная оценка'
    ))

    # Эмоциональные отзывы без конкретики → story (чаще связаны с сюжетом)
    rules.append(Rule(
        ContainsAny(['шедевр', 'гениально', 'эмоции', 'восторг', 'пережива',
                     'любов', 'обож', 'душевн', 'впечатлил', 'потряс']),
        'story', 28, 'эмоциональная оценка (шедевр)'
    ))

    # Рекомендация (без конкретики) → gameplay
    rules.append(Rule(
        ContainsAny(['рекоменд', 'совету', 'must have', 'обязателен']),
        'gameplay', 27, 'рекомендация'
    ))

    # Игра как целое (общие фразы) → gameplay
    rules.append(Rule(
        ContainsAny(['игра супер', 'игра класс', 'игра отличн', 'игра топ', 'игра бомба']),
        'gameplay', 26, 'оценка игры в целом'
    ))

    # ========================================================
    # СПЕЦИАЛЬНЫЕ ПРАВИЛА С ИСКЛЮЧЕНИЯМИ (стоп-слова)
    # ========================================================

    # Исключить случаи, когда слово "графика" относится к настройкам или железу
    rules.append(Rule(
        And(Contains('график'),
            NotContains('настройк'), NotContains('видеокарт'),
            NotContains('драйвер'), NotContains('процессор')),
        'graphics', 50, 'графика (не техническая)'
    ))

    # Сортировка по приоритету (от большего к меньшему)
    rules.sort(key=lambda r: r.priority, reverse=True)
    return rules


# ============================================================
# 4. КЛАССИФИКАТОР (ОБНОВЛЁННЫЙ)
# ============================================================

class RuleBasedClassifier:
    def __init__(self, rules: List[Rule] = None):
        self.rules = rules if rules is not None else get_rules()

    def classify(self, text: str) -> Optional[str]:
        if not text or not isinstance(text, str):
            return None
        text = self._preprocess(text)
        for rule in self.rules:
            if rule.applies(text):
                return rule.category
        return None

    def classify_with_rule(self, text: str) -> Tuple[Optional[str], Optional[Rule]]:
        if not text or not isinstance(text, str):
            return None, None
        text = self._preprocess(text)
        for rule in self.rules:
            if rule.applies(text):
                return rule.category, rule
        return None, None

    def _preprocess(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^a-zа-яё0-9\s]', ' ', text)  # удаляем пунктуацию
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def print_rules(self):
        print("ПРАВИЛА КЛАССИФИКАЦИИ:")
        print("-" * 65)
        print(f"{'#':3} | {'Категория':12} | {'Приор':5} | {'Название правила'}")
        print("-" * 65)
        for i, rule in enumerate(self.rules, 1):
            print(f"{i:3} | {rule.category:12} | {rule.priority:5} | {rule.name}")
        print("-" * 65)


# ============================================================
# 5. РАЗМЕТКА ДАТАСЕТА
# ============================================================

def label_reviews(df: pd.DataFrame,
                  text_column: str = 'review',
                  classifier: RuleBasedClassifier = None,
                  inplace: bool = False) -> Tuple[pd.DataFrame, RuleBasedClassifier]:
    if classifier is None:
        classifier = RuleBasedClassifier()

    df_copy = df if inplace else df.copy()

    df_copy['label'] = df_copy[text_column].apply(classifier.classify)

    # Сохраняем название сработавшего правила для отладки
    def get_rule_name(text):
        _, rule = classifier.classify_with_rule(text)
        return rule.name if rule else None

    df_copy['rule_name'] = df_copy[text_column].apply(get_rule_name)

    return df_copy, classifier


def analyze_labels(df: pd.DataFrame, label_column: str = 'label', num_examples: int = 3):
    """Выводит статистику по разметке и примеры для каждого аспекта"""
    labeled = df[df[label_column].notna()]
    unlabeled = df[df[label_column].isna()]

    print("=" * 65)
    print("СТАТИСТИКА РАЗМЕТКИ")
    print("=" * 65)
    print(f"Всего отзывов:          {len(df)}")
    print(f"Размечено:              {len(labeled)} ({len(labeled) / len(df) * 100:.1f}%)")
    print(f"Не размечено:           {len(unlabeled)} ({len(unlabeled) / len(df) * 100:.1f}%)")
    print()
    print("Распределение по аспектам:")

    categories = ['performance', 'gameplay', 'story', 'graphics']
    category_names = {
        'performance': '⚙️ Производительность',
        'gameplay': '🎮 Геймплей',
        'story': '📖 Сюжет',
        'graphics': '🎨 Графика'
    }

    for cat in categories:
        count = len(labeled[labeled[label_column] == cat])
        if len(labeled) > 0:
            pct = count / len(labeled) * 100
            bar = '█' * int(pct / 2) + '░' * (50 - int(pct / 2))
            print(f"  {cat:12} : {count:5} ({pct:5.1f}%) {bar}")
    print("=" * 65)

    # Статистика по сработавшим правилам
    if 'rule_name' in df.columns:
        print("\n📊 СРАБОТАВШИЕ ПРАВИЛА (топ-15):")
        rule_stats = df[df['rule_name'].notna()]['rule_name'].value_counts().head(15)
        for rule_name, count in rule_stats.items():
            print(f"  {rule_name:40} : {count:5}")

    # ============================================================
    # НОВОЕ: ПРИМЕРЫ РАЗМЕЧЕННЫХ ОТЗЫВОВ ПО КАЖДОМУ АСПЕКТУ
    # ============================================================
    print("\n" + "=" * 65)
    print("📝 ПРИМЕРЫ РАЗМЕЧЕННЫХ ОТЗЫВОВ")
    print("=" * 65)

    for cat in categories:
        cat_label = category_names.get(cat, cat)
        cat_samples = labeled[labeled[label_column] == cat].head(num_examples)

        print(f"\n[{cat_label}] ({len(labeled[labeled[label_column] == cat])} отзывов)")
        print("-" * 50)

        if len(cat_samples) == 0:
            print("  (нет примеров)")
        else:
            for i, (idx, row) in enumerate(cat_samples.iterrows(), 1):
                text = row['review']
                # Обрезаем длинные отзывы
                preview = text[:200] + "..." if len(text) > 200 else text
                # Убираем лишние переносы строк
                preview = preview.replace('\n', ' ')
                # Получаем название сработавшего правила
                rule_name = row.get('rule_name', 'неизвестно')
                print(f"  {i}. [{rule_name[:35]}] {preview}")

    return labeled, unlabeled


def save_labeled_data(df: pd.DataFrame, output_path: str = 'labeled_reviews.csv'):
    """Сохраняет размеченный датасет"""
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nРазмеченные данные сохранены в: {output_path}")


# ============================================================
# 6. ЗАПУСК
# ============================================================

if __name__ == "__main__":
    INPUT_FILE = 'Data/cleaned_reviews.csv'
    OUTPUT_FILE = 'Data/labeled_reviews.csv'

    # Сколько примеров показывать для каждого аспекта (можно изменить)
    EXAMPLES_PER_CATEGORY = 5

    print("Загрузка данных...")
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"Загружено {len(df)} записей")
    except FileNotFoundError:
        print(f"Файл {INPUT_FILE} не найден!")
        print("Укажите правильный путь к файлу.")
        exit(1)

    if 'review' not in df.columns:
        print(f"Колонка 'review' не найдена. Доступны: {list(df.columns)}")
        exit(1)

    # Создаём классификатор и выводим правила
    classifier = RuleBasedClassifier()
    classifier.print_rules()

    # Размечаем
    print("\nЗапуск разметки...")
    df_labeled, classifier = label_reviews(df, classifier=classifier)

    # Анализируем (с примерами)
    analyze_labels(df_labeled, num_examples=EXAMPLES_PER_CATEGORY)

    # Сохраняем
    save_labeled_data(df_labeled, OUTPUT_FILE)

    # Показываем неразмеченные отзывы для анализа
    unlabeled = df_labeled[df_labeled['label'].isna()]
    if len(unlabeled) > 0:
        print("\n" + "=" * 65)
        print("❓ ПРИМЕРЫ НЕРАЗМЕЧЕННЫХ ОТЗЫВОВ (возможно, нужны доп. правила):")
        print("=" * 65)
        for i, row in enumerate(unlabeled.head(10).iterrows(), 1):
            text = row[1]['review']
            preview = text[:150] + "..." if len(text) > 150 else text
            preview = preview.replace('\n', ' ')
            print(f"{i}. {preview}\n")
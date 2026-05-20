import pandas as pd
import re
import sys


# ============================================================
# 1. ОЧИСТКА ТЕКСТА
# ============================================================

def clean_review_text(text):
    """
    Очистка текста отзыва:
    - Приведение к нижнему регистру
    - Удаление URL, ссылок
    - Удаление спецсимволов (оставляем буквы, цифры, пробелы, базовые знаки)
    - Удаление лишних пробелов
    - Удаление слишком коротких слов (менее 2 символов)
    - Опционально: удаление стоп-слов (пока оставляем, можно добавить позже)
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""

    # Приводим к нижнему регистру
    text = text.lower()

    # Удаляем URL и ссылки
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'steamcommunity\.com/\S+', '', text)
    text = re.sub(r'youtu\.be/\S+', '', text)
    text = re.sub(r'youtube\.com/\S+', '', text)
    text = re.sub(r'https?://youtube\.be/\S+', '', text)

    # Удаляем HTML-теги и BB-коды (например, [h1], [/h1], [spoiler])
    text = re.sub(r'\[.*?\]', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)

    # Удаляем эмодзи и спецсимволы (оставляем буквы, цифры, пробелы, .!?,-)
    text = re.sub(r'[^a-zа-яё0-9\s\.\!\?\,\-]', ' ', text)

    # Удаляем повторяющиеся буквы (например, "ааааа" -> "а")
    text = re.sub(r'(.)\1{3,}', r'\1', text)

    # Удаляем одиночные символы и слишком короткие слова (менее 2 букв)
    words = text.split()
    words = [w for w in words if len(w) >= 2 and not w.isdigit()]
    text = ' '.join(words)

    # Удаляем лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def is_russian(text):
    """
    Проверяет, содержит ли текст русские буквы.
    Возвращает True, если в тексте есть хотя бы одна русская буква.
    """
    if not text:
        return False
    return bool(re.search(r'[а-яё]', text.lower()))


def is_too_short(text, min_length=10):
    """
    Проверяет, не слишком ли короткий отзыв.
    """
    if not text:
        return True
    return len(text) < min_length


# ============================================================
# 2. ОСНОВНАЯ ФУНКЦИЯ ПРЕДОБРАБОТКИ
# ============================================================

def preprocess_dataset(input_file,
                       output_file='cleaned_reviews.csv',
                       min_text_length=10,
                       keep_only_russian=True,
                       remove_duplicates=True):
    """
    Полная предобработка датасета.

    Аргументы:
        input_file: путь к исходному CSV
        output_file: путь для сохранения очищенного CSV
        min_text_length: минимальная длина текста отзыва
        keep_only_russian: оставлять только русские отзывы
        remove_duplicates: удалять дубликаты отзывов

    Возвращает:
        DataFrame с очищенными данными
    """
    print("=" * 60)
    print("ПРЕДОБРАБОТКА ДАТАСЕТА")
    print("=" * 60)

    # Шаг 1: Загрузка данных
    print("\n[1/6] Загрузка данных...")
    try:
        df = pd.read_csv(input_file)
        print(f"      Загружено записей: {len(df)}")
    except FileNotFoundError:
        print(f"      ОШИБКА: Файл {input_file} не найден!")
        sys.exit(1)
    except Exception as e:
        print(f"      ОШИБКА: {e}")
        sys.exit(1)

    # Шаг 2: Проверка наличия колонки 'review'
    print("\n[2/6] Проверка структуры данных...")
    if 'review' not in df.columns:
        print(f"      ОШИБКА: Колонка 'review' не найдена!")
        print(f"      Доступные колонки: {list(df.columns)}")
        sys.exit(1)
    print(f"      Колонка 'review' найдена")

    # Шаг 3: Очистка текста
    print("\n[3/6] Очистка текста отзывов...")
    df['review_cleaned'] = df['review'].apply(clean_review_text)

    # Подсчёт результатов очистки
    empty_before = (df['review'].isna() | (df['review'] == '')).sum()
    empty_after = (df['review_cleaned'] == '').sum()
    print(f"      Пустых отзывов до очистки: {empty_before}")
    print(f"      Пустых отзывов после очистки: {empty_after}")

    # Шаг 5: Фильтрация по длине
    print("\n[5/6] Фильтрация: удаление слишком коротких отзывов...")
    df['is_long_enough'] = df['review_cleaned'].apply(
        lambda x: not is_too_short(x, min_text_length)
    )
    long_count = df['is_long_enough'].sum()
    print(f"      Отзывов длиннее {min_text_length} символов: {long_count} ({long_count / len(df) * 100:.1f}%)")
    df = df[df['is_long_enough']].copy()

    # Шаг 6: Удаление дубликатов
    if remove_duplicates:
        print("\n[6/6] Удаление дубликатов...")
        before = len(df)
        df = df.drop_duplicates(subset=['review_cleaned'], keep='first')
        after = len(df)
        print(f"      Удалено дубликатов: {before - after}")

    # Итоговая статистика
    print("\n" + "=" * 60)
    print("ИТОГИ ПРЕДОБРАБОТКИ")
    print("=" * 60)
    print(f"Исходное количество записей:      {original_count}")
    print(f"После очистки и фильтрации:       {len(df)}")
    print(f"Сохранение в файл:                {output_file}")

    # Сохранение результата
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print("\nГотово!")

    return df


# ============================================================
# 3. АНАЛИЗ КАЧЕСТВА ОЧИСТКИ (ОПЦИОНАЛЬНО)
# ============================================================

def show_sample(df, n=5):
    """
    Показывает примеры очищенных отзывов.
    """
    print("\n" + "=" * 60)
    print(f"ПРИМЕРЫ ОЧИЩЕННЫХ ОТЗЫВОВ (первые {n})")
    print("=" * 60)

    for i, row in df.head(n).iterrows():
        original = row['review'][:100] + "..." if len(str(row['review'])) > 100 else row['review']
        cleaned = row['review_cleaned'][:100] + "..." if len(row['review_cleaned']) > 100 else row['review_cleaned']
        print(f"\n[{i}]")
        print(f"  Было:  {original}")
        print(f"  Стало: {cleaned}")


# ============================================================
# 4. ЗАПУСК
# ============================================================


INPUT_FILE = 'Data/steam_reviews_russian_only.csv'  # исходный файл
OUTPUT_FILE = 'Data/cleaned_reviews.csv'  # результат предобработки

# Запоминаем исходное количество (для статистики)
try:
    temp_df = pd.read_csv(INPUT_FILE)
    original_count = len(temp_df)
except:
    original_count = 0

# Запуск предобработки
df_cleaned = preprocess_dataset(
    input_file=INPUT_FILE,
    output_file=OUTPUT_FILE,
    min_text_length=10,  # минимальная длина текста
)

# Показываем примеры
show_sample(df_cleaned, n=5)

# Дополнительная информация
print("\n" + "=" * 60)
print("СТРУКТУРА ОЧИЩЕННОГО ДАТАСЕТА")
print("=" * 60)
print(f"Колонки: {list(df_cleaned.columns)}")
print(f"Размер: {df_cleaned.shape}")
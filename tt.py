import pandas as pd

# Параметры
chunk_size = 50000  # строк в одном чанке
target_true = 5000
target_false = 5000

# Контейнеры для накопления нужных записей
true_samples = []
false_samples = []

# Читаем файл частями
for chunk in pd.read_csv('Data/Raw/steam_reviews_full.csv',
                         chunksize=chunk_size,
                         low_memory=False):
    # Оставляем только русские отзывы
    chunk_russian = chunk[chunk['language'] == 'russian']

    # Разделяем на рекомендованные и нет
    true_chunk = chunk_russian[chunk_russian['recommended'] == True]
    false_chunk = chunk_russian[chunk_russian['recommended'] == False]

    # Добавляем в накопители
    true_samples.append(true_chunk)
    false_samples.append(false_chunk)

    # Объединяем накопленные данные
    true_so_far = pd.concat(true_samples, ignore_index=True)
    false_so_far = pd.concat(false_samples, ignore_index=True)

    # Если набрали достаточно, останавливаемся
    if len(true_so_far) >= target_true and len(false_so_far) >= target_false:
        print(f"Достигнуто нужное количество: true={len(true_so_far)}, false={len(false_so_far)}")
        break

# Теперь у нас есть true_so_far и false_so_far (возможно, больше целевого)
# Берём ровно по target_true и target_false (или меньше, если не хватило)
true_sample = true_so_far.sample(n=min(target_true, len(true_so_far)), random_state=42)
false_sample = false_so_far.sample(n=min(target_false, len(false_so_far)), random_state=42)

# Объединяем и перемешиваем
sampled_df = pd.concat([true_sample, false_sample], ignore_index=True)
sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Сохраняем
sampled_df.to_csv('Data/steam_reviews_russian_balanced.csv', index=False)

print(f"Итоговый размер выборки: {len(sampled_df)}")
print(
    f"Рекомендовано: {sampled_df['recommended'].sum()}, не рекомендовано: {len(sampled_df) - sampled_df['recommended'].sum()}")
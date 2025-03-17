import os
import shutil
import pandas as pd

# Исходные папки
source_dirs = ["data/train", "data/test"]
target_dir = "data/all_data"

# Создаем главную папку
os.makedirs(target_dir, exist_ok=True)

# Список для хранения информации о метках
data_list = []

# Проходим по всем исходным директориям (train и test)
for source_dir in source_dirs:
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)

        # Пропускаем, если не папка
        if not os.path.isdir(class_path):
            continue

        # Создаем такую же папку в `all_data/`
        class_target_dir = os.path.join(target_dir, class_name)
        os.makedirs(class_target_dir, exist_ok=True)

        # Проходим по файлам в этой категории
        for file_name in os.listdir(class_path):
            old_path = os.path.join(class_path, file_name)
            new_path = os.path.join(class_target_dir, file_name)

            # Копируем файл
            shutil.copy2(old_path, new_path)

            # Добавляем запись в список
            data_list.append([class_name, file_name])

# Создаем DataFrame и сохраняем метки в CSV
df = pd.DataFrame(data_list, columns=["class", "filename"])
df.to_csv(os.path.join(target_dir, "labels.csv"), index=False)

for e in source_dirs:
    shutil.rmtree(e)  # Удаляем исходные папки train и test

print(f"✅ Все файлы объединены в {target_dir} с сохранением структуры!")

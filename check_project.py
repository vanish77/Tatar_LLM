# -*- coding: utf-8 -*-
"""
Скрипт для проверки готовности проекта к сдаче
Проверяет наличие всех необходимых файлов и компонентов
"""

import os
from pathlib import Path


def check_file(filepath, critical=True):
    """Проверяет наличие файла"""
    exists = os.path.exists(filepath)
    status = "?" if exists else ("?" if critical else "??")
    filename = Path(filepath).name
    print(f"{status} {filename}")
    return exists


def check_directory(dirpath, critical=True):
    """Проверяет наличие директории"""
    exists = os.path.exists(dirpath) and os.path.isdir(dirpath)
    status = "?" if exists else ("?" if critical else "??")
    dirname = Path(dirpath).name
    print(f"{status} {dirname}/")
    return exists


def check_project():
    """Проверяет готовность проекта"""
    
    print("="*60)
    print("ПРОВЕРКА ГОТОВНОСТИ ПРОЕКТА")
    print("="*60)
    print()
    
    all_good = True
    
    # Основные скрипты
    print("?? Основные скрипты:")
    all_good &= check_file("01_prepare_data.py")
    all_good &= check_file("02_train_tokenizer.py")
    all_good &= check_file("03_train.py")
    all_good &= check_file("model.py")
    all_good &= check_file("config.py")
    all_good &= check_file("utils.py")
    print()
    
    # Демо и тестирование
    print("?? Демо и тестирование:")
    all_good &= check_file("demo_inference.ipynb")
    all_good &= check_file("quick_test.py")
    check_file("app.py", critical=False)  # Опционально
    print()
    
    # Документация
    print("?? Документация:")
    all_good &= check_file("README.md")
    check_file("QUICKSTART.md", critical=False)
    check_file("ARCHITECTURE.md", critical=False)
    check_file("EXAMPLES.md", critical=False)
    check_file("SUBMISSION.md", critical=False)
    print()
    
    # Конфигурация
    print("?? Конфигурация:")
    all_good &= check_file("requirements.txt")
    all_good &= check_file(".gitignore")
    check_file("run_pipeline.sh", critical=False)
    print()
    
    # Данные (создаются при запуске)
    print("?? Генерируемые данные:")
    has_data = check_directory("data/processed", critical=False)
    has_tokenizer = check_directory("tokenizer", critical=False)
    has_model = check_directory("models", critical=False)
    
    if has_data:
        check_file("data/processed/train.txt", critical=False)
        check_file("data/processed/val.txt", critical=False)
    
    if has_tokenizer:
        check_file("tokenizer/tokenizer.json", critical=False)
        check_file("tokenizer/config.json", critical=False)
    
    if has_model:
        check_file("models/best_model.pt", critical=False)
        check_file("models/final_model.pt", critical=False)
    
    print()
    
    # Итоги
    print("="*60)
    print("ИТОГИ ПРОВЕРКИ")
    print("="*60)
    print()
    
    if all_good:
        print("? Все критичные файлы на месте!")
        print()
    else:
        print("? Отсутствуют некоторые критичные файлы!")
        print("   Проверьте отмеченные ? файлы выше.")
        print()
    
    # Проверка обученной модели
    if has_model and os.path.exists("models/best_model.pt"):
        print("? Модель обучена и сохранена")
        
        # Проверяем размер модели
        model_size = os.path.getsize("models/best_model.pt") / (1024 * 1024)
        print(f"   Размер модели: {model_size:.1f} МБ")
        
        if model_size < 10:
            print("   ??  Модель кажется слишком маленькой. Убедитесь что обучение завершилось.")
        
        print()
    else:
        print("??  Модель ещё не обучена")
        print("   Запустите: python 03_train.py")
        print()
    
    # Проверка токенизатора
    if has_tokenizer and os.path.exists("tokenizer/tokenizer.json"):
        print("? Токенизатор обучен")
        print()
    else:
        print("??  Токенизатор ещё не обучен")
        print("   Запустите: python 02_train_tokenizer.py")
        print()
    
    # Проверка данных
    if has_data and os.path.exists("data/processed/train.txt"):
        print("? Данные подготовлены")
        
        # Проверяем размер данных
        data_size = os.path.getsize("data/processed/train.txt") / (1024 * 1024)
        print(f"   Размер train.txt: {data_size:.1f} МБ")
        
        if data_size < 1:
            print("   ??  Данных кажется мало. Рекомендуется хотя бы 10-100 МБ.")
        
        print()
    else:
        print("??  Данные ещё не подготовлены")
        print("   Запустите: python 01_prepare_data.py")
        print()
    
    # Чеклист для сдачи
    print("="*60)
    print("ЧЕКЛИСТ ДЛЯ СДАЧИ")
    print("="*60)
    print()
    
    checklist = [
        ("Код загружен в репозиторий (GitHub)", False),
        ("README.md заполнен", os.path.exists("README.md")),
        ("Модель обучена", has_model and os.path.exists("models/best_model.pt")),
        ("Модель загружена на HF/GDrive/GH Release", False),
        ("demo_inference.ipynb работает", os.path.exists("demo_inference.ipynb")),
        ("Есть примеры с разными формулировками", os.path.exists("EXAMPLES.md")),
        ("requirements.txt актуален", os.path.exists("requirements.txt")),
        (".gitignore настроен", os.path.exists(".gitignore")),
    ]
    
    for task, done in checklist:
        status = "?" if done else "?"
        print(f"{status} {task}")
    
    print()
    
    # Следующие шаги
    print("="*60)
    print("СЛЕДУЮЩИЕ ШАГИ")
    print("="*60)
    print()
    
    steps_needed = []
    
    if not (has_data and os.path.exists("data/processed/train.txt")):
        steps_needed.append("1. Подготовить данные: python 01_prepare_data.py")
    
    if not (has_tokenizer and os.path.exists("tokenizer/tokenizer.json")):
        steps_needed.append("2. Обучить токенизатор: python 02_train_tokenizer.py")
    
    if not (has_model and os.path.exists("models/best_model.pt")):
        steps_needed.append("3. Обучить модель: python 03_train.py")
    
    if has_model and os.path.exists("models/best_model.pt"):
        steps_needed.append("4. Протестировать модель: python quick_test.py")
        steps_needed.append("5. Открыть demo_inference.ipynb и запустить")
        steps_needed.append("6. Загрузить модель на Hugging Face / Google Drive")
        steps_needed.append("7. Создать GitHub репозиторий и загрузить код")
        steps_needed.append("8. Обновить README.md со ссылкой на модель")
        steps_needed.append("9. Сдать преподавателю!")
    
    if steps_needed:
        for step in steps_needed:
            print(f"? {step}")
    else:
        print("?? Всё готово к сдаче!")
        print()
        print("Финальные шаги:")
        print("1. Загрузите модель на HF/GDrive/GH Release")
        print("2. Создайте GitHub репозиторий")
        print("3. Обновите README со ссылкой на модель")
        print("4. Сдайте преподавателю!")
    
    print()
    print("="*60)
    
    return all_good


if __name__ == "__main__":
    check_project()


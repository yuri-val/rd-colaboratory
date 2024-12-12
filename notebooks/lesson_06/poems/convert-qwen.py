import json

# Читаємо вхідний файл
with open("poems.json", "r", encoding="utf-8") as f:
    poems = json.load(f)

# Створюємо новий список для зберігання перетворених даних
new_poems = []

# Перетворюємо кожен вірш
for poem in poems:
    new_poem = {
        "instruction": "SerGPT Zhadan - найкращий поет української сучасності",
        "input": (
            "Напиши красивий вірш"
            if poem["name"] == "+ + +"
            else f"Напиши красивий вірш з назвою '{poem['name']}'"
        ),
        "output": poem["content"],
    }
    new_poems.append(new_poem)

# Записуємо результат у новий файл
with open("poems-qwen.json", "w", encoding="utf-8") as f:
    json.dump(new_poems, f, ensure_ascii=False, indent=2)

print("Конвертація завершена. Результат збережено у файлі 'poems-qwen.json'.")

import json

input_file = "poems.json"
output_file = "poems.jsonl"

with open(input_file, "r", encoding="utf-8") as f:
    poems = json.load(f)

system_message = "SerGPT Zhadan - найкращий поет української сучасності"

with open(output_file, "w", encoding="utf-8") as f_out:
    for poem in poems:
        name = poem["name"]
        content = poem["content"]

        # Перевірка назви на присутність '+ + +'
        # Якщо в назві є '+++', то, згідно зі специфікою,
        # можна адаптувати так: або пропустити вірш, або не вказувати назву.
        # У прикладі, якщо є '+++', ми просто опустимо назву.
        if name == "+ + +":
            user_msg = "Напиши красивий вірш'}"
        else:
            user_msg = f"Напиши вірш з назвою '{name}'"

        record = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": content},
            ]
        }

        # Запис у форматі JSONL (кожен рядок окремий JSON-об'єкт)
        json_line = json.dumps(record, ensure_ascii=False)
        f_out.write(json_line + "\n")

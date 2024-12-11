import json

input_file = "poems.txt"
output_file = "poems.json"

with open(input_file, "r", encoding="utf-8") as f:
    lines = [line.strip("\n") for line in f]

poems = []
current_name = None
current_content = []

# Структура:
# -----
# ---
# <name>
# ---
# <content>
# -----

# Логіка така: коли зустрічаємо "-----" - це розділювач між віршами.
# Між ними є також "---" перед назвою та після назви.
# Після другої "---" починається текст вірша до наступних "-----".
# Наприклад:
#
# -----
# ---
# назва
# ---
# вірш
# вірш
# -----
#
# Повторюється

# Видалимо можливі початкові або зайві роздільники на початку файла
while lines and lines[0].strip() == "-----":
    lines.pop(0)

# Тепер ітеруємося по рядках
i = 0
while i < len(lines):
    line = lines[i].strip()
    if line == "---":
        # наступний рядок - це назва
        i += 1
        current_name = lines[i].strip()
        # наступний рядок знову буде "---"
        i += 2
        # Тепер читаємо вірш до наступного "-----"
        current_content = []
        while i < len(lines) and lines[i].strip() != "-----":
            current_content.append(lines[i])
            i += 1
        # Коли натрапили на "-----" або кінець файлу, завершуємо поточний вірш
        poems.append(
            {"name": current_name, "content": "\n".join(current_content).strip()}
        )
    else:
        i += 1

# Записуємо у файл JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(poems, f, ensure_ascii=False, indent=2)

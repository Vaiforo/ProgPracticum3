import random
import nltk
from nltk.corpus import words

# Загружаем список английских слов из NLTK
nltk.download("words")
word_list = [word.lower() for word in words.words() if 3 <= len(word) <= 10]

# Генерируем 100000 случайных слов (с возможными повторениями)
generated_words = random.choices(word_list, k=10_000_000)

# Сохраняем в файл
with open("../source/bigsource/english_words_10kk.txt", "w") as f:
    f.write("\n".join(generated_words))

print("Файл сгенерирован!")

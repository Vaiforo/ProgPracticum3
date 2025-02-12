import random
import nltk
from nltk.corpus import words

# Загружаем список английских слов из NLTK
nltk.download("words")
word_list = [word.lower() for word in words.words() if 3 <= len(word) <= 10]

# Генерируем 100000 случайных слов (с возможными повторениями)
generated_words = random.choices(word_list, k=1_000)

# Сохраняем в файл
with open("../source/english_words_1k.txt", "w") as f:
    f.write("\n".join(generated_words))

print("Файл english_words_100k.txt сгенерирован!")

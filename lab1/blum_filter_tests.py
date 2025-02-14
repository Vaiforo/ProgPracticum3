from lab1.blum_filter import BlumFilter

blum = BlumFilter(1000)

with open('../source/english_words_1k.txt', 'r', encoding='utf-8') as f:
    for word in f:
        word = word.strip()
        blum.add(word)

print(blum)

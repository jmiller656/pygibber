import random
import string

import keras
from nltk.corpus import words as words
from nltk.corpus import names as names
import numpy as np

class Generator(keras.utils.Sequence):

    def __init__(self, num_examples=500, batch_size=32):
        self.examples = num_examples
        self.max_length = 30
        self.batch_size = batch_size
        self.int_to_char_map = {ind:val for ind,val in enumerate(string.ascii_lowercase)}
        self.char_to_int_map = {val:ind for ind,val in enumerate(string.ascii_lowercase)}
        self.word_set = set(words.words()).union(set(names.words()))
        self.word_list = list(self.word_set)

    def get_word(self):
        word = self.word_list[random.randint(0,len(self.word_list)-1)]
        rem_chars = [x for x in self.split(word) if x not in string.ascii_lowercase]
        for char in rem_chars:
            word = word.replace(char, "")
        return word


    def get_gibberish(self):
        len = random.randint(0, self.max_length)
        tmp = ""
        valid = False
        while not valid:
            for i in range(len):
                tmp += random.choice(string.ascii_lowercase)
            valid = not tmp in self.word_set

        return tmp

    def split(self, word):
        return [char for char in word]

    def __len__(self):
        return self.examples * 2

    def map_to_chars(self, text):
        return list(map(lambda x: self.char_to_int_map[x], self.split(text)))

    def postprocess(self, ex):
        if len(ex) < self.max_length:
            extra = [26 for x in range(self.max_length - len(ex))]
            ex += extra
        tmp = np.zeros((30, len(self.char_to_int_map)+1))
        for i in range(len(ex)):
            tmp[i][ex[i]] = 1
        return tmp

    def __getitem__(self, index):
        batch = []
        labels = []

        for i in range(self.batch_size // 2):
            batch.append(self.get_word())
            labels.append(0)
            batch.append(self.get_gibberish())
            labels.append(1)

        batch = list(map(lambda x: self.postprocess(self.map_to_chars(x)), batch))
        tmp = list(zip(batch, labels))
        random.shuffle(tmp)
        batch, labels = zip(*tmp)

        return np.asarray(batch), np.asarray(labels)

from bitarray import bitarray


# djb2 Хэш-функция
def hashfun(hash_input, shift):
    res = 5381

    res = ((res << 5) + res) + shift

    for ch in hash_input:
        res = ((res << 5) + res) + ord(ch)

    return res


class BloomFilter:
    def __init__(self, search_words, size, hash_count):
        self.bloomArray = bitarray('0' * size)
        self.hash_count = hash_count
        self.size = size

        for item in search_words:
            for i in range(self.hash_count):
                self.bloomArray[hashfun(item, i) % self.size] = 1

    def check(self, search_input):
        for shift in range(self.hash_count):
            if self.bloomArray[hashfun(search_input, shift) % self.size] != 1:
                return False
        return True

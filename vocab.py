

class Vocab:
    def __init__(self):
        self.w2i = {}
        self.i2w = []

    def add_token(self, token):
        if token not in self.w2i:
            self.w2i[token] = len(self.i2w)
            self.i2w.append(token)

    def get_id(self, token):
        if token in self.w2i:
            return self.w2i[token]
        return -1

    def get_token(self, id):
        return self.i2w[id]

    def size(self):
        return len(self.i2w)

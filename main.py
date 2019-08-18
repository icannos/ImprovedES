
class A:

    def __init__(self):
        self.v = self.a()

    def a(self):
        return 8

class B(A):
    def a(self):
        return 0

b = B()

print(b.v)
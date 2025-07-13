from functools import partial

class SomeClass:
    def foo(self, hi):
        print(hi)

A = SomeClass()
A.foo = partial(A.foo, hi = 'hello')
A.foo()

B = SomeClass()
B.foo('yuh')
from zhihu.hello.hello import add, foo

def test_hello():
    assert add(1, 2) == 3
    assert add("Hello ", "Python!") == "Hello Python!"
    assert type(foo()) == str
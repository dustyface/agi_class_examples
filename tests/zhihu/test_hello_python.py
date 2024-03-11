from zhihu.hello.hello_python import add

def test_hello_python():
    assert add(1, 2) == 3
    assert add("Hello ", "Python!") == "Hello Python!"
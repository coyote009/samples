def func(a, b, *args, **kwargs):
    print(f"a={a}")
    print(f"b={b}")
    print(f"args={args}")
    print(f"kwargs={kwargs}")
    print("")

func(0, 1, 2, 3, c=5)
""" Result
a=0
b=1
args=(2, 3)
kwargs={'c': 5}
"""
    
def func2(a, b, c=10, *args, **kwargs):
    print(f"a={a}")
    print(f"b={b}")
    print(f"c={c}")
    print(f"args={args}")
    print(f"kwargs={kwargs}")
    print("")

#func2(0, 1, 2, 3, c=5, d=20)
""" Result
TypeError: func2() got multiple values for argument 'c'
"""

func2(0, 1, d=20)
""" Result
a=0
b=1
c=10
args=()
kwargs={'d': 20}
"""

func2(0, 1, 2, 3, d=20)
""" Result
a=0
b=1
c=2
args=(3,)
kwargs={'d': 20}
"""

#func2(0, 1, 2, 3, d=20, a=100)
""" Result
TypeError: func() got multiple values for argument 'a'
"""

#func2(0, d=20)
""" Result
TypeError: func() missing 1 required positional argument: 'b'
"""

def func3(a, b, c=10, d=20, *args, **kwargs):
    print(f"a={a}")
    print(f"b={b}")
    print(f"c={c}")
    print(f"d={d}")
    print(f"args={args}")
    print(f"kwargs={kwargs}")
    print("")

func3(0, 1, d=100, e=200)
""" Result
a=0
b=1
c=10
d=100
args=()
kwargs={'e': 200}
"""

"""
定義してある変数は、変数名で参照できる
　→足りない場合はエラー
定義してない変数のうち、
　値で渡されたものはargsへ
　名前で渡されたものはkwargsへ
"""

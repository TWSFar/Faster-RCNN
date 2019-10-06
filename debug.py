class test():
    def __init__(self):
        self.a = 3
    
    def fun(self):
        self.a = 5
        return self

t = test()
t.fun()
pass
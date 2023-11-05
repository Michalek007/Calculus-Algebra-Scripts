class Complex:
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag

    def __eq__(self, other):
        return self.real == other.real and self.imag == other.imag

    def __add__(self, other):
        real = self.real + other.real
        imag = self.imag + other.imag
        return Complex(real, imag)

    def __sub__(self, other):
        real = self.real - other.real
        imag = self.imag - other.imag
        return Complex(real, imag)

    def __str__(self):
        if self.real != 0:
            self_str = f'{self.real}'
            if self.imag > 0:
                self_str += f' + {self.imag}i'
            elif self.imag < 0:
                self_str += f' - {-self.imag}i'
        else:
            if self.imag != 0:
                self_str = f'{self.imag}i'
            else:
                self_str = '0'
        return self_str

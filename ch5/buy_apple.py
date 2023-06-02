apple=100
apple_num=2
tax=1.1

import layers

#layer
mul_apple_layer=layer.Mullayer()
mul_tax_layer=layer.Mullayer()

#forward
apple_price=mul_apple_layer.forward(apple,apple_num)
price=mul_tax_layer.forward(apple_price,tax)

print(price,"price")

#backword
dprice=1
dapple_price,dtax=mul_tax_layer.backward(dprice)
dapple,dapple_num=mul_apple_layer.backward(dapple_price)
print(dapple,"    dapple\n")
print(dtax,"   dtax\n")
print(dapple_num,"  dapple_num")
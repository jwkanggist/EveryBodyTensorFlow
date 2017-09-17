#-*- coding: utf-8 -*-

"""
#---------------------------------------------
  filename: ex_runTFmatmul.py
  - Construct a computational graph which calculate
    a matrix multiplication in Tensorflow
  - Use tf.constant() in a matrix form
  Written by Jaewook Kang
  2017 Aug.
#-------------------------------------------
"""

import tensorflow as tf

# 계산 그래프 construction =====================
# 1x2 형렬을 만드는 Constant 연산을 생성합니다.
# 이 연산자는 기본 그래프의 노드로 추가됩니다.
#
# 생성자에 의해 반환된 값(matrix1)은 Constant 연산의 출력을 나타냅니다.
matrix1 = tf.constant([[3., 3.]])

# 2x1 행렬을 만드는 또 다른 Constant를 생성합니다.
matrix2 = tf.constant([[2.],[2.]])

# 'matrix1'과 'matrix2'를 입력으로 받는 Matmul 연산을 생성합니다.
# 반환값인 'product'는 행렬을 곱한 결과를 나타냅니다.
product = tf.matmul(matrix1, matrix2)



# 계산 그래프로 세션을 생성==================
sess = tf.Session()

# matmul 연산을 실행하려면 matmul 연산의 출력을 나타내는 'product'를
# 인자로 넘겨주면서 세션의 'run()' 메소드를 호출합니다. 이렇게 호출하면
# matmul 연산의 출력을 다시 얻고 싶다는 뜻입니다.

#
# 연산이 필요로 하는 모든 입력은 세션에 의해 자동으로 실행됩니다.
# 일반적으로 병렬적으로 실행되지요.
#
# 따라서 'run(product)' 호출은 그래프 내의 세 개 연산을 실행하게 됩니다.
# 두 개의 상수와 matmul이 바로 그 연산이지요.
#
# 연산의 출력은 'result'에 numpy의 `ndarray` 객체 형태로 저장됩니다.
result = sess.run(product)
print(result)
# ==> [[ 12.]]

# 다 끝났으면 세션을 닫아줍니다.
sess.close()
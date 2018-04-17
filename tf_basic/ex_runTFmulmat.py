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

# computational TF graph construction ================================
# 1x2 형렬을 만드는 Constant 연산을 생성합니다.
# 이 연산자는 기본 그래프의 노드로 추가됩니다.
#
# 생성자에 의해 반환된 값(matrix1)은 Constant 연산의 출력을 나타냅니다.
matrix1 = tf.constant([[3., 3.]]) # 1 by 2

# 2x1 행렬을 만드는 또 다른 Constant를 생성합니다.
matrix2 = tf.constant([[2.],[2.]]) # 2 by 1

# 'matrix1'과 'matrix2'를 입력으로 받는 Matmul 연산을 생성합니다.
# 반환값인 'product'는 행렬을 곱한 결과를 나타냅니다.
product = tf.matmul(matrix1, matrix2)


# 계산 그래프로 세션을 생성==================
sess = tf.Session()

result = sess.run(product)
print(result)
# ==> [[ 12.]]

sess.close()
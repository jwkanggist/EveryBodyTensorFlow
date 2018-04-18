#-*- coding: utf-8 -*-

"""
#---------------------------------------------
  filename: ex_runTFvariable.py
  - tensorflow의 기본 계산 그래프를 생성하고 평가해본다
  - tf.Variable()을 사용해 본다
  Written by Jaewook Kang
  2017 Aug.
#-------------------------------------------
"""
import tensorflow as tf

g = tf.Graph()

with g.as_default():
  # 변수를 하나 생성하고 스칼라 값인 0으로 초기화
  state = tf.Variable(0, name="cnt")

  # one을 state에 더하는 계산 그래프 생성
  one = tf.constant(1) #  상수 노드 
  add_one = tf.add(state, one) # 연산 노드
  cnt_update = tf.assign(state, add_one) # state에 다시 연산결과를 대입

  # 변수는 그래프가 올라간 뒤 'init' 연산을 실행해서 반드시 초기화 되어야함
  # 그 전에 먼저 'init' 연산을 그래프에 추가해야 합니다.
  init_op = tf.global_variables_initializer()


# 계산 그래프를 올리고 연산을 실행합니다.
with tf.Session(graph=g) as sess:

  # 변수를 사용하기 위해서 'init' 연산을 실행
  sess.run(init_op)

  #
  # 'state'의 초기값 출력
  print('init state = %s' % sess.run(state))
  # 'state'를 갱신하는 연산 실행 후 'state'를 출력
  for _ in range(3):
    sess.run(cnt_update)
    print('state = %s' % sess.run(state))


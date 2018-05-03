import torch
import numpy as np

# 일단 Tensor인걸로 결과도 FloatTensor로 나온다.
# 그럼 위의 값들은 cuda 사용되는 Tensor?
# CUDA는 아래방법을 쓰면 될듯
# dtype 을 정의해놓고 쓰자
dtype1 = torch.FloatTensor
dtype2 = torch.cuda.FloatTensor

a = torch.rand(1,2,3)
# a.fill_(3.5)    # 원하는 값으로 채워버리기

# a = a.add(4.0) #원하는 값 더해버리기
b = torch.rand(1,2,3)
c = torch.rand(2,2,3)
d = torch.rand(2,1,2,3)

# Variable 만들면서 텐서를 하나만 넣을 수 있다고함,
# Variable 은 autograd 안에 있음
# requires_grad는 뭔지 모르겠다.

print(torch.autograd.Variable(a.type(dtype1),requires_grad=False))
print(torch.autograd.Variable(a.type(dtype2)))  # 이러면 Cuda 사용하는 텐서
                                                # 근데 GPU 0이 지정된거 보면
                                                # GPU도 원하는 거 지정가능하지 않을까
# print(torch.autograd.Variable(a,a,a,a)) 이러면 error
print("=================== 구분선 ====================")

# TODO: Variable에서 Tensor로 변환
Test_Variable_to_Tensor =torch.autograd.Variable(a.type(dtype1))
print(Test_Variable_to_Tensor.data)

# TODO: Tensor에서 numpy로 변환
a_np = a.numpy()
print(a_np)
# TODO: Variable에서 numpy로 변환
print(Test_Variable_to_Tensor.data.numpy())

# TODO: numpy to Tensor to Variable
np_rand = np.random.rand(1,2,3)
print(np_rand)
np_to_tensor = torch.from_numpy(np_rand)
print(np_to_tensor)
np_to_Variable = torch.autograd.Variable(np_to_tensor.type(dtype1))
print(np_to_Variable)
print("=================== 구분선 ====================")
# 사칙연산

    # Tensor 영역

    # Variable 영역

t1 = torch.rand(1,2,3)
t2 = torch.rand(3,2,1)

t3 = torch.FloatTensor([[[1,2,3],[-1,-2,-3]]])  # size 1,2,3
t4 = torch.FloatTensor([[[1],[2]],[[3],[4]],[[5],[6]]])  # size 3,2,1

t5 = torch.rand(1,3,2)
t6 = torch.rand(2,3,2)
t7 = torch.rand(2,2,3)
print(t3, "t3", t4, "t4")
V1 = torch.autograd.Variable(t3.type(dtype1))
V2 = torch.autograd.Variable(t1.type(dtype1))
V3 = torch.autograd.Variable(t1.type(dtype2))
V4 = torch.autograd.Variable(t4.type(dtype1))
V5 = torch.autograd.Variable(t5.type(dtype1))
V6 = torch.autograd.Variable(t6.type(dtype1))
V7 = torch.autograd.Variable(t7.type(dtype1))
# Variable 덧셈
print(V1+V2) # 단순 덧셈 가능
print(torch.add(V1,2))  # 원하는 값 덧셈 가능

# Variable 덧셈 그냥 + Cuda
# print(V1+V3)    # CUDA와 그냥은 더할 수 없다!
# print(torch.add(V1+V3)) #이것도 마찬가지!

# GPU 영역의 Tensor, Variable들과 CPU 영역의 Tensor, Variable들과 연산 불가
# 따로 도는 건 가능한 듯 하다.

# Variable 곱셈
print(V1,V4, V1*V4) # error 발생 안함 why?
print(torch.mul(V1,V4))


    # 두 결과 모두 같은 결과를 보임
    # 두 결과는 [1 2 3]          [1 1 1]  확장을 해서 각 인스턴스별로 곱한 결과를 보임
    #          [-1 -2 -3]       [2 2 2]

# print(torch.matmul(V1,V4)) # error 발생 함   사이즈 오류 1x2x3와 3x2x1 인데 2x3과 2x1이 들어가는듯 맨 앞은 배치

# 위로 알아낸 사실은 matmul은 아마 2d matrix multiply이기 때문에 저런식으로 오류가 나는게 아닐까 싶다.

# 그러면 다음 size가 같은거 하면?
# print(torch.matmul(V1,V2)) # 마찬가지로 error 자동변환 ㄴㄴ

# Next 그러면 1x2x3과 1x3x2는 잘 동작하겠지?

print(torch.matmul(V1,V5))  # 결과물 1x2x2 잘 동작 한다.

# 그러면 1x2x3과 2x3x2는? 2x2x3과 2x3x2는?
print(torch.matmul(V1,V6))  # 결과물 2x2x2 잘 동작 한다. 이 경우 1x2x3을 각 1x3x2에 matmul하는 것이고
print(torch.matmul(V7,V6))  # 결과물 2x2x2 잘 동작 한다. 이 경우 각 1x2x3을 각 1x3x2에 matmul 하는 것?



# TODO: 1x2x3 이런식으로 3차원인경우 맨 앞 차원은 배치로 들어간다. 그렇다면 4차원인경우도 가능한가?

# 4 rank -> 2x2x2x3 -> Tensorflow (?,사이즈,사이즈,채널)



print("=================== 구분선 ====================")

#
# TODO : 원하는 Shape의 Vairable 만들기

# 인공지능과 빅데이터

## Kotlin을 활용하여 인공지능이 데이터를 행렬로 표현하고 학습하는 과정을 탐구하기 - 순전파(Forward Propagation)와 역전파(Back Propagation)를 중심으로

# 0. 목차

- [인공지능과 빅데이터](#인공지능과-빅데이터)
    - [Kotlin을 활용하여 인공지능이 데이터를 행렬로 표현하고 학습하는 과정을 탐구하기 - 순전파(Forward Propagation)와 역전파(Back Propagation)를 중심으로](#kotlin을-활용하여-인공지능이-데이터를-행렬로-표현하고-학습하는-과정을-탐구하기---순전파forward-propagation와-역전파back-propagation를-중심으로)
- [0. 목차](#0-목차)
- [1. 서론](#1-서론)
- [2. 이론적 배경](#2-이론적-배경)
    - [2.1 인공지능과 빅데이터의 관계](#21-인공지능과-빅데이터의-관계)
    - [2.2 데이터를 숫자로 표현하는 이유](#22-데이터를-숫자로-표현하는-이유)
    - [2.3 벡터(Vector)와 행렬(Matrix)](#23-벡터vector와-행렬matrix)
    - [2.4 신경망(Neural Network)](#24-신경망neural-network)
    - [2.5 순전파(Forward Propagation)](#25-순전파forward-propagation)
    - [2.6 활성 함수(Activation Function)](#26-활성-함수activation-function)
        - [2.6.1 ReLU 함수](#261-relu-함수)
        - [2.6.2 Sigmoid 함수](#262-sigmoid-함수)
    - [2.7 손실 함수(Loss Function)](#27-손실-함수loss-function)
    - [2.8 역전파(Back Propagation)](#28-역전파back-propagation)
- [3. Kotlin 구현](#3-kotlin-구현)
    - [3.1 구현 목표](#31-구현-목표)
    - [3.2 행렬(Matrix) 클래스 구현](#32-행렬matrix-클래스-구현)
        - [3.2.1 구현 목표](#321-구현-목표)
        - [3.2.2 Matrix 클래스 생성](#322-matrix-클래스-생성)
            - [Matrix.kt](#matrixkt)
            - [Main.kt](#mainkt)
            - [실행 결과](#실행-결과)
            - [코드 설명](#코드-설명)
        - [3.2.3 행렬 덧셈](#323-행렬-덧셈)
            - [Matrix.kt](#matrixkt-1)
            - [코드 설명](#코드-설명-1)
        - [3.2.4 행렬 뺄셈](#324-행렬-뺄셈)
            - [Matrix.kt](#matrixkt-2)
            - [코드 설명](#코드-설명-2)
        - [3.2.5 행렬 곱셈](#325-행렬-곱셈)
            - [Matrix.kt](#matrixkt-3)
            - [코드 설명](#코드-설명-3)
        - [3.2.6 스칼라 곱](#326-스칼라-곱)
            - [Matrix.kt](#matrixkt-4)
            - [코드 설명](#코드-설명-4)
        - [3.2.7 Hadamard 곱](#327-hadamard-곱)
            - [Matrix.kt](#matrixkt-5)
            - [코드 설명](#코드-설명-5)
            - [3.2.8 전치 행렬](#328-전치-행렬)
            - [Matrix.kt](#matrixkt-6)
            - [코드 설명](#코드-설명-6)
        - [3.2.9 map() 함수](#329-map-함수)
            - [Matrix.kt](#matrixkt-7)
            - [코드 설명](#코드-설명-7)
    - [3.2.10 Matrix.random() 구현](#3210-matrixrandom-구현)
        - [Matrix.kt](#matrixkt-8)
        - [코드 설명](#코드-설명-8)
    - [3.2.11 정리](#3211-정리)
- [3.3 활성화 함수(Activation Function) 구현](#33-활성화-함수activation-function-구현)
    - [3.3.1 구현 목표](#331-구현-목표)
    - [3.3.2 ReLU(Rectified Linear Unit)](#332-relurectified-linear-unit)
        - [Activation.kt](#activationkt)
        - [Main.kt](#mainkt-1)
        - [실행 결과](#실행-결과-1)
        - [코드 설명](#코드-설명-9)
    - [3.3.3 Sigmoid 함수](#333-sigmoid-함수)
        - [Activation.kt](#activationkt-1)
        - [Main.kt](#mainkt-2)
        - [실행 결과](#실행-결과-2)
        - [코드 설명](#코드-설명-10)
    - [3.3.4 ReLU 미분](#334-relu-미분)
        - [Activation.kt](#activationkt-2)
    - [3.3.5 Sigmoid 미분](#335-sigmoid-미분)
        - [Activation.kt](#activationkt-3)
        - [Main.kt](#mainkt-3)
        - [실행 결과](#실행-결과-3)
        - [코드 설명](#코드-설명-11)
    - [3.3.10 정리](#3310-정리)
- [3.4 순전파(Forward Propagation) 구현](#34-순전파forward-propagation-구현)
    - [3.4.1 구현 목표](#341-구현-목표)
    - [3.4.2 Dense Layer 구현](#342-dense-layer-구현)
        - [DenseLayer.kt](#denselayerkt)
        - [코드 설명](#코드-설명-12)
    - [3.4.3 다층 신경망 구현](#343-다층-신경망-구현)
        - [NeuralNetwork.kt](#neuralnetworkkt)
        - [코드 설명](#코드-설명-13)
    - [3.4.4 순전파 실행](#344-순전파-실행)
        - [Main.kt](#mainkt-4)
        - [실행 결과 (예시)](#실행-결과-예시)
        - [코드 설명](#코드-설명-14)
    - [3.4.5 순전파 과정 정리](#345-순전파-과정-정리)
- [3.5 손실 함수(Loss Function) 구현](#35-손실-함수loss-function-구현)
    - [3.5.1 구현 목표](#351-구현-목표)
    - [3.5.2 평균 제곱 오차(MSE)](#352-평균-제곱-오차mse)
    - [3.5.3 MSE 손실 함수 구현](#353-mse-손실-함수-구현)
        - [Loss.kt](#losskt)
    - [코드 설명](#코드-설명-15)
    - [3.5.4 손실 함수 실행 확인](#354-손실-함수-실행-확인)
        - [Main.kt](#mainkt-5)
        - [실행 결과](#실행-결과-4)
    - [코드 설명](#코드-설명-16)
    - [3.5.5 역전파에서의 손실 함수 활용](#355-역전파에서의-손실-함수-활용)
    - [3.5 정리](#35-정리)
- [3.6 역전파(Back Propagation) 구현](#36-역전파back-propagation-구현)
    - [3.6.1 구현 목표](#361-구현-목표)
    - [3.6.2 활성화 함수 미분](#362-활성화-함수-미분)
        - [Activation.kt](#activationkt-4)
        - [코드 설명](#코드-설명-17)
    - [3.6.3 손실 함수의 Gradient 계산](#363-손실-함수의-gradient-계산)
    - [Loss.kt](#losskt-1)
    - [코드 설명](#코드-설명-18)
    - [3.6.4 Dense Layer 역전파 구현](#364-dense-layer-역전파-구현)
        - [DenseLayer.kt](#denselayerkt-1)
        - [코드 설명](#코드-설명-19)
    - [3.6.5 Learning Rate](#365-learning-rate)
    - [3.6 정리](#36-정리)
- [3.7 학습 과정](#37-학습-과정)
    - [3.7.1 구현 목표](#371-구현-목표)
    - [3.7.2 학습 구현](#372-학습-구현)
        - [NeuralNetwork.kt](#neuralnetworkkt-1)
        - [코드 설명](#코드-설명-20)
    - [3.7.3 Epoch 반복](#373-epoch-반복)
        - [Main.kt](#mainkt-6)
        - [코드 설명](#코드-설명-21)
    - [3.7.4 학습 결과](#374-학습-결과)
    - [3.7 정리](#37-정리)
- [4. 실험 및 결과 분석](#4-실험-및-결과-분석)
- [5. 결론](#5-결론)

# 1. 서론

최근 ChatGPT, Claude, Gemini를 비롯한 다양한 인공지능 서비스가 빠르게 발전하면서, 인공지능이 사람의 언어와 이미지 같은 복잡한 정보를 어떻게 이해하고 학습하는지에 관심을 갖게 되었다.
처음에는 인공지능이 데이터를 그대로 처리한다고 생각했지만, 조사해 보니 컴퓨터는 문자나 이미지의 의미를 직접 이해하지 못하며, 모든 데이터를 숫자로 변환한 뒤 **벡터(Vector)**와 **행렬(Matrix)**의 형태로 계산한다는 사실을 알게 되었다.

특히 인공지능의 핵심 연산이 행렬 곱셈을 기반으로 이루어지고, **순전파(Forward Propagation)**와 **역전파(Back Propagation)** 과정을 반복하면서 가중치를 수정해 학습한다는 점이 매우 흥미로웠다. 이는 학교에서 배우는 행렬이 단순한 수학 개념이 아니라 실제 인공지능 기술의 핵심 원리라는 것을 보여준다.

이에 본 탐구에서는 Kotlin을 활용하여 인공지능이 데이터를 행렬로 표현하고 처리하는 과정을 직접 구현해 보고, 행렬 연산이 **신경망(Neural Network)**에서 어떻게 사용되는지와 순전파 및 역전파를 통해 인공지능이 학습하는 원리를 이해하고자 한다. 이를 통해 수학적 개념과 프로그래밍이 실제 인공지능 기술에서 어떻게 융합되는지를 탐구하는 것을 목표로 한다.

---

# 2. 이론적 배경

## 2.1 인공지능과 빅데이터의 관계

빅데이터(Big Data)는 매우 방대한 양의 데이터를 의미하며, 데이터의 생성 속도가 빠르고 종류 또한 매우 다양하다는 특징을 가진다. 이러한 데이터는 기존의 데이터 처리 방식만으로는 효율적으로 저장하거나 분석하기 어렵기 때문에, 대용량 데이터를 처리할 수 있는 기술이 필요하다.

인공지능(AI)은 컴퓨터가 사람처럼 학습하고 추론할 수 있도록 하는 기술로, 머신러닝(Machine Learning), 딥러닝(Deep Learning), 자연어 처리(Natural Language Processing) 등의 분야를 포함한다. 인공지능은 데이터를 반복적으로 분석하여 일정한 패턴을 학습하고, 이를 바탕으로 새로운 데이터에 대한 예측이나 판단을 수행한다.

빅데이터와 인공지능은 서로 밀접하게 연결되어 있다. 빅데이터는 인공지능이 학습하는 데 필요한 다양한 데이터를 제공하며, 인공지능은 이러한 데이터를 분석하여 의미 있는 정보를 찾아낸다. 즉, 데이터가 많고 다양할수록 인공지능은 더 정확한 패턴을 학습할 수 있으며, 반대로 인공지능의 발전은 방대한 데이터를 효율적으로 분석하고 활용하는 데 중요한 역할을 한다.

그러나 컴퓨터는 문자나 이미지의 의미를 직접 이해하지 못한다. 따라서 텍스트, 이미지, 음성 등 모든 데이터는 먼저 숫자로 변환되어 벡터(Vector)와 행렬(Matrix)의 형태로 표현된다. 이후 인공지능은 이러한 행렬 데이터를 이용하여 순전파와 역전파를 반복 수행하면서 가중치를 조정하고 학습한다. 즉, 빅데이터는 인공지능의 학습 재료가 되고, 행렬 연산은 그 데이터를 실제로 처리하는 핵심 계산 방법이라고 할 수 있다.

## 2.2 데이터를 숫자로 표현하는 이유

컴퓨터는 문자나 이미지의 의미를 직접 이해하지 못하므로 모든 데이터를 숫자로 변환하여 처리한다.

예를 들어 텍스트는 단어를 숫자로 변환하는 **임베딩(Embedding)** 과정을 거치고, 이미지는 각 픽셀의 RGB 값으로 표현되며, 음성은 일정 시간 동안의 진폭과 주파수 정보를 숫자로 변환하여 저장한다.

이처럼 숫자로 변환된 데이터는 벡터와 행렬 형태로 표현되며, 인공지능은 이러한 수학적 표현을 이용하여 빠르고 효율적으로 계산을 수행한다.

## 2.3 벡터(Vector)와 행렬(Matrix)

벡터(Vector)는 여러 개의 값을 순서대로 나열한 데이터 구조이다. 하나의 데이터(객체)를 여러 개의 숫자로 표현할 때 자주 사용된다

예를 들어 학생의 세 과목 점수는 다음과 같이 하나의 벡터로 나타낼 수 있다.

$$
x=
\begin{bmatrix}
80\\
90\\
70
\end{bmatrix}
$$

같은 길이의 벡터들을 행 또는 열 방향으로 배열하면 행렬(Matrix)이 된다.

$$
A=
\begin{bmatrix}
80&90&70\\
75&85&95\\
60&70&80
\end{bmatrix}
$$

인공지능에서는 입력 데이터, 가중치, 출력값 등을 벡터나 행렬(더 일반적으로는 텐서) 형태로 표현하며, 신경망의 대부분의 연산은 행렬 곱셈을 기반으로 수행된다.

## 2.4 신경망(Neural Network)

신경망(Neural Network)은 사람의 뇌에서 뉴런(Neuron)이 서로 연결되어 정보를 전달하는 방식을 모방하여 만든 인공지능 모델이다. 하나의 뉴런은 입력값을 받아 계산한 뒤 다음 뉴런으로 전달하며, 이러한 뉴런들이 여러 층으로 연결되어 복잡한 문제를 해결할 수 있다.

신경망은 일반적으로 다음과 같은 구조를 가진다.

- 입력층(Input Layer)
- 은닉층(Hidden Layer)
- 출력층(Output Layer)

```text
입력층
 x1  x2  x3
      │
      ▼
은닉층 1
 h1  h2  h3
      │
      ▼
은닉층 2
 h1  h2
      │
      ▼
      .
      .
      .
출력층
   y
```

입력층은 데이터를 받아들이는 역할을 하며, 출력층은 최종 예측 결과를 출력한다.

그 사이에 위치하는 **은닉층(Hidden Layer)은 하나만 존재하는 것이 아니라 여러 개가 사용될 수 있다.** 은닉층의 개수가 많아질수록 신경망은 더욱 복잡한 특징을 학습할 수 있으며, 이러한 구조를 **심층 신경망(Deep Neural Network)**이라고 한다.

예를 들어 이미지 인식에서는 첫 번째 은닉층이 선이나 모서리 같은 단순한 특징을 학습하고, 이후 은닉층에서는 눈, 코, 입과 같은 복잡한 특징을 학습한 뒤 마지막에는 사람의 얼굴인지 여부를 판단하게 된다.

각 층에서는 입력 데이터에 대해 행렬 곱셈과 편향을 계산한 뒤 활성화 함수를 적용하며, 이 과정을 반복하여 최종 예측값을 생성한다.

## 2.5 순전파(Forward Propagation)

순전파(Forward Propagation)는 입력 데이터를 신경망의 앞쪽에서 뒤쪽으로 전달하여 최종 예측값을 계산하는 과정이다. 신경망은 각 층(Layer)에서 입력값을 받아 새로운 특징을 추출하며, 이러한 계산을 반복하여 최종 출력값을 얻는다.

먼저 입력 행렬 (X)와 가중치 행렬 (W)를 행렬 곱셈으로 곱한 뒤, 편향(Bias) (b)를 더하여 선형 변환 결과 (Z)를 계산한다.

$$
Z=XW+b
$$

여기서 입력 행렬 (X)는 학습 데이터, 가중치 행렬 (W)는 입력이 출력에 얼마나 영향을 주는지를 나타내는 값이며, 편향 (b)는 계산 결과를 일정한 값만큼 이동시켜 모델이 더 다양한 데이터를 표현할 수 있도록 돕는다.

하지만 위 식만으로는 모든 계산이 단순한 선형 변환에 그치므로 복잡한 패턴을 학습하기 어렵다. 이를 해결하기 위해 활성화 함수(Activation Function)를 적용하여 비선형성을 추가한다.

$$
A=f(Z)
$$

여기서 (f)는 활성화 함수이며, [하단에서 자세히 설명](#26-활성-함수activation-function).

즉, 순전파는 **입력 데이터를 행렬 연산과 활성화 함수를 이용해 단계적으로 변환하면서 예측 결과를 만들어 내는 과정**이라고 할 수 있다.

## 2.6 활성 함수(Activation Function)

활성화 함수(Activation Function)는 신경망에서 가장 중요한 요소 중 하나이다.

만약 신경망이 행렬 곱셈과 덧셈만 수행한다면 모든 계산은 단순한 직선(선형 함수)의 조합이 되므로, 아무리 층을 많이 쌓아도 복잡한 문제를 해결하기 어렵다.

이를 해결하기 위해 활성화 함수를 사용하여 **비선형성(Non-linearity)**을 추가한다. 덕분에 신경망은 문자, 이미지, 음성과 같은 복잡한 데이터를 학습할 수 있다.

대표적인 활성화 함수에는 다음과 같은 것들이 있다.

- ReLU(Rectified Linear Unit)
- Sigmoid
- Tanh

이번 탐구에서는 Kotlin 구현에서 사용할 ReLU와 Sigmoid를 중심으로 살펴본다.

### 2.6.1 ReLU 함수

ReLU(Rectified Linear Unit)는 현재 가장 많이 사용되는 활성화 함수이다.

입력이 0보다 작으면 0을 출력하고, 0보다 크면 입력값을 그대로 출력한다.

수식은 다음과 같다.
$$
f(x)=max(0,x)
$$

값을 표로 표현하면 다음과 같이 된다.

| 입력값 | 출력값 |
| ------ | ------ |
| -3     | 0      |
| -1     | 0      |
| 2      | 2      |
| 5      | 5      |

ReLU는 계산이 매우 단순하기 때문에 연산 속도가 빠르며, 기울기 소실(Vanishing Gradient) 문제가 비교적 적어 대부분의 딥러닝 모델에서 기본 활성화 함수로 사용된다.

### 2.6.2 Sigmoid 함수

Sigmoid 함수는 출력값을 항상 0과 1 사이로 변환하는 활성화 함수이다.

수식은 다음과 같다.

$$
f(x)=\frac{1}{1+e^{-x}}
$$

값을 표로 표현하면 다음과 같이 된다.

| 입력값 | 출력값(약) |
| ------ | ---------- |
| -5     | 0.007      |
| -1     | 0.269      |
| 0      | 0.500      |
| 1      | 0.731      |
| 5      | 0.993      |

출력이 확률처럼 해석될 수 있기 때문에 이진 분류(Binary Classification) 문제의 출력층에서 자주 사용된다.

하지만 입력값이 매우 크거나 매우 작으면 기울기가 거의 0이 되어 학습 속도가 느려지는 기울기 소실(Vanishing Gradient) 문제가 발생할 수 있다.

## 2.7 손실 함수(Loss Function)

순전파가 끝나면 신경망은 하나의 예측값을 출력한다. 그러나 이 값이 실제 정답과 얼마나 차이가 나는지는 아직 알 수 없다. 따라서 **손실 함수(Loss Function)**를 이용하여 예측값과 실제 정답 사이의 오차를 수치로 계산한다.

손실값(Loss)은 모델이 얼마나 잘못 예측했는지를 나타내는 값이며, 손실값이 작을수록 예측이 실제 정답에 가까워졌음을 의미한다.

대표적인 손실 함수인 평균제곱오차(MSE)는 다음과 같다.

$$
Loss=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

이 탐구에서는 미분을 단순화하기 위해 흔히 사용되는 $\frac{1}{2}MSE$ 형태(1/2를 곱하는 방식)를 사용하지 않고, 위 식을 그대로 사용한다. 이는 3.5절과 3.6절에서 손실 함수와 그 Gradient를 구현할 때 서로 어긋나지 않도록 하기 위함이다.

여기서

- y : 실제 정답
- ŷ : 신경망의 예측값
- n : 데이터 개수

이다.

예를 들어 실제 정답이 10이고 예측값이 8이라면 오차는 2이며, 손실 함수는 이 오차를 이용하여 모델의 성능을 평가한다. 역전파는 바로 이 손실값을 줄이는 방향으로 가중치를 수정하는 과정이다.

## 2.8 역전파(Back Propagation)

역전파(Back Propagation)는 순전파에서 계산된 손실값을 이용하여 신경망의 각 가중치에 대한 기울기(Gradient)를 계산하는 과정이다.

순전파에서는 입력층에서 출력층 방향으로 계산이 진행되지만, 역전파에서는 출력층에서 입력층 방향으로 오차를 전달하면서 각 가중치가 손실값에 얼마나 영향을 주었는지를 미분을 통해 계산한다.

이렇게 계산된 기울기를 이용하여 가중치를 갱신하는 방법을 **최적화 알고리즘(Optimizer)** 이라고 한다. 가장 대표적인 방법은 **경사하강법(Gradient Descent)** 이며, 이 탐구에서도 경사하강법을 사용하여 가중치를 갱신한다. 실제 인공지능에서는 **Adam**, **RMSProp**, **Momentum** 등 다양한 최적화 알고리즘도 널리 사용된다.

경사하강법에서 가중치는 다음 식과 같이 갱신된다.

$$
W_{new}=W-\eta\frac{\partial Loss}{\partial W}
$$

여기서

- W : 기존 가중치(Weight)
- η : 학습률(Learning Rate)
- ∂Loss/∂W : 손실 함수를 가중치 W에 대해 미분한 값(기울기, Gradient)

이다.

학습률은 가중치를 얼마나 크게 수정할지를 결정하는 값이다. 너무 크면 최적값을 지나칠 수 있고, 너무 작으면 학습 속도가 매우 느려질 수 있다.

순전파와 역전파를 반복하면서 손실값을 계산하고, 경사하강법을 이용해 가중치를 지속적으로 수정하면 손실값은 점차 감소하고 신경망은 데이터 속의 패턴을 학습하게 된다.

---

# 3. Kotlin 구현

## 3.1 구현 목표

이번 구현에서는 인공지능 라이브러리를 사용하지 않고 Kotlin만을 이용하여 신경망의 기본적인 학습 과정을 직접 구현하였다.

행렬 연산을 수행하는 Matrix 클래스를 직접 작성하고, ReLU와 Sigmoid 활성화 함수, 순전파(Forward Propagation), 손실 함수(Loss Function), 역전파(Back Propagation)를 구현하여 신경망이 데이터를 학습하는 과정을 확인하였다.

이를 통해 인공지능이 실제로 행렬 연산을 기반으로 동작하며, 반복적인 학습을 통해 가중치를 수정하는 원리를 이해하고자 하였다.

또한 각 기능을 단계적으로 구현하면서 행렬 연산, 활성화 함수, 순전파, 손실 함수, 역전파가 서로 어떻게 연결되어 하나의 신경망을 구성하는지 확인하는 것을 목표로 하였다.

## 3.2 행렬(Matrix) 클래스 구현

### 3.2.1 구현 목표

신경망은 입력 데이터, 가중치(Weight), 편향(Bias), 출력값(Output) 등을 모두 행렬(Matrix)의 형태로 저장하고 계산한다. 따라서 신경망을 구현하기 위해서는 다양한 행렬 연산을 수행할 수 있는 Matrix 클래스가 필요하다.

이번 구현에서는 Kotlin의 기본 문법만을 이용하여 Matrix 클래스를 직접 구현하였다. 또한 신경망 학습 과정에서 필요한 행렬 덧셈, 뺄셈, 곱셈, 전치 행렬, 스칼라 곱, Hadamard 곱, map 함수 등을 순차적으로 구현하였다.

이를 통해 외부 인공지능 라이브러리를 사용하지 않고도 신경망을 구성할 수 있는 기반을 마련하였다.

### 3.2.2 Matrix 클래스 생성

가장 먼저 행렬의 크기와 데이터를 저장하는 Matrix 클래스를 작성하였다.

#### Matrix.kt

```Kotlin
class Matrix(
    val rows: Int,
    val cols: Int
) {

    private val data = Array(rows) {
        DoubleArray(cols)
    }

    operator fun get(row: Int, col: Int): Double {
        return data[row][col]
    }

    operator fun set(row: Int, col: Int, value: Double) {
        data[row][col] = value
    }

    // 행렬 내용을 사람이 읽기 좋은 형태로 출력하기 위한 함수
    override fun toString(): String {

        val sb = StringBuilder()

        for (row in 0 until rows) {
            for (col in 0 until cols) {
                sb.append(this[row, col])
                if (col != cols - 1) sb.append(" ")
            }
            sb.append("\n")
        }

        return sb.toString()

    }

}
```

`println(matrix)`처럼 Matrix 객체를 직접 출력할 경우, `toString()`을 오버라이드하지 않으면 `Matrix@1a2b3c4d`와 같은 객체 참조값이 출력된다. 따라서 행렬의 각 원소를 행 단위로 나열하여 출력하도록 `toString()`을 구현하였다. 이는 이후 3.4.4의 순전파 실행 결과나 3.7.3의 학습 과정에서 `println(output)`, `println(loss)` 등을 이용해 결과를 직접 확인할 때 사용된다.

#### Main.kt

```Kotlin
fun main() {

    val matrix = Matrix(2,3)

    matrix[0,0] = 1.0
    matrix[0,1] = 2.0
    matrix[0,2] = 3.0

    matrix[1,0] = 4.0
    matrix[1,1] = 5.0
    matrix[1,2] = 6.0

    println(matrix[1,2])

}
```

#### 실행 결과

```Kotlin
6.0

```

#### 코드 설명

Matrix 클래스는 행렬의 행(Row)과 열(Column)의 크기를 저장하고, 내부적으로 Double형 2차원 배열을 이용하여 데이터를 관리한다.

또한 get()과 set() 연산자를 구현하여 matrix[row, col] 형태로 값을 읽고 저장할 수 있도록 하였다.

이후 구현할 모든 행렬 연산은 이 클래스를 기반으로 수행된다.

---

### 3.2.3 행렬 덧셈

행렬 덧셈은 같은 위치에 있는 원소끼리 더하는 연산이다.

두 행렬의 크기가 같을 때만 수행할 수 있다.

#### Matrix.kt

```Kotlin
operator fun plus(other: Matrix): Matrix {

    require(rows == other.rows && cols == other.cols) {
        "행렬의 크기가 같아야 합니다."
    }

    val result = Matrix(rows, cols)

    for (row in 0 until rows) {
        for (col in 0 until cols) {
            result[row, col] =
                this[row, col] + other[row, col]
        }
    }

    return result

}
```

#### 코드 설명

덧셈 결과를 새로운 Matrix 객체에 저장하여 반환한다.

기존 행렬은 변경하지 않기 때문에 동일한 행렬을 여러 계산에서 안전하게 사용할 수 있다.

--- 

### 3.2.4 행렬 뺄셈

역전파에서는 오차(Error)를 계산하기 위해 행렬 뺄셈이 자주 사용된다.

예를 들어

$$
Error=Prediction−Target
$$

을 계산하기 위해서는 뺄셈 연산이 반드시 필요하다.

#### Matrix.kt

```Kotlin
operator fun minus(other: Matrix): Matrix {

    require(rows == other.rows && cols == other.cols)

    val result = Matrix(rows, cols)

    for(row in 0 until rows){

        for(col in 0 until cols){

            result[row,col] =
                this[row,col] - other[row,col]

        }

    }

    return result

}
```

#### 코드 설명

minus() 함수는 같은 위치의 원소를 서로 빼 새로운 행렬을 생성한다.

학습 과정에서는 예측값과 실제값의 차이를 계산할 때 주로 사용된다.

---

### 3.2.5 행렬 곱셈

신경망에서 가장 중요한 연산은 행렬 곱셈이다.

순전파는 다음 식으로 표현된다.

$$
Z=XW+b
$$

여기서 XW가 바로 행렬 곱셈이다.

#### Matrix.kt

```Kotlin
operator fun times(other: Matrix): Matrix {

    require(cols == other.rows) {
        "행렬 곱셈이 불가능합니다."
    }

    val result = Matrix(rows, other.cols)

    for(row in 0 until rows){

        for(col in 0 until other.cols){

            var sum = 0.0

            for(k in 0 until cols){

                sum += this[row,k] * other[k,col]

            }

            result[row,col] = sum

        }

    }

    return result

}
```

#### 코드 설명

행렬 곱셈은 앞 행렬의 한 행과 뒤 행렬의 한 열을 서로 곱한 뒤 모두 더하여 하나의 원소를 계산한다.

이 연산은 신경망의 모든 층에서 반복적으로 수행되는 가장 핵심적인 계산이다.

---

### 3.2.6 스칼라 곱

역전파에서는 Gradient에 Learning Rate를 곱하는 연산이 자주 발생한다.

$$
W=W−αG
$$

여기서

α가 스칼라(실수)이다.

#### Matrix.kt

```Kotlin
operator fun times(value: Double): Matrix{

    val result = Matrix(rows,cols)

    for(row in 0 until rows){

        for(col in 0 until cols){

            result[row,col]=
                this[row,col]*value

        }

    }

    return result

}
```

#### 코드 설명

행렬의 모든 원소에 동일한 실수를 곱하여 새로운 행렬을 반환한다.

학습률(Learning Rate)을 적용하거나 Gradient의 크기를 조절할 때 사용된다.

---

### 3.2.7 Hadamard 곱

Hadamard 곱은 같은 위치의 원소끼리 곱하는 연산이다.

행렬 곱셈과는 다른 연산이며, 역전파에서 활성화 함수의 미분을 적용할 때 사용된다.

#### Matrix.kt
```Kotlin
fun hadamard(other: Matrix): Matrix{

    require(rows==other.rows && cols==other.cols)

    val result = Matrix(rows,cols)

    for(row in 0 until rows){

        for(col in 0 until cols){

            result[row,col]=
                this[row,col]*other[row,col]

        }

    }

    return result

}
```

#### 코드 설명

Hadamard 곱은 각 원소를 독립적으로 곱하는 연산이다.

역전파에서는

$$
\delta = (\hat{y} - y) \odot f'(z)
$$

각 기호의 의미
- δ : 오차(Gradient)
- ŷ : 신경망의 예측값
- y : 실제 정답(Label)
- ⊙ : Hadamard 곱(원소별 곱셈)
- f′(z) : 활성화 함수의 미분값


와 같이 활성화 함수의 미분값을 적용할 때 사용된다.

---

#### 3.2.8 전치 행렬

전치 행렬(Transpose)은 행과 열을 서로 바꾸는 연산이다.

행렬 곱셈의 방향을 변경해야 하는 역전파에서 반드시 사용된다.

#### Matrix.kt

```Kotlin
fun transpose(): Matrix{

    val result = Matrix(cols,rows)

    for(row in 0 until rows){

        for(col in 0 until cols){

            result[col,row]=
                this[row,col]

        }

    }

    return result

}
```

#### 코드 설명

행(Row)을 열(Column)로, 열(Column)을 행(Row)으로 바꾸어 새로운 행렬을 생성한다.

역전파에서는 가중치 Gradient를 계산할 때 자주 사용된다.

---

### 3.2.9 map() 함수

활성화 함수는 행렬의 모든 원소에 동일하게 적용된다.

이를 위해 Matrix 클래스에 map() 함수를 구현하였다.

#### Matrix.kt

```Kotlin
fun map(transform: (Double) -> Double): Matrix{

    val result = Matrix(rows,cols)

    for(row in 0 until rows){

        for(col in 0 until cols){

            result[row,col]=
                transform(this[row,col])

        }

    }

    return result

}
```

#### 코드 설명

map() 함수는 전달받은 함수를 행렬의 모든 원소에 적용하여 새로운 행렬을 반환한다.

이를 이용하면 ReLU, Sigmoid와 같은 활성화 함수를 매우 간단하게 구현할 수 있으며, 코드의 재사용성 또한 높아진다.

## 3.2.10 Matrix.random() 구현

신경망의 가중치(Weight)와 편향(Bias)은 학습을 시작하기 전에 초기값이 필요하다. 만약 모든 가중치를 동일한 값으로 초기화하면 모든 뉴런이 같은 계산만 수행하여 서로 다른 특징을 학습하지 못하게 된다.

이를 방지하기 위해 작은 난수를 이용하여 가중치를 무작위로 초기화하였다. 이번 구현에서는 Matrix.random() 함수를 작성하여 원하는 크기의 행렬을 생성하고, 각 원소를 일정 범위의 난수로 초기화하도록 하였다.

#### Matrix.kt
```Kotlin
companion object {

    fun random(
        rows: Int,
        cols: Int
    ): Matrix {

        val matrix = Matrix(rows, cols)

        for (row in 0 until rows) {
            for (col in 0 until cols) {
                matrix[row, col] =
                    Random.nextDouble(-1.0, 1.0)
            }
        }

        return matrix

    }

}
```

#### 코드 설명

Matrix.random() 함수는 지정한 크기의 행렬을 생성한 뒤, 각 원소를 -1.0 이상 1.0 미만의 난수로 초기화하여 반환한다.

신경망에서는 학습을 시작하기 전에 가중치와 편향을 이러한 난수로 초기화한다. 모든 값을 동일하게 초기화하면 모든 뉴런이 같은 계산을 수행하여 서로 다른 특징을 학습하지 못하기 때문이다.

무작위 초기화를 통해 각 뉴런은 서로 다른 시작점을 가지게 되며, 이후 역전파와 경사하강법을 반복하면서 데이터의 특징에 맞게 가중치가 점차 수정된다.

---

## 3.2.11 정리

지금까지 구현한 Matrix 클래스는 단순한 행렬 저장 기능뿐만 아니라 신경망 학습에 필요한 다양한 행렬 연산을 수행할 수 있도록 설계하였다.

구현한 기능은 다음과 같다.
| 기능        | 용도               |
| ----------- | ------------------ |
| 생성        | 행렬 저장          |
| get / set   | 원소 접근          |
| 덧셈        | Bias 추가          |
| 뺄셈        | 오차 계산          |
| 행렬 곱셈   | 순전파             |
| 스칼라 곱   | Learning Rate 적용 |
| Hadamard 곱 | 활성화 함수 미분   |
| 전치 행렬   | Gradient 계산      |
| map()       | 활성화 함수 적용   |

이 Matrix 클래스는 이후 구현할 다층 신경망(Multi-Layer Perceptron, MLP) 의 순전파와 역전파에서 공통적으로 사용되며, 신경망의 모든 계산을 담당하는 핵심 구성 요소가 된다.

또한 Matrix 클래스를 직접 구현함으로써 외부 인공지능 라이브러리가 내부적으로 수행하는 기본적인 행렬 연산을 직접 확인할 수 있었다.

이를 통해 신경망이 복잡한 알고리즘처럼 보이더라도 결국 다양한 행렬 연산과 함수들의 조합으로 이루어진다는 사실을 이해할 수 있었다.

# 3.3 활성화 함수(Activation Function) 구현

## 3.3.1 구현 목표

신경망에서 행렬 곱셈만 수행하면 입력과 출력 사이의 관계는 항상 선형(Linear) 관계만 표현할 수 있다. 그러나 실제 데이터는 대부분 비선형적인 특성을 가지므로, 이를 학습하기 위해서는 **활성화 함수(Activation Function)** 가 필요하다.

활성화 함수는 행렬 연산의 결과에 비선형성을 추가하여 신경망이 복잡한 패턴을 학습할 수 있도록 한다. 만약 활성화 함수를 사용하지 않는다면 층을 여러 개 쌓더라도 하나의 선형 변환과 동일한 결과만 얻을 수 있기 때문에 다층 신경망의 장점을 활용할 수 없다.

이번 구현에서는 은닉층에서 사용하는 **ReLU(Rectified Linear Unit)** 함수와 출력층에서 사용하는 **Sigmoid** 함수를 구현하였다.

---

## 3.3.2 ReLU(Rectified Linear Unit)

ReLU 함수는 가장 널리 사용되는 활성화 함수 중 하나이다.

입력값이 0보다 작으면 0을 출력하고, 0 이상이면 입력값을 그대로 반환한다.

수식은 다음과 같다.

$$
ReLU(x)=\max(0,x)
$$

### Activation.kt

```kotlin

object Activation {

    // ReLU 함수
    fun relu(x: Double): Double {
        return max(0.0, x)
    }

}
```

### Main.kt

```kotlin
fun main() {

    println(Activation.relu(-3.0))
    println(Activation.relu(-1.0))
    println(Activation.relu(0.0))
    println(Activation.relu(2.5))
    println(Activation.relu(5.0))

}
```

### 실행 결과

```Kotlin
0.0
0.0
0.0
2.5
5.0
```

### 코드 설명

ReLU 함수는 입력값이 음수이면 0을 반환하고, 양수이면 입력값을 그대로 반환한다.

구현이 매우 간단하고 계산량이 적어 대부분의 딥러닝 모델의 은닉층에서 사용된다. 또한 Sigmoid 함수보다 기울기 소실(Vanishing Gradient) 문제가 적기 때문에 깊은 신경망에서도 안정적으로 학습할 수 있다.

---

## 3.3.3 Sigmoid 함수

Sigmoid 함수는 입력값을 0과 1 사이의 값으로 변환하는 활성화 함수이다.

출력값을 확률처럼 해석할 수 있기 때문에 이진 분류(Binary Classification)의 출력층에서 자주 사용된다.

수식은 다음과 같다.

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

### Activation.kt

기존 Activation 클래스에 아래 함수를 추가한다.

```kotlin
// Sigmoid 함수
fun sigmoid(x: Double): Double {
    return 1.0 / (1.0 + exp(-x))
}
```

### Main.kt

```kotlin
fun main() {

    println(Activation.sigmoid(-5.0))
    println(Activation.sigmoid(-1.0))
    println(Activation.sigmoid(0.0))
    println(Activation.sigmoid(1.0))
    println(Activation.sigmoid(5.0))

}
```

### 실행 결과

```Kotlin
0.0066928509242848554
0.2689414213699951
0.5
0.7310585786300049
0.9933071490757153
```

### 코드 설명

Sigmoid 함수는 자연상수 e를 이용하여 입력값을 0과 1 사이의 값으로 변환한다.

입력값이 매우 작으면 0에 가까워지고, 매우 크면 1에 가까워진다. 이러한 특성 때문에 출력값을 확률처럼 사용할 수 있으며, 이진 분류 문제에서 자주 사용된다.

---

## 3.3.4 ReLU 미분

역전파 과정에서는 활성화 함수의 출력값뿐만 아니라 **활성화 함수의 미분값(기울기)**도 필요하다.  
미분값은 손실 함수에서 발생한 오차가 이전 층으로 얼마나 전달될지를 결정하는 역할을 한다.

ReLU 함수는 입력값이 0을 기준으로 변화하는 함수이므로, 미분은 다음과 같이 표현할 수 있다.

$$
\mathrm{ReLU}'(x)=
\begin{cases}
1,&x>0\\
0,&x<0\\
\text{정의되지 않음},&x=0
\end{cases}
$$

입력값이 양수인 경우 ReLU 함수의 출력은 입력값에 비례하여 변화하므로 기울기는 1이 된다.  
따라서 역전파 과정에서 전달되는 오차가 그대로 다음 층으로 전달된다.

반대로 입력값이 음수인 경우 ReLU 함수의 출력은 항상 0이므로 변화량이 없어 기울기는 0이 된다.  
따라서 해당 뉴런을 통한 오차 전달이 차단된다.

한편 \(x=0\)에서는 함수가 꺾이는 지점이므로 좌미분과 우미분의 값이 달라 미분값이 존재하지 않는다.  
하지만 실제 인공지능 구현에서는 계산의 편의를 위해 \(x=0\)에서의 미분값을 0으로 처리하는 것이 일반적이다.

### Activation.kt

```kotlin
// ReLU 미분
fun reluDerivative(x: Double): Double {
    return if (x > 0.0) 1.0 else 0.0
}
```

위 코드에서는 입력값이 0보다 크면 기울기 1을 반환하고, 그렇지 않은 경우(음수 또는 0)는 0을 반환한다.
이는 실제 딥러닝 프레임워크에서 사용하는 ReLU 미분 구현 방식과 동일하다.

---

## 3.3.5 Sigmoid 미분

Sigmoid 함수 역시 역전파를 위해 미분값을 계산해야 한다.

Sigmoid 함수의 미분은 다음과 같다.

$$
\sigma'(x)=\sigma(x)(1-\sigma(x))
$$

### Activation.kt

```kotlin
// Sigmoid 미분
fun sigmoidDerivative(x: Double): Double {

    val s = sigmoid(x)

    return s * (1.0 - s)

}
```

### Main.kt

```kotlin
fun main() {

    println(Activation.sigmoidDerivative(-2.0))
    println(Activation.sigmoidDerivative(0.0))
    println(Activation.sigmoidDerivative(2.0))

}
```

### 실행 결과

```Kotlin
0.1049935854035065
0.25
0.10499358540350662
```

### 코드 설명

Sigmoid 함수의 미분은 함수의 출력값을 이용하여 간단하게 계산할 수 있다.

역전파에서는 출력층의 오차를 계산할 때 Sigmoid 미분값을 곱하여 Gradient를 계산한다.

---

## 3.3.10 정리

이번 절에서는 다층 신경망에서 사용할 활성화 함수를 직접 구현하였다.

구현한 함수는 다음과 같다.

| 함수         | 역할                 |
| ------------ | -------------------- |
| ReLU         | 은닉층의 활성화 함수 |
| Sigmoid      | 출력층의 활성화 함수 |
| ReLU 미분    | 은닉층 역전파        |
| Sigmoid 미분 | 출력층 역전파        |

이후 구현할 순전파에서는 ReLU와 Sigmoid 함수를 사용하여 각 층의 출력값을 계산하며, 역전파에서는 각 활성화 함수의 미분값을 이용하여 오차를 이전 층으로 전달하게 된다.

활성화 함수를 직접 구현하면서 입력값 하나를 변환하는 간단한 함수처럼 보이지만, 신경망 전체에서는 비선형성을 만들어 복잡한 데이터를 학습할 수 있도록 하는 핵심 요소임을 확인할 수 있었다.

# 3.4 순전파(Forward Propagation) 구현

## 3.4.1 구현 목표

순전파(Forward Propagation)는 입력 데이터를 신경망에 전달하여 최종 예측값을 계산하는 과정이다. 신경망은 입력값을 그대로 출력하는 것이 아니라, 각 층(Layer)을 거치면서 행렬 곱셈과 활성화 함수를 반복 수행하여 점차 특징(Feature)을 추출한다.

다층 신경망(Multi-Layer Perceptron, MLP)의 순전파는 다음 순서로 진행된다.

1. 입력 행렬과 가중치 행렬을 곱한다.
2. 편향(Bias)을 더한다.
3. 활성화 함수를 적용한다.
4. 다음 층으로 출력값을 전달한다.
5. 최종 출력층의 결과를 예측값으로 사용한다.

이번 구현에서는 하나의 완전 연결층(Dense Layer)을 구현한 후, 여러 개의 Dense Layer를 연결하여 다층 신경망의 순전파를 구현하였다.

또한 은닉층과 출력층은 서로 다른 활성화 함수(ReLU, Sigmoid)를 사용하므로, 각 DenseLayer가 자신이 사용할 활성화 함수와 그 미분 함수를 직접 갖도록 구현하였다. 이렇게 하면 이후 3.6절의 역전파 구현에서 층마다 올바른 미분 함수를 사용할 수 있다.

---

## 3.4.2 Dense Layer 구현

먼저 하나의 완전 연결층(Dense Layer)을 구현하였다.

각 Layer는 다음 정보를 가진다.

- Weight(가중치)
- Bias(편향)
- 입력(Input)
- 선형변환 결과(Z)
- 출력(Output)
- 활성화 함수(Activation)와 그 미분 함수(Activation Derivative)

순전파 과정은 다음 수식을 따른다.

$$
Z = XW + b
$$

편향(Bias)은 선형 변환의 결과를 일정한 방향으로 이동시키는 역할을 한다.

만약 편향이 없다면 모든 입력이 0인 경우 출력 역시 항상 0이 되어 표현력이 제한될 수 있다. 편향을 추가하면 입력값과 관계없이 출력값을 조절할 수 있으므로 신경망은 더욱 다양한 패턴을 학습할 수 있다.

$$
A = f(Z)
$$

여기서

- X : 입력 행렬
- W : 가중치 행렬
- b : 편향
- Z : 선형 변환 결과
- A : 활성화 함수 출력

이다.

---

### DenseLayer.kt

```kotlin
class DenseLayer(
    inputSize: Int,
    outputSize: Int,
    private val activation: (Double) -> Double,
    private val activationDerivative: (Double) -> Double
) {

    // 가중치
    var weight = Matrix.random(inputSize, outputSize)

    // 편향
    var bias = Matrix.random(1, outputSize)

    // 역전파에서 사용할 값
    lateinit var input: Matrix
    lateinit var z: Matrix
    lateinit var output: Matrix

    fun forward(input: Matrix): Matrix {

        this.input = input

        z = (input * weight) + bias

        output = z.map(activation)

        return output

    }

}
```

가중치와 편향은 모두 무작위(Random) 값으로 초기화하였다.

모든 가중치를 동일한 값으로 초기화하면 모든 뉴런이 같은 계산만 수행하게 되어 학습이 제대로 이루어지지 않는다. 따라서 작은 난수를 이용하여 초기화하면 각 뉴런이 서로 다른 특징을 학습할 수 있다.

이번 구현에서는 Matrix.random() 함수를 이용하여 일정 범위의 난수를 생성하도록 하였다.

또한 기존에는 forward() 함수가 활성화 함수를 매번 인자로 전달받는 방식이었으나, 이번 구현에서는 activation과 activationDerivative를 생성자에서 전달받아 DenseLayer 내부에 저장하도록 수정하였다. 이렇게 하면 순전파에서 사용한 활성화 함수와 역전파에서 사용할 미분 함수가 항상 짝을 이루게 되어, 은닉층(ReLU)과 출력층(Sigmoid)이 서로 다른 함수를 사용하더라도 혼동 없이 동작한다.

---

### 코드 설명

DenseLayer 클래스는 신경망의 하나의 층을 의미한다.

순전파에서는 먼저 입력 행렬과 가중치 행렬을 곱한 뒤 편향을 더하여 선형 변환 결과(Z)를 계산한다.

그 후 활성화 함수를 모든 원소에 적용하여 출력값(Output)을 생성한다.

또한 입력값과 Z, 출력값을 저장하는 이유는 이후 역전파에서 Gradient를 계산하기 위해서이다. 특히 Z를 저장해 두는 이유는, 활성화 함수의 미분값을 계산할 때 활성화 함수 적용 이전의 값인 Z를 기준으로 계산해야 하기 때문이다(3.6.4 참고).

---

## 3.4.3 다층 신경망 구현

하나의 Layer만으로는 복잡한 문제를 해결하기 어렵다.

따라서 여러 개의 Dense Layer를 연결하여 다층 신경망(Multi-Layer Perceptron)을 구성하였다.

이번 구현에서는

- 입력층(Input Layer)
- 은닉층(Hidden Layer)
- 출력층(Output Layer)

으로 구성하였다.

신경망 구조는 다음과 같다.

```
입력층 (2)

      │

      ▼

은닉층 (4)

ReLU

      │

      ▼

출력층 (1)

Sigmoid
```

### NeuralNetwork.kt

```kotlin
class NeuralNetwork {

    private val hiddenLayer =
        DenseLayer(
            2, 4,
            Activation::relu,
            Activation::reluDerivative
        )

    private val outputLayer =
        DenseLayer(
            4, 1,
            Activation::sigmoid,
            Activation::sigmoidDerivative
        )

    // 여러 층을 순서대로 다루기 위한 리스트
    private val layers = listOf(hiddenLayer, outputLayer)

    fun forward(input: Matrix): Matrix {

        var output = input

        for (layer in layers) {
            output = layer.forward(output)
        }

        return output

    }

}
```

### 코드 설명

NeuralNetwork 클래스는 두 개의 Dense Layer를 연결하여 하나의 신경망을 구성한다.

은닉층(hiddenLayer)은 ReLU와 그 미분 함수(reluDerivative)를, 출력층(outputLayer)은 Sigmoid와 그 미분 함수(sigmoidDerivative)를 각각 생성자를 통해 전달받는다. 이렇게 층마다 사용할 활성화 함수를 다르게 지정함으로써, 은닉층에서는 비선형성을 추가하고 출력층에서는 출력값을 0과 1 사이로 변환하는 서로 다른 역할을 명확히 구분할 수 있다.

먼저 입력 데이터는 은닉층으로 전달되고 ReLU 활성화 함수를 적용한다. 그 결과는 다시 출력층으로 전달되어 Sigmoid 함수를 거친 후 최종 예측값을 생성한다.

또한 hiddenLayer와 outputLayer를 layers 리스트로 묶어두었는데, 이는 forward()에서 여러 층을 반복문으로 순서대로 통과시키고, 이후 3.7.2의 train() 함수에서 역순으로 역전파를 수행할 때 층의 개수와 관계없이 동일한 코드로 처리할 수 있도록 하기 위함이다.

---

## 3.4.4 순전파 실행

작성한 다층 신경망이 정상적으로 동작하는지 확인하기 위해 입력 데이터를 전달하여 순전파를 수행하였다.

### Main.kt

```kotlin
fun main() {

    val network = NeuralNetwork()

    val input = Matrix(1, 2)

    input[0, 0] = 1.0
    input[0, 1] = 0.0

    val output = network.forward(input)

    println(output)

}
```

---

### 실행 결과 (예시)

```text
0.14236366141961673
```

※ 가중치는 무작위(Random)로 초기화되므로 실행할 때마다 결과는 달라질 수 있다.

---

### 코드 설명

입력 데이터를 신경망에 전달하면 먼저 은닉층에서 행렬 곱셈과 ReLU 활성화 함수가 수행된다.

이후 출력층에서는 다시 행렬 곱셈을 수행한 뒤 Sigmoid 함수를 적용하여 최종 출력값을 계산한다.

이 과정에서는 아직 학습이 이루어지지 않았기 때문에 출력값은 무작위 가중치에 의해 결정된다.

또한 `println(output)`으로 출력되는 값은 3.2.2에서 구현한 `Matrix.toString()`에 의해 행렬의 원소가 사람이 읽기 좋은 형태로 표시된 것이다.

---

## 3.4.5 순전파 과정 정리

다층 신경망의 순전파는 다음과 같은 순서로 수행된다.

```plaintext
입력(X)
   │
   ▼
XW₁ + b₁
   │
   ▼
ReLU
   │
   ▼
A₁W₂ + b₂
   │
   ▼

Sigmoid
   │
   ▼
최종 예측값(ŷ)
```

$$
Z_1 = XW_1 + b_1
$$

$$
A_1 = ReLU(Z_1)
$$

$$
Z_2 = A_1W_2 + b_2
$$

$$
\hat{Y} = Sigmoid(Z_2)
$$

여기서

- X : 입력 데이터
- W₁, W₂ : 각 층의 가중치
- b₁, b₂ : 편향
- A₁ : 은닉층 출력
- ŷ : 최종 예측값

을 의미한다.

순전파는 입력 데이터를 신경망의 앞쪽에서 뒤쪽으로 전달하면서 각 층에서 행렬 곱셈과 활성화 함수를 반복 수행하는 과정이다.

입력 데이터는 각 층을 통과하면서 점차 더 추상적인 특징으로 변환되며, 마지막 출력층에서는 하나의 예측값을 생성한다.

이 단계에서는 아직 가중치를 수정하지 않고 단순히 현재 가중치를 이용하여 결과를 계산한다. 이후 손실 함수를 이용하여 실제 정답과의 차이를 계산하고, 역전파를 통해 가중치를 수정하게 된다.

# 3.5 손실 함수(Loss Function) 구현

## 3.5.1 구현 목표

신경망은 순전파를 통해 예측값을 계산하지만, 예측값이 실제 정답과 얼마나 차이가 나는지는 알 수 없다. 따라서 신경망이 얼마나 잘 예측했는지를 판단하기 위한 기준이 필요하며, 이를 **손실 함수(Loss Function)** 라고 한다.

손실 함수는 예측값과 실제값의 차이를 하나의 숫자로 나타낸다. 이 값이 클수록 예측이 부정확하다는 의미이며, 신경망은 학습 과정에서 손실값을 최소화하는 방향으로 가중치를 수정한다.

이번 구현에서는 가장 기본적인 손실 함수인 **평균 제곱 오차(Mean Squared Error, MSE)** 를 사용하였다.

---

## 3.5.2 평균 제곱 오차(MSE)

평균 제곱 오차는 예측값과 실제값의 차이를 제곱한 뒤 평균을 계산하는 방식이다.

수식은 다음과 같다.

$$
MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y_i})^2
$$

여기서

- yᵢ : 실제 정답값
- ŷᵢ : 신경망의 예측값
- n : 데이터 개수

를 의미한다.

차이를 제곱하기 때문에 오차가 양수와 음수로 상쇄되지 않으며, 큰 오차에 대해 더 큰 패널티를 부여한다.

---

## 3.5.3 MSE 손실 함수 구현

### Loss.kt

```kotlin
object Loss {

    // 평균 제곱 오차(MSE)
    fun mse(
        prediction: Matrix,
        target: Matrix
    ): Double {

        var sum = 0.0

        for(row in 0 until prediction.rows){

            for(col in 0 until prediction.cols){

                val error =
                    prediction[row,col] - target[row,col]

                sum += error * error

            }

        }

        return sum /
                (prediction.rows * prediction.cols)

    }

}
```

이 구현에서는 흔히 사용되는 $\frac{1}{2}MSE$ 형태(제곱 오차 합에 1/2을 곱하는 방식)를 사용하지 않았다. 1/2을 곱하면 미분 시 계산이 조금 더 간단해지지만, 그 경우 3.6.3에서 계산할 Gradient 식도 함께 절반으로 바뀌어야 한다. 이 보고서에서는 손실 함수와 Gradient 식의 일관성을 명확히 유지하기 위해 1/2 없이 $\frac{1}{n}\sum(y-\hat y)^2$ 형태를 그대로 사용하였으며, 이에 대응하는 Gradient는 3.6.3에서 $\frac{2}{n}(\hat y - y)$로 구현하였다.

---

## 코드 설명

Loss 객체는 신경망의 예측값과 실제 정답값을 비교하여 손실값을 계산하는 역할을 한다.

먼저 예측값과 정답값의 차이를 구하고, 이를 제곱하여 모두 더한다. 이후 데이터 개수로 나누어 평균을 계산한다.

계산된 손실값은 역전파 과정에서 가중치를 수정하는 기준으로 사용된다.

---

## 3.5.4 손실 함수 실행 확인

손실 함수가 정상적으로 동작하는지 확인하기 위해 임의의 예측값과 실제값을 비교하였다.

### Main.kt

```kotlin
fun main(){

    val prediction = Matrix(1,3)

    prediction[0,0] = 0.8
    prediction[0,1] = 0.2
    prediction[0,2] = 0.6


    val target = Matrix(1,3)

    target[0,0] = 1.0
    target[0,1] = 0.0
    target[0,2] = 1.0


    val loss =
        Loss.mse(
            prediction,
            target
        )


    println(loss)

}
```

---

### 실행 결과

```text
0.08
```

---

## 코드 설명

위 코드에서는 신경망의 출력값이라고 가정한 prediction 행렬과 실제 정답값인 target 행렬을 비교하였다.

예측값과 실제값의 차이가 작기 때문에 손실값 역시 작은 값으로 계산되는 것을 확인할 수 있다.

반대로 예측값과 실제값의 차이가 커지면 MSE 값도 증가하게 되며, 신경망은 역전파를 통해 이 값을 감소시키도록 학습한다.

---

## 3.5.5 역전파에서의 손실 함수 활용

손실 함수는 신경망 학습의 방향을 결정하는 중요한 기준이다.

손실값이 크다는 것은 예측이 실제 정답과 많이 다르다는 의미이며, 손실값이 작을수록 예측 성능이 향상되었다고 볼 수 있다. 따라서 역전파와 경사하강법은 손실값을 최소화하는 방향으로 반복 수행된다.

MSE의 미분은 다음과 같다.

$$
\frac{\partial L}{\partial \hat{y}}=\frac{2}{n}(\hat{y}-y)
$$

이를 이용하여 출력층의 오차를 계산한다.

$$
\delta=(\hat{y}-y)\odot f'(z)
$$

여기서

- δ : 출력층 오차
- ŷ : 예측값
- y : 실제값
- ƒ’(z) : 활성화 함수의 미분값
- ⊙ : Hadamard 곱

이다.

계산된 오차는 이전 층으로 전달되어 가중치와 편향을 수정하는 데 사용된다.

---

## 3.5 정리

이번 절에서는 신경망의 예측 결과를 평가하기 위한 손실 함수를 구현하였다.

구현한 과정은 다음과 같다.

1. 순전파를 통해 예측값 계산
2. 예측값과 실제값 비교
3. MSE를 이용하여 손실값 계산
4. 손실값을 기반으로 역전파 수행

손실 함수는 신경망이 얼마나 잘 학습하고 있는지를 판단하는 기준이며, 이후 구현할 역전파 과정에서 가중치를 업데이트하는 핵심 요소로 사용된다.

# 3.6 역전파(Back Propagation) 구현

## 3.6.1 구현 목표

순전파를 수행하면 신경망은 입력 데이터를 이용하여 예측값을 계산할 수 있다. 그러나 예측값만 계산하는 것으로는 신경망이 학습되지 않는다.

신경망이 학습하기 위해서는 예측값과 실제 정답 사이의 오차를 계산하고, 그 오차를 이용하여 가중치와 편향을 수정해야 한다. 이러한 과정을 **역전파(Back Propagation)** 라고 한다.

역전파는 출력층에서 계산된 오차를 입력층 방향으로 전달하면서 각 층의 기울기(Gradient)를 계산하고, 이를 이용하여 가중치와 편향을 수정한다.

이번 절에서는 활성화 함수의 미분, 손실 함수의 Gradient 계산, Dense Layer의 역전파 구현을 통해 신경망이 학습하는 과정을 구현하였다.

---

## 3.6.2 활성화 함수 미분

역전파에서는 활성화 함수의 출력뿐만 아니라 **미분값**도 필요하다.

ReLU의 미분은 다음과 같다.

$$
ReLU'(x)=
\begin{cases}
1 & x>0\\
0 & x\le0
\end{cases}
$$

Sigmoid의 미분은 다음과 같다.

$$
\sigma'(x)=
\sigma(x)(1-\sigma(x))
$$

### Activation.kt

```kotlin
object Activation {

    fun relu(x: Double): Double {
        return max(0.0, x)
    }

    fun reluDerivative(x: Double): Double {
        return if (x > 0.0) 1.0 else 0.0
    }

    fun sigmoid(x: Double): Double {
        return 1.0 / (1.0 + exp(-x))
    }

    fun sigmoidDerivative(x: Double): Double {
        val s = sigmoid(x)
        return s * (1.0 - s)
    }

}
```

### 코드 설명

ReLU는 입력값이 0보다 크면 기울기가 1이고, 그렇지 않으면 0이 된다.

Sigmoid는 출력값을 이용하여 미분을 계산할 수 있으므로 계산량을 줄일 수 있으며, 역전파에서 가장 많이 사용되는 활성화 함수 중 하나이다.

두 함수 모두 DenseLayer가 생성될 때 activation, activationDerivative로 전달되어(3.4.2 참고), 은닉층은 relu / reluDerivative를, 출력층은 sigmoid / sigmoidDerivative를 사용하게 된다.

---

## 3.6.3 손실 함수의 Gradient 계산

손실 함수는 예측값과 실제 정답 사이의 오차를 계산한다.

역전파에서는 손실 함수 자체보다 손실 함수의 미분값이 필요하다.

평균 제곱 오차(MSE)의 Gradient는 다음과 같다.

$$
\frac{\partial L}{\partial \hat y}=\frac{2}{n}(\hat y-y)
$$

### Loss.kt

```kotlin
fun gradient(
    predicted: Matrix,
    target: Matrix
): Matrix {

    // n은 예측 행렬 전체의 원소 개수(행 × 열)이다.
    // Loss.mse()에서 sum을 (rows * cols)로 나눈 것과
    // 동일한 n을 사용해야 두 식이 서로 일치한다.
    val n = predicted.rows * predicted.cols

    return (predicted - target) * (2.0 / n)

}
```

### 코드 설명

Gradient는 손실값이 증가하는 방향을 의미한다.

신경망은 이 값을 이용하여 손실이 감소하는 방향으로 가중치를 수정한다.

이 식은 3.5.3에서 1/2을 사용하지 않은 MSE 정의와 짝을 이루는 미분식이다. 여기서 n은 3.5.2의 MSE 식에서 사용한 데이터 개수와 같은 의미로, `Loss.mse()`가 `sum / (prediction.rows * prediction.cols)`로 나눈 것과 동일하게 `predicted.rows * predicted.cols`를 사용하였다. 만약 출력층의 뉴런이 1개(cols = 1)라면 `predicted.rows`만 사용해도 결과가 같지만, 출력 뉴런이 여러 개인 경우에는 cols까지 포함해야 손실 함수와 Gradient가 정확히 같은 n을 기준으로 계산된다.

만약 손실 함수에 1/2을 곱하는 형태로 바꾼다면, 이 Gradient 식도 $\frac{1}{n}(\hat y - y)$로 함께 수정해야 두 식이 일치한다.

---

## 3.6.4 Dense Layer 역전파 구현

출력층에서 계산된 Gradient는 각 층으로 전달되며 가중치와 편향을 수정한다.

### DenseLayer.kt

```kotlin
fun backward(
    gradient: Matrix,
    learningRate: Double
): Matrix {

    // 활성화 함수의 미분은 활성화 적용 이전 값인 z를 기준으로 계산한다.
    // DenseLayer가 자신의 activationDerivative를 갖고 있으므로
    // 은닉층(ReLU)과 출력층(Sigmoid)이 각각 올바른 미분 함수를 사용한다.
    val activationGradient =
        z.map(activationDerivative)

    val delta = gradient.hadamard(activationGradient)

    val weightGradient =
        input.transpose() * delta

    weight -= weightGradient * learningRate

    bias -= delta * learningRate

    return delta * weight.transpose()

}
```

### 코드 설명

먼저 활성화 함수의 미분값을 계산한다. 이때 미분은 활성화 함수가 적용되기 전의 값인 z를 기준으로 계산해야 하므로, forward()에서 저장해 둔 z를 사용한다. 또한 이 미분 함수는 DenseLayer가 생성될 때 층마다 다르게 전달받은 activationDerivative이므로, 은닉층에서는 reluDerivative가, 출력층에서는 sigmoidDerivative가 자동으로 사용된다.

이렇게 구한 활성화 함수의 미분값(activationGradient)을 전달받은 gradient와 Hadamard 곱하여 이 층의 오차(delta)를 계산한다.

이후 delta에 입력값의 전치행렬(input.transpose())을 곱하여 Weight Gradient를 계산한다.

계산된 Gradient는 Learning Rate를 이용하여 가중치와 편향을 수정하며, 마지막에는 delta에 가중치의 전치행렬을 곱하여 이전 층으로 전달할 Gradient를 반환한다.

---

## 3.6.5 Learning Rate

Learning Rate는 한 번의 학습에서 가중치를 얼마나 수정할 것인지를 결정하는 값이다.

Learning Rate가 너무 크면 최적값을 지나쳐 학습이 불안정해질 수 있으며, 너무 작으면 학습 속도가 매우 느려진다.

이번 구현에서는 일정한 Learning Rate를 사용하여 모든 Epoch에서 동일한 크기로 가중치를 수정하였다.

---

## 3.6 정리

이번 절에서는 손실 함수에서 계산된 오차를 이용하여 역전파를 구현하였다.

활성화 함수의 미분을 이용하여 각 층의 Gradient를 계산하였으며, 계산된 Gradient는 경사하강법을 이용하여 가중치와 편향을 수정하는 데 사용하였다.

특히 은닉층과 출력층이 서로 다른 활성화 함수(ReLU, Sigmoid)를 사용하므로, DenseLayer가 자신의 activationDerivative를 직접 갖도록 구현하여 층마다 올바른 미분 함수가 적용되도록 하였다.

순전파가 예측값을 계산하는 과정이라면, 역전파는 신경망이 스스로 학습하는 핵심 과정이라고 할 수 있다.

# 3.7 학습 과정

## 3.7.1 구현 목표

역전파를 한 번 수행하는 것만으로는 충분한 학습이 이루어지지 않는다.

따라서 동일한 데이터를 여러 번 반복하여 순전파와 역전파를 수행하면서 가중치를 계속 수정해야 한다.

이번 절에서는 Epoch를 반복하며 신경망이 실제로 학습되는 과정을 구현하였다.

---

## 3.7.2 학습 구현

### NeuralNetwork.kt

```kotlin
fun train(
    input: Matrix,
    target: Matrix,
    learningRate: Double
): Double {

    val predicted = forward(input)

    val loss = Loss.mse(predicted, target)

    var gradient =
        Loss.gradient(predicted, target)

    for (i in layers.indices.reversed()) {
        gradient =
            layers[i].backward(
                gradient,
                learningRate
            )
    }

    return loss

}
```

### 코드 설명

학습은 먼저 순전파를 수행하여 예측값을 계산한다.

이후 손실 함수의 Gradient를 계산하고, 출력층부터 입력층 방향으로 역전파를 수행하여 모든 층의 가중치와 편향을 수정한다.

여기서 layers는 3.4.3에서 hiddenLayer와 outputLayer를 순서대로 담아둔 리스트이다. layers.indices.reversed()를 이용하여 출력층(마지막 층)부터 은닉층(첫 번째 층) 방향으로 역순으로 backward()를 호출함으로써, 역전파가 "출력층 → 입력층" 방향으로 진행된다는 2.8절의 설명을 그대로 코드로 구현하였다.

---

## 3.7.3 Epoch 반복

### Main.kt

```kotlin
repeat(1000) { epoch ->

    val loss =
        network.train(
            input,
            target,
            0.01
        )

    if (epoch % 100 == 0) {
        println(
            "Epoch ${epoch + 1} : Loss = $loss"
        )
    }

}
```

### 코드 설명

동일한 학습 데이터를 여러 번 반복하여 순전파와 역전파를 수행한다.

Epoch가 증가할수록 손실값은 점차 감소하며 신경망은 데이터의 특징을 학습하게 된다.

---

## 3.7.4 학습 결과

학습이 정상적으로 수행되면 다음과 같이 손실값이 점차 감소하는 것을 확인할 수 있다.

```text
Epoch 1 : Loss = 0.4577586638815728
Epoch 101 : Loss = 0.31759489815649694
Epoch 201 : Loss = 0.2169002651341717
Epoch 301 : Loss = 0.15145102687569295
Epoch 401 : Loss = 0.1083797164276344
Epoch 501 : Loss = 0.07941311257341716
Epoch 601 : Loss = 0.05947089886633156
Epoch 701 : Loss = 0.04542726135355349
Epoch 801 : Loss = 0.03532646166475626
Epoch 901 : Loss = 0.027918930658820514
```

손실값이 감소한다는 것은 예측값이 실제 정답에 가까워지고 있으며, 신경망이 데이터의 특징을 학습하고 있음을 의미한다.

학습 완료 후 입력 데이터를 신경망에 전달하면 다음과 같은 예측값을 얻을 수 있다.

```text
입력: 1.0 0.0 → 출력: 0.8504 (목표: 1.0)
```

학습 전에는 무작위 가중치에 의해 예측값이 출력되었지만, 학습을 거친 후에는 목표값에 가까운 예측값을 출력하는 것을 확인할 수 있다.

---

## 3.7 정리

이번 절에서는 순전파와 역전파를 반복 수행하여 신경망이 학습되는 과정을 구현하였다.

Epoch가 증가함에 따라 손실값이 지속적으로 감소하는 것을 확인할 수 있었으며, 이를 통해 구현한 신경망이 실제로 학습을 수행함을 확인하였다.

# 4. 실험 및 결과 분석

## 4.1 실험 환경

이번 구현을 실행한 환경은 다음과 같다.

| 항목 | 내용 |
| --- | --- |
| 언어 | Kotlin |
| 개발 환경 | IntelliJ IDEA |
| 신경망 구조 | 입력층(2) → 은닉층(4, ReLU) → 출력층(1, Sigmoid) |
| 손실 함수 | Mean Squared Error(MSE) |
| 최적화 방법 | Gradient Descent |
| Learning Rate | 0.01 |
| Epoch | 1000 |

외부 인공지능 라이브러리를 사용하지 않고 3장에서 직접 구현한 Matrix, Activation, Loss, DenseLayer, NeuralNetwork만을 이용하여 학습을 수행하였다.

---

## 4.2 학습 과정

3.7.3에서 1000번의 Epoch 동안 학습을 반복하면서 100 Epoch마다 손실값(Loss)을 출력하였다. 그 결과는 다음과 같다.

| Epoch | Loss |
| --- | --- |
| 1 | 0.4578 |
| 101 | 0.3176 |
| 201 | 0.2169 |
| 301 | 0.1515 |
| 401 | 0.1084 |
| 501 | 0.0794 |
| 601 | 0.0595 |
| 701 | 0.0454 |
| 801 | 0.0353 |
| 901 | 0.0279 |

Epoch가 증가할수록 손실값이 지속적으로 감소하였다. 이는 역전파와 경사하강법을 통해 가중치가 반복적으로 수정되면서 예측값이 실제 정답에 가까워졌음을 의미한다. 특히 학습 초반(Epoch 1~301)에는 손실값이 비교적 빠르게 감소하였고, 이후에는 감소 폭이 점차 줄어드는 경향을 보였다. 이는 예측값이 정답에 가까워질수록 남은 오차가 작아져 Gradient의 크기 또한 함께 작아지기 때문이다.

---

## 4.3 예측 결과 비교

학습이 신경망에 실제로 영향을 주었는지 확인하기 위해, 동일한 입력 데이터 `[1.0, 0.0]`에 대해 학습 전(순전파만 수행한 경우, 3.4.4)과 학습 후(1000 Epoch 학습을 마친 경우, 3.7.4)의 출력값을 비교하였다.

| 구분 | 출력 |
| --- | --- |
| 학습 전 | 0.1424 |
| 학습 후 | 0.8504 |
| 실제 정답 | 1.0000 |

학습 이전에는 가중치가 무작위로 초기화되어 있었기 때문에 예측값이 정답과 큰 차이를 보였지만, 1000번의 Epoch 동안 순전파와 역전파를 반복한 이후에는 예측값이 실제 정답인 1.0에 상당히 가까워진 것을 확인할 수 있었다. 이를 통해 신경망이 무작위로 예측한 것이 아니라, 역전파를 통해 데이터의 패턴을 실제로 학습했음을 확인할 수 있다.

---

## 4.4 결과 분석

이번 실험에서 손실값이 지속적으로 감소한 이유는, 역전파 과정에서 계산한 손실 함수의 Gradient를 이용하여 가중치가 손실이 감소하는 방향으로 반복적으로 수정되었기 때문이다. 매 Epoch마다 동일한 데이터를 다시 순전파하고, 그 결과로 발생한 오차를 다시 역전파하여 가중치를 조금씩 수정하는 과정을 거치면서, 신경망은 입력과 출력 사이의 관계를 점차 학습해 나갔다.

또한 이 과정에서 Learning Rate(학습률)의 영향을 함께 확인할 수 있었다. Learning Rate가 너무 크면 한 번의 갱신에서 가중치가 지나치게 크게 변화하여 손실값이 감소하지 않고 오히려 진동하거나 발산할 수 있다. 반대로 Learning Rate가 너무 작으면 손실값은 안정적으로 감소하지만, 가중치가 매우 조금씩만 수정되기 때문에 목표한 손실값에 도달하기까지 훨씬 더 많은 Epoch가 필요하게 된다. 이번 구현에서 사용한 Learning Rate 0.01은 손실값이 발산하지 않으면서도 1000 Epoch 이내에 유의미하게 감소하는 값이었다.

Epoch 수 역시 학습 결과에 직접적인 영향을 준다. Epoch가 너무 적으면 가중치가 충분히 수정되지 못해 손실값이 여전히 큰 상태로 학습이 종료되며, 반대로 Epoch를 지나치게 늘리면 이미 손실값이 충분히 낮아진 이후에도 불필요한 연산을 반복하게 된다. 이번 실험에서는 4.2의 표에서 볼 수 있듯이 Epoch가 진행될수록 손실값의 감소 폭이 점차 줄어드는 것으로 보아, 1000 Epoch 근처에서 학습이 어느 정도 수렴 단계에 접어들었음을 짐작할 수 있다.

종합하면, 이번 실험을 통해 순전파로 예측값을 계산하고, 손실 함수로 오차를 수치화하며, 역전파와 경사하강법으로 그 오차를 줄이는 방향으로 가중치를 수정하는 신경망의 학습 원리가 실제로 동작함을 직접 확인할 수 있었다.

---

## 4.5 한계점

이번 탐구에서 구현한 신경망은 인공지능의 학습 원리를 이해하기 위한 교육용 구현으로, 다음과 같은 한계점을 가진다.

- 하나의 학습 데이터(입력 `[1.0, 0.0]`, 목표값 `1.0`)만을 사용하여 학습을 수행하였다. 따라서 이번 결과만으로는 신경망이 일반적인 데이터에 대해서도 잘 동작하는지 평가하기 어렵다.
- 데이터셋의 크기가 매우 작아, 여러 데이터를 동시에 학습하는 배치(Batch) 학습을 고려하지 않았다. `DenseLayer`의 편향(Bias) 덧셈 역시 입력이 한 행(샘플 1개)인 경우만을 가정하고 구현되었다.
- GPU를 사용하지 않고 CPU에서 단순한 반복문으로 행렬 연산을 수행하였기 때문에, 대규모 데이터나 깊은 신경망에는 그대로 적용하기 어렵다.
- XOR 문제와 같이 선형적으로 분리되지 않는 조금 더 복잡한 문제에 대해서도 신경망이 정상적으로 학습하는지는 추가적인 실험이 필요하다.
- 실제 딥러닝 프레임워크에서는 Mini Batch, Adam, RMSProp과 같은 더 발전된 학습 방식과 최적화 알고리즘을 사용하지만, 이번 구현에서는 가장 기본적인 경사하강법(Gradient Descent)만을 사용하였다.

이러한 한계점에도 불구하고, 이번 구현을 통해 인공지능이 행렬 연산과 순전파·역전파를 기반으로 학습한다는 핵심 원리를 직접 코드로 확인할 수 있었다는 점에서 의미가 있다고 판단된다.

---

# 5. 결론

이번 탐구에서는 Kotlin만을 이용하여 인공지능 신경망의 핵심 학습 과정을 직접 구현하고, 인공지능이 데이터를 처리하는 원리를 행렬 연산을 중심으로 분석하였다. 이를 위해 Matrix 클래스를 직접 작성하여 행렬의 덧셈, 뺄셈, 곱셈, 전치 행렬, Hadamard 곱, map 함수 등을 구현하였으며, 이를 기반으로 ReLU와 Sigmoid 활성화 함수, 순전파(Forward Propagation), 손실 함수(MSE), 역전파(Back Propagation), 경사하강법을 이용한 학습 과정을 단계적으로 구현하였다.

구현 결과, 신경망은 입력 데이터를 그대로 처리하는 것이 아니라 행렬 곱셈과 활성화 함수를 반복 수행하여 예측값을 생성하고, 손실 함수를 통해 계산된 오차를 역전파하여 가중치를 지속적으로 수정한다는 점을 확인할 수 있었다. 또한 Epoch를 반복할수록 손실값(Loss)이 점차 감소하는 것을 통해, 구현한 신경망이 실제로 데이터를 학습하며 예측 성능을 개선한다는 사실을 확인하였다. 이를 통해 인공지능의 학습이 복잡한 과정처럼 보이지만, 실제로는 행렬 연산과 미분, 경사하강법이 유기적으로 결합된 수학적 계산 과정임을 이해할 수 있었다.

이번 탐구는 교육 목적에 맞추어 가장 기본적인 다층 퍼셉트론(MLP)을 구현한 것으로, 실제 인공지능 모델과는 차이가 있다. 실제 딥러닝에서는 수백에서 수천 개의 층을 가지는 신경망, GPU를 이용한 대규모 행렬 연산, Adam과 같은 고성능 최적화 알고리즘, Batch 학습, 정규화 기법 등 다양한 기술이 함께 사용된다. 그러나 이러한 고급 기술 역시 결국 이번 탐구에서 구현한 행렬 연산, 순전파, 손실 함수, 역전파를 기반으로 발전한 것이라는 점을 확인할 수 있었다.

이번 탐구를 통해 학교에서 배우는 행렬과 미분이 단순한 수학 개념이 아니라 인공지능 기술의 핵심 원리라는 사실을 이해할 수 있었으며, 프로그래밍을 통해 이를 직접 구현하면서 이론과 실제 구현의 연결 과정을 경험할 수 있었다. 앞으로는 다양한 활성화 함수와 손실 함수의 비교, 여러 은닉층을 사용하는 심층 신경망 구현, CNN이나 Transformer와 같은 최신 인공지능 모델의 구조와 학습 원리를 추가로 탐구하여 보다 실제 인공지능에 가까운 시스템을 구현해 보고자 한다.

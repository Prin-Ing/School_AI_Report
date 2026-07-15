class DenseLayer(
    inputSize: Int,
    outputSize: Int
) {

    // 가중치
    var weight = Matrix.random(inputSize, outputSize)

    // 편향
    var bias = Matrix.random(1, outputSize)

    // 순전파에서 저장할 값
    lateinit var input: Matrix
    lateinit var z: Matrix
    lateinit var output: Matrix

    fun forward(
        input: Matrix,
        activation: (Double) -> Double
    ): Matrix {

        this.input = input

        z = (input * weight) + bias

        output = z.map(activation)

        return output

    }

    fun backward(
        gradient: Matrix,
        learningRate: Double
    ): Matrix {

        val activationGradient =
            output.map {
                Activation.sigmoidDerivative(it)
            }

        val delta = gradient.hadamard(activationGradient)

        val weightGradient =
            input.transpose() * delta

        weight -= weightGradient * learningRate

        bias -= delta * learningRate

        return delta * weight.transpose()

    }

}

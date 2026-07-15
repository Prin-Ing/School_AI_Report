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

    fun backward(
        gradient: Matrix,
        learningRate: Double
    ): Matrix {

        val activationGradient =
            z.map(activationDerivative)

        val delta = gradient.hadamard(activationGradient)

        val weightGradient =
            input.transpose() * delta

        weight -= weightGradient * learningRate

        bias -= delta * learningRate

        return delta * weight.transpose()

    }

}

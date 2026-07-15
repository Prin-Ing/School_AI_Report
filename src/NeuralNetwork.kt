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

}

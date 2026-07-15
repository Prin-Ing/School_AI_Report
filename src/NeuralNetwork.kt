class NeuralNetwork {

    private val hiddenLayer =
        DenseLayer(2, 4)

    private val outputLayer =
        DenseLayer(4, 1)

    private val layers = listOf(hiddenLayer, outputLayer)

    fun forward(input: Matrix): Matrix {

        val hiddenOutput =
            hiddenLayer.forward(
                input,
                Activation::relu
            )

        return outputLayer.forward(
            hiddenOutput,
            Activation::sigmoid
        )

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

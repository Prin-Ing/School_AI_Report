fun main() {

    val network = NeuralNetwork()

    val input = Matrix(1, 2)

    input[0, 0] = 1.0
    input[0, 1] = 0.0

    val target = Matrix(1, 1)

    target[0, 0] = 1.0

    println("=== 학습 시작 ===")
    println("입력: ${input[0,0]} ${input[0,1]} | 목표: ${target[0,0]}")
    println()

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

    println()
    println("=== 학습 결과 ===")
    val output = network.forward(input)
    println("입력: ${input[0,0]} ${input[0,1]} → 출력: ${"%.4f".format(output[0,0])} (목표: ${target[0,0]})")

}

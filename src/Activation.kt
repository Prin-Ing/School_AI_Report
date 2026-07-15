import kotlin.math.max
import kotlin.math.exp

object Activation {

    // ReLU 함수
    fun relu(x: Double): Double {
        return max(0.0, x)
    }

    // Sigmoid 함수
    fun sigmoid(x: Double): Double {
        return 1.0 / (1.0 + exp(-x))
    }

    // ReLU 미분
    fun reluDerivative(x: Double): Double {

        return if (x > 0.0) 1.0 else 0.0

    }

    // Sigmoid 미분
    fun sigmoidDerivative(x: Double): Double {

        val s = sigmoid(x)

        return s * (1.0 - s)

    }

}

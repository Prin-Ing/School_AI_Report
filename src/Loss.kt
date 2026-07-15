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

    fun gradient(
        predicted: Matrix,
        target: Matrix
    ): Matrix {

        return (predicted - target) * (2.0 / predicted.rows)

    }

}

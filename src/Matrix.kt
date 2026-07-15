import kotlin.random.Random

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

    override fun toString(): String {

        val sb = StringBuilder()

        for(row in 0 until rows){

            for(col in 0 until cols){

                sb.append(data[row][col])

                if(col < cols - 1) sb.append("\t")

            }

            sb.append("\n")

        }

        return sb.toString()

    }

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

}

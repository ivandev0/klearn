package klearn.models.klearn.model

import klearn.backend.jvm.DoubleMatrix
import klearn.backend.jvm.DoubleVector
import klearn.linalg.Matrix
import klearn.linalg.Vector
import klearn.models.SVM
import klearn.models.klearn.backend.jvm.JvmMatrixTest
import org.junit.Assert
import org.junit.Before
import org.junit.Test
import java.io.BufferedReader
import java.io.FileReader
import kotlin.math.sign

class SvmTest {

    val csv = SimpleCsvReader("/src/main/resources/iris.csv")
    val exp = 1e-6

    @Test
    fun irisTest() {
        val expected = vectorOf(82.856486, -103.484448, -135.850070)
        val svm = SVM()
        svm.fit(csv.readFeaturesFromCSV(), csv.readLabelsFromCSV())
        Assert.assertEquals(expected[0], svm.w[0], exp)
        Assert.assertEquals(expected[1], svm.w[1], exp)
        Assert.assertEquals(expected[2], svm.w[2], exp)

        Assert.assertEquals(1, svm.predict(vectorOf(5.0, 2.5, 1.0))[0].sign.toInt())
        Assert.assertEquals(-1, svm.predict(vectorOf(5.0, 3.0, 1.0))[0].sign.toInt())
    }

    fun vectorOf(vararg x: Double): Vector<Double> {
        val elems = x.toList()
        val res = DoubleVector(elems.size, false)
        elems.forEachIndexed { index, value -> res.a[index] = value }
        return res
    }
}

class SimpleCsvReader(private val fileName: String) {
    fun readFeaturesFromCSV(): Matrix<Double> {
        BufferedReader(FileReader(System.getProperty("user.dir") + fileName)).use {
            val list: MutableList<Pair<Double, Double>> = mutableListOf()

            // Read CSV header
            it.readLine()

            // Read the file line by line starting from the second line
            var line: String? = it.readLine()
            while (line != null) {
                val tokens = line.split(",")
                if (tokens.isNotEmpty() && tokens[4] != "virginica") {
                    list.add(Pair(tokens[0].toDouble(),tokens[1].toDouble()))
                }
                line = it.readLine()
            }

            val matrix = DoubleMatrix(list.size, 2)
            for (i in 0 until list.size){
                matrix[i, 0] = list[i].first
                matrix[i, 1] = list[i].second
            }
            return matrix
        }
    }

    fun readLabelsFromCSV(): Vector<Double> {
        BufferedReader(FileReader(System.getProperty("user.dir") + fileName)).use {
            val list: MutableList<Double> = mutableListOf()

            // Read CSV header
            it.readLine()

            // Read the file line by line starting from the second line
            var line: String? = it.readLine()
            while (line != null) {
                val tokens = line.split(",")
                if (tokens.isNotEmpty() && tokens[4] != "virginica") {
                    list.add(if (tokens[4] == "versicolor") 1.0 else -1.0)
                }
                line = it.readLine()
            }

            val vector = DoubleVector(list.size, true)
            for (i in 0 until list.size){
                vector[i] = list[i]
            }
            return vector
        }
    }
}

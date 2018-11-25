package klearn.models

import klearn.*
import klearn.backend.jvm.DoubleVector
import klearn.linalg.Matrix
import klearn.linalg.Vector

class SVM(private val epochs: Int = 10000,
          private val eta: Double = 1.0) : Model {

    private lateinit var labels: Vector<Double>
    private lateinit var df: Matrix<Double>
    lateinit var w: Vector<Double>

    override fun fit(df: DataFrame, col: Column<*>) {
        TODO("not implemented")
    }

    override fun predict(data: DataFrame): Column<*> {
        TODO("not implemented")
    }

    fun predict(data: Matrix<Double>): Vector<Double> {
        val result = DoubleVector(data.dim.rows, true)
        for (i in 0 until data.dim.rows){
            result[i] = w.dot(data.row(i))
        }
        return result
    }

    fun fit(df: Matrix<Double>, labels: Vector<Double>) {
        this.labels = labels
        this.df = df.cbind(ones(labels.size))
        //Init weights
        w = DoubleVector(this.df.dim.cols, false)
        // Update weights
        w = trainEpochs(w, epochs)
    }

    private fun trainEpochs(w: Vector<Double>, epochs: Int, epochCount: Int = 1): Vector<Double> {
        if(epochs == 0)
            return w
        return trainEpochs(trainOneEpoch(w,0, epochCount), epochs - 1, epochCount + 1)
    }

    private fun trainOneEpoch(w: Vector<Double>, currentIndex: Int, epoch: Int): Vector<Double> {
        if(df.dim.rows == currentIndex)
            return w
        return if (misClassification(w, df.row(currentIndex), labels[currentIndex])){
            trainOneEpoch(gradient(w, df.row(currentIndex), labels[currentIndex], epoch), currentIndex + 1, epoch)
        } else {
            trainOneEpoch(regularizationGradient(w, epoch), currentIndex + 1, epoch)
        }
    }

    // Will only be called if classification is wrong.
    private fun gradient(w: Vector<Double>, data: Vector<Double>, label: Double, epoch: Int): Vector<Double> {
        val matrixW = w * (-2 * (1.0 / epoch))
        val matrixData = data * label
        val result = (matrixW + matrixData) * eta + w
        return if (result.dim.cols == 1) result.col(0) else result.row(0)
    }

    private fun regularizationGradient(w: Vector<Double>, epoch: Int): Vector<Double> {
        val matrix = w + w * (eta * (-2  * (1 / epoch)))
        return if (matrix.dim.cols == 1) matrix.col(0) else matrix.row(0)
    }

    // Misclassification threshold.
    private fun misClassification(w: Vector<Double>, x: Vector<Double>,  label: Double): Boolean {
        return x.dot(w) * label < 1
    }

    private fun ones(n: Int): Vector<Double> {
        val arr = DoubleArray(n) { 1.0 }
        return DoubleVector(n, true, arr)
    }

}


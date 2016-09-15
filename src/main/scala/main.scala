import com.neuralnetwork.Network
import org.apache.spark.mllib.linalg.{Vectors,Vector}

object Main {
  def main(args: Array[String]) {
    val n = new Network(Array(1,3,1), learningRate=0.001)

    val feats = (for (i <- -50 to 49) yield Vectors.dense(i / 100.0)).toArray
    val targs = (for (i <- -50 to 49)
                  yield Vectors.dense((Math.pow(i, 3) - (2 * Math.pow(i, 2)) + 4) / 10000.0)).toArray

    n.fit(feats, targs, iterations=10000)

    feats foreach { feat =>
      val predicted = n.predict(feat)
      println(predicted.apply(0) * 10000)
    }

  }
}

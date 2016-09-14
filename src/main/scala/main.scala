import com.neuralnetwork.Network
import org.apache.spark.mllib.linalg.{Vectors,Vector}

object Main {
  def main(args: Array[String]) {
    val n = new Network(Array(1,2,1), learningRate=0.001)

    val feats = (for (i <- 0 to 99) yield Vectors.dense(i / 100.0)).toArray
    val targs = (for (i <- 0 to 99) yield Vectors.dense(Math.pow(i, 2) / 1000.0)).toArray

    println(n)
    n.fit(feats, targs, iterations=10000)

    feats foreach { feat =>
      val predicted = n.predict(feat)
      println(predicted.apply(0) * 1000)
    }

  }
}

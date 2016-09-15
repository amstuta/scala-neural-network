package com.neuralnetwork

import org.apache.spark.mllib.linalg.{Vectors,Vector}
import org.apache.spark.mllib.linalg.{Matrices,Matrix}


class Network(sizes: Array[Int],
              private val learningRate: Double=0.01)
{

  private val layers = sizes.zipWithIndex map { case (elem, idx) =>
    idx match {
      case 0                              => new Layer(elem, sizes(idx + 1), learningRate, inputLayer=true)
      case i if (idx == sizes.length - 1) => new Layer(elem, learningRate=learningRate, outputLayer=true)
      case _                              => new Layer(elem, sizes(idx + 1), learningRate)
    }
  }


  def fit(features: Array[Vector], targets: Array[Vector], iterations: Int=1000) = {
    0 to iterations foreach { i =>
      features.zip(targets) foreach {
        case (f, t) => updateWeights(f, t)
      }
    }
  }


  def predict(features: Vector): Vector =
    feedForward(features)


  private def feedForward(features: Vector) = {
    var inputs = features
    var prev = layers.head

    prev set inputs
    layers.slice(1, layers.length) foreach { layer =>
      inputs = layer.feed(inputs, prev)
      prev = layer
    }
    inputs
  }


  private def updateWeights(feature: Vector, target: Vector) = {
    val output = feedForward(feature)
    val errors = layers.last.update(target)

    layers.slice(0, layers.length - 1).reverse.foldLeft(errors) {
      case (err, layer) => layer.update(target, Some(err))
    }
  }


  override def toString =
    layers.map(_.toString).reduce(_+_)

}

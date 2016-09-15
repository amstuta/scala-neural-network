package com.neuralnetwork

import org.apache.spark.mllib.linalg.{Vectors,Vector}
import org.apache.spark.mllib.linalg.{Matrices,Matrix}


/**
 * Represents the neural network.
 *
 * @param sizes Array representing the number of layers in the network and
 *              their size
 * @param learningRate The learning rate / alpha of the network
 *
 * Example of usage
 *  {{{
 *  scala> import com.neuralnetwork.Network
 *  scala> val net = new Network(Array(1,2,1), 0.001)
 *  net: => 1 neuron(s)=> 2 neuron(s)=> 1 neuron(s)
 *  }}}
 */
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


  /**
   * Trains the neural network with the provided inputs and outputs
   *
   * @param features   Array containing the inputs of the network. The number of
   *                   values in each Vector must correspond to the number of
   *                   input neurons
   * @param targets    Array containing the outputs of the network. The number of
   *                   values in each Vector must correspond to the number of
   *                   output neurons
   * @param iterations The number of iterations to perform to train the network
   */
  def fit(features: Array[Vector], targets: Array[Vector], iterations: Int=1000) = {
    0 to iterations foreach { i =>
      features.zip(targets) foreach {
        case (f, t) => updateWeights(f, t)
      }
    }
  }


  /**
   * Makes a prediction given an input Vector
   *
   * @param features  Vector containing the input values. It must contain the
   *                  same number of values that the number of input neurons
   * @return          A Vector containing the prediction
   */
  def predict(features: Vector): Vector =
    feedForward(features)


  /**
   * Propagates an input in all the layers and returns the output
   *
   * @param features  Vector containing the input values
   * @return          A vector containing the prediction
   */
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


  /**
   * Used during the training to update the weights in the network
   *
   * @param feature Input of the network
   * @param target  Output corresponding to the input given
   */
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

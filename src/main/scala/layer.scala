package com.neuralnetwork

import scala.util.control.Exception
import scala.util.Random
import org.apache.spark.mllib.linalg.{Vectors,Vector}
import org.apache.spark.mllib.linalg.{Matrices,Matrix}


/**
 * Provides the activation functions contained in a neuron
 */
object Neuron {

  def sigmoid(x: Vector) =
    Vectors.dense( x.toArray.map { elem => 1.0 / (1.0 + Math.exp(-elem)) }  )

  def sigmoidPrime(x: Vector) =
    Vectors.dense(x.toArray map { elem =>
      val sig = 1.0 / (1.0 + Math.exp(-elem))
      sig * (1.0 - sig)
    })

  def identity(x: Vector) = x.copy

  def identityPrime(x: Vector) =
    Vectors.dense((for (i <- 0 to x.size) yield 1.0).toArray)

}


/**
 * Class representing a layer of neurons in a neural network
 *
 * @param size          Number of neuron in this layer
 * @param learningRate  The learning rate to use for training
 * @param inputLayer    Boolean that says if this layer is the input layer
 * @param outputLayer   Boolean that says if this layer if the output layer
 */
class Layer(size: Int, nextSize: Int=0,
            private val learningRate: Double=0.01,
            private val inputLayer: Boolean=false,
            private val outputLayer: Boolean=false)
{
  import com.mathutils.MathUtils._


  private var (weights, bias) = outputLayer match {
    case true  => (Matrices.zeros(1, 1), Vectors.zeros(1))
    case false =>
      (Matrices.dense(nextSize, size, (for (i <- 0 to (nextSize * size - 1))
          yield Random.nextDouble).toArray),
        Vectors.dense((for (i <- 0 to (nextSize - 1))
          yield Random.nextDouble).toArray))
  }

  private var inputs = Vectors.zeros(size)
  private var outputs = Vectors.zeros(size)
  private var errors = Vectors.zeros(size)

  private val activation = if (outputLayer) Neuron.identity(_) else Neuron.sigmoid(_)
  private val derivate = if (outputLayer) Neuron.identityPrime(_) else Neuron.sigmoidPrime(_)


  /**
   * Feeds this layer with the inputs from the previous layer and returns
   * the output
   *
   * @param features  Outputs from the previous layer
   * @param prevLayer The previous layer in the network
   * @return          Outputs of this layer
   */
  def feed(features: Vector, prevLayer: Layer): Vector = {
    val (prevW, prevB) = prevLayer.get

    inputs = prevW.multiply(features) + prevB
    outputs = activation(inputs)
    outputs
  }


  /**
   * Updates the weights, bias and errors contained in this layer. If this is
   * the output layer, juste computes the initial error, else it uses the
   * backpropagation algorithm to adjust the weights and biases
   *
   * @param target    Used only by the output layer to compute the network error
   * @param nextError Used in all layers except the output layer. It is the
   *                  error propagated from the following layer in the network
   * @return          The error of this layer
   */
  def update(target: Vector, nextError: Option[Vector]=None): Vector = {

    if (outputLayer) {
      errors = derivate(outputs).elementProd(outputs - target)
    }
    else {
      nextError match {
        case Some(err) =>
          if (!inputLayer) {
            errors = weights.transpose.multiply(err).elementProd(derivate(inputs))
          }
          weights -= err.outerProd(outputs) * learningRate
          bias -= err * learningRate

        case None      => throw new Exception("Missing error from next layer")
      }
    }
    errors
  }


  /**
   * Returns the weights and bias contained in this layer
   */
  def get = (weights, bias)


  /**
   * Sets the inputs and outputs of this layer.
   * This function is used only to set the values of the input layer
   */
  def set(input: Vector) = {
    inputs = input
    outputs = inputs
  }

  override def toString = s"=> ${outputs.size} neuron(s)"

}

package com.neuralnetwork

import scala.util.control.Exception
import scala.util.Random
import org.apache.spark.mllib.linalg.{Vectors,Vector}
import org.apache.spark.mllib.linalg.{Matrices,Matrix}


object Layer {

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

  private val activation = if (outputLayer) Layer.identity(_) else Layer.sigmoid(_)
  private val derivate = if (outputLayer) Layer.identityPrime(_) else Layer.sigmoidPrime(_)


  def feed(features: Vector, prevLayer: Layer): Vector = {
    val (prevW, prevB) = prevLayer.get

    inputs = prevW.multiply(features) + prevB
    outputs = activation(inputs)
    outputs
  }


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


  def get = (weights, bias)

  def set(input: Vector) = {
    inputs = input
    outputs = inputs
  }

  override def toString = s"=> ${outputs.size} neuron(s)"

}

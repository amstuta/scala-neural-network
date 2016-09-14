package com.neuralnetwork

import scala.util.control.Exception
import scala.util.Random
import org.apache.spark.mllib.linalg.{Vectors,Vector}
import org.apache.spark.mllib.linalg.{Matrices,Matrix}


object RealVector {

  def add(lhs:Vector, rhs: Vector) =
    Vectors.dense(lhs.toArray.zip(rhs.toArray) map {
      case (l,r)  => l + r
      case _      => throw new Exception("Can't perform vectors addition: sizes mismatch")
    })


  def min(lhs: Vector, rhs: Vector) =
    Vectors.dense(lhs.toArray.zip(rhs.toArray) map {
      case (l,r)  => l - r
      case _      => throw new Exception("Can't perform vectors substraction: sizes mismatch")
    })


  def mul(lhs: Vector, rhs: Double) =
    Vectors.dense(lhs.toArray map { e => e * rhs })


  def elementProd(lhs: Vector, rhs: Vector) =
    Vectors.dense(lhs.toArray.zip(rhs.toArray) map {
      case (l,r)   => l * r
      case _       => throw new Exception("Can't perform vectors element product: sizes mismatch")
    })


  def outerProd(lhs: Vector, rhs: Vector): Matrix =
    Matrices.dense(lhs.size, rhs.size,
      (rhs.toArray.map { r =>
        lhs.toArray.map { l => l * r }
      }).flatten
    )
}


object RealMatrix {

  def min(lhs: Matrix, rhs: Matrix) =
    Matrices.dense(lhs.numRows, lhs.numCols,
      lhs.toArray.zip(rhs.toArray) map {
      case (l,r) => l - r
      case _     => throw new Exception("Cant substract two matrices with different dimensions")
    })


    def mul(lhs: Matrix, rhs: Double) =
      Matrices.dense(lhs.numRows, lhs.numCols,
        lhs.toArray map { elem => elem * rhs }
      )

}


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
  import RealVector._
  import RealMatrix.{min=>mmin, mul=>mmul}

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

    inputs = add(prevW.multiply(features), prevB)
    outputs = activation(inputs)
    outputs
  }


  def update(target: Vector, nextError: Option[Vector]=None): Vector = {

    if (outputLayer) {
      errors = elementProd(derivate(outputs), min(outputs, target))
    }
    else {
      nextError match {
        case Some(err) =>
          if (!inputLayer) {
            errors = elementProd(weights.transpose.multiply(err), derivate(inputs))
          }
          weights = mmin(weights, mmul(outerProd(err, outputs), learningRate))
          bias = min(bias, mul(err, learningRate))

        case None      => throw new Exception("Missing error from next layer")
      }
    }
    errors
  }


  def set(input: Vector) = {
    inputs = input
    outputs = inputs
  }


  def get = (weights, bias)


  override def toString = s"=> ${outputs.size} neuron(s)"

}

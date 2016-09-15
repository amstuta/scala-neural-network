package com.mathutils
/**
 * Provides implicit classes to add operations to Matix and Vector classes
 */

import scala.util.control.Exception
import org.apache.spark.mllib.linalg.{Vectors,Vector}
import org.apache.spark.mllib.linalg.{Matrices,Matrix}


object MathUtils {

  implicit class VectorUtils(val v: Vector) {

    def +(rhs: Vector) =
      Vectors.dense(v.toArray.zip(rhs.toArray) map {
        case (l,r)  => l + r
        case _      => throw new Exception("Can't perform vectors addition: sizes mismatch")
      })


    def -(rhs: Vector) =
      Vectors.dense(v.toArray.zip(rhs.toArray) map {
        case (l,r)  => l - r
        case _      => throw new Exception("Can't perform vectors substraction: sizes mismatch")
      })


    def *(rhs: Double) =
      Vectors.dense(v.toArray map { e => e * rhs })


    def elementProd(rhs: Vector) =
      Vectors.dense(v.toArray.zip(rhs.toArray) map {
        case (l,r)   => l * r
        case _       => throw new Exception("Can't perform vectors element product: sizes mismatch")
      })


    def outerProd(rhs: Vector): Matrix =
      Matrices.dense(v.size, rhs.size,
        (rhs.toArray.map { r =>
          v.toArray.map { l => l * r }
        }).flatten
      )
  }


  implicit class MatrixUtils(val m: Matrix) {

    def -(rhs: Matrix) =
      Matrices.dense(m.numRows, m.numCols,
        m.toArray.zip(rhs.toArray) map {
          case (l,r) => l - r
          case _     => throw new Exception("Cant substract two matrices with different dimensions")
      })


    def *(rhs: Double) =
      Matrices.dense(m.numRows, m.numCols,
        m.toArray map { elem => elem * rhs }
      )
  }

}

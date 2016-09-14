name := """play-getting-started"""

version := "1.0-SNAPSHOT"

scalaVersion := "2.10.6"

libraryDependencies ++= Seq(
  "org.apache.spark" % "spark-core_2.10" % "2.0.0",
  "org.apache.spark" % "spark-mllib_2.10" % "2.0.0"
)

libraryDependencies <+= scalaVersion("org.scala-lang" % "scala-compiler" % _ )

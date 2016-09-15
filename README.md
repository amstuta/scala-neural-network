# Feed forward neural network written in Scala

The main function contains a simple test that makes the network learn the ```f(x) = x^3 - 2x^2 + 4``` function.
After the training, the program prints out the results of predicting values from -50 to 50.

To compile and run, just type ```sbt run```.

Dependencies:
  - org.apache.spark.mllib.linalg


## Result plot example

![alt tag](https://raw.githubusercontent.com/amstuta/scala_neural_network/master/resources/plot.png)

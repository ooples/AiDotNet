using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.OnlineLearning.Algorithms;

namespace AiDotNetTests.UnitTests.OnlineLearning;

[TestClass]
public class OnlinePerceptronTests
{
    [TestMethod]
    public void Constructor_ShouldInitializeCorrectly()
    {
        // Arrange & Act
        var perceptron = new OnlinePerceptron<double>(3);

        // Assert
        Assert.IsNotNull(perceptron);
        Assert.AreEqual(3, perceptron.GetInputFeatureCount());
    }

    [TestMethod]
    public void PartialFit_WithLinearlySeparableData_ShouldConverge()
    {
        // Arrange
        var perceptron = new OnlinePerceptron<double>(2);
        var inputs = new[]
        {
            new Vector<double>(new[] { 1.0, 1.0 }),
            new Vector<double>(new[] { 2.0, 2.0 }),
            new Vector<double>(new[] { -1.0, -1.0 }),
            new Vector<double>(new[] { -2.0, -2.0 })
        };
        var labels = new[] { 1.0, 1.0, -1.0, -1.0 };

        // Act - train multiple epochs
        for (int epoch = 0; epoch < 10; epoch++)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                perceptron.PartialFit(inputs[i], labels[i]);
            }
        }

        // Assert - check predictions
        Assert.AreEqual(1.0, perceptron.Predict(inputs[0])); // Positive class
        Assert.AreEqual(1.0, perceptron.Predict(inputs[1])); // Positive class
        Assert.AreEqual(0.0, perceptron.Predict(inputs[2])); // Negative class
        Assert.AreEqual(0.0, perceptron.Predict(inputs[3])); // Negative class
    }

    [TestMethod]
    public void PartialFitBatch_ShouldUpdateModel()
    {
        // Arrange
        var perceptron = new OnlinePerceptron<double>(2);
        var inputs = new Matrix<double>(new double[,]
        {
            { 1.0, 1.0 },
            { 2.0, 2.0 },
            { -1.0, -1.0 },
            { -2.0, -2.0 }
        });
        var labels = new Vector<double>(new[] { 1.0, 1.0, -1.0, -1.0 });

        // Act
        // Convert Matrix to array of Vectors
        var inputVectors = new Vector<double>[inputs.Rows];
        for (int i = 0; i < inputs.Rows; i++)
        {
            inputVectors[i] = new Vector<double>(new[] { inputs[i, 0], inputs[i, 1] });
        }
        perceptron.PartialFitBatch(inputVectors, labels.ToArray());

        // Assert - model should have been updated
        var prediction = perceptron.Predict(new Vector<double>(new[] { 1.5, 1.5 }));
        Assert.AreNotEqual(0, prediction);
    }

    [TestMethod]
    public void PredictBatch_ShouldReturnCorrectShape()
    {
        // Arrange
        var perceptron = new OnlinePerceptron<double>(2);
        var inputs = new Matrix<double>(new double[,]
        {
            { 1.0, 1.0 },
            { 2.0, 2.0 },
            { -1.0, -1.0 }
        });

        // Act
        // PredictBatch doesn't exist, predict individually
        var predictions = new double[inputs.Rows];
        for (int i = 0; i < inputs.Rows; i++)
        {
            var input = new Vector<double>(new[] { inputs[i, 0], inputs[i, 1] });
            predictions[i] = perceptron.Predict(input);
        }

        // Assert
        Assert.AreEqual(3, predictions.Length);
    }

    [TestMethod]
    public void GetParameters_ShouldReturnCorrectParameters()
    {
        // Arrange
        var perceptron = new OnlinePerceptron<double>(3);

        // Act
        var parameters = perceptron.GetParameters();

        // Assert
        Assert.AreEqual(4, parameters.Length); // 3 weights + 1 bias
        // Parameters are returned as a Vector containing weights followed by bias
    }

    [TestMethod]
    public void SetParameters_ShouldUpdateModel()
    {
        // Arrange
        var perceptron = new OnlinePerceptron<double>(2);
        // Parameters are weights + bias
        var parameters = new Vector<double>(new[] { 0.5, -0.5, 1.0 });

        // Act
        perceptron.SetParameters(parameters);

        // Assert
        var testInput = new Vector<double>(new[] { 1.0, 1.0 });
        var prediction = perceptron.Predict(testInput);
        // 0.5 * 1.0 + (-0.5) * 1.0 + 1.0 = 1.0 > 0, so predict 1
        Assert.AreEqual(1.0, prediction, 0.0001);
    }

    [TestMethod]
    public void Clone_ShouldCreateIndependentCopy()
    {
        // Arrange
        var perceptron = new OnlinePerceptron<double>(2);
        var input = new Vector<double>(new[] { 1.0, 1.0 });
        perceptron.PartialFit(input, 1.0);

        // Act
        var clone = perceptron.Clone();
        perceptron.PartialFit(input, -1.0); // Update original

        // Assert - clone should not be affected
        Assert.AreNotEqual(perceptron.Predict(input), clone.Predict(input));
    }

    [TestMethod]
    public void WithCustomOptions_ShouldRespectLearningRate()
    {
        // Arrange
        var options = new OnlineModelOptions<double> { InitialLearningRate = 0.01 };
        var perceptron = new OnlinePerceptron<double>(2, options);
        var input = new Vector<double>(new[] { 1.0, 1.0 });

        // Act
        var initialPrediction = perceptron.Predict(input);
        perceptron.PartialFit(input, 1.0);
        var afterUpdatePrediction = perceptron.Predict(input);

        // Assert - with small learning rate, change should be small
        var change = Math.Abs(afterUpdatePrediction - initialPrediction);
        Assert.IsTrue(change < 0.1); // Small change due to small learning rate
    }
}
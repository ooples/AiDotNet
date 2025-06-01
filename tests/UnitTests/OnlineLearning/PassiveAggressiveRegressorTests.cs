using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.OnlineLearning.Algorithms;

namespace AiDotNetTests.UnitTests.OnlineLearning;

[TestClass]
public class PassiveAggressiveRegressorTests
{
    [TestMethod]
    public void Constructor_ShouldInitializeCorrectly()
    {
        // Arrange & Act
        var regressor = new PassiveAggressiveRegressor<double>(3);

        // Assert
        Assert.IsNotNull(regressor);
        Assert.AreEqual(3, regressor.GetInputFeatureCount());
    }

    [TestMethod]
    public void PartialFit_WithLinearData_ShouldFitWell()
    {
        // Arrange
        var regressor = new PassiveAggressiveRegressor<double>(1);
        var inputs = new[]
        {
            new Vector<double>(new[] { 1.0 }),
            new Vector<double>(new[] { 2.0 }),
            new Vector<double>(new[] { 3.0 }),
            new Vector<double>(new[] { 4.0 })
        };
        var outputs = new[] { 2.0, 4.0, 6.0, 8.0 }; // y = 2x

        // Act - train multiple epochs
        for (int epoch = 0; epoch < 50; epoch++)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                regressor.PartialFit(inputs[i], outputs[i]);
            }
        }

        // Assert - check predictions
        Assert.AreEqual(10.0, regressor.Predict(new Vector<double>(new[] { 5.0 })), 1);
        Assert.AreEqual(12.0, regressor.Predict(new Vector<double>(new[] { 6.0 })), 1);
    }

    [TestMethod]
    public void PartialFitBatch_ShouldUpdateModel()
    {
        // Arrange
        var regressor = new PassiveAggressiveRegressor<double>(2);
        var inputs = new Matrix<double>(new double[,]
        {
            { 1.0, 0.0 },
            { 0.0, 1.0 },
            { 1.0, 1.0 },
            { 2.0, 2.0 }
        });
        var outputs = new Vector<double>(new[] { 1.0, 2.0, 3.0, 6.0 });

        // Act
        for (int i = 0; i < 10; i++)
        {
            // Convert Matrix to array of Vectors
            var inputVectors = new Vector<double>[inputs.Rows];
            for (int j = 0; j < inputs.Rows; j++)
            {
                inputVectors[j] = new Vector<double>(new[] { inputs[j, 0], inputs[j, 1] });
            }
            regressor.PartialFitBatch(inputVectors, outputs.ToArray());
        }

        // Assert
        var prediction = regressor.Predict(new Vector<double>(new[] { 1.5, 1.5 }));
        Assert.IsTrue(Math.Abs(prediction - 4.5) < 1.0); // Should be close to 4.5
    }

    [TestMethod]
    public void WithCustomC_ShouldAffectRegularization()
    {
        // Arrange
        var options1 = new OnlineModelOptions<double> { AggressivenessParameter = 0.1 };
        var options2 = new OnlineModelOptions<double> { AggressivenessParameter = 10.0 };
        var regressor1 = new PassiveAggressiveRegressor<double>(2, options1);
        var regressor2 = new PassiveAggressiveRegressor<double>(2, options2);

        var input = new Vector<double>(new[] { 1.0, 1.0 });
        var target = 5.0;

        // Act - same update on both
        regressor1.PartialFit(input, target);
        regressor2.PartialFit(input, target);

        // Assert - different regularization should lead to different predictions
        var pred1 = regressor1.Predict(input);
        var pred2 = regressor2.Predict(input);
        Assert.AreNotEqual(pred1, pred2);
    }

    [TestMethod]
    public void Clone_ShouldCreateIndependentCopy()
    {
        // Arrange
        var regressor = new PassiveAggressiveRegressor<double>(2);
        var input = new Vector<double>(new[] { 1.0, 2.0 });
        regressor.PartialFit(input, 3.0);

        // Act
        var clone = regressor.Clone() as PassiveAggressiveRegressor<double>;
        regressor.PartialFit(input, 5.0); // Update original

        // Assert
        Assert.IsNotNull(clone);
        Assert.AreNotEqual(regressor.Predict(input), clone.Predict(input));
    }

    [TestMethod]
    public void GetSetParameters_ShouldWorkCorrectly()
    {
        // Arrange
        var regressor = new PassiveAggressiveRegressor<double>(2);
        // Parameters are weights + bias
        var parameters = new Vector<double>(new[] { 1.5, -0.5, 2.0 });

        // Act
        regressor.SetParameters(parameters);
        var retrievedParams = regressor.GetParameters();

        // Assert
        Assert.IsNotNull(retrievedParams);
        Assert.AreEqual(3, retrievedParams.Length); // 2 weights + 1 bias
        Assert.AreEqual(1.5, retrievedParams[0]);
        Assert.AreEqual(-0.5, retrievedParams[1]);
        Assert.AreEqual(2.0, retrievedParams[2]); // bias
    }

    [TestMethod]
    public void WithEpsilon_ShouldAffectInsensitiveLoss()
    {
        // Arrange
        var options = new OnlineModelOptions<double> { Epsilon = 0.5 };
        var regressor = new PassiveAggressiveRegressor<double>(1, options);
        var input = new Vector<double>(new[] { 1.0 });

        // Act - update with small error (within epsilon)
        var initialPred = regressor.Predict(input);
        regressor.PartialFit(input, initialPred + 0.3); // Error < epsilon
        var afterPred = regressor.Predict(input);

        // Assert - should not update much due to epsilon-insensitive loss
        Assert.AreEqual(initialPred, afterPred, 0.000001);
    }
}
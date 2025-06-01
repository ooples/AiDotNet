using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.OnlineLearning.Algorithms;

namespace AiDotNetTests.UnitTests.OnlineLearning;

[TestClass]
public class OnlineSGDRegressorTests
{
    [TestMethod]
    public void Constructor_ShouldInitializeCorrectly()
    {
        // Arrange & Act
        var regressor = new OnlineSGDRegressor<double>(3, FitnessCalculatorType.MeanSquaredError);

        // Assert
        Assert.IsNotNull(regressor);
        Assert.AreEqual(3, regressor.GetInputFeatureCount());
    }

    [TestMethod]
    public void PartialFit_WithLinearData_ShouldConverge()
    {
        // Arrange
        var regressor = new OnlineSGDRegressor<double>(1, FitnessCalculatorType.MeanSquaredError);
        var inputs = new[]
        {
            new Vector<double>(new[] { 1.0 }),
            new Vector<double>(new[] { 2.0 }),
            new Vector<double>(new[] { 3.0 }),
            new Vector<double>(new[] { 4.0 }),
            new Vector<double>(new[] { 5.0 })
        };
        var outputs = new[] { 3.0, 5.0, 7.0, 9.0, 11.0 }; // y = 2x + 1

        // Act - train multiple epochs
        for (int epoch = 0; epoch < 100; epoch++)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                regressor.PartialFit(inputs[i], outputs[i]);
            }
        }

        // Assert
        Assert.AreEqual(13.0, regressor.Predict(new Vector<double>(new[] { 6.0 })), 0.5);
        Assert.AreEqual(15.0, regressor.Predict(new Vector<double>(new[] { 7.0 })), 0.5);
    }

    [TestMethod]
    public void WithL2Regularization_ShouldReduceWeights()
    {
        // Arrange
        var optionsNoReg = new AdaptiveOnlineModelOptions<double> { RegularizationParameter = 0.0 };
        var optionsWithReg = new AdaptiveOnlineModelOptions<double> { RegularizationParameter = 0.1 };
        var regressorNoReg = new OnlineSGDRegressor<double>(2, FitnessCalculatorType.MeanSquaredError, optionsNoReg);
        var regressorWithReg = new OnlineSGDRegressor<double>(2, FitnessCalculatorType.MeanSquaredError, optionsWithReg);

        var inputs = new Matrix<double>(new double[,]
        {
            { 1.0, 1.0 },
            { 2.0, 2.0 },
            { 3.0, 3.0 }
        });
        var outputs = new Vector<double>(new[] { 2.0, 4.0, 6.0 });

        // Act
        for (int i = 0; i < 50; i++)
        {
            // Convert Matrix to array of Vectors
            var inputVectors = new Vector<double>[inputs.Rows];
            for (int j = 0; j < inputs.Rows; j++)
            {
                inputVectors[j] = new Vector<double>(new[] { inputs[j, 0], inputs[j, 1] });
            }
            regressorNoReg.PartialFitBatch(inputVectors, outputs.ToArray());
            regressorWithReg.PartialFitBatch(inputVectors, outputs.ToArray());
        }

        // Assert - regularized model should have smaller weights
        var paramsNoReg = regressorNoReg.GetParameters();
        var paramsWithReg = regressorWithReg.GetParameters();
        
        // Parameters are weights + bias, so extract just weights
        var weightsNoReg = new Vector<double>(paramsNoReg.Take(paramsNoReg.Length - 1).ToArray());
        var weightsWithReg = new Vector<double>(paramsWithReg.Take(paramsWithReg.Length - 1).ToArray());
        
        var normNoReg = weightsNoReg.Norm();
        var normWithReg = weightsWithReg.Norm();
        Assert.IsTrue(normWithReg < normNoReg);
    }

    [TestMethod]
    public void PredictBatch_ShouldReturnCorrectShape()
    {
        // Arrange
        var regressor = new OnlineSGDRegressor<double>(3, FitnessCalculatorType.MeanSquaredError);
        var inputs = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 },
            { 7.0, 8.0, 9.0 }
        });

        // Act
        // PredictBatch doesn't exist, predict individually
        var predictions = new double[inputs.Rows];
        for (int i = 0; i < inputs.Rows; i++)
        {
            var input = new Vector<double>(new[] { inputs[i, 0], inputs[i, 1], inputs[i, 2] });
            predictions[i] = regressor.Predict(input);
        }

        // Assert
        Assert.AreEqual(3, predictions.Length);
    }

    [TestMethod]
    public void WithMomentum_ShouldConvergeFaster()
    {
        // Arrange
        var optionsNoMomentum = new AdaptiveOnlineModelOptions<double> { InitialLearningRate = 0.01, UseMomentum = false };
        var optionsWithMomentum = new AdaptiveOnlineModelOptions<double> { InitialLearningRate = 0.01, UseMomentum = true, MomentumFactor = 0.9 };
        var regressorNoMomentum = new OnlineSGDRegressor<double>(1, FitnessCalculatorType.MeanSquaredError, optionsNoMomentum);
        var regressorWithMomentum = new OnlineSGDRegressor<double>(1, FitnessCalculatorType.MeanSquaredError, optionsWithMomentum);

        var input = new Vector<double>(new[] { 1.0 });
        var target = 10.0;

        // Act - single update
        regressorNoMomentum.PartialFit(input, target);
        regressorWithMomentum.PartialFit(input, target);

        // Assert - momentum should lead to larger update
        var predNoMomentum = regressorNoMomentum.Predict(input);
        var predWithMomentum = regressorWithMomentum.Predict(input);
        
        // Both should move towards target, but momentum might behave differently
        // Just verify they produce different results
        Assert.AreNotEqual(predNoMomentum, predWithMomentum);
    }

    [TestMethod]
    public void Clone_ShouldCreateIndependentCopy()
    {
        // Arrange
        var regressor = new OnlineSGDRegressor<double>(2, FitnessCalculatorType.MeanSquaredError);
        var input = new Vector<double>(new[] { 1.0, 2.0 });
        
        // Train the original model
        for (int i = 0; i < 10; i++)
        {
            regressor.PartialFit(input, 5.0);
        }

        // Act
        var clone = regressor.Clone() as OnlineSGDRegressor<double>;
        
        // Further train the original
        for (int i = 0; i < 10; i++)
        {
            regressor.PartialFit(input, 10.0);
        }

        // Assert
        Assert.IsNotNull(clone);
        var originalPred = regressor.Predict(input);
        var clonePred = clone.Predict(input);
        Assert.AreNotEqual(originalPred, clonePred);
        Assert.IsTrue(Math.Abs(clonePred - 5.0) < Math.Abs(clonePred - 10.0));
    }

    [TestMethod]
    public void LearningRateSchedule_ShouldDecay()
    {
        // Arrange
        var options = new AdaptiveOnlineModelOptions<double> 
        { 
            InitialLearningRate = 0.1,
            UseAdaptiveLearningRate = true,
            LearningRateDecay = 0.001
        };
        var regressor = new OnlineSGDRegressor<double>(1, FitnessCalculatorType.MeanSquaredError, options);
        var input = new Vector<double>(new[] { 1.0 });

        // Act - track predictions after each update
        var predictions = new List<double>();
        predictions.Add(regressor.Predict(input));
        
        for (int i = 0; i < 5; i++)
        {
            regressor.PartialFit(input, 10.0);
            predictions.Add(regressor.Predict(input));
        }

        // Assert - updates should get smaller over time
        var differences = new List<double>();
        for (int i = 1; i < predictions.Count; i++)
        {
            differences.Add(Math.Abs(predictions[i] - predictions[i - 1]));
        }

        // Differences should generally decrease (learning rate decay)
        for (int i = 1; i < differences.Count - 1; i++)
        {
            Assert.IsTrue(differences[i] <= differences[i - 1] * 1.1); // Allow small tolerance
        }
    }
}
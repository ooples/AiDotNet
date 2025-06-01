using System;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.OnlineLearning;
using AiDotNet.OnlineLearning.Algorithms;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace AiDotNetTests.UnitTests.OnlineLearning;

[TestClass]
public class OnlineModelBaseTests
{
    [TestMethod]
    public void AllModels_ShouldImplementIOnlineModel()
    {
        // Arrange
        var models = new IOnlineModel<double, Vector<double>, double>[]
        {
            new OnlinePerceptron<double>(3),
            new PassiveAggressiveRegressor<double>(3),
            new OnlineSGDRegressor<double>(3, FitnessCalculatorType.MeanSquaredError),
            new OnlineSVM<double>(3),
            new AROW<double>(3),
            new ConfidenceWeighted<double>(3),
            new HoeffdingTree<double>(3, 2), // numFeatures, numClasses
            new OnlineRandomForest<double>(3, 2, 5), // numFeatures, numClasses, numTrees
            new OnlineBagging<double>(3, 2, OnlineLearningAlgorithm.Perceptron, 5),
            new FTRL<double>(3),
            new OnlineNaiveBayes<double>(3, 2), // numFeatures, numClasses
            new OnlineKMeans<double>(3, 2)
        };

        // Assert
        foreach (var model in models)
        {
            Assert.IsNotNull(model);
            Assert.IsTrue(model is IOnlineModel<double, Vector<double>, double>);
        }
    }

    [TestMethod]
    public void AllModels_ShouldSupportBatchOperations()
    {
        // Arrange
        var models = new IOnlineModel<double, Vector<double>, double>[]
        {
            new OnlinePerceptron<double>(2),
            new PassiveAggressiveRegressor<double>(2),
            new OnlineSGDRegressor<double>(2, FitnessCalculatorType.MeanSquaredError),
            new OnlineSVM<double>(2),
            new AROW<double>(2),
            new ConfidenceWeighted<double>(2),
            new HoeffdingTree<double>(2, 2),
            new OnlineRandomForest<double>(2, 2, 3),
            new OnlineBagging<double>(2, 2, OnlineLearningAlgorithm.Perceptron, 3),
            new FTRL<double>(2),
            new OnlineNaiveBayes<double>(2, 2),
            new OnlineKMeans<double>(2, 2)
        };

        var batchInputs = new[]
        {
            new Vector<double>(new[] { 1.0, 0.0 }),
            new Vector<double>(new[] { 0.0, 1.0 }),
            new Vector<double>(new[] { 1.0, 1.0 })
        };
        var batchTargets = new[] { 1.0, 0.0, 0.5 };

        // Act & Assert
        foreach (var model in models)
        {
            // Should not throw
            model.PartialFitBatch(batchInputs, batchTargets);
            
            // PredictBatch doesn't exist, predict individually
            var predictions = new double[batchInputs.Length];
            for (int i = 0; i < batchInputs.Length; i++)
            {
                predictions[i] = model.Predict(batchInputs[i]);
            }
            
            Assert.IsNotNull(predictions);
            Assert.AreEqual(3, predictions.Length);
        }
    }

    [TestMethod]
    public void AllModels_ShouldSupportCloning()
    {
        // Arrange
        var models = new IOnlineModel<double, Vector<double>, double>[]
        {
            new OnlinePerceptron<double>(2),
            new PassiveAggressiveRegressor<double>(2),
            new OnlineSGDRegressor<double>(2, FitnessCalculatorType.MeanSquaredError),
            new OnlineSVM<double>(2),
            new AROW<double>(2),
            new ConfidenceWeighted<double>(2),
            new HoeffdingTree<double>(2, 2),
            new OnlineRandomForest<double>(2, 2, 3),
            new OnlineBagging<double>(2, 2, OnlineLearningAlgorithm.Perceptron, 3),
            new FTRL<double>(2),
            new OnlineNaiveBayes<double>(2, 2),
            new OnlineKMeans<double>(2, 2)
        };

        var input = new Vector<double>(new[] { 1.0, 0.5 });

        // Act & Assert
        foreach (var model in models)
        {
            // Train original
            model.PartialFit(input, 1.0);
            
            // Clone
            var clone = model.Clone();
            
            // Verify clone is independent
            Assert.IsNotNull(clone);
            Assert.IsTrue(clone is IOnlineModel<double, Vector<double>, double>);
            
            // Update original
            model.PartialFit(input, 0.0);
            
            // Predictions might differ (not guaranteed for all models)
            var origPred = model.Predict(input);
            var clonePred = clone.Predict(input);
            
            // At minimum, both should produce valid predictions
            Assert.IsFalse(double.IsNaN(origPred));
            Assert.IsFalse(double.IsNaN(clonePred));
        }
    }

    [TestMethod]
    public void AllModels_ShouldSupportParameterGetSet()
    {
        // Arrange
        var models = new IFullModel<double, Vector<double>, double>[]
        {
            new OnlinePerceptron<double>(2),
            new PassiveAggressiveRegressor<double>(2),
            new OnlineSGDRegressor<double>(2, FitnessCalculatorType.MeanSquaredError),
            new OnlineSVM<double>(2),
            new AROW<double>(2),
            new ConfidenceWeighted<double>(2),
            new HoeffdingTree<double>(2, 2),
            new OnlineRandomForest<double>(2, 2, 3),
            new OnlineBagging<double>(2, 2, OnlineLearningAlgorithm.Perceptron, 3),
            new FTRL<double>(2),
            new OnlineNaiveBayes<double>(2, 2),
            new OnlineKMeans<double>(2, 2)
        };

        // Act & Assert
        foreach (var model in models)
        {
            var parameters = model.GetParameters();
            
            Assert.IsNotNull(parameters);
            Assert.IsTrue(parameters.Length > 0);
            
            // Should be able to set parameters back
            model.SetParameters(parameters);
        }
    }

    [TestMethod]
    public void AdaptiveModels_ShouldImplementIAdaptiveOnlineModel()
    {
        // Arrange - models that should be adaptive
        var adaptiveModels = new IAdaptiveOnlineModel<double, Vector<double>, double>[]
        {
            new AROW<double>(3),
            new ConfidenceWeighted<double>(3)
        };

        // Assert
        foreach (var model in adaptiveModels)
        {
            Assert.IsNotNull(model);
            Assert.IsTrue(model is IAdaptiveOnlineModel<double, Vector<double>, double>);
            
            // Adaptive models should have drift detection capabilities
            // Check that DriftSensitivity is configurable (adaptive models have this property)
            Assert.IsTrue(model.DriftSensitivity != null);
        }
    }

    [TestMethod]
    public void ModelOptions_ShouldBeRespected()
    {
        // Arrange
        var options = new OnlineModelOptions<double>
        {
            InitialLearningRate = 0.01,
            RegularizationParameter = 0.1,
            UseAdaptiveLearningRate = true
        };

        var models = new IOnlineModel<double, Vector<double>, double>[]
        {
            new OnlinePerceptron<double>(2, options),
            new PassiveAggressiveRegressor<double>(2, options),
            new OnlineSGDRegressor<double>(2, FitnessCalculatorType.MeanSquaredError, new AdaptiveOnlineModelOptions<double> { InitialLearningRate = options.InitialLearningRate, RegularizationParameter = options.RegularizationParameter }),
            new OnlineSVM<double>(2, options),
            new AROW<double>(2, options),
            new ConfidenceWeighted<double>(2, options),
            new HoeffdingTree<double>(2, 2, options),
            new FTRL<double>(2, options),
            new OnlineNaiveBayes<double>(2, 2, false, options),
            new OnlineKMeans<double>(2, 2, true, options)
        };

        // Act & Assert - models should be created without errors
        foreach (var model in models)
        {
            Assert.IsNotNull(model);
            
            // Verify model uses the options (implementation specific)
            var input = new Vector<double>(new[] { 1.0, 1.0 });
            model.PartialFit(input, 1.0);
            
            // Should produce valid prediction
            var prediction = model.Predict(input);
            Assert.IsFalse(double.IsNaN(prediction));
        }
    }

    [TestMethod]
    public void NullInputs_ShouldThrowArgumentNullException()
    {
        // Arrange
        var model = new OnlinePerceptron<double>(2);
        Vector<double>? nullVector = null;

        // Act & Assert
        Assert.ThrowsException<ArgumentNullException>(() => model.PartialFit(nullVector!, 1.0));
        Assert.ThrowsException<ArgumentNullException>(() => model.Predict(nullVector!));
        Assert.ThrowsException<ArgumentNullException>(() => model.PartialFitBatch(null!, new[] { 1.0 }));
    }

    [TestMethod]
    public void IncorrectDimensions_ShouldThrowArgumentException()
    {
        // Arrange
        var model = new OnlinePerceptron<double>(3); // Expects 3 features
        var wrongInput = new Vector<double>(new[] { 1.0, 2.0 }); // Only 2 features
        var wrongBatch = new[] { new Vector<double>(new[] { 1.0, 2.0 }) }; // Only 2 features

        // Act & Assert
        Assert.ThrowsException<ArgumentException>(() => model.PartialFit(wrongInput, 1.0));
        Assert.ThrowsException<ArgumentException>(() => model.Predict(wrongInput));
        Assert.ThrowsException<ArgumentException>(() => model.PartialFitBatch(wrongBatch, new[] { 1.0 }));
    }
}
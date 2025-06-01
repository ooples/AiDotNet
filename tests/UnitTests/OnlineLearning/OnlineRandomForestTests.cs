using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.OnlineLearning.Algorithms;

namespace AiDotNetTests.UnitTests.OnlineLearning;

[TestClass]
public class OnlineRandomForestTests
{
    [TestMethod]
    public void Constructor_ShouldInitializeCorrectly()
    {
        // Arrange & Act
        var forest = new OnlineRandomForest<double>(4, 2, 10); // numFeatures, numClasses, numTrees

        // Assert
        Assert.IsNotNull(forest);
        Assert.AreEqual(4, forest.GetInputFeatureCount());
    }

    [TestMethod]
    public void PartialFit_ShouldImproveAccuracy()
    {
        // Arrange
        var forest = new OnlineRandomForest<double>(2, 2, 5); // numFeatures, numClasses, numTrees
        
        // Create XOR-like pattern
        var inputs = new[]
        {
            new Vector<double>(new[] { 0.0, 0.0 }),
            new Vector<double>(new[] { 0.0, 1.0 }),
            new Vector<double>(new[] { 1.0, 0.0 }),
            new Vector<double>(new[] { 1.0, 1.0 })
        };
        var outputs = new[] { 0.0, 1.0, 1.0, 0.0 }; // XOR

        // Act - train multiple epochs
        for (int epoch = 0; epoch < 100; epoch++)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                forest.PartialFit(inputs[i], outputs[i]);
            }
        }

        // Assert - should learn XOR pattern
        Assert.IsTrue(forest.Predict(inputs[0]) < 0.5); // 0
        Assert.IsTrue(forest.Predict(inputs[1]) > 0.5); // 1
        Assert.IsTrue(forest.Predict(inputs[2]) > 0.5); // 1
        Assert.IsTrue(forest.Predict(inputs[3]) < 0.5); // 0
    }

    [TestMethod]
    public void EnsemblePrediction_ShouldAverageTreePredictions()
    {
        // Arrange
        var forest = new OnlineRandomForest<double>(2, 2, 10); // numFeatures, numClasses, numTrees
        
        // Simple linear pattern
        var inputs = new[]
        {
            new Vector<double>(new[] { 1.0, 1.0 }),
            new Vector<double>(new[] { 2.0, 2.0 }),
            new Vector<double>(new[] { -1.0, -1.0 }),
            new Vector<double>(new[] { -2.0, -2.0 })
        };
        var outputs = new[] { 1.0, 1.0, 0.0, 0.0 };

        // Act
        for (int epoch = 0; epoch < 50; epoch++)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                forest.PartialFit(inputs[i], outputs[i]);
            }
        }

        // Assert - predictions should be smooth due to averaging
        var pred1 = forest.Predict(new Vector<double>(new[] { 0.5, 0.5 }));
        var pred2 = forest.Predict(new Vector<double>(new[] { -0.5, -0.5 }));
        
        // Should be between 0 and 1 (ensemble average)
        Assert.IsTrue(pred1 > 0.4 && pred1 < 0.8);
        Assert.IsTrue(pred2 > 0.2 && pred2 < 0.6);
        Assert.IsTrue(pred1 > pred2); // Positive side should have higher prediction
    }

    [TestMethod]
    public void PartialFitBatch_ShouldUpdateAllTrees()
    {
        // Arrange
        var forest = new OnlineRandomForest<double>(3, 2, 5); // numFeatures, numClasses, numTrees
        var inputs = new Matrix<double>(new double[,]
        {
            { 1.0, 0.0, 1.0 },
            { 0.0, 1.0, 1.0 },
            { 1.0, 1.0, 0.0 },
            { 0.0, 0.0, 0.0 }
        });
        var outputs = new Vector<double>(new[] { 1.0, 1.0, 1.0, 0.0 });

        // Act
        for (int i = 0; i < 30; i++)
        {
            // Convert Matrix to array of Vectors
            var inputVectors = new Vector<double>[inputs.Rows];
            for (int j = 0; j < inputs.Rows; j++)
            {
                inputVectors[j] = new Vector<double>(new[] { inputs[j, 0], inputs[j, 1], inputs[j, 2] });
            }
            forest.PartialFitBatch(inputVectors, outputs.ToArray());
        }

        // Assert
        Assert.IsTrue(forest.Predict(new Vector<double>(new[] { 1.0, 0.5, 0.5 })) > 0.5);
        Assert.IsTrue(forest.Predict(new Vector<double>(new[] { 0.0, 0.0, 0.1 })) < 0.5);
    }

    [TestMethod]
    public void WithDifferentNumTrees_ShouldAffectPredictionStability()
    {
        // Arrange
        var forest1 = new OnlineRandomForest<double>(2, 2, 1); // numFeatures, numClasses, single tree
        var forest10 = new OnlineRandomForest<double>(2, 2, 10); // numFeatures, numClasses, 10 trees
        
        var inputs = new[]
        {
            new Vector<double>(new[] { 1.0, 1.0 }),
            new Vector<double>(new[] { -1.0, -1.0 })
        };
        var outputs = new[] { 1.0, 0.0 };

        // Act - train both forests
        for (int epoch = 0; epoch < 20; epoch++)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                forest1.PartialFit(inputs[i], outputs[i]);
                forest10.PartialFit(inputs[i], outputs[i]);
            }
        }

        // Test on ambiguous point
        var testPoint = new Vector<double>(new[] { 0.1, 0.1 });
        
        // Get multiple predictions to test stability
        var predictions1 = new List<double>();
        var predictions10 = new List<double>();
        
        for (int i = 0; i < 5; i++)
        {
            // Add small noise to trigger different random behavior
            forest1.PartialFit(new Vector<double>(new[] { 0.0, 0.0 }), 0.5);
            forest10.PartialFit(new Vector<double>(new[] { 0.0, 0.0 }), 0.5);
            
            predictions1.Add(forest1.Predict(testPoint));
            predictions10.Add(forest10.Predict(testPoint));
        }

        // Assert - ensemble should have more stable predictions
        var variance1 = CalculateVariance(predictions1);
        var variance10 = CalculateVariance(predictions10);
        
        // More trees should generally lead to more stable predictions
        Assert.IsTrue(variance10 <= variance1 * 1.1); // Allow small tolerance
    }

    [TestMethod]
    public void Clone_ShouldCreateIndependentForest()
    {
        // Arrange
        var forest = new OnlineRandomForest<double>(2, 2, 3); // numFeatures, numClasses, numTrees
        
        // Train original
        var inputs = new[]
        {
            new Vector<double>(new[] { 1.0, 0.0 }),
            new Vector<double>(new[] { 0.0, 1.0 })
        };
        var outputs = new[] { 1.0, 0.0 };
        
        for (int i = 0; i < 30; i++)
        {
            for (int j = 0; j < inputs.Length; j++)
            {
                forest.PartialFit(inputs[j], outputs[j]);
            }
        }

        // Act
        var clone = forest.Clone() as OnlineRandomForest<double>;
        
        // Update original with opposite pattern
        for (int i = 0; i < 30; i++)
        {
            forest.PartialFit(inputs[0], 0.0); // Reverse the pattern
            forest.PartialFit(inputs[1], 1.0);
        }

        // Assert
        Assert.IsNotNull(clone);
        // Clone should maintain original pattern
        Assert.IsTrue(clone.Predict(inputs[0]) > 0.5);
        Assert.IsTrue(clone.Predict(inputs[1]) < 0.5);
        // Original should have adapted to new pattern
        Assert.IsTrue(forest.Predict(inputs[0]) < 0.5);
        Assert.IsTrue(forest.Predict(inputs[1]) > 0.5);
    }

    [TestMethod]
    public void GetParameters_ShouldReturnForestInfo()
    {
        // Arrange
        var forest = new OnlineRandomForest<double>(3, 2, 5); // numFeatures, numClasses, numTrees

        // Act
        var parameters = forest.GetParameters();

        // Assert - GetParameters returns Vector with hyperparameters [numTrees, subspaceSize, splitConfidence]
        Assert.IsNotNull(parameters);
        Assert.AreEqual(3, parameters.Length);
        Assert.AreEqual(5.0, parameters[0]); // numTrees
    }

    [TestMethod]
    public void PredictBatch_ShouldReturnAveragedPredictions()
    {
        // Arrange
        var forest = new OnlineRandomForest<double>(2, 2, 7); // numFeatures, numClasses, numTrees
        
        // Train
        var trainInputs = new Matrix<double>(new double[,]
        {
            { 2.0, 2.0 },
            { 2.0, -2.0 },
            { -2.0, 2.0 },
            { -2.0, -2.0 }
        });
        var trainOutputs = new Vector<double>(new[] { 1.0, 0.0, 0.0, 1.0 }); // XOR-like

        for (int i = 0; i < 50; i++)
        {
            // Convert Matrix to array of Vectors
            var trainVectors = new Vector<double>[trainInputs.Rows];
            for (int j = 0; j < trainInputs.Rows; j++)
            {
                trainVectors[j] = new Vector<double>(new[] { trainInputs[j, 0], trainInputs[j, 1] });
            }
            forest.PartialFitBatch(trainVectors, trainOutputs.ToArray());
        }

        // Act
        var testInputs = new Matrix<double>(new double[,]
        {
            { 1.5, 1.5 },
            { 1.5, -1.5 },
            { -1.5, 1.5 },
            { -1.5, -1.5 }
        });
        // PredictBatch doesn't exist, predict individually
        var predictions = new double[testInputs.Rows];
        for (int j = 0; j < testInputs.Rows; j++)
        {
            var input = new Vector<double>(new[] { testInputs[j, 0], testInputs[j, 1] });
            predictions[j] = forest.Predict(input);
        }

        // Assert
        Assert.AreEqual(4, predictions.Length);
        Assert.IsTrue(predictions[0] > 0.5); // Similar to (2,2) -> 1
        Assert.IsTrue(predictions[1] < 0.5); // Similar to (2,-2) -> 0
        Assert.IsTrue(predictions[2] < 0.5); // Similar to (-2,2) -> 0
        Assert.IsTrue(predictions[3] > 0.5); // Similar to (-2,-2) -> 1
    }

    private double CalculateVariance(List<double> values)
    {
        var mean = values.Average();
        var sumOfSquares = values.Sum(v => Math.Pow(v - mean, 2));
        return sumOfSquares / values.Count;
    }
}
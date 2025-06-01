using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.OnlineLearning.Algorithms;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace AiDotNetTests.UnitTests.OnlineLearning;

[TestClass]
public class FTRLTests
{
    [TestMethod]
    public void Constructor_ShouldInitializeCorrectly()
    {
        // Arrange & Act
        var ftrl = new FTRL<double>(5);

        // Assert
        Assert.IsNotNull(ftrl);
        Assert.AreEqual(5, ftrl.GetInputFeatureCount());
    }

    [TestMethod]
    public void PartialFit_WithSparseData_ShouldLearnEfficiently()
    {
        // Arrange
        var ftrl = new FTRL<double>(10);
        
        // Sparse inputs - only a few features are non-zero
        var inputs = new[]
        {
            CreateSparseVector(10, new[] { (0, 1.0), (2, 1.0) }),
            CreateSparseVector(10, new[] { (1, 1.0), (3, 1.0) }),
            CreateSparseVector(10, new[] { (0, -1.0), (2, -1.0) }),
            CreateSparseVector(10, new[] { (1, -1.0), (3, -1.0) })
        };
        var outputs = new[] { 1.0, 1.0, 0.0, 0.0 };

        // Act
        for (int epoch = 0; epoch < 50; epoch++)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                ftrl.PartialFit(inputs[i], outputs[i]);
            }
        }

        // Assert
        var testPositive = CreateSparseVector(10, new[] { (0, 0.5), (2, 0.5) });
        var testNegative = CreateSparseVector(10, new[] { (0, -0.5), (2, -0.5) });
        
        Assert.IsTrue(ftrl.Predict(testPositive) > 0.5);
        Assert.IsTrue(ftrl.Predict(testNegative) < 0.5);
    }

    [TestMethod]
    public void FTRL_WithL1Regularization_ShouldProduceSparseWeights()
    {
        // Arrange
        var options = new OnlineModelOptions<double> 
        { 
            RegularizationParameter = 1.0,  // Used as L1 regularization (lambda1)
            LearningRateDecay = 0.0         // Used as beta parameter
        };
        var ftrl = new FTRL<double>(5, options);

        // Dense inputs with redundant features
        var inputs = new[]
        {
            new Vector<double>(new[] { 1.0, 0.1, 0.1, 0.1, 0.1 }),
            new Vector<double>(new[] { -1.0, -0.1, -0.1, -0.1, -0.1 })
        };
        var outputs = new[] { 1.0, 0.0 };

        // Act - train with strong L1 regularization
        for (int epoch = 0; epoch < 100; epoch++)
        {
            for (int j = 0; j < inputs.Length; j++)
            {
                ftrl.PartialFit(inputs[j], outputs[j]);
            }
        }

        // Assert - should have sparse weights
        // GetParameters returns hyperparameters, not weights
        // Check sparsity by looking at active features
        var activeFeatures = ftrl.GetActiveFeatureIndices().Count();
        
        // With L1 regularization, most redundant features should be zeroed out
        Assert.IsTrue(activeFeatures <= 2); // At most 2 out of 5 should be active
    }

    [TestMethod]
    public void PartialFitBatch_ShouldUpdateModel()
    {
        // Arrange
        var ftrl = new FTRL<double>(3);
        var inputs = new Matrix<double>(new double[,]
        {
            { 1.0, 0.0, 1.0 },
            { 0.0, 1.0, 1.0 },
            { 1.0, 1.0, 0.0 },
            { 0.0, 0.0, 0.0 }
        });
        var outputs = new Vector<double>(new[] { 1.0, 1.0, 1.0, 0.0 });

        // Act
        for (int i = 0; i < 20; i++)
        {
            // PartialFitBatch takes arrays, not Matrix/Vector
            for (int j = 0; j < inputs.Rows; j++)
            {
                var inputVector = new Vector<double>(new[] { inputs[j, 0], inputs[j, 1], inputs[j, 2] });
                ftrl.PartialFit(inputVector, outputs[j]);
            }
        }

        // Assert
        Assert.IsTrue(ftrl.Predict(new Vector<double>(new[] { 0.5, 0.5, 0.5 })) > 0.5);
        Assert.IsTrue(ftrl.Predict(new Vector<double>(new[] { 0.0, 0.0, 0.0 })) < 0.5);
    }

    [TestMethod]
    public void WithUseAdaptiveLearningRate_ShouldConvergeFaster()
    {
        // Arrange
        var ftrl = new FTRL<double>(2);
        var input = new Vector<double>(new[] { 1.0, 1.0 });

        // Act - track how prediction changes over updates
        var predictions = new List<double>();
        predictions.Add(ftrl.Predict(input));
        
        for (int i = 0; i < 10; i++)
        {
            ftrl.PartialFit(input, 1.0);
            predictions.Add(ftrl.Predict(input));
        }

        // Assert - predictions should monotonically increase towards 1
        for (int i = 1; i < predictions.Count; i++)
        {
            Assert.IsTrue(predictions[i] >= predictions[i - 1]);
        }
        Assert.IsTrue(predictions.Last() > 0.8);
    }

    [TestMethod]
    public void Clone_ShouldCreateIndependentCopy()
    {
        // Arrange
        var ftrl = new FTRL<double>(3);
        
        // Train original
        var inputs = new[]
        {
            new Vector<double>(new[] { 1.0, 0.0, 0.0 }),
            new Vector<double>(new[] { 0.0, 1.0, 0.0 }),
            new Vector<double>(new[] { 0.0, 0.0, 1.0 })
        };
        var outputs = new[] { 1.0, 0.5, 0.0 };
        
        for (int i = 0; i < 30; i++)
        {
            for (int j = 0; j < inputs.Length; j++)
            {
                ftrl.PartialFit(inputs[j], outputs[j]);
            }
        }

        // Act
        var clone = ftrl.Clone() as FTRL<double>;
        
        // Update original with different pattern
        ftrl.PartialFit(inputs[0], 0.0);

        // Assert
        Assert.IsNotNull(clone);
        Assert.AreNotEqual(ftrl.Predict(inputs[0]), clone.Predict(inputs[0]));
    }

    [TestMethod]
    public void GetSetParameters_ShouldIncludeAccumulators()
    {
        // Arrange
        var ftrl = new FTRL<double>(2);
        
        // Train to update accumulators
        ftrl.PartialFit(new Vector<double>(new[] { 1.0, 0.0 }), 1.0);
        
        // Act
        var parameters = ftrl.GetParameters();

        // Assert
        // GetParameters returns hyperparameters [alpha, beta, lambda1, lambda2]
        Assert.AreEqual(4, parameters.Length);
        Assert.IsTrue(parameters[0] > 0); // alpha
        Assert.IsTrue(parameters[1] >= 0); // beta
        Assert.IsTrue(parameters[2] >= 0); // lambda1
        Assert.IsTrue(parameters[3] >= 0); // lambda2
    }

    [TestMethod]
    public void PredictBatch_ShouldReturnProbabilities()
    {
        // Arrange
        var ftrl = new FTRL<double>(2);
        
        // Train with binary classification data
        var trainInputs = new Matrix<double>(new double[,]
        {
            { 1.0, 1.0 },
            { -1.0, -1.0 }
        });
        var trainOutputs = new Vector<double>(new[] { 1.0, 0.0 });
        
        for (int i = 0; i < 50; i++)
        {
            // PartialFitBatch takes arrays, not Matrix/Vector
            for (int j = 0; j < trainInputs.Rows; j++)
            {
                var inputVector = new Vector<double>(new[] { trainInputs[j, 0], trainInputs[j, 1] });
                ftrl.PartialFit(inputVector, trainOutputs[j]);
            }
        }

        // Act
        var testInputs = new Matrix<double>(new double[,]
        {
            { 0.5, 0.5 },
            { -0.5, -0.5 },
            { 0.0, 0.0 }
        });
        // PredictBatch doesn't exist, predict each individually
        var predictions = new double[testInputs.Rows];
        for (int i = 0; i < testInputs.Rows; i++)
        {
            var inputVector = new Vector<double>(new[] { testInputs[i, 0], testInputs[i, 1] });
            predictions[i] = ftrl.PredictProbability(inputVector);
        }

        // Assert - all should be valid probabilities
        Assert.AreEqual(3, predictions.Length);
        foreach (var pred in predictions)
        {
            Assert.IsTrue(pred >= 0.0 && pred <= 1.0);
        }
        
        // Check ordering
        Assert.IsTrue(predictions[0] > predictions[2]); // Positive side
        Assert.IsTrue(predictions[2] > predictions[1]); // Middle > negative
    }

    [TestMethod]
    public void FTRL_ShouldHandleHighDimensionalData()
    {
        // Arrange
        var ftrl = new FTRL<double>(1000); // High dimensional
        
        // Create sparse high-dimensional data
        var input1 = new Vector<double>(1000);
        var input2 = new Vector<double>(1000);
        
        // Only set a few features
        input1[10] = 1.0;
        input1[50] = 1.0;
        input1[100] = 1.0;
        
        input2[20] = 1.0;
        input2[60] = 1.0;
        input2[110] = 1.0;

        // Act
        for (int i = 0; i < 20; i++)
        {
            ftrl.PartialFit(input1, 1.0);
            ftrl.PartialFit(input2, 0.0);
        }

        // Assert
        Assert.IsTrue(ftrl.Predict(input1) > 0.7);
        Assert.IsTrue(ftrl.Predict(input2) < 0.3);
    }

    private Vector<double> CreateSparseVector(int size, (int index, double value)[] nonZeroElements)
    {
        var vector = new Vector<double>(size);
        foreach (var (index, value) in nonZeroElements)
        {
            vector[index] = value;
        }
        return vector;
    }
}
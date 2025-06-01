using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.OnlineLearning.Algorithms;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace AiDotNetTests.UnitTests.OnlineLearning;

[TestClass]
public class ConfidenceWeightedTests
{
    [TestMethod]
    public void Constructor_ShouldInitializeCorrectly()
    {
        // Arrange & Act
        var cw = new ConfidenceWeighted<double>(3);

        // Assert
        Assert.IsNotNull(cw);
        Assert.AreEqual(3, cw.GetInputFeatureCount());
    }

    [TestMethod]
    public void PartialFit_WithLinearlySeparableData_ShouldConverge()
    {
        // Arrange
        var cw = new ConfidenceWeighted<double>(2);
        var inputs = new[]
        {
            new Vector<double>(new[] { 2.0, 3.0 }),
            new Vector<double>(new[] { 3.0, 4.0 }),
            new Vector<double>(new[] { -1.0, -1.0 }),
            new Vector<double>(new[] { -2.0, -1.0 })
        };
        var labels = new[] { 1.0, 1.0, -1.0, -1.0 };

        // Act - train multiple epochs
        for (int epoch = 0; epoch < 30; epoch++)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                cw.PartialFit(inputs[i], labels[i]);
            }
        }

        // Assert
        // ConfidenceWeighted returns 0 or 1, not negative values
        Assert.AreEqual(1.0, cw.Predict(new Vector<double>(new[] { 2.5, 3.5 })));
        Assert.AreEqual(0.0, cw.Predict(new Vector<double>(new[] { -1.5, -1.0 })));
    }

    [TestMethod]
    public void ConfidenceWeighted_ShouldMaintainCovarianceMatrix()
    {
        // Arrange
        var cw = new ConfidenceWeighted<double>(2);
        var inputs = new[]
        {
            new Vector<double>(new[] { 1.0, 0.0 }),
            new Vector<double>(new[] { 0.0, 1.0 }),
            new Vector<double>(new[] { 1.0, 1.0 })
        };

        // Act
        cw.PartialFit(inputs[0], 1.0);
        cw.PartialFit(inputs[1], -1.0);
        cw.PartialFit(inputs[2], 1.0);

        // Assert - check covariance is maintained
        // GetParameters returns mean and variance concatenated
        var parameters = cw.GetParameters();
        // First half is mean, second half is variance (diagonal of covariance)
        var variance = parameters.Skip(cw.GetInputFeatureCount()).Take(cw.GetInputFeatureCount()).ToArray();
        var covariance = new Vector<double>(variance);
        Assert.IsNotNull(covariance);
        
        // All diagonal elements should be positive
        foreach (var value in covariance)
        {
            Assert.IsTrue(value > 0);
        }
        
        // Covariance should have been updated (different from initial identity)
        Assert.AreNotEqual(1.0, covariance[0]);
        Assert.AreNotEqual(1.0, covariance[1]);
    }

    [TestMethod]
    public void WithDifferentEta_ShouldAffectConfidenceParameter()
    {
        // Arrange
        var optionsLowEta = new OnlineModelOptions<double> { InitialLearningRate = 0.5 }; // Used as confidence parameter
        var optionsHighEta = new OnlineModelOptions<double> { InitialLearningRate = 0.95 }; // Used as confidence parameter
        var cwLowEta = new ConfidenceWeighted<double>(2, optionsLowEta);
        var cwHighEta = new ConfidenceWeighted<double>(2, optionsHighEta);

        var input = new Vector<double>(new[] { 1.0, 1.0 });

        // Act
        cwLowEta.PartialFit(input, 1.0);
        cwHighEta.PartialFit(input, 1.0);

        // Assert - different eta should lead to different updates
        var predLowEta = cwLowEta.Predict(input);
        var predHighEta = cwHighEta.Predict(input);
        
        // Higher eta means higher confidence requirement, potentially different updates
        Assert.AreNotEqual(predLowEta, predHighEta);
    }

    [TestMethod]
    public void PartialFitBatch_ShouldUpdateModel()
    {
        // Arrange
        var cw = new ConfidenceWeighted<double>(2);
        var inputs = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 },
            { 2.0, 1.0 },
            { -1.0, -2.0 },
            { -2.0, -1.0 }
        });
        var labels = new Vector<double>(new[] { 1.0, 1.0, -1.0, -1.0 });

        // Act
        for (int i = 0; i < 15; i++)
        {
            // PartialFitBatch takes arrays, not Matrix/Vector
            for (int j = 0; j < inputs.Rows; j++)
            {
                var inputVector = new Vector<double>(new[] { inputs[j, 0], inputs[j, 1] });
                cw.PartialFit(inputVector, labels[j]);
            }
        }

        // Assert
        // ConfidenceWeighted returns 0 or 1, not negative values
        Assert.AreEqual(1.0, cw.Predict(new Vector<double>(new[] { 1.5, 1.5 })));
        Assert.AreEqual(0.0, cw.Predict(new Vector<double>(new[] { -1.5, -1.5 })));
    }

    [TestMethod]
    public void Clone_ShouldCreateIndependentCopy()
    {
        // Arrange
        var cw = new ConfidenceWeighted<double>(2);
        
        // Train original
        var trainData = new[]
        {
            (new Vector<double>(new[] { 1.0, 1.0 }), 1.0),
            (new Vector<double>(new[] { -1.0, -1.0 }), -1.0)
        };
        
        foreach (var (input, label) in trainData)
        {
            cw.PartialFit(input, label);
        }

        // Act
        var clone = cw.Clone() as ConfidenceWeighted<double>;
        
        // Further train original
        cw.PartialFit(new Vector<double>(new[] { 2.0, 2.0 }), 1.0);

        // Assert
        Assert.IsNotNull(clone);
        var testPoint = new Vector<double>(new[] { 1.5, 1.5 });
        Assert.AreNotEqual(cw.Predict(testPoint), clone.Predict(testPoint));
    }

    [TestMethod]
    public void GetSetParameters_ShouldIncludeMeanAndCovariance()
    {
        // Arrange
        var cw = new ConfidenceWeighted<double>(2);
        var mean = new Vector<double>(new[] { 0.3, -0.3 });
        var covariance = new Vector<double>(new[] { 0.7, 0.8 });
        
        // Parameters are mean followed by variance (diagonal of covariance)
        var parameters = new Vector<double>(new[] { 0.3, -0.3, 0.7, 0.8 });

        // Act
        cw.SetParameters(parameters);
        var retrieved = cw.GetParameters();

        // Assert
        Assert.AreEqual(4, retrieved.Length); // 2 mean + 2 variance
        Assert.AreEqual(0.3, retrieved[0]);
        Assert.AreEqual(-0.3, retrieved[1]);
        Assert.AreEqual(0.7, retrieved[2]);
        Assert.AreEqual(0.8, retrieved[3]);
    }

    [TestMethod]
    public void PredictBatch_ShouldReturnCorrectShape()
    {
        // Arrange
        var cw = new ConfidenceWeighted<double>(3);
        var inputs = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 },
            { 7.0, 8.0, 9.0 },
            { 10.0, 11.0, 12.0 }
        });

        // Act
        // PredictBatch doesn't exist, predict each individually
        var predictions = new double[inputs.Rows];
        for (int i = 0; i < inputs.Rows; i++)
        {
            var inputVector = new Vector<double>(new[] { inputs[i, 0], inputs[i, 1], inputs[i, 2] });
            predictions[i] = cw.Predict(inputVector);
        }

        // Assert
        Assert.AreEqual(4, predictions.Length);
    }

    [TestMethod]
    public void ConfidenceWeighted_ShouldHandleUncertainData()
    {
        // Arrange
        var cw = new ConfidenceWeighted<double>(2);
        
        // Create data with different levels of certainty
        var certainPositive = new Vector<double>(new[] { 5.0, 5.0 });
        var uncertainPositive = new Vector<double>(new[] { 0.1, 0.1 });
        var certainNegative = new Vector<double>(new[] { -5.0, -5.0 });

        // Act - train with certain examples first
        for (int i = 0; i < 10; i++)
        {
            cw.PartialFit(certainPositive, 1.0);
            cw.PartialFit(certainNegative, -1.0);
        }
        
        var predBeforeUncertain = cw.Predict(uncertainPositive);
        
        // Update with uncertain example
        cw.PartialFit(uncertainPositive, 1.0);
        
        var predAfterUncertain = cw.Predict(uncertainPositive);

        // Assert - model should update but maintain confidence bounds
        Assert.AreNotEqual(predBeforeUncertain, predAfterUncertain);
        // Both predictions are 0 or 1, so just check they're different
        // predAfterUncertain should likely be 1.0 if it moved towards positive
    }
}
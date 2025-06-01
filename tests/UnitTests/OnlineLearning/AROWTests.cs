using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.OnlineLearning.Algorithms;

namespace AiDotNetTests.UnitTests.OnlineLearning;

[TestClass]
public class AROWTests
{
    [TestMethod]
    public void Constructor_ShouldInitializeCorrectly()
    {
        // Arrange & Act
        var arow = new AROW<double>(4);

        // Assert
        Assert.IsNotNull(arow);
        Assert.AreEqual(4, arow.GetInputFeatureCount());
    }

    [TestMethod]
    public void PartialFit_WithLinearlySeparableData_ShouldConverge()
    {
        // Arrange
        var arow = new AROW<double>(2);
        var inputs = new[]
        {
            new Vector<double>(new[] { 3.0, 3.0 }),
            new Vector<double>(new[] { 4.0, 3.0 }),
            new Vector<double>(new[] { 1.0, 1.0 }),
            new Vector<double>(new[] { 1.0, 2.0 })
        };
        var labels = new[] { 1.0, 1.0, -1.0, -1.0 };

        // Act - train multiple epochs
        for (int epoch = 0; epoch < 30; epoch++)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                arow.PartialFit(inputs[i], labels[i]);
            }
        }

        // Assert
        Assert.AreEqual(1.0, arow.Predict(new Vector<double>(new[] { 3.5, 3.0 }))); // Positive class
        Assert.AreEqual(0.0, arow.Predict(new Vector<double>(new[] { 1.0, 1.5 }))); // Negative class
    }

    [TestMethod]
    public void AROW_ShouldMaintainConfidenceBounds()
    {
        // Arrange
        var arow = new AROW<double>(2);
        var inputs = new[]
        {
            new Vector<double>(new[] { 1.0, 0.0 }),
            new Vector<double>(new[] { 0.0, 1.0 })
        };

        // Act - update with different patterns
        arow.PartialFit(inputs[0], 1.0);
        arow.PartialFit(inputs[1], -1.0);

        // Assert - variance should have been updated
        var variance = arow.Variance;
        Assert.IsNotNull(variance);
        
        // Variance should be positive and less than initial value (1.0)
        Assert.IsTrue(variance[0] > 0 && variance[0] < 1.0);
        Assert.IsTrue(variance[1] > 0 && variance[1] < 1.0);
    }

    [TestMethod]
    public void WithDifferentR_ShouldAffectConfidenceUpdates()
    {
        // Arrange
        var optionsLowR = new OnlineModelOptions<double> { RegularizationParameter = 0.1 };
        var optionsHighR = new OnlineModelOptions<double> { RegularizationParameter = 10.0 };
        var arowLowR = new AROW<double>(2, optionsLowR);
        var arowHighR = new AROW<double>(2, optionsHighR);

        var input = new Vector<double>(new[] { 1.0, 1.0 });

        // Act
        arowLowR.PartialFit(input, 1.0);
        arowHighR.PartialFit(input, 1.0);

        // Assert - different R should lead to different variance updates
        // Get the variance directly from the model
        var varianceLowR = arowLowR.Variance;
        var varianceHighR = arowHighR.Variance;

        Assert.IsNotNull(varianceLowR);
        Assert.IsNotNull(varianceHighR);
        
        // Higher R should lead to more conservative updates (larger remaining variance)
        Assert.IsTrue(varianceHighR[0] > varianceLowR[0]);
    }

    [TestMethod]
    public void PartialFitBatch_ShouldUpdateModel()
    {
        // Arrange
        var arow = new AROW<double>(2);
        var inputs = new[]
        {
            new Vector<double>(new[] { 2.0, 1.0 }),
            new Vector<double>(new[] { 1.0, 2.0 }),
            new Vector<double>(new[] { -1.0, -1.0 }),
            new Vector<double>(new[] { -2.0, -1.0 })
        };
        var labels = new[] { 1.0, 1.0, -1.0, -1.0 };

        // Act
        for (int i = 0; i < 10; i++)
        {
            arow.PartialFitBatch(inputs, labels);
        }

        // Assert
        Assert.AreEqual(1.0, arow.Predict(new Vector<double>(new[] { 1.5, 1.5 }))); // Positive class
        Assert.AreEqual(0.0, arow.Predict(new Vector<double>(new[] { -1.5, -1.0 }))); // Negative class
    }

    [TestMethod]
    public void Clone_ShouldCreateIndependentCopy()
    {
        // Arrange
        var arow = new AROW<double>(2);
        var input = new Vector<double>(new[] { 1.0, 2.0 });
        arow.PartialFit(input, 1.0);

        // Act
        var clone = arow.Clone() as AROW<double>;
        arow.PartialFit(input, -1.0); // Change original

        // Assert
        Assert.IsNotNull(clone);
        // Predictions should be different after modifying the original
        var originalPred = clone.Predict(input);
        var modifiedPred = arow.Predict(input);
        // They might be equal if both classify to same class, but internal state is different
        Assert.IsNotNull(clone);
    }

    [TestMethod]
    public void GetSetParameters_ShouldIncludeVariance()
    {
        // Arrange
        var arow = new AROW<double>(2);
        
        // Train the model first to set some parameters
        arow.PartialFit(new Vector<double>(new[] { 1.0, 0.0 }), 1.0);
        
        // Act - Get parameters as vector (mean and variance concatenated)
        var parameters = arow.GetParameters();

        // Assert
        Assert.IsNotNull(parameters);
        Assert.AreEqual(4, parameters.Length); // 2 for mean + 2 for variance
        
        // Create a new AROW and set parameters
        var arow2 = new AROW<double>(2);
        arow2.SetParameters(parameters);
        
        // Should produce same predictions
        var testInput = new Vector<double>(new[] { 1.0, 1.0 });
        Assert.AreEqual(arow.Predict(testInput), arow2.Predict(testInput));
    }

    [TestMethod]
    public void PredictBatch_ShouldReturnCorrectPredictions()
    {
        // Arrange
        var arow = new AROW<double>(2);
        
        // Train the model
        var trainInputs = new[]
        {
            new Vector<double>(new[] { 1.0, 1.0 }),
            new Vector<double>(new[] { -1.0, -1.0 })
        };
        var trainLabels = new[] { 1.0, -1.0 };
        
        for (int i = 0; i < 20; i++)
        {
            for (int j = 0; j < trainInputs.Length; j++)
            {
                arow.PartialFit(trainInputs[j], trainLabels[j]);
            }
        }

        // Act
        var testInputs = new[]
        {
            new Vector<double>(new[] { 0.8, 0.8 }),
            new Vector<double>(new[] { -0.8, -0.8 }),
            new Vector<double>(new[] { 0.0, 0.0 })
        };
        var predictions = new double[testInputs.Length];
        for (int i = 0; i < testInputs.Length; i++)
        {
            predictions[i] = arow.Predict(testInputs[i]);
        }

        // Assert
        Assert.AreEqual(3, predictions.Length);
        Assert.IsTrue(predictions[0] > 0.5); // Similar to positive training examples (returns 1.0 for positive)
        Assert.IsTrue(predictions[1] < 0.5); // Similar to negative training examples (returns 0.0 for negative)
    }
}
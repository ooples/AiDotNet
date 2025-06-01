using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.OnlineLearning.Algorithms;

namespace AiDotNetTests.UnitTests.OnlineLearning;

[TestClass]
public class OnlineBaggingTests
{
    [TestMethod]
    public void Constructor_ShouldInitializeCorrectly()
    {
        // Arrange & Act
        var bagging = new OnlineBagging<double>(
            numFeatures: 3, 
            numClasses: 2,
            baseAlgorithm: OnlineLearningAlgorithm.Perceptron,
            ensembleSize: 5);

        // Assert
        Assert.IsNotNull(bagging);
        Assert.AreEqual(3, bagging.GetInputFeatureCount());
        Assert.AreEqual(2, bagging.GetOutputFeatureCount());
    }

    [TestMethod]
    public void PartialFit_WithPerceptrons_ShouldLearnBinaryClassification()
    {
        // Arrange
        var bagging = new OnlineBagging<double>(
            numFeatures: 2, 
            numClasses: 2,
            baseAlgorithm: OnlineLearningAlgorithm.Perceptron,
            ensembleSize: 10,
            isClassification: true);
        
        // Linearly separable data
        var inputs = new[]
        {
            new Vector<double>(new[] { 1.0, 2.0 }),
            new Vector<double>(new[] { 2.0, 3.0 }),
            new Vector<double>(new[] { -1.0, -2.0 }),
            new Vector<double>(new[] { -2.0, -3.0 })
        };
        var outputs = new[] { 1.0, 1.0, 0.0, 0.0 }; // Binary classification: 0 or 1

        // Act
        for (int epoch = 0; epoch < 50; epoch++)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                bagging.PartialFit(inputs[i], outputs[i]);
            }
        }

        // Assert
        Assert.AreEqual(1.0, bagging.Predict(new Vector<double>(new[] { 1.5, 2.5 })));
        Assert.AreEqual(0.0, bagging.Predict(new Vector<double>(new[] { -1.5, -2.5 })));
    }

    [TestMethod]
    public void PartialFit_WithSGDRegressors_ShouldFitRegression()
    {
        // Arrange
        var bagging = new OnlineBagging<double>(
            numFeatures: 1, 
            numClasses: 1, // For regression
            baseAlgorithm: OnlineLearningAlgorithm.StochasticGradientDescent,
            ensembleSize: 5,
            isClassification: false);
        
        // Linear regression data: y = 2x + 1
        var inputs = new[]
        {
            new Vector<double>(new[] { 0.0 }),
            new Vector<double>(new[] { 1.0 }),
            new Vector<double>(new[] { 2.0 }),
            new Vector<double>(new[] { 3.0 })
        };
        var outputs = new[] { 1.0, 3.0, 5.0, 7.0 };

        // Act
        for (int epoch = 0; epoch < 100; epoch++)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                bagging.PartialFit(inputs[i], outputs[i]);
            }
        }

        // Assert
        var pred = bagging.Predict(new Vector<double>(new[] { 4.0 }));
        Assert.IsTrue(pred >= 7.0 && pred <= 11.0); // Should predict close to 9
    }

    [TestMethod]
    public void PredictProbabilities_ShouldReturnValidProbabilities()
    {
        // Arrange
        var bagging = new OnlineBagging<double>(
            numFeatures: 2, 
            numClasses: 3,
            baseAlgorithm: OnlineLearningAlgorithm.AROW,
            ensembleSize: 5,
            isClassification: true);
        
        // Multi-class data
        var inputs = new[]
        {
            new Vector<double>(new[] { 1.0, 0.0 }),
            new Vector<double>(new[] { 0.0, 1.0 }),
            new Vector<double>(new[] { -1.0, -1.0 })
        };
        var outputs = new[] { 0.0, 1.0, 2.0 };

        // Train
        for (int epoch = 0; epoch < 30; epoch++)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                bagging.PartialFit(inputs[i], outputs[i]);
            }
        }

        // Act
        var probs = bagging.PredictProbabilities(new Vector<double>(new[] { 0.5, 0.5 }));

        // Assert
        Assert.AreEqual(3, probs.Length);
        var sum = probs[0] + probs[1] + probs[2];
        Assert.IsTrue(sum >= 0.99 && sum <= 1.01); // Probabilities should sum to 1
        Assert.IsTrue(probs.All(p => p >= 0 && p <= 1)); // All probabilities in [0, 1]
    }

    [TestMethod]
    public void Serialize_Deserialize_ShouldPreserveModel()
    {
        // Arrange
        var bagging = new OnlineBagging<double>(
            numFeatures: 2, 
            numClasses: 2,
            baseAlgorithm: OnlineLearningAlgorithm.OnlineSVM,
            ensembleSize: 3,
            isClassification: true);
        
        // Train
        var inputs = new[]
        {
            new Vector<double>(new[] { 1.0, 1.0 }),
            new Vector<double>(new[] { -1.0, -1.0 })
        };
        var outputs = new[] { 1.0, 0.0 };
        
        for (int i = 0; i < 20; i++)
        {
            bagging.PartialFit(inputs[0], outputs[0]);
            bagging.PartialFit(inputs[1], outputs[1]);
        }

        var originalPred1 = bagging.Predict(inputs[0]);
        var originalPred2 = bagging.Predict(inputs[1]);

        // Act
        var serialized = bagging.Serialize();
        var deserialized = new OnlineBagging<double>(2, 2, OnlineLearningAlgorithm.OnlineSVM, 3, true);
        deserialized.Deserialize(serialized);

        // Assert
        Assert.AreEqual(originalPred1, deserialized.Predict(inputs[0]));
        Assert.AreEqual(originalPred2, deserialized.Predict(inputs[1]));
        // SamplesSeen is protected, check through metadata instead
        Assert.AreEqual(bagging.GetModelMetaData().AdditionalInfo["SamplesSeen"], 
                       deserialized.GetModelMetaData().AdditionalInfo["SamplesSeen"]);
    }

    [TestMethod]
    public void GetEnsembleDiversity_ShouldCalculateDiversity()
    {
        // Arrange
        var bagging = new OnlineBagging<double>(
            numFeatures: 2, 
            numClasses: 2,
            baseAlgorithm: OnlineLearningAlgorithm.Perceptron,
            ensembleSize: 5,
            isClassification: true);
        
        // Train with noisy data to encourage diversity
        var random = new Random(42);
        for (int i = 0; i < 50; i++)
        {
            var x = random.NextDouble() * 2 - 1;
            var y = random.NextDouble() * 2 - 1;
            var label = (x + y > 0) ? 1.0 : 0.0;
            
            // Add some noise to labels occasionally
            if (random.NextDouble() < 0.1)
                label = 1.0 - label;
                
            bagging.PartialFit(new Vector<double>(new[] { x, y }), label);
        }

        // Act
        var testInputs = new[]
        {
            new Vector<double>(new[] { 0.5, 0.5 }),
            new Vector<double>(new[] { -0.5, -0.5 }),
            new Vector<double>(new[] { 0.5, -0.5 }),
            new Vector<double>(new[] { -0.5, 0.5 })
        };
        
        var diversity = bagging.GetEnsembleDiversity(testInputs);

        // Assert
        Assert.IsTrue(diversity >= 0.0); // Diversity should be non-negative
        Assert.IsTrue(diversity <= 1.0); // Diversity should be at most 1 (100% disagreement)
    }

    [TestMethod]
    public void SetParameters_ShouldUpdateLearningRate()
    {
        // Arrange
        var bagging = new OnlineBagging<double>(
            numFeatures: 2, 
            numClasses: 2,
            baseAlgorithm: OnlineLearningAlgorithm.PassiveAggressive,
            ensembleSize: 3,
            isClassification: true);

        var initialParams = bagging.GetParameters();
        
        // Act
        var newParams = new Vector<double>(new[] { 3.0, 0.5, 0.01 }); // ensembleSize, learningRate, regularization
        bagging.SetParameters(newParams);
        
        var updatedParams = bagging.GetParameters();

        // Assert
        Assert.AreEqual(0.5, updatedParams[1]); // Learning rate should be updated
        Assert.AreEqual(0.01, updatedParams[2]); // Regularization should be updated
    }

    [TestMethod]
    public void Clone_ShouldCreateIndependentCopy()
    {
        // Arrange
        var bagging = new OnlineBagging<double>(
            numFeatures: 2, 
            numClasses: 2,
            baseAlgorithm: OnlineLearningAlgorithm.ConfidenceWeighted,
            ensembleSize: 3,
            isClassification: true);
        
        // Train original
        var input1 = new Vector<double>(new[] { 1.0, 1.0 });
        var input2 = new Vector<double>(new[] { -1.0, -1.0 });
        
        for (int i = 0; i < 20; i++)
        {
            bagging.PartialFit(input1, 1.0);
            bagging.PartialFit(input2, 0.0);
        }

        // Act
        var clone = bagging.Clone() as OnlineBagging<double>;
        
        // Train original with opposite labels
        for (int i = 0; i < 40; i++)
        {
            bagging.PartialFit(input1, 0.0);
            bagging.PartialFit(input2, 1.0);
        }

        // Assert
        Assert.IsNotNull(clone);
        // Clone should maintain original behavior
        Assert.AreEqual(1.0, clone.Predict(input1));
        Assert.AreEqual(0.0, clone.Predict(input2));
        // Original should have adapted to new pattern
        Assert.AreEqual(0.0, bagging.Predict(input1));
        Assert.AreEqual(1.0, bagging.Predict(input2));
    }

    [TestMethod]
    public void WithParameters_Dictionary_ShouldCreateNewInstance()
    {
        // Arrange
        var bagging = new OnlineBagging<double>(
            numFeatures: 3, 
            numClasses: 2,
            baseAlgorithm: OnlineLearningAlgorithm.HoeffdingTree,
            ensembleSize: 5,
            isClassification: true);
        
        var parameters = new Dictionary<string, object>
        {
            ["EnsembleSize"] = 10,
            ["LearningRate"] = 0.5,
            ["AdaptiveLearningRate"] = true
        };

        // Act
        var newBagging = bagging.WithParameters(parameters) as OnlineBagging<double>;

        // Assert
        Assert.IsNotNull(newBagging);
        Assert.AreNotSame(bagging, newBagging);
        var metadata = newBagging.GetModelMetaData();
        Assert.AreEqual(10, metadata.AdditionalInfo["EnsembleSize"]);
    }
}
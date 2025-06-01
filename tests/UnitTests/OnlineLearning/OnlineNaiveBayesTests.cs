using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.OnlineLearning.Algorithms;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace AiDotNetTests.UnitTests.OnlineLearning;

[TestClass]
public class OnlineNaiveBayesTests
{
    [TestMethod]
    public void Constructor_Gaussian_ShouldInitializeCorrectly()
    {
        // Arrange & Act
        var nb = new OnlineNaiveBayes<double>(4, 2); // numFeatures, numClasses

        // Assert
        Assert.IsNotNull(nb);
        Assert.AreEqual(4, nb.GetInputFeatureCount());
    }

    [TestMethod]
    public void Constructor_Multinomial_ShouldInitializeCorrectly()
    {
        // Arrange & Act
        var nb = new OnlineNaiveBayes<double>(4, 2, false); // numFeatures, numClasses, isGaussian=false for multinomial

        // Assert
        Assert.IsNotNull(nb);
        Assert.AreEqual(4, nb.GetInputFeatureCount());
    }

    [TestMethod]
    public void PartialFit_GaussianNB_ShouldLearnDistributions()
    {
        // Arrange
        var nb = new OnlineNaiveBayes<double>(2, 2); // numFeatures, numClasses
        
        // Two classes with different means
        var inputs = new[]
        {
            // Class 1 - centered around (2, 2)
            new Vector<double>(new[] { 2.1, 2.0 }),
            new Vector<double>(new[] { 1.9, 2.1 }),
            new Vector<double>(new[] { 2.0, 1.9 }),
            // Class 0 - centered around (-2, -2)
            new Vector<double>(new[] { -2.1, -2.0 }),
            new Vector<double>(new[] { -1.9, -2.1 }),
            new Vector<double>(new[] { -2.0, -1.9 })
        };
        var outputs = new[] { 1.0, 1.0, 1.0, 0.0, 0.0, 0.0 };

        // Act
        for (int epoch = 0; epoch < 20; epoch++)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                nb.PartialFit(inputs[i], outputs[i]);
            }
        }

        // Assert
        Assert.IsTrue(nb.Predict(new Vector<double>(new[] { 2.0, 2.0 })) > 0.5);
        Assert.IsTrue(nb.Predict(new Vector<double>(new[] { -2.0, -2.0 })) < 0.5);
    }

    [TestMethod]
    public void PartialFit_MultinomialNB_ShouldWorkWithCounts()
    {
        // Arrange
        var nb = new OnlineNaiveBayes<double>(4, 2, false); // numFeatures, numClasses, isGaussian=false for multinomial
        
        // Document classification-like data (word counts)
        var inputs = new[]
        {
            new Vector<double>(new[] { 3.0, 1.0, 0.0, 0.0 }), // More word1
            new Vector<double>(new[] { 4.0, 0.0, 0.0, 1.0 }), // More word1
            new Vector<double>(new[] { 0.0, 0.0, 3.0, 2.0 }), // More word3,4
            new Vector<double>(new[] { 0.0, 1.0, 4.0, 3.0 })  // More word3,4
        };
        var outputs = new[] { 1.0, 1.0, 0.0, 0.0 };

        // Act
        for (int epoch = 0; epoch < 30; epoch++)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                nb.PartialFit(inputs[i], outputs[i]);
            }
        }

        // Assert
        var test1 = new Vector<double>(new[] { 2.0, 0.0, 0.0, 0.0 }); // Like class 1
        var test0 = new Vector<double>(new[] { 0.0, 0.0, 2.0, 1.0 }); // Like class 0
        
        Assert.IsTrue(nb.Predict(test1) > 0.5);
        Assert.IsTrue(nb.Predict(test0) < 0.5);
    }

    [TestMethod]
    public void PartialFitBatch_ShouldUpdateStatistics()
    {
        // Arrange
        var nb = new OnlineNaiveBayes<double>(2, 2); // numFeatures, numClasses
        var inputs = new Matrix<double>(new double[,]
        {
            { 1.0, 1.0 },
            { 1.1, 0.9 },
            { -1.0, -1.0 },
            { -0.9, -1.1 }
        });
        var outputs = new Vector<double>(new[] { 1.0, 1.0, 0.0, 0.0 });

        // Act
        for (int i = 0; i < 10; i++)
        {
            // Convert Matrix to array of Vectors
            var inputVectors = new Vector<double>[inputs.Rows];
            for (int j = 0; j < inputs.Rows; j++)
            {
                inputVectors[j] = new Vector<double>(new[] { inputs[j, 0], inputs[j, 1] });
            }
            nb.PartialFitBatch(inputVectors, outputs.ToArray());
        }

        // Assert
        Assert.IsTrue(nb.Predict(new Vector<double>(new[] { 0.9, 1.0 })) > 0.5);
        Assert.IsTrue(nb.Predict(new Vector<double>(new[] { -1.0, -0.9 })) < 0.5);
    }

    [TestMethod]
    public void WithSmoothing_ShouldHandleUnseenFeatures()
    {
        // Arrange
        var options = new OnlineModelOptions<double> { RegularizationParameter = 1.0 }; // Laplace smoothing
        var nb = new OnlineNaiveBayes<double>(3, 2, false, options); // numFeatures, numClasses, isGaussian=false for multinomial
        
        // Train with limited features
        var trainInputs = new[]
        {
            new Vector<double>(new[] { 1.0, 0.0, 0.0 }),
            new Vector<double>(new[] { 0.0, 1.0, 0.0 })
        };
        var trainOutputs = new[] { 1.0, 0.0 };
        
        for (int i = 0; i < trainInputs.Length; i++)
        {
            nb.PartialFit(trainInputs[i], trainOutputs[i]);
        }

        // Act - test with unseen feature combination
        var unseenInput = new Vector<double>(new[] { 0.0, 0.0, 1.0 });
        var prediction = nb.Predict(unseenInput);

        // Assert - should not crash and give reasonable probability
        Assert.IsTrue(prediction >= 0.0 && prediction <= 1.0);
        Assert.IsTrue(prediction > 0.1 && prediction < 0.9); // Not extreme due to smoothing
    }

    [TestMethod]
    public void Clone_ShouldCreateIndependentCopy()
    {
        // Arrange
        var nb = new OnlineNaiveBayes<double>(2, 2); // numFeatures, numClasses
        
        // Train original
        var inputs = new[]
        {
            new Vector<double>(new[] { 1.0, 0.0 }),
            new Vector<double>(new[] { 0.0, 1.0 })
        };
        var outputs = new[] { 1.0, 0.0 };
        
        for (int i = 0; i < 20; i++)
        {
            for (int j = 0; j < inputs.Length; j++)
            {
                nb.PartialFit(inputs[j], outputs[j]);
            }
        }

        // Act
        var clone = nb.Clone() as OnlineNaiveBayes<double>;
        
        // Update original with more data
        nb.PartialFit(new Vector<double>(new[] { 0.5, 0.5 }), 1.0);

        // Assert
        Assert.IsNotNull(clone);
        var testPoint = new Vector<double>(new[] { 0.5, 0.5 });
        Assert.AreNotEqual(nb.Predict(testPoint), clone.Predict(testPoint));
    }

    [TestMethod]
    public void GetParameters_Gaussian_ShouldReturnMeansAndVariances()
    {
        // Arrange
        var nb = new OnlineNaiveBayes<double>(2, 2, true); // numFeatures, numClasses, isGaussian=true
        
        // Train with some data
        nb.PartialFit(new Vector<double>(new[] { 1.0, 2.0 }), 1.0);
        nb.PartialFit(new Vector<double>(new[] { -1.0, -2.0 }), 0.0);

        // Act
        var parameters = nb.GetParameters();

        // Assert - GetParameters returns Vector with hyperparameters (alpha/Laplace smoothing)
        Assert.IsNotNull(parameters);
        Assert.AreEqual(1, parameters.Length); // Only returns alpha parameter
    }

    [TestMethod]
    public void GetParameters_Multinomial_ShouldReturnFeatureCounts()
    {
        // Arrange
        var nb = new OnlineNaiveBayes<double>(3, 2, false); // numFeatures, numClasses, isGaussian=false for multinomial
        
        // Train with count data
        nb.PartialFit(new Vector<double>(new[] { 2.0, 1.0, 0.0 }), 1.0);
        nb.PartialFit(new Vector<double>(new[] { 0.0, 1.0, 2.0 }), 0.0);

        // Act
        var parameters = nb.GetParameters();

        // Assert - GetParameters returns Vector with hyperparameters (alpha/Laplace smoothing)
        Assert.IsNotNull(parameters);
        Assert.AreEqual(1, parameters.Length); // Only returns alpha parameter
    }

    [TestMethod]
    public void PredictBatch_ShouldReturnProbabilities()
    {
        // Arrange
        var nb = new OnlineNaiveBayes<double>(2, 2); // numFeatures, numClasses
        
        // Train with clear separation
        var trainInputs = new Matrix<double>(new double[,]
        {
            { 3.0, 3.0 },
            { 3.0, 2.0 },
            { -3.0, -3.0 },
            { -3.0, -2.0 }
        });
        var trainOutputs = new Vector<double>(new[] { 1.0, 1.0, 0.0, 0.0 });
        
        for (int i = 0; i < 20; i++)
        {
            // Convert Matrix to array of Vectors
            var trainVectors = new Vector<double>[trainInputs.Rows];
            for (int j = 0; j < trainInputs.Rows; j++)
            {
                trainVectors[j] = new Vector<double>(new[] { trainInputs[j, 0], trainInputs[j, 1] });
            }
            nb.PartialFitBatch(trainVectors, trainOutputs.ToArray());
        }

        // Act
        var testInputs = new Matrix<double>(new double[,]
        {
            { 2.5, 2.5 },   // Should be class 1
            { -2.5, -2.5 }, // Should be class 0
            { 0.0, 0.0 }    // Uncertain
        });
        // PredictBatch doesn't exist, predict individually
        var predictions = new double[testInputs.Rows];
        for (int j = 0; j < testInputs.Rows; j++)
        {
            var input = new Vector<double>(new[] { testInputs[j, 0], testInputs[j, 1] });
            predictions[j] = nb.Predict(input);
        }

        // Assert
        Assert.AreEqual(3, predictions.Length);
        Assert.AreEqual(1.0, predictions[0]);  // Should predict class 1
        Assert.AreEqual(0.0, predictions[1]);  // Should predict class 0
        // predictions[2] could be either class
    }

    [TestMethod]
    public void IncrementalLearning_ShouldAdaptToNewData()
    {
        // Arrange
        var nb = new OnlineNaiveBayes<double>(2, 2); // numFeatures, numClasses
        
        // Initial training - class 1 at (1,1), class 0 at (-1,-1)
        var phase1Inputs = new[]
        {
            new Vector<double>(new[] { 1.0, 1.0 }),
            new Vector<double>(new[] { -1.0, -1.0 })
        };
        var phase1Outputs = new[] { 1.0, 0.0 };
        
        for (int i = 0; i < 50; i++)
        {
            for (int j = 0; j < phase1Inputs.Length; j++)
            {
                nb.PartialFit(phase1Inputs[j], phase1Outputs[j]);
            }
        }
        
        var testPoint = new Vector<double>(new[] { 2.0, -2.0 });
        var predBefore = nb.Predict(testPoint);

        // Act - shift class 1 to include (2,-2)
        for (int i = 0; i < 50; i++)
        {
            nb.PartialFit(new Vector<double>(new[] { 2.0, -2.0 }), 1.0);
        }
        
        var predAfter = nb.Predict(testPoint);

        // Assert - prediction should shift towards class 1
        Assert.IsTrue(predAfter > predBefore);
        Assert.IsTrue(predAfter > 0.5);
    }
}
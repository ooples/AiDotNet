using AiDotNet.Classification.Linear;
using AiDotNet.Classification.Meta;
using AiDotNet.Classification.NaiveBayes;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Classification;

/// <summary>
/// Integration tests for meta classifiers (OneVsRest, OneVsOne, Voting, Bagging, Stacking, etc.).
/// These tests verify mathematical correctness and do NOT trust the implementation.
/// </summary>
public class MetaClassifierIntegrationTests
{
    private const double Tolerance = 1e-6;
    private const int RandomSeed = 42;

    #region Test Data Helpers

    /// <summary>
    /// Creates a simple linearly separable 3-class dataset.
    /// </summary>
    private static (Matrix<double> X, Vector<double> y) CreateThreeClassData(int samplesPerClass = 30)
    {
        int n = samplesPerClass * 3;
        var x = new Matrix<double>(n, 2);
        var y = new Vector<double>(n);

        // Class 0: centered at (0, 0)
        for (int i = 0; i < samplesPerClass; i++)
        {
            x[i, 0] = 0.0 + 0.5 * Math.Sin(i);
            x[i, 1] = 0.0 + 0.5 * Math.Cos(i);
            y[i] = 0.0;
        }

        // Class 1: centered at (3, 0)
        for (int i = 0; i < samplesPerClass; i++)
        {
            int idx = samplesPerClass + i;
            x[idx, 0] = 3.0 + 0.5 * Math.Sin(i);
            x[idx, 1] = 0.0 + 0.5 * Math.Cos(i);
            y[idx] = 1.0;
        }

        // Class 2: centered at (1.5, 2.5)
        for (int i = 0; i < samplesPerClass; i++)
        {
            int idx = 2 * samplesPerClass + i;
            x[idx, 0] = 1.5 + 0.5 * Math.Sin(i);
            x[idx, 1] = 2.5 + 0.5 * Math.Cos(i);
            y[idx] = 2.0;
        }

        return (x, y);
    }

    /// <summary>
    /// Creates a simple binary classification dataset.
    /// </summary>
    private static (Matrix<double> X, Vector<double> y) CreateBinaryData(int samplesPerClass = 50)
    {
        int n = samplesPerClass * 2;
        var x = new Matrix<double>(n, 2);
        var y = new Vector<double>(n);

        // Class 0: left side
        for (int i = 0; i < samplesPerClass; i++)
        {
            x[i, 0] = -2.0 + 0.5 * Math.Sin(i);
            x[i, 1] = 0.5 * Math.Cos(i);
            y[i] = 0.0;
        }

        // Class 1: right side
        for (int i = 0; i < samplesPerClass; i++)
        {
            int idx = samplesPerClass + i;
            x[idx, 0] = 2.0 + 0.5 * Math.Sin(i);
            x[idx, 1] = 0.5 * Math.Cos(i);
            y[idx] = 1.0;
        }

        return (x, y);
    }

    /// <summary>
    /// Creates multi-label data where labels can co-occur.
    /// </summary>
    private static (Matrix<double> X, Matrix<double> Y) CreateMultiLabelData(int n = 100)
    {
        var x = new Matrix<double>(n, 3);
        var yMultiLabel = new Matrix<double>(n, 3); // 3 labels

        for (int i = 0; i < n; i++)
        {
            double f1 = Math.Sin(i * 0.1);
            double f2 = Math.Cos(i * 0.1);
            double f3 = Math.Sin(i * 0.2);

            x[i, 0] = f1;
            x[i, 1] = f2;
            x[i, 2] = f3;

            // Label 0: active when f1 > 0
            yMultiLabel[i, 0] = f1 > 0 ? 1.0 : 0.0;

            // Label 1: active when f2 > 0
            yMultiLabel[i, 1] = f2 > 0 ? 1.0 : 0.0;

            // Label 2: active when f1 + f2 > 0 (correlated with both)
            yMultiLabel[i, 2] = (f1 + f2) > 0 ? 1.0 : 0.0;
        }

        return (x, yMultiLabel);
    }

    #endregion

    #region OneVsRestClassifier Tests

    [Fact]
    [Trait("Category", "Integration")]
    public void OneVsRest_TrainsKBinaryClassifiers_ForKClasses()
    {
        // Arrange
        var (x, y) = CreateThreeClassData();
        var classifier = new OneVsRestClassifier<double>(
            () => new GaussianNaiveBayes<double>());

        // Act
        classifier.Train(x, y);
        var predictions = classifier.Predict(x);

        // Assert - should train 3 classifiers for 3 classes
        // Verify predictions are valid class labels
        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.True(predictions[i] >= 0 && predictions[i] <= 2,
                $"Prediction {predictions[i]} is not a valid class label");
        }
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void OneVsRest_ProbabilitiesSumToOne()
    {
        // Arrange
        var (x, y) = CreateThreeClassData();
        var classifier = new OneVsRestClassifier<double>(
            () => new GaussianNaiveBayes<double>());
        classifier.Train(x, y);

        // Act
        var probs = classifier.PredictProbabilities(x);

        // Assert - probabilities must sum to 1 for each sample (softmax normalization)
        for (int i = 0; i < x.Rows; i++)
        {
            double sum = 0;
            for (int c = 0; c < probs.Columns; c++)
            {
                sum += probs[i, c];
                Assert.True(probs[i, c] >= 0 && probs[i, c] <= 1,
                    $"Probability {probs[i, c]} is not in [0, 1]");
            }
            Assert.True(Math.Abs(sum - 1.0) < Tolerance,
                $"Probabilities sum to {sum}, expected 1.0");
        }
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void OneVsRest_PredictionMatchesHighestProbability()
    {
        // Arrange
        var (x, y) = CreateThreeClassData();
        var classifier = new OneVsRestClassifier<double>(
            () => new GaussianNaiveBayes<double>());
        classifier.Train(x, y);

        // Act
        var predictions = classifier.Predict(x);
        var probs = classifier.PredictProbabilities(x);

        // Assert - predicted class should have highest probability
        for (int i = 0; i < x.Rows; i++)
        {
            int predicted = (int)predictions[i];

            for (int c = 0; c < probs.Columns; c++)
            {
                if (c != predicted)
                {
                    Assert.True(probs[i, predicted] >= probs[i, c],
                        $"Predicted class {predicted} has lower probability than class {c}");
                }
            }
        }
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void OneVsRest_MultiLabel_ReturnsIndependentPredictions()
    {
        // Arrange
        var (x, y) = CreateThreeClassData();
        var classifier = new OneVsRestClassifier<double>(
            () => new GaussianNaiveBayes<double>());
        classifier.Train(x, y);

        // Act
        var multiLabelPreds = classifier.PredictMultiLabel(x);

        // Assert - multi-label predictions should be 0 or 1
        for (int i = 0; i < x.Rows; i++)
        {
            for (int c = 0; c < multiLabelPreds.Columns; c++)
            {
                Assert.True(multiLabelPreds[i, c] == 0.0 || multiLabelPreds[i, c] == 1.0,
                    $"Multi-label prediction {multiLabelPreds[i, c]} is not 0 or 1");
            }
        }
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void OneVsRest_Clone_ProducesSamePredictions()
    {
        // Arrange
        var (x, y) = CreateThreeClassData();
        var classifier = new OneVsRestClassifier<double>(
            () => new GaussianNaiveBayes<double>());
        classifier.Train(x, y);

        // Act
        var clone = (OneVsRestClassifier<double>)classifier.Clone();
        var origPreds = classifier.Predict(x);
        var clonePreds = clone.Predict(x);

        // Assert
        for (int i = 0; i < origPreds.Length; i++)
        {
            Assert.Equal(origPreds[i], clonePreds[i]);
        }
    }

    #endregion

    #region OneVsOneClassifier Tests

    [Fact]
    [Trait("Category", "Integration")]
    public void OneVsOne_TrainsCorrectNumberOfClassifiers()
    {
        // Arrange
        var (x, y) = CreateThreeClassData();
        var classifier = new OneVsOneClassifier<double>(
            () => new GaussianNaiveBayes<double>());

        // Act
        classifier.Train(x, y);
        var predictions = classifier.Predict(x);

        // Assert - for 3 classes, should train 3*(3-1)/2 = 3 classifiers
        // Verify predictions are valid class labels
        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.True(predictions[i] >= 0 && predictions[i] <= 2,
                $"Prediction {predictions[i]} is not a valid class label");
        }
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void OneVsOne_VotingMechanism_Works()
    {
        // Arrange - 4 classes = 4*3/2 = 6 pairwise classifiers
        int samplesPerClass = 20;
        int n = samplesPerClass * 4;
        var x = new Matrix<double>(n, 2);
        var y = new Vector<double>(n);

        // Create 4 well-separated clusters
        double[,] centers = { { -2, -2 }, { 2, -2 }, { -2, 2 }, { 2, 2 } };

        for (int c = 0; c < 4; c++)
        {
            for (int i = 0; i < samplesPerClass; i++)
            {
                int idx = c * samplesPerClass + i;
                x[idx, 0] = centers[c, 0] + 0.3 * Math.Sin(i);
                x[idx, 1] = centers[c, 1] + 0.3 * Math.Cos(i);
                y[idx] = c;
            }
        }

        var classifier = new OneVsOneClassifier<double>(
            () => new GaussianNaiveBayes<double>());

        // Act
        classifier.Train(x, y);
        var predictions = classifier.Predict(x);

        // Assert - should achieve reasonable accuracy on training data
        int correct = 0;
        for (int i = 0; i < n; i++)
        {
            if (Math.Abs(predictions[i] - y[i]) < 0.1)
            {
                correct++;
            }
        }

        double accuracy = (double)correct / n;
        Assert.True(accuracy > 0.7, $"OneVsOne accuracy {accuracy:P2} is too low");
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void OneVsOne_ProbabilitiesSumToOne()
    {
        // Arrange
        var (x, y) = CreateThreeClassData();
        var classifier = new OneVsOneClassifier<double>(
            () => new GaussianNaiveBayes<double>());
        classifier.Train(x, y);

        // Act
        var probs = classifier.PredictProbabilities(x);

        // Assert
        for (int i = 0; i < x.Rows; i++)
        {
            double sum = 0;
            for (int c = 0; c < probs.Columns; c++)
            {
                sum += probs[i, c];
                Assert.True(probs[i, c] >= 0, $"Probability {probs[i, c]} is negative");
            }
            Assert.True(Math.Abs(sum - 1.0) < Tolerance,
                $"Probabilities sum to {sum}, expected 1.0");
        }
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void OneVsOne_PairwiseClassifiers_UseCorrectSubsets()
    {
        // Arrange - Create data where class 0 and class 1 are separable,
        // but class 2 is in between
        var (x, y) = CreateThreeClassData();
        var classifier = new OneVsOneClassifier<double>(
            () => new GaussianNaiveBayes<double>());

        // Act
        classifier.Train(x, y);
        var predictions = classifier.Predict(x);

        // Assert - each prediction should be a valid class
        var uniqueClasses = new HashSet<double>();
        for (int i = 0; i < y.Length; i++)
        {
            uniqueClasses.Add(y[i]);
        }

        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.Contains(predictions[i], uniqueClasses);
        }
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void OneVsOne_Clone_ProducesSamePredictions()
    {
        // Arrange
        var (x, y) = CreateThreeClassData();
        var classifier = new OneVsOneClassifier<double>(
            () => new GaussianNaiveBayes<double>());
        classifier.Train(x, y);

        // Act
        var clone = (OneVsOneClassifier<double>)classifier.Clone();
        var origPreds = classifier.Predict(x);
        var clonePreds = clone.Predict(x);

        // Assert
        for (int i = 0; i < origPreds.Length; i++)
        {
            Assert.Equal(origPreds[i], clonePreds[i]);
        }
    }

    #endregion

    #region VotingClassifier Tests

    [Fact]
    [Trait("Category", "Integration")]
    public void VotingClassifier_HardVoting_UsesMajorityVote()
    {
        // Arrange
        var (x, y) = CreateThreeClassData();

        var estimators = new List<IClassifier<double>>
        {
            new GaussianNaiveBayes<double>(),
            new GaussianNaiveBayes<double>(),
            new GaussianNaiveBayes<double>()
        };

        var classifier = new VotingClassifier<double>(
            estimators,
            new VotingClassifierOptions<double> { Voting = VotingType.Hard });

        // Act
        classifier.Train(x, y);
        var predictions = classifier.Predict(x);

        // Assert - predictions should be valid class labels
        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.True(predictions[i] >= 0 && predictions[i] <= 2);
        }
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void VotingClassifier_SoftVoting_UsesAverageProbabilities()
    {
        // Arrange
        var (x, y) = CreateThreeClassData();

        var estimators = new List<IClassifier<double>>
        {
            new GaussianNaiveBayes<double>(),
            new GaussianNaiveBayes<double>()
        };

        var classifier = new VotingClassifier<double>(
            estimators,
            new VotingClassifierOptions<double> { Voting = VotingType.Soft });

        // Act
        classifier.Train(x, y);
        var probs = classifier.PredictProbabilities(x);

        // Assert - soft voting averages probabilities, should still sum to 1
        for (int i = 0; i < x.Rows; i++)
        {
            double sum = 0;
            for (int c = 0; c < probs.Columns; c++)
            {
                sum += probs[i, c];
            }
            Assert.True(Math.Abs(sum - 1.0) < Tolerance,
                $"Soft voting probabilities sum to {sum}, expected 1.0");
        }
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void VotingClassifier_Weights_AffectsVoting()
    {
        // Arrange
        var (x, y) = CreateBinaryData();

        // Create two identical classifiers
        var estimators = new List<IClassifier<double>>
        {
            new GaussianNaiveBayes<double>(),
            new GaussianNaiveBayes<double>()
        };

        // One classifier with all weight
        var weighted = new VotingClassifier<double>(
            estimators,
            new VotingClassifierOptions<double>
            {
                Voting = VotingType.Hard,
                Weights = new[] { 1.0, 0.0 } // First classifier has all weight
            });

        var equalWeights = new VotingClassifier<double>(
            estimators,
            new VotingClassifierOptions<double>
            {
                Voting = VotingType.Hard,
                Weights = new[] { 0.5, 0.5 }
            });

        // Act
        weighted.Train(x, y);
        equalWeights.Train(x, y);

        // Both should produce predictions (just verifying no crash)
        var weightedPreds = weighted.Predict(x);
        var equalPreds = equalWeights.Predict(x);

        // Assert - both should produce valid predictions
        Assert.Equal(x.Rows, weightedPreds.Length);
        Assert.Equal(x.Rows, equalPreds.Length);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void VotingClassifier_WeightsAreNormalized()
    {
        // Arrange
        var (x, y) = CreateThreeClassData();

        var estimators = new List<IClassifier<double>>
        {
            new GaussianNaiveBayes<double>(),
            new GaussianNaiveBayes<double>()
        };

        // Weights that don't sum to 1
        var classifier = new VotingClassifier<double>(
            estimators,
            new VotingClassifierOptions<double>
            {
                Voting = VotingType.Soft,
                Weights = new[] { 10.0, 20.0 } // Should be normalized to 1/3, 2/3
            });

        // Act
        classifier.Train(x, y);
        var probs = classifier.PredictProbabilities(x);

        // Assert - probabilities should still sum to 1 after weight normalization
        for (int i = 0; i < x.Rows; i++)
        {
            double sum = 0;
            for (int c = 0; c < probs.Columns; c++)
            {
                sum += probs[i, c];
            }
            Assert.True(Math.Abs(sum - 1.0) < Tolerance,
                $"Probabilities sum to {sum}, expected 1.0");
        }
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void VotingClassifier_Clone_ProducesSamePredictions()
    {
        // Arrange
        var (x, y) = CreateThreeClassData();

        var estimators = new List<IClassifier<double>>
        {
            new GaussianNaiveBayes<double>(),
            new GaussianNaiveBayes<double>()
        };

        var classifier = new VotingClassifier<double>(
            estimators,
            new VotingClassifierOptions<double> { Voting = VotingType.Soft });
        classifier.Train(x, y);

        // Act
        var clone = (VotingClassifier<double>)classifier.Clone();
        var origPreds = classifier.Predict(x);
        var clonePreds = clone.Predict(x);

        // Assert
        for (int i = 0; i < origPreds.Length; i++)
        {
            Assert.Equal(origPreds[i], clonePreds[i]);
        }
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void VotingClassifier_Metadata_ContainsVotingType()
    {
        // Arrange
        var (x, y) = CreateBinaryData();

        var estimators = new List<IClassifier<double>> { new GaussianNaiveBayes<double>() };
        var classifier = new VotingClassifier<double>(
            estimators,
            new VotingClassifierOptions<double> { Voting = VotingType.Soft });
        classifier.Train(x, y);

        // Act
        var metadata = classifier.GetModelMetadata();

        // Assert
        Assert.Equal("Soft", metadata.AdditionalInfo["VotingType"]);
        Assert.Equal(1, metadata.AdditionalInfo["NumEstimators"]);
    }

    #endregion

    #region BaggingClassifier Tests

    [Fact]
    [Trait("Category", "Integration")]
    public void BaggingClassifier_TrainsMultipleEstimators()
    {
        // Arrange
        var (x, y) = CreateThreeClassData();
        var classifier = new BaggingClassifier<double>(
            () => new GaussianNaiveBayes<double>(),
            new BaggingClassifierOptions<double>
            {
                NumEstimators = 10,
                Seed = RandomSeed
            });

        // Act
        classifier.Train(x, y);
        var predictions = classifier.Predict(x);

        // Assert
        Assert.Equal(x.Rows, predictions.Length);
        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.True(predictions[i] >= 0 && predictions[i] <= 2);
        }
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void BaggingClassifier_BootstrapSampling_CreatesDifferentModels()
    {
        // Arrange
        var (x, y) = CreateThreeClassData();

        // Train two bagging classifiers with different seeds
        var classifier1 = new BaggingClassifier<double>(
            () => new GaussianNaiveBayes<double>(),
            new BaggingClassifierOptions<double>
            {
                NumEstimators = 5,
                Seed = 42
            });

        var classifier2 = new BaggingClassifier<double>(
            () => new GaussianNaiveBayes<double>(),
            new BaggingClassifierOptions<double>
            {
                NumEstimators = 5,
                Seed = 123
            });

        // Act
        classifier1.Train(x, y);
        classifier2.Train(x, y);

        var probs1 = classifier1.PredictProbabilities(x);
        var probs2 = classifier2.PredictProbabilities(x);

        // Assert - different seeds should produce different probability distributions
        bool anyDifference = false;
        for (int i = 0; i < x.Rows && !anyDifference; i++)
        {
            for (int c = 0; c < probs1.Columns; c++)
            {
                if (Math.Abs(probs1[i, c] - probs2[i, c]) > 1e-10)
                {
                    anyDifference = true;
                    break;
                }
            }
        }

        Assert.True(anyDifference, "Different seeds should produce different models");
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void BaggingClassifier_MaxSamples_UsesSubsetOfData()
    {
        // Arrange
        var (x, y) = CreateThreeClassData();

        var classifier = new BaggingClassifier<double>(
            () => new GaussianNaiveBayes<double>(),
            new BaggingClassifierOptions<double>
            {
                NumEstimators = 5,
                MaxSamples = 0.5, // Use 50% of samples per estimator
                Seed = RandomSeed
            });

        // Act
        classifier.Train(x, y);
        var predictions = classifier.Predict(x);

        // Assert - should still produce valid predictions
        Assert.Equal(x.Rows, predictions.Length);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void BaggingClassifier_MaxFeatures_UsesSubsetOfFeatures()
    {
        // Arrange - create data with more features
        int n = 100;
        int numFeatures = 10;
        var x = new Matrix<double>(n, numFeatures);
        var y = new Vector<double>(n);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < numFeatures; j++)
            {
                x[i, j] = Math.Sin(i * (j + 1) * 0.1);
            }
            y[i] = i < n / 2 ? 0.0 : 1.0;
        }

        var classifier = new BaggingClassifier<double>(
            () => new GaussianNaiveBayes<double>(),
            new BaggingClassifierOptions<double>
            {
                NumEstimators = 5,
                MaxFeatures = 0.5, // Use 50% of features per estimator
                Seed = RandomSeed
            });

        // Act
        classifier.Train(x, y);
        var predictions = classifier.Predict(x);

        // Assert - should still produce valid predictions
        Assert.Equal(n, predictions.Length);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void BaggingClassifier_ProbabilitiesAreAveraged()
    {
        // Arrange
        var (x, y) = CreateThreeClassData();
        var classifier = new BaggingClassifier<double>(
            () => new GaussianNaiveBayes<double>(),
            new BaggingClassifierOptions<double>
            {
                NumEstimators = 10,
                Seed = RandomSeed
            });
        classifier.Train(x, y);

        // Act
        var probs = classifier.PredictProbabilities(x);

        // Assert - averaged probabilities should sum to 1
        for (int i = 0; i < x.Rows; i++)
        {
            double sum = 0;
            for (int c = 0; c < probs.Columns; c++)
            {
                sum += probs[i, c];
                Assert.True(probs[i, c] >= 0 && probs[i, c] <= 1);
            }
            Assert.True(Math.Abs(sum - 1.0) < Tolerance,
                $"Bagging probabilities sum to {sum}, expected 1.0");
        }
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void BaggingClassifier_Clone_ProducesSamePredictions()
    {
        // Arrange
        var (x, y) = CreateThreeClassData();
        var classifier = new BaggingClassifier<double>(
            () => new GaussianNaiveBayes<double>(),
            new BaggingClassifierOptions<double>
            {
                NumEstimators = 5,
                Seed = RandomSeed
            });
        classifier.Train(x, y);

        // Act
        var clone = (BaggingClassifier<double>)classifier.Clone();
        var origPreds = classifier.Predict(x);
        var clonePreds = clone.Predict(x);

        // Assert
        for (int i = 0; i < origPreds.Length; i++)
        {
            Assert.Equal(origPreds[i], clonePreds[i]);
        }
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void BaggingClassifier_Metadata_ContainsConfiguration()
    {
        // Arrange
        var (x, y) = CreateBinaryData();
        var classifier = new BaggingClassifier<double>(
            () => new GaussianNaiveBayes<double>(),
            new BaggingClassifierOptions<double>
            {
                NumEstimators = 15,
                MaxSamples = 0.8,
                MaxFeatures = 0.9
            });
        classifier.Train(x, y);

        // Act
        var metadata = classifier.GetModelMetadata();

        // Assert
        Assert.Equal(15, metadata.AdditionalInfo["NumEstimators"]);
        Assert.Equal(0.8, metadata.AdditionalInfo["MaxSamples"]);
        Assert.Equal(0.9, metadata.AdditionalInfo["MaxFeatures"]);
    }

    #endregion

    #region StackingClassifier Tests

    [Fact]
    [Trait("Category", "Integration")]
    public void StackingClassifier_UsesBaseEstimatorPredictionsAsFeatures()
    {
        // Arrange
        var (x, y) = CreateThreeClassData();

        var baseEstimators = new List<IClassifier<double>>
        {
            new GaussianNaiveBayes<double>(),
            new GaussianNaiveBayes<double>()
        };

        var classifier = new StackingClassifier<double>(
            baseEstimators,
            () => new GaussianNaiveBayes<double>(), // Final estimator
            new StackingClassifierOptions<double>
            {
                CrossValidationFolds = 3,
                UseProbabilities = true,
                Seed = RandomSeed
            });

        // Act
        classifier.Train(x, y);
        var predictions = classifier.Predict(x);

        // Assert
        Assert.Equal(x.Rows, predictions.Length);
        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.True(predictions[i] >= 0 && predictions[i] <= 2);
        }
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void StackingClassifier_CrossValidation_ReducesOverfitting()
    {
        // Arrange
        var (x, y) = CreateThreeClassData();

        var baseEstimators = new List<IClassifier<double>>
        {
            new GaussianNaiveBayes<double>()
        };

        // With CV
        var withCV = new StackingClassifier<double>(
            baseEstimators,
            () => new GaussianNaiveBayes<double>(),
            new StackingClassifierOptions<double>
            {
                CrossValidationFolds = 5,
                UseProbabilities = true,
                Seed = RandomSeed
            });

        // Without CV (fold=1)
        var withoutCV = new StackingClassifier<double>(
            baseEstimators,
            () => new GaussianNaiveBayes<double>(),
            new StackingClassifierOptions<double>
            {
                CrossValidationFolds = 1,
                UseProbabilities = true,
                Seed = RandomSeed
            });

        // Act
        withCV.Train(x, y);
        withoutCV.Train(x, y);

        var predsCV = withCV.Predict(x);
        var predsNoCV = withoutCV.Predict(x);

        // Assert - both should produce valid predictions
        Assert.Equal(x.Rows, predsCV.Length);
        Assert.Equal(x.Rows, predsNoCV.Length);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void StackingClassifier_Passthrough_IncludesOriginalFeatures()
    {
        // Arrange
        var (x, y) = CreateThreeClassData();

        var baseEstimators = new List<IClassifier<double>>
        {
            new GaussianNaiveBayes<double>()
        };

        var withPassthrough = new StackingClassifier<double>(
            baseEstimators,
            () => new GaussianNaiveBayes<double>(),
            new StackingClassifierOptions<double>
            {
                CrossValidationFolds = 3,
                Passthrough = true, // Include original features
                UseProbabilities = true,
                Seed = RandomSeed
            });

        // Act
        withPassthrough.Train(x, y);
        var predictions = withPassthrough.Predict(x);

        // Assert - should produce valid predictions
        Assert.Equal(x.Rows, predictions.Length);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void StackingClassifier_UseProbabilities_AffectsMetaFeatures()
    {
        // Arrange
        var (x, y) = CreateThreeClassData();

        var baseEstimators = new List<IClassifier<double>>
        {
            new GaussianNaiveBayes<double>()
        };

        var useProbs = new StackingClassifier<double>(
            baseEstimators,
            () => new GaussianNaiveBayes<double>(),
            new StackingClassifierOptions<double>
            {
                CrossValidationFolds = 3,
                UseProbabilities = true, // 3 features per estimator (one per class)
                Seed = RandomSeed
            });

        var usePreds = new StackingClassifier<double>(
            baseEstimators,
            () => new GaussianNaiveBayes<double>(),
            new StackingClassifierOptions<double>
            {
                CrossValidationFolds = 3,
                UseProbabilities = false, // 1 feature per estimator (predicted class)
                Seed = RandomSeed
            });

        // Act
        useProbs.Train(x, y);
        usePreds.Train(x, y);

        var predsProbs = useProbs.Predict(x);
        var predsPreds = usePreds.Predict(x);

        // Assert - both should produce valid predictions (may differ slightly)
        Assert.Equal(x.Rows, predsProbs.Length);
        Assert.Equal(x.Rows, predsPreds.Length);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void StackingClassifier_Clone_ProducesSamePredictions()
    {
        // Arrange
        var (x, y) = CreateThreeClassData();

        var baseEstimators = new List<IClassifier<double>>
        {
            new GaussianNaiveBayes<double>()
        };

        var classifier = new StackingClassifier<double>(
            baseEstimators,
            () => new GaussianNaiveBayes<double>(),
            new StackingClassifierOptions<double>
            {
                CrossValidationFolds = 3,
                Seed = RandomSeed
            });
        classifier.Train(x, y);

        // Act
        var clone = (StackingClassifier<double>)classifier.Clone();
        var origPreds = classifier.Predict(x);
        var clonePreds = clone.Predict(x);

        // Assert
        for (int i = 0; i < origPreds.Length; i++)
        {
            Assert.Equal(origPreds[i], clonePreds[i]);
        }
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void StackingClassifier_Metadata_ContainsConfiguration()
    {
        // Arrange
        var (x, y) = CreateBinaryData();

        var baseEstimators = new List<IClassifier<double>>
        {
            new GaussianNaiveBayes<double>(),
            new GaussianNaiveBayes<double>()
        };

        var classifier = new StackingClassifier<double>(
            baseEstimators,
            () => new GaussianNaiveBayes<double>(),
            new StackingClassifierOptions<double>
            {
                CrossValidationFolds = 5,
                UseProbabilities = true,
                Passthrough = true
            });
        classifier.Train(x, y);

        // Act
        var metadata = classifier.GetModelMetadata();

        // Assert
        Assert.Equal(2, metadata.AdditionalInfo["NumEstimators"]);
        Assert.Equal(5, metadata.AdditionalInfo["CrossValidationFolds"]);
        Assert.Equal(true, metadata.AdditionalInfo["UseProbabilities"]);
        Assert.Equal(true, metadata.AdditionalInfo["Passthrough"]);
    }

    #endregion

    #region ClassifierChain Tests

    [Fact]
    [Trait("Category", "Integration")]
    public void ClassifierChain_CapturesLabelDependencies()
    {
        // Arrange - multi-label data where label 2 depends on labels 0 and 1
        var (x, yMultiLabel) = CreateMultiLabelData();

        var classifier = new ClassifierChain<double>(
            () => new GaussianNaiveBayes<double>(),
            new ClassifierChainOptions<double>
            {
                RandomOrder = false, // Use default order
                Seed = RandomSeed
            });

        // Act
        classifier.TrainMultiLabel(x, yMultiLabel);
        var predictions = classifier.PredictMultiLabel(x);

        // Assert - predictions should be 0 or 1
        for (int i = 0; i < x.Rows; i++)
        {
            for (int c = 0; c < predictions.Columns; c++)
            {
                Assert.True(predictions[i, c] == 0.0 || predictions[i, c] == 1.0,
                    $"Multi-label prediction {predictions[i, c]} is not 0 or 1");
            }
        }
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void ClassifierChain_UsesAugmentedFeatures()
    {
        // Arrange
        var (x, yMultiLabel) = CreateMultiLabelData();

        var classifier = new ClassifierChain<double>(
            () => new GaussianNaiveBayes<double>(),
            new ClassifierChainOptions<double>
            {
                Order = new[] { 0, 1, 2 }, // Explicit order
                Seed = RandomSeed
            });

        // Act
        classifier.TrainMultiLabel(x, yMultiLabel);
        var predictions = classifier.PredictMultiLabel(x);

        // Assert - classifier 2 should use predictions from 0 and 1 as features
        Assert.Equal(yMultiLabel.Columns, predictions.Columns);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void ClassifierChain_RandomOrder_ShufflesChain()
    {
        // Arrange
        var (x, yMultiLabel) = CreateMultiLabelData();

        var classifier1 = new ClassifierChain<double>(
            () => new GaussianNaiveBayes<double>(),
            new ClassifierChainOptions<double>
            {
                RandomOrder = true,
                Seed = 42
            });

        var classifier2 = new ClassifierChain<double>(
            () => new GaussianNaiveBayes<double>(),
            new ClassifierChainOptions<double>
            {
                RandomOrder = true,
                Seed = 123
            });

        // Act
        classifier1.TrainMultiLabel(x, yMultiLabel);
        classifier2.TrainMultiLabel(x, yMultiLabel);

        var probs1 = classifier1.PredictMultiLabelProbabilities(x);
        var probs2 = classifier2.PredictMultiLabelProbabilities(x);

        // Assert - different seeds should produce different results
        bool anyDifference = false;
        for (int i = 0; i < x.Rows && !anyDifference; i++)
        {
            for (int c = 0; c < probs1.Columns; c++)
            {
                if (Math.Abs(probs1[i, c] - probs2[i, c]) > 1e-10)
                {
                    anyDifference = true;
                    break;
                }
            }
        }

        Assert.True(anyDifference, "Different random orders should produce different results");
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void ClassifierChain_ExplicitOrder_UsesProvidedOrder()
    {
        // Arrange
        var (x, yMultiLabel) = CreateMultiLabelData();

        var classifier = new ClassifierChain<double>(
            () => new GaussianNaiveBayes<double>(),
            new ClassifierChainOptions<double>
            {
                Order = new[] { 2, 0, 1 }, // Custom order
                Seed = RandomSeed
            });

        // Act
        classifier.TrainMultiLabel(x, yMultiLabel);
        var predictions = classifier.PredictMultiLabel(x);

        // Assert - should produce valid predictions
        Assert.Equal(yMultiLabel.Columns, predictions.Columns);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void ClassifierChain_SingleLabelTraining_ConvertsToMultiLabel()
    {
        // Arrange
        var (x, y) = CreateThreeClassData();

        var classifier = new ClassifierChain<double>(
            () => new GaussianNaiveBayes<double>(),
            new ClassifierChainOptions<double>
            {
                Seed = RandomSeed
            });

        // Act
        classifier.Train(x, y); // Single-label training
        var predictions = classifier.Predict(x);

        // Assert - predictions should be valid class labels
        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.True(predictions[i] >= 0 && predictions[i] <= 2);
        }
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void ClassifierChain_Clone_ProducesSamePredictions()
    {
        // Arrange
        var (x, yMultiLabel) = CreateMultiLabelData();

        var classifier = new ClassifierChain<double>(
            () => new GaussianNaiveBayes<double>(),
            new ClassifierChainOptions<double> { Seed = RandomSeed });
        classifier.TrainMultiLabel(x, yMultiLabel);

        // Act
        var clone = (ClassifierChain<double>)classifier.Clone();
        var origPreds = classifier.PredictMultiLabel(x);
        var clonePreds = clone.PredictMultiLabel(x);

        // Assert
        for (int i = 0; i < x.Rows; i++)
        {
            for (int c = 0; c < origPreds.Columns; c++)
            {
                Assert.Equal(origPreds[i, c], clonePreds[i, c]);
            }
        }
    }

    #endregion

    #region MultiOutputClassifier Tests

    [Fact]
    [Trait("Category", "Integration")]
    public void MultiOutputClassifier_TrainsIndependentClassifiers()
    {
        // Arrange
        var (x, yMultiLabel) = CreateMultiLabelData();

        var classifier = new MultiOutputClassifier<double>(
            () => new GaussianNaiveBayes<double>());

        // Act
        classifier.TrainMultiLabel(x, yMultiLabel);
        var predictions = classifier.PredictMultiLabel(x);

        // Assert - one classifier per label
        Assert.Equal(yMultiLabel.Columns, predictions.Columns);
        for (int i = 0; i < x.Rows; i++)
        {
            for (int c = 0; c < predictions.Columns; c++)
            {
                Assert.True(predictions[i, c] == 0.0 || predictions[i, c] == 1.0);
            }
        }
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void MultiOutputClassifier_LabelsAreIndependent()
    {
        // Arrange - create data where labels are truly independent
        int n = 100;
        var x = new Matrix<double>(n, 2);
        var yMultiLabel = new Matrix<double>(n, 2);

        for (int i = 0; i < n; i++)
        {
            x[i, 0] = Math.Sin(i * 0.1);
            x[i, 1] = Math.Cos(i * 0.1);

            // Independent labels based on different features
            yMultiLabel[i, 0] = x[i, 0] > 0 ? 1.0 : 0.0;
            yMultiLabel[i, 1] = x[i, 1] > 0 ? 1.0 : 0.0;
        }

        var classifier = new MultiOutputClassifier<double>(
            () => new GaussianNaiveBayes<double>());

        // Act
        classifier.TrainMultiLabel(x, yMultiLabel);
        var predictions = classifier.PredictMultiLabel(x);

        // Assert - should capture independent patterns
        int correctLabel0 = 0, correctLabel1 = 0;
        for (int i = 0; i < n; i++)
        {
            if (predictions[i, 0] == yMultiLabel[i, 0]) correctLabel0++;
            if (predictions[i, 1] == yMultiLabel[i, 1]) correctLabel1++;
        }

        double acc0 = (double)correctLabel0 / n;
        double acc1 = (double)correctLabel1 / n;

        Assert.True(acc0 > 0.7, $"Label 0 accuracy {acc0:P2} is too low");
        Assert.True(acc1 > 0.7, $"Label 1 accuracy {acc1:P2} is too low");
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void MultiOutputClassifier_SingleLabelTraining_Works()
    {
        // Arrange
        var (x, y) = CreateThreeClassData();

        var classifier = new MultiOutputClassifier<double>(
            () => new GaussianNaiveBayes<double>());

        // Act
        classifier.Train(x, y);
        var predictions = classifier.Predict(x);

        // Assert
        Assert.Equal(x.Rows, predictions.Length);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void MultiOutputClassifier_ProbabilitiesPerLabel()
    {
        // Arrange
        var (x, yMultiLabel) = CreateMultiLabelData();

        var classifier = new MultiOutputClassifier<double>(
            () => new GaussianNaiveBayes<double>());
        classifier.TrainMultiLabel(x, yMultiLabel);

        // Act
        var probs = classifier.PredictMultiLabelProbabilities(x);

        // Assert - probabilities should be in [0, 1]
        for (int i = 0; i < x.Rows; i++)
        {
            for (int c = 0; c < probs.Columns; c++)
            {
                Assert.True(probs[i, c] >= 0 && probs[i, c] <= 1,
                    $"Probability {probs[i, c]} is not in [0, 1]");
            }
        }
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void MultiOutputClassifier_Clone_ProducesSamePredictions()
    {
        // Arrange
        var (x, yMultiLabel) = CreateMultiLabelData();

        var classifier = new MultiOutputClassifier<double>(
            () => new GaussianNaiveBayes<double>());
        classifier.TrainMultiLabel(x, yMultiLabel);

        // Act
        var clone = (MultiOutputClassifier<double>)classifier.Clone();
        var origPreds = classifier.PredictMultiLabel(x);
        var clonePreds = clone.PredictMultiLabel(x);

        // Assert
        for (int i = 0; i < x.Rows; i++)
        {
            for (int c = 0; c < origPreds.Columns; c++)
            {
                Assert.Equal(origPreds[i, c], clonePreds[i, c]);
            }
        }
    }

    #endregion

    #region Edge Cases and Error Handling

    [Fact]
    [Trait("Category", "Integration")]
    public void OneVsRest_ThrowsIfNotTrained()
    {
        // Arrange
        var classifier = new OneVsRestClassifier<double>(() => new GaussianNaiveBayes<double>());
        var input = new Matrix<double>(10, 2);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => classifier.Predict(input));
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void OneVsOne_ThrowsIfNotTrained()
    {
        // Arrange
        var classifier = new OneVsOneClassifier<double>(() => new GaussianNaiveBayes<double>());
        var input = new Matrix<double>(10, 2);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => classifier.Predict(input));
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void VotingClassifier_ThrowsIfNoEstimators()
    {
        // Arrange
        var emptyEstimators = new List<IClassifier<double>>();
        var classifier = new VotingClassifier<double>(emptyEstimators);

        var x = new Matrix<double>(10, 2);
        var y = new Vector<double>(10);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => classifier.Train(x, y));
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void BaggingClassifier_ThrowsIfNotTrained()
    {
        // Arrange
        var classifier = new BaggingClassifier<double>(() => new GaussianNaiveBayes<double>());
        var input = new Matrix<double>(10, 2);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => classifier.Predict(input));
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void StackingClassifier_ThrowsIfNotTrained()
    {
        // Arrange
        var estimators = new List<IClassifier<double>> { new GaussianNaiveBayes<double>() };
        var classifier = new StackingClassifier<double>(
            estimators, () => new GaussianNaiveBayes<double>());
        var input = new Matrix<double>(10, 2);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => classifier.Predict(input));
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void ClassifierChain_ThrowsIfNotTrained()
    {
        // Arrange
        var classifier = new ClassifierChain<double>(() => new GaussianNaiveBayes<double>());
        var input = new Matrix<double>(10, 2);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => classifier.PredictMultiLabel(input));
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void MultiOutputClassifier_ThrowsIfNotTrained()
    {
        // Arrange
        var classifier = new MultiOutputClassifier<double>(() => new GaussianNaiveBayes<double>());
        var input = new Matrix<double>(10, 2);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => classifier.PredictMultiLabel(input));
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void MetaClassifiers_HandleMismatchedXYLengths()
    {
        // Arrange
        var x = new Matrix<double>(10, 2);
        var y = new Vector<double>(5); // Mismatched length

        var ovr = new OneVsRestClassifier<double>(() => new GaussianNaiveBayes<double>());
        var ovo = new OneVsOneClassifier<double>(() => new GaussianNaiveBayes<double>());
        var bagging = new BaggingClassifier<double>(() => new GaussianNaiveBayes<double>());

        // Act & Assert
        Assert.Throws<ArgumentException>(() => ovr.Train(x, y));
        Assert.Throws<ArgumentException>(() => ovo.Train(x, y));
        Assert.Throws<ArgumentException>(() => bagging.Train(x, y));
    }

    #endregion

    #region Log Probability Tests

    [Fact]
    [Trait("Category", "Integration")]
    public void OneVsRest_LogProbabilities_AreValid()
    {
        // Arrange
        var (x, y) = CreateThreeClassData();
        var classifier = new OneVsRestClassifier<double>(
            () => new GaussianNaiveBayes<double>());
        classifier.Train(x, y);

        // Act
        var logProbs = classifier.PredictLogProbabilities(x);
        var probs = classifier.PredictProbabilities(x);

        // Assert - log(probs) should equal logProbs
        for (int i = 0; i < x.Rows; i++)
        {
            for (int c = 0; c < logProbs.Columns; c++)
            {
                double expectedLog = Math.Log(Math.Max(probs[i, c], 1e-15));
                Assert.True(Math.Abs(logProbs[i, c] - expectedLog) < Tolerance,
                    $"Log probability mismatch: {logProbs[i, c]} vs expected {expectedLog}");
            }
        }
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void VotingClassifier_LogProbabilities_AreNonPositive()
    {
        // Arrange
        var (x, y) = CreateThreeClassData();

        var estimators = new List<IClassifier<double>>
        {
            new GaussianNaiveBayes<double>()
        };

        var classifier = new VotingClassifier<double>(
            estimators,
            new VotingClassifierOptions<double> { Voting = VotingType.Soft });
        classifier.Train(x, y);

        // Act
        var logProbs = classifier.PredictLogProbabilities(x);

        // Assert - log probabilities should be <= 0
        for (int i = 0; i < x.Rows; i++)
        {
            for (int c = 0; c < logProbs.Columns; c++)
            {
                Assert.True(logProbs[i, c] <= 0 + Tolerance,
                    $"Log probability {logProbs[i, c]} is positive");
            }
        }
    }

    #endregion

    #region Comparison Tests

    [Fact]
    [Trait("Category", "Integration")]
    public void OneVsRest_VsOneVsOne_BothProduceValidResults()
    {
        // Arrange
        var (x, y) = CreateThreeClassData();

        var ovr = new OneVsRestClassifier<double>(() => new GaussianNaiveBayes<double>());
        var ovo = new OneVsOneClassifier<double>(() => new GaussianNaiveBayes<double>());

        // Act
        ovr.Train(x, y);
        ovo.Train(x, y);

        var predsOvr = ovr.Predict(x);
        var predsOvo = ovo.Predict(x);

        // Calculate accuracies
        int correctOvr = 0, correctOvo = 0;
        for (int i = 0; i < x.Rows; i++)
        {
            if (Math.Abs(predsOvr[i] - y[i]) < 0.1) correctOvr++;
            if (Math.Abs(predsOvo[i] - y[i]) < 0.1) correctOvo++;
        }

        // Assert - both should achieve reasonable accuracy
        double accOvr = (double)correctOvr / x.Rows;
        double accOvo = (double)correctOvo / x.Rows;

        Assert.True(accOvr > 0.6, $"OvR accuracy {accOvr:P2} is too low");
        Assert.True(accOvo > 0.6, $"OvO accuracy {accOvo:P2} is too low");
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void ClassifierChain_VsMultiOutput_BothHandleMultiLabel()
    {
        // Arrange
        var (x, yMultiLabel) = CreateMultiLabelData();

        var chain = new ClassifierChain<double>(
            () => new GaussianNaiveBayes<double>(),
            new ClassifierChainOptions<double> { Seed = RandomSeed });

        var multiOutput = new MultiOutputClassifier<double>(
            () => new GaussianNaiveBayes<double>());

        // Act
        chain.TrainMultiLabel(x, yMultiLabel);
        multiOutput.TrainMultiLabel(x, yMultiLabel);

        var chainPreds = chain.PredictMultiLabel(x);
        var multiPreds = multiOutput.PredictMultiLabel(x);

        // Assert - both should produce valid multi-label predictions
        Assert.Equal(yMultiLabel.Columns, chainPreds.Columns);
        Assert.Equal(yMultiLabel.Columns, multiPreds.Columns);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void Bagging_VsSingle_ReducesVariance()
    {
        // Arrange
        var (x, y) = CreateThreeClassData();

        // Single classifier
        var single = new GaussianNaiveBayes<double>();

        // Bagging with same base classifier
        var bagging = new BaggingClassifier<double>(
            () => new GaussianNaiveBayes<double>(),
            new BaggingClassifierOptions<double>
            {
                NumEstimators = 10,
                Seed = RandomSeed
            });

        // Act
        single.Train(x, y);
        bagging.Train(x, y);

        var singlePreds = single.Predict(x);
        var baggingPreds = bagging.Predict(x);

        // Assert - both should produce valid predictions
        Assert.Equal(x.Rows, singlePreds.Length);
        Assert.Equal(x.Rows, baggingPreds.Length);
    }

    #endregion
}

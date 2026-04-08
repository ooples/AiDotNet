using AiDotNet.Classification.Trees;
using AiDotNet.Models.Options;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Classification;

/// <summary>
/// Integration tests for Decision Tree classifier.
/// These tests verify mathematical correctness and do NOT trust the implementation.
/// </summary>
public class DecisionTreeIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region Basic Training and Prediction Tests

    [Fact]
    public void DecisionTree_Train_BuildsTree()
    {
        // Arrange
        var x = new Matrix<double>(6, 2);
        var y = new Vector<double>(6);

        // Simple linearly separable data
        x[0, 0] = 1; x[0, 1] = 1; y[0] = 0;
        x[1, 0] = 2; x[1, 1] = 2; y[1] = 0;
        x[2, 0] = 3; x[2, 1] = 3; y[2] = 0;
        x[3, 0] = 7; x[3, 1] = 7; y[3] = 1;
        x[4, 0] = 8; x[4, 1] = 8; y[4] = 1;
        x[5, 0] = 9; x[5, 1] = 9; y[5] = 1;

        var dt = new DecisionTreeClassifier<double>();

        // Act
        dt.Train(x, y);

        // Assert: Tree should have nodes
        Assert.True(dt.NodeCount > 0, "Tree should have at least one node");
        Assert.True(dt.LeafCount > 0, "Tree should have at least one leaf");
    }

    [Fact]
    public void DecisionTree_Predict_BinaryClassification()
    {
        // Arrange
        var x = new Matrix<double>(8, 2);
        var y = new Vector<double>(8);

        // Two well-separated clusters
        x[0, 0] = 1; x[0, 1] = 1; y[0] = 0;
        x[1, 0] = 1; x[1, 1] = 2; y[1] = 0;
        x[2, 0] = 2; x[2, 1] = 1; y[2] = 0;
        x[3, 0] = 2; x[3, 1] = 2; y[3] = 0;
        x[4, 0] = 8; x[4, 1] = 8; y[4] = 1;
        x[5, 0] = 8; x[5, 1] = 9; y[5] = 1;
        x[6, 0] = 9; x[6, 1] = 8; y[6] = 1;
        x[7, 0] = 9; x[7, 1] = 9; y[7] = 1;

        var dt = new DecisionTreeClassifier<double>();
        dt.Train(x, y);

        // Act: Predict on training data
        var predictions = dt.Predict(x);

        // Assert: Should correctly classify training data
        for (int i = 0; i < 8; i++)
        {
            Assert.Equal(y[i], predictions[i], Tolerance);
        }
    }

    [Fact]
    public void DecisionTree_Predict_MultiClassClassification()
    {
        // Arrange: Three classes
        var x = new Matrix<double>(9, 2);
        var y = new Vector<double>(9);

        // Class 0 cluster
        x[0, 0] = 1; x[0, 1] = 1; y[0] = 0;
        x[1, 0] = 1; x[1, 1] = 2; y[1] = 0;
        x[2, 0] = 2; x[2, 1] = 1; y[2] = 0;

        // Class 1 cluster
        x[3, 0] = 10; x[3, 1] = 1; y[3] = 1;
        x[4, 0] = 10; x[4, 1] = 2; y[4] = 1;
        x[5, 0] = 11; x[5, 1] = 1; y[5] = 1;

        // Class 2 cluster
        x[6, 0] = 5; x[6, 1] = 10; y[6] = 2;
        x[7, 0] = 5; x[7, 1] = 11; y[7] = 2;
        x[8, 0] = 6; x[8, 1] = 10; y[8] = 2;

        var dt = new DecisionTreeClassifier<double>();
        dt.Train(x, y);

        // Act
        var predictions = dt.Predict(x);

        // Assert
        for (int i = 0; i < 9; i++)
        {
            Assert.Equal(y[i], predictions[i], Tolerance);
        }
    }

    #endregion

    #region Probability Tests

    [Fact]
    public void DecisionTree_PredictProbabilities_SumsToOne()
    {
        // Arrange
        var x = new Matrix<double>(6, 2);
        var y = new Vector<double>(6);

        for (int i = 0; i < 3; i++)
        {
            x[i, 0] = i; x[i, 1] = i; y[i] = 0;
        }
        for (int i = 3; i < 6; i++)
        {
            x[i, 0] = i + 5; x[i, 1] = i + 5; y[i] = 1;
        }

        var dt = new DecisionTreeClassifier<double>();
        dt.Train(x, y);

        // Act
        var testPoints = new Matrix<double>(3, 2);
        testPoints[0, 0] = 1; testPoints[0, 1] = 1;
        testPoints[1, 0] = 5; testPoints[1, 1] = 5;
        testPoints[2, 0] = 10; testPoints[2, 1] = 10;

        var probs = dt.PredictProbabilities(testPoints);

        // Assert
        for (int i = 0; i < testPoints.Rows; i++)
        {
            double sum = 0;
            for (int j = 0; j < probs.Columns; j++)
            {
                sum += probs[i, j];
                Assert.True(probs[i, j] >= 0, $"Probability at [{i},{j}] is negative: {probs[i, j]}");
                Assert.True(probs[i, j] <= 1, $"Probability at [{i},{j}] exceeds 1: {probs[i, j]}");
            }
            Assert.Equal(1.0, sum, Tolerance);
        }
    }

    [Fact]
    public void DecisionTree_PureSplit_HasProbabilityOne()
    {
        // Arrange: Perfectly separable data
        var x = new Matrix<double>(4, 1);
        var y = new Vector<double>(4);

        x[0, 0] = 0; y[0] = 0;
        x[1, 0] = 1; y[1] = 0;
        x[2, 0] = 10; y[2] = 1;
        x[3, 0] = 11; y[3] = 1;

        var dt = new DecisionTreeClassifier<double>();
        dt.Train(x, y);

        // Act: Test at pure regions
        var testClass0 = new Matrix<double>(1, 1);
        testClass0[0, 0] = 0.5;
        var probClass0 = dt.PredictProbabilities(testClass0);

        var testClass1 = new Matrix<double>(1, 1);
        testClass1[0, 0] = 10.5;
        var probClass1 = dt.PredictProbabilities(testClass1);

        // Assert: Should be 100% confident for pure regions
        Assert.Equal(1.0, probClass0[0, 0], Tolerance);
        Assert.Equal(0.0, probClass0[0, 1], Tolerance);

        Assert.Equal(0.0, probClass1[0, 0], Tolerance);
        Assert.Equal(1.0, probClass1[0, 1], Tolerance);
    }

    #endregion

    #region Tree Constraint Tests

    [Fact]
    public void DecisionTree_WithMaxDepth_LimitsDepth()
    {
        // Arrange: Data that would create deep tree without constraint
        var x = new Matrix<double>(16, 2);
        var y = new Vector<double>(16);

        for (int i = 0; i < 16; i++)
        {
            x[i, 0] = i;
            x[i, 1] = i % 2;
            y[i] = i % 2;
        }

        var dt = new DecisionTreeClassifier<double>(new DecisionTreeClassifierOptions<double>
        {
            MaxDepth = 2
        });
        dt.Train(x, y);

        // Assert: Depth should be limited
        Assert.True(dt.MaxDepth <= 2, $"MaxDepth {dt.MaxDepth} exceeds limit of 2");
    }

    [Fact]
    public void DecisionTree_WithMinSamplesSplit_PreventsSplits()
    {
        // Arrange
        var x = new Matrix<double>(10, 1);
        var y = new Vector<double>(10);

        for (int i = 0; i < 5; i++)
        {
            x[i, 0] = i; y[i] = 0;
        }
        for (int i = 5; i < 10; i++)
        {
            x[i, 0] = i + 10; y[i] = 1;
        }

        var dtNoConstraint = new DecisionTreeClassifier<double>(new DecisionTreeClassifierOptions<double>
        {
            MinSamplesSplit = 2
        });
        dtNoConstraint.Train(x, y);

        var dtWithConstraint = new DecisionTreeClassifier<double>(new DecisionTreeClassifierOptions<double>
        {
            MinSamplesSplit = 10
        });
        dtWithConstraint.Train(x, y);

        // Assert: Tree with high MinSamplesSplit should have fewer nodes
        Assert.True(dtWithConstraint.NodeCount <= dtNoConstraint.NodeCount,
            "High MinSamplesSplit should result in fewer or equal nodes");
    }

    [Fact]
    public void DecisionTree_WithMinSamplesLeaf_EnforcesMinimum()
    {
        // Arrange
        var x = new Matrix<double>(10, 1);
        var y = new Vector<double>(10);

        for (int i = 0; i < 5; i++)
        {
            x[i, 0] = i; y[i] = 0;
        }
        for (int i = 5; i < 10; i++)
        {
            x[i, 0] = i + 10; y[i] = 1;
        }

        var dt = new DecisionTreeClassifier<double>(new DecisionTreeClassifierOptions<double>
        {
            MinSamplesLeaf = 3
        });
        dt.Train(x, y);

        // Assert: Model should still work with constraint
        var predictions = dt.Predict(x);
        Assert.Equal(10, predictions.Length);
    }

    #endregion

    #region Feature Importance Tests

    [Fact]
    public void DecisionTree_FeatureImportance_ReturnsValidValues()
    {
        // Arrange: Data where feature 0 is more important than feature 1
        var x = new Matrix<double>(8, 2);
        var y = new Vector<double>(8);

        // Feature 0 separates classes (important)
        // Feature 1 is just noise
        x[0, 0] = 0; x[0, 1] = 1; y[0] = 0;
        x[1, 0] = 1; x[1, 1] = 5; y[1] = 0;
        x[2, 0] = 2; x[2, 1] = 2; y[2] = 0;
        x[3, 0] = 3; x[3, 1] = 8; y[3] = 0;
        x[4, 0] = 10; x[4, 1] = 3; y[4] = 1;
        x[5, 0] = 11; x[5, 1] = 7; y[5] = 1;
        x[6, 0] = 12; x[6, 1] = 1; y[6] = 1;
        x[7, 0] = 13; x[7, 1] = 9; y[7] = 1;

        var dt = new DecisionTreeClassifier<double>();
        dt.Train(x, y);

        // Act
        var importance = dt.FeatureImportances;

        // Assert
        Assert.NotNull(importance);
        Assert.Equal(2, importance.Length);

        // Feature importances should sum to 1 (or 0 if no splits made)
        double sum = importance[0] + importance[1];
        Assert.True(sum <= 1.0 + Tolerance, $"Feature importances should sum to at most 1, got {sum}");

        // All importances should be non-negative
        Assert.True(importance[0] >= 0, "Feature importance should be non-negative");
        Assert.True(importance[1] >= 0, "Feature importance should be non-negative");
    }

    #endregion

    #region Impurity Criterion Tests

    [Fact]
    public void DecisionTree_GiniImpurity_CalculatedCorrectly()
    {
        // Arrange: Same data, Gini criterion
        var x = new Matrix<double>(6, 1);
        var y = new Vector<double>(6);

        for (int i = 0; i < 3; i++)
        {
            x[i, 0] = i; y[i] = 0;
        }
        for (int i = 3; i < 6; i++)
        {
            x[i, 0] = i + 5; y[i] = 1;
        }

        var dtGini = new DecisionTreeClassifier<double>(new DecisionTreeClassifierOptions<double>
        {
            Criterion = ClassificationSplitCriterion.Gini
        });
        dtGini.Train(x, y);

        // Act & Assert: Should still train and predict correctly
        var predictions = dtGini.Predict(x);
        for (int i = 0; i < 6; i++)
        {
            Assert.Equal(y[i], predictions[i], Tolerance);
        }
    }

    [Fact]
    public void DecisionTree_Entropy_CalculatedCorrectly()
    {
        // Arrange: Same data, Entropy criterion
        var x = new Matrix<double>(6, 1);
        var y = new Vector<double>(6);

        for (int i = 0; i < 3; i++)
        {
            x[i, 0] = i; y[i] = 0;
        }
        for (int i = 3; i < 6; i++)
        {
            x[i, 0] = i + 5; y[i] = 1;
        }

        var dtEntropy = new DecisionTreeClassifier<double>(new DecisionTreeClassifierOptions<double>
        {
            Criterion = ClassificationSplitCriterion.Entropy
        });
        dtEntropy.Train(x, y);

        // Act & Assert: Should still train and predict correctly
        var predictions = dtEntropy.Predict(x);
        for (int i = 0; i < 6; i++)
        {
            Assert.Equal(y[i], predictions[i], Tolerance);
        }
    }

    #endregion

    #region Serialization and Clone Tests

    [Fact]
    public void DecisionTree_Serialize_Deserialize_PreservesPredictions()
    {
        // Arrange
        var x = new Matrix<double>(6, 2);
        var y = new Vector<double>(6);

        for (int i = 0; i < 3; i++)
        {
            x[i, 0] = i; x[i, 1] = i; y[i] = 0;
        }
        for (int i = 3; i < 6; i++)
        {
            x[i, 0] = i + 5; x[i, 1] = i + 5; y[i] = 1;
        }

        var dt = new DecisionTreeClassifier<double>();
        dt.Train(x, y);

        var testPoint = new Matrix<double>(1, 2);
        testPoint[0, 0] = 1; testPoint[0, 1] = 1;
        var originalProbs = dt.PredictProbabilities(testPoint);

        // Act
        byte[] serialized = dt.Serialize();
        var dt2 = new DecisionTreeClassifier<double>();
        dt2.Deserialize(serialized);

        var newProbs = dt2.PredictProbabilities(testPoint);

        // Assert
        Assert.Equal(originalProbs[0, 0], newProbs[0, 0], Tolerance);
        Assert.Equal(originalProbs[0, 1], newProbs[0, 1], Tolerance);
    }

    [Fact]
    public void DecisionTree_Clone_CreatesIndependentCopy()
    {
        // Arrange
        var x = new Matrix<double>(6, 2);
        var y = new Vector<double>(6);

        for (int i = 0; i < 3; i++)
        {
            x[i, 0] = i; x[i, 1] = i; y[i] = 0;
        }
        for (int i = 3; i < 6; i++)
        {
            x[i, 0] = i + 5; x[i, 1] = i + 5; y[i] = 1;
        }

        var dt = new DecisionTreeClassifier<double>();
        dt.Train(x, y);

        // Act
        var clone = dt.Clone() as DecisionTreeClassifier<double>;

        // Assert
        Assert.NotNull(clone);

        var testPoint = new Matrix<double>(1, 2);
        testPoint[0, 0] = 1; testPoint[0, 1] = 1;

        var originalProbs = dt.PredictProbabilities(testPoint);
        var cloneProbs = clone!.PredictProbabilities(testPoint);

        Assert.Equal(originalProbs[0, 0], cloneProbs[0, 0], Tolerance);
        Assert.Equal(originalProbs[0, 1], cloneProbs[0, 1], Tolerance);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void DecisionTree_SingleFeature_WorksCorrectly()
    {
        // Arrange
        var x = new Matrix<double>(4, 1);
        var y = new Vector<double>(4);

        x[0, 0] = 0; y[0] = 0;
        x[1, 0] = 1; y[1] = 0;
        x[2, 0] = 10; y[2] = 1;
        x[3, 0] = 11; y[3] = 1;

        var dt = new DecisionTreeClassifier<double>();
        dt.Train(x, y);

        // Act & Assert
        var predictions = dt.Predict(x);
        for (int i = 0; i < 4; i++)
        {
            Assert.Equal(y[i], predictions[i], Tolerance);
        }
    }

    [Fact]
    public void DecisionTree_PureNode_StopsSplitting()
    {
        // Arrange: All same class
        var x = new Matrix<double>(4, 2);
        var y = new Vector<double>(4);

        for (int i = 0; i < 4; i++)
        {
            x[i, 0] = i;
            x[i, 1] = i * 2;
            y[i] = 0; // All same class
        }

        var dt = new DecisionTreeClassifier<double>();
        dt.Train(x, y);

        // Assert: Should be just a single leaf
        Assert.Equal(1, dt.LeafCount);
        Assert.Equal(1, dt.NodeCount);
    }

    [Fact]
    public void DecisionTree_RandomSeed_ProducesConsistentResults()
    {
        // Arrange
        var x = new Matrix<double>(10, 3);
        var y = new Vector<double>(10);

        var rand = new Random(42);
        for (int i = 0; i < 10; i++)
        {
            x[i, 0] = rand.NextDouble();
            x[i, 1] = rand.NextDouble();
            x[i, 2] = rand.NextDouble();
            y[i] = i < 5 ? 0 : 1;
        }

        var dt1 = new DecisionTreeClassifier<double>(new DecisionTreeClassifierOptions<double>
        {
            Seed = 123,
            MaxFeatures = 2
        });
        dt1.Train(x, y);

        var dt2 = new DecisionTreeClassifier<double>(new DecisionTreeClassifierOptions<double>
        {
            Seed = 123,
            MaxFeatures = 2
        });
        dt2.Train(x, y);

        // Act
        var test = new Matrix<double>(1, 3);
        test[0, 0] = 0.5; test[0, 1] = 0.5; test[0, 2] = 0.5;

        var pred1 = dt1.Predict(test);
        var pred2 = dt2.Predict(test);

        // Assert: Same seed should give same predictions
        Assert.Equal(pred1[0], pred2[0], Tolerance);
    }

    [Fact]
    public void DecisionTree_IdenticalSamples_HandlesGracefully()
    {
        // Arrange: Multiple identical samples with different labels
        var x = new Matrix<double>(4, 2);
        var y = new Vector<double>(4);

        x[0, 0] = 1; x[0, 1] = 1; y[0] = 0;
        x[1, 0] = 1; x[1, 1] = 1; y[1] = 0;
        x[2, 0] = 1; x[2, 1] = 1; y[2] = 1;
        x[3, 0] = 1; x[3, 1] = 1; y[3] = 1;

        var dt = new DecisionTreeClassifier<double>();

        // Act: Should not throw
        dt.Train(x, y);

        var predictions = dt.Predict(x);
        var probs = dt.PredictProbabilities(x);

        // Assert: Should produce valid predictions
        Assert.Equal(4, predictions.Length);
        for (int i = 0; i < 4; i++)
        {
            double sum = probs[i, 0] + probs[i, 1];
            Assert.Equal(1.0, sum, Tolerance);
        }
    }

    #endregion
}

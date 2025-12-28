using AiDotNet.Classification.NaiveBayes;
using AiDotNet.Models.Options;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Classification;

/// <summary>
/// Integration tests for Naive Bayes classifiers.
/// These tests verify mathematical correctness and do NOT trust the implementation.
/// </summary>
public class NaiveBayesIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region GaussianNaiveBayes Tests

    [Fact]
    public void GaussianNB_Train_ComputesCorrectMeans()
    {
        // Arrange: Create data where class 0 has mean [1, 2] and class 1 has mean [3, 4]
        var x = new Matrix<double>(6, 2);
        var y = new Vector<double>(6);

        // Class 0 samples (mean should be [1, 2])
        x[0, 0] = 0.5; x[0, 1] = 1.5; y[0] = 0;
        x[1, 0] = 1.0; x[1, 1] = 2.0; y[1] = 0;
        x[2, 0] = 1.5; x[2, 1] = 2.5; y[2] = 0;

        // Class 1 samples (mean should be [3, 4])
        x[3, 0] = 2.5; x[3, 1] = 3.5; y[3] = 1;
        x[4, 0] = 3.0; x[4, 1] = 4.0; y[4] = 1;
        x[5, 0] = 3.5; x[5, 1] = 4.5; y[5] = 1;

        var gnb = new GaussianNaiveBayes<double>();

        // Act
        gnb.Train(x, y);

        // Assert: Test predictions on class centroids
        // A sample at the exact mean of class 0 should be classified as class 0
        var testClass0 = new Matrix<double>(1, 2);
        testClass0[0, 0] = 1.0; testClass0[0, 1] = 2.0;
        var pred0 = gnb.Predict(testClass0);
        Assert.Equal(0.0, pred0[0], Tolerance);

        // A sample at the exact mean of class 1 should be classified as class 1
        var testClass1 = new Matrix<double>(1, 2);
        testClass1[0, 0] = 3.0; testClass1[0, 1] = 4.0;
        var pred1 = gnb.Predict(testClass1);
        Assert.Equal(1.0, pred1[0], Tolerance);
    }

    [Fact]
    public void GaussianNB_Train_ComputesCorrectVariances()
    {
        // Arrange: Create data with known variance
        // Class 0: all samples at (1, 1) - zero variance (will be clamped to min)
        // Class 1: samples with known spread
        var x = new Matrix<double>(6, 2);
        var y = new Vector<double>(6);

        // Class 0: constant values
        x[0, 0] = 1.0; x[0, 1] = 1.0; y[0] = 0;
        x[1, 0] = 1.0; x[1, 1] = 1.0; y[1] = 0;
        x[2, 0] = 1.0; x[2, 1] = 1.0; y[2] = 0;

        // Class 1: spread values
        x[3, 0] = 4.0; x[3, 1] = 4.0; y[3] = 1;
        x[4, 0] = 5.0; x[4, 1] = 5.0; y[4] = 1;
        x[5, 0] = 6.0; x[5, 1] = 6.0; y[5] = 1;

        var gnb = new GaussianNaiveBayes<double>();

        // Act
        gnb.Train(x, y);

        // Assert: Points at class means should be classified correctly
        var testClass0 = new Matrix<double>(1, 2);
        testClass0[0, 0] = 1.0; testClass0[0, 1] = 1.0;
        var pred0 = gnb.Predict(testClass0);
        Assert.Equal(0.0, pred0[0], Tolerance);

        var testClass1 = new Matrix<double>(1, 2);
        testClass1[0, 0] = 5.0; testClass1[0, 1] = 5.0;
        var pred1 = gnb.Predict(testClass1);
        Assert.Equal(1.0, pred1[0], Tolerance);
    }

    [Fact]
    public void GaussianNB_Predict_BinaryClassification()
    {
        // Arrange: Simple linearly separable data
        var x = new Matrix<double>(10, 2);
        var y = new Vector<double>(10);

        // Class 0: centered around (0, 0)
        for (int i = 0; i < 5; i++)
        {
            x[i, 0] = -1.0 + i * 0.1;
            x[i, 1] = -1.0 + i * 0.1;
            y[i] = 0;
        }

        // Class 1: centered around (5, 5)
        for (int i = 5; i < 10; i++)
        {
            x[i, 0] = 4.0 + (i - 5) * 0.1;
            x[i, 1] = 4.0 + (i - 5) * 0.1;
            y[i] = 1;
        }

        var gnb = new GaussianNaiveBayes<double>();
        gnb.Train(x, y);

        // Act & Assert: Test on training data (should classify correctly)
        var predictions = gnb.Predict(x);
        for (int i = 0; i < 10; i++)
        {
            Assert.Equal(y[i], predictions[i], Tolerance);
        }
    }

    [Fact]
    public void GaussianNB_Predict_MultiClassClassification()
    {
        // Arrange: Three well-separated classes
        var x = new Matrix<double>(9, 2);
        var y = new Vector<double>(9);

        // Class 0: centered around (0, 0)
        x[0, 0] = -0.5; x[0, 1] = -0.5; y[0] = 0;
        x[1, 0] = 0.0; x[1, 1] = 0.0; y[1] = 0;
        x[2, 0] = 0.5; x[2, 1] = 0.5; y[2] = 0;

        // Class 1: centered around (5, 0)
        x[3, 0] = 4.5; x[3, 1] = -0.5; y[3] = 1;
        x[4, 0] = 5.0; x[4, 1] = 0.0; y[4] = 1;
        x[5, 0] = 5.5; x[5, 1] = 0.5; y[5] = 1;

        // Class 2: centered around (2.5, 5)
        x[6, 0] = 2.0; x[6, 1] = 4.5; y[6] = 2;
        x[7, 0] = 2.5; x[7, 1] = 5.0; y[7] = 2;
        x[8, 0] = 3.0; x[8, 1] = 5.5; y[8] = 2;

        var gnb = new GaussianNaiveBayes<double>();
        gnb.Train(x, y);

        // Act & Assert: Test on training data
        var predictions = gnb.Predict(x);
        for (int i = 0; i < 9; i++)
        {
            Assert.Equal(y[i], predictions[i], Tolerance);
        }
    }

    [Fact]
    public void GaussianNB_PredictProbabilities_SumsToOne()
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

        var gnb = new GaussianNaiveBayes<double>();
        gnb.Train(x, y);

        // Act: Predict probabilities on various test points
        var testPoints = new Matrix<double>(5, 2);
        testPoints[0, 0] = 0; testPoints[0, 1] = 0;
        testPoints[1, 0] = 5; testPoints[1, 1] = 5;
        testPoints[2, 0] = 10; testPoints[2, 1] = 10;
        testPoints[3, 0] = -5; testPoints[3, 1] = -5;
        testPoints[4, 0] = 2.5; testPoints[4, 1] = 2.5;

        var probs = gnb.PredictProbabilities(testPoints);

        // Assert: Each row should sum to 1.0
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
    public void GaussianNB_PredictLogProbabilities_NumericallyStable()
    {
        // Arrange: Use extreme values that could cause numerical issues
        var x = new Matrix<double>(6, 2);
        var y = new Vector<double>(6);

        for (int i = 0; i < 3; i++)
        {
            x[i, 0] = i * 1000; x[i, 1] = i * 1000; y[i] = 0;
        }
        for (int i = 3; i < 6; i++)
        {
            x[i, 0] = (i - 3) * 1000 + 100000; x[i, 1] = (i - 3) * 1000 + 100000; y[i] = 1;
        }

        var gnb = new GaussianNaiveBayes<double>();
        gnb.Train(x, y);

        // Act
        var testPoint = new Matrix<double>(1, 2);
        testPoint[0, 0] = 1000; testPoint[0, 1] = 1000;
        var logProbs = gnb.PredictLogProbabilities(testPoint);

        // Assert: Log probabilities should be finite (not NaN or Infinity)
        for (int j = 0; j < logProbs.Columns; j++)
        {
            Assert.True(!double.IsNaN(logProbs[0, j]), $"Log probability at column {j} is NaN");
            Assert.True(!double.IsPositiveInfinity(logProbs[0, j]), $"Log probability at column {j} is +Infinity");
            Assert.True(logProbs[0, j] <= 0, $"Log probability at column {j} is positive: {logProbs[0, j]}");
        }
    }

    [Fact]
    public void GaussianNB_WithZeroVariance_HandlesGracefully()
    {
        // Arrange: All samples have the same feature value (zero variance)
        var x = new Matrix<double>(4, 2);
        var y = new Vector<double>(4);

        // Both classes have constant feature values
        x[0, 0] = 1.0; x[0, 1] = 5.0; y[0] = 0;
        x[1, 0] = 1.0; x[1, 1] = 5.0; y[1] = 0;
        x[2, 0] = 1.0; x[2, 1] = 10.0; y[2] = 1;
        x[3, 0] = 1.0; x[3, 1] = 10.0; y[3] = 1;

        var gnb = new GaussianNaiveBayes<double>(new NaiveBayesOptions<double> { MinVariance = 1e-9 });

        // Act: Should not throw
        gnb.Train(x, y);

        var testPoint = new Matrix<double>(1, 2);
        testPoint[0, 0] = 1.0; testPoint[0, 1] = 7.0;
        var probs = gnb.PredictProbabilities(testPoint);

        // Assert: Should produce valid probabilities
        double sum = probs[0, 0] + probs[0, 1];
        Assert.Equal(1.0, sum, Tolerance);
    }

    [Fact]
    public void GaussianNB_VarSmoothing_PreventsZeroVariance()
    {
        // Arrange
        var x = new Matrix<double>(4, 1);
        var y = new Vector<double>(4);

        // Class 0: constant value
        x[0, 0] = 5.0; y[0] = 0;
        x[1, 0] = 5.0; y[1] = 0;

        // Class 1: constant value (different)
        x[2, 0] = 10.0; y[2] = 1;
        x[3, 0] = 10.0; y[3] = 1;

        var gnb = new GaussianNaiveBayes<double>(new NaiveBayesOptions<double> { MinVariance = 0.1 });
        gnb.Train(x, y);

        // Act
        var testPoint = new Matrix<double>(1, 1);
        testPoint[0, 0] = 7.5; // Midpoint
        var probs = gnb.PredictProbabilities(testPoint);

        // Assert: Both classes should have positive probability
        Assert.True(probs[0, 0] > 0, "Class 0 probability should be positive");
        Assert.True(probs[0, 1] > 0, "Class 1 probability should be positive");
    }

    [Fact]
    public void GaussianNB_Serialize_Deserialize_PreservesPredictions()
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

        var gnb = new GaussianNaiveBayes<double>();
        gnb.Train(x, y);

        // Get original predictions
        var testPoint = new Matrix<double>(1, 2);
        testPoint[0, 0] = 1; testPoint[0, 1] = 1;
        var originalProbs = gnb.PredictProbabilities(testPoint);

        // Act: Serialize and deserialize
        byte[] serialized = gnb.Serialize();
        var gnb2 = new GaussianNaiveBayes<double>();
        gnb2.Deserialize(serialized);

        var newProbs = gnb2.PredictProbabilities(testPoint);

        // Assert: Predictions should match
        Assert.Equal(originalProbs[0, 0], newProbs[0, 0], Tolerance);
        Assert.Equal(originalProbs[0, 1], newProbs[0, 1], Tolerance);
    }

    [Fact]
    public void GaussianNB_Clone_CreatesIndependentCopy()
    {
        // Arrange
        var x = new Matrix<double>(4, 2);
        var y = new Vector<double>(4);

        x[0, 0] = 0; x[0, 1] = 0; y[0] = 0;
        x[1, 0] = 1; x[1, 1] = 1; y[1] = 0;
        x[2, 0] = 5; x[2, 1] = 5; y[2] = 1;
        x[3, 0] = 6; x[3, 1] = 6; y[3] = 1;

        var gnb = new GaussianNaiveBayes<double>();
        gnb.Train(x, y);

        // Act
        var clone = gnb.Clone() as GaussianNaiveBayes<double>;

        // Assert
        Assert.NotNull(clone);

        var testPoint = new Matrix<double>(1, 2);
        testPoint[0, 0] = 0.5; testPoint[0, 1] = 0.5;

        var originalProbs = gnb.PredictProbabilities(testPoint);
        var cloneProbs = clone!.PredictProbabilities(testPoint);

        Assert.Equal(originalProbs[0, 0], cloneProbs[0, 0], Tolerance);
        Assert.Equal(originalProbs[0, 1], cloneProbs[0, 1], Tolerance);
    }

    #endregion

    #region MultinomialNaiveBayes Tests

    [Fact]
    public void MultinomialNB_Predict_BinaryClassification()
    {
        // Arrange: Document classification scenario with word counts
        var x = new Matrix<double>(6, 4);
        var y = new Vector<double>(6);

        // Class 0: "spam" documents - high counts for features 0, 1
        x[0, 0] = 5; x[0, 1] = 4; x[0, 2] = 1; x[0, 3] = 0; y[0] = 0;
        x[1, 0] = 6; x[1, 1] = 5; x[1, 2] = 0; x[1, 3] = 1; y[1] = 0;
        x[2, 0] = 4; x[2, 1] = 3; x[2, 2] = 1; x[2, 3] = 0; y[2] = 0;

        // Class 1: "ham" documents - high counts for features 2, 3
        x[3, 0] = 1; x[3, 1] = 0; x[3, 2] = 5; x[3, 3] = 4; y[3] = 1;
        x[4, 0] = 0; x[4, 1] = 1; x[4, 2] = 6; x[4, 3] = 5; y[4] = 1;
        x[5, 0] = 1; x[5, 1] = 1; x[5, 2] = 4; x[5, 3] = 3; y[5] = 1;

        var mnb = new MultinomialNaiveBayes<double>();
        mnb.Train(x, y);

        // Act: Predict on test samples
        var testSpam = new Matrix<double>(1, 4);
        testSpam[0, 0] = 7; testSpam[0, 1] = 6; testSpam[0, 2] = 0; testSpam[0, 3] = 0;

        var testHam = new Matrix<double>(1, 4);
        testHam[0, 0] = 0; testHam[0, 1] = 0; testHam[0, 2] = 7; testHam[0, 3] = 6;

        var predSpam = mnb.Predict(testSpam);
        var predHam = mnb.Predict(testHam);

        // Assert
        Assert.Equal(0.0, predSpam[0], Tolerance); // Should predict spam (class 0)
        Assert.Equal(1.0, predHam[0], Tolerance);  // Should predict ham (class 1)
    }

    [Fact]
    public void MultinomialNB_PredictProbabilities_SumsToOne()
    {
        // Arrange
        var x = new Matrix<double>(6, 3);
        var y = new Vector<double>(6);

        for (int i = 0; i < 3; i++)
        {
            x[i, 0] = 5; x[i, 1] = 1; x[i, 2] = 1; y[i] = 0;
        }
        for (int i = 3; i < 6; i++)
        {
            x[i, 0] = 1; x[i, 1] = 5; x[i, 2] = 1; y[i] = 1;
        }

        var mnb = new MultinomialNaiveBayes<double>();
        mnb.Train(x, y);

        // Act
        var testPoints = new Matrix<double>(3, 3);
        testPoints[0, 0] = 3; testPoints[0, 1] = 3; testPoints[0, 2] = 1;
        testPoints[1, 0] = 10; testPoints[1, 1] = 0; testPoints[1, 2] = 0;
        testPoints[2, 0] = 0; testPoints[2, 1] = 10; testPoints[2, 2] = 0;

        var probs = mnb.PredictProbabilities(testPoints);

        // Assert
        for (int i = 0; i < testPoints.Rows; i++)
        {
            double sum = 0;
            for (int j = 0; j < probs.Columns; j++)
            {
                sum += probs[i, j];
                Assert.True(probs[i, j] >= 0, $"Probability at [{i},{j}] is negative");
                Assert.True(probs[i, j] <= 1, $"Probability at [{i},{j}] exceeds 1");
            }
            Assert.Equal(1.0, sum, Tolerance);
        }
    }

    [Fact]
    public void MultinomialNB_WithAlphaSmoothing_PreventsZeroProbabilities()
    {
        // Arrange: Feature 2 never appears in class 0
        var x = new Matrix<double>(4, 3);
        var y = new Vector<double>(4);

        x[0, 0] = 5; x[0, 1] = 3; x[0, 2] = 0; y[0] = 0;
        x[1, 0] = 4; x[1, 1] = 4; x[1, 2] = 0; y[1] = 0;
        x[2, 0] = 1; x[2, 1] = 1; x[2, 2] = 5; y[2] = 1;
        x[3, 0] = 2; x[3, 1] = 1; x[3, 2] = 6; y[3] = 1;

        var mnb = new MultinomialNaiveBayes<double>(new NaiveBayesOptions<double> { Alpha = 1.0 });
        mnb.Train(x, y);

        // Act: Test with a sample that has feature 2 (unseen in class 0 without smoothing)
        var testPoint = new Matrix<double>(1, 3);
        testPoint[0, 0] = 3; testPoint[0, 1] = 2; testPoint[0, 2] = 3;

        var probs = mnb.PredictProbabilities(testPoint);

        // Assert: Class 0 should still have non-zero probability due to smoothing
        Assert.True(probs[0, 0] > 0, "Class 0 probability should be positive with smoothing");
        Assert.True(probs[0, 1] > 0, "Class 1 probability should be positive");
    }

    #endregion

    #region BernoulliNaiveBayes Tests

    [Fact]
    public void BernoulliNB_Train_ComputesBinaryFeatureProbabilities()
    {
        // Arrange: Binary features (0 or 1)
        var x = new Matrix<double>(6, 3);
        var y = new Vector<double>(6);

        // Class 0: Features 0, 1 tend to be 1
        x[0, 0] = 1; x[0, 1] = 1; x[0, 2] = 0; y[0] = 0;
        x[1, 0] = 1; x[1, 1] = 1; x[1, 2] = 0; y[1] = 0;
        x[2, 0] = 1; x[2, 1] = 0; x[2, 2] = 1; y[2] = 0;

        // Class 1: Feature 2 tends to be 1
        x[3, 0] = 0; x[3, 1] = 0; x[3, 2] = 1; y[3] = 1;
        x[4, 0] = 0; x[4, 1] = 1; x[4, 2] = 1; y[4] = 1;
        x[5, 0] = 1; x[5, 1] = 0; x[5, 2] = 1; y[5] = 1;

        var bnb = new BernoulliNaiveBayes<double>();
        bnb.Train(x, y);

        // Act: Predict
        var testClass0 = new Matrix<double>(1, 3);
        testClass0[0, 0] = 1; testClass0[0, 1] = 1; testClass0[0, 2] = 0;

        var testClass1 = new Matrix<double>(1, 3);
        testClass1[0, 0] = 0; testClass1[0, 1] = 0; testClass1[0, 2] = 1;

        var pred0 = bnb.Predict(testClass0);
        var pred1 = bnb.Predict(testClass1);

        // Assert
        Assert.Equal(0.0, pred0[0], Tolerance);
        Assert.Equal(1.0, pred1[0], Tolerance);
    }

    [Fact]
    public void BernoulliNB_PredictProbabilities_SumsToOne()
    {
        // Arrange
        var x = new Matrix<double>(4, 3);
        var y = new Vector<double>(4);

        x[0, 0] = 1; x[0, 1] = 1; x[0, 2] = 0; y[0] = 0;
        x[1, 0] = 1; x[1, 1] = 0; x[1, 2] = 0; y[1] = 0;
        x[2, 0] = 0; x[2, 1] = 0; x[2, 2] = 1; y[2] = 1;
        x[3, 0] = 0; x[3, 1] = 1; x[3, 2] = 1; y[3] = 1;

        var bnb = new BernoulliNaiveBayes<double>();
        bnb.Train(x, y);

        // Act
        var testPoints = new Matrix<double>(4, 3);
        testPoints[0, 0] = 1; testPoints[0, 1] = 1; testPoints[0, 2] = 1;
        testPoints[1, 0] = 0; testPoints[1, 1] = 0; testPoints[1, 2] = 0;
        testPoints[2, 0] = 1; testPoints[2, 1] = 0; testPoints[2, 2] = 1;
        testPoints[3, 0] = 0; testPoints[3, 1] = 1; testPoints[3, 2] = 0;

        var probs = bnb.PredictProbabilities(testPoints);

        // Assert
        for (int i = 0; i < testPoints.Rows; i++)
        {
            double sum = 0;
            for (int j = 0; j < probs.Columns; j++)
            {
                sum += probs[i, j];
                Assert.True(probs[i, j] >= 0);
                Assert.True(probs[i, j] <= 1);
            }
            Assert.Equal(1.0, sum, Tolerance);
        }
    }

    #endregion

    #region ComplementNaiveBayes Tests

    [Fact]
    public void ComplementNB_Predict_MultiClassClassification()
    {
        // Arrange: Multi-class text classification scenario
        var x = new Matrix<double>(9, 4);
        var y = new Vector<double>(9);

        // Class 0
        x[0, 0] = 5; x[0, 1] = 1; x[0, 2] = 0; x[0, 3] = 0; y[0] = 0;
        x[1, 0] = 4; x[1, 1] = 2; x[1, 2] = 0; x[1, 3] = 1; y[1] = 0;
        x[2, 0] = 6; x[2, 1] = 0; x[2, 2] = 1; x[2, 3] = 0; y[2] = 0;

        // Class 1
        x[3, 0] = 0; x[3, 1] = 5; x[3, 2] = 1; x[3, 3] = 0; y[3] = 1;
        x[4, 0] = 1; x[4, 1] = 4; x[4, 2] = 2; x[4, 3] = 0; y[4] = 1;
        x[5, 0] = 0; x[5, 1] = 6; x[5, 2] = 0; x[5, 3] = 1; y[5] = 1;

        // Class 2
        x[6, 0] = 0; x[6, 1] = 0; x[6, 2] = 5; x[6, 3] = 1; y[6] = 2;
        x[7, 0] = 1; x[7, 1] = 0; x[7, 2] = 4; x[7, 3] = 2; y[7] = 2;
        x[8, 0] = 0; x[8, 1] = 1; x[8, 2] = 6; x[8, 3] = 0; y[8] = 2;

        var cnb = new ComplementNaiveBayes<double>();
        cnb.Train(x, y);

        // Act: Predict on class-representative samples
        var test0 = new Matrix<double>(1, 4);
        test0[0, 0] = 7; test0[0, 1] = 0; test0[0, 2] = 0; test0[0, 3] = 0;

        var test1 = new Matrix<double>(1, 4);
        test1[0, 0] = 0; test1[0, 1] = 7; test1[0, 2] = 0; test1[0, 3] = 0;

        var test2 = new Matrix<double>(1, 4);
        test2[0, 0] = 0; test2[0, 1] = 0; test2[0, 2] = 7; test2[0, 3] = 0;

        var pred0 = cnb.Predict(test0);
        var pred1 = cnb.Predict(test1);
        var pred2 = cnb.Predict(test2);

        // Assert
        Assert.Equal(0.0, pred0[0], Tolerance);
        Assert.Equal(1.0, pred1[0], Tolerance);
        Assert.Equal(2.0, pred2[0], Tolerance);
    }

    [Fact]
    public void ComplementNB_PredictProbabilities_SumsToOne()
    {
        // Arrange
        var x = new Matrix<double>(6, 3);
        var y = new Vector<double>(6);

        for (int i = 0; i < 3; i++)
        {
            x[i, 0] = 5; x[i, 1] = 1; x[i, 2] = 1; y[i] = 0;
        }
        for (int i = 3; i < 6; i++)
        {
            x[i, 0] = 1; x[i, 1] = 5; x[i, 2] = 1; y[i] = 1;
        }

        var cnb = new ComplementNaiveBayes<double>();
        cnb.Train(x, y);

        // Act
        var testPoints = new Matrix<double>(2, 3);
        testPoints[0, 0] = 3; testPoints[0, 1] = 3; testPoints[0, 2] = 1;
        testPoints[1, 0] = 10; testPoints[1, 1] = 0; testPoints[1, 2] = 0;

        var probs = cnb.PredictProbabilities(testPoints);

        // Assert
        for (int i = 0; i < testPoints.Rows; i++)
        {
            double sum = 0;
            for (int j = 0; j < probs.Columns; j++)
            {
                sum += probs[i, j];
                Assert.True(probs[i, j] >= 0);
                Assert.True(probs[i, j] <= 1);
            }
            Assert.Equal(1.0, sum, Tolerance);
        }
    }

    #endregion

    #region CategoricalNaiveBayes Tests

    [Fact]
    public void CategoricalNB_Predict_WithCategoricalFeatures()
    {
        // Arrange: Categorical features encoded as integers
        var x = new Matrix<double>(6, 2);
        var y = new Vector<double>(6);

        // Feature 0: 3 categories (0, 1, 2)
        // Feature 1: 2 categories (0, 1)
        // Class 0 tends to have low category values
        x[0, 0] = 0; x[0, 1] = 0; y[0] = 0;
        x[1, 0] = 0; x[1, 1] = 1; y[1] = 0;
        x[2, 0] = 1; x[2, 1] = 0; y[2] = 0;

        // Class 1 tends to have high category values
        x[3, 0] = 2; x[3, 1] = 1; y[3] = 1;
        x[4, 0] = 2; x[4, 1] = 0; y[4] = 1;
        x[5, 0] = 1; x[5, 1] = 1; y[5] = 1;

        var cnb = new CategoricalNaiveBayes<double>();
        cnb.Train(x, y);

        // Act
        var testClass0 = new Matrix<double>(1, 2);
        testClass0[0, 0] = 0; testClass0[0, 1] = 0;

        var testClass1 = new Matrix<double>(1, 2);
        testClass1[0, 0] = 2; testClass1[0, 1] = 1;

        var pred0 = cnb.Predict(testClass0);
        var pred1 = cnb.Predict(testClass1);

        // Assert
        Assert.Equal(0.0, pred0[0], Tolerance);
        Assert.Equal(1.0, pred1[0], Tolerance);
    }

    [Fact]
    public void CategoricalNB_PredictProbabilities_SumsToOne()
    {
        // Arrange
        var x = new Matrix<double>(4, 2);
        var y = new Vector<double>(4);

        x[0, 0] = 0; x[0, 1] = 0; y[0] = 0;
        x[1, 0] = 1; x[1, 1] = 0; y[1] = 0;
        x[2, 0] = 0; x[2, 1] = 1; y[2] = 1;
        x[3, 0] = 1; x[3, 1] = 1; y[3] = 1;

        var cnb = new CategoricalNaiveBayes<double>();
        cnb.Train(x, y);

        // Act
        var testPoints = new Matrix<double>(4, 2);
        testPoints[0, 0] = 0; testPoints[0, 1] = 0;
        testPoints[1, 0] = 1; testPoints[1, 1] = 1;
        testPoints[2, 0] = 0; testPoints[2, 1] = 1;
        testPoints[3, 0] = 1; testPoints[3, 1] = 0;

        var probs = cnb.PredictProbabilities(testPoints);

        // Assert
        for (int i = 0; i < testPoints.Rows; i++)
        {
            double sum = 0;
            for (int j = 0; j < probs.Columns; j++)
            {
                sum += probs[i, j];
                Assert.True(probs[i, j] >= 0);
                Assert.True(probs[i, j] <= 1);
            }
            Assert.Equal(1.0, sum, Tolerance);
        }
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void GaussianNB_SingleSamplePerClass_HandlesGracefully()
    {
        // Arrange: Only one sample per class
        var x = new Matrix<double>(2, 2);
        var y = new Vector<double>(2);

        x[0, 0] = 1; x[0, 1] = 1; y[0] = 0;
        x[1, 0] = 5; x[1, 1] = 5; y[1] = 1;

        var gnb = new GaussianNaiveBayes<double>(new NaiveBayesOptions<double> { MinVariance = 0.1 });

        // Act: Should not throw
        gnb.Train(x, y);

        var testPoint = new Matrix<double>(1, 2);
        testPoint[0, 0] = 3; testPoint[0, 1] = 3;
        var probs = gnb.PredictProbabilities(testPoint);

        // Assert
        double sum = probs[0, 0] + probs[0, 1];
        Assert.Equal(1.0, sum, Tolerance);
    }

    [Fact]
    public void GaussianNB_HighDimensionalData_Stable()
    {
        // Arrange: High dimensional data
        int numFeatures = 50;
        int numSamples = 20;

        var x = new Matrix<double>(numSamples, numFeatures);
        var y = new Vector<double>(numSamples);

        var rand = new Random(42);
        for (int i = 0; i < numSamples; i++)
        {
            int classLabel = i < numSamples / 2 ? 0 : 1;
            double offset = classLabel * 5.0;
            for (int j = 0; j < numFeatures; j++)
            {
                x[i, j] = offset + rand.NextDouble();
            }
            y[i] = classLabel;
        }

        var gnb = new GaussianNaiveBayes<double>();
        gnb.Train(x, y);

        // Act
        var predictions = gnb.Predict(x);
        var probs = gnb.PredictProbabilities(x);

        // Assert: Predictions should be stable and probabilities valid
        for (int i = 0; i < numSamples; i++)
        {
            Assert.True(predictions[i] == 0 || predictions[i] == 1, "Invalid prediction");

            double sum = 0;
            for (int j = 0; j < 2; j++)
            {
                Assert.True(!double.IsNaN(probs[i, j]), $"Probability at [{i},{j}] is NaN");
                Assert.True(!double.IsInfinity(probs[i, j]), $"Probability at [{i},{j}] is Infinity");
                sum += probs[i, j];
            }
            Assert.Equal(1.0, sum, Tolerance);
        }
    }

    [Fact]
    public void NaiveBayes_ImbalancedData_HandlesClassWeights()
    {
        // Arrange: Heavily imbalanced data (90% class 0, 10% class 1)
        var x = new Matrix<double>(20, 2);
        var y = new Vector<double>(20);

        // 18 samples of class 0
        for (int i = 0; i < 18; i++)
        {
            x[i, 0] = 1.0 + i * 0.1;
            x[i, 1] = 1.0 + i * 0.1;
            y[i] = 0;
        }

        // 2 samples of class 1
        x[18, 0] = 10.0; x[18, 1] = 10.0; y[18] = 1;
        x[19, 0] = 11.0; x[19, 1] = 11.0; y[19] = 1;

        var gnb = new GaussianNaiveBayes<double>();
        gnb.Train(x, y);

        // Act: Predict probabilities for a borderline sample
        var testPoint = new Matrix<double>(1, 2);
        testPoint[0, 0] = 5; testPoint[0, 1] = 5;
        var probs = gnb.PredictProbabilities(testPoint);

        // Assert: Probabilities should still sum to 1
        Assert.Equal(1.0, probs[0, 0] + probs[0, 1], Tolerance);

        // The prior should favor class 0 due to imbalance
        // (class 0 has 90% of samples)
    }

    #endregion

    #region CategoricalNaiveBayes Tests

    [Fact]
    public void CategoricalNB_BasicClassification_Works()
    {
        // Arrange: Categorical features (color=0/1/2, size=0/1/2)
        // Class 0: tends to have color=0, size=0
        // Class 1: tends to have color=2, size=2
        var x = new Matrix<double>(8, 2);
        var y = new Vector<double>(8);

        // Class 0 samples
        x[0, 0] = 0; x[0, 1] = 0; y[0] = 0;
        x[1, 0] = 0; x[1, 1] = 1; y[1] = 0;
        x[2, 0] = 1; x[2, 1] = 0; y[2] = 0;
        x[3, 0] = 0; x[3, 1] = 0; y[3] = 0;

        // Class 1 samples
        x[4, 0] = 2; x[4, 1] = 2; y[4] = 1;
        x[5, 0] = 2; x[5, 1] = 1; y[5] = 1;
        x[6, 0] = 1; x[6, 1] = 2; y[6] = 1;
        x[7, 0] = 2; x[7, 1] = 2; y[7] = 1;

        var cnb = new CategoricalNaiveBayes<double>();

        // Act
        cnb.Train(x, y);

        // Test: Sample with category values typical of class 0
        var testClass0 = new Matrix<double>(1, 2);
        testClass0[0, 0] = 0; testClass0[0, 1] = 0;
        var pred0 = cnb.Predict(testClass0);
        Assert.Equal(0.0, pred0[0], Tolerance);

        // Test: Sample with category values typical of class 1
        var testClass1 = new Matrix<double>(1, 2);
        testClass1[0, 0] = 2; testClass1[0, 1] = 2;
        var pred1 = cnb.Predict(testClass1);
        Assert.Equal(1.0, pred1[0], Tolerance);
    }

    [Fact]
    public void CategoricalNB_ProbabilitiesSumToOne()
    {
        // Arrange
        var x = new Matrix<double>(6, 3);
        var y = new Vector<double>(6);

        // 3 categories per feature (0, 1, 2)
        x[0, 0] = 0; x[0, 1] = 0; x[0, 2] = 0; y[0] = 0;
        x[1, 0] = 0; x[1, 1] = 1; x[1, 2] = 0; y[1] = 0;
        x[2, 0] = 1; x[2, 1] = 0; x[2, 2] = 1; y[2] = 0;
        x[3, 0] = 2; x[3, 1] = 2; x[3, 2] = 2; y[3] = 1;
        x[4, 0] = 2; x[4, 1] = 1; x[4, 2] = 2; y[4] = 1;
        x[5, 0] = 1; x[5, 1] = 2; x[5, 2] = 1; y[5] = 1;

        var cnb = new CategoricalNaiveBayes<double>();
        cnb.Train(x, y);

        // Act: Get probabilities for a test sample
        var testPoint = new Matrix<double>(1, 3);
        testPoint[0, 0] = 1; testPoint[0, 1] = 1; testPoint[0, 2] = 1;
        var probs = cnb.PredictProbabilities(testPoint);

        // Assert: Probabilities must sum to 1
        double sum = probs[0, 0] + probs[0, 1];
        Assert.Equal(1.0, sum, Tolerance);
        Assert.True(probs[0, 0] >= 0 && probs[0, 0] <= 1);
        Assert.True(probs[0, 1] >= 0 && probs[0, 1] <= 1);
    }

    [Fact]
    public void CategoricalNB_LaplaceSmoothingWorks()
    {
        // Arrange: Data where not all categories appear in training
        var x = new Matrix<double>(4, 2);
        var y = new Vector<double>(4);

        // Only categories 0 and 1 appear, but we'll test with category 2
        x[0, 0] = 0; x[0, 1] = 0; y[0] = 0;
        x[1, 0] = 1; x[1, 1] = 1; y[1] = 0;
        x[2, 0] = 0; x[2, 1] = 1; y[2] = 1;
        x[3, 0] = 1; x[3, 1] = 0; y[3] = 1;

        var cnb = new CategoricalNaiveBayes<double>(new NaiveBayesOptions<double> { Alpha = 1.0 });
        cnb.Train(x, y);

        // Act: Test with unseen category value (2)
        var testPoint = new Matrix<double>(1, 2);
        testPoint[0, 0] = 2; testPoint[0, 1] = 2;

        // Should not throw due to Laplace smoothing
        var probs = cnb.PredictProbabilities(testPoint);

        // Assert: Probabilities should still be valid
        double sum = probs[0, 0] + probs[0, 1];
        Assert.Equal(1.0, sum, Tolerance);
    }

    [Fact]
    public void CategoricalNB_MultiClassClassification_Works()
    {
        // Arrange: 3 classes with distinct categorical patterns
        var x = new Matrix<double>(9, 2);
        var y = new Vector<double>(9);

        // Class 0: category pattern (0, 0)
        x[0, 0] = 0; x[0, 1] = 0; y[0] = 0;
        x[1, 0] = 0; x[1, 1] = 1; y[1] = 0;
        x[2, 0] = 1; x[2, 1] = 0; y[2] = 0;

        // Class 1: category pattern (1, 1)
        x[3, 0] = 1; x[3, 1] = 1; y[3] = 1;
        x[4, 0] = 1; x[4, 1] = 2; y[4] = 1;
        x[5, 0] = 2; x[5, 1] = 1; y[5] = 1;

        // Class 2: category pattern (2, 2)
        x[6, 0] = 2; x[6, 1] = 2; y[6] = 2;
        x[7, 0] = 2; x[7, 1] = 2; y[7] = 2;
        x[8, 0] = 2; x[8, 1] = 1; y[8] = 2;

        var cnb = new CategoricalNaiveBayes<double>();
        cnb.Train(x, y);

        // Act & Assert: Test representative samples for each class
        var testClass0 = new Matrix<double>(1, 2);
        testClass0[0, 0] = 0; testClass0[0, 1] = 0;
        var pred0 = cnb.Predict(testClass0);
        Assert.Equal(0.0, pred0[0], Tolerance);

        var testClass2 = new Matrix<double>(1, 2);
        testClass2[0, 0] = 2; testClass2[0, 1] = 2;
        var pred2 = cnb.Predict(testClass2);
        Assert.Equal(2.0, pred2[0], Tolerance);
    }

    [Fact]
    public void CategoricalNB_Clone_PreservesState()
    {
        // Arrange
        var x = new Matrix<double>(6, 2);
        var y = new Vector<double>(6);

        x[0, 0] = 0; x[0, 1] = 0; y[0] = 0;
        x[1, 0] = 0; x[1, 1] = 1; y[1] = 0;
        x[2, 0] = 1; x[2, 1] = 0; y[2] = 0;
        x[3, 0] = 2; x[3, 1] = 2; y[3] = 1;
        x[4, 0] = 2; x[4, 1] = 1; y[4] = 1;
        x[5, 0] = 1; x[5, 1] = 2; y[5] = 1;

        var cnb = new CategoricalNaiveBayes<double>();
        cnb.Train(x, y);

        // Act
        var clone = cnb.Clone() as CategoricalNaiveBayes<double>;

        // Assert
        Assert.NotNull(clone);

        var testPoint = new Matrix<double>(1, 2);
        testPoint[0, 0] = 0; testPoint[0, 1] = 0;

        var origProbs = cnb.PredictProbabilities(testPoint);
        var cloneProbs = clone!.PredictProbabilities(testPoint);

        Assert.Equal(origProbs[0, 0], cloneProbs[0, 0], Tolerance);
        Assert.Equal(origProbs[0, 1], cloneProbs[0, 1], Tolerance);
    }

    [Fact]
    public void CategoricalNB_AlphaSmoothing_AffectsProbabilities()
    {
        // Arrange
        var x = new Matrix<double>(4, 2);
        var y = new Vector<double>(4);

        x[0, 0] = 0; x[0, 1] = 0; y[0] = 0;
        x[1, 0] = 0; x[1, 1] = 0; y[1] = 0;
        x[2, 0] = 1; x[2, 1] = 1; y[2] = 1;
        x[3, 0] = 1; x[3, 1] = 1; y[3] = 1;

        var cnbLowAlpha = new CategoricalNaiveBayes<double>(new NaiveBayesOptions<double> { Alpha = 0.01 });
        var cnbHighAlpha = new CategoricalNaiveBayes<double>(new NaiveBayesOptions<double> { Alpha = 10.0 });

        cnbLowAlpha.Train(x, y);
        cnbHighAlpha.Train(x, y);

        // Act
        var testPoint = new Matrix<double>(1, 2);
        testPoint[0, 0] = 0; testPoint[0, 1] = 0;

        var probsLow = cnbLowAlpha.PredictProbabilities(testPoint);
        var probsHigh = cnbHighAlpha.PredictProbabilities(testPoint);

        // Assert: Higher alpha should make predictions less extreme (closer to uniform)
        // Low alpha should give higher probability to class 0 for sample (0,0)
        Assert.True(probsLow[0, 0] > probsHigh[0, 0],
            "Low alpha should give more extreme (higher) probability for class 0");
    }

    #endregion
}

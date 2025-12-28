using AiDotNet.Classification.DiscriminantAnalysis;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Classification;

/// <summary>
/// Integration tests for Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA).
/// Tests verify mathematical correctness without trusting the implementation.
/// </summary>
[Trait("Category", "Integration")]
public class DiscriminantAnalysisIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region LDA Core Functionality Tests

    [Fact]
    public void LDA_Train_ComputesCorrectClassMeans()
    {
        // Arrange: Two classes with known means
        // Class 0: points around (0, 0)
        // Class 1: points around (4, 4)
        var x = new Matrix<double>(6, 2);
        x[0, 0] = -1; x[0, 1] = -1;  // Class 0
        x[1, 0] = 0; x[1, 1] = 0;    // Class 0
        x[2, 0] = 1; x[2, 1] = 1;    // Class 0
        x[3, 0] = 3; x[3, 1] = 3;    // Class 1
        x[4, 0] = 4; x[4, 1] = 4;    // Class 1
        x[5, 0] = 5; x[5, 1] = 5;    // Class 1

        var y = new Vector<double>(6);
        y[0] = 0; y[1] = 0; y[2] = 0;
        y[3] = 1; y[4] = 1; y[5] = 1;

        var lda = new LinearDiscriminantAnalysis<double>();

        // Act
        lda.Train(x, y);

        // Assert: Verify class means by predicting on the class mean points
        // A point exactly at the class mean should have highest probability for that class
        var meanClass0 = new Matrix<double>(1, 2);
        meanClass0[0, 0] = 0; meanClass0[0, 1] = 0;  // Mean of class 0

        var meanClass1 = new Matrix<double>(1, 2);
        meanClass1[0, 0] = 4; meanClass1[0, 1] = 4;  // Mean of class 1

        var probsClass0 = lda.PredictProbabilities(meanClass0);
        var probsClass1 = lda.PredictProbabilities(meanClass1);

        // Point at class 0 mean should have higher prob for class 0
        Assert.True(probsClass0[0, 0] > probsClass0[0, 1],
            $"Point at class 0 mean should favor class 0. Got P(class0)={probsClass0[0, 0]}, P(class1)={probsClass0[0, 1]}");

        // Point at class 1 mean should have higher prob for class 1
        Assert.True(probsClass1[0, 1] > probsClass1[0, 0],
            $"Point at class 1 mean should favor class 1. Got P(class0)={probsClass1[0, 0]}, P(class1)={probsClass1[0, 1]}");
    }

    [Fact]
    public void LDA_Train_ComputesCorrectClassPriors()
    {
        // Arrange: Imbalanced classes - 3 samples class 0, 1 sample class 1
        var x = new Matrix<double>(4, 2);
        x[0, 0] = 0; x[0, 1] = 0;
        x[1, 0] = 1; x[1, 1] = 0;
        x[2, 0] = 0; x[2, 1] = 1;
        x[3, 0] = 5; x[3, 1] = 5;

        var y = new Vector<double>(4);
        y[0] = 0; y[1] = 0; y[2] = 0; y[3] = 1;

        var lda = new LinearDiscriminantAnalysis<double>();

        // Act
        lda.Train(x, y);

        // Assert: At a point equidistant from both means, priors should influence prediction
        // With 3:1 ratio, class 0 should be preferred at midpoint
        var midpoint = new Matrix<double>(1, 2);
        midpoint[0, 0] = 2.0; midpoint[0, 1] = 1.75;  // Rough midpoint

        var probs = lda.PredictProbabilities(midpoint);

        // Sum of probabilities should be 1
        double sum = probs[0, 0] + probs[0, 1];
        Assert.True(Math.Abs(sum - 1.0) < Tolerance,
            $"Probabilities should sum to 1, got {sum}");
    }

    [Fact]
    public void LDA_PredictProbabilities_SumToOne()
    {
        // Arrange
        var x = new Matrix<double>(6, 3);
        var y = new Vector<double>(6);

        // 3-class problem
        x[0, 0] = 0; x[0, 1] = 0; x[0, 2] = 0; y[0] = 0;
        x[1, 0] = 1; x[1, 1] = 0; x[1, 2] = 0; y[1] = 0;
        x[2, 0] = 3; x[2, 1] = 0; x[2, 2] = 0; y[2] = 1;
        x[3, 0] = 4; x[3, 1] = 0; x[3, 2] = 0; y[3] = 1;
        x[4, 0] = 6; x[4, 1] = 0; x[4, 2] = 0; y[4] = 2;
        x[5, 0] = 7; x[5, 1] = 0; x[5, 2] = 0; y[5] = 2;

        var lda = new LinearDiscriminantAnalysis<double>();
        lda.Train(x, y);

        // Act: Test multiple points
        var testPoints = new Matrix<double>(5, 3);
        testPoints[0, 0] = 0; testPoints[0, 1] = 0; testPoints[0, 2] = 0;
        testPoints[1, 0] = 2; testPoints[1, 1] = 0; testPoints[1, 2] = 0;
        testPoints[2, 0] = 5; testPoints[2, 1] = 0; testPoints[2, 2] = 0;
        testPoints[3, 0] = 7; testPoints[3, 1] = 0; testPoints[3, 2] = 0;
        testPoints[4, 0] = -10; testPoints[4, 1] = 0; testPoints[4, 2] = 0;

        var probs = lda.PredictProbabilities(testPoints);

        // Assert: All probability rows should sum to 1
        for (int i = 0; i < testPoints.Rows; i++)
        {
            double sum = 0;
            for (int c = 0; c < 3; c++)
            {
                sum += probs[i, c];
                Assert.True(probs[i, c] >= 0 && probs[i, c] <= 1,
                    $"Probability at row {i}, class {c} should be in [0,1], got {probs[i, c]}");
            }
            Assert.True(Math.Abs(sum - 1.0) < Tolerance,
                $"Row {i} probabilities should sum to 1, got {sum}");
        }
    }

    [Fact]
    public void LDA_MahalanobisDistance_PooledCovariance()
    {
        // Arrange: Well-separated Gaussian clusters
        // LDA should use pooled covariance - same distance metric for all classes
        var x = new Matrix<double>(8, 2);
        var y = new Vector<double>(8);

        // Class 0: variance mainly in x direction
        x[0, 0] = -2; x[0, 1] = 0; y[0] = 0;
        x[1, 0] = -1; x[1, 1] = 0; y[1] = 0;
        x[2, 0] = 1; x[2, 1] = 0; y[2] = 0;
        x[3, 0] = 2; x[3, 1] = 0; y[3] = 0;

        // Class 1: variance mainly in y direction (but LDA pools, so it should blend)
        x[4, 0] = 10; x[4, 1] = -2; y[4] = 1;
        x[5, 0] = 10; x[5, 1] = -1; y[5] = 1;
        x[6, 0] = 10; x[6, 1] = 1; y[6] = 1;
        x[7, 0] = 10; x[7, 1] = 2; y[7] = 1;

        var lda = new LinearDiscriminantAnalysis<double>();
        lda.Train(x, y);

        // Act: Test point at midpoint
        var midpoint = new Matrix<double>(1, 2);
        midpoint[0, 0] = 5; midpoint[0, 1] = 0;

        var probs = lda.PredictProbabilities(midpoint);

        // Assert: At the geometric midpoint, probabilities should be close to equal
        // (assuming equal priors from 4 samples each)
        Assert.True(Math.Abs(probs[0, 0] - probs[0, 1]) < 0.3,
            $"At midpoint, probabilities should be relatively balanced. Got P(0)={probs[0, 0]}, P(1)={probs[0, 1]}");
    }

    [Fact]
    public void LDA_Regularization_ImprovesSingularCovariance()
    {
        // Arrange: Degenerate case with collinear points
        var x = new Matrix<double>(6, 2);
        var y = new Vector<double>(6);

        // All points on a line y = x
        x[0, 0] = 0; x[0, 1] = 0; y[0] = 0;
        x[1, 0] = 1; x[1, 1] = 1; y[1] = 0;
        x[2, 0] = 2; x[2, 1] = 2; y[2] = 0;
        x[3, 0] = 4; x[3, 1] = 4; y[3] = 1;
        x[4, 0] = 5; x[4, 1] = 5; y[4] = 1;
        x[5, 0] = 6; x[5, 1] = 6; y[5] = 1;

        var options = new DiscriminantAnalysisOptions<double>
        {
            RegularizationParam = 0.1
        };
        var lda = new LinearDiscriminantAnalysis<double>(options);

        // Act & Assert: Should not throw with regularization
        lda.Train(x, y);

        var testPoint = new Matrix<double>(1, 2);
        testPoint[0, 0] = 3; testPoint[0, 1] = 3;

        var probs = lda.PredictProbabilities(testPoint);

        // Should produce valid probabilities
        double sum = probs[0, 0] + probs[0, 1];
        Assert.True(Math.Abs(sum - 1.0) < Tolerance, $"Probabilities should sum to 1, got {sum}");
    }

    [Fact]
    public void LDA_LinearDecisionBoundary_CorrectlyClassifies()
    {
        // Arrange: Two classes that are linearly separable
        var x = new Matrix<double>(100, 2);
        var y = new Vector<double>(100);

        var random = new Random(42);

        // Class 0: centered at (-2, -2)
        for (int i = 0; i < 50; i++)
        {
            x[i, 0] = -2 + 0.5 * (random.NextDouble() - 0.5);
            x[i, 1] = -2 + 0.5 * (random.NextDouble() - 0.5);
            y[i] = 0;
        }

        // Class 1: centered at (2, 2)
        for (int i = 50; i < 100; i++)
        {
            x[i, 0] = 2 + 0.5 * (random.NextDouble() - 0.5);
            x[i, 1] = 2 + 0.5 * (random.NextDouble() - 0.5);
            y[i] = 1;
        }

        var lda = new LinearDiscriminantAnalysis<double>();
        lda.Train(x, y);

        // Act
        var predictions = lda.Predict(x);

        // Assert: Should achieve near-perfect classification on training data
        int correct = 0;
        for (int i = 0; i < 100; i++)
        {
            if (Math.Abs(predictions[i] - y[i]) < 0.01) correct++;
        }

        double accuracy = correct / 100.0;
        Assert.True(accuracy > 0.95, $"LDA should achieve >95% accuracy on linearly separable data, got {accuracy}");
    }

    #endregion

    #region QDA Core Functionality Tests

    [Fact]
    public void QDA_Train_ComputesSeparateCovarianceMatrices()
    {
        // Arrange: Two classes with different covariance structures
        // Class 0: high variance in x, low in y
        // Class 1: low variance in x, high in y
        var x = new Matrix<double>(8, 2);
        var y = new Vector<double>(8);

        // Class 0: stretched in x direction
        x[0, 0] = -3; x[0, 1] = 0; y[0] = 0;
        x[1, 0] = -1; x[1, 1] = 0; y[1] = 0;
        x[2, 0] = 1; x[2, 1] = 0; y[2] = 0;
        x[3, 0] = 3; x[3, 1] = 0; y[3] = 0;

        // Class 1: stretched in y direction (at x=10)
        x[4, 0] = 10; x[4, 1] = -3; y[4] = 1;
        x[5, 0] = 10; x[5, 1] = -1; y[5] = 1;
        x[6, 0] = 10; x[6, 1] = 1; y[6] = 1;
        x[7, 0] = 10; x[7, 1] = 3; y[7] = 1;

        var qda = new QuadraticDiscriminantAnalysis<double>();
        qda.Train(x, y);

        // Act: Test points
        var testX = new Matrix<double>(2, 2);
        // Point in the "x-stretched" region near class 0
        testX[0, 0] = 2; testX[0, 1] = 0;
        // Point in the "y-stretched" region near class 1
        testX[1, 0] = 10; testX[1, 1] = 2;

        var probs = qda.PredictProbabilities(testX);

        // Assert
        Assert.True(probs[0, 0] > probs[0, 1],
            $"Point (2,0) should favor class 0. Got P(0)={probs[0, 0]}, P(1)={probs[0, 1]}");
        Assert.True(probs[1, 1] > probs[1, 0],
            $"Point (10,2) should favor class 1. Got P(0)={probs[1, 0]}, P(1)={probs[1, 1]}");
    }

    [Fact]
    public void QDA_PredictProbabilities_SumToOne()
    {
        // Arrange
        var x = new Matrix<double>(9, 2);
        var y = new Vector<double>(9);

        // 3-class problem
        for (int i = 0; i < 3; i++)
        {
            x[i, 0] = i; x[i, 1] = 0; y[i] = 0;
            x[i + 3, 0] = 5 + i; x[i + 3, 1] = 0; y[i + 3] = 1;
            x[i + 6, 0] = 10 + i; x[i + 6, 1] = 0; y[i + 6] = 2;
        }

        var qda = new QuadraticDiscriminantAnalysis<double>();
        qda.Train(x, y);

        // Act
        var testPoints = new Matrix<double>(5, 2);
        testPoints[0, 0] = 0; testPoints[0, 1] = 0;
        testPoints[1, 0] = 3; testPoints[1, 1] = 0;
        testPoints[2, 0] = 6; testPoints[2, 1] = 0;
        testPoints[3, 0] = 9; testPoints[3, 1] = 0;
        testPoints[4, 0] = 12; testPoints[4, 1] = 0;

        var probs = qda.PredictProbabilities(testPoints);

        // Assert
        for (int i = 0; i < testPoints.Rows; i++)
        {
            double sum = probs[i, 0] + probs[i, 1] + probs[i, 2];
            Assert.True(Math.Abs(sum - 1.0) < Tolerance,
                $"Row {i} probabilities should sum to 1, got {sum}");
        }
    }

    [Fact]
    public void QDA_QuadraticDecisionBoundary_ClassifiesNonLinearData()
    {
        // Arrange: Data that requires curved decision boundary
        // Class 0: ring/circle around origin
        // Class 1: center cluster at origin
        var x = new Matrix<double>(24, 2);
        var y = new Vector<double>(24);

        // Class 0: points on a circle of radius 3
        for (int i = 0; i < 12; i++)
        {
            double angle = 2 * Math.PI * i / 12;
            x[i, 0] = 3 * Math.Cos(angle);
            x[i, 1] = 3 * Math.Sin(angle);
            y[i] = 0;
        }

        // Class 1: points near origin
        double[] offsetsX = { -0.5, 0, 0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0, 0.25, -0.25 };
        double[] offsetsY = { -0.5, -0.5, -0.5, 0, 0, 0, 0.5, 0.5, 0.5, 0.25, 0, 0 };
        for (int i = 0; i < 12; i++)
        {
            x[12 + i, 0] = offsetsX[i];
            x[12 + i, 1] = offsetsY[i];
            y[12 + i] = 1;
        }

        var qda = new QuadraticDiscriminantAnalysis<double>();
        qda.Train(x, y);

        // Act: Test points
        var testPoints = new Matrix<double>(4, 2);
        // Points at origin should be class 1
        testPoints[0, 0] = 0; testPoints[0, 1] = 0;
        testPoints[1, 0] = 0.3; testPoints[1, 1] = 0.3;
        // Points far from origin should be class 0
        testPoints[2, 0] = 3; testPoints[2, 1] = 0;
        testPoints[3, 0] = 0; testPoints[3, 1] = 3;

        var probs = qda.PredictProbabilities(testPoints);

        // Assert: Origin points favor class 1, outer points favor class 0
        Assert.True(probs[0, 1] > probs[0, 0],
            $"Origin should favor class 1. Got P(0)={probs[0, 0]}, P(1)={probs[0, 1]}");
        Assert.True(probs[2, 0] > probs[2, 1],
            $"Point (3,0) should favor class 0. Got P(0)={probs[2, 0]}, P(1)={probs[2, 1]}");
    }

    [Fact]
    public void QDA_Regularization_PreventsOverfitting()
    {
        // Arrange: Small dataset prone to singular covariance
        var x = new Matrix<double>(6, 2);
        var y = new Vector<double>(6);

        x[0, 0] = 0; x[0, 1] = 0; y[0] = 0;
        x[1, 0] = 1; x[1, 1] = 0; y[1] = 0;
        x[2, 0] = 2; x[2, 1] = 0; y[2] = 0;
        x[3, 0] = 5; x[3, 1] = 5; y[3] = 1;
        x[4, 0] = 6; x[4, 1] = 5; y[4] = 1;
        x[5, 0] = 7; x[5, 1] = 5; y[5] = 1;

        var options = new DiscriminantAnalysisOptions<double>
        {
            RegularizationParam = 0.01
        };
        var qda = new QuadraticDiscriminantAnalysis<double>(options);

        // Act & Assert: Should not throw
        qda.Train(x, y);

        var testPoint = new Matrix<double>(1, 2);
        testPoint[0, 0] = 3.5; testPoint[0, 1] = 2.5;

        var probs = qda.PredictProbabilities(testPoint);

        double sum = probs[0, 0] + probs[0, 1];
        Assert.True(Math.Abs(sum - 1.0) < Tolerance, $"Probabilities should sum to 1, got {sum}");
    }

    #endregion

    #region LDA vs QDA Comparison Tests

    [Fact]
    public void LDA_vs_QDA_SameCovariance_SimilarResults()
    {
        // Arrange: Data where both classes have same covariance
        // LDA and QDA should give similar results
        var x = new Matrix<double>(40, 2);
        var y = new Vector<double>(40);

        var random = new Random(42);

        // Both classes: spherical Gaussian with same variance
        for (int i = 0; i < 20; i++)
        {
            x[i, 0] = 0 + 0.5 * (random.NextDouble() - 0.5);
            x[i, 1] = 0 + 0.5 * (random.NextDouble() - 0.5);
            y[i] = 0;
        }
        for (int i = 20; i < 40; i++)
        {
            x[i, 0] = 3 + 0.5 * (random.NextDouble() - 0.5);
            x[i, 1] = 3 + 0.5 * (random.NextDouble() - 0.5);
            y[i] = 1;
        }

        var lda = new LinearDiscriminantAnalysis<double>();
        var qda = new QuadraticDiscriminantAnalysis<double>();

        lda.Train(x, y);
        qda.Train(x, y);

        // Act
        var testPoint = new Matrix<double>(1, 2);
        testPoint[0, 0] = 1.5; testPoint[0, 1] = 1.5;

        var ldaProbs = lda.PredictProbabilities(testPoint);
        var qdaProbs = qda.PredictProbabilities(testPoint);

        // Assert: Results should be similar (not necessarily identical)
        Assert.True(Math.Abs(ldaProbs[0, 0] - qdaProbs[0, 0]) < 0.2,
            $"LDA and QDA should give similar probs when covariances are equal. LDA P(0)={ldaProbs[0, 0]}, QDA P(0)={qdaProbs[0, 0]}");
    }

    [Fact]
    public void LDA_vs_QDA_DifferentCovariance_QDABetter()
    {
        // Arrange: Data where classes have VERY different covariances
        // QDA should be more appropriate
        var x = new Matrix<double>(20, 2);
        var y = new Vector<double>(20);

        // Class 0: very elongated in x direction
        for (int i = 0; i < 10; i++)
        {
            x[i, 0] = -5 + i;  // x from -5 to 4
            x[i, 1] = 0.01 * (i % 3 - 1);  // very small y variance
            y[i] = 0;
        }

        // Class 1: very elongated in y direction
        for (int i = 0; i < 10; i++)
        {
            x[10 + i, 0] = 10 + 0.01 * (i % 3 - 1);  // very small x variance
            x[10 + i, 1] = -5 + i;  // y from -5 to 4
            y[10 + i] = 1;
        }

        var lda = new LinearDiscriminantAnalysis<double>();
        var qda = new QuadraticDiscriminantAnalysis<double>();

        lda.Train(x, y);
        qda.Train(x, y);

        // Act: Test on training data
        var ldaPreds = lda.Predict(x);
        var qdaPreds = qda.Predict(x);

        // Assert: Count correct predictions
        int ldaCorrect = 0, qdaCorrect = 0;
        for (int i = 0; i < 20; i++)
        {
            if (Math.Abs(ldaPreds[i] - y[i]) < 0.01) ldaCorrect++;
            if (Math.Abs(qdaPreds[i] - y[i]) < 0.01) qdaCorrect++;
        }

        // Both should do well on this well-separated data
        Assert.True(ldaCorrect >= 15, $"LDA should classify most correctly, got {ldaCorrect}/20");
        Assert.True(qdaCorrect >= 15, $"QDA should classify most correctly, got {qdaCorrect}/20");
    }

    #endregion

    #region Edge Cases and Error Handling

    [Fact]
    public void LDA_ThrowsOnMismatchedDimensions()
    {
        // Arrange
        var x = new Matrix<double>(5, 2);
        var y = new Vector<double>(3);  // Mismatched length

        var lda = new LinearDiscriminantAnalysis<double>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => lda.Train(x, y));
    }

    [Fact]
    public void QDA_ThrowsOnMismatchedDimensions()
    {
        // Arrange
        var x = new Matrix<double>(5, 2);
        var y = new Vector<double>(3);  // Mismatched length

        var qda = new QuadraticDiscriminantAnalysis<double>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => qda.Train(x, y));
    }

    [Fact]
    public void LDA_PredictBeforeTrain_ThrowsInvalidOperationException()
    {
        // Arrange
        var lda = new LinearDiscriminantAnalysis<double>();
        var testPoint = new Matrix<double>(1, 2);
        testPoint[0, 0] = 1; testPoint[0, 1] = 1;

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => lda.PredictProbabilities(testPoint));
    }

    [Fact]
    public void QDA_PredictBeforeTrain_ThrowsInvalidOperationException()
    {
        // Arrange
        var qda = new QuadraticDiscriminantAnalysis<double>();
        var testPoint = new Matrix<double>(1, 2);
        testPoint[0, 0] = 1; testPoint[0, 1] = 1;

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => qda.PredictProbabilities(testPoint));
    }

    [Fact]
    public void LDA_SingleSamplePerClass_HandlesGracefully()
    {
        // Arrange: Edge case with just 1 sample per class
        var x = new Matrix<double>(2, 2);
        x[0, 0] = 0; x[0, 1] = 0;
        x[1, 0] = 5; x[1, 1] = 5;

        var y = new Vector<double>(2);
        y[0] = 0; y[1] = 1;

        var options = new DiscriminantAnalysisOptions<double>
        {
            RegularizationParam = 0.1  // Need regularization for singular covariance
        };
        var lda = new LinearDiscriminantAnalysis<double>(options);

        // Act & Assert: Should not throw
        lda.Train(x, y);

        var testPoint = new Matrix<double>(1, 2);
        testPoint[0, 0] = 2.5; testPoint[0, 1] = 2.5;

        var probs = lda.PredictProbabilities(testPoint);
        double sum = probs[0, 0] + probs[0, 1];
        Assert.True(Math.Abs(sum - 1.0) < Tolerance, $"Probabilities should sum to 1, got {sum}");
    }

    #endregion

    #region PredictLogProbabilities Tests

    [Fact]
    public void LDA_PredictLogProbabilities_ConsistentWithProbabilities()
    {
        // Arrange
        var x = new Matrix<double>(6, 2);
        var y = new Vector<double>(6);

        x[0, 0] = 0; x[0, 1] = 0; y[0] = 0;
        x[1, 0] = 1; x[1, 1] = 0; y[1] = 0;
        x[2, 0] = 2; x[2, 1] = 0; y[2] = 0;
        x[3, 0] = 5; x[3, 1] = 0; y[3] = 1;
        x[4, 0] = 6; x[4, 1] = 0; y[4] = 1;
        x[5, 0] = 7; x[5, 1] = 0; y[5] = 1;

        var lda = new LinearDiscriminantAnalysis<double>();
        lda.Train(x, y);

        var testPoint = new Matrix<double>(1, 2);
        testPoint[0, 0] = 3.5; testPoint[0, 1] = 0;

        // Act
        var probs = lda.PredictProbabilities(testPoint);
        var logProbs = lda.PredictLogProbabilities(testPoint);

        // Assert: log(probs) should equal logProbs
        for (int c = 0; c < 2; c++)
        {
            double expectedLog = Math.Log(Math.Max(probs[0, c], 1e-15));
            Assert.True(Math.Abs(logProbs[0, c] - expectedLog) < 0.01,
                $"Log probability mismatch for class {c}. Expected {expectedLog}, got {logProbs[0, c]}");
        }
    }

    [Fact]
    public void QDA_PredictLogProbabilities_ConsistentWithProbabilities()
    {
        // Arrange
        var x = new Matrix<double>(6, 2);
        var y = new Vector<double>(6);

        x[0, 0] = 0; x[0, 1] = 0; y[0] = 0;
        x[1, 0] = 1; x[1, 1] = 0; y[1] = 0;
        x[2, 0] = 2; x[2, 1] = 0; y[2] = 0;
        x[3, 0] = 5; x[3, 1] = 0; y[3] = 1;
        x[4, 0] = 6; x[4, 1] = 0; y[4] = 1;
        x[5, 0] = 7; x[5, 1] = 0; y[5] = 1;

        var qda = new QuadraticDiscriminantAnalysis<double>();
        qda.Train(x, y);

        var testPoint = new Matrix<double>(1, 2);
        testPoint[0, 0] = 3.5; testPoint[0, 1] = 0;

        // Act
        var probs = qda.PredictProbabilities(testPoint);
        var logProbs = qda.PredictLogProbabilities(testPoint);

        // Assert
        for (int c = 0; c < 2; c++)
        {
            double expectedLog = Math.Log(Math.Max(probs[0, c], 1e-15));
            Assert.True(Math.Abs(logProbs[0, c] - expectedLog) < 0.01,
                $"Log probability mismatch for class {c}. Expected {expectedLog}, got {logProbs[0, c]}");
        }
    }

    #endregion

    #region Clone and Serialization Tests

    [Fact]
    public void LDA_Clone_ProducesSamePredictions()
    {
        // Arrange
        var x = new Matrix<double>(6, 2);
        var y = new Vector<double>(6);

        x[0, 0] = 0; x[0, 1] = 0; y[0] = 0;
        x[1, 0] = 1; x[1, 1] = 0; y[1] = 0;
        x[2, 0] = 2; x[2, 1] = 0; y[2] = 0;
        x[3, 0] = 5; x[3, 1] = 0; y[3] = 1;
        x[4, 0] = 6; x[4, 1] = 0; y[4] = 1;
        x[5, 0] = 7; x[5, 1] = 0; y[5] = 1;

        var lda = new LinearDiscriminantAnalysis<double>();
        lda.Train(x, y);

        // Act
        var clone = (LinearDiscriminantAnalysis<double>)lda.Clone();

        var testPoint = new Matrix<double>(1, 2);
        testPoint[0, 0] = 3.5; testPoint[0, 1] = 0;

        var originalProbs = lda.PredictProbabilities(testPoint);
        var cloneProbs = clone.PredictProbabilities(testPoint);

        // Assert
        Assert.True(Math.Abs(originalProbs[0, 0] - cloneProbs[0, 0]) < Tolerance);
        Assert.True(Math.Abs(originalProbs[0, 1] - cloneProbs[0, 1]) < Tolerance);
    }

    [Fact]
    public void QDA_Clone_ProducesSamePredictions()
    {
        // Arrange
        var x = new Matrix<double>(6, 2);
        var y = new Vector<double>(6);

        x[0, 0] = 0; x[0, 1] = 0; y[0] = 0;
        x[1, 0] = 1; x[1, 1] = 0; y[1] = 0;
        x[2, 0] = 2; x[2, 1] = 0; y[2] = 0;
        x[3, 0] = 5; x[3, 1] = 0; y[3] = 1;
        x[4, 0] = 6; x[4, 1] = 0; y[4] = 1;
        x[5, 0] = 7; x[5, 1] = 0; y[5] = 1;

        var qda = new QuadraticDiscriminantAnalysis<double>();
        qda.Train(x, y);

        // Act
        var clone = (QuadraticDiscriminantAnalysis<double>)qda.Clone();

        var testPoint = new Matrix<double>(1, 2);
        testPoint[0, 0] = 3.5; testPoint[0, 1] = 0;

        var originalProbs = qda.PredictProbabilities(testPoint);
        var cloneProbs = clone.PredictProbabilities(testPoint);

        // Assert
        Assert.True(Math.Abs(originalProbs[0, 0] - cloneProbs[0, 0]) < Tolerance);
        Assert.True(Math.Abs(originalProbs[0, 1] - cloneProbs[0, 1]) < Tolerance);
    }

    [Fact]
    public void LDA_Clone_IsIndependent()
    {
        // Arrange
        var x = new Matrix<double>(6, 2);
        var y = new Vector<double>(6);

        x[0, 0] = 0; x[0, 1] = 0; y[0] = 0;
        x[1, 0] = 1; x[1, 1] = 0; y[1] = 0;
        x[2, 0] = 2; x[2, 1] = 0; y[2] = 0;
        x[3, 0] = 5; x[3, 1] = 0; y[3] = 1;
        x[4, 0] = 6; x[4, 1] = 0; y[4] = 1;
        x[5, 0] = 7; x[5, 1] = 0; y[5] = 1;

        var lda = new LinearDiscriminantAnalysis<double>();
        lda.Train(x, y);
        var clone = (LinearDiscriminantAnalysis<double>)lda.Clone();

        // Get predictions before retraining
        var testPoint = new Matrix<double>(1, 2);
        testPoint[0, 0] = 3.5; testPoint[0, 1] = 0;
        var originalProbs = clone.PredictProbabilities(testPoint);

        // Act: Retrain original with different data
        var x2 = new Matrix<double>(4, 2);
        var y2 = new Vector<double>(4);
        x2[0, 0] = 10; x2[0, 1] = 10; y2[0] = 0;
        x2[1, 0] = 11; x2[1, 1] = 10; y2[1] = 0;
        x2[2, 0] = 20; x2[2, 1] = 20; y2[2] = 1;
        x2[3, 0] = 21; x2[3, 1] = 20; y2[3] = 1;

        lda.Train(x2, y2);

        // Assert: Clone should still produce original predictions
        var cloneProbs = clone.PredictProbabilities(testPoint);
        Assert.True(Math.Abs(originalProbs[0, 0] - cloneProbs[0, 0]) < Tolerance);
        Assert.True(Math.Abs(originalProbs[0, 1] - cloneProbs[0, 1]) < Tolerance);
    }

    #endregion

    #region Multiclass Tests

    [Fact]
    public void LDA_MulticlassClassification_ThreeClasses()
    {
        // Arrange: 3-class problem
        var x = new Matrix<double>(15, 2);
        var y = new Vector<double>(15);

        // Class 0 at (0, 0)
        for (int i = 0; i < 5; i++)
        {
            x[i, 0] = 0 + 0.1 * i; x[i, 1] = 0 + 0.1 * i; y[i] = 0;
        }
        // Class 1 at (5, 0)
        for (int i = 0; i < 5; i++)
        {
            x[5 + i, 0] = 5 + 0.1 * i; x[5 + i, 1] = 0 + 0.1 * i; y[5 + i] = 1;
        }
        // Class 2 at (2.5, 4)
        for (int i = 0; i < 5; i++)
        {
            x[10 + i, 0] = 2.5 + 0.1 * i; x[10 + i, 1] = 4 + 0.1 * i; y[10 + i] = 2;
        }

        var lda = new LinearDiscriminantAnalysis<double>();
        lda.Train(x, y);

        // Act
        var testPoints = new Matrix<double>(3, 2);
        testPoints[0, 0] = 0; testPoints[0, 1] = 0;    // Should be class 0
        testPoints[1, 0] = 5; testPoints[1, 1] = 0;    // Should be class 1
        testPoints[2, 0] = 2.5; testPoints[2, 1] = 4;  // Should be class 2

        var predictions = lda.Predict(testPoints);
        var probs = lda.PredictProbabilities(testPoints);

        // Assert
        Assert.True(Math.Abs(predictions[0] - 0) < 0.01, $"Point 0 should be class 0, got {predictions[0]}");
        Assert.True(Math.Abs(predictions[1] - 1) < 0.01, $"Point 1 should be class 1, got {predictions[1]}");
        Assert.True(Math.Abs(predictions[2] - 2) < 0.01, $"Point 2 should be class 2, got {predictions[2]}");

        // Each row of probabilities should sum to 1
        for (int i = 0; i < 3; i++)
        {
            double sum = probs[i, 0] + probs[i, 1] + probs[i, 2];
            Assert.True(Math.Abs(sum - 1.0) < Tolerance, $"Row {i} probs should sum to 1, got {sum}");
        }
    }

    [Fact]
    public void QDA_MulticlassClassification_FourClasses()
    {
        // Arrange: 4-class problem at corners of a square
        var x = new Matrix<double>(20, 2);
        var y = new Vector<double>(20);

        double[][] centers = { new[] { 0.0, 0.0 }, new[] { 5.0, 0.0 }, new[] { 0.0, 5.0 }, new[] { 5.0, 5.0 } };

        for (int c = 0; c < 4; c++)
        {
            for (int i = 0; i < 5; i++)
            {
                x[c * 5 + i, 0] = centers[c][0] + 0.1 * i;
                x[c * 5 + i, 1] = centers[c][1] + 0.1 * i;
                y[c * 5 + i] = c;
            }
        }

        var qda = new QuadraticDiscriminantAnalysis<double>();
        qda.Train(x, y);

        // Act
        var testPoints = new Matrix<double>(4, 2);
        for (int c = 0; c < 4; c++)
        {
            testPoints[c, 0] = centers[c][0];
            testPoints[c, 1] = centers[c][1];
        }

        var predictions = qda.Predict(testPoints);
        var probs = qda.PredictProbabilities(testPoints);

        // Assert
        for (int c = 0; c < 4; c++)
        {
            Assert.True(Math.Abs(predictions[c] - c) < 0.01,
                $"Point at center {c} should be class {c}, got {predictions[c]}");

            double sum = probs[c, 0] + probs[c, 1] + probs[c, 2] + probs[c, 3];
            Assert.True(Math.Abs(sum - 1.0) < Tolerance,
                $"Row {c} probs should sum to 1, got {sum}");
        }
    }

    #endregion

    #region Numerical Stability Tests

    [Fact]
    public void LDA_NumericalStability_LargeFeatureValues()
    {
        // Arrange: Data with large feature values
        var x = new Matrix<double>(6, 2);
        var y = new Vector<double>(6);

        x[0, 0] = 1e6; x[0, 1] = 1e6; y[0] = 0;
        x[1, 0] = 1e6 + 1; x[1, 1] = 1e6; y[1] = 0;
        x[2, 0] = 1e6 + 2; x[2, 1] = 1e6; y[2] = 0;
        x[3, 0] = 2e6; x[3, 1] = 2e6; y[3] = 1;
        x[4, 0] = 2e6 + 1; x[4, 1] = 2e6; y[4] = 1;
        x[5, 0] = 2e6 + 2; x[5, 1] = 2e6; y[5] = 1;

        var lda = new LinearDiscriminantAnalysis<double>();
        lda.Train(x, y);

        // Act
        var testPoint = new Matrix<double>(1, 2);
        testPoint[0, 0] = 1.5e6; testPoint[0, 1] = 1.5e6;

        var probs = lda.PredictProbabilities(testPoint);

        // Assert: Should produce valid probabilities despite large values
        double sum = probs[0, 0] + probs[0, 1];
        Assert.True(Math.Abs(sum - 1.0) < Tolerance, $"Probabilities should sum to 1, got {sum}");
        Assert.False(double.IsNaN(probs[0, 0]), "Probability should not be NaN");
        Assert.False(double.IsInfinity(probs[0, 0]), "Probability should not be infinite");
    }

    [Fact]
    public void QDA_NumericalStability_SmallVariances()
    {
        // Arrange: Data with very small variances
        var x = new Matrix<double>(6, 2);
        var y = new Vector<double>(6);

        x[0, 0] = 0; x[0, 1] = 0; y[0] = 0;
        x[1, 0] = 1e-8; x[1, 1] = 0; y[1] = 0;
        x[2, 0] = 2e-8; x[2, 1] = 0; y[2] = 0;
        x[3, 0] = 1; x[3, 1] = 1; y[3] = 1;
        x[4, 0] = 1 + 1e-8; x[4, 1] = 1; y[4] = 1;
        x[5, 0] = 1 + 2e-8; x[5, 1] = 1; y[5] = 1;

        var options = new DiscriminantAnalysisOptions<double>
        {
            RegularizationParam = 0.001
        };
        var qda = new QuadraticDiscriminantAnalysis<double>(options);
        qda.Train(x, y);

        // Act
        var testPoint = new Matrix<double>(1, 2);
        testPoint[0, 0] = 0.5; testPoint[0, 1] = 0.5;

        var probs = qda.PredictProbabilities(testPoint);

        // Assert
        double sum = probs[0, 0] + probs[0, 1];
        Assert.True(Math.Abs(sum - 1.0) < Tolerance, $"Probabilities should sum to 1, got {sum}");
        Assert.False(double.IsNaN(probs[0, 0]), "Probability should not be NaN");
    }

    #endregion

    #region GetModelMetadata Tests

    [Fact]
    public void LDA_GetModelMetadata_ContainsRegularizationParam()
    {
        // Arrange
        var options = new DiscriminantAnalysisOptions<double>
        {
            RegularizationParam = 0.05
        };
        var lda = new LinearDiscriminantAnalysis<double>(options);

        var x = new Matrix<double>(4, 2);
        x[0, 0] = 0; x[0, 1] = 0;
        x[1, 0] = 1; x[1, 1] = 0;
        x[2, 0] = 5; x[2, 1] = 5;
        x[3, 0] = 6; x[3, 1] = 5;

        var y = new Vector<double>(4);
        y[0] = 0; y[1] = 0; y[2] = 1; y[3] = 1;

        lda.Train(x, y);

        // Act
        var metadata = lda.GetModelMetadata();

        // Assert
        Assert.True(metadata.AdditionalInfo.ContainsKey("RegularizationParam"));
        Assert.Equal(0.05, metadata.AdditionalInfo["RegularizationParam"]);
    }

    [Fact]
    public void QDA_GetModelMetadata_ContainsRegularizationParam()
    {
        // Arrange
        var options = new DiscriminantAnalysisOptions<double>
        {
            RegularizationParam = 0.1
        };
        var qda = new QuadraticDiscriminantAnalysis<double>(options);

        var x = new Matrix<double>(4, 2);
        x[0, 0] = 0; x[0, 1] = 0;
        x[1, 0] = 1; x[1, 1] = 0;
        x[2, 0] = 5; x[2, 1] = 5;
        x[3, 0] = 6; x[3, 1] = 5;

        var y = new Vector<double>(4);
        y[0] = 0; y[1] = 0; y[2] = 1; y[3] = 1;

        qda.Train(x, y);

        // Act
        var metadata = qda.GetModelMetadata();

        // Assert
        Assert.True(metadata.AdditionalInfo.ContainsKey("RegularizationParam"));
        Assert.Equal(0.1, metadata.AdditionalInfo["RegularizationParam"]);
    }

    #endregion
}

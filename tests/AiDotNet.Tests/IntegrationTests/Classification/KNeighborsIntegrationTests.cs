using AiDotNet.Classification.Neighbors;
using AiDotNet.Models.Options;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Classification;

/// <summary>
/// Integration tests for K-Nearest Neighbors classifier.
/// These tests verify mathematical correctness and do NOT trust the implementation.
/// </summary>
public class KNeighborsIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region Basic Prediction Tests

    [Fact]
    public void KNN_Train_StoresTrainingData()
    {
        // Arrange
        var x = new Matrix<double>(4, 2);
        var y = new Vector<double>(4);

        x[0, 0] = 1; x[0, 1] = 1; y[0] = 0;
        x[1, 0] = 2; x[1, 1] = 2; y[1] = 0;
        x[2, 0] = 8; x[2, 1] = 8; y[2] = 1;
        x[3, 0] = 9; x[3, 1] = 9; y[3] = 1;

        var knn = new KNeighborsClassifier<double>(new KNeighborsOptions<double> { NNeighbors = 3 });

        // Act
        knn.Train(x, y);

        // Assert: Model should be able to predict on training data
        var predictions = knn.Predict(x);
        Assert.Equal(4, predictions.Length);
    }

    [Fact]
    public void KNN_Predict_WithK1_ReturnsNearestNeighbor()
    {
        // Arrange: Simple 4-point dataset
        var x = new Matrix<double>(4, 2);
        var y = new Vector<double>(4);

        x[0, 0] = 0; x[0, 1] = 0; y[0] = 0;  // Point at origin, class 0
        x[1, 0] = 1; x[1, 1] = 0; y[1] = 0;  // Point at (1,0), class 0
        x[2, 0] = 10; x[2, 1] = 0; y[2] = 1; // Point at (10,0), class 1
        x[3, 0] = 11; x[3, 1] = 0; y[3] = 1; // Point at (11,0), class 1

        var knn = new KNeighborsClassifier<double>(new KNeighborsOptions<double> { NNeighbors = 1 });
        knn.Train(x, y);

        // Act: Test points very close to training points
        var test = new Matrix<double>(2, 2);
        test[0, 0] = 0.1; test[0, 1] = 0; // Closest to (0,0) - class 0
        test[1, 0] = 10.1; test[1, 1] = 0; // Closest to (10,0) - class 1

        var predictions = knn.Predict(test);

        // Assert
        Assert.Equal(0.0, predictions[0], Tolerance);
        Assert.Equal(1.0, predictions[1], Tolerance);
    }

    [Fact]
    public void KNN_Predict_WithK3_UsesMajorityVoting()
    {
        // Arrange: 5 points where k=3 voting matters
        var x = new Matrix<double>(5, 2);
        var y = new Vector<double>(5);

        // 3 class 0 points clustered together
        x[0, 0] = 0; x[0, 1] = 0; y[0] = 0;
        x[1, 0] = 0.5; x[1, 1] = 0; y[1] = 0;
        x[2, 0] = 0; x[2, 1] = 0.5; y[2] = 0;

        // 2 class 1 points further away
        x[3, 0] = 5; x[3, 1] = 5; y[3] = 1;
        x[4, 0] = 6; x[4, 1] = 6; y[4] = 1;

        var knn = new KNeighborsClassifier<double>(new KNeighborsOptions<double> { NNeighbors = 3 });
        knn.Train(x, y);

        // Act: Test point at origin - nearest 3 are all class 0
        var test = new Matrix<double>(1, 2);
        test[0, 0] = 0.25; test[0, 1] = 0.25;

        var prediction = knn.Predict(test);

        // Assert: Majority of 3 nearest neighbors should be class 0
        Assert.Equal(0.0, prediction[0], Tolerance);
    }

    [Fact]
    public void KNN_Predict_BinaryClassification()
    {
        // Arrange: Two well-separated clusters
        var x = new Matrix<double>(10, 2);
        var y = new Vector<double>(10);

        // Class 0 cluster around (0, 0)
        for (int i = 0; i < 5; i++)
        {
            x[i, 0] = i * 0.5;
            x[i, 1] = i * 0.5;
            y[i] = 0;
        }

        // Class 1 cluster around (10, 10)
        for (int i = 5; i < 10; i++)
        {
            x[i, 0] = 10 + (i - 5) * 0.5;
            x[i, 1] = 10 + (i - 5) * 0.5;
            y[i] = 1;
        }

        var knn = new KNeighborsClassifier<double>(new KNeighborsOptions<double> { NNeighbors = 3 });
        knn.Train(x, y);

        // Act: Predict on training data
        var predictions = knn.Predict(x);

        // Assert: Should correctly classify all training points
        for (int i = 0; i < 10; i++)
        {
            Assert.Equal(y[i], predictions[i], Tolerance);
        }
    }

    [Fact]
    public void KNN_Predict_MultiClassClassification()
    {
        // Arrange: Three well-separated clusters
        var x = new Matrix<double>(9, 2);
        var y = new Vector<double>(9);

        // Class 0 cluster
        x[0, 0] = 0; x[0, 1] = 0; y[0] = 0;
        x[1, 0] = 0.5; x[1, 1] = 0.5; y[1] = 0;
        x[2, 0] = 1; x[2, 1] = 0; y[2] = 0;

        // Class 1 cluster
        x[3, 0] = 10; x[3, 1] = 0; y[3] = 1;
        x[4, 0] = 10.5; x[4, 1] = 0.5; y[4] = 1;
        x[5, 0] = 11; x[5, 1] = 0; y[5] = 1;

        // Class 2 cluster
        x[6, 0] = 5; x[6, 1] = 10; y[6] = 2;
        x[7, 0] = 5.5; x[7, 1] = 10.5; y[7] = 2;
        x[8, 0] = 5; x[8, 1] = 11; y[8] = 2;

        var knn = new KNeighborsClassifier<double>(new KNeighborsOptions<double> { NNeighbors = 3 });
        knn.Train(x, y);

        // Act: Predict on cluster centers
        var test = new Matrix<double>(3, 2);
        test[0, 0] = 0.5; test[0, 1] = 0.25; // Near class 0
        test[1, 0] = 10.5; test[1, 1] = 0.25; // Near class 1
        test[2, 0] = 5.25; test[2, 1] = 10.5; // Near class 2

        var predictions = knn.Predict(test);

        // Assert
        Assert.Equal(0.0, predictions[0], Tolerance);
        Assert.Equal(1.0, predictions[1], Tolerance);
        Assert.Equal(2.0, predictions[2], Tolerance);
    }

    #endregion

    #region Probability Tests

    [Fact]
    public void KNN_PredictProbabilities_SumsToOne()
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

        var knn = new KNeighborsClassifier<double>(new KNeighborsOptions<double> { NNeighbors = 3 });
        knn.Train(x, y);

        // Act
        var testPoints = new Matrix<double>(5, 2);
        testPoints[0, 0] = 0; testPoints[0, 1] = 0;
        testPoints[1, 0] = 5; testPoints[1, 1] = 5;
        testPoints[2, 0] = 10; testPoints[2, 1] = 10;
        testPoints[3, 0] = -5; testPoints[3, 1] = -5;
        testPoints[4, 0] = 2.5; testPoints[4, 1] = 2.5;

        var probs = knn.PredictProbabilities(testPoints);

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
    public void KNN_WithUniformWeights_EqualVotes()
    {
        // Arrange: 3 neighbors with 2 class 0 and 1 class 1
        var x = new Matrix<double>(3, 1);
        var y = new Vector<double>(3);

        x[0, 0] = 0; y[0] = 0;
        x[1, 0] = 1; y[1] = 0;
        x[2, 0] = 2; y[2] = 1;

        var knn = new KNeighborsClassifier<double>(new KNeighborsOptions<double>
        {
            NNeighbors = 3,
            Weights = WeightingScheme.Uniform
        });
        knn.Train(x, y);

        // Act: Query at position 0.5 (equidistant from 0 and 1)
        var test = new Matrix<double>(1, 1);
        test[0, 0] = 1; // Closest to middle point

        var probs = knn.PredictProbabilities(test);

        // Assert: With uniform weights, probability should be 2/3 for class 0, 1/3 for class 1
        Assert.Equal(2.0 / 3.0, probs[0, 0], 0.01);
        Assert.Equal(1.0 / 3.0, probs[0, 1], 0.01);
    }

    [Fact]
    public void KNN_WithDistanceWeights_InverseDistanceVotes()
    {
        // Arrange: 2 neighbors - one very close, one far
        var x = new Matrix<double>(2, 1);
        var y = new Vector<double>(2);

        x[0, 0] = 0; y[0] = 0;   // Close neighbor (class 0)
        x[1, 0] = 10; y[1] = 1;  // Far neighbor (class 1)

        var knn = new KNeighborsClassifier<double>(new KNeighborsOptions<double>
        {
            NNeighbors = 2,
            Weights = WeightingScheme.Distance
        });
        knn.Train(x, y);

        // Act: Query at position 0.1 (very close to class 0)
        var test = new Matrix<double>(1, 1);
        test[0, 0] = 0.1;

        var probs = knn.PredictProbabilities(test);

        // Assert: Distance-weighted, class 0 should have much higher probability
        Assert.True(probs[0, 0] > probs[0, 1], "Closer neighbor should have more weight");
        Assert.True(probs[0, 0] > 0.9, "Very close neighbor should dominate probability");
    }

    #endregion

    #region Distance Metric Tests

    [Fact]
    public void KNN_WithEuclideanDistance_CorrectDistances()
    {
        // Arrange: Points where Euclidean distance is well-defined
        var x = new Matrix<double>(3, 2);
        var y = new Vector<double>(3);

        x[0, 0] = 0; x[0, 1] = 0; y[0] = 0;  // Origin
        x[1, 0] = 3; x[1, 1] = 0; y[1] = 1;  // (3, 0) - distance 3 from origin
        x[2, 0] = 0; x[2, 1] = 4; y[2] = 2;  // (0, 4) - distance 4 from origin

        var knn = new KNeighborsClassifier<double>(new KNeighborsOptions<double>
        {
            NNeighbors = 1,
            Metric = DistanceMetric.Euclidean
        });
        knn.Train(x, y);

        // Act: Query at (1, 0) - closer to origin than (3, 0)
        var test = new Matrix<double>(1, 2);
        test[0, 0] = 1; test[0, 1] = 0;

        var prediction = knn.Predict(test);

        // Assert: Should be classified as class 0 (origin is closer)
        Assert.Equal(0.0, prediction[0], Tolerance);
    }

    [Fact]
    public void KNN_WithManhattanDistance_CorrectDistances()
    {
        // Arrange: Points where Manhattan distance differs from Euclidean
        var x = new Matrix<double>(3, 2);
        var y = new Vector<double>(3);

        x[0, 0] = 0; x[0, 1] = 0; y[0] = 0;  // Origin, Manhattan = 0
        x[1, 0] = 2; x[1, 1] = 2; y[1] = 1;  // (2, 2), Manhattan = 4 from origin
        x[2, 0] = 3; x[2, 1] = 0; y[2] = 2;  // (3, 0), Manhattan = 3 from origin

        var knn = new KNeighborsClassifier<double>(new KNeighborsOptions<double>
        {
            NNeighbors = 1,
            Metric = DistanceMetric.Manhattan
        });
        knn.Train(x, y);

        // Act: Query at (1.5, 0) - Manhattan to origin=1.5, to (3,0)=1.5, to (2,2)=3.5
        var test = new Matrix<double>(1, 2);
        test[0, 0] = 1.5; test[0, 1] = 0;

        var prediction = knn.Predict(test);

        // Assert: Equidistant from origin and (3,0), but should pick one
        Assert.True(prediction[0] == 0 || prediction[0] == 2);
    }

    [Fact]
    public void KNN_WithChebyshevDistance_CorrectDistances()
    {
        // Arrange: Points where Chebyshev distance (max of abs differences) matters
        var x = new Matrix<double>(3, 2);
        var y = new Vector<double>(3);

        x[0, 0] = 0; x[0, 1] = 0; y[0] = 0;
        x[1, 0] = 2; x[1, 1] = 0; y[1] = 1;  // Chebyshev = 2 from origin
        x[2, 0] = 1; x[2, 1] = 3; y[2] = 2;  // Chebyshev = 3 from origin

        var knn = new KNeighborsClassifier<double>(new KNeighborsOptions<double>
        {
            NNeighbors = 1,
            Metric = DistanceMetric.Chebyshev
        });
        knn.Train(x, y);

        // Act: Query at (0.5, 0) - Chebyshev to origin=0.5, to (2,0)=1.5
        var test = new Matrix<double>(1, 2);
        test[0, 0] = 0.5; test[0, 1] = 0;

        var prediction = knn.Predict(test);

        // Assert: Origin is closest
        Assert.Equal(0.0, prediction[0], Tolerance);
    }

    [Fact]
    public void KNN_WithMinkowskiDistance_P2_EqualsEuclidean()
    {
        // Arrange
        var x = new Matrix<double>(3, 2);
        var y = new Vector<double>(3);

        x[0, 0] = 0; x[0, 1] = 0; y[0] = 0;
        x[1, 0] = 3; x[1, 1] = 4; y[1] = 1; // Distance 5 from origin (3-4-5 triangle)
        x[2, 0] = 10; x[2, 1] = 0; y[2] = 2;

        var knnMinkowski = new KNeighborsClassifier<double>(new KNeighborsOptions<double>
        {
            NNeighbors = 1,
            Metric = DistanceMetric.Minkowski,
            P = 2.0
        });
        knnMinkowski.Train(x, y);

        var knnEuclidean = new KNeighborsClassifier<double>(new KNeighborsOptions<double>
        {
            NNeighbors = 1,
            Metric = DistanceMetric.Euclidean
        });
        knnEuclidean.Train(x, y);

        // Act
        var test = new Matrix<double>(1, 2);
        test[0, 0] = 1; test[0, 1] = 1;

        var predMinkowski = knnMinkowski.Predict(test);
        var predEuclidean = knnEuclidean.Predict(test);

        // Assert: Should give same result
        Assert.Equal(predEuclidean[0], predMinkowski[0], Tolerance);
    }

    [Fact]
    public void KNN_WithCosineDistance_AngleBasedSimilarity()
    {
        // Arrange: Points with different magnitudes but same/different directions
        var x = new Matrix<double>(3, 2);
        var y = new Vector<double>(3);

        x[0, 0] = 1; x[0, 1] = 0; y[0] = 0;   // Direction (1, 0)
        x[1, 0] = 10; x[1, 1] = 0; y[1] = 0;  // Same direction (1, 0), different magnitude
        x[2, 0] = 0; x[2, 1] = 1; y[2] = 1;   // Orthogonal direction (0, 1)

        var knn = new KNeighborsClassifier<double>(new KNeighborsOptions<double>
        {
            NNeighbors = 1,
            Metric = DistanceMetric.Cosine
        });
        knn.Train(x, y);

        // Act: Query in direction (2, 0) - same direction as class 0
        var test = new Matrix<double>(1, 2);
        test[0, 0] = 2; test[0, 1] = 0;

        var prediction = knn.Predict(test);

        // Assert: Should be class 0 (same direction)
        Assert.Equal(0.0, prediction[0], Tolerance);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void KNN_WithKGreaterThanSamples_ThrowsException()
    {
        // Arrange
        var x = new Matrix<double>(3, 2);
        var y = new Vector<double>(3);

        x[0, 0] = 0; x[0, 1] = 0; y[0] = 0;
        x[1, 0] = 1; x[1, 1] = 1; y[1] = 1;
        x[2, 0] = 2; x[2, 1] = 2; y[2] = 1;

        var knn = new KNeighborsClassifier<double>(new KNeighborsOptions<double> { NNeighbors = 10 });

        // Act & Assert
        Assert.Throws<ArgumentException>(() => knn.Train(x, y));
    }

    [Fact]
    public void KNN_Serialize_Deserialize_PreservesPredictions()
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

        var knn = new KNeighborsClassifier<double>(new KNeighborsOptions<double> { NNeighbors = 3 });
        knn.Train(x, y);

        // Get original predictions
        var testPoint = new Matrix<double>(1, 2);
        testPoint[0, 0] = 1; testPoint[0, 1] = 1;
        var originalProbs = knn.PredictProbabilities(testPoint);

        // Act: Serialize and deserialize
        byte[] serialized = knn.Serialize();
        var knn2 = new KNeighborsClassifier<double>();
        knn2.Deserialize(serialized);

        var newProbs = knn2.PredictProbabilities(testPoint);

        // Assert
        Assert.Equal(originalProbs[0, 0], newProbs[0, 0], Tolerance);
        Assert.Equal(originalProbs[0, 1], newProbs[0, 1], Tolerance);
    }

    [Fact]
    public void KNN_Clone_CreatesIndependentCopy()
    {
        // Arrange
        var x = new Matrix<double>(4, 2);
        var y = new Vector<double>(4);

        x[0, 0] = 0; x[0, 1] = 0; y[0] = 0;
        x[1, 0] = 1; x[1, 1] = 1; y[1] = 0;
        x[2, 0] = 5; x[2, 1] = 5; y[2] = 1;
        x[3, 0] = 6; x[3, 1] = 6; y[3] = 1;

        var knn = new KNeighborsClassifier<double>(new KNeighborsOptions<double> { NNeighbors = 2 });
        knn.Train(x, y);

        // Act
        var clone = knn.Clone() as KNeighborsClassifier<double>;

        // Assert
        Assert.NotNull(clone);

        var testPoint = new Matrix<double>(1, 2);
        testPoint[0, 0] = 0.5; testPoint[0, 1] = 0.5;

        var originalProbs = knn.PredictProbabilities(testPoint);
        var cloneProbs = clone!.PredictProbabilities(testPoint);

        Assert.Equal(originalProbs[0, 0], cloneProbs[0, 0], Tolerance);
        Assert.Equal(originalProbs[0, 1], cloneProbs[0, 1], Tolerance);
    }

    [Fact]
    public void KNN_SingleFeature_WorksCorrectly()
    {
        // Arrange: 1D classification
        var x = new Matrix<double>(4, 1);
        var y = new Vector<double>(4);

        x[0, 0] = 0; y[0] = 0;
        x[1, 0] = 1; y[1] = 0;
        x[2, 0] = 10; y[2] = 1;
        x[3, 0] = 11; y[3] = 1;

        var knn = new KNeighborsClassifier<double>(new KNeighborsOptions<double> { NNeighbors = 2 });
        knn.Train(x, y);

        // Act
        var test = new Matrix<double>(2, 1);
        test[0, 0] = 0.5;  // Near class 0
        test[1, 0] = 10.5; // Near class 1

        var predictions = knn.Predict(test);

        // Assert
        Assert.Equal(0.0, predictions[0], Tolerance);
        Assert.Equal(1.0, predictions[1], Tolerance);
    }

    [Fact]
    public void KNN_HighDimensionalData_Stable()
    {
        // Arrange: High dimensional data
        int numFeatures = 20;
        int numSamples = 10;

        var x = new Matrix<double>(numSamples, numFeatures);
        var y = new Vector<double>(numSamples);

        var rand = new Random(42);
        for (int i = 0; i < numSamples; i++)
        {
            int classLabel = i < numSamples / 2 ? 0 : 1;
            double offset = classLabel * 10.0;
            for (int j = 0; j < numFeatures; j++)
            {
                x[i, j] = offset + rand.NextDouble();
            }
            y[i] = classLabel;
        }

        var knn = new KNeighborsClassifier<double>(new KNeighborsOptions<double> { NNeighbors = 3 });
        knn.Train(x, y);

        // Act
        var predictions = knn.Predict(x);
        var probs = knn.PredictProbabilities(x);

        // Assert
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
    public void KNN_WithTies_HandlesGracefully()
    {
        // Arrange: Create a tie situation with k=2, 1 neighbor from each class
        var x = new Matrix<double>(2, 1);
        var y = new Vector<double>(2);

        x[0, 0] = 0; y[0] = 0;
        x[1, 0] = 2; y[1] = 1;

        var knn = new KNeighborsClassifier<double>(new KNeighborsOptions<double>
        {
            NNeighbors = 2,
            Weights = WeightingScheme.Uniform
        });
        knn.Train(x, y);

        // Act: Query at midpoint (equidistant from both)
        var test = new Matrix<double>(1, 1);
        test[0, 0] = 1;

        var probs = knn.PredictProbabilities(test);

        // Assert: With uniform weights and tie, each class should have 0.5 probability
        Assert.Equal(0.5, probs[0, 0], Tolerance);
        Assert.Equal(0.5, probs[0, 1], Tolerance);
    }

    [Fact]
    public void KNN_ExactMatch_ReturnsCorrectClass()
    {
        // Arrange
        var x = new Matrix<double>(4, 2);
        var y = new Vector<double>(4);

        x[0, 0] = 1; x[0, 1] = 1; y[0] = 0;
        x[1, 0] = 2; x[1, 1] = 2; y[1] = 0;
        x[2, 0] = 5; x[2, 1] = 5; y[2] = 1;
        x[3, 0] = 6; x[3, 1] = 6; y[3] = 1;

        var knn = new KNeighborsClassifier<double>(new KNeighborsOptions<double>
        {
            NNeighbors = 1,
            Weights = WeightingScheme.Distance
        });
        knn.Train(x, y);

        // Act: Query exactly at a training point
        var test = new Matrix<double>(1, 2);
        test[0, 0] = 1; test[0, 1] = 1; // Exact match to point at (1,1)

        var prediction = knn.Predict(test);
        var probs = knn.PredictProbabilities(test);

        // Assert
        Assert.Equal(0.0, prediction[0], Tolerance);
        Assert.True(probs[0, 0] > 0.99, "Exact match should have very high probability");
    }

    #endregion
}

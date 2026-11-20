using AiDotNet.Classification.Calibration;
using AiDotNet.Classification.DiscriminantAnalysis;
using AiDotNet.Classification.ImbalancedEnsemble;
using AiDotNet.Classification.Neighbors;
using AiDotNet.Models.Options;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.Classification;

/// <summary>
/// Integration tests for K-Nearest Neighbors classifier.
/// These tests verify mathematical correctness and do NOT trust the implementation.
/// </summary>
public class KNeighborsIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region Basic Prediction Tests

    [Fact(Timeout = 120000)]
    public async Task KNN_Train_StoresTrainingData()
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

    [Fact(Timeout = 120000)]
    public async Task KNN_Predict_WithK1_ReturnsNearestNeighbor()
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

    [Fact(Timeout = 120000)]
    public async Task KNN_Predict_WithK3_UsesMajorityVoting()
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

    [Fact(Timeout = 120000)]
    public async Task KNN_Predict_BinaryClassification()
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

    [Fact(Timeout = 120000)]
    public async Task KNN_Predict_MultiClassClassification()
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

    [Fact(Timeout = 120000)]
    public async Task KNN_PredictProbabilities_SumsToOne()
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

    [Fact(Timeout = 120000)]
    public async Task KNN_WithUniformWeights_EqualVotes()
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

    [Fact(Timeout = 120000)]
    public async Task KNN_WithDistanceWeights_InverseDistanceVotes()
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

    [Fact(Timeout = 120000)]
    public async Task KNN_WithEuclideanDistance_CorrectDistances()
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

    [Fact(Timeout = 120000)]
    public async Task KNN_WithManhattanDistance_CorrectDistances()
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

    [Fact(Timeout = 120000)]
    public async Task KNN_WithChebyshevDistance_CorrectDistances()
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

    [Fact(Timeout = 120000)]
    public async Task KNN_WithMinkowskiDistance_P2_EqualsEuclidean()
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

    [Fact(Timeout = 120000)]
    public async Task KNN_WithCosineDistance_AngleBasedSimilarity()
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

    [Fact(Timeout = 120000)]
    public async Task KNN_WithKGreaterThanSamples_ThrowsException()
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

    [Fact(Timeout = 120000)]
    public async Task KNN_Serialize_Deserialize_PreservesPredictions()
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

    [Fact(Timeout = 120000)]
    public async Task KNN_Clone_CreatesIndependentCopy()
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
        var cloneProbs = clone.PredictProbabilities(testPoint);

        Assert.Equal(originalProbs[0, 0], cloneProbs[0, 0], Tolerance);
        Assert.Equal(originalProbs[0, 1], cloneProbs[0, 1], Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task KNN_SingleFeature_WorksCorrectly()
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

    [Fact(Timeout = 120000)]
    public async Task KNN_HighDimensionalData_Stable()
    {
        // Arrange: High dimensional data
        int numFeatures = 20;
        int numSamples = 10;

        var x = new Matrix<double>(numSamples, numFeatures);
        var y = new Vector<double>(numSamples);

        var rand = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(42);
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

    [Fact(Timeout = 120000)]
    public async Task KNN_WithTies_HandlesGracefully()
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

    [Fact(Timeout = 120000)]
    public async Task KNN_ExactMatch_ReturnsCorrectClass()
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

    [Fact(Timeout = 120000)]
    public async Task KNN_TiedDistancesAtTheKBoundary_SelectInPartialSelectionSortOrder()
    {
        // Arrange: distances from the query at 0 are [3, 1, 3, 2] for samples 0..3, so samples 0 and 2 tie
        // for the third neighbor. The partial selection sort (swap the first strict minimum into place)
        // takes sample 1, then sample 3 (swapping sample 0 to position 3), then sample 2, which now sits
        // before sample 0. A stable sort would take sample 0 instead.
        var x = new Matrix<double>(4, 1);
        var y = new Vector<double>(4);
        x[0, 0] = 3; y[0] = 0;
        x[1, 0] = 1; y[1] = 1;
        x[2, 0] = -3; y[2] = 2;
        x[3, 0] = 2; y[3] = 1;

        var knn = new KNeighborsClassifier<double>(new KNeighborsOptions<double> { NNeighbors = 3 });
        knn.Train(x, y);

        var test = new Matrix<double>(1, 1);
        test[0, 0] = 0;

        // Act
        var probs = knn.PredictProbabilities(test);

        // Assert: neighbors are samples 1, 3 and 2 -> classes {1, 1, 2}
        Assert.Equal(0.0, probs[0, 0], Tolerance);
        Assert.Equal(2.0 / 3.0, probs[0, 1], Tolerance);
        Assert.Equal(1.0 / 3.0, probs[0, 2], Tolerance);
    }

    #endregion

    #region Shared Classifier Helper Regressions

    // 1 + 2^-23 and 1 + 2^-22: adjacent floats whose midpoint (a + b) / 2 rounds up to b in float, so a
    // threshold taken as the plain midpoint sends both values to the "<= threshold" side.
    private static float FloatFromBits(int bits) => BitConverter.ToSingle(BitConverter.GetBytes(bits), 0);

    [Fact(Timeout = 120000)]
    public async Task BalancedRandomForest_AdjacentFloatValues_SplitSeparatesThem()
    {
        // Arrange: one sample per class, at adjacent float values. The best Gini split lies between them; its
        // threshold must stay below the upper value (scikit-learn uses the lower value when the midpoint rounds
        // up to the upper one), otherwise every sample goes left and the leaves predict the majority class.
        float lower = FloatFromBits(0x3F800001);
        float upper = FloatFromBits(0x3F800002);
        var x = new Matrix<float>(2, 1);
        var y = new Vector<float>(2);
        x[0, 0] = lower; y[0] = 0f;
        x[1, 0] = upper; y[1] = 1f;

        var forest = new BalancedRandomForestClassifier<float>(nEstimators: 3, maxDepth: 4, seed: 7);
        forest.Train(x, y);

        // Act
        var predictions = forest.Predict(x);

        // Assert
        Assert.Equal(0f, predictions[0]);
        Assert.Equal(1f, predictions[1]);
    }

    [Fact(Timeout = 120000)]
    public async Task EasyEnsemble_AdjacentFloatValues_StumpSeparatesThem()
    {
        // Arrange: the stump between two adjacent floats has zero weighted error only if its threshold stays
        // below the upper value; a threshold that rounds up to it predicts class 0 for both samples.
        float lower = FloatFromBits(0x3F800001);
        float upper = FloatFromBits(0x3F800002);
        var x = new Matrix<float>(2, 1);
        var y = new Vector<float>(2);
        x[0, 0] = lower; y[0] = 0f;
        x[1, 0] = upper; y[1] = 1f;

        var ensemble = new EasyEnsembleClassifier<float>(nSubsets: 1, nEstimatorsPerSubset: 1, seed: 3);
        ensemble.Train(x, y);

        // Act
        var predictions = ensemble.Predict(x);

        // Assert
        Assert.Equal(0f, predictions[0]);
        Assert.Equal(1f, predictions[1]);
    }

    [Fact(Timeout = 120000)]
    public async Task EasyEnsemble_Half_SeparableData_KeepsFiniteBoostingWeights()
    {
        // Arrange: a perfectly separating stump has weighted error 0. The error is clamped away from 0 before
        // log((1 - err) / err); 1e-10 rounds to 0 in Half, which made alpha infinite, the sample weights NaN
        // and every later prediction class 0.
        var x = new Matrix<Half>(4, 1);
        var y = new Vector<Half>(4);
        for (int i = 0; i < 4; i++)
        {
            x[i, 0] = (Half)(float)i;
            y[i] = i < 2 ? (Half)0f : (Half)1f;
        }

        var ensemble = new EasyEnsembleClassifier<Half>(nSubsets: 1, nEstimatorsPerSubset: 3, softVoting: false, seed: 5);
        ensemble.Train(x, y);

        // Act
        var predictions = ensemble.Predict(x);

        // Assert
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<Half>();
        Assert.Equal(0.0, ops.ToDouble(predictions[0]));
        Assert.Equal(0.0, ops.ToDouble(predictions[1]));
        Assert.Equal(1.0, ops.ToDouble(predictions[2]));
        Assert.Equal(1.0, ops.ToDouble(predictions[3]));
    }

    [Fact(Timeout = 120000)]
    public async Task CalibratedClassifier_Float_PlattScalingOnCertainProbabilities_StaysFinite()
    {
        // Arrange: a 1-nearest-neighbor base model gives probabilities of exactly 0 and 1. Platt scaling clamps
        // them to [1e-10, 1 - 1e-10] before taking logit(p); 1 - 1e-10 rounds to 1 in float, so logit(1) was
        // infinite and the fitted A and B (and every calibrated probability) became NaN.
        var x = new Matrix<float>(20, 1);
        var y = new Vector<float>(20);
        for (int i = 0; i < 20; i++)
        {
            x[i, 0] = i;
            y[i] = i < 10 ? 0f : 1f;
        }

        var calibrated = new CalibratedClassifier<float>(
            new KNeighborsClassifier<float>(new KNeighborsOptions<float> { NNeighbors = 1 }),
            new CalibratedClassifierOptions<float> { CalibrationMethod = AiDotNet.Enums.ProbabilityCalibrationMethod.PlattScaling, Seed = 11 });
        calibrated.Train(x, y);

        // Act
        var probs = calibrated.PredictProbabilities(x);

        // Assert: finite, and the positive class is more likely at the class-1 end than at the class-0 end
        for (int i = 0; i < probs.Rows; i++)
        {
            for (int c = 0; c < probs.Columns; c++)
            {
                Assert.False(float.IsNaN(probs[i, c]) || float.IsInfinity(probs[i, c]), $"probs[{i}, {c}] = {probs[i, c]}");
            }
        }

        Assert.True(probs[19, 1] > probs[0, 1]);
    }

    [Fact(Timeout = 120000)]
    public async Task LinearDiscriminantAnalysis_Half_PriorsCountEveryClassSample()
    {
        // Arrange: both classes have mean about 0 (alternating +1/-1), so the posterior equals the class prior,
        // 3001 / 4001, about 0.75. Counting class samples as a sum of ones in Half stops at 2048, which gave priors
        // 2048/4000 and 1000/4000 and a posterior of about 0.67.
        int classZero = 3001;
        int classOne = 1000;
        int n = classZero + classOne;
        var x = new Matrix<Half>(n, 1);
        var y = new Vector<Half>(n);
        for (int i = 0; i < n; i++)
        {
            x[i, 0] = i % 2 == 0 ? (Half)1f : (Half)(-1f);
            y[i] = i < classZero ? (Half)0f : (Half)1f;
        }

        var lda = new LinearDiscriminantAnalysis<Half>();
        lda.Train(x, y);
        var query = new Matrix<Half>(1, 1);

        // Act
        var probs = lda.PredictProbabilities(query);

        // Assert: the class counts are exact ints (3001 is not a Half value, so no count kept in Half can
        // equal it), and the posterior is the prior 3001 / 4001
        Assert.Equal(new[] { 3001, 1000 }, lda.CountClasses(y).ToArray());
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<Half>();
        Assert.Equal(0.75, ops.ToDouble(probs[0, 0]), 2);
        Assert.Equal(0.25, ops.ToDouble(probs[0, 1]), 2);
    }

    [Fact(Timeout = 120000)]
    public async Task KNN_Double_DistanceWeights_ExactMatchTakesAllWeight()
    {
        // Arrange: as scikit-learn's weights='distance', neighbors at distance 0 get weight 1 and the others 0,
        // so the query at a class-0 training point is class 0 with probability exactly 1. A fixed weight of 1e10
        // left 1.5 / (1e10 + 1.5) for class 1.
        var x = new Matrix<double>(3, 1);
        var y = new Vector<double>(3);
        x[0, 0] = 0; y[0] = 0;
        x[1, 0] = 1; y[1] = 1;
        x[2, 0] = 2; y[2] = 1;
        var knn = new KNeighborsClassifier<double>(new KNeighborsOptions<double> { NNeighbors = 3, Weights = WeightingScheme.Distance });
        knn.Train(x, y);

        // Act
        var probs = knn.PredictProbabilities(new Matrix<double>(1, 1));

        // Assert
        Assert.Equal(1.0, probs[0, 0]);
        Assert.Equal(0.0, probs[0, 1]);
    }

    [Fact(Timeout = 120000)]
    public async Task KNN_Half_DistanceWeights_ExactMatchTakesAllWeight()
    {
        // Arrange: the fixed exact-match weight 1e10 is infinite in Half, so class 0 got inf / inf = NaN.
        var x = new Matrix<Half>(3, 1);
        var y = new Vector<Half>(3);
        x[0, 0] = (Half)0f; y[0] = (Half)0f;
        x[1, 0] = (Half)1f; y[1] = (Half)1f;
        x[2, 0] = (Half)2f; y[2] = (Half)1f;
        var knn = new KNeighborsClassifier<Half>(new KNeighborsOptions<Half> { NNeighbors = 3, Weights = WeightingScheme.Distance });
        knn.Train(x, y);

        // Act
        var probs = knn.PredictProbabilities(new Matrix<Half>(1, 1));

        // Assert
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<Half>();
        Assert.Equal(1.0, ops.ToDouble(probs[0, 0]));
        Assert.Equal(0.0, ops.ToDouble(probs[0, 1]));
    }

    [Fact(Timeout = 120000)]
    public async Task KNN_Half_EuclideanDistancesPast255_PickTheNearestNeighbor()
    {
        // Arrange: from 0 the class-1 point (300) is nearer than the class-0 point (-1000), but both squared
        // differences overflow Half, so both distances were infinite and the first training sample won.
        var x = new Matrix<Half>(2, 1);
        var y = new Vector<Half>(2);
        x[0, 0] = (Half)(-1000f); y[0] = (Half)0f;
        x[1, 0] = (Half)300f; y[1] = (Half)1f;
        var knn = new KNeighborsClassifier<Half>(new KNeighborsOptions<Half> { NNeighbors = 1 });
        knn.Train(x, y);

        // Act
        var predictions = knn.Predict(new Matrix<Half>(1, 1));

        // Assert
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<Half>();
        Assert.Equal(1.0, ops.ToDouble(predictions[0]));
    }

    [Fact(Timeout = 120000)]
    public async Task KNN_Half_CosineOfLargeRows_PicksTheSameDirection()
    {
        // Arrange: the query (600, 10) points along the class-0 row (300, 0). Its dot product with that row and
        // the squared norms overflow Half, which made that cosine distance inf / inf = NaN and the class-1 row
        // (distance 1) the nearest.
        var x = new Matrix<Half>(2, 2);
        var y = new Vector<Half>(2);
        x[0, 0] = (Half)0f; x[0, 1] = (Half)300f; y[0] = (Half)1f;
        x[1, 0] = (Half)300f; x[1, 1] = (Half)0f; y[1] = (Half)0f;
        var knn = new KNeighborsClassifier<Half>(new KNeighborsOptions<Half> { NNeighbors = 1, Metric = DistanceMetric.Cosine });
        knn.Train(x, y);
        var query = new Matrix<Half>(1, 2);
        query[0, 0] = (Half)600f;
        query[0, 1] = (Half)10f;

        // Act
        var predictions = knn.Predict(query);

        // Assert
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<Half>();
        Assert.Equal(0.0, ops.ToDouble(predictions[0]));
    }

    [Fact(Timeout = 120000)]
    public async Task NuSupportVectorClassifier_Half_SmallNu_KeepsSupportVectors()
    {
        // Arrange: each class's multipliers sum to nu / 2 = 5e-5, below Half's smallest normal value 6.1e-5.
        // Raising the support-vector threshold 1e-10 to that value dropped every sample from the support set.
        var x = new Matrix<Half>(20, 1);
        var y = new Vector<Half>(20);
        for (int i = 0; i < 10; i++)
        {
            x[i, 0] = (Half)(-1f - i); y[i] = (Half)0f;
            x[10 + i, 0] = (Half)(1f + i); y[10 + i] = (Half)1f;
        }

        var options = new AiDotNet.Models.Options.SVMOptions<Half> { Kernel = AiDotNet.Enums.KernelType.Linear, MaxIterations = 50, Seed = 42 };
        var nuSvc = new AiDotNet.Classification.SVM.NuSupportVectorClassifier<Half>(options, null, nu: 0.0001);

        // Act
        nuSvc.Train(x, y);

        // Assert
        Assert.True(nuSvc.NSupportVectors > 0, "Every sample with a positive multiplier is a support vector.");
    }

    [Fact(Timeout = 120000)]
    public async Task OrdinalRidgeRegression_Half_TargetSumPast65504_PredictsBothClasses()
    {
        // Arrange: the target mean was a Half sum of 100 labels near 1000 (about 100100, infinite) divided by 100,
        // which made every coefficient NaN and every prediction the lowest class.
        var x = new Matrix<Half>(100, 1);
        var y = new Vector<Half>(100);
        for (int i = 0; i < 100; i++)
        {
            bool upper = i % 2 == 1;
            x[i, 0] = upper ? (Half)1f : (Half)0f;
            y[i] = upper ? (Half)1002f : (Half)1000f;
        }

        var ordinal = new AiDotNet.Classification.Ordinal.OrdinalRidgeRegression<Half>();
        ordinal.Train(x, y);

        // Act
        var predictions = ordinal.Predict(x);

        // Assert
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<Half>();
        Assert.Equal(1000.0, ops.ToDouble(predictions[0]));
        Assert.Equal(1002.0, ops.ToDouble(predictions[1]));
    }

    [Fact(Timeout = 120000)]
    public async Task MLkNN_Half_DistancesPast255_UseTheNearestNeighbor()
    {
        // Arrange: label 1 is held by 300 and 600, label 0 by -1000, -1300 and -1600. From 0 the nearest training
        // sample is 300 (label 1), so P(label 1) = 0.75 * 3/7 / (0.75 * 3/7 + 0.2 * 4/7), about 0.74. Every squared
        // difference overflowed Half, so all distances were infinite and the neighbors were arbitrary.
        float[] positions = { -1000f, -1300f, 300f, 600f, -1600f };
        var x = new Matrix<Half>(5, 1);
        var labels = new Matrix<Half>(5, 2);
        for (int i = 0; i < 5; i++)
        {
            x[i, 0] = (Half)positions[i];
            bool hasLabelOne = positions[i] > 0f;
            labels[i, 0] = hasLabelOne ? (Half)0f : (Half)1f;
            labels[i, 1] = hasLabelOne ? (Half)1f : (Half)0f;
        }

        var mlknn = new AiDotNet.Classification.MultiLabel.MLkNNClassifier<Half>(new MLkNNOptions<Half> { KNeighbors = 1, Smoothing = 1.0 });
        mlknn.Train(x, labels);

        // Act
        var probs = mlknn.PredictMultiLabelProbabilities(new Matrix<Half>(1, 1));

        // Assert
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<Half>();
        Assert.Equal(2.25 / 3.05, ops.ToDouble(probs[0, 1]), 2);
    }

    [Fact(Timeout = 120000)]
    public async Task GaussianNaiveBayes_Half_ClassSumsPast65504_KeepFiniteMeans()
    {
        // Arrange: 100 samples per class near 1000 and 1010. The per-class feature sums (about 100000) overflowed
        // Half before the division by the count, so the means were infinite and every prediction class 0.
        var x = new Matrix<Half>(200, 1);
        var y = new Vector<Half>(200);
        for (int i = 0; i < 100; i++)
        {
            float offset = i % 2 == 0 ? -1f : 1f;
            x[i, 0] = (Half)(1000f + offset); y[i] = (Half)0f;
            x[100 + i, 0] = (Half)(1010f + offset); y[100 + i] = (Half)1f;
        }

        var gnb = new AiDotNet.Classification.NaiveBayes.GaussianNaiveBayes<Half>();
        gnb.Train(x, y);
        var query = new Matrix<Half>(2, 1);
        query[0, 0] = (Half)1000f;
        query[1, 0] = (Half)1010f;

        // Act
        var predictions = gnb.Predict(query);

        // Assert
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<Half>();
        Assert.Equal(0.0, ops.ToDouble(predictions[0]));
        Assert.Equal(1.0, ops.ToDouble(predictions[1]));
    }

    [Fact(Timeout = 120000)]
    public async Task AddClassVotes_Half_CountsPast2048Exactly()
    {
        // Arrange: ensemble votes are counted per class; a sum of one-hot rows in Half stops at 2048.
        var x = new Matrix<Half>(2, 1);
        var y = new Vector<Half>(2);
        x[1, 0] = (Half)1f;
        y[1] = (Half)1f;
        var knn = new KNeighborsClassifier<Half>(new KNeighborsOptions<Half> { NNeighbors = 1 });
        knn.Train(x, y);
        var votes = new AiDotNet.Tensors.LinearAlgebra.Tensor<int>(new[] { 1, 2 });
        var classZero = new Vector<Half>(new[] { (Half)0f });

        // Act
        for (int i = 0; i < 3001; i++)
        {
            votes = knn.AddClassVotes(votes, classZero);
        }

        // Assert
        Assert.Equal(3001, votes[0, 0]);
        Assert.Equal(0, votes[0, 1]);
    }

    [Fact(Timeout = 120000)]
    public async Task KNN_ExactlyEquidistantNeighbors_KeepTheLowerTrainingIndex()
    {
        await Task.Yield();

        // Arrange: (2, 9) and (6, 7) are both sqrt(85) from the origin. Scaling each difference by its own largest
        // component before squaring rounds the two norms one ulp apart (9.219544457292889 against ...887), so the
        // second point won instead of the first.
        var x = new Matrix<double>(2, 2);
        x[0, 0] = 2; x[0, 1] = 9;
        x[1, 0] = 6; x[1, 1] = 7;
        var y = new Vector<double>(new[] { 0.0, 1.0 });
        var knn = new KNeighborsClassifier<double>(new KNeighborsOptions<double> { NNeighbors = 1 });
        knn.Train(x, y);

        // Act
        var prediction = knn.Predict(new Matrix<double>(1, 2));

        // Assert
        Assert.Equal(0.0, prediction[0]);
    }

    [Fact(Timeout = 120000)]
    public async Task KNN_DistanceWeights_Decimal_ExactMatch_DoesNotDivideByZero()
    {
        await Task.Yield();

        // Arrange: predicting a training point with distance weighting. The inverse distances were computed for
        // every neighbor before the exact matches were masked, which throws DivideByZeroException for decimal.
        var x = new Matrix<decimal>(4, 1);
        x[0, 0] = 0m; x[1, 0] = 1m; x[2, 0] = 10m; x[3, 0] = 11m;
        var y = new Vector<decimal>(new[] { 0m, 0m, 1m, 1m });
        var knn = new KNeighborsClassifier<decimal>(
            new KNeighborsOptions<decimal> { NNeighbors = 3, Weights = WeightingScheme.Distance });
        knn.Train(x, y);

        // Act
        var probabilities = knn.PredictProbabilities(x);

        // Assert: an exact match takes all the weight
        Assert.Equal(1m, probabilities[0, 0]);
        Assert.Equal(0m, probabilities[0, 1]);
        Assert.Equal(1m, probabilities[3, 1]);
    }

    #endregion
}

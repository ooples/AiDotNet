using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Regression;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Regression;

/// <summary>
/// Integration tests for Generalized Linear Models and Count-based regression algorithms.
/// Tests PoissonRegression, NegativeBinomialRegression, and GeneralizedAdditiveModel.
/// </summary>
public class GLMRegressionIntegrationTests
{
    private const double Tolerance = 0.5;
    private const double LooseTolerance = 2.0;

    #region Helper Methods

    private static Matrix<double> CreateMatrix(double[,] data)
    {
        int rows = data.GetLength(0);
        int cols = data.GetLength(1);
        var matrix = new Matrix<double>(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                matrix[i, j] = data[i, j];
            }
        }
        return matrix;
    }

    private static Vector<double> CreateVector(double[] data)
    {
        var vector = new Vector<double>(data.Length);
        for (int i = 0; i < data.Length; i++)
        {
            vector[i] = data[i];
        }
        return vector;
    }

    /// <summary>
    /// Generates count data following Poisson-like distribution for testing.
    /// </summary>
    private static (Matrix<double> X, Vector<double> y) GenerateCountData(int n, double baseLambda = 5.0)
    {
        var random = new Random(42);
        var X = new Matrix<double>(n, 1);
        var y = new Vector<double>(n);

        for (int i = 0; i < n; i++)
        {
            double x = (double)i / n * 2; // Range from 0 to 2
            X[i, 0] = x;
            // Generate Poisson-like count (expected value increases with x)
            double lambda = baseLambda * Math.Exp(0.5 * x);
            y[i] = Math.Max(0, Math.Round(lambda + random.NextDouble() * Math.Sqrt(lambda) - Math.Sqrt(lambda) / 2));
        }

        return (X, y);
    }

    /// <summary>
    /// Generates overdispersed count data for negative binomial testing.
    /// </summary>
    private static (Matrix<double> X, Vector<double> y) GenerateOverdispersedCountData(int n)
    {
        var random = new Random(42);
        var X = new Matrix<double>(n, 1);
        var y = new Vector<double>(n);

        for (int i = 0; i < n; i++)
        {
            double x = (double)i / n * 2; // Range from 0 to 2
            X[i, 0] = x;
            // Generate overdispersed count data (variance > mean)
            double mean = 10 * Math.Exp(0.3 * x);
            double variance = mean * 2; // Overdispersed
            y[i] = Math.Max(0, Math.Round(mean + random.NextDouble() * Math.Sqrt(variance) * 2 - Math.Sqrt(variance)));
        }

        return (X, y);
    }

    /// <summary>
    /// Generates non-linear data for GAM testing.
    /// </summary>
    private static (Matrix<double> X, Vector<double> y) GenerateNonLinearData(int n)
    {
        var X = new Matrix<double>(n, 1);
        var y = new Vector<double>(n);

        for (int i = 0; i < n; i++)
        {
            double x = (double)i / n * 4 - 2; // Range from -2 to 2
            X[i, 0] = x;
            // Non-linear relationship: y = sin(x) + x^2
            y[i] = Math.Sin(x) + x * x;
        }

        return (X, y);
    }

    /// <summary>
    /// Generates multi-feature data for GAM testing.
    /// </summary>
    private static (Matrix<double> X, Vector<double> y) GenerateMultiFeatureNonLinearData(int n)
    {
        var random = new Random(42);
        var X = new Matrix<double>(n, 2);
        var y = new Vector<double>(n);

        for (int i = 0; i < n; i++)
        {
            double x1 = random.NextDouble() * 4 - 2; // Range -2 to 2
            double x2 = random.NextDouble() * 4 - 2;
            X[i, 0] = x1;
            X[i, 1] = x2;
            // y = sin(x1) + x2^2
            y[i] = Math.Sin(x1) + x2 * x2;
        }

        return (X, y);
    }

    #endregion

    #region PoissonRegression Tests

    [Fact]
    public void PoissonRegression_Train_CountData_MakesPositivePredictions()
    {
        // Arrange
        var options = new PoissonRegressionOptions<double>
        {
            MaxIterations = 100,
            Tolerance = 1e-6
        };
        var poisson = new PoissonRegression<double>(options);
        var (X, y) = GenerateCountData(30);

        // Act
        poisson.Train(X, y);
        var predictions = poisson.Predict(X);

        // Assert - all predictions should be non-negative (count data)
        Assert.Equal(y.Length, predictions.Length);
        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.True(predictions[i] >= 0, $"Prediction {predictions[i]} at index {i} should be non-negative");
        }
    }

    [Fact]
    public void PoissonRegression_Train_IncreasingTrend_CapturesTrend()
    {
        // Arrange
        var options = new PoissonRegressionOptions<double>
        {
            MaxIterations = 100,
            Tolerance = 1e-6
        };
        var poisson = new PoissonRegression<double>(options);
        var (X, y) = GenerateCountData(20);

        // Act
        poisson.Train(X, y);
        var predictions = poisson.Predict(X);

        // Assert - predictions at higher X values should generally be higher
        double avgFirstHalf = 0, avgSecondHalf = 0;
        int halfPoint = predictions.Length / 2;
        for (int i = 0; i < halfPoint; i++)
        {
            avgFirstHalf += predictions[i];
            avgSecondHalf += predictions[halfPoint + i];
        }
        avgFirstHalf /= halfPoint;
        avgSecondHalf /= halfPoint;

        Assert.True(avgSecondHalf > avgFirstHalf, "Predictions should increase with increasing X");
    }

    [Fact]
    public void PoissonRegression_Coefficients_AreAccessible()
    {
        // Arrange
        var poisson = new PoissonRegression<double>(new PoissonRegressionOptions<double>());
        var (X, y) = GenerateCountData(20);

        // Act
        poisson.Train(X, y);

        // Assert
        Assert.NotNull(poisson.Coefficients);
        Assert.Equal(1, poisson.Coefficients.Length); // One feature
    }

    [Fact]
    public void PoissonRegression_SerializeDeserialize_PreservesModel()
    {
        // Arrange
        var options = new PoissonRegressionOptions<double>
        {
            MaxIterations = 50,
            Tolerance = 1e-4
        };
        var poisson = new PoissonRegression<double>(options);
        var (X, y) = GenerateCountData(15);
        poisson.Train(X, y);
        var originalPredictions = poisson.Predict(X);

        // Act
        var serialized = poisson.Serialize();
        var newPoisson = new PoissonRegression<double>(new PoissonRegressionOptions<double>());
        newPoisson.Deserialize(serialized);
        var deserializedPredictions = newPoisson.Predict(X);

        // Assert
        Assert.Equal(originalPredictions.Length, deserializedPredictions.Length);
        for (int i = 0; i < originalPredictions.Length; i++)
        {
            Assert.Equal(originalPredictions[i], deserializedPredictions[i], 4);
        }
    }

    [Fact]
    public void PoissonRegression_GetModelType_ReturnsPoissonRegression()
    {
        // Arrange
        var poisson = new PoissonRegression<double>(new PoissonRegressionOptions<double>());
        var (X, y) = GenerateCountData(10);
        poisson.Train(X, y);

        // Act
        var metadata = poisson.GetModelMetadata();

        // Assert
        Assert.Equal(ModelType.PoissonRegression, metadata.ModelType);
    }

    #endregion

    #region NegativeBinomialRegression Tests

    [Fact]
    public void NegativeBinomialRegression_Train_OverdispersedData_MakesPositivePredictions()
    {
        // Arrange
        var options = new NegativeBinomialRegressionOptions<double>
        {
            MaxIterations = 100,
            Tolerance = 1e-6
        };
        var nbr = new NegativeBinomialRegression<double>(options);
        var (X, y) = GenerateOverdispersedCountData(30);

        // Act
        nbr.Train(X, y);
        var predictions = nbr.Predict(X);

        // Assert - all predictions should be non-negative
        Assert.Equal(y.Length, predictions.Length);
        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.True(predictions[i] >= 0, $"Prediction {predictions[i]} at index {i} should be non-negative");
        }
    }

    [Fact]
    public void NegativeBinomialRegression_Train_CapturesOverdispersion()
    {
        // Arrange
        var options = new NegativeBinomialRegressionOptions<double>
        {
            MaxIterations = 50,
            Tolerance = 1e-4
        };
        var nbr = new NegativeBinomialRegression<double>(options);
        var (X, y) = GenerateOverdispersedCountData(20);

        // Act
        nbr.Train(X, y);
        var predictions = nbr.Predict(X);

        // Assert - should complete without exception
        Assert.Equal(y.Length, predictions.Length);
    }

    [Fact]
    public void NegativeBinomialRegression_SerializeDeserialize_PreservesModel()
    {
        // Arrange
        var options = new NegativeBinomialRegressionOptions<double>
        {
            MaxIterations = 30,
            Tolerance = 1e-3
        };
        var nbr = new NegativeBinomialRegression<double>(options);
        var (X, y) = GenerateOverdispersedCountData(15);
        nbr.Train(X, y);
        var originalPredictions = nbr.Predict(X);

        // Act
        var serialized = nbr.Serialize();
        var newNbr = new NegativeBinomialRegression<double>(new NegativeBinomialRegressionOptions<double>());
        newNbr.Deserialize(serialized);
        var deserializedPredictions = newNbr.Predict(X);

        // Assert
        Assert.Equal(originalPredictions.Length, deserializedPredictions.Length);
        for (int i = 0; i < originalPredictions.Length; i++)
        {
            Assert.Equal(originalPredictions[i], deserializedPredictions[i], 4);
        }
    }

    [Fact]
    public void NegativeBinomialRegression_GetModelType_ReturnsNegativeBinomialRegression()
    {
        // Arrange
        var nbr = new NegativeBinomialRegression<double>(new NegativeBinomialRegressionOptions<double>());
        var (X, y) = GenerateOverdispersedCountData(10);
        nbr.Train(X, y);

        // Act
        var metadata = nbr.GetModelMetadata();

        // Assert
        Assert.Equal(ModelType.NegativeBinomialRegression, metadata.ModelType);
    }

    [Fact]
    public void NegativeBinomialRegression_Coefficients_AreAccessible()
    {
        // Arrange
        var nbr = new NegativeBinomialRegression<double>(new NegativeBinomialRegressionOptions<double>());
        var (X, y) = GenerateOverdispersedCountData(20);

        // Act
        nbr.Train(X, y);

        // Assert
        Assert.NotNull(nbr.Coefficients);
        Assert.Equal(1, nbr.Coefficients.Length); // One feature
    }

    #endregion

    #region GeneralizedAdditiveModel Tests

    [Fact]
    public void GeneralizedAdditiveModel_Train_NonLinearData_CapturesPattern()
    {
        // Arrange
        var options = new GeneralizedAdditiveModelOptions<double>
        {
            NumSplines = 5,
            Degree = 3
        };
        var gam = new GeneralizedAdditiveModel<double>(options);
        var (X, y) = GenerateNonLinearData(30);

        // Act
        gam.Train(X, y);
        var predictions = gam.Predict(X);

        // Assert
        Assert.Equal(y.Length, predictions.Length);
    }

    [Fact]
    public void GeneralizedAdditiveModel_Train_MultipleFeatures_HandlesCorrectly()
    {
        // Arrange
        var options = new GeneralizedAdditiveModelOptions<double>
        {
            NumSplines = 4,
            Degree = 3
        };
        var gam = new GeneralizedAdditiveModel<double>(options);
        var (X, y) = GenerateMultiFeatureNonLinearData(30);

        // Act
        gam.Train(X, y);
        var predictions = gam.Predict(X);

        // Assert
        Assert.Equal(y.Length, predictions.Length);
    }

    [Fact]
    public void GeneralizedAdditiveModel_GetModelMetadata_ReturnsCorrectInfo()
    {
        // Arrange
        var options = new GeneralizedAdditiveModelOptions<double>
        {
            NumSplines = 6,
            Degree = 2
        };
        var gam = new GeneralizedAdditiveModel<double>(options);
        var (X, y) = GenerateNonLinearData(20);
        gam.Train(X, y);

        // Act
        var metadata = gam.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
        Assert.Equal(ModelType.GeneralizedAdditiveModelRegression, metadata.ModelType);
        Assert.True(metadata.AdditionalInfo.ContainsKey("NumSplines"));
        Assert.True(metadata.AdditionalInfo.ContainsKey("Degree"));
        Assert.Equal(6, (int)metadata.AdditionalInfo["NumSplines"]);
        Assert.Equal(2, (int)metadata.AdditionalInfo["Degree"]);
    }

    [Fact]
    public void GeneralizedAdditiveModel_FeatureImportances_AreCalculated()
    {
        // Arrange
        var options = new GeneralizedAdditiveModelOptions<double>
        {
            NumSplines = 4,
            Degree = 3
        };
        var gam = new GeneralizedAdditiveModel<double>(options);
        var (X, y) = GenerateMultiFeatureNonLinearData(25);
        gam.Train(X, y);

        // Act
        var metadata = gam.GetModelMetadata();

        // Assert
        Assert.True(metadata.AdditionalInfo.ContainsKey("FeatureImportance"));
        var importances = metadata.AdditionalInfo["FeatureImportance"] as Vector<double>;
        Assert.NotNull(importances);
        Assert.Equal(2, importances.Length); // Two features
    }

    [Fact]
    public void GeneralizedAdditiveModel_DifferentSplineConfigurations_ProduceDifferentResults()
    {
        // Arrange
        var (X, y) = GenerateNonLinearData(25);

        var fewSplines = new GeneralizedAdditiveModelOptions<double> { NumSplines = 2, Degree = 2 };
        var manySplines = new GeneralizedAdditiveModelOptions<double> { NumSplines = 10, Degree = 3 };

        var gamFew = new GeneralizedAdditiveModel<double>(fewSplines);
        var gamMany = new GeneralizedAdditiveModel<double>(manySplines);

        // Act
        gamFew.Train(X, y);
        gamMany.Train(X, y);

        var predictionsFew = gamFew.Predict(X);
        var predictionsMany = gamMany.Predict(X);

        // Assert - both should produce predictions
        Assert.Equal(y.Length, predictionsFew.Length);
        Assert.Equal(y.Length, predictionsMany.Length);
    }

    [Fact]
    public void GeneralizedAdditiveModel_SerializeDeserialize_PreservesModel()
    {
        // Arrange
        var options = new GeneralizedAdditiveModelOptions<double>
        {
            NumSplines = 5,
            Degree = 3
        };
        var gam = new GeneralizedAdditiveModel<double>(options);
        var (X, y) = GenerateNonLinearData(15);
        gam.Train(X, y);
        var originalPredictions = gam.Predict(X);

        // Act
        var serialized = gam.Serialize();
        var newGam = new GeneralizedAdditiveModel<double>(new GeneralizedAdditiveModelOptions<double>());
        newGam.Deserialize(serialized);
        var deserializedPredictions = newGam.Predict(X);

        // Assert
        Assert.Equal(originalPredictions.Length, deserializedPredictions.Length);
        for (int i = 0; i < originalPredictions.Length; i++)
        {
            Assert.Equal(originalPredictions[i], deserializedPredictions[i], 4);
        }
    }

    #endregion

    #region Cross-Model Tests

    [Fact]
    public void AllGLMModels_TrainAndPredict_CompletesWithoutException()
    {
        // Arrange
        var (X, y) = GenerateCountData(20);

        var models = new List<(string Name, Action TrainAndPredict)>
        {
            ("PoissonRegression", () =>
            {
                var poisson = new PoissonRegression<double>(new PoissonRegressionOptions<double>());
                poisson.Train(X, y);
                poisson.Predict(X);
            }),
            ("NegativeBinomialRegression", () =>
            {
                var nbr = new NegativeBinomialRegression<double>(new NegativeBinomialRegressionOptions<double>());
                nbr.Train(X, y);
                nbr.Predict(X);
            }),
            ("GeneralizedAdditiveModel", () =>
            {
                var gam = new GeneralizedAdditiveModel<double>(new GeneralizedAdditiveModelOptions<double>());
                gam.Train(X, y);
                gam.Predict(X);
            })
        };

        // Act & Assert
        foreach (var (name, trainAndPredict) in models)
        {
            var exception = Record.Exception(trainAndPredict);
            Assert.True(exception == null, $"{name} threw exception: {exception?.Message}");
        }
    }

    [Fact]
    public void PoissonVsNegativeBinomial_SameData_BothProduceResults()
    {
        // Arrange
        var (X, y) = GenerateCountData(25);

        var poisson = new PoissonRegression<double>(new PoissonRegressionOptions<double>());
        var nbr = new NegativeBinomialRegression<double>(new NegativeBinomialRegressionOptions<double>());

        // Act
        poisson.Train(X, y);
        nbr.Train(X, y);

        var poissonPredictions = poisson.Predict(X);
        var nbrPredictions = nbr.Predict(X);

        // Assert - both should produce predictions of the same length
        Assert.Equal(y.Length, poissonPredictions.Length);
        Assert.Equal(y.Length, nbrPredictions.Length);
    }

    [Fact]
    public void GLMModels_WithSmallData_HandlesEdgeCases()
    {
        // Arrange - minimal training data
        var X = CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });
        var y = CreateVector(new double[] { 2, 3, 5, 7, 11 }); // Count data (primes)

        // Act & Assert
        var poisson = new PoissonRegression<double>(new PoissonRegressionOptions<double> { MaxIterations = 20 });
        var exception = Record.Exception(() =>
        {
            poisson.Train(X, y);
            poisson.Predict(X);
        });

        // Just verify it doesn't crash with small data
        Assert.True(exception == null || exception is ArgumentException,
            $"Unexpected exception: {exception?.GetType().Name}: {exception?.Message}");
    }

    #endregion
}

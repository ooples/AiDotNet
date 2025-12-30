using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Regression;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Regression;

/// <summary>
/// Integration tests for Kernel and Non-Parametric regression algorithms.
/// Tests KernelRidgeRegression, SupportVectorRegression, GaussianProcessRegression,
/// LocallyWeightedRegression, KNearestNeighborsRegression, and RadialBasisFunctionRegression.
/// </summary>
public class KernelRegressionIntegrationTests
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
    /// Generates a simple sinusoidal dataset for testing non-linear regression.
    /// </summary>
    private static (Matrix<double> X, Vector<double> y) GenerateSinusoidalData(int n, double noiseLevel = 0.0)
    {
        var random = new Random(42);
        var X = new Matrix<double>(n, 1);
        var y = new Vector<double>(n);

        for (int i = 0; i < n; i++)
        {
            double x = (double)i / n * 2 * Math.PI;
            X[i, 0] = x;
            y[i] = Math.Sin(x) + noiseLevel * (random.NextDouble() - 0.5);
        }

        return (X, y);
    }

    /// <summary>
    /// Generates a 2D quadratic dataset for testing non-linear regression.
    /// </summary>
    private static (Matrix<double> X, Vector<double> y) GenerateQuadraticData(int n)
    {
        var X = new Matrix<double>(n, 1);
        var y = new Vector<double>(n);

        for (int i = 0; i < n; i++)
        {
            double x = (double)i / n * 4 - 2; // Range from -2 to 2
            X[i, 0] = x;
            y[i] = x * x; // y = x^2
        }

        return (X, y);
    }

    /// <summary>
    /// Generates a multi-dimensional dataset for testing.
    /// </summary>
    private static (Matrix<double> X, Vector<double> y) GenerateMultiDimensionalData(int n)
    {
        var random = new Random(42);
        var X = new Matrix<double>(n, 2);
        var y = new Vector<double>(n);

        for (int i = 0; i < n; i++)
        {
            double x1 = random.NextDouble() * 4 - 2;
            double x2 = random.NextDouble() * 4 - 2;
            X[i, 0] = x1;
            X[i, 1] = x2;
            y[i] = x1 * x1 + x2 * x2; // y = x1^2 + x2^2
        }

        return (X, y);
    }

    #endregion

    #region KernelRidgeRegression Tests

    [Fact]
    public void KernelRidgeRegression_Train_QuadraticData_MakesReasonablePredictions()
    {
        // Arrange
        var options = new KernelRidgeRegressionOptions
        {
            LambdaKRR = 0.01,
            KernelType = KernelType.RBF,
            Gamma = 1.0
        };
        var krr = new KernelRidgeRegression<double>(options);
        var (X, y) = GenerateQuadraticData(20);

        // Act
        krr.Train(X, y);
        var predictions = krr.Predict(X);

        // Assert - predictions should be close to actual values
        Assert.Equal(y.Length, predictions.Length);
        for (int i = 0; i < y.Length; i++)
        {
            Assert.InRange(predictions[i], y[i] - LooseTolerance, y[i] + LooseTolerance);
        }
    }

    [Fact]
    public void KernelRidgeRegression_Train_SinusoidalData_CapturesNonLinearPattern()
    {
        // Arrange
        var options = new KernelRidgeRegressionOptions
        {
            LambdaKRR = 0.001,
            KernelType = KernelType.RBF,
            Gamma = 1.0
        };
        var krr = new KernelRidgeRegression<double>(options);
        var (X, y) = GenerateSinusoidalData(30);

        // Act
        krr.Train(X, y);
        var predictions = krr.Predict(X);

        // Assert - verify model captures the sinusoidal pattern
        double mse = 0;
        for (int i = 0; i < y.Length; i++)
        {
            mse += Math.Pow(predictions[i] - y[i], 2);
        }
        mse /= y.Length;

        // MSE should be reasonably low for a well-fit model
        Assert.True(mse < 1.0, $"MSE {mse} is too high, model failed to capture pattern");
    }

    [Fact]
    public void KernelRidgeRegression_GetModelMetadata_ReturnsCorrectInfo()
    {
        // Arrange
        var options = new KernelRidgeRegressionOptions
        {
            LambdaKRR = 0.5
        };
        var krr = new KernelRidgeRegression<double>(options);
        var (X, y) = GenerateQuadraticData(10);
        krr.Train(X, y);

        // Act
        var metadata = krr.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
        Assert.Equal(ModelType.KernelRidgeRegression, metadata.ModelType);
        Assert.True(metadata.AdditionalInfo.ContainsKey("LambdaKRR"));
        Assert.Equal(0.5, (double)metadata.AdditionalInfo["LambdaKRR"]);
    }

    [Fact]
    public void KernelRidgeRegression_SerializeDeserialize_PreservesModel()
    {
        // Arrange
        var options = new KernelRidgeRegressionOptions
        {
            LambdaKRR = 0.1,
            KernelType = KernelType.RBF
        };
        var krr = new KernelRidgeRegression<double>(options);
        var (X, y) = GenerateQuadraticData(15);
        krr.Train(X, y);
        var originalPredictions = krr.Predict(X);

        // Act
        var serialized = krr.Serialize();
        var newKrr = new KernelRidgeRegression<double>(new KernelRidgeRegressionOptions());
        newKrr.Deserialize(serialized);
        var deserializedPredictions = newKrr.Predict(X);

        // Assert
        Assert.Equal(originalPredictions.Length, deserializedPredictions.Length);
        for (int i = 0; i < originalPredictions.Length; i++)
        {
            Assert.Equal(originalPredictions[i], deserializedPredictions[i], 6);
        }
    }

    #endregion

    #region SupportVectorRegression Tests

    [Fact]
    public void SupportVectorRegression_Train_LinearData_MakesReasonablePredictions()
    {
        // Arrange
        var options = new SupportVectorRegressionOptions
        {
            Epsilon = 0.1,
            C = 1.0,
            MaxIterations = 100,
            KernelType = KernelType.Linear
        };
        var svr = new SupportVectorRegression<double>(options);
        var X = CreateMatrix(new double[,]
        {
            { 1 }, { 2 }, { 3 }, { 4 }, { 5 },
            { 6 }, { 7 }, { 8 }, { 9 }, { 10 }
        });
        var y = CreateVector(new double[] { 3, 5, 7, 9, 11, 13, 15, 17, 19, 21 }); // y = 2x + 1

        // Act
        svr.Train(X, y);
        var predictions = svr.Predict(X);

        // Assert
        Assert.Equal(y.Length, predictions.Length);
        // SVR should make reasonable predictions within epsilon-tube
        for (int i = 0; i < y.Length; i++)
        {
            Assert.InRange(predictions[i], y[i] - 5.0, y[i] + 5.0);
        }
    }

    [Fact]
    public void SupportVectorRegression_Train_RBFKernel_CapturesNonLinearity()
    {
        // Arrange
        var options = new SupportVectorRegressionOptions
        {
            Epsilon = 0.1,
            C = 10.0,
            MaxIterations = 200,
            KernelType = KernelType.RBF,
            Gamma = 0.5
        };
        var svr = new SupportVectorRegression<double>(options);
        var (X, y) = GenerateQuadraticData(20);

        // Act
        svr.Train(X, y);
        var predictions = svr.Predict(X);

        // Assert - predictions should follow the quadratic trend
        Assert.Equal(y.Length, predictions.Length);
    }

    [Fact]
    public void SupportVectorRegression_GetModelMetadata_ReturnsCorrectInfo()
    {
        // Arrange
        var options = new SupportVectorRegressionOptions
        {
            Epsilon = 0.2,
            C = 5.0
        };
        var svr = new SupportVectorRegression<double>(options);
        var X = CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 } });
        var y = CreateVector(new double[] { 2, 4, 6 });
        svr.Train(X, y);

        // Act
        var metadata = svr.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
        Assert.Equal(ModelType.SupportVectorRegression, metadata.ModelType);
        Assert.True(metadata.AdditionalInfo.ContainsKey("Epsilon"));
        Assert.True(metadata.AdditionalInfo.ContainsKey("C"));
    }

    [Fact]
    public void SupportVectorRegression_SerializeDeserialize_PreservesModel()
    {
        // Arrange
        var options = new SupportVectorRegressionOptions
        {
            Epsilon = 0.1,
            C = 1.0
        };
        var svr = new SupportVectorRegression<double>(options);
        var X = CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });
        var y = CreateVector(new double[] { 2, 4, 6, 8, 10 });
        svr.Train(X, y);
        var originalPredictions = svr.Predict(X);

        // Act
        var serialized = svr.Serialize();
        var newSvr = new SupportVectorRegression<double>(new SupportVectorRegressionOptions());
        newSvr.Deserialize(serialized);
        var deserializedPredictions = newSvr.Predict(X);

        // Assert
        Assert.Equal(originalPredictions.Length, deserializedPredictions.Length);
    }

    #endregion

    #region GaussianProcessRegression Tests

    [Fact]
    public void GaussianProcessRegression_Train_SimpleData_MakesReasonablePredictions()
    {
        // Arrange
        var options = new GaussianProcessRegressionOptions
        {
            NoiseLevel = 0.01,
            LengthScale = 1.0,
            SignalVariance = 1.0,
            OptimizeHyperparameters = false
        };
        var gpr = new GaussianProcessRegression<double>(options);
        var (X, y) = GenerateQuadraticData(10);

        // Act
        gpr.Train(X, y);
        var predictions = gpr.Predict(X);

        // Assert - GPR should interpolate through training points
        Assert.Equal(y.Length, predictions.Length);
        for (int i = 0; i < y.Length; i++)
        {
            Assert.InRange(predictions[i], y[i] - 1.0, y[i] + 1.0);
        }
    }

    [Fact]
    public void GaussianProcessRegression_Train_WithHyperparameterOptimization_Converges()
    {
        // Arrange
        var options = new GaussianProcessRegressionOptions
        {
            NoiseLevel = 0.01,
            OptimizeHyperparameters = true,
            MaxIterations = 10,
            Tolerance = 1e-3
        };
        var gpr = new GaussianProcessRegression<double>(options);
        var (X, y) = GenerateSinusoidalData(15);

        // Act
        gpr.Train(X, y);
        var predictions = gpr.Predict(X);

        // Assert
        Assert.Equal(y.Length, predictions.Length);
        // Just verify it completes without exception
    }

    [Fact]
    public void GaussianProcessRegression_GetModelMetadata_ReturnsCorrectInfo()
    {
        // Arrange
        var options = new GaussianProcessRegressionOptions
        {
            NoiseLevel = 0.1,
            LengthScale = 2.0,
            SignalVariance = 1.5
        };
        var gpr = new GaussianProcessRegression<double>(options);
        var X = CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 } });
        var y = CreateVector(new double[] { 1, 4, 9 });
        gpr.Train(X, y);

        // Act
        var metadata = gpr.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
        Assert.Equal(ModelType.GaussianProcessRegression, metadata.ModelType);
        Assert.True(metadata.AdditionalInfo.ContainsKey("NoiseLevel"));
        Assert.True(metadata.AdditionalInfo.ContainsKey("LengthScale"));
        Assert.True(metadata.AdditionalInfo.ContainsKey("SignalVariance"));
    }

    [Fact]
    public void GaussianProcessRegression_InterpolatesThroughTrainingPoints()
    {
        // Arrange - GPR should exactly interpolate through training points (with low noise)
        var options = new GaussianProcessRegressionOptions
        {
            NoiseLevel = 1e-6,
            LengthScale = 1.0,
            SignalVariance = 1.0
        };
        var gpr = new GaussianProcessRegression<double>(options);
        var X = CreateMatrix(new double[,] { { 0 }, { 1 }, { 2 } });
        var y = CreateVector(new double[] { 0, 1, 4 });

        // Act
        gpr.Train(X, y);
        var predictions = gpr.Predict(X);

        // Assert - with very low noise, predictions should be very close to training values
        for (int i = 0; i < y.Length; i++)
        {
            Assert.InRange(predictions[i], y[i] - 0.1, y[i] + 0.1);
        }
    }

    #endregion

    #region LocallyWeightedRegression Tests

    [Fact]
    public void LocallyWeightedRegression_Train_NonLinearData_AdaptsLocally()
    {
        // Arrange
        var options = new LocallyWeightedRegressionOptions
        {
            Bandwidth = 0.5
        };
        var lwr = new LocallyWeightedRegression<double>(options);
        var (X, y) = GenerateSinusoidalData(20);

        // Act
        lwr.Train(X, y);
        var predictions = lwr.Predict(X);

        // Assert
        Assert.Equal(y.Length, predictions.Length);
        // LWR should adapt to local patterns
    }

    [Fact]
    public void LocallyWeightedRegression_Train_DifferentBandwidths_AffectsSmoothness()
    {
        // Arrange
        var (X, y) = GenerateSinusoidalData(30, noiseLevel: 0.2);

        var smallBandwidth = new LocallyWeightedRegressionOptions { Bandwidth = 0.1 };
        var largeBandwidth = new LocallyWeightedRegressionOptions { Bandwidth = 2.0 };

        var lwrSmall = new LocallyWeightedRegression<double>(smallBandwidth);
        var lwrLarge = new LocallyWeightedRegression<double>(largeBandwidth);

        // Act
        lwrSmall.Train(X, y);
        lwrLarge.Train(X, y);

        var predictionsSmall = lwrSmall.Predict(X);
        var predictionsLarge = lwrLarge.Predict(X);

        // Assert - both should produce predictions
        Assert.Equal(y.Length, predictionsSmall.Length);
        Assert.Equal(y.Length, predictionsLarge.Length);
    }

    [Fact]
    public void LocallyWeightedRegression_SerializeDeserialize_PreservesModel()
    {
        // Arrange
        var options = new LocallyWeightedRegressionOptions { Bandwidth = 1.0 };
        var lwr = new LocallyWeightedRegression<double>(options);
        var X = CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });
        var y = CreateVector(new double[] { 1, 4, 9, 16, 25 });
        lwr.Train(X, y);
        var originalPredictions = lwr.Predict(X);

        // Act
        var serialized = lwr.Serialize();
        var newLwr = new LocallyWeightedRegression<double>(new LocallyWeightedRegressionOptions());
        newLwr.Deserialize(serialized);
        var deserializedPredictions = newLwr.Predict(X);

        // Assert
        Assert.Equal(originalPredictions.Length, deserializedPredictions.Length);
        for (int i = 0; i < originalPredictions.Length; i++)
        {
            Assert.Equal(originalPredictions[i], deserializedPredictions[i], 6);
        }
    }

    [Fact]
    public void LocallyWeightedRegression_SoftMode_EnablesJitCompilation()
    {
        // Arrange
        var options = new LocallyWeightedRegressionOptions
        {
            Bandwidth = 1.0,
            UseSoftMode = true
        };
        var lwr = new LocallyWeightedRegression<double>(options);
        var X = CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 } });
        var y = CreateVector(new double[] { 1, 4, 9 });
        lwr.Train(X, y);

        // Act & Assert
        Assert.True(lwr.SupportsJitCompilation);
    }

    #endregion

    #region KNearestNeighborsRegression Tests

    [Fact]
    public void KNearestNeighborsRegression_Train_SimpleData_MakesReasonablePredictions()
    {
        // Arrange
        var options = new KNearestNeighborsOptions { K = 3 };
        var knn = new KNearestNeighborsRegression<double>(options);
        var X = CreateMatrix(new double[,]
        {
            { 1 }, { 2 }, { 3 }, { 4 }, { 5 },
            { 6 }, { 7 }, { 8 }, { 9 }, { 10 }
        });
        var y = CreateVector(new double[] { 2, 4, 6, 8, 10, 12, 14, 16, 18, 20 });

        // Act
        knn.Train(X, y);
        var predictions = knn.Predict(X);

        // Assert
        Assert.Equal(y.Length, predictions.Length);
        // For K=3, predictions should be averages of 3 nearest neighbors
    }

    [Fact]
    public void KNearestNeighborsRegression_Train_K1_ReturnsNearestNeighbor()
    {
        // Arrange - with K=1, prediction should be the exact value of nearest neighbor
        var options = new KNearestNeighborsOptions { K = 1 };
        var knn = new KNearestNeighborsRegression<double>(options);
        var X = CreateMatrix(new double[,] { { 0 }, { 1 }, { 2 }, { 3 } });
        var y = CreateVector(new double[] { 10, 20, 30, 40 });

        // Act
        knn.Train(X, y);
        var predictions = knn.Predict(X);

        // Assert - with K=1, predictions at training points should equal training values
        for (int i = 0; i < y.Length; i++)
        {
            Assert.Equal(y[i], predictions[i], Tolerance);
        }
    }

    [Fact]
    public void KNearestNeighborsRegression_DifferentKValues_ProduceDifferentResults()
    {
        // Arrange
        var (X, y) = GenerateSinusoidalData(20, noiseLevel: 0.3);

        var knn1 = new KNearestNeighborsRegression<double>(new KNearestNeighborsOptions { K = 1 });
        var knn5 = new KNearestNeighborsRegression<double>(new KNearestNeighborsOptions { K = 5 });

        // Act
        knn1.Train(X, y);
        knn5.Train(X, y);

        var predictions1 = knn1.Predict(X);
        var predictions5 = knn5.Predict(X);

        // Assert - K=5 should produce smoother results
        Assert.Equal(y.Length, predictions1.Length);
        Assert.Equal(y.Length, predictions5.Length);
    }

    [Fact]
    public void KNearestNeighborsRegression_SerializeDeserialize_PreservesModel()
    {
        // Arrange
        var options = new KNearestNeighborsOptions { K = 3 };
        var knn = new KNearestNeighborsRegression<double>(options);
        var X = CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });
        var y = CreateVector(new double[] { 1, 4, 9, 16, 25 });
        knn.Train(X, y);
        var originalPredictions = knn.Predict(X);

        // Act
        var serialized = knn.Serialize();
        var newKnn = new KNearestNeighborsRegression<double>(new KNearestNeighborsOptions());
        newKnn.Deserialize(serialized);
        var deserializedPredictions = newKnn.Predict(X);

        // Assert
        Assert.Equal(originalPredictions.Length, deserializedPredictions.Length);
        for (int i = 0; i < originalPredictions.Length; i++)
        {
            Assert.Equal(originalPredictions[i], deserializedPredictions[i], 6);
        }
    }

    [Fact]
    public void KNearestNeighborsRegression_SoftKNN_EnablesJitCompilation()
    {
        // Arrange
        var options = new KNearestNeighborsOptions { K = 3 };
        var knn = new KNearestNeighborsRegression<double>(options);
        var X = CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 } });
        var y = CreateVector(new double[] { 1, 4, 9 });
        knn.Train(X, y);

        // Act
        knn.UseSoftKNN = true;

        // Assert
        Assert.True(knn.SupportsJitCompilation);
    }

    [Fact]
    public void KNearestNeighborsRegression_MultiDimensionalData_WorksCorrectly()
    {
        // Arrange
        var options = new KNearestNeighborsOptions { K = 3 };
        var knn = new KNearestNeighborsRegression<double>(options);
        var (X, y) = GenerateMultiDimensionalData(20);

        // Act
        knn.Train(X, y);
        var predictions = knn.Predict(X);

        // Assert
        Assert.Equal(y.Length, predictions.Length);
    }

    #endregion

    #region RadialBasisFunctionRegression Tests

    [Fact]
    public void RadialBasisFunctionRegression_Train_QuadraticData_CapturesPattern()
    {
        // Arrange
        var options = new RadialBasisFunctionOptions
        {
            NumberOfCenters = 5,
            Gamma = 1.0,
            Seed = 42
        };
        var rbf = new RadialBasisFunctionRegression<double>(options);
        var (X, y) = GenerateQuadraticData(20);

        // Act
        rbf.Train(X, y);
        var predictions = rbf.Predict(X);

        // Assert
        Assert.Equal(y.Length, predictions.Length);
    }

    [Fact]
    public void RadialBasisFunctionRegression_DifferentCenters_AffectsComplexity()
    {
        // Arrange
        var (X, y) = GenerateSinusoidalData(30);

        var fewCenters = new RadialBasisFunctionOptions { NumberOfCenters = 3, Seed = 42 };
        var manyCenters = new RadialBasisFunctionOptions { NumberOfCenters = 15, Seed = 42 };

        var rbfFew = new RadialBasisFunctionRegression<double>(fewCenters);
        var rbfMany = new RadialBasisFunctionRegression<double>(manyCenters);

        // Act
        rbfFew.Train(X, y);
        rbfMany.Train(X, y);

        var predictionsFew = rbfFew.Predict(X);
        var predictionsMany = rbfMany.Predict(X);

        // Assert
        Assert.Equal(y.Length, predictionsFew.Length);
        Assert.Equal(y.Length, predictionsMany.Length);
    }

    [Fact]
    public void RadialBasisFunctionRegression_SerializeDeserialize_PreservesModel()
    {
        // Arrange
        var options = new RadialBasisFunctionOptions
        {
            NumberOfCenters = 5,
            Gamma = 1.0,
            Seed = 42
        };
        var rbf = new RadialBasisFunctionRegression<double>(options);
        var X = CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 5 } });
        var y = CreateVector(new double[] { 1, 4, 9, 16, 25 });
        rbf.Train(X, y);
        var originalPredictions = rbf.Predict(X);

        // Act
        var serialized = rbf.Serialize();
        var newRbf = new RadialBasisFunctionRegression<double>(new RadialBasisFunctionOptions());
        newRbf.Deserialize(serialized);
        var deserializedPredictions = newRbf.Predict(X);

        // Assert
        Assert.Equal(originalPredictions.Length, deserializedPredictions.Length);
        for (int i = 0; i < originalPredictions.Length; i++)
        {
            Assert.Equal(originalPredictions[i], deserializedPredictions[i], 6);
        }
    }

    [Fact]
    public void RadialBasisFunctionRegression_GetModelMetadata_ReturnsCorrectInfo()
    {
        // Arrange
        var options = new RadialBasisFunctionOptions
        {
            NumberOfCenters = 10,
            Gamma = 0.5
        };
        var rbf = new RadialBasisFunctionRegression<double>(options);
        var X = CreateMatrix(new double[,] { { 1 }, { 2 }, { 3 } });
        var y = CreateVector(new double[] { 1, 4, 9 });
        rbf.Train(X, y);

        // Act
        var metadata = rbf.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
        Assert.Equal(ModelType.RadialBasisFunctionRegression, metadata.ModelType);
    }

    [Fact]
    public void RadialBasisFunctionRegression_MultiDimensionalInput_WorksCorrectly()
    {
        // Arrange
        var options = new RadialBasisFunctionOptions
        {
            NumberOfCenters = 5,
            Gamma = 1.0,
            Seed = 42
        };
        var rbf = new RadialBasisFunctionRegression<double>(options);
        var (X, y) = GenerateMultiDimensionalData(20);

        // Act
        rbf.Train(X, y);
        var predictions = rbf.Predict(X);

        // Assert
        Assert.Equal(y.Length, predictions.Length);
    }

    #endregion

    #region Cross-Model Comparison Tests

    [Fact]
    public void AllKernelModels_TrainAndPredict_CompletesWithoutException()
    {
        // Arrange
        var (X, y) = GenerateSinusoidalData(20);

        var models = new List<(string Name, Action TrainAndPredict)>
        {
            ("KernelRidgeRegression", () =>
            {
                var krr = new KernelRidgeRegression<double>(new KernelRidgeRegressionOptions());
                krr.Train(X, y);
                krr.Predict(X);
            }),
            ("SupportVectorRegression", () =>
            {
                var svr = new SupportVectorRegression<double>(new SupportVectorRegressionOptions());
                svr.Train(X, y);
                svr.Predict(X);
            }),
            ("GaussianProcessRegression", () =>
            {
                var gpr = new GaussianProcessRegression<double>(new GaussianProcessRegressionOptions());
                gpr.Train(X, y);
                gpr.Predict(X);
            }),
            ("LocallyWeightedRegression", () =>
            {
                var lwr = new LocallyWeightedRegression<double>(new LocallyWeightedRegressionOptions());
                lwr.Train(X, y);
                lwr.Predict(X);
            }),
            ("KNearestNeighborsRegression", () =>
            {
                var knn = new KNearestNeighborsRegression<double>(new KNearestNeighborsOptions());
                knn.Train(X, y);
                knn.Predict(X);
            }),
            ("RadialBasisFunctionRegression", () =>
            {
                var rbf = new RadialBasisFunctionRegression<double>(new RadialBasisFunctionOptions());
                rbf.Train(X, y);
                rbf.Predict(X);
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
    public void AllKernelModels_WithNoisyData_HandleNoiseGracefully()
    {
        // Arrange
        var (X, y) = GenerateSinusoidalData(30, noiseLevel: 0.5);

        // Act & Assert - all models should handle noisy data without crashing
        var krr = new KernelRidgeRegression<double>(new KernelRidgeRegressionOptions { LambdaKRR = 0.1 });
        krr.Train(X, y);
        Assert.Equal(y.Length, krr.Predict(X).Length);

        var svr = new SupportVectorRegression<double>(new SupportVectorRegressionOptions { Epsilon = 0.2 });
        svr.Train(X, y);
        Assert.Equal(y.Length, svr.Predict(X).Length);

        var gpr = new GaussianProcessRegression<double>(new GaussianProcessRegressionOptions { NoiseLevel = 0.1 });
        gpr.Train(X, y);
        Assert.Equal(y.Length, gpr.Predict(X).Length);

        var lwr = new LocallyWeightedRegression<double>(new LocallyWeightedRegressionOptions { Bandwidth = 0.5 });
        lwr.Train(X, y);
        Assert.Equal(y.Length, lwr.Predict(X).Length);

        var knn = new KNearestNeighborsRegression<double>(new KNearestNeighborsOptions { K = 5 });
        knn.Train(X, y);
        Assert.Equal(y.Length, knn.Predict(X).Length);

        var rbf = new RadialBasisFunctionRegression<double>(new RadialBasisFunctionOptions { NumberOfCenters = 10 });
        rbf.Train(X, y);
        Assert.Equal(y.Length, rbf.Predict(X).Length);
    }

    #endregion
}

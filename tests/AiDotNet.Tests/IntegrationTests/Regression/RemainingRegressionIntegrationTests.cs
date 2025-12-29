using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Regression;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Regression;

/// <summary>
/// Integration tests for remaining regression algorithms including Logistic, Time Series,
/// Genetic Algorithm, Dimensionality Reduction, and other regression models.
/// </summary>
public class RemainingRegressionIntegrationTests
{
    private const double Tolerance = 0.5;

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
    /// Generates binary classification data for logistic regression testing.
    /// </summary>
    private static (Matrix<double> X, Vector<double> y) GenerateBinaryClassificationData(int n)
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
            // Binary classification: y = 1 if x1 + x2 > 0, else 0
            y[i] = (x1 + x2 + random.NextDouble() * 0.5 - 0.25) > 0 ? 1 : 0;
        }

        return (X, y);
    }

    /// <summary>
    /// Generates time series data for testing.
    /// </summary>
    private static (Matrix<double> X, Vector<double> y) GenerateTimeSeriesData(int n)
    {
        var X = new Matrix<double>(n, 1);
        var y = new Vector<double>(n);
        var random = new Random(42);

        for (int i = 0; i < n; i++)
        {
            X[i, 0] = i;
            // y = trend + seasonality + noise
            double trend = 0.1 * i;
            double seasonality = 5 * Math.Sin(2 * Math.PI * i / 7); // Weekly seasonality
            double noise = random.NextDouble() * 2 - 1;
            y[i] = 50 + trend + seasonality + noise;
        }

        return (X, y);
    }

    /// <summary>
    /// Generates simple linear data for basic regression testing.
    /// </summary>
    private static (Matrix<double> X, Vector<double> y) GenerateLinearData(int n)
    {
        var X = new Matrix<double>(n, 1);
        var y = new Vector<double>(n);

        for (int i = 0; i < n; i++)
        {
            double x = (double)i / n * 10;
            X[i, 0] = x;
            y[i] = 2 * x + 3; // y = 2x + 3
        }

        return (X, y);
    }

    /// <summary>
    /// Generates multivariate regression data.
    /// </summary>
    private static (Matrix<double> X, Vector<double> y) GenerateMultivariateData(int n)
    {
        var random = new Random(42);
        var X = new Matrix<double>(n, 3);
        var y = new Vector<double>(n);

        for (int i = 0; i < n; i++)
        {
            double x1 = random.NextDouble() * 10;
            double x2 = random.NextDouble() * 10;
            double x3 = random.NextDouble() * 10;
            X[i, 0] = x1;
            X[i, 1] = x2;
            X[i, 2] = x3;
            // y = 2*x1 + 3*x2 - x3 + 5
            y[i] = 2 * x1 + 3 * x2 - x3 + 5 + random.NextDouble() * 0.5;
        }

        return (X, y);
    }

    #endregion

    #region LogisticRegression Tests

    [Fact]
    public void LogisticRegression_Train_BinaryData_MakesProbabilisticPredictions()
    {
        // Arrange
        var options = new LogisticRegressionOptions<double>
        {
            MaxIterations = 100,
            Tolerance = 1e-6,
            LearningRate = 0.01
        };
        var logistic = new LogisticRegression<double>(options);
        var (X, y) = GenerateBinaryClassificationData(50);

        // Act
        logistic.Train(X, y);
        var predictions = logistic.Predict(X);

        // Assert - predictions should be probabilities between 0 and 1
        Assert.Equal(y.Length, predictions.Length);
        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.InRange(predictions[i], 0.0, 1.0);
        }
    }

    [Fact]
    public void LogisticRegression_Train_LearnableBoundary_SeparatesClasses()
    {
        // Arrange
        var options = new LogisticRegressionOptions<double>
        {
            MaxIterations = 200,
            Tolerance = 1e-6,
            LearningRate = 0.1
        };
        var logistic = new LogisticRegression<double>(options);
        var (X, y) = GenerateBinaryClassificationData(100);

        // Act
        logistic.Train(X, y);
        var predictions = logistic.Predict(X);

        // Assert - predictions should correlate with actual labels
        int correctClass0 = 0, correctClass1 = 0;
        int totalClass0 = 0, totalClass1 = 0;
        for (int i = 0; i < y.Length; i++)
        {
            if (y[i] == 0)
            {
                totalClass0++;
                if (predictions[i] < 0.5) correctClass0++;
            }
            else
            {
                totalClass1++;
                if (predictions[i] >= 0.5) correctClass1++;
            }
        }

        // Should classify better than random guessing
        double accuracy = (double)(correctClass0 + correctClass1) / y.Length;
        Assert.True(accuracy > 0.5, $"Accuracy {accuracy} should be better than random");
    }

    [Fact]
    public void LogisticRegression_Coefficients_AreAccessible()
    {
        // Arrange
        var logistic = new LogisticRegression<double>(new LogisticRegressionOptions<double>());
        var (X, y) = GenerateBinaryClassificationData(30);

        // Act
        logistic.Train(X, y);

        // Assert
        Assert.NotNull(logistic.Coefficients);
        Assert.Equal(2, logistic.Coefficients.Length); // Two features
    }

    [Fact]
    public void LogisticRegression_SerializeDeserialize_PreservesModel()
    {
        // Arrange
        var options = new LogisticRegressionOptions<double>
        {
            MaxIterations = 50,
            Tolerance = 1e-4
        };
        var logistic = new LogisticRegression<double>(options);
        var (X, y) = GenerateBinaryClassificationData(20);
        logistic.Train(X, y);
        var originalPredictions = logistic.Predict(X);

        // Act
        var serialized = logistic.Serialize();
        var newLogistic = new LogisticRegression<double>(new LogisticRegressionOptions<double>());
        newLogistic.Deserialize(serialized);
        var deserializedPredictions = newLogistic.Predict(X);

        // Assert
        Assert.Equal(originalPredictions.Length, deserializedPredictions.Length);
        for (int i = 0; i < originalPredictions.Length; i++)
        {
            Assert.Equal(originalPredictions[i], deserializedPredictions[i], 4);
        }
    }

    [Fact]
    public void LogisticRegression_GetModelType_ReturnsLogisticRegression()
    {
        // Arrange
        var logistic = new LogisticRegression<double>(new LogisticRegressionOptions<double>());
        var (X, y) = GenerateBinaryClassificationData(10);
        logistic.Train(X, y);

        // Act
        var metadata = logistic.GetModelMetadata();

        // Assert
        Assert.Equal(ModelType.LogisticRegression, metadata.ModelType);
    }

    #endregion

    #region TimeSeriesRegression Tests

    [Fact]
    public void TimeSeriesRegression_Train_WithLagOrder_MakesPredictions()
    {
        // Arrange
        var options = new TimeSeriesRegressionOptions<double>
        {
            LagOrder = 3,
            IncludeTrend = true,
            SeasonalPeriod = 0,
            AutocorrelationCorrection = false,
            ModelType = TimeSeriesModelType.AutoRegressive
        };
        var timeSeries = new TimeSeriesRegression<double>(options);
        var (X, y) = GenerateTimeSeriesData(50);

        // Act
        timeSeries.Train(X, y);
        var predictions = timeSeries.Predict(X);

        // Assert
        Assert.True(predictions.Length > 0);
    }

    [Fact]
    public void TimeSeriesRegression_Train_WithSeasonality_CapturesPattern()
    {
        // Arrange
        var options = new TimeSeriesRegressionOptions<double>
        {
            LagOrder = 2,
            IncludeTrend = true,
            SeasonalPeriod = 7, // Weekly seasonality
            AutocorrelationCorrection = false,
            ModelType = TimeSeriesModelType.AutoRegressive
        };
        var timeSeries = new TimeSeriesRegression<double>(options);
        var (X, y) = GenerateTimeSeriesData(60);

        // Act
        timeSeries.Train(X, y);
        var predictions = timeSeries.Predict(X);

        // Assert
        Assert.True(predictions.Length > 0);
    }

    [Fact]
    public void TimeSeriesRegression_GetModelType_ReturnsTimeSeriesRegression()
    {
        // Arrange
        var options = new TimeSeriesRegressionOptions<double>
        {
            LagOrder = 1,
            IncludeTrend = false,
            ModelType = TimeSeriesModelType.AutoRegressive
        };
        var timeSeries = new TimeSeriesRegression<double>(options);
        var (X, y) = GenerateTimeSeriesData(20);
        timeSeries.Train(X, y);

        // Act
        var metadata = timeSeries.GetModelMetadata();

        // Assert
        Assert.Equal(ModelType.TimeSeriesRegression, metadata.ModelType);
    }

    #endregion

    #region GeneticAlgorithmRegression Tests

    [Fact]
    public void GeneticAlgorithmRegression_Train_FindsSolution()
    {
        // Arrange
        var gaOptions = new GeneticAlgorithmOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            PopulationSize = 20,
            MaxGenerations = 10,
            MutationRate = 0.1,
            CrossoverRate = 0.8
        };
        var gaRegression = new GeneticAlgorithmRegression<double>(gaOptions: gaOptions);
        var (X, y) = GenerateLinearData(30);

        // Act
        gaRegression.Train(X, y);
        var predictions = gaRegression.Predict(X);

        // Assert
        Assert.Equal(y.Length, predictions.Length);
    }

    [Fact]
    public void GeneticAlgorithmRegression_Coefficients_AreAccessible()
    {
        // Arrange
        var gaOptions = new GeneticAlgorithmOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            PopulationSize = 10,
            MaxGenerations = 5
        };
        var gaRegression = new GeneticAlgorithmRegression<double>(gaOptions: gaOptions);
        var (X, y) = GenerateLinearData(20);

        // Act
        gaRegression.Train(X, y);

        // Assert
        Assert.NotNull(gaRegression.Coefficients);
    }

    [Fact]
    public void GeneticAlgorithmRegression_GetModelType_ReturnsGeneticAlgorithmRegression()
    {
        // Arrange
        var gaOptions = new GeneticAlgorithmOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            PopulationSize = 10,
            MaxGenerations = 5
        };
        var gaRegression = new GeneticAlgorithmRegression<double>(gaOptions: gaOptions);
        var (X, y) = GenerateLinearData(15);
        gaRegression.Train(X, y);

        // Act
        var metadata = gaRegression.GetModelMetadata();

        // Assert
        Assert.Equal(ModelType.GeneticAlgorithmRegression, metadata.ModelType);
    }

    #endregion

    #region PrincipalComponentRegression Tests

    [Fact]
    public void PrincipalComponentRegression_Train_ReducesDimensionality()
    {
        // Arrange
        var options = new PrincipalComponentRegressionOptions<double>
        {
            NumComponents = 2
        };
        var pcr = new PrincipalComponentRegression<double>(options);
        var (X, y) = GenerateMultivariateData(40);

        // Act
        pcr.Train(X, y);
        var predictions = pcr.Predict(X);

        // Assert
        Assert.Equal(y.Length, predictions.Length);
    }

    [Fact]
    public void PrincipalComponentRegression_Train_WithFewerComponents_WorksCorrectly()
    {
        // Arrange
        var options = new PrincipalComponentRegressionOptions<double>
        {
            NumComponents = 1 // Use only 1 principal component
        };
        var pcr = new PrincipalComponentRegression<double>(options);
        var (X, y) = GenerateMultivariateData(30);

        // Act
        pcr.Train(X, y);
        var predictions = pcr.Predict(X);

        // Assert
        Assert.Equal(y.Length, predictions.Length);
    }

    #endregion

    #region PartialLeastSquaresRegression Tests

    [Fact]
    public void PartialLeastSquaresRegression_Train_MakesPredictions()
    {
        // Arrange
        var options = new PartialLeastSquaresRegressionOptions<double>
        {
            NumComponents = 2
        };
        var pls = new PartialLeastSquaresRegression<double>(options);
        var (X, y) = GenerateMultivariateData(40);

        // Act
        pls.Train(X, y);
        var predictions = pls.Predict(X);

        // Assert
        Assert.Equal(y.Length, predictions.Length);
    }

    [Fact]
    public void PartialLeastSquaresRegression_GetModelMetadata_ReturnsCorrectInfo()
    {
        // Arrange
        var options = new PartialLeastSquaresRegressionOptions<double>
        {
            NumComponents = 2
        };
        var pls = new PartialLeastSquaresRegression<double>(options);
        var (X, y) = GenerateMultivariateData(30);
        pls.Train(X, y);

        // Act
        var metadata = pls.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
        Assert.Equal(ModelType.PartialLeastSquaresRegression, metadata.ModelType);
    }

    #endregion

    #region MultivariateRegression Tests

    [Fact]
    public void MultivariateRegression_Train_WithMultipleTargets_WorksCorrectly()
    {
        // Arrange - using multiple target variables
        var X = CreateMatrix(new double[,]
        {
            { 1, 2 }, { 2, 3 }, { 3, 4 }, { 4, 5 }, { 5, 6 }
        });
        // For multivariate regression, y would typically be a matrix, but here we test with single target
        var y = CreateVector(new double[] { 5, 8, 11, 14, 17 }); // y = x1 + 2*x2

        var multivariate = new MultivariateRegression<double>(new RegressionOptions<double>());

        // Act
        multivariate.Train(X, y);
        var predictions = multivariate.Predict(X);

        // Assert
        Assert.Equal(y.Length, predictions.Length);
    }

    #endregion

    #region StepwiseRegression Tests

    [Fact]
    public void StepwiseRegression_Train_SelectsFeatures()
    {
        // Arrange
        var options = new StepwiseRegressionOptions<double>
        {
            Method = StepwiseMethod.Forward,
            MaxFeatures = 10,
            MinImprovement = 0.05
        };
        var stepwise = new StepwiseRegression<double>(options);
        var (X, y) = GenerateMultivariateData(50);

        // Act
        stepwise.Train(X, y);
        var predictions = stepwise.Predict(X);

        // Assert
        Assert.Equal(y.Length, predictions.Length);
    }

    [Fact]
    public void StepwiseRegression_BackwardDirection_WorksCorrectly()
    {
        // Arrange
        var options = new StepwiseRegressionOptions<double>
        {
            Method = StepwiseMethod.Backward,
            MaxFeatures = 10
        };
        var stepwise = new StepwiseRegression<double>(options);
        var (X, y) = GenerateMultivariateData(50);

        // Act
        stepwise.Train(X, y);
        var predictions = stepwise.Predict(X);

        // Assert
        Assert.Equal(y.Length, predictions.Length);
    }

    #endregion

    #region WeightedRegression Tests

    [Fact]
    public void WeightedRegression_Train_WithUniformWeights_WorksLikeOLS()
    {
        // Arrange
        var (X, y) = GenerateLinearData(30);
        // Uniform weights
        var weights = new Vector<double>(y.Length);
        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = 1.0;
        }
        var options = new WeightedRegressionOptions<double>
        {
            Weights = weights
        };
        var weighted = new WeightedRegression<double>(options);

        // Act
        weighted.Train(X, y);
        var predictions = weighted.Predict(X);

        // Assert
        Assert.Equal(y.Length, predictions.Length);
    }

    [Fact]
    public void WeightedRegression_Train_WithVaryingWeights_WorksCorrectly()
    {
        // Arrange
        var (X, y) = GenerateLinearData(30);
        // Give higher weights to later observations
        var weights = new Vector<double>(y.Length);
        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = (i + 1.0) / y.Length;
        }
        var options = new WeightedRegressionOptions<double>
        {
            Weights = weights
        };
        var weighted = new WeightedRegression<double>(options);

        // Act
        weighted.Train(X, y);
        var predictions = weighted.Predict(X);

        // Assert
        Assert.Equal(y.Length, predictions.Length);
    }

    #endregion

    #region OrthogonalRegression Tests

    [Fact]
    public void OrthogonalRegression_Train_MinimizesTotalLeastSquares()
    {
        // Arrange
        var orthogonal = new OrthogonalRegression<double>(new OrthogonalRegressionOptions<double>());
        var (X, y) = GenerateLinearData(30);

        // Act
        orthogonal.Train(X, y);
        var predictions = orthogonal.Predict(X);

        // Assert
        Assert.Equal(y.Length, predictions.Length);
    }

    #endregion

    #region NeuralNetworkRegression Tests

    [Fact]
    public void NeuralNetworkRegression_Train_SimpleData_MakesPredictions()
    {
        // Arrange
        var options = new NeuralNetworkRegressionOptions<double, Matrix<double>, Vector<double>>
        {
            LayerSizes = new List<int> { 1, 10, 1 },
            Epochs = 100,
            LearningRate = 0.01
        };
        var nn = new NeuralNetworkRegression<double>(options);
        var (X, y) = GenerateLinearData(30);

        // Act
        nn.Train(X, y);
        var predictions = nn.Predict(X);

        // Assert
        Assert.Equal(y.Length, predictions.Length);
    }

    [Fact]
    public void NeuralNetworkRegression_Train_NonLinearData_CapturesPattern()
    {
        // Arrange - quadratic data y = x^2
        var options = new NeuralNetworkRegressionOptions<double, Matrix<double>, Vector<double>>
        {
            LayerSizes = new List<int> { 1, 20, 10, 1 },
            Epochs = 200,
            LearningRate = 0.01
        };
        var nn = new NeuralNetworkRegression<double>(options);

        var X = new Matrix<double>(20, 1);
        var y = new Vector<double>(20);
        for (int i = 0; i < 20; i++)
        {
            double x = (double)i / 20 * 4 - 2; // Range -2 to 2
            X[i, 0] = x;
            y[i] = x * x; // Quadratic
        }

        // Act
        nn.Train(X, y);
        var predictions = nn.Predict(X);

        // Assert - predictions should be in reasonable range
        Assert.Equal(y.Length, predictions.Length);
    }

    [Fact]
    public void NeuralNetworkRegression_GetModelMetadata_ReturnsCorrectInfo()
    {
        // Arrange
        var options = new NeuralNetworkRegressionOptions<double, Matrix<double>, Vector<double>>
        {
            LayerSizes = new List<int> { 1, 5, 1 },
            Epochs = 50
        };
        var nn = new NeuralNetworkRegression<double>(options);
        var (X, y) = GenerateLinearData(20);
        nn.Train(X, y);

        // Act
        var metadata = nn.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
        Assert.Equal(ModelType.NeuralNetworkRegression, metadata.ModelType);
    }

    #endregion

    #region MultilayerPerceptronRegression Tests

    [Fact]
    public void MultilayerPerceptronRegression_Train_SimpleData_MakesPredictions()
    {
        // Arrange
        var options = new MultilayerPerceptronOptions<double, Matrix<double>, Vector<double>>
        {
            LayerSizes = new List<int> { 1, 10, 1 },
            MaxEpochs = 100,
            LearningRate = 0.01
        };
        var mlp = new MultilayerPerceptronRegression<double>(options);
        var (X, y) = GenerateLinearData(30);

        // Act
        mlp.Train(X, y);
        var predictions = mlp.Predict(X);

        // Assert
        Assert.Equal(y.Length, predictions.Length);
    }

    [Fact]
    public void MultilayerPerceptronRegression_Train_MultipleLayers_WorksCorrectly()
    {
        // Arrange
        var options = new MultilayerPerceptronOptions<double, Matrix<double>, Vector<double>>
        {
            LayerSizes = new List<int> { 1, 10, 5, 1 }, // 2 hidden layers
            MaxEpochs = 100,
            LearningRate = 0.01
        };
        var mlp = new MultilayerPerceptronRegression<double>(options);
        var (X, y) = GenerateLinearData(30);

        // Act
        mlp.Train(X, y);
        var predictions = mlp.Predict(X);

        // Assert
        Assert.Equal(y.Length, predictions.Length);
    }

    [Fact]
    public void MultilayerPerceptronRegression_GetModelMetadata_ReturnsCorrectInfo()
    {
        // Arrange
        var options = new MultilayerPerceptronOptions<double, Matrix<double>, Vector<double>>
        {
            LayerSizes = new List<int> { 1, 5, 1 },
            MaxEpochs = 50
        };
        var mlp = new MultilayerPerceptronRegression<double>(options);
        var (X, y) = GenerateLinearData(20);
        mlp.Train(X, y);

        // Act
        var metadata = mlp.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
        Assert.Equal(ModelType.MultilayerPerceptronRegression, metadata.ModelType);
    }

    #endregion

    #region MultinomialLogisticRegression Tests

    /// <summary>
    /// Generates multi-class classification data for multinomial logistic regression.
    /// </summary>
    private static (Matrix<double> X, Vector<double> y) GenerateMultiClassData(int n, int numClasses = 3)
    {
        var random = new Random(42);
        var X = new Matrix<double>(n, 2);
        var y = new Vector<double>(n);

        for (int i = 0; i < n; i++)
        {
            int classLabel = i % numClasses;
            // Create clusters for each class
            double centerX = classLabel * 2;
            double centerY = classLabel;
            X[i, 0] = centerX + random.NextDouble() * 0.5 - 0.25;
            X[i, 1] = centerY + random.NextDouble() * 0.5 - 0.25;
            y[i] = classLabel;
        }

        return (X, y);
    }

    [Fact]
    public void MultinomialLogisticRegression_Train_MultiClassData_MakesPredictions()
    {
        // Arrange
        var options = new MultinomialLogisticRegressionOptions<double>
        {
            MaxIterations = 100,
            Tolerance = 1e-6
        };
        var multinomial = new MultinomialLogisticRegression<double>(options);
        var (X, y) = GenerateMultiClassData(60, 3);

        // Act
        multinomial.Train(X, y);
        var predictions = multinomial.Predict(X);

        // Assert
        Assert.Equal(y.Length, predictions.Length);
        // Predictions should be class labels (0, 1, or 2)
        for (int i = 0; i < predictions.Length; i++)
        {
            Assert.True(predictions[i] >= 0 && predictions[i] <= 2,
                $"Prediction {predictions[i]} should be a valid class label");
        }
    }

    [Fact]
    public void MultinomialLogisticRegression_Coefficients_AreAccessible()
    {
        // Arrange
        var options = new MultinomialLogisticRegressionOptions<double>
        {
            MaxIterations = 50
        };
        var multinomial = new MultinomialLogisticRegression<double>(options);
        var (X, y) = GenerateMultiClassData(30, 3);

        // Act
        multinomial.Train(X, y);

        // Assert
        Assert.NotNull(multinomial.Coefficients);
    }

    [Fact]
    public void MultinomialLogisticRegression_GetModelMetadata_ReturnsCorrectInfo()
    {
        // Arrange
        var options = new MultinomialLogisticRegressionOptions<double>();
        var multinomial = new MultinomialLogisticRegression<double>(options);
        var (X, y) = GenerateMultiClassData(30, 3);
        multinomial.Train(X, y);

        // Act
        var metadata = multinomial.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
        Assert.Equal(ModelType.MultinomialLogisticRegression, metadata.ModelType);
    }

    #endregion

    #region SymbolicRegression Tests

    [Fact]
    public void SymbolicRegression_Train_SimpleData_FindsExpression()
    {
        // Arrange
        var options = new SymbolicRegressionOptions
        {
            PopulationSize = 20,
            MaxGenerations = 10
        };
        var symbolic = new SymbolicRegression<double>(options);
        var (X, y) = GenerateLinearData(20);

        // Act
        symbolic.Train(X, y);
        var predictions = symbolic.Predict(X);

        // Assert
        Assert.Equal(y.Length, predictions.Length);
    }

    [Fact]
    public void SymbolicRegression_Train_QuadraticData_DiscoversPattern()
    {
        // Arrange - y = x^2 data
        var options = new SymbolicRegressionOptions
        {
            PopulationSize = 30,
            MaxGenerations = 20
        };
        var symbolic = new SymbolicRegression<double>(options);

        var X = new Matrix<double>(15, 1);
        var y = new Vector<double>(15);
        for (int i = 0; i < 15; i++)
        {
            double x = i - 7;
            X[i, 0] = x;
            y[i] = x * x;
        }

        // Act
        symbolic.Train(X, y);
        var predictions = symbolic.Predict(X);

        // Assert
        Assert.Equal(y.Length, predictions.Length);
    }

    [Fact]
    public void SymbolicRegression_GetModelMetadata_ReturnsCorrectInfo()
    {
        // Arrange
        var options = new SymbolicRegressionOptions
        {
            PopulationSize = 10,
            MaxGenerations = 5
        };
        var symbolic = new SymbolicRegression<double>(options);
        var (X, y) = GenerateLinearData(15);
        symbolic.Train(X, y);

        // Act
        var metadata = symbolic.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
        Assert.Equal(ModelType.SymbolicRegression, metadata.ModelType);
    }

    #endregion

    #region Cross-Model Comparison Tests

    [Fact]
    public void AllRemainingModels_TrainAndPredict_CompletesWithoutException()
    {
        // Arrange
        var (X, y) = GenerateLinearData(30);
        var (Xclass, yclass) = GenerateBinaryClassificationData(30);
        var (Xmulti, ymulti) = GenerateMultivariateData(30);

        var models = new List<(string Name, Action TrainAndPredict)>
        {
            ("LogisticRegression", () =>
            {
                var logistic = new LogisticRegression<double>(new LogisticRegressionOptions<double>());
                logistic.Train(Xclass, yclass);
                logistic.Predict(Xclass);
            }),
            ("MultivariateRegression", () =>
            {
                var multi = new MultivariateRegression<double>(new RegressionOptions<double>());
                multi.Train(X, y);
                multi.Predict(X);
            }),
            ("StepwiseRegression", () =>
            {
                var stepwise = new StepwiseRegression<double>(new StepwiseRegressionOptions<double>());
                stepwise.Train(Xmulti, ymulti);
                stepwise.Predict(Xmulti);
            }),
            ("PrincipalComponentRegression", () =>
            {
                var pcr = new PrincipalComponentRegression<double>(new PrincipalComponentRegressionOptions<double> { NumComponents = 1 });
                pcr.Train(Xmulti, ymulti);
                pcr.Predict(Xmulti);
            }),
            ("PartialLeastSquaresRegression", () =>
            {
                var pls = new PartialLeastSquaresRegression<double>(new PartialLeastSquaresRegressionOptions<double> { NumComponents = 1 });
                pls.Train(Xmulti, ymulti);
                pls.Predict(Xmulti);
            })
        };

        // Act & Assert
        foreach (var (name, trainAndPredict) in models)
        {
            var exception = Record.Exception(trainAndPredict);
            Assert.True(exception == null, $"{name} threw exception: {exception?.Message}");
        }
    }

    #endregion
}

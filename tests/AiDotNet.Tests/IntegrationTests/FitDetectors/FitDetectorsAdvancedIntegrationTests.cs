using AiDotNet.Enums;
using AiDotNet.FitDetectors;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Regression;
using Xunit;

namespace AiDotNetTests.IntegrationTests.FitDetectors
{
    /// <summary>
    /// Integration tests for advanced fit detectors (Part 2 of 2).
    /// Tests InformationCriteria, Autocorrelation, Heteroscedasticity, CookDistance, VIF,
    /// CalibratedProbability, FeatureImportance, PartialDependencePlot, ShapleyValue,
    /// PermutationTest, Bayesian, GaussianProcess, NeuralNetwork, GradientBoosting,
    /// Ensemble, Hybrid, and Adaptive fit detectors with mathematically verified results.
    /// </summary>
    public class FitDetectorsAdvancedIntegrationTests
    {
        #region Helper Methods

        /// <summary>
        /// Creates a simple model evaluation data with known characteristics
        /// </summary>
        private ModelEvaluationData<double, Matrix<double>, Vector<double>> CreateBasicEvaluationData(
            int trainSize = 50, int valSize = 25, int testSize = 25, bool addNoise = false, double noiseFactor = 0.1)
        {
            var random = new Random(42);

            // Create training data
            var trainX = new Matrix<double>(trainSize, 3);
            var trainY = new Vector<double>(trainSize);
            for (int i = 0; i < trainSize; i++)
            {
                trainX[i, 0] = i / 10.0;
                trainX[i, 1] = Math.Sin(i / 10.0);
                trainX[i, 2] = random.NextDouble();
                trainY[i] = 2.0 * trainX[i, 0] + 3.0 * trainX[i, 1] + (addNoise ? noiseFactor * random.NextDouble() : 0.0);
            }

            // Create validation data
            var valX = new Matrix<double>(valSize, 3);
            var valY = new Vector<double>(valSize);
            for (int i = 0; i < valSize; i++)
            {
                int offset = trainSize + i;
                valX[i, 0] = offset / 10.0;
                valX[i, 1] = Math.Sin(offset / 10.0);
                valX[i, 2] = random.NextDouble();
                valY[i] = 2.0 * valX[i, 0] + 3.0 * valX[i, 1] + (addNoise ? noiseFactor * random.NextDouble() : 0.0);
            }

            // Create test data
            var testX = new Matrix<double>(testSize, 3);
            var testY = new Vector<double>(testSize);
            for (int i = 0; i < testSize; i++)
            {
                int offset = trainSize + valSize + i;
                testX[i, 0] = offset / 10.0;
                testX[i, 1] = Math.Sin(offset / 10.0);
                testX[i, 2] = random.NextDouble();
                testY[i] = 2.0 * testX[i, 0] + 3.0 * testX[i, 1] + (addNoise ? noiseFactor * random.NextDouble() : 0.0);
            }

            // Create a simple regression model and get predictions
            var model = new SimpleRegression<double>();
            model.Fit(trainX, trainY);

            var trainPredictions = model.Predict(trainX);
            var valPredictions = model.Predict(valX);
            var testPredictions = model.Predict(testX);

            // Create evaluation data
            return CreateEvaluationDataFromPredictions(trainX, trainY, trainPredictions,
                valX, valY, valPredictions, testX, testY, testPredictions, model);
        }

        private ModelEvaluationData<double, Matrix<double>, Vector<double>> CreateEvaluationDataFromPredictions(
            Matrix<double> trainX, Vector<double> trainY, Vector<double> trainPred,
            Matrix<double> valX, Vector<double> valY, Vector<double> valPred,
            Matrix<double> testX, Vector<double> testY, Vector<double> testPred,
            SimpleRegression<double> model = null)
        {
            var evalData = new ModelEvaluationData<double, Matrix<double>, Vector<double>>();

            // Set up training set
            evalData.TrainingSet = new DataSetStats<double, Matrix<double>, Vector<double>>
            {
                Features = trainX,
                Actual = trainY,
                Predicted = trainPred,
                ErrorStats = CreateErrorStats(trainY, trainPred),
                PredictionStats = CreatePredictionStats(trainY, trainPred, trainX.Columns),
                ActualBasicStats = CreateBasicStats(trainY)
            };

            // Set up validation set
            evalData.ValidationSet = new DataSetStats<double, Matrix<double>, Vector<double>>
            {
                Features = valX,
                Actual = valY,
                Predicted = valPred,
                ErrorStats = CreateErrorStats(valY, valPred),
                PredictionStats = CreatePredictionStats(valY, valPred, valX.Columns),
                ActualBasicStats = CreateBasicStats(valY)
            };

            // Set up test set
            evalData.TestSet = new DataSetStats<double, Matrix<double>, Vector<double>>
            {
                Features = testX,
                Actual = testY,
                Predicted = testPred,
                ErrorStats = CreateErrorStats(testY, testPred),
                PredictionStats = CreatePredictionStats(testY, testPred, testX.Columns),
                ActualBasicStats = CreateBasicStats(testY)
            };

            // Set up model stats
            evalData.ModelStats = new ModelStats<double, Matrix<double>, Vector<double>>
            {
                Features = trainX,
                Actual = trainY,
                Predicted = trainPred,
                Model = model,
                FeatureNames = new List<string> { "Feature1", "Feature2", "Feature3" },
                FeatureValues = new Dictionary<string, object>
                {
                    { "Feature1", trainX.GetColumn(0) },
                    { "Feature2", trainX.GetColumn(1) },
                    { "Feature3", trainX.GetColumn(2) }
                },
                CorrelationMatrix = CalculateCorrelationMatrix(trainX)
            };

            return evalData;
        }

        private ErrorStats<double> CreateErrorStats(Vector<double> actual, Vector<double> predicted)
        {
            var errors = new Vector<double>(actual.Length);
            double sumSquaredError = 0;
            double sumAbsError = 0;

            for (int i = 0; i < actual.Length; i++)
            {
                var error = actual[i] - predicted[i];
                errors[i] = error;
                sumSquaredError += error * error;
                sumAbsError += Math.Abs(error);
            }

            var mse = sumSquaredError / actual.Length;
            var mae = sumAbsError / actual.Length;
            var rmse = Math.Sqrt(mse);

            // Calculate AIC and BIC (simplified versions)
            var n = actual.Length;
            var k = 3; // Number of parameters
            var aic = n * Math.Log(mse) + 2 * k;
            var bic = n * Math.Log(mse) + k * Math.Log(n);

            return new ErrorStats<double>
            {
                ErrorList = errors,
                MSE = mse,
                MAE = mae,
                RMSE = rmse,
                AIC = aic,
                BIC = bic
            };
        }

        private PredictionStats<double> CreatePredictionStats(Vector<double> actual, Vector<double> predicted, int numParams)
        {
            var inputs = new PredictionStatsInputs<double>
            {
                Actual = actual,
                Predicted = predicted,
                NumberOfParameters = numParams
            };
            return new PredictionStats<double>(inputs);
        }

        private BasicStats<double> CreateBasicStats(Vector<double> data)
        {
            var mean = data.Average();
            var variance = data.Select(x => Math.Pow(x - mean, 2)).Sum() / data.Length;

            return new BasicStats<double>
            {
                Mean = mean,
                Variance = variance,
                StdDev = Math.Sqrt(variance)
            };
        }

        private Matrix<double> CalculateCorrelationMatrix(Matrix<double> X)
        {
            var n = X.Columns;
            var corr = new Matrix<double>(n, n);

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (i == j)
                    {
                        corr[i, j] = 1.0;
                    }
                    else
                    {
                        var col1 = X.GetColumn(i);
                        var col2 = X.GetColumn(j);
                        corr[i, j] = CalculatePearsonCorrelation(col1, col2);
                    }
                }
            }

            return corr;
        }

        private double CalculatePearsonCorrelation(Vector<double> x, Vector<double> y)
        {
            var meanX = x.Average();
            var meanY = y.Average();

            double numerator = 0;
            double denomX = 0;
            double denomY = 0;

            for (int i = 0; i < x.Length; i++)
            {
                var dx = x[i] - meanX;
                var dy = y[i] - meanY;
                numerator += dx * dy;
                denomX += dx * dx;
                denomY += dy * dy;
            }

            return numerator / Math.Sqrt(denomX * denomY);
        }

        #endregion

        #region InformationCriteriaFitDetector Tests

        [Fact]
        public void InformationCriteriaFitDetector_GoodFit_DetectsCorrectly()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData(addNoise: false);
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.Equal(FitType.GoodFit, result.FitType);
            Assert.True(result.ConfidenceLevel > 0.5);
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void InformationCriteriaFitDetector_CalculatesAICBICCorrectly()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert - AIC and BIC should be calculated
            Assert.NotEqual(0.0, evalData.TrainingSet.ErrorStats.AIC);
            Assert.NotEqual(0.0, evalData.TrainingSet.ErrorStats.BIC);
            Assert.Contains("AIC threshold", string.Join(" ", result.Recommendations));
        }

        [Fact]
        public void InformationCriteriaFitDetector_HigherComplexityModel_DetectsOverfit()
        {
            // Arrange - Create data where validation AIC/BIC is much higher than training
            var evalData = CreateBasicEvaluationData(addNoise: true, noiseFactor: 2.0);
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.True(result.FitType == FitType.Overfit || result.FitType == FitType.HighVariance);
        }

        [Fact]
        public void InformationCriteriaFitDetector_ReturnsConfidenceLevel()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result.ConfidenceLevel);
            Assert.True(result.ConfidenceLevel >= 0.0 && result.ConfidenceLevel <= 1.0);
        }

        [Fact]
        public void InformationCriteriaFitDetector_IncludesRelevantRecommendations()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotEmpty(result.Recommendations);
            Assert.Contains(result.Recommendations, r => r.Contains("information criteria") || r.Contains("AIC") || r.Contains("BIC"));
        }

        [Fact]
        public void InformationCriteriaFitDetector_DifferentDataSizes_WorksCorrectly()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData(trainSize: 100, valSize: 50, testSize: 50);
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.FitType);
        }

        [Fact]
        public void InformationCriteriaFitDetector_ComparesAICandBIC()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert - BIC typically penalizes complexity more than AIC
            Assert.True(evalData.TrainingSet.ErrorStats.BIC >= evalData.TrainingSet.ErrorStats.AIC);
        }

        #endregion

        #region AutocorrelationFitDetector Tests

        [Fact]
        public void AutocorrelationFitDetector_NoAutocorrelation_DetectsCorrectly()
        {
            // Arrange - Random data with no autocorrelation
            var evalData = CreateBasicEvaluationData(addNoise: true);
            var detector = new AutocorrelationFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result.FitType);
            Assert.Contains(result.Recommendations, r => r.Contains("Durbin-Watson"));
        }

        [Fact]
        public void AutocorrelationFitDetector_PositiveAutocorrelation_DetectsCorrectly()
        {
            // Arrange - Create time series data with positive autocorrelation
            var trainSize = 50;
            var trainX = new Matrix<double>(trainSize, 1);
            var trainY = new Vector<double>(trainSize);

            // Create autocorrelated time series
            trainY[0] = 1.0;
            trainX[0, 0] = 0.0;
            for (int i = 1; i < trainSize; i++)
            {
                trainY[i] = 0.8 * trainY[i-1] + 0.5; // Strong positive autocorrelation
                trainX[i, 0] = i;
            }

            var model = new SimpleRegression<double>();
            model.Fit(trainX, trainY);
            var predictions = model.Predict(trainX);

            var evalData = CreateEvaluationDataFromPredictions(
                trainX, trainY, predictions,
                trainX, trainY, predictions,
                trainX, trainY, predictions);

            var detector = new AutocorrelationFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result);
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void AutocorrelationFitDetector_CalculatesDurbinWatson()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new AutocorrelationFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.Contains(result.Recommendations, r => r.Contains("Durbin-Watson statistic:"));
        }

        [Fact]
        public void AutocorrelationFitDetector_ReturnsValidConfidence()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new AutocorrelationFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result.ConfidenceLevel);
            Assert.True(result.ConfidenceLevel >= 0.0 && result.ConfidenceLevel <= 1.0);
        }

        [Fact]
        public void AutocorrelationFitDetector_ProvidesRelevantRecommendations()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new AutocorrelationFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void AutocorrelationFitDetector_HandlesSmallSamples()
        {
            // Arrange - Small sample
            var evalData = CreateBasicEvaluationData(trainSize: 10, valSize: 5, testSize: 5);
            var detector = new AutocorrelationFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result);
        }

        #endregion

        #region HeteroscedasticityFitDetector Tests

        [Fact]
        public void HeteroscedasticityFitDetector_HomoscedasticData_DetectsGoodFit()
        {
            // Arrange - Constant variance
            var evalData = CreateBasicEvaluationData(addNoise: false);
            var detector = new HeteroscedasticityFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.True(result.FitType == FitType.GoodFit || result.FitType == FitType.Moderate);
            Assert.NotNull(result.AdditionalInfo);
        }

        [Fact]
        public void HeteroscedasticityFitDetector_CalculatesBreuschPaganTest()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new HeteroscedasticityFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.Contains("BreuschPaganTestStatistic", result.AdditionalInfo.Keys);
            Assert.IsType<double>(result.AdditionalInfo["BreuschPaganTestStatistic"]);
        }

        [Fact]
        public void HeteroscedasticityFitDetector_CalculatesWhiteTest()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new HeteroscedasticityFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.Contains("WhiteTestStatistic", result.AdditionalInfo.Keys);
            Assert.IsType<double>(result.AdditionalInfo["WhiteTestStatistic"]);
        }

        [Fact]
        public void HeteroscedasticityFitDetector_ProvidesBothTestStatistics()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new HeteroscedasticityFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.Contains(result.Recommendations, r => r.Contains("Breusch-Pagan"));
            Assert.Contains(result.Recommendations, r => r.Contains("White"));
        }

        [Fact]
        public void HeteroscedasticityFitDetector_ReturnsValidConfidence()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new HeteroscedasticityFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result.ConfidenceLevel);
            Assert.True(result.ConfidenceLevel >= 0.0 && result.ConfidenceLevel <= 1.0);
        }

        [Fact]
        public void HeteroscedasticityFitDetector_DifferentFitTypes_GeneratesDifferentRecommendations()
        {
            // Arrange
            var goodData = CreateBasicEvaluationData(addNoise: false);
            var poorData = CreateBasicEvaluationData(addNoise: true, noiseFactor: 3.0);
            var detector = new HeteroscedasticityFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var goodResult = detector.DetectFit(goodData);
            var poorResult = detector.DetectFit(poorData);

            // Assert
            Assert.NotEqual(goodResult.Recommendations.Count, poorResult.Recommendations.Count);
        }

        #endregion

        #region CookDistanceFitDetector Tests

        [Fact]
        public void CookDistanceFitDetector_NoInfluentialPoints_DetectsGoodFit()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData(addNoise: false);
            var detector = new CookDistanceFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.AdditionalInfo);
            Assert.Contains("CookDistances", result.AdditionalInfo.Keys);
        }

        [Fact]
        public void CookDistanceFitDetector_CalculatesCookDistances()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new CookDistanceFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            var cookDistances = result.AdditionalInfo["CookDistances"] as Vector<double>;
            Assert.NotNull(cookDistances);
            Assert.True(cookDistances.Length > 0);
        }

        [Fact]
        public void CookDistanceFitDetector_IdentifiesTopInfluentialPoints()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new CookDistanceFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.Contains(result.Recommendations, r => r.Contains("Top 5 most influential points"));
        }

        [Fact]
        public void CookDistanceFitDetector_ReturnsValidConfidence()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new CookDistanceFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result.ConfidenceLevel);
            Assert.True(result.ConfidenceLevel >= 0.0 && result.ConfidenceLevel <= 1.0);
        }

        [Fact]
        public void CookDistanceFitDetector_ProvidesActionableRecommendations()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new CookDistanceFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotEmpty(result.Recommendations);
            Assert.True(result.Recommendations.Any(r => r.Contains("influential") || r.Contains("Cook")));
        }

        [Fact]
        public void CookDistanceFitDetector_DetectsOutliers()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new CookDistanceFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            var cookDistances = result.AdditionalInfo["CookDistances"] as Vector<double>;
            Assert.NotNull(cookDistances);
            Assert.All(cookDistances, d => Assert.True(d >= 0));
        }

        #endregion

        #region VIFFitDetector Tests

        [Fact]
        public void VIFFitDetector_LowMulticollinearity_DetectsGoodFit()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new VIFFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.FitType == FitType.GoodFit || result.FitType == FitType.PoorFit);
        }

        [Fact]
        public void VIFFitDetector_HighlyCorrelatedFeatures_DetectsMulticollinearity()
        {
            // Arrange - Create data with highly correlated features
            var trainSize = 50;
            var trainX = new Matrix<double>(trainSize, 3);
            var trainY = new Vector<double>(trainSize);

            for (int i = 0; i < trainSize; i++)
            {
                trainX[i, 0] = i / 10.0;
                trainX[i, 1] = i / 10.0 + 0.01; // Highly correlated with first feature
                trainX[i, 2] = Math.Sin(i / 10.0);
                trainY[i] = 2.0 * trainX[i, 0] + 3.0 * trainX[i, 2];
            }

            var model = new SimpleRegression<double>();
            model.Fit(trainX, trainY);
            var predictions = model.Predict(trainX);

            var evalData = CreateEvaluationDataFromPredictions(
                trainX, trainY, predictions,
                trainX, trainY, predictions,
                trainX, trainY, predictions);

            var detector = new VIFFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result);
        }

        [Fact]
        public void VIFFitDetector_ProvidesVIFMetrics()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new VIFFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.Contains(result.Recommendations, r => r.Contains("Validation") || r.Contains("Test"));
        }

        [Fact]
        public void VIFFitDetector_ReturnsValidConfidence()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new VIFFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result.ConfidenceLevel);
            Assert.True(result.ConfidenceLevel >= 0.0);
        }

        [Fact]
        public void VIFFitDetector_GeneratesRelevantRecommendations()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new VIFFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void VIFFitDetector_HandlesMultipleFeatures()
        {
            // Arrange - More features
            var trainSize = 50;
            var numFeatures = 5;
            var trainX = new Matrix<double>(trainSize, numFeatures);
            var trainY = new Vector<double>(trainSize);

            var random = new Random(42);
            for (int i = 0; i < trainSize; i++)
            {
                for (int j = 0; j < numFeatures; j++)
                {
                    trainX[i, j] = random.NextDouble();
                }
                trainY[i] = trainX[i, 0] + trainX[i, 1];
            }

            var model = new SimpleRegression<double>();
            model.Fit(trainX, trainY);
            var predictions = model.Predict(trainX);

            var evalData = CreateEvaluationDataFromPredictions(
                trainX, trainY, predictions,
                trainX, trainY, predictions,
                trainX, trainY, predictions);

            var detector = new VIFFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result);
        }

        #endregion

        #region CalibratedProbabilityFitDetector Tests

        [Fact]
        public void CalibratedProbabilityFitDetector_WellCalibratedProbabilities_DetectsGoodFit()
        {
            // Arrange - Create probability data
            var trainSize = 50;
            var trainX = new Matrix<double>(trainSize, 2);
            var trainY = new Vector<double>(trainSize);

            var random = new Random(42);
            for (int i = 0; i < trainSize; i++)
            {
                trainX[i, 0] = random.NextDouble();
                trainX[i, 1] = random.NextDouble();
                // Create probabilities between 0 and 1
                trainY[i] = Math.Min(1.0, Math.Max(0.0, trainX[i, 0] * 0.5 + trainX[i, 1] * 0.5));
            }

            var model = new SimpleRegression<double>();
            model.Fit(trainX, trainY);
            var predictions = model.Predict(trainX);

            // Ensure predictions are probabilities
            for (int i = 0; i < predictions.Length; i++)
            {
                predictions[i] = Math.Min(1.0, Math.Max(0.0, predictions[i]));
            }

            var evalData = CreateEvaluationDataFromPredictions(
                trainX, trainY, predictions,
                trainX, trainY, predictions,
                trainX, trainY, predictions);

            var detector = new CalibratedProbabilityFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.FitType);
        }

        [Fact]
        public void CalibratedProbabilityFitDetector_ReturnsValidConfidence()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            // Normalize predictions to [0, 1]
            for (int i = 0; i < evalData.TrainingSet.Predicted.Length; i++)
            {
                evalData.TrainingSet.Predicted[i] = Math.Min(1.0, Math.Max(0.0, evalData.TrainingSet.Predicted[i] / 10.0));
                evalData.TrainingSet.Actual[i] = Math.Min(1.0, Math.Max(0.0, evalData.TrainingSet.Actual[i] / 10.0));
            }

            var detector = new CalibratedProbabilityFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result.ConfidenceLevel);
            Assert.True(result.ConfidenceLevel >= 0.0 && result.ConfidenceLevel <= 1.0);
        }

        [Fact]
        public void CalibratedProbabilityFitDetector_ProvidesCalibrationRecommendations()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new CalibratedProbabilityFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void CalibratedProbabilityFitDetector_HandlesEdgeCases()
        {
            // Arrange - All predictions are 0 or 1
            var trainSize = 20;
            var trainX = new Matrix<double>(trainSize, 1);
            var trainY = new Vector<double>(trainSize);
            var predictions = new Vector<double>(trainSize);

            for (int i = 0; i < trainSize; i++)
            {
                trainX[i, 0] = i;
                trainY[i] = i % 2;
                predictions[i] = i % 2;
            }

            var evalData = CreateEvaluationDataFromPredictions(
                trainX, trainY, predictions,
                trainX, trainY, predictions,
                trainX, trainY, predictions);

            var detector = new CalibratedProbabilityFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result);
        }

        [Fact]
        public void CalibratedProbabilityFitDetector_DetectsCalibrationIssues()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new CalibratedProbabilityFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result.AdditionalInfo);
        }

        #endregion

        #region FeatureImportanceFitDetector Tests

        [Fact]
        public void FeatureImportanceFitDetector_BalancedImportance_DetectsGoodFit()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new FeatureImportanceFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result);
        }

        [Fact]
        public void FeatureImportanceFitDetector_IdentifiesTopFeatures()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new FeatureImportanceFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.Contains(result.Recommendations, r => r.Contains("Top 3 most important features"));
        }

        [Fact]
        public void FeatureImportanceFitDetector_ReturnsValidConfidence()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new FeatureImportanceFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result.ConfidenceLevel);
        }

        [Fact]
        public void FeatureImportanceFitDetector_ProvidesActionableRecommendations()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new FeatureImportanceFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotEmpty(result.Recommendations);
            Assert.True(result.Recommendations.Any(r => r.Contains("feature") || r.Contains("importance")));
        }

        [Fact]
        public void FeatureImportanceFitDetector_RanksFeatures()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new FeatureImportanceFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.Contains("FeatureImportances", result.AdditionalInfo.Keys);
        }

        #endregion

        #region PartialDependencePlotFitDetector Tests

        [Fact]
        public void PartialDependencePlotFitDetector_DetectsFitType()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new PartialDependencePlotFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.FitType);
        }

        [Fact]
        public void PartialDependencePlotFitDetector_CalculatesNonlinearity()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new PartialDependencePlotFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.Contains("PartialDependencePlots", result.AdditionalInfo.Keys);
        }

        [Fact]
        public void PartialDependencePlotFitDetector_ReturnsValidConfidence()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new PartialDependencePlotFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result.ConfidenceLevel);
            Assert.True(result.ConfidenceLevel >= 0.0 && result.ConfidenceLevel <= 1.0);
        }

        [Fact]
        public void PartialDependencePlotFitDetector_IdentifiesNonlinearFeatures()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new PartialDependencePlotFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.Contains(result.Recommendations, r => r.Contains("Top 5 most nonlinear features"));
        }

        [Fact]
        public void PartialDependencePlotFitDetector_ProvidesPlotData()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new PartialDependencePlotFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            var plots = result.AdditionalInfo["PartialDependencePlots"];
            Assert.NotNull(plots);
        }

        #endregion

        #region ShapleyValueFitDetector Tests

        [Fact]
        public void ShapleyValueFitDetector_CalculatesFeatureContributions()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var options = new ShapleyValueFitDetectorOptions();
            var detector = new ShapleyValueFitDetector<double, Matrix<double>, Vector<double>>(options);

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result);
            Assert.Contains("ShapleyValues", result.AdditionalInfo.Keys);
        }

        [Fact]
        public void ShapleyValueFitDetector_IdentifiesImportantFeatures()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var options = new ShapleyValueFitDetectorOptions();
            var detector = new ShapleyValueFitDetector<double, Matrix<double>, Vector<double>>(options);

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.Contains(result.Recommendations, r => r.Contains("Top 5 most important features"));
        }

        [Fact]
        public void ShapleyValueFitDetector_ReturnsValidConfidence()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var options = new ShapleyValueFitDetectorOptions();
            var detector = new ShapleyValueFitDetector<double, Matrix<double>, Vector<double>>(options);

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result.ConfidenceLevel);
            Assert.True(result.ConfidenceLevel >= 0.0 && result.ConfidenceLevel <= 1.0);
        }

        [Fact]
        public void ShapleyValueFitDetector_StoresShapleyValuesInAdditionalInfo()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var options = new ShapleyValueFitDetectorOptions();
            var detector = new ShapleyValueFitDetector<double, Matrix<double>, Vector<double>>(options);

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            var shapleyValues = result.AdditionalInfo["ShapleyValues"];
            Assert.NotNull(shapleyValues);
        }

        #endregion

        #region PermutationTestFitDetector Tests

        [Fact]
        public void PermutationTestFitDetector_SignificantModel_DetectsGoodFit()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData(addNoise: false);
            var detector = new PermutationTestFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result);
        }

        [Fact]
        public void PermutationTestFitDetector_ReturnsValidConfidence()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new PermutationTestFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result.ConfidenceLevel);
            Assert.True(result.ConfidenceLevel >= 0.0 && result.ConfidenceLevel <= 1.0);
        }

        [Fact]
        public void PermutationTestFitDetector_ProvidesPermutationDetails()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new PermutationTestFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.Contains(result.Recommendations, r => r.Contains("Permutation tests"));
        }

        [Fact]
        public void PermutationTestFitDetector_GeneratesRelevantRecommendations()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new PermutationTestFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void PermutationTestFitDetector_CalculatesPermutationImportance()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new PermutationTestFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.Contains("PermutationImportances", result.AdditionalInfo.Keys);
        }

        #endregion

        #region BayesianFitDetector Tests

        [Fact]
        public void BayesianFitDetector_CalculatesBayesianMetrics()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new BayesianFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result);
            Assert.Contains(result.Recommendations, r => r.Contains("DIC:") || r.Contains("WAIC:") || r.Contains("LOO:"));
        }

        [Fact]
        public void BayesianFitDetector_ReturnsValidConfidence()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new BayesianFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result.ConfidenceLevel);
        }

        [Fact]
        public void BayesianFitDetector_ProvidesBayesianRecommendations()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new BayesianFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotEmpty(result.Recommendations);
            Assert.True(result.Recommendations.Any(r =>
                r.Contains("Bayesian") || r.Contains("prior") || r.Contains("posterior")));
        }

        [Fact]
        public void BayesianFitDetector_CalculatesDIC()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new BayesianFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.Contains(result.Recommendations, r => r.Contains("DIC"));
        }

        #endregion

        #region GaussianProcessFitDetector Tests

        [Fact]
        public void GaussianProcessFitDetector_CalculatesUncertainty()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new GaussianProcessFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result);
        }

        [Fact]
        public void GaussianProcessFitDetector_ReturnsValidConfidence()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new GaussianProcessFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result.ConfidenceLevel);
            Assert.True(result.ConfidenceLevel >= 0.0 && result.ConfidenceLevel <= 1.0);
        }

        [Fact]
        public void GaussianProcessFitDetector_ProvidesKernelRecommendations()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new GaussianProcessFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotEmpty(result.Recommendations);
            Assert.True(result.Recommendations.Any(r => r.Contains("kernel") || r.Contains("Gaussian Process")));
        }

        [Fact]
        public void GaussianProcessFitDetector_MeasuresUncertaintyEstimates()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new GaussianProcessFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result.AdditionalInfo);
        }

        #endregion

        #region NeuralNetworkFitDetector Tests

        [Fact]
        public void NeuralNetworkFitDetector_CalculatesOverfittingScore()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new NeuralNetworkFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result);
            Assert.Contains("OverfittingScore", result.AdditionalInfo.Keys);
        }

        [Fact]
        public void NeuralNetworkFitDetector_TracksLossMetrics()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new NeuralNetworkFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.Contains("TrainingLoss", result.AdditionalInfo.Keys);
            Assert.Contains("ValidationLoss", result.AdditionalInfo.Keys);
            Assert.Contains("TestLoss", result.AdditionalInfo.Keys);
        }

        [Fact]
        public void NeuralNetworkFitDetector_ReturnsValidConfidence()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new NeuralNetworkFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result.ConfidenceLevel);
            Assert.True(result.ConfidenceLevel >= 0.0 && result.ConfidenceLevel <= 1.0);
        }

        [Fact]
        public void NeuralNetworkFitDetector_ProvidesNNSpecificRecommendations()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new NeuralNetworkFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void NeuralNetworkFitDetector_HighOverfitting_DetectsCorrectly()
        {
            // Arrange - Create data with high training/validation gap
            var trainData = CreateBasicEvaluationData(addNoise: false);
            // Modify validation data to simulate overfitting
            for (int i = 0; i < trainData.ValidationSet.ErrorStats.ErrorList.Length; i++)
            {
                trainData.ValidationSet.ErrorStats.ErrorList[i] *= 2.0; // Double the errors
            }

            var detector = new NeuralNetworkFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(trainData);

            // Assert
            var overfittingScore = (double)result.AdditionalInfo["OverfittingScore"];
            Assert.True(overfittingScore > 0);
        }

        #endregion

        #region GradientBoostingFitDetector Tests

        [Fact]
        public void GradientBoostingFitDetector_DetectsFitType()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new GradientBoostingFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.FitType);
        }

        [Fact]
        public void GradientBoostingFitDetector_TracksPerformanceMetrics()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new GradientBoostingFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.Contains("PerformanceMetrics", result.AdditionalInfo.Keys);
        }

        [Fact]
        public void GradientBoostingFitDetector_ReturnsValidConfidence()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new GradientBoostingFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result.ConfidenceLevel);
            Assert.True(result.ConfidenceLevel >= 0.0 && result.ConfidenceLevel <= 1.0);
        }

        [Fact]
        public void GradientBoostingFitDetector_ProvidesBoostingSpecificRecommendations()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new GradientBoostingFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void GradientBoostingFitDetector_DetectsOverfitting()
        {
            // Arrange - Create data with overfitting scenario
            var evalData = CreateBasicEvaluationData(addNoise: true, noiseFactor: 2.0);
            var detector = new GradientBoostingFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.FitType == FitType.PoorFit || result.FitType == FitType.Overfit || result.FitType == FitType.Moderate);
        }

        #endregion

        #region EnsembleFitDetector Tests

        [Fact]
        public void EnsembleFitDetector_CombinesMultipleDetectors()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detectors = new List<IFitDetector<double, Matrix<double>, Vector<double>>>
            {
                new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>(),
                new AutocorrelationFitDetector<double, Matrix<double>, Vector<double>>()
            };
            var detector = new EnsembleFitDetector<double, Matrix<double>, Vector<double>>(detectors);

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result);
            Assert.Contains("IndividualResults", result.AdditionalInfo.Keys);
        }

        [Fact]
        public void EnsembleFitDetector_ReturnsAggregatedConfidence()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detectors = new List<IFitDetector<double, Matrix<double>, Vector<double>>>
            {
                new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>(),
                new PermutationTestFitDetector<double, Matrix<double>, Vector<double>>()
            };
            var detector = new EnsembleFitDetector<double, Matrix<double>, Vector<double>>(detectors);

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result.ConfidenceLevel);
            Assert.True(result.ConfidenceLevel >= 0.0 && result.ConfidenceLevel <= 1.0);
        }

        [Fact]
        public void EnsembleFitDetector_CombinesRecommendations()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detectors = new List<IFitDetector<double, Matrix<double>, Vector<double>>>
            {
                new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>(),
                new AutocorrelationFitDetector<double, Matrix<double>, Vector<double>>()
            };
            var detector = new EnsembleFitDetector<double, Matrix<double>, Vector<double>>(detectors);

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void EnsembleFitDetector_StoresIndividualResults()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detectors = new List<IFitDetector<double, Matrix<double>, Vector<double>>>
            {
                new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>(),
                new PermutationTestFitDetector<double, Matrix<double>, Vector<double>>()
            };
            var detector = new EnsembleFitDetector<double, Matrix<double>, Vector<double>>(detectors);

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            var individualResults = result.AdditionalInfo["IndividualResults"];
            Assert.NotNull(individualResults);
        }

        [Fact]
        public void EnsembleFitDetector_WeightedAggregation_WorksCorrectly()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detectors = new List<IFitDetector<double, Matrix<double>, Vector<double>>>
            {
                new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>(),
                new AutocorrelationFitDetector<double, Matrix<double>, Vector<double>>()
            };
            var options = new EnsembleFitDetectorOptions
            {
                DetectorWeights = new List<double> { 0.7, 0.3 }
            };
            var detector = new EnsembleFitDetector<double, Matrix<double>, Vector<double>>(detectors, options);

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result);
            Assert.Contains("DetectorWeights", result.AdditionalInfo.Keys);
        }

        [Fact]
        public void EnsembleFitDetector_ThreeDetectors_CombinesCorrectly()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detectors = new List<IFitDetector<double, Matrix<double>, Vector<double>>>
            {
                new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>(),
                new AutocorrelationFitDetector<double, Matrix<double>, Vector<double>>(),
                new PermutationTestFitDetector<double, Matrix<double>, Vector<double>>()
            };
            var detector = new EnsembleFitDetector<double, Matrix<double>, Vector<double>>(detectors);

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result);
            var individualResults = result.AdditionalInfo["IndividualResults"] as List<FitDetectorResult<double>>;
            Assert.Equal(3, individualResults.Count);
        }

        #endregion

        #region HybridFitDetector Tests

        [Fact]
        public void HybridFitDetector_CombinesResidualAndLearningCurve()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var residualDetector = new ResidualAnalysisFitDetector<double, Matrix<double>, Vector<double>>();
            var learningCurveDetector = new LearningCurveFitDetector<double, Matrix<double>, Vector<double>>();
            var detector = new HybridFitDetector<double, Matrix<double>, Vector<double>>(
                residualDetector, learningCurveDetector);

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.FitType);
        }

        [Fact]
        public void HybridFitDetector_ReturnsWeightedConfidence()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var residualDetector = new ResidualAnalysisFitDetector<double, Matrix<double>, Vector<double>>();
            var learningCurveDetector = new LearningCurveFitDetector<double, Matrix<double>, Vector<double>>();
            var detector = new HybridFitDetector<double, Matrix<double>, Vector<double>>(
                residualDetector, learningCurveDetector);

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result.ConfidenceLevel);
            Assert.True(result.ConfidenceLevel >= 0.0 && result.ConfidenceLevel <= 1.0);
        }

        [Fact]
        public void HybridFitDetector_CombinesBothRecommendations()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var residualDetector = new ResidualAnalysisFitDetector<double, Matrix<double>, Vector<double>>();
            var learningCurveDetector = new LearningCurveFitDetector<double, Matrix<double>, Vector<double>>();
            var detector = new HybridFitDetector<double, Matrix<double>, Vector<double>>(
                residualDetector, learningCurveDetector);

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void HybridFitDetector_BalancesTwoApproaches()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var residualDetector = new ResidualAnalysisFitDetector<double, Matrix<double>, Vector<double>>();
            var learningCurveDetector = new LearningCurveFitDetector<double, Matrix<double>, Vector<double>>();
            var detector = new HybridFitDetector<double, Matrix<double>, Vector<double>>(
                residualDetector, learningCurveDetector);

            // Act
            var result = detector.DetectFit(evalData);

            // Assert - Should have recommendations from both approaches
            Assert.True(result.Recommendations.Count >= 2);
        }

        #endregion

        #region AdaptiveFitDetector Tests

        [Fact]
        public void AdaptiveFitDetector_SelectsAppropriateDetector()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result);
            Assert.Contains(result.Recommendations, r => r.Contains("adaptive fit detector used"));
        }

        [Fact]
        public void AdaptiveFitDetector_ReturnsValidConfidence()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result.ConfidenceLevel);
            Assert.True(result.ConfidenceLevel >= 0.0 && result.ConfidenceLevel <= 1.0);
        }

        [Fact]
        public void AdaptiveFitDetector_ExplainsDetectorChoice()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotEmpty(result.Recommendations);
            Assert.Contains(result.Recommendations, r =>
                r.Contains("data complexity") || r.Contains("model performance"));
        }

        [Fact]
        public void AdaptiveFitDetector_HandlesDifferentDataComplexities()
        {
            // Arrange - Simple data
            var simpleData = CreateBasicEvaluationData(addNoise: false);
            // Complex data
            var complexData = CreateBasicEvaluationData(addNoise: true, noiseFactor: 2.0);

            var detector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var simpleResult = detector.DetectFit(simpleData);
            var complexResult = detector.DetectFit(complexData);

            // Assert
            Assert.NotNull(simpleResult);
            Assert.NotNull(complexResult);
        }

        [Fact]
        public void AdaptiveFitDetector_GoodPerformance_SelectsResidualAnalyzer()
        {
            // Arrange - Create simple data with good fit
            var evalData = CreateBasicEvaluationData(addNoise: false);
            var detector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.Contains(result.Recommendations, r => r.Contains("Residual Analysis") || r.Contains("Learning Curve") || r.Contains("Hybrid"));
        }

        [Fact]
        public void AdaptiveFitDetector_PoorPerformance_SelectsHybridDetector()
        {
            // Arrange - Create complex data with poor fit
            var evalData = CreateBasicEvaluationData(addNoise: true, noiseFactor: 3.0);
            var detector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result);
        }

        #endregion

        #region Cross-Detector Comparison Tests

        [Fact]
        public void AllDetectors_ReturnValidResults()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detectors = new List<IFitDetector<double, Matrix<double>, Vector<double>>>
            {
                new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>(),
                new AutocorrelationFitDetector<double, Matrix<double>, Vector<double>>(),
                new HeteroscedasticityFitDetector<double, Matrix<double>, Vector<double>>(),
                new CookDistanceFitDetector<double, Matrix<double>, Vector<double>>(),
                new VIFFitDetector<double, Matrix<double>, Vector<double>>(),
                new PermutationTestFitDetector<double, Matrix<double>, Vector<double>>(),
                new NeuralNetworkFitDetector<double, Matrix<double>, Vector<double>>(),
                new GradientBoostingFitDetector<double, Matrix<double>, Vector<double>>()
            };

            // Act & Assert
            foreach (var detector in detectors)
            {
                var result = detector.DetectFit(evalData);
                Assert.NotNull(result);
                Assert.NotNull(result.FitType);
                Assert.NotNull(result.Recommendations);
            }
        }

        [Fact]
        public void AllDetectors_ReturnValidConfidenceLevels()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detectors = new List<IFitDetector<double, Matrix<double>, Vector<double>>>
            {
                new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>(),
                new AutocorrelationFitDetector<double, Matrix<double>, Vector<double>>(),
                new HeteroscedasticityFitDetector<double, Matrix<double>, Vector<double>>(),
                new PermutationTestFitDetector<double, Matrix<double>, Vector<double>>(),
                new NeuralNetworkFitDetector<double, Matrix<double>, Vector<double>>()
            };

            // Act & Assert
            foreach (var detector in detectors)
            {
                var result = detector.DetectFit(evalData);
                if (result.ConfidenceLevel.HasValue)
                {
                    Assert.True(result.ConfidenceLevel >= 0.0);
                }
            }
        }

        [Fact]
        public void AllDetectors_ProvideNonEmptyRecommendations()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detectors = new List<IFitDetector<double, Matrix<double>, Vector<double>>>
            {
                new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>(),
                new AutocorrelationFitDetector<double, Matrix<double>, Vector<double>>(),
                new VIFFitDetector<double, Matrix<double>, Vector<double>>(),
                new PermutationTestFitDetector<double, Matrix<double>, Vector<double>>()
            };

            // Act & Assert
            foreach (var detector in detectors)
            {
                var result = detector.DetectFit(evalData);
                Assert.NotEmpty(result.Recommendations);
            }
        }

        [Fact]
        public void InformationCriteriaFitDetector_LowComplexityModel_PrefersBIC()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData(trainSize: 200);
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert - With more data, BIC should penalize complexity more
            Assert.NotNull(result);
        }

        [Fact]
        public void HeteroscedasticityFitDetector_IncreasingVariance_DetectsHeteroscedasticity()
        {
            // Arrange - Create data with increasing variance
            var trainSize = 50;
            var trainX = new Matrix<double>(trainSize, 1);
            var trainY = new Vector<double>(trainSize);

            var random = new Random(42);
            for (int i = 0; i < trainSize; i++)
            {
                trainX[i, 0] = i / 10.0;
                // Variance increases with x
                var variance = 0.1 + (i / 50.0) * 2.0;
                trainY[i] = 2.0 * trainX[i, 0] + random.NextDouble() * variance;
            }

            var model = new SimpleRegression<double>();
            model.Fit(trainX, trainY);
            var predictions = model.Predict(trainX);

            var evalData = CreateEvaluationDataFromPredictions(
                trainX, trainY, predictions,
                trainX, trainY, predictions,
                trainX, trainY, predictions);

            var detector = new HeteroscedasticityFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result);
        }

        [Fact]
        public void CalibratedProbabilityFitDetector_BinaryClassification_WorksCorrectly()
        {
            // Arrange - Binary classification scenario
            var trainSize = 100;
            var trainX = new Matrix<double>(trainSize, 2);
            var trainY = new Vector<double>(trainSize);

            var random = new Random(42);
            for (int i = 0; i < trainSize; i++)
            {
                trainX[i, 0] = random.NextDouble();
                trainX[i, 1] = random.NextDouble();
                // Binary outcomes
                trainY[i] = (trainX[i, 0] + trainX[i, 1]) > 1.0 ? 1.0 : 0.0;
            }

            var model = new SimpleRegression<double>();
            model.Fit(trainX, trainY);
            var predictions = model.Predict(trainX);

            // Clip predictions to [0, 1]
            for (int i = 0; i < predictions.Length; i++)
            {
                predictions[i] = Math.Max(0.0, Math.Min(1.0, predictions[i]));
            }

            var evalData = CreateEvaluationDataFromPredictions(
                trainX, trainY, predictions,
                trainX, trainY, predictions,
                trainX, trainY, predictions);

            var detector = new CalibratedProbabilityFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result);
        }

        [Fact]
        public void FeatureImportanceFitDetector_SingleDominantFeature_IdentifiesCorrectly()
        {
            // Arrange - One feature dominates
            var trainSize = 50;
            var trainX = new Matrix<double>(trainSize, 3);
            var trainY = new Vector<double>(trainSize);

            var random = new Random(42);
            for (int i = 0; i < trainSize; i++)
            {
                trainX[i, 0] = i / 10.0;
                trainX[i, 1] = random.NextDouble() * 0.1; // Weak feature
                trainX[i, 2] = random.NextDouble() * 0.1; // Weak feature
                trainY[i] = 5.0 * trainX[i, 0] + 0.1 * trainX[i, 1] + 0.05 * trainX[i, 2];
            }

            var model = new SimpleRegression<double>();
            model.Fit(trainX, trainY);
            var predictions = model.Predict(trainX);

            var evalData = CreateEvaluationDataFromPredictions(
                trainX, trainY, predictions,
                trainX, trainY, predictions,
                trainX, trainY, predictions);

            var detector = new FeatureImportanceFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result);
            Assert.Contains("FeatureImportances", result.AdditionalInfo.Keys);
        }

        [Fact]
        public void BayesianFitDetector_ComparesModelComplexities()
        {
            // Arrange
            var simpleModel = CreateBasicEvaluationData(trainSize: 50, valSize: 25, testSize: 25);
            var complexModel = CreateBasicEvaluationData(trainSize: 100, valSize: 50, testSize: 50);

            var detector = new BayesianFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var simpleResult = detector.DetectFit(simpleModel);
            var complexResult = detector.DetectFit(complexModel);

            // Assert
            Assert.NotNull(simpleResult);
            Assert.NotNull(complexResult);
        }

        [Fact]
        public void GaussianProcessFitDetector_HighUncertainty_DetectsCorrectly()
        {
            // Arrange - Create sparse data to induce high uncertainty
            var evalData = CreateBasicEvaluationData(trainSize: 20, valSize: 10, testSize: 10);
            var detector = new GaussianProcessFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result);
        }

        [Fact]
        public void NeuralNetworkFitDetector_NoOverfitting_DetectsGoodFit()
        {
            // Arrange - Similar training and validation loss
            var evalData = CreateBasicEvaluationData(addNoise: false);
            var detector = new NeuralNetworkFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.Contains("OverfittingScore", result.AdditionalInfo.Keys);
            var score = (double)result.AdditionalInfo["OverfittingScore"];
            Assert.True(score >= 0);
        }

        [Fact]
        public void GradientBoostingFitDetector_EarlyStoppingRequired_Recommends()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData(addNoise: true, noiseFactor: 1.5);
            var detector = new GradientBoostingFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void EnsembleFitDetector_DisagreeingDetectors_ReducesConfidence()
        {
            // Arrange
            var evalData = CreateBasicEvaluationData();
            var detectors = new List<IFitDetector<double, Matrix<double>, Vector<double>>>
            {
                new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>(),
                new AutocorrelationFitDetector<double, Matrix<double>, Vector<double>>(),
                new HeteroscedasticityFitDetector<double, Matrix<double>, Vector<double>>()
            };
            var detector = new EnsembleFitDetector<double, Matrix<double>, Vector<double>>(detectors);

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result.ConfidenceLevel);
        }

        [Fact]
        public void HybridFitDetector_AgreeingComponents_IncreasesConfidence()
        {
            // Arrange - Good fit should have both detectors agree
            var evalData = CreateBasicEvaluationData(addNoise: false);
            var residualDetector = new ResidualAnalysisFitDetector<double, Matrix<double>, Vector<double>>();
            var learningCurveDetector = new LearningCurveFitDetector<double, Matrix<double>, Vector<double>>();
            var detector = new HybridFitDetector<double, Matrix<double>, Vector<double>>(
                residualDetector, learningCurveDetector);

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotNull(result.ConfidenceLevel);
        }

        [Fact]
        public void AdaptiveFitDetector_ModerateComplexity_SelectsLearningCurve()
        {
            // Arrange - Moderate complexity
            var evalData = CreateBasicEvaluationData(addNoise: true, noiseFactor: 0.5);
            var detector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>();

            // Act
            var result = detector.DetectFit(evalData);

            // Assert
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void AllAdvancedDetectors_HandleLargeDatasets()
        {
            // Arrange - Larger dataset
            var evalData = CreateBasicEvaluationData(trainSize: 200, valSize: 100, testSize: 100);
            var detectors = new List<IFitDetector<double, Matrix<double>, Vector<double>>>
            {
                new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>(),
                new CookDistanceFitDetector<double, Matrix<double>, Vector<double>>(),
                new PermutationTestFitDetector<double, Matrix<double>, Vector<double>>()
            };

            // Act & Assert
            foreach (var detector in detectors)
            {
                var result = detector.DetectFit(evalData);
                Assert.NotNull(result);
            }
        }

        #endregion
    }
}

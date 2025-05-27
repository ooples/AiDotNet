using StockTradingTest.Configuration;
using StockTradingTest.Models;
using AiDotNet.Interfaces;
using AiDotNet;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.Normalizers;
using AiDotNet.Enums;

namespace StockTradingTest.Services
{
    public class ModelTrainer
    {
        private readonly StockDataService _dataService;
        private readonly ModelCompetitionConfig _config;
        private readonly Random _random = new Random(42); // Fixed seed for reproducibility

        public ModelTrainer(StockDataService dataService, ModelCompetitionConfig config)
        {
            _dataService = dataService;
            _config = config;
        }

        public List<ModelCompetitor> CreateCompetitors()
        {
            var competitors = new List<ModelCompetitor>
            {
                CreateNeuralNetworkCompetitor("FeedForward Neural Network"),
                CreateLSTMCompetitor("LSTM Neural Network"),
                CreateTransformerCompetitor("Transformer Model"),
                CreateRandomForestCompetitor("Random Forest"),
                CreateGradientBoostingCompetitor("Gradient Boosting"),
                CreateSVRCompetitor("Support Vector Regression"),
                CreateGaussianProcessCompetitor("Gaussian Process"),
                CreateARIMACompetitor("ARIMA Model")
            };

            return competitors;
        }

        public async Task<ModelCompetitor> TrainModelAsync(ModelCompetitor competitor, string symbol)
        {
            DateTime trainStartDate = _config.StartDate;
            DateTime trainEndDate = _config.StartDate.AddDays((DateTime.Now - _config.StartDate).TotalDays * (1 - _config.ValidationDataSplit - _config.TestDataSplit));
            DateTime validationEndDate = trainEndDate.AddDays((DateTime.Now - _config.StartDate).TotalDays * _config.ValidationDataSplit);

            var (trainFeatures, trainTargets) = _dataService.PrepareModelData(
                symbol,
                trainStartDate,
                trainEndDate,
                _config.LookbackPeriod,
                _config.PredictionHorizon
            );

            var (validationFeatures, validationTargets) = _dataService.PrepareModelData(
                symbol,
                trainEndDate.AddDays(1),
                validationEndDate,
                _config.LookbackPeriod,
                _config.PredictionHorizon
            );

            var normalizer = new MeanVarianceNormalizer<double, Matrix<double>, Matrix<double>>();
            var normalizedTrainFeatures = normalizer.Normalize(trainFeatures);
            var normalizedValidationFeatures = normalizer.Normalize(validationFeatures);

            var startTime = DateTime.Now;
            await Task.Run(() => competitor.Model.Fit(normalizedTrainFeatures, trainTargets));
            var endTime = DateTime.Now;

            // Calculate training metrics
            var trainPredictions = competitor.Model.Predict(normalizedTrainFeatures);
            competitor.TrainingMetrics = CalculateMetrics(trainTargets, trainPredictions, endTime - startTime, trainFeatures.Rows);

            // Calculate validation metrics
            var validationStartTime = DateTime.Now;
            var validationPredictions = competitor.Model.Predict(normalizedValidationFeatures);
            var validationEndTime = DateTime.Now;
            competitor.ValidationMetrics = CalculateMetrics(validationTargets, validationPredictions, validationEndTime - validationStartTime, validationFeatures.Rows);

            return competitor;
        }

        private ModelMetrics CalculateMetrics(Vector<double> actual, Vector<double> predicted, TimeSpan processingTime, int dataPoints)
        {
            double mse = 0;
            double mae = 0;
            int directionalCorrect = 0;

            for (int i = 0; i < actual.Length; i++)
            {
                double diff = predicted[i] - actual[i];
                mse += diff * diff;
                mae += Math.Abs(diff);

                // Directional accuracy
                if ((predicted[i] > 0 && actual[i] > 0) || (predicted[i] < 0 && actual[i] < 0))
                {
                    directionalCorrect++;
                }
            }

            mse /= actual.Length;
            mae /= actual.Length;
            double rmse = Math.Sqrt(mse);

            // Calculate R2 score
            double meanActual = 0;
            for (int i = 0; i < actual.Length; i++)
            {
                meanActual += actual[i];
            }
            meanActual /= actual.Length;

            double ssTotal = 0;
            double ssResidual = 0;
            for (int i = 0; i < actual.Length; i++)
            {
                ssTotal += Math.Pow(actual[i] - meanActual, 2);
                ssResidual += Math.Pow(actual[i] - predicted[i], 2);
            }

            double r2 = 1 - (ssResidual / ssTotal);

            return new ModelMetrics
            {
                MeanAbsoluteError = mae,
                MeanSquaredError = mse,
                RootMeanSquaredError = rmse,
                R2Score = r2,
                DirectionalAccuracy = (double)directionalCorrect / actual.Length,
                TrainingTime = processingTime,
                InferenceTime = TimeSpan.FromMilliseconds(processingTime.TotalMilliseconds / actual.Length),
                TrainingDataPoints = dataPoints,
                TestingDataPoints = actual.Length
            };
        }

        #region Model Creation Methods

        private ModelCompetitor CreateNeuralNetworkCompetitor(string name)
        {
            var builder = new PredictionModelBuilder();
            builder.UseConfiguration(new NeuralNetworkRegressionOptions
            {
                Layers = new int[] { 50, 30, 15, 1 },
                Epochs = 100,
                BatchSize = 32,
                LearningRate = 0.001,
                Optimizer = OptimizerType.Adam,
                RegularizationType = RegularizationType.L2,
                RegularizationRate = 0.0001,
                ActivationFunction = ActivationFunction.ReLU
            });

            return new ModelCompetitor
            {
                Name = name,
                Model = builder.BuildFullModel(),
                Type = ModelType.NeuralNetwork
            };
        }

        private ModelCompetitor CreateLSTMCompetitor(string name)
        {
            var builder = new PredictionModelBuilder();
            builder.UseConfiguration(new NeuralNetworkRegressionOptions
            {
                Layers = new int[] { 128, 64, 1 },
                Epochs = 100,
                BatchSize = 32,
                LearningRate = 0.001,
                Optimizer = OptimizerType.Adam,
                RegularizationType = RegularizationType.L2,
                RegularizationRate = 0.0001,
                NeuralNetworkTaskType = NeuralNetworkTaskType.TimeSeriesForecasting
            });

            return new ModelCompetitor
            {
                Name = name,
                Model = builder.BuildFullModel(),
                Type = ModelType.LSTM
            };
        }

        private ModelCompetitor CreateTransformerCompetitor(string name)
        {
            var builder = new PredictionModelBuilder();
            builder.UseConfiguration(new NeuralNetworkRegressionOptions
            {
                Layers = new int[] { 128, 128, 64, 1 },
                Epochs = 100,
                BatchSize = 32,
                LearningRate = 0.0001,
                Optimizer = OptimizerType.Adam,
                RegularizationType = RegularizationType.L2,
                RegularizationRate = 0.0001,
                NeuralNetworkTaskType = NeuralNetworkTaskType.TimeSeriesForecasting,
                TransformerTaskType = TransformerTaskType.Forecasting
            });

            return new ModelCompetitor
            {
                Name = name,
                Model = builder.BuildFullModel(),
                Type = ModelType.Transformer
            };
        }

        private ModelCompetitor CreateRandomForestCompetitor(string name)
        {
            var builder = new PredictionModelBuilder();
            builder.UseConfiguration(new RandomForestRegressionOptions
            {
                NumberOfTrees = 100,
                MaxDepth = 15,
                MinSamplesSplit = 2,
                MaxFeatures = 0.7,
                Bootstrap = true
            });

            return new ModelCompetitor
            {
                Name = name,
                Model = builder.BuildFullModel(),
                Type = ModelType.RandomForest
            };
        }

        private ModelCompetitor CreateGradientBoostingCompetitor(string name)
        {
            var builder = new PredictionModelBuilder();
            builder.UseConfiguration(new GradientBoostingRegressionOptions
            {
                NumberOfTrees = 100,
                LearningRate = 0.1,
                MaxDepth = 5,
                SubsampleRatio = 0.8,
                LossFunction = FitnessCalculatorType.MeanSquaredError
            });

            return new ModelCompetitor
            {
                Name = name,
                Model = builder.BuildFullModel(),
                Type = ModelType.GradientBoosting
            };
        }

        private ModelCompetitor CreateSVRCompetitor(string name)
        {
            var builder = new PredictionModelBuilder();
            builder.UseConfiguration(new SupportVectorRegressionOptions
            {
                Kernel = KernelType.RBF,
                C = 1.0,
                Epsilon = 0.1,
                Gamma = 0.1,
                CacheSize = 200,
                Shrinking = true
            });

            return new ModelCompetitor
            {
                Name = name,
                Model = builder.BuildFullModel(),
                Type = ModelType.SupportVectorMachine
            };
        }

        private ModelCompetitor CreateGaussianProcessCompetitor(string name)
        {
            var builder = new PredictionModelBuilder();
            builder.UseConfiguration(new GaussianProcessRegressionOptions
            {
                Kernel = KernelType.RBF,
                Alpha = 0.00001,
                Optimizer = OptimizerType.LevenbergMarquardt,
                NumberOfRestarts = 3
            });

            return new ModelCompetitor
            {
                Name = name,
                Model = builder.BuildFullModel(),
                Type = ModelType.GaussianProcess
            };
        }

        private ModelCompetitor CreateARIMACompetitor(string name)
        {
            var builder = new PredictionModelBuilder();
            builder.UseConfiguration(new ARIMAOptions
            {
                AR = 5,
                I = 1,
                MA = 2,
                FitIntercept = true
            });

            return new ModelCompetitor
            {
                Name = name,
                Model = builder.BuildFullModel(),
                Type = ModelType.ARIMA
            };
        }

        #endregion
    }
}
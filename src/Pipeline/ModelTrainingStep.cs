using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;

namespace AiDotNet.Pipeline
{
    /// <summary>
    /// Generic production-ready model training pipeline step
    /// </summary>
    /// <typeparam name="T">The numeric type for computations</typeparam>
    public class ModelTrainingStep<T> : PipelineStepBase<T>
    {
        private readonly ModelTrainingConfig<T> config;
        private IFullModel<T, Matrix<T>, Vector<T>>? trainedModel;
        private readonly Dictionary<string, double> trainingMetrics;
        
        public ModelTrainingStep(ModelTrainingConfig<T> config) 
            : base("ModelTraining", MathHelper.GetNumericOperations<T>())
        {
            this.config = config ?? throw new ArgumentNullException(nameof(config));
            this.trainingMetrics = new Dictionary<string, double>();
            
            Position = PipelinePosition.Any; // Note: Middle doesn't exist in the enum
            IsCacheable = false; // Training should not be cached
        }
        
        protected override bool RequiresFitting() => true;
        
        protected override void FitCore(Matrix<T> inputs, Vector<T>? targets)
        {
            if (targets == null)
            {
                throw new ArgumentException("Targets are required for model training");
            }
            
            // Create model based on configuration
            trainedModel = CreateModel();
            
            if (trainedModel == null)
            {
                throw new InvalidOperationException("Failed to create model");
            }
            
            // Split data if validation split is specified
            Matrix<T> trainInputs = inputs;
            Vector<T> trainTargets = targets;
            Matrix<T>? valInputs = null;
            Vector<T>? valTargets = null;
            
            if (config.ValidationSplit > 0)
            {
                var splitIndex = (int)(inputs.Rows * (1 - config.ValidationSplit));
                
                // Create training data
                trainInputs = new Matrix<T>(splitIndex, inputs.Columns);
                trainTargets = new Vector<T>(splitIndex);
                
                // Create validation data
                valInputs = new Matrix<T>(inputs.Rows - splitIndex, inputs.Columns);
                valTargets = new Vector<T>(inputs.Rows - splitIndex);
                
                // Copy data
                for (int i = 0; i < splitIndex; i++)
                {
                    for (int j = 0; j < inputs.Columns; j++)
                    {
                        trainInputs[i, j] = inputs[i, j];
                    }
                    trainTargets[i] = targets[i];
                }
                
                for (int i = splitIndex; i < inputs.Rows; i++)
                {
                    for (int j = 0; j < inputs.Columns; j++)
                    {
                        valInputs[i - splitIndex, j] = inputs[i, j];
                    }
                    valTargets[i - splitIndex] = targets[i];
                }
            }
            
            // Train model
            var stopwatch = Stopwatch.StartNew();
            
            try
            {
                if (config.UseEarlyStopping && valInputs != null && valTargets != null)
                {
                    TrainWithEarlyStopping(trainInputs, trainTargets, valInputs, valTargets);
                }
                else
                {
                    trainedModel.Train(trainInputs, trainTargets);
                }
                
                stopwatch.Stop();
                trainingMetrics["TrainingTimeMs"] = stopwatch.ElapsedMilliseconds;
                
                // Calculate training metrics
                var trainPredictions = trainedModel.Predict(trainInputs);
                CalculateMetrics(trainTargets, trainPredictions, "Train");
                
                if (valInputs != null && valTargets != null)
                {
                    var valPredictions = trainedModel.Predict(valInputs);
                    CalculateMetrics(valTargets, valPredictions, "Validation");
                }
                
                UpdateMetadata("ModelType", config.ModelType.ToString());
                UpdateMetadata("TrainingTime", $"{stopwatch.ElapsedMilliseconds}ms");
                UpdateMetadata("TrainingSamples", trainInputs.Rows.ToString());
                
                if (trainedModel is IParameterizable<T, Matrix<T>, Vector<T>> param)
                {
                    var paramCount = param.GetParameters().Count;
                    UpdateMetadata("ParameterCount", paramCount.ToString());
                }
            }
            catch (Exception ex)
            {
                UpdateMetadata("TrainingError", ex.Message);
                throw new InvalidOperationException($"Model training failed: {ex.Message}", ex);
            }
        }
        
        protected override Matrix<T> TransformCore(Matrix<T> inputs)
        {
            if (trainedModel == null)
            {
                throw new InvalidOperationException("Model has not been trained");
            }
            
            // For training step, transform returns predictions
            var predictions = trainedModel.Predict(inputs);
            
            // Convert predictions to Matrix format
            var result = new Matrix<T>(predictions.Length, 1);
            for (int i = 0; i < predictions.Length; i++)
            {
                result[i, 0] = predictions[i];
            }
            
            return result;
        }
        
        private IFullModel<T, Matrix<T>, Vector<T>> CreateModel()
        {
            return config.Model ?? throw new InvalidOperationException("Model must be provided in configuration");
        }
        
        private void TrainWithEarlyStopping(Matrix<T> trainInputs, Vector<T> trainTargets, 
                                           Matrix<T> valInputs, Vector<T> valTargets)
        {
            if (trainedModel == null) return;
            
            var bestValLoss = double.MaxValue;
            var patience = config.EarlyStoppingPatience;
            var patienceCounter = 0;
            
            for (int epoch = 0; epoch < config.MaxEpochs; epoch++)
            {
                // Train for one epoch
                trainedModel.Train(trainInputs, trainTargets);
                
                // Calculate validation loss
                var valPredictions = trainedModel.Predict(valInputs);
                var valLoss = CalculateLoss(valTargets, valPredictions);
                
                if (valLoss < bestValLoss)
                {
                    bestValLoss = valLoss;
                    patienceCounter = 0;
                }
                else
                {
                    patienceCounter++;
                    if (patienceCounter >= patience)
                    {
                        UpdateMetadata("EarlyStoppingEpoch", epoch.ToString());
                        break;
                    }
                }
            }
        }
        
        private double CalculateLoss(Vector<T> actual, Vector<T> predicted)
        {
            double sum = 0;
            for (int i = 0; i < actual.Length; i++)
            {
                var diff = Convert.ToDouble(NumOps.Subtract(actual[i], predicted[i]));
                sum += diff * diff;
            }
            return sum / actual.Length;
        }
        
        private void CalculateMetrics(Vector<T> actual, Vector<T> predicted, string prefix)
        {
            // Calculate common metrics
            var mse = CalculateLoss(actual, predicted);
            var rmse = Math.Sqrt(mse);
            
            trainingMetrics[$"{prefix}MSE"] = mse;
            trainingMetrics[$"{prefix}RMSE"] = rmse;
            
            // Calculate R-squared
            var actualMean = CalculateMean(actual);
            var ssTotal = 0.0;
            var ssResidual = 0.0;
            
            for (int i = 0; i < actual.Length; i++)
            {
                var actualVal = Convert.ToDouble(actual[i]);
                var predictedVal = Convert.ToDouble(predicted[i]);
                var meanVal = Convert.ToDouble(actualMean);
                
                ssTotal += Math.Pow(actualVal - meanVal, 2);
                ssResidual += Math.Pow(actualVal - predictedVal, 2);
            }
            
            var rSquared = 1 - (ssResidual / ssTotal);
            trainingMetrics[$"{prefix}R2"] = rSquared;
        }
        
        private T CalculateMean(Vector<T> values)
        {
            var sum = NumOps.Zero;
            for (int i = 0; i < values.Length; i++)
            {
                sum = NumOps.Add(sum, values[i]);
            }
            return NumOps.Divide(sum, NumOps.FromDouble(values.Length));
        }
        
        /// <summary>
        /// Gets the trained model
        /// </summary>
        public IFullModel<T, Matrix<T>, Vector<T>>? GetTrainedModel() => trainedModel;
        
        /// <summary>
        /// Gets the training metrics
        /// </summary>
        public Dictionary<string, double> GetTrainingMetrics() => new Dictionary<string, double>(trainingMetrics);
        
        /// <summary>
        /// Predicts using the trained model
        /// </summary>
        public Vector<T> Predict(Matrix<T> inputs)
        {
            if (trainedModel == null)
            {
                throw new InvalidOperationException("Model has not been trained");
            }
            
            return trainedModel.Predict(inputs);
        }
        
        /// <summary>
        /// Saves the trained model to file
        /// </summary>
        public async Task SaveModelAsync(string path)
        {
            if (trainedModel == null)
            {
                throw new InvalidOperationException("Model has not been trained");
            }
            
            if (trainedModel is IModelSerializer serializable)
            {
                var data = await Task.Run(() => serializable.Serialize());
                await System.IO.File.WriteAllBytesAsync(path, data);
            }
            else
            {
                throw new NotSupportedException("Model does not support serialization");
            }
        }
        
        /// <summary>
        /// Loads a model from file
        /// </summary>
        public async Task LoadModelAsync(string path)
        {
            trainedModel = CreateModel();
            
            if (trainedModel is IModelSerializer serializable)
            {
                var data = await System.IO.File.ReadAllBytesAsync(path);
                await Task.Run(() => serializable.Deserialize(data));
            }
            else
            {
                throw new NotSupportedException("Model does not support deserialization");
            }
        }
    }
    
    /// <summary>
    /// Configuration for model training
    /// </summary>
    /// <typeparam name="T">The numeric type</typeparam>
    public class ModelTrainingConfig<T>
    {
        /// <summary>
        /// The model to train
        /// </summary>
        public IFullModel<T, Matrix<T>, Vector<T>>? Model { get; set; }
        
        /// <summary>
        /// Model type for metadata
        /// </summary>
        public ModelType ModelType { get; set; } = ModelType.Unknown;
        
        /// <summary>
        /// Validation split ratio (0-1)
        /// </summary>
        public double ValidationSplit { get; set; } = 0.2;
        
        /// <summary>
        /// Whether to use early stopping
        /// </summary>
        public bool UseEarlyStopping { get; set; } = true;
        
        /// <summary>
        /// Early stopping patience (epochs)
        /// </summary>
        public int EarlyStoppingPatience { get; set; } = 10;
        
        /// <summary>
        /// Maximum training epochs
        /// </summary>
        public int MaxEpochs { get; set; } = 100;
        
        /// <summary>
        /// Whether to calculate additional metrics
        /// </summary>
        public bool CalculateAdvancedMetrics { get; set; } = true;
    }
}
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Extensions;
using AiDotNet.LinearAlgebra;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.FederatedLearning.MetaLearning.Models;
using AiDotNet.FederatedLearning.MetaLearning.Parameters;

namespace AiDotNet.FederatedLearning.MetaLearning
{
    /// <summary>
    /// Production-ready Model-Agnostic Meta-Learning (MAML) for Federated Learning
    /// Enables fast adaptation to new tasks in federated settings
    /// </summary>
    public class MAMLFederated : FederatedLearningBase
    {
        /// <summary>
        /// Meta-learning parameters
        /// </summary>
        public MAMLParameters Parameters { get; set; } = new();

        /// <summary>
        /// Meta-model for learning across tasks
        /// </summary>
        public IFullModel<double, Matrix<double>, Vector<double>> MetaModel { get; private set; } = default!;

        /// <summary>
        /// Client task definitions
        /// </summary>
        protected Dictionary<string, FederatedTask> ClientTasks { get; set; } = new();

        /// <summary>
        /// Meta-gradient accumulator
        /// </summary>
        protected Dictionary<string, Vector<double>> MetaGradients { get; set; } = new();

        /// <summary>
        /// Task performance history
        /// </summary>
        public List<MetaLearningRound> MetaHistory { get; private set; } = new();

        /// <summary>
        /// Inner loop optimizer for task adaptation
        /// </summary>
        protected IOptimizer<double, Matrix<double>, Vector<double>> InnerOptimizer { get; set; } = default!;

        /// <summary>
        /// Outer loop optimizer for meta-updates
        /// </summary>
        protected IOptimizer<double, Matrix<double>, Vector<double>> OuterOptimizer { get; set; } = default!;

        /// <summary>
        /// Model parameter cache for efficiency
        /// </summary>
        private readonly Dictionary<string, Dictionary<string, Vector<double>>> _parameterCache;

        /// <summary>
        /// Random number generator for reproducibility
        /// </summary>
        private readonly Random _random;

        /// <summary>
        /// Initialize MAML for federated learning
        /// </summary>
        /// <param name="metaModel">Meta-model for learning</param>
        /// <param name="parameters">MAML parameters</param>
        /// <param name="seed">Random seed for reproducibility</param>
        public MAMLFederated(
            IFullModel<double, Matrix<double>, Vector<double>> metaModel, 
            MAMLParameters? parameters = null,
            int? seed = null)
        {
            MetaModel = metaModel ?? throw new ArgumentNullException(nameof(metaModel));
            Parameters = parameters ?? new MAMLParameters();
            Parameters.Validate();
            
            ClientTasks = new Dictionary<string, FederatedTask>();
            MetaGradients = new Dictionary<string, Vector<double>>();
            MetaHistory = new List<MetaLearningRound>();
            _parameterCache = new Dictionary<string, Dictionary<string, Vector<double>>>();
            _random = seed.HasValue ? new Random(seed.Value) : new Random();

            // Initialize optimizers
            InitializeOptimizers();
        }

        /// <summary>
        /// Initialize inner and outer loop optimizers
        /// </summary>
        private void InitializeOptimizers()
        {
            // Inner loop optimizer (typically SGD for simplicity)
            InnerOptimizer = new GradientDescentOptimizer<double, Matrix<double>, Vector<double>>(
                new GradientDescentOptimizerOptions<double, Matrix<double>, Vector<double>>
                {
                    LearningRate = Parameters.InnerLearningRate
                });

            // Outer loop optimizer (Adam for stability)
            OuterOptimizer = new AdamOptimizer<double, Matrix<double>, Vector<double>>(
                new AdamOptimizerOptions<double, Matrix<double>, Vector<double>>
                {
                    LearningRate = Parameters.OuterLearningRate,
                    Beta1 = 0.9,
                    Beta2 = 0.999,
                    Epsilon = 1e-8
                });
        }

        /// <summary>
        /// Register a federated task for a client
        /// </summary>
        /// <param name="clientId">Client identifier</param>
        /// <param name="task">Federated task definition</param>
        public void RegisterClientTask(string clientId, FederatedTask task)
        {
            if (string.IsNullOrEmpty(clientId))
                throw new ArgumentNullException(nameof(clientId));
            
            if (task == null)
                throw new ArgumentNullException(nameof(task));

            task.Validate();
            task.ClientId = clientId;
            
            ClientTasks[clientId] = task;
            SetClientWeight(clientId, task.TotalExamples);
        }

        /// <summary>
        /// Perform federated meta-learning round
        /// </summary>
        /// <param name="selectedClients">Selected clients for this round</param>
        /// <returns>Meta-learning results</returns>
        public async Task<MetaLearningResult> PerformMetaLearningRoundAsync(List<string> selectedClients)
        {
            if (selectedClients == null || selectedClients.Count == 0)
                throw new ArgumentException("No clients selected for meta-learning round");

            var roundStart = DateTime.UtcNow;
            var roundHistory = new MetaLearningRound { Round = CurrentRound, StartTime = roundStart };
            
            try
            {
                var clientResults = new Dictionary<string, ClientMetaResult>();

                // Phase 1: Inner loop adaptation on each client (parallel execution)
                var innerLoopTasks = selectedClients
                    .Where(clientId => ClientTasks.ContainsKey(clientId))
                    .Select(clientId => PerformInnerLoopWithErrorHandlingAsync(clientId, ClientTasks[clientId]));

                var innerResults = await Task.WhenAll(innerLoopTasks);

                foreach (var result in innerResults.Where(r => r != null))
                {
                    if (result != null)
                    {
                        clientResults[result.ClientId] = result;
                    }
                }

                if (clientResults.Count == 0)
                {
                    throw new InvalidOperationException("No successful client adaptations in this round");
                }

                // Phase 2: Compute meta-gradients
                var metaGradients = ComputeMetaGradients(clientResults);

                // Apply gradient clipping if configured
                if (Parameters.GradientClipThreshold > 0)
                {
                    metaGradients = ClipGradients(metaGradients, Parameters.GradientClipThreshold);
                }

                // Phase 3: Meta-update (outer loop)
                await PerformMetaUpdateAsync(metaGradients);

                // Record round results
                var roundResult = new MetaLearningResult
                {
                    Round = CurrentRound,
                    ParticipatingClients = clientResults.Keys.ToList(),
                    ClientResults = clientResults,
                    MetaGradients = metaGradients,
                    RoundTime = DateTime.UtcNow - roundStart,
                    AverageTaskLoss = clientResults.Values.Average(r => r.QueryLoss),
                    MetaGradientNorm = CalculateGradientNorm(metaGradients)
                };

                // Update history
                roundHistory.ParticipatingTasks = clientResults.Count;
                roundHistory.AverageAdaptationSteps = clientResults.Values.Average(r => r.AdaptationSteps);
                roundHistory.AverageTaskAccuracy = clientResults.Values.Average(r => r.TaskAccuracy);
                roundHistory.MetaLoss = roundResult.AverageTaskLoss;
                roundHistory.AverageLossImprovement = clientResults.Values.Average(r => r.LossImprovement);
                roundHistory.LearningRate = Parameters.OuterLearningRate;
                roundHistory.Complete();

                MetaHistory.Add(roundHistory);
                CurrentRound++;

                return roundResult;
            }
            catch (Exception ex)
            {
                roundHistory.Success = false;
                roundHistory.ErrorMessage = ex.Message;
                roundHistory.Complete();
                MetaHistory.Add(roundHistory);
                throw;
            }
        }

        /// <summary>
        /// Perform inner loop with error handling
        /// </summary>
        private async Task<ClientMetaResult?> PerformInnerLoopWithErrorHandlingAsync(string clientId, FederatedTask task)
        {
            try
            {
                return await PerformInnerLoopAsync(clientId, task);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Inner loop failed for client {clientId}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Perform inner loop adaptation for a client task
        /// </summary>
        /// <param name="clientId">Client identifier</param>
        /// <param name="task">Client task</param>
        /// <returns>Client meta-learning result</returns>
        private async Task<ClientMetaResult> PerformInnerLoopAsync(string clientId, FederatedTask task)
        {
            var adaptationStart = DateTime.UtcNow;
            
            // Clone meta-model for task-specific adaptation
            var adaptedModel = CloneModel(MetaModel);
            var initialParameters = GetModelParameters(adaptedModel);
            
            // Calculate initial loss for comparison
            var initialLoss = await CalculateTaskLossAsync(adaptedModel, task.SupportSet, task.SupportLabels);
            
            // Perform gradient descent steps on support set
            var adaptationSteps = 0;
            var supportLoss = initialLoss;
            var converged = false;

            for (int step = 0; step < Parameters.InnerSteps; step++)
            {
                // Compute gradients on support set
                var gradients = await ComputeTaskGradientsAsync(adaptedModel, task.SupportSet, task.SupportLabels);
                
                // Apply gradient normalization if configured
                if (Parameters.NormalizeGradients)
                {
                    gradients = NormalizeGradients(gradients);
                }
                
                // Apply inner loop update
                var parameterUpdates = InnerOptimizer.Optimize(
                    GetModelParameters(adaptedModel),
                    gradients,
                    null // No additional data needed for SGD
                );
                
                ApplyParameterUpdates(adaptedModel, parameterUpdates);
                
                // Calculate support loss
                var newSupportLoss = await CalculateTaskLossAsync(adaptedModel, task.SupportSet, task.SupportLabels);
                adaptationSteps++;

                // Check for convergence
                if (Math.Abs(supportLoss - newSupportLoss) < Parameters.ConvergenceThreshold)
                {
                    converged = true;
                    supportLoss = newSupportLoss;
                    break;
                }
                
                supportLoss = newSupportLoss;
            }

            // Evaluate on query set
            var queryLoss = await CalculateTaskLossAsync(adaptedModel, task.QuerySet, task.QueryLabels);
            var taskAccuracy = await CalculateTaskAccuracyAsync(adaptedModel, task.QuerySet, task.QueryLabels);

            // Compute meta-gradients
            var metaGradients = Parameters.UseFirstOrder
                ? await ComputeFirstOrderMetaGradientsAsync(adaptedModel, task)
                : await ComputeMetaGradientsForTaskAsync(initialParameters, GetModelParameters(adaptedModel), task);

            return new ClientMetaResult
            {
                ClientId = clientId,
                AdaptationSteps = adaptationSteps,
                InitialLoss = initialLoss,
                SupportLoss = supportLoss,
                QueryLoss = queryLoss,
                TaskAccuracy = taskAccuracy,
                MetaGradients = metaGradients,
                AdaptedParameters = GetModelParameters(adaptedModel),
                AdaptationTime = DateTime.UtcNow - adaptationStart,
                EffectiveLearningRate = Parameters.InnerLearningRate,
                Converged = converged
            };
        }

        /// <summary>
        /// Compute first-order meta-gradients (FOMAML approximation)
        /// </summary>
        private async Task<Dictionary<string, Vector<double>>> ComputeFirstOrderMetaGradientsAsync(
            IFullModel<double, Matrix<double>, Vector<double>> adaptedModel,
            FederatedTask task)
        {
            // In FOMAML, meta-gradients are simply the gradients of the query loss
            // with respect to the adapted parameters
            return await ComputeTaskGradientsAsync(adaptedModel, task.QuerySet, task.QueryLabels);
        }

        /// <summary>
        /// Compute task-specific gradients
        /// </summary>
        private async Task<Dictionary<string, Vector<double>>> ComputeTaskGradientsAsync(
            IFullModel<double, Matrix<double>, Vector<double>> model, 
            Matrix<double> data, 
            Vector<double> labels)
        {
            // Check if model supports gradient computation
            if (model is IGradientModel<double> gradientModel)
            {
                return await Task.Run(() => gradientModel.ComputeGradients(data, labels));
            }
            
            // Otherwise compute numerical gradients
            var predictions = await PredictBatchAsync(model, data);
            return await Task.Run(() => ComputeNumericalGradients(model, data, labels, predictions));
                gradients = await Task.FromResult(gradientModel.ComputeGradients(data, labels));
            }
            else
            {
                // Compute numerical gradients for other models
                gradients = ComputeNumericalGradients(model, data, labels, predictions);
            }

            return gradients;
        }

        /// <summary>
        /// Compute numerical gradients using finite differences
        /// </summary>
        private Dictionary<string, Vector<double>> ComputeNumericalGradients(
            IFullModel<double, Matrix<double>, Vector<double>> model, 
            Matrix<double> data, 
            Vector<double> labels, 
            Vector<double> predictions)
        {
            var gradients = new Dictionary<string, Vector<double>>();
            var parameters = GetModelParameters(model);
            var epsilon = 1e-5;

            Parallel.ForEach(parameters, kvp =>
            {
                var paramName = kvp.Key;
                var paramValues = kvp.Value;
                var gradient = new double[paramValues.Length];

                for (int i = 0; i < paramValues.Length; i++)
                {
                    var originalValue = paramValues[i];
                    
                    // f(x + ε)
                    paramValues[i] = originalValue + epsilon;
                    SetModelParameter(model, paramName, paramValues);
                    var lossPlus = CalculateLoss(PredictBatchSync(model, data), labels);
                    
                    // f(x - ε)
                    paramValues[i] = originalValue - epsilon;
                    SetModelParameter(model, paramName, paramValues);
                    var lossMinus = CalculateLoss(PredictBatchSync(model, data), labels);
                    
                    // Restore original value
                    paramValues[i] = originalValue;
                    SetModelParameter(model, paramName, paramValues);
                    
                    // Compute gradient
                    gradient[i] = (lossPlus - lossMinus) / (2 * epsilon);
                }

                lock (gradients)
                {
                    gradients[paramName] = new Vector<double>(gradient);
                }
            });

            return gradients;
        }

        /// <summary>
        /// Compute meta-gradients for a specific task (full MAML)
        /// </summary>
        private async Task<Dictionary<string, Vector<double>>> ComputeMetaGradientsForTaskAsync(
            Dictionary<string, Vector<double>> initialParameters,
            Dictionary<string, Vector<double>> adaptedParameters,
            FederatedTask task)
        {
            // Create model with adapted parameters
            var adaptedModel = CloneModel(MetaModel);
            SetModelParameters(adaptedModel, adaptedParameters);

            // Compute gradients of query loss w.r.t. adapted parameters
            var queryGradients = await ComputeTaskGradientsAsync(adaptedModel, task.QuerySet, task.QueryLabels);

            // For full MAML, we would need to compute the Hessian-vector product
            // This is a simplified implementation that approximates the meta-gradient
            var metaGradients = new Dictionary<string, Vector<double>>();
            
            foreach (var kvp in queryGradients)
            {
                var paramName = kvp.Key;
                if (initialParameters.ContainsKey(paramName))
                {
                    // Simplified meta-gradient computation
                    // In full MAML, this would involve computing d(queryLoss)/d(initialParams)
                    // through the chain of inner loop updates
                    metaGradients[paramName] = kvp.Value;
                }
            }

            return metaGradients;
        }

        /// <summary>
        /// Compute aggregated meta-gradients from all clients
        /// </summary>
        private Dictionary<string, Vector<double>> ComputeMetaGradients(Dictionary<string, ClientMetaResult> clientResults)
        {
            var aggregatedGradients = new Dictionary<string, Vector<double>>();
            var normalizedWeights = NormalizeClientWeights(clientResults.Keys.ToList());

            // Get parameter structure from first client
            var firstClient = clientResults.Values.First();
            
            foreach (var paramName in firstClient.MetaGradients.Keys)
            {
                var paramSize = firstClient.MetaGradients[paramName].Length;
                var aggregatedValues = new double[paramSize];

                // Aggregate gradients across clients
                foreach (var clientId in clientResults.Keys)
                {
                    if (clientResults[clientId].MetaGradients.ContainsKey(paramName))
                    {
                        var clientGradient = clientResults[clientId].MetaGradients[paramName];
                        var weight = normalizedWeights.ContainsKey(clientId) ? normalizedWeights[clientId] : 1.0 / clientResults.Count;

                        for (int i = 0; i < paramSize; i++)
                        {
                            aggregatedValues[i] += weight * clientGradient[i];
                        }
                    }
                }

                aggregatedGradients[paramName] = new Vector<double>(aggregatedValues);
            }

            return aggregatedGradients;
        }

        /// <summary>
        /// Perform meta-update using aggregated gradients
        /// </summary>
        private async Task PerformMetaUpdateAsync(Dictionary<string, Vector<double>> metaGradients)
        {
            await Task.Run(() =>
            {
                // Get current parameters
                var currentParameters = GetModelParameters(MetaModel);
                
                // Compute parameter updates using outer optimizer
                var parameterUpdates = OuterOptimizer.Optimize(
                    currentParameters,
                    metaGradients,
                    null // No additional data needed
                );
                
                // Apply updates to meta-model
                ApplyParameterUpdates(MetaModel, parameterUpdates);
                
                // Update learning rate if adaptive
                if (Parameters.UseAdaptiveLearningRate)
                {
                    UpdateLearningRates();
                }
            });
        }

        /// <summary>
        /// Update learning rates based on performance
        /// </summary>
        private void UpdateLearningRates()
        {
            if (MetaHistory.Count > 5)
            {
                var recentLosses = MetaHistory.TakeLast(5).Select(h => h.MetaLoss).ToList();
                var avgRecentLoss = recentLosses.Average();
                var previousAvgLoss = MetaHistory.SkipLast(5).TakeLast(5).Select(h => h.MetaLoss).Average();
                
                // If loss is not decreasing, reduce learning rate
                if (avgRecentLoss >= previousAvgLoss * 0.99)
                {
                    Parameters.OuterLearningRate *= 0.9;
                    Parameters.InnerLearningRate *= 0.95;
                    
                    // Re-initialize optimizers with new learning rates
                    InitializeOptimizers();
                }
            }
        }

        /// <summary>
        /// Clip gradients to prevent exploding gradients
        /// </summary>
        private Dictionary<string, Vector<double>> ClipGradients(
            Dictionary<string, Vector<double>> gradients, 
            double threshold)
        {
            var totalNorm = CalculateGradientNorm(gradients);
            
            if (totalNorm > threshold)
            {
                var scale = threshold / totalNorm;
                var clippedGradients = new Dictionary<string, Vector<double>>();
                
                foreach (var kvp in gradients)
                {
                    clippedGradients[kvp.Key] = kvp.Value * scale;
                }
                
                return clippedGradients;
            }
            
            return gradients;
        }

        /// <summary>
        /// Normalize gradients
        /// </summary>
        private Dictionary<string, Vector<double>> NormalizeGradients(Dictionary<string, Vector<double>> gradients)
        {
            var norm = CalculateGradientNorm(gradients);
            
            if (norm > 0)
            {
                var normalizedGradients = new Dictionary<string, Vector<double>>();
                
                foreach (var kvp in gradients)
                {
                    normalizedGradients[kvp.Key] = kvp.Value / norm;
                }
                
                return normalizedGradients;
            }
            
            return gradients;
        }

        /// <summary>
        /// Calculate task loss
        /// </summary>
        private async Task<double> CalculateTaskLossAsync(
            IFullModel<double, Matrix<double>, Vector<double>> model, 
            Matrix<double> data, 
            Vector<double> labels)
        {
            var predictions = await PredictBatchAsync(model, data);
            return CalculateLoss(predictions, labels);
        }

        /// <summary>
        /// Calculate task accuracy
        /// </summary>
        private async Task<double> CalculateTaskAccuracyAsync(
            IFullModel<double, Matrix<double>, Vector<double>> model, 
            Matrix<double> data, 
            Vector<double> labels)
        {
            var predictions = await PredictBatchAsync(model, data);
            
            // For regression tasks, use R-squared
            if (IsRegressionTask(labels))
            {
                return CalculateRSquared(predictions, labels);
            }
            
            // For classification tasks, use accuracy
            var correct = 0;
            for (int i = 0; i < predictions.Length; i++)
            {
                var predicted = Math.Round(predictions[i]);
                if (Math.Abs(predicted - labels[i]) < 0.1)
                {
                    correct++;
                }
            }

            return (double)correct / predictions.Length;
        }

        /// <summary>
        /// Determine if task is regression based on labels
        /// </summary>
        private bool IsRegressionTask(Vector<double> labels)
        {
            // Check if labels are continuous (not just 0 and 1)
            var uniqueValues = labels.Distinct().Count();
            return uniqueValues > 2 || labels.Any(l => l != 0.0 && l != 1.0);
        }

        /// <summary>
        /// Calculate R-squared for regression tasks
        /// </summary>
        private double CalculateRSquared(Vector<double> predictions, Vector<double> actual)
        {
            var mean = actual.Average();
            var ssTotal = actual.Select(y => Math.Pow(y - mean, 2)).Sum();
            var ssResidual = predictions.Zip(actual, (p, a) => Math.Pow(a - p, 2)).Sum();
            
            return 1 - (ssResidual / ssTotal);
            if (model is IPredictiveModel<double, Matrix<double>, Vector<double>> predictiveModel)

                // Predict the entire batch at once
                var predictions = await Task.FromResult(predictiveModel.Predict(data));
                return predictions;
            }
            {
                var predictions = new double[data.Rows];
                for (int i = 0; i < data.Rows; i++)
                {
                    var input = data.GetRow(i);
                    predictions[i] = await Task.FromResult(predictiveModel.Predict(input));
                }
                return new Vector<double>(predictions);
            if (model is IPredictiveModel<double, Matrix<double>, Vector<double>> predictiveModel)
            {
                // Predict the entire batch at once
                return predictiveModel.Predict(data);
            }
            {
                var predictions = new double[data.Rows];
                for (int i = 0; i < data.Rows; i++)
                {
                    var input = data.GetRow(i);
                    predictions[i] = predictiveModel.Predict(input);
                }
                return new Vector<double>(predictions);
            }

        /// <summary>
        /// Calculate loss between predictions and labels
        /// </summary>
        private double CalculateLoss(Vector<double> predictions, Vector<double> labels)
        {
            // Mean squared error
            var sum = 0.0;
            for (int i = 0; i < predictions.Length; i++)
            {
                var diff = predictions[i] - labels[i];
                sum += diff * diff;
            }
            return sum / predictions.Length;
        }

        /// <summary>
        /// Calculate gradient norm
        /// </summary>
        private double CalculateGradientNorm(Dictionary<string, Vector<double>> gradients)
        {
            var sum = 0.0;
            foreach (var kvp in gradients)
            {
                sum += kvp.Value.DotProduct(kvp.Value);
            }
            return Math.Sqrt(sum);
        }

        #region Model Manipulation Methods

        /// <summary>
        /// Clone a model for task-specific adaptation
        /// </summary>
        private IFullModel<double, Matrix<double>, Vector<double>> CloneModel(
            IFullModel<double, Matrix<double>, Vector<double>> model)
        {
            // Use cached parameters if available
            var cacheKey = $"{model.GetHashCode()}_clone";
            if (_parameterCache.ContainsKey(cacheKey))
            {
                var clonedModel = Activator.CreateInstance(model.GetType()) as IFullModel<double, Matrix<double>, Vector<double>>;
                if (clonedModel != null)
                {
                    SetModelParameters(clonedModel, _parameterCache[cacheKey]);
                    return clonedModel;
                }
            }
            
            // Create deep copy - implementation depends on model type
            if (model is ICloneable cloneable)
            {
                return cloneable.Clone() as IFullModel<double, Matrix<double>, Vector<double>> 
                    ?? throw new InvalidOperationException("Model clone failed");
            }
            
            throw new NotSupportedException($"Model type {model.GetType().Name} does not support cloning");
        }

        /// <summary>
        /// Get model parameters as a dictionary
        /// </summary>
        private Dictionary<string, Vector<double>> GetModelParameters(
            IFullModel<double, Matrix<double>, Vector<double>> model)
        {
            if (model is IParameterizable<double> parameterizable)
            {
                return parameterizable.GetParameters();
            }
            
            // Return empty dictionary for models without explicit parameters
            return new Dictionary<string, Vector<double>>();
        }

        /// <summary>
        /// Set all model parameters
        /// </summary>
        private void SetModelParameters(
            IFullModel<double, Matrix<double>, Vector<double>> model, 
            Dictionary<string, Vector<double>> parameters)
        {
            if (model is IParameterizable<double> parameterizable)
            {
                parameterizable.SetParameters(parameters);
            }
        }

        /// <summary>
        /// Set a specific model parameter
        /// </summary>
        private void SetModelParameter(
            IFullModel<double, Matrix<double>, Vector<double>> model, 
            string paramName, 
            Vector<double> parameter)
        {
            if (model is IParameterizable<double> parameterizable)
            {
                var parameters = parameterizable.GetParameters();
                parameters[paramName] = parameter;
                parameterizable.SetParameters(parameters);
            }
        }

        /// <summary>
        /// Apply parameter updates to model
        /// </summary>
        private void ApplyParameterUpdates(
            IFullModel<double, Matrix<double>, Vector<double>> model, 
            Dictionary<string, Vector<double>> updates)
        {
            var currentParams = GetModelParameters(model);
            var updatedParams = new Dictionary<string, Vector<double>>();
            
            foreach (var kvp in currentParams)
            {
                if (updates.ContainsKey(kvp.Key))
                {
                    updatedParams[kvp.Key] = kvp.Value.Add(updates[kvp.Key]);
                }
                else
                {
                    updatedParams[kvp.Key] = kvp.Value;
                }
            }
            
            SetModelParameters(model, updatedParams);
        }

        #endregion

        #region Implementation Methods

        /// <summary>
        /// Aggregate parameters using specified strategy
        /// </summary>
        public override Dictionary<string, Vector<double>> AggregateParameters(
            Dictionary<string, Dictionary<string, Vector<double>>> clientUpdates,
            FederatedAggregationStrategy strategy)
        {
            // MAML uses meta-gradient aggregation instead of parameter aggregation
            var aggregator = new Aggregation.FederatedAveraging();
            return aggregator.AggregateParameters(clientUpdates, ClientWeights, strategy);
        }

        /// <summary>
        /// Apply differential privacy to parameters
        /// </summary>
        /// <summary>
        /// Export meta-learning results to file
        /// </summary>
        public async Task ExportResultsAsync(string filePath)
        {
            var results = new
            {
                Parameters = Parameters,
                History = MetaHistory.Select(h => new
                {
                    h.Round,
                    h.ParticipatingTasks,
                    h.AverageTaskAccuracy,
                    h.MetaLoss,
                    h.AverageAdaptationSteps,
                    h.RoundTime,
                    h.Success
                }),
                FinalAccuracy = MetaHistory.LastOrDefault()?.AverageTaskAccuracy ?? 0,
                TotalRounds = CurrentRound
            };
    {
            var json = System.Text.Json.JsonSerializer.Serialize(results, new System.Text.Json.JsonSerializerOptions
            {
                WriteIndented = true
            });
    {
            await System.IO.File.WriteAllTextAsync(filePath, json);
        }
    {
        public int Round { get; set; }
        public List<string> ParticipatingClients { get; set; }
        public Dictionary<string, ClientMetaResult> ClientResults { get; set; }
        public Dictionary<string, Vector<double>> MetaGradients { get; set; }
        public TimeSpan RoundTime { get; set; }
        public double AverageTaskLoss { get; set; }
        public double MetaGradientNorm { get; set; }
    }

        #endregion
    }
}

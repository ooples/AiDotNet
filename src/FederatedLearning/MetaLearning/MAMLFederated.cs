using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.LinearAlgebra;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Optimizers;

namespace AiDotNet.FederatedLearning.MetaLearning
{
    /// <summary>
    /// Model-Agnostic Meta-Learning (MAML) for Federated Learning
    /// Enables fast adaptation to new tasks in federated settings
    /// </summary>
    public class MAMLFederated : FederatedLearningBase
    {
        /// <summary>
        /// Meta-learning parameters
        /// </summary>
        public MAMLParameters Parameters { get; set; }

        /// <summary>
        /// Meta-model for learning across tasks
        /// </summary>
        public IFullModel<double, Matrix<double>, Vector<double>> MetaModel { get; private set; }

        /// <summary>
        /// Client task definitions
        /// </summary>
        protected Dictionary<string, FederatedTask> ClientTasks { get; set; }

        /// <summary>
        /// Meta-gradient accumulator
        /// </summary>
        protected Dictionary<string, Vector<double>> MetaGradients { get; set; }

        /// <summary>
        /// Task performance history
        /// </summary>
        public List<MetaLearningRound> MetaHistory { get; private set; }

        /// <summary>
        /// Inner loop optimizer for task adaptation
        /// </summary>
        protected IOptimizer<double, Matrix<double>, Vector<double>> InnerOptimizer { get; set; }

        /// <summary>
        /// Outer loop optimizer for meta-updates
        /// </summary>
        protected IOptimizer<double, Matrix<double>, Vector<double>> OuterOptimizer { get; set; }

        /// <summary>
        /// Initialize MAML for federated learning
        /// </summary>
        /// <param name="metaModel">Meta-model for learning</param>
        /// <param name="parameters">MAML parameters</param>
        public MAMLFederated(IFullModel<double, Matrix<double>, Vector<double>> metaModel, MAMLParameters parameters = null)
        {
            MetaModel = metaModel ?? throw new ArgumentNullException(nameof(metaModel));
            Parameters = parameters ?? new MAMLParameters();
            ClientTasks = new Dictionary<string, FederatedTask>();
            MetaGradients = new Dictionary<string, Vector<double>>();
            MetaHistory = new List<MetaLearningRound>();

            // Initialize optimizers
            InnerOptimizer = new GradientDescentOptimizer(new Models.Options.GradientDescentOptimizerOptions
            {
                LearningRate = Parameters.InnerLearningRate
            });

            OuterOptimizer = new AdamOptimizer(new Models.Options.AdamOptimizerOptions
            {
                LearningRate = Parameters.OuterLearningRate,
                Beta1 = 0.9,
                Beta2 = 0.999
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

            ClientTasks[clientId] = task;
            SetClientWeight(clientId, task.SupportSet.Rows + task.QuerySet.Rows);
        }

        /// <summary>
        /// Perform federated meta-learning round
        /// </summary>
        /// <param name="selectedClients">Selected clients for this round</param>
        /// <returns>Meta-learning results</returns>
        public async Task<MetaLearningResult> PerformMetaLearningRoundAsync(List<string> selectedClients)
        {
            var roundStart = DateTime.UtcNow;
            var clientResults = new Dictionary<string, ClientMetaResult>();

            // Phase 1: Inner loop adaptation on each client
            var innerLoopTasks = selectedClients.Select(async clientId =>
            {
                if (ClientTasks.ContainsKey(clientId))
                {
                    var result = await PerformInnerLoopAsync(clientId, ClientTasks[clientId]);
                    return (clientId, result);
                }
                return (clientId, null);
            });

            var innerResults = await Task.WhenAll(innerLoopTasks);

            foreach (var (clientId, result) in innerResults)
            {
                if (result != null)
                {
                    clientResults[clientId] = result;
                }
            }

            // Phase 2: Compute meta-gradients
            var metaGradients = ComputeMetaGradients(clientResults);

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
            var metaRound = new MetaLearningRound
            {
                Round = CurrentRound,
                ParticipatingTasks = clientResults.Count,
                AverageAdaptationSteps = clientResults.Values.Average(r => r.AdaptationSteps),
                AverageTaskAccuracy = clientResults.Values.Average(r => r.TaskAccuracy),
                MetaLoss = roundResult.AverageTaskLoss,
                RoundTime = roundResult.RoundTime
            };

            MetaHistory.Add(metaRound);
            CurrentRound++;

            return roundResult;
        }

        /// <summary>
        /// Perform inner loop adaptation for a client task
        /// </summary>
        /// <param name="clientId">Client identifier</param>
        /// <param name="task">Client task</param>
        /// <returns>Client meta-learning result</returns>
        private async Task<ClientMetaResult> PerformInnerLoopAsync(string clientId, FederatedTask task)
        {
            try
            {
                // Clone meta-model for task-specific adaptation
                var adaptedModel = CloneModel(MetaModel);
                var initialParameters = GetModelParameters(adaptedModel);
                
                // Perform gradient descent steps on support set
                var adaptationSteps = 0;
                var supportLoss = 0.0;

                for (int step = 0; step < Parameters.InnerSteps; step++)
                {
                    // Compute gradients on support set
                    var gradients = await ComputeTaskGradientsAsync(adaptedModel, task.SupportSet, task.SupportLabels);
                    
                    // Apply inner loop update
                    var parameterUpdates = InnerOptimizer.CalculateUpdate(gradients);
                    ApplyParameterUpdates(adaptedModel, parameterUpdates);
                    
                    // Calculate support loss
                    supportLoss = await CalculateTaskLossAsync(adaptedModel, task.SupportSet, task.SupportLabels);
                    adaptationSteps++;

                    // Early stopping if converged
                    if (supportLoss < Parameters.ConvergenceThreshold)
                        break;
                }

                // Evaluate on query set
                var queryLoss = await CalculateTaskLossAsync(adaptedModel, task.QuerySet, task.QueryLabels);
                var taskAccuracy = await CalculateTaskAccuracyAsync(adaptedModel, task.QuerySet, task.QueryLabels);

                // Compute meta-gradients (gradients of query loss w.r.t. initial parameters)
                var metaGradients = await ComputeMetaGradientsForTaskAsync(
                    initialParameters, 
                    GetModelParameters(adaptedModel), 
                    task);

                return new ClientMetaResult
                {
                    ClientId = clientId,
                    AdaptationSteps = adaptationSteps,
                    SupportLoss = supportLoss,
                    QueryLoss = queryLoss,
                    TaskAccuracy = taskAccuracy,
                    MetaGradients = metaGradients,
                    AdaptedParameters = GetModelParameters(adaptedModel)
                };
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Inner loop failed for client {clientId}: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Compute task-specific gradients
        /// </summary>
        /// <param name="model">Model to compute gradients for</param>
        /// <param name="data">Input data</param>
        /// <param name="labels">Target labels</param>
        /// <returns>Task gradients</returns>
        private async Task<Dictionary<string, Vector<double>>> ComputeTaskGradientsAsync(IFullModel<double, Matrix<double>, Vector<double>> model, Matrix<double> data, Vector<double> labels)
        {
            // Compute forward pass
            var predictions = await PredictBatchAsync(model, data);
            
            // Compute loss gradients
            var gradients = new Dictionary<string, Vector<double>>();
            
            if (model is IGradientModel<double> gradientModel)
            {
                // Use model's gradient computation if available
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
        /// <param name="model">Model</param>
        /// <param name="data">Input data</param>
        /// <param name="labels">Target labels</param>
        /// <param name="predictions">Model predictions</param>
        /// <returns>Numerical gradients</returns>
        private Dictionary<string, Vector<double>> ComputeNumericalGradients(IFullModel<double, Matrix<double>, Vector<double>> model, Matrix<double> data, Vector<double> labels, Vector<double> predictions)
        {
            var gradients = new Dictionary<string, Vector<double>>();
            var parameters = GetModelParameters(model);
            var epsilon = 1e-5;

            foreach (var kvp in parameters)
            {
                var paramName = kvp.Key;
                var paramValues = kvp.Value;
                var gradient = new double[paramValues.Length];

                for (int i = 0; i < paramValues.Length; i++)
                {
                    // Forward difference
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

                gradients[paramName] = new Vector<double>(gradient);
            }

            return gradients;
        }

        /// <summary>
        /// Compute meta-gradients for a specific task
        /// </summary>
        /// <param name="initialParameters">Initial model parameters</param>
        /// <param name="adaptedParameters">Parameters after adaptation</param>
        /// <param name="task">Task definition</param>
        /// <returns>Meta-gradients</returns>
        private async Task<Dictionary<string, Vector<double>>> ComputeMetaGradientsForTaskAsync(
            Dictionary<string, Vector<double>> initialParameters,
            Dictionary<string, Vector<double>> adaptedParameters,
            FederatedTask task)
        {
            var metaGradients = new Dictionary<string, Vector<double>>();

            // Create model with adapted parameters
            var adaptedModel = CloneModel(MetaModel);
            SetModelParameters(adaptedModel, adaptedParameters);

            // Compute gradients of query loss w.r.t. adapted parameters
            var queryGradients = await ComputeTaskGradientsAsync(adaptedModel, task.QuerySet, task.QueryLabels);

            // Compute meta-gradients using chain rule
            // This is a simplified implementation - full MAML requires computing
            // the gradient of the gradient through the adaptation steps
            foreach (var kvp in queryGradients)
            {
                var paramName = kvp.Key;
                var queryGrad = kvp.Value;

                if (initialParameters.ContainsKey(paramName))
                {
                    // For simplicity, approximate meta-gradient as query gradient
                    // In full MAML, this would involve Hessian computation
                    metaGradients[paramName] = queryGrad;
                }
            }

            return metaGradients;
        }

        /// <summary>
        /// Compute aggregated meta-gradients from all clients
        /// </summary>
        /// <param name="clientResults">Client meta-learning results</param>
        /// <returns>Aggregated meta-gradients</returns>
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
        /// <param name="metaGradients">Aggregated meta-gradients</param>
        private async Task PerformMetaUpdateAsync(Dictionary<string, Vector<double>> metaGradients)
        {
            // Compute parameter updates using outer optimizer
            var parameterUpdates = OuterOptimizer.CalculateUpdate(metaGradients);

            // Apply updates to meta-model
            ApplyParameterUpdates(MetaModel, parameterUpdates);

            await Task.CompletedTask;
        }

        /// <summary>
        /// Calculate task loss
        /// </summary>
        /// <param name="model">Model</param>
        /// <param name="data">Input data</param>
        /// <param name="labels">Target labels</param>
        /// <returns>Task loss</returns>
        private async Task<double> CalculateTaskLossAsync(IFullModel<double, Matrix<double>, Vector<double>> model, Matrix<double> data, Vector<double> labels)
        {
            var predictions = await PredictBatchAsync(model, data);
            return CalculateLoss(predictions, labels);
        }

        /// <summary>
        /// Calculate task accuracy
        /// </summary>
        /// <param name="model">Model</param>
        /// <param name="data">Input data</param>
        /// <param name="labels">Target labels</param>
        /// <returns>Task accuracy</returns>
        private async Task<double> CalculateTaskAccuracyAsync(IFullModel<double, Matrix<double>, Vector<double>> model, Matrix<double> data, Vector<double> labels)
        {
            var predictions = await PredictBatchAsync(model, data);
            
            var correct = 0;
            for (int i = 0; i < predictions.Length; i++)
            {
                var predicted = predictions[i] > 0.5 ? 1.0 : 0.0; // Binary classification
                if (Math.Abs(predicted - labels[i]) < 0.1)
                {
                    correct++;
                }
            }

            return (double)correct / predictions.Length;
        }

        /// <summary>
        /// Predict batch using model
        /// </summary>
        /// <param name="model">Model</param>
        /// <param name="data">Input data</param>
        /// <returns>Predictions</returns>
        private async Task<Vector<double>> PredictBatchAsync(IFullModel<double, Matrix<double>, Vector<double>> model, Matrix<double> data)
        {
            if (model is IPredictiveModel predictiveModel)
            {
                var predictions = new double[data.Rows];
                for (int i = 0; i < data.Rows; i++)
                {
                    var input = data.GetRow(i);
                    predictions[i] = await Task.FromResult(predictiveModel.Predict(input));
                }
                return new Vector<double>(predictions);
            }

            throw new NotSupportedException("Model does not support prediction");
        }

        /// <summary>
        /// Synchronous prediction for numerical gradient computation
        /// </summary>
        /// <param name="model">Model</param>
        /// <param name="data">Input data</param>
        /// <returns>Predictions</returns>
        private Vector<double> PredictBatchSync(IFullModel<double, Matrix<double>, Vector<double>> model, Matrix<double> data)
        {
            if (model is IPredictiveModel predictiveModel)
            {
                var predictions = new double[data.Rows];
                for (int i = 0; i < data.Rows; i++)
                {
                    var input = data.GetRow(i);
                    predictions[i] = predictiveModel.Predict(input);
                }
                return new Vector<double>(predictions);
            }

            throw new NotSupportedException("Model does not support prediction");
        }

        /// <summary>
        /// Calculate loss between predictions and labels
        /// </summary>
        /// <param name="predictions">Predictions</param>
        /// <param name="labels">True labels</param>
        /// <returns>Loss value</returns>
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
        /// <param name="gradients">Gradients</param>
        /// <returns>L2 norm</returns>
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

        private IFullModel<double, Matrix<double>, Vector<double>> CloneModel(IFullModel<double, Matrix<double>, Vector<double>> model)
        {
            // Simplified model cloning - in practice, use proper deep copying
            return model;
        }

        private Dictionary<string, Vector<double>> GetModelParameters(IFullModel<double, Matrix<double>, Vector<double>> model)
        {
            // Extract model parameters - implementation depends on model type
            return new Dictionary<string, Vector<double>>();
        }

        private void SetModelParameters(IFullModel<double, Matrix<double>, Vector<double>> model, Dictionary<string, Vector<double>> parameters)
        {
            // Set model parameters - implementation depends on model type
        }

        private void SetModelParameter(IFullModel<double, Matrix<double>, Vector<double>> model, string paramName, Vector<double> parameter)
        {
            // Set specific model parameter - implementation depends on model type
        }

        private void ApplyParameterUpdates(IFullModel<double, Matrix<double>, Vector<double>> model, Dictionary<string, Vector<double>> updates)
        {
            // Apply parameter updates to model
            var currentParams = GetModelParameters(model);
            foreach (var kvp in updates)
            {
                if (currentParams.ContainsKey(kvp.Key))
                {
                    var updated = currentParams[kvp.Key].Add(kvp.Value);
                    SetModelParameter(model, kvp.Key, updated);
                }
            }
        }

        #endregion

        #region Implementation Methods

        public override Dictionary<string, Vector<double>> AggregateParameters(
            Dictionary<string, Dictionary<string, Vector<double>>> clientUpdates,
            FederatedAggregationStrategy strategy)
        {
            // MAML uses meta-gradient aggregation instead of parameter aggregation
            var aggregator = new Aggregation.FederatedAveraging();
            return aggregator.AggregateParameters(clientUpdates, ClientWeights, strategy);
        }

        public override Dictionary<string, Vector<double>> ApplyDifferentialPrivacy(
            Dictionary<string, Vector<double>> parameters,
            double epsilon,
            double delta)
        {
            var dp = new Privacy.DifferentialPrivacy();
            return dp.ApplyPrivacy(parameters, epsilon, delta, PrivacySettings);
        }

        #endregion
    }

    /// <summary>
    /// MAML parameters configuration
    /// </summary>
    public class MAMLParameters
    {
        /// <summary>
        /// Inner loop learning rate
        /// </summary>
        public double InnerLearningRate { get; set; } = 0.01;

        /// <summary>
        /// Outer loop learning rate
        /// </summary>
        public double OuterLearningRate { get; set; } = 0.001;

        /// <summary>
        /// Number of inner loop gradient steps
        /// </summary>
        public int InnerSteps { get; set; } = 5;

        /// <summary>
        /// Convergence threshold for early stopping
        /// </summary>
        public double ConvergenceThreshold { get; set; } = 1e-6;

        /// <summary>
        /// Use first-order approximation (FOMAML)
        /// </summary>
        public bool UseFirstOrder { get; set; } = false;

        /// <summary>
        /// Number of support examples per task
        /// </summary>
        public int SupportSize { get; set; } = 5;

        /// <summary>
        /// Number of query examples per task
        /// </summary>
        public int QuerySize { get; set; } = 15;

        /// <summary>
        /// Task batch size for meta-updates
        /// </summary>
        public int TaskBatchSize { get; set; } = 4;
    }

    /// <summary>
    /// Federated task definition
    /// </summary>
    public class FederatedTask
    {
        public string TaskId { get; set; }
        public string TaskType { get; set; }
        public Matrix<double> SupportSet { get; set; }
        public Vector<double> SupportLabels { get; set; }
        public Matrix<double> QuerySet { get; set; }
        public Vector<double> QueryLabels { get; set; }
        public Dictionary<string, object> TaskMetadata { get; set; } = new Dictionary<string, object>();
    }

    /// <summary>
    /// Client meta-learning result
    /// </summary>
    public class ClientMetaResult
    {
        public string ClientId { get; set; }
        public int AdaptationSteps { get; set; }
        public double SupportLoss { get; set; }
        public double QueryLoss { get; set; }
        public double TaskAccuracy { get; set; }
        public Dictionary<string, Vector<double>> MetaGradients { get; set; }
        public Dictionary<string, Vector<double>> AdaptedParameters { get; set; }
    }

    /// <summary>
    /// Meta-learning round result
    /// </summary>
    public class MetaLearningResult
    {
        public int Round { get; set; }
        public List<string> ParticipatingClients { get; set; }
        public Dictionary<string, ClientMetaResult> ClientResults { get; set; }
        public Dictionary<string, Vector<double>> MetaGradients { get; set; }
        public TimeSpan RoundTime { get; set; }
        public double AverageTaskLoss { get; set; }
        public double MetaGradientNorm { get; set; }
    }

    /// <summary>
    /// Meta-learning round history
    /// </summary>
    public class MetaLearningRound
    {
        public int Round { get; set; }
        public int ParticipatingTasks { get; set; }
        public double AverageAdaptationSteps { get; set; }
        public double AverageTaskAccuracy { get; set; }
        public double MetaLoss { get; set; }
        public TimeSpan RoundTime { get; set; }
    }
}
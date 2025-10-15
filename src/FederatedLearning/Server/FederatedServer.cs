using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.LinearAlgebra;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Enums;
using AiDotNet.FederatedLearning.Aggregation;
using AiDotNet.FederatedLearning.Privacy;
using AiDotNet.FederatedLearning.Communication;
using AiDotNet.FederatedLearning.Communication.Interfaces;

namespace AiDotNet.FederatedLearning.Server
{
    /// <summary>
    /// Federated learning server for coordinating distributed training
    /// </summary>
    public class FederatedServer : FederatedLearningBase
    {
        /// <summary>
        /// Server identifier
        /// </summary>
        public string ServerId { get; private set; }

        /// <summary>
        /// Connected clients
        /// </summary>
        protected Dictionary<string, ClientInfo> ConnectedClients { get; set; }

        /// <summary>
        /// Parameter aggregator
        /// </summary>
        protected IParameterAggregator ParameterAggregator { get; set; }

        /// <summary>
        /// Differential privacy handler
        /// </summary>
        protected IDifferentialPrivacy DifferentialPrivacy { get; set; }

        /// <summary>
        /// Communication manager
        /// </summary>
        protected ICommunicationManager CommunicationManager { get; set; }

        /// <summary>
        /// Server training history
        /// </summary>
        public List<ServerRound> TrainingHistory { get; private set; }

        /// <summary>
        /// Server status
        /// </summary>
        public ServerStatus Status { get; private set; }

        /// <summary>
        /// Convergence threshold for early stopping
        /// </summary>
        public double ConvergenceThreshold { get; set; }

        /// <summary>
        /// Maximum wait time for client responses
        /// </summary>
        public TimeSpan ClientTimeout { get; set; }

        /// <summary>
        /// Initialize federated server
        /// </summary>
        /// <param name="serverId">Unique server identifier</param>
        public FederatedServer(string serverId)
        {
            ServerId = serverId ?? throw new ArgumentNullException(nameof(serverId));
            ConnectedClients = new Dictionary<string, ClientInfo>();
            TrainingHistory = new List<ServerRound>();
            Status = ServerStatus.Initializing;
            ConvergenceThreshold = 1e-6;
            ClientTimeout = TimeSpan.FromMinutes(10);

            // Initialize components
            ParameterAggregator = new FederatedAveraging();
            DifferentialPrivacy = new DifferentialPrivacy();
            CommunicationManager = new CommunicationManager();

            Status = ServerStatus.Ready;
        }

        /// <summary>
        /// Register a new client with the server
        /// </summary>
        /// <param name="clientId">Client identifier</param>
        /// <param name="clientInfo">Client information</param>
        public void RegisterClient(string clientId, ClientInfo clientInfo)
        {
            if (string.IsNullOrEmpty(clientId))
                throw new ArgumentNullException(nameof(clientId));
            
            if (clientInfo == null)
                throw new ArgumentNullException(nameof(clientInfo));

            ConnectedClients[clientId] = clientInfo;
            clientInfo.Status = ClientConnectionStatus.Connected;
            clientInfo.RegistrationTime = DateTime.UtcNow;
        }

        /// <summary>
        /// Unregister a client from the server
        /// </summary>
        /// <param name="clientId">Client identifier</param>
        public void UnregisterClient(string clientId)
        {
            if (ConnectedClients.ContainsKey(clientId))
            {
                ConnectedClients[clientId].Status = ClientConnectionStatus.Disconnected;
                ConnectedClients.Remove(clientId);
            }
        }

        /// <summary>
        /// Start federated learning process
        /// </summary>
        /// <returns>Training results</returns>
        public async Task<FederatedTrainingResult> StartFederatedLearningAsync()
        {
            try
            {
                Status = ServerStatus.Training;
                var trainingStart = DateTime.UtcNow;
                var convergenceHistory = new List<double>();

                while (ShouldContinueTraining())
                {
                    var roundResult = await ExecuteTrainingRoundAsync();
                    
                    // Check for convergence
                    convergenceHistory.Add(roundResult.ConvergenceMetric);
                    if (HasConverged(convergenceHistory))
                    {
                        break;
                    }

                    // Update global model
                    UpdateGlobalModel(roundResult.AggregatedParameters);
                }

                Status = ServerStatus.Completed;
                var trainingTime = DateTime.UtcNow - trainingStart;

                return new FederatedTrainingResult
                {
                    TotalRounds = CurrentRound,
                    TrainingTime = trainingTime,
                    FinalGlobalParameters = GetGlobalParameters(),
                    ConvergenceHistory = convergenceHistory,
                    TrainingHistory = TrainingHistory.ToList()
                };
            }
            catch (Exception ex)
            {
                Status = ServerStatus.Error;
                throw new InvalidOperationException($"Federated learning failed: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Execute a single training round
        /// </summary>
        /// <returns>Round results</returns>
        private async Task<RoundResult> ExecuteTrainingRoundAsync()
        {
            var roundStart = DateTime.UtcNow;
            
            // Select clients for this round
            var availableClients = GetAvailableClients();
            var selectedClients = SelectClientsForRound(availableClients);

            if (selectedClients.Count < MinimumClients)
            {
                throw new InvalidOperationException($"Insufficient clients for training. Required: {MinimumClients}, Available: {selectedClients.Count}");
            }

            // Send global model to selected clients
            await BroadcastGlobalModelAsync(selectedClients);

            // Collect client updates
            var clientUpdates = await CollectClientUpdatesAsync(selectedClients);

            // Aggregate parameters
            var aggregatedParameters = AggregateParameters(clientUpdates, FederatedAggregationStrategy.FederatedAveraging);

            // Apply differential privacy if enabled
            if (PrivacySettings.UseDifferentialPrivacy)
            {
                aggregatedParameters = ApplyDifferentialPrivacy(
                    aggregatedParameters,
                    PrivacySettings.Epsilon,
                    PrivacySettings.Delta);
            }

            // Calculate convergence metric
            var convergenceMetric = CalculateConvergenceMetric(aggregatedParameters);

            // Record round results
            var roundResult = new RoundResult
            {
                Round = CurrentRound,
                SelectedClients = selectedClients,
                ParticipatingClients = clientUpdates.Keys.ToList(),
                AggregatedParameters = aggregatedParameters,
                ConvergenceMetric = convergenceMetric,
                RoundTime = DateTime.UtcNow - roundStart
            };

            var serverRound = new ServerRound
            {
                Round = CurrentRound,
                ParticipatingClients = clientUpdates.Count,
                ConvergenceMetric = convergenceMetric,
                RoundTime = roundResult.RoundTime,
                GlobalParameterNorm = CalculateParameterNorm(aggregatedParameters)
            };

            TrainingHistory.Add(serverRound);
            return roundResult;
        }

        /// <summary>
        /// Get list of available clients
        /// </summary>
        /// <returns>Available client IDs</returns>
        private List<string> GetAvailableClients()
        {
            return ConnectedClients
                .Where(kvp => kvp.Value.Status == ClientConnectionStatus.Connected)
                .Select(kvp => kvp.Key)
                .ToList();
        }

        /// <summary>
        /// Broadcast global model to selected clients
        /// </summary>
        /// <param name="selectedClients">Selected client IDs</param>
        private async Task BroadcastGlobalModelAsync(List<string> selectedClients)
        {
            var globalParameters = GetGlobalParameters();
            var tasks = new List<Task>();

            foreach (var clientId in selectedClients)
            {
                var task = CommunicationManager.SendGlobalModelAsync(clientId, globalParameters);
                tasks.Add(task);
            }

            await Task.WhenAll(tasks);
        }

        /// <summary>
        /// Collect parameter updates from clients
        /// </summary>
        /// <param name="selectedClients">Selected client IDs</param>
        /// <returns>Client parameter updates</returns>
        private async Task<Dictionary<string, Dictionary<string, Vector<double>>>> CollectClientUpdatesAsync(List<string> selectedClients)
        {
            var clientUpdates = new Dictionary<string, Dictionary<string, Vector<double>>>();
            var tasks = new Dictionary<string, Task<Dictionary<string, Vector<double>>?>>();

            // Start collection from all clients
            foreach (var clientId in selectedClients)
            {
                var task = CommunicationManager.ReceiveClientUpdateAsync(clientId, ClientTimeout);
                tasks[clientId] = task;
            }

            // Wait for responses with timeout
            var completedTasks = await Task.WhenAll(tasks.Values);

            // Collect successful responses
            var index = 0;
            foreach (var clientId in selectedClients)
            {
                try
                {
                    var update = completedTasks[index];
                    if (update != null && update.Count > 0)
                    {
                        clientUpdates[clientId] = update;
                    }
                }
                catch (Exception ex)
                {
                    // Log client failure and continue
                    Console.WriteLine($"Failed to receive update from client {clientId}: {ex.Message}");
                    ConnectedClients[clientId].Status = ClientConnectionStatus.Error;
                }
                index++;
            }

            return clientUpdates;
        }

        /// <summary>
        /// Calculate convergence metric based on parameter changes
        /// </summary>
        /// <param name="newParameters">New parameters</param>
        /// <returns>Convergence metric</returns>
        private double CalculateConvergenceMetric(Dictionary<string, Vector<double>> newParameters)
        {
            if (GlobalParameters.Count == 0)
                return double.MaxValue;

            var totalChange = 0.0;
            var totalNorm = 0.0;

            foreach (var kvp in newParameters)
            {
                if (GlobalParameters.ContainsKey(kvp.Key))
                {
                    var difference = kvp.Value.Subtract(GlobalParameters[kvp.Key]);
                    var changeNorm = Math.Sqrt(difference.DotProduct(difference));
                    var paramNorm = Math.Sqrt(kvp.Value.DotProduct(kvp.Value));

                    totalChange += changeNorm;
                    totalNorm += paramNorm;
                }
            }

            return totalNorm > 0 ? totalChange / totalNorm : 0.0;
        }

        /// <summary>
        /// Check if training has converged
        /// </summary>
        /// <param name="convergenceHistory">History of convergence metrics</param>
        /// <returns>True if converged</returns>
        private bool HasConverged(List<double> convergenceHistory)
        {
            if (convergenceHistory.Count < 3)
                return false;

            // TakeLast is not available in older .NET versions, use Skip instead
            var count = convergenceHistory.Count;
            var recent = convergenceHistory.Skip(Math.Max(0, count - 3)).ToList();
            return recent.All(x => x < ConvergenceThreshold);
        }

        /// <summary>
        /// Calculate L2 norm of parameters
        /// </summary>
        /// <param name="parameters">Parameters</param>
        /// <returns>L2 norm</returns>
        private double CalculateParameterNorm(Dictionary<string, Vector<double>> parameters)
        {
            var sum = 0.0;
            foreach (var kvp in parameters)
            {
                sum += kvp.Value.DotProduct(kvp.Value);
            }
            return Math.Sqrt(sum);
        }

        #region Implementation Methods

        public override Dictionary<string, Vector<double>> AggregateParameters(
            Dictionary<string, Dictionary<string, Vector<double>>> clientUpdates,
            FederatedAggregationStrategy strategy)
        {
            return ParameterAggregator.AggregateParameters(clientUpdates, ClientWeights, strategy);
        }

        public override Dictionary<string, Vector<double>> ApplyDifferentialPrivacy(
            Dictionary<string, Vector<double>> parameters,
            double epsilon,
            double delta)
        {
            // Convert FederatedLearning.PrivacySettings to Privacy.PrivacySettings if needed
            var privacySettings = new AiDotNet.FederatedLearning.Privacy.PrivacySettings
            {
                UseDifferentialPrivacy = PrivacySettings.UseDifferentialPrivacy,
                Epsilon = PrivacySettings.Epsilon,
                Delta = PrivacySettings.Delta,
                ClippingThreshold = PrivacySettings.ClippingThreshold
            };
            return DifferentialPrivacy.ApplyPrivacy(parameters, epsilon, delta, privacySettings);
        }

        #endregion

        /// <summary>
        /// Get server statistics
        /// </summary>
        /// <returns>Server statistics</returns>
        public ServerStatistics GetStatistics()
        {
            return new ServerStatistics
            {
                ServerId = ServerId,
                CurrentRound = CurrentRound,
                TotalConnectedClients = ConnectedClients.Count,
                ActiveClients = ConnectedClients.Values.Count(c => c.Status == ClientConnectionStatus.Connected),
                TotalRounds = TotalRounds,
                Status = Status,
                AverageRoundTime = TrainingHistory.Count > 0 ? 
                    TimeSpan.FromTicks((long)TrainingHistory.Average(r => r.RoundTime.Ticks)) : 
                    TimeSpan.Zero
            };
        }
    }
}

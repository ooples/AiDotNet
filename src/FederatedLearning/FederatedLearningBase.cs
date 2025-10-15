using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.LinearAlgebra;
using AiDotNet.Interfaces;
using AiDotNet.Enums;
using AiDotNet.Models;

namespace AiDotNet.FederatedLearning
{
    /// <summary>
    /// Base class for federated learning implementations providing common functionality
    /// for secure and private distributed machine learning
    /// </summary>
    public abstract class FederatedLearningBase
    {
        /// <summary>
        /// Current round of federated learning
        /// </summary>
        protected int CurrentRound { get; set; }

        /// <summary>
        /// Total number of federated learning rounds
        /// </summary>
        protected int TotalRounds { get; set; }

        /// <summary>
        /// Minimum number of clients required for aggregation
        /// </summary>
        protected int MinimumClients { get; set; }

        /// <summary>
        /// Privacy budget for differential privacy
        /// </summary>
        protected double PrivacyBudget { get; set; }

        /// <summary>
        /// Learning rate for federated optimization
        /// </summary>
        protected double LearningRate { get; set; }

        /// <summary>
        /// Client selection probability
        /// </summary>
        protected double ClientSamplingRate { get; set; }

        /// <summary>
        /// Global model parameters
        /// </summary>
        protected Dictionary<string, Vector<double>> GlobalParameters { get; set; }

        /// <summary>
        /// Local client parameters
        /// </summary>
        protected Dictionary<string, Dictionary<string, Vector<double>>> ClientParameters { get; set; }

        /// <summary>
        /// Client weights for aggregation
        /// </summary>
        protected Dictionary<string, double> ClientWeights { get; set; }

        /// <summary>
        /// Security settings for federated learning
        /// </summary>
        protected FederatedSecuritySettings SecuritySettings { get; set; }

        /// <summary>
        /// Privacy settings for differential privacy
        /// </summary>
        protected Privacy.PrivacySettings PrivacySettings { get; set; }

        /// <summary>
        /// Communication settings for client-server interaction
        /// </summary>
        protected CommunicationSettings CommunicationSettings { get; set; }

        /// <summary>
        /// Initialize federated learning base
        /// </summary>
        protected FederatedLearningBase()
        {
            CurrentRound = 0;
            TotalRounds = 100;
            MinimumClients = 2;
            PrivacyBudget = 1.0;
            LearningRate = 0.01;
            ClientSamplingRate = 0.1;
            GlobalParameters = new Dictionary<string, Vector<double>>();
            ClientParameters = new Dictionary<string, Dictionary<string, Vector<double>>>();
            ClientWeights = new Dictionary<string, double>();
            SecuritySettings = new FederatedSecuritySettings();
            PrivacySettings = new Privacy.PrivacySettings();
            CommunicationSettings = new CommunicationSettings();
        }

        /// <summary>
        /// Initialize global model parameters
        /// </summary>
        /// <param name="parameterShapes">Parameter shapes for initialization</param>
        public virtual void InitializeGlobalModel(Dictionary<string, int[]> parameterShapes)
        {
            var random = new Random();
            foreach (var kvp in parameterShapes)
            {
                var shape = kvp.Value;
                var size = 1;
                foreach (var dim in shape)
                {
                    size *= dim;
                }

                var data = new double[size];
                for (int i = 0; i < size; i++)
                {
                    data[i] = random.NextGaussian() * 0.01; // Xavier initialization
                }

                GlobalParameters[kvp.Key] = new Vector<double>(data);
            }
        }

        /// <summary>
        /// Select clients for current round
        /// </summary>
        /// <param name="availableClients">List of available client IDs</param>
        /// <returns>Selected client IDs</returns>
        public virtual List<string> SelectClientsForRound(List<string> availableClients)
        {
            var random = new Random();
            var selectedClients = new List<string>();
            
            foreach (var clientId in availableClients)
            {
                if (random.NextDouble() < ClientSamplingRate)
                {
                    selectedClients.Add(clientId);
                }
            }

            // Ensure minimum number of clients
            while (selectedClients.Count < MinimumClients && selectedClients.Count < availableClients.Count)
            {
                var remainingClients = availableClients.FindAll(c => !selectedClients.Contains(c));
                if (remainingClients.Count > 0)
                {
                    selectedClients.Add(remainingClients[random.Next(remainingClients.Count)]);
                }
                else
                {
                    break;
                }
            }

            return selectedClients;
        }

        /// <summary>
        /// Aggregate client parameters using specified strategy
        /// </summary>
        /// <param name="clientUpdates">Client parameter updates</param>
        /// <param name="strategy">Aggregation strategy</param>
        /// <returns>Aggregated parameters</returns>
        public abstract Dictionary<string, Vector<double>> AggregateParameters(
            Dictionary<string, Dictionary<string, Vector<double>>> clientUpdates,
            FederatedAggregationStrategy strategy);

        /// <summary>
        /// Apply differential privacy to parameters
        /// </summary>
        /// <param name="parameters">Parameters to privatize</param>
        /// <param name="epsilon">Privacy parameter</param>
        /// <param name="delta">Privacy parameter</param>
        /// <returns>Privatized parameters</returns>
        public abstract Dictionary<string, Vector<double>> ApplyDifferentialPrivacy(
            Dictionary<string, Vector<double>> parameters,
            double epsilon,
            double delta);

        /// <summary>
        /// Update global model with aggregated parameters
        /// </summary>
        /// <param name="aggregatedParameters">Aggregated parameters from clients</param>
        public virtual void UpdateGlobalModel(Dictionary<string, Vector<double>> aggregatedParameters)
        {
            foreach (var kvp in aggregatedParameters)
            {
                if (GlobalParameters.ContainsKey(kvp.Key))
                {
                    GlobalParameters[kvp.Key] = kvp.Value;
                }
            }
            CurrentRound++;
        }

        /// <summary>
        /// Check if federated learning should continue
        /// </summary>
        /// <returns>True if should continue training</returns>
        public virtual bool ShouldContinueTraining()
        {
            return CurrentRound < TotalRounds;
        }

        /// <summary>
        /// Get current global model parameters
        /// </summary>
        /// <returns>Global model parameters</returns>
        public virtual Dictionary<string, Vector<double>> GetGlobalParameters()
        {
            return new Dictionary<string, Vector<double>>(GlobalParameters);
        }

        /// <summary>
        /// Calculate client contribution weight based on data size
        /// </summary>
        /// <param name="clientId">Client identifier</param>
        /// <param name="dataSize">Size of client's training data</param>
        public virtual void SetClientWeight(string clientId, int dataSize)
        {
            ClientWeights[clientId] = dataSize;
        }

        /// <summary>
        /// Normalize client weights for aggregation
        /// </summary>
        /// <param name="selectedClients">Selected clients for current round</param>
        /// <returns>Normalized weights</returns>
        protected virtual Dictionary<string, double> NormalizeClientWeights(List<string> selectedClients)
        {
            var normalizedWeights = new Dictionary<string, double>();
            var totalWeight = 0.0;

            foreach (var clientId in selectedClients)
            {
                if (ClientWeights.ContainsKey(clientId))
                {
                    totalWeight += ClientWeights[clientId];
                }
            }

            if (totalWeight > 0)
            {
                foreach (var clientId in selectedClients)
                {
                    if (ClientWeights.ContainsKey(clientId))
                    {
                        normalizedWeights[clientId] = ClientWeights[clientId] / totalWeight;
                    }
                    else
                    {
                        normalizedWeights[clientId] = 1.0 / selectedClients.Count;
                    }
                }
            }
            else
            {
                // Equal weights if no data size information
                foreach (var clientId in selectedClients)
                {
                    normalizedWeights[clientId] = 1.0 / selectedClients.Count;
                }
            }

            return normalizedWeights;
        }
    }

    /// <summary>
    /// Federated aggregation strategies
    /// </summary>
    public enum FederatedAggregationStrategy
    {
        FederatedAveraging,
        SecureAggregation,
        WeightedAveraging,
        MedianAggregation,
        TrimmedMean,
        Byzantine,
        FederatedProx
    }
}
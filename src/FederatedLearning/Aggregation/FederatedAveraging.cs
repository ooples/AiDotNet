using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.FederatedLearning.Aggregation
{
    /// <summary>
    /// Federated Averaging (FedAvg) algorithm implementation for parameter aggregation
    /// </summary>
    public class FederatedAveraging : IParameterAggregator
    {
        /// <summary>
        /// Aggregation strategy type
        /// </summary>
        public FederatedAggregationStrategy Strategy => FederatedAggregationStrategy.FederatedAveraging;

        /// <summary>
        /// Aggregate parameters from multiple clients using FedAvg algorithm
        /// </summary>
        /// <param name="clientUpdates">Client parameter updates</param>
        /// <param name="clientWeights">Client weights for aggregation</param>
        /// <param name="strategy">Aggregation strategy</param>
        /// <returns>Aggregated parameters</returns>
        public Dictionary<string, Vector<double>> AggregateParameters(
            Dictionary<string, Dictionary<string, Vector<double>>> clientUpdates,
            Dictionary<string, double> clientWeights,
            FederatedAggregationStrategy strategy)
        {
            if (clientUpdates == null || clientUpdates.Count == 0)
                throw new ArgumentException("Client updates cannot be null or empty");

            switch (strategy)
            {
                case FederatedAggregationStrategy.FederatedAveraging:
                    return PerformFederatedAveraging(clientUpdates, clientWeights);
                case FederatedAggregationStrategy.WeightedAveraging:
                    return PerformWeightedAveraging(clientUpdates, clientWeights);
                case FederatedAggregationStrategy.MedianAggregation:
                    return PerformMedianAggregation(clientUpdates);
                case FederatedAggregationStrategy.TrimmedMean:
                    return PerformTrimmedMeanAggregation(clientUpdates, 0.1); // 10% trimming
                case FederatedAggregationStrategy.Byzantine:
                    return PerformByzantineRobustAggregation(clientUpdates, clientWeights);
                case FederatedAggregationStrategy.FederatedProx:
                    return PerformFederatedProx(clientUpdates, clientWeights, 0.1); // μ = 0.1
                default:
                    return PerformFederatedAveraging(clientUpdates, clientWeights);
            }
        }

        /// <summary>
        /// Perform standard federated averaging
        /// </summary>
        /// <param name="clientUpdates">Client parameter updates</param>
        /// <param name="clientWeights">Client weights</param>
        /// <returns>Averaged parameters</returns>
        private Dictionary<string, Vector<double>> PerformFederatedAveraging(
            Dictionary<string, Dictionary<string, Vector<double>>> clientUpdates,
            Dictionary<string, double> clientWeights)
        {
            var aggregatedParameters = new Dictionary<string, Vector<double>>();
            var normalizedWeights = NormalizeWeights(clientUpdates.Keys.ToList(), clientWeights);

            // Get all parameter names from first client
            var firstClient = clientUpdates.Values.First();
            foreach (var parameterName in firstClient.Keys)
            {
                var parameterSize = firstClient[parameterName].Length;
                var aggregatedValues = new double[parameterSize];

                // Aggregate parameter values across all clients
                foreach (var clientId in clientUpdates.Keys)
                {
                    if (clientUpdates[clientId].ContainsKey(parameterName))
                    {
                        var clientParameter = clientUpdates[clientId][parameterName];
                        var weight = normalizedWeights.ContainsKey(clientId) ? normalizedWeights[clientId] : 1.0 / clientUpdates.Count;

                        for (int i = 0; i < parameterSize; i++)
                        {
                            aggregatedValues[i] += weight * clientParameter[i];
                        }
                    }
                }

                aggregatedParameters[parameterName] = new Vector<double>(aggregatedValues);
            }

            return aggregatedParameters;
        }

        /// <summary>
        /// Perform weighted averaging based on client data sizes
        /// </summary>
        /// <param name="clientUpdates">Client parameter updates</param>
        /// <param name="clientWeights">Client weights based on data size</param>
        /// <returns>Weighted averaged parameters</returns>
        private Dictionary<string, Vector<double>> PerformWeightedAveraging(
            Dictionary<string, Dictionary<string, Vector<double>>> clientUpdates,
            Dictionary<string, double> clientWeights)
        {
            // Weighted averaging is the same as FedAvg when proper weights are provided
            return PerformFederatedAveraging(clientUpdates, clientWeights);
        }

        /// <summary>
        /// Perform median-based aggregation for Byzantine fault tolerance
        /// </summary>
        /// <param name="clientUpdates">Client parameter updates</param>
        /// <returns>Median aggregated parameters</returns>
        private Dictionary<string, Vector<double>> PerformMedianAggregation(
            Dictionary<string, Dictionary<string, Vector<double>>> clientUpdates)
        {
            var aggregatedParameters = new Dictionary<string, Vector<double>>();
            var firstClient = clientUpdates.Values.First();

            foreach (var parameterName in firstClient.Keys)
            {
                var parameterSize = firstClient[parameterName].Length;
                var aggregatedValues = new double[parameterSize];

                for (int i = 0; i < parameterSize; i++)
                {
                    var values = new List<double>();
                    
                    foreach (var clientId in clientUpdates.Keys)
                    {
                        if (clientUpdates[clientId].ContainsKey(parameterName))
                        {
                            values.Add(clientUpdates[clientId][parameterName][i]);
                        }
                    }

                    values.Sort();
                    aggregatedValues[i] = CalculateMedian(values);
                }

                aggregatedParameters[parameterName] = new Vector<double>(aggregatedValues);
            }

            return aggregatedParameters;
        }

        /// <summary>
        /// Perform trimmed mean aggregation for outlier robustness
        /// </summary>
        /// <param name="clientUpdates">Client parameter updates</param>
        /// <param name="trimRatio">Ratio of values to trim from each end</param>
        /// <returns>Trimmed mean aggregated parameters</returns>
        private Dictionary<string, Vector<double>> PerformTrimmedMeanAggregation(
            Dictionary<string, Dictionary<string, Vector<double>>> clientUpdates,
            double trimRatio)
        {
            var aggregatedParameters = new Dictionary<string, Vector<double>>();
            var firstClient = clientUpdates.Values.First();

            foreach (var parameterName in firstClient.Keys)
            {
                var parameterSize = firstClient[parameterName].Length;
                var aggregatedValues = new double[parameterSize];

                for (int i = 0; i < parameterSize; i++)
                {
                    var values = new List<double>();
                    
                    foreach (var clientId in clientUpdates.Keys)
                    {
                        if (clientUpdates[clientId].ContainsKey(parameterName))
                        {
                            values.Add(clientUpdates[clientId][parameterName][i]);
                        }
                    }

                    values.Sort();
                    aggregatedValues[i] = CalculateTrimmedMean(values, trimRatio);
                }

                aggregatedParameters[parameterName] = new Vector<double>(aggregatedValues);
            }

            return aggregatedParameters;
        }

        /// <summary>
        /// Perform Byzantine-robust aggregation using coordinate-wise median
        /// </summary>
        /// <param name="clientUpdates">Client parameter updates</param>
        /// <param name="clientWeights">Client weights</param>
        /// <returns>Byzantine-robust aggregated parameters</returns>
        private Dictionary<string, Vector<double>> PerformByzantineRobustAggregation(
            Dictionary<string, Dictionary<string, Vector<double>>> clientUpdates,
            Dictionary<string, double> clientWeights)
        {
            // For simplicity, use coordinate-wise median for Byzantine robustness
            // In practice, more sophisticated algorithms like Krum or Bulyan could be used
            return PerformMedianAggregation(clientUpdates);
        }

        /// <summary>
        /// Perform FederatedProx aggregation with proximal term
        /// </summary>
        /// <param name="clientUpdates">Client parameter updates</param>
        /// <param name="clientWeights">Client weights</param>
        /// <param name="mu">Proximal term coefficient</param>
        /// <returns>FederatedProx aggregated parameters</returns>
        private Dictionary<string, Vector<double>> PerformFederatedProx(
            Dictionary<string, Dictionary<string, Vector<double>>> clientUpdates,
            Dictionary<string, double> clientWeights,
            double mu)
        {
            // FederatedProx is similar to FedAvg but with a proximal term
            // For aggregation, we can use standard averaging
            // The proximal term is typically applied during local training
            return PerformFederatedAveraging(clientUpdates, clientWeights);
        }

        /// <summary>
        /// Normalize client weights to sum to 1
        /// </summary>
        /// <param name="clientIds">Client identifiers</param>
        /// <param name="clientWeights">Original client weights</param>
        /// <returns>Normalized weights</returns>
        private Dictionary<string, double> NormalizeWeights(List<string> clientIds, Dictionary<string, double> clientWeights)
        {
            var normalizedWeights = new Dictionary<string, double>();
            var totalWeight = 0.0;

            // Calculate total weight
            foreach (var clientId in clientIds)
            {
                if (clientWeights != null && clientWeights.ContainsKey(clientId))
                {
                    totalWeight += clientWeights[clientId];
                }
                else
                {
                    totalWeight += 1.0; // Equal weight if not specified
                }
            }

            // Normalize weights
            if (totalWeight > 0)
            {
                foreach (var clientId in clientIds)
                {
                    if (clientWeights != null && clientWeights.ContainsKey(clientId))
                    {
                        normalizedWeights[clientId] = clientWeights[clientId] / totalWeight;
                    }
                    else
                    {
                        normalizedWeights[clientId] = 1.0 / totalWeight;
                    }
                }
            }
            else
            {
                // Equal weights if total is zero
                foreach (var clientId in clientIds)
                {
                    normalizedWeights[clientId] = 1.0 / clientIds.Count;
                }
            }

            return normalizedWeights;
        }

        /// <summary>
        /// Calculate median of a sorted list of values
        /// </summary>
        /// <param name="sortedValues">Sorted values</param>
        /// <returns>Median value</returns>
        private double CalculateMedian(List<double> sortedValues)
        {
            if (sortedValues.Count == 0)
                return 0.0;

            if (sortedValues.Count % 2 == 0)
            {
                var mid1 = sortedValues[sortedValues.Count / 2 - 1];
                var mid2 = sortedValues[sortedValues.Count / 2];
                return (mid1 + mid2) / 2.0;
            }
            else
            {
                return sortedValues[sortedValues.Count / 2];
            }
        }

        /// <summary>
        /// Calculate trimmed mean by removing outliers
        /// </summary>
        /// <param name="sortedValues">Sorted values</param>
        /// <param name="trimRatio">Ratio to trim from each end</param>
        /// <returns>Trimmed mean</returns>
        private double CalculateTrimmedMean(List<double> sortedValues, double trimRatio)
        {
            if (sortedValues.Count == 0)
                return 0.0;

            var trimCount = (int)(sortedValues.Count * trimRatio);
            var startIndex = trimCount;
            var endIndex = sortedValues.Count - trimCount;

            if (startIndex >= endIndex)
            {
                // If trimming removes all values, return median
                return CalculateMedian(sortedValues);
            }

            var sum = 0.0;
            var count = 0;

            for (int i = startIndex; i < endIndex; i++)
            {
                sum += sortedValues[i];
                count++;
            }

            return count > 0 ? sum / count : 0.0;
        }

        /// <summary>
        /// Validate client updates for consistency
        /// </summary>
        /// <param name="clientUpdates">Client parameter updates</param>
        /// <returns>True if valid</returns>
        public bool ValidateClientUpdates(Dictionary<string, Dictionary<string, Vector<double>>> clientUpdates)
        {
            if (clientUpdates == null || clientUpdates.Count == 0)
                return false;

            var firstClient = clientUpdates.Values.First();
            var parameterNames = new HashSet<string>(firstClient.Keys);

            // Check that all clients have the same parameter structure
            foreach (var clientUpdate in clientUpdates.Values)
            {
                if (!parameterNames.SetEquals(clientUpdate.Keys))
                    return false;

                // Check parameter dimensions
                foreach (var paramName in parameterNames)
                {
                    if (clientUpdate[paramName].Length != firstClient[paramName].Length)
                        return false;
                }
            }

            return true;
        }

        /// <summary>
        /// Calculate aggregation quality metrics
        /// </summary>
        /// <param name="clientUpdates">Client parameter updates</param>
        /// <param name="aggregatedParameters">Aggregated parameters</param>
        /// <returns>Quality metrics</returns>
        public AggregationMetrics CalculateAggregationMetrics(
            Dictionary<string, Dictionary<string, Vector<double>>> clientUpdates,
            Dictionary<string, Vector<double>> aggregatedParameters)
        {
            var metrics = new AggregationMetrics();
            
            // Calculate variance across client updates
            var totalVariance = 0.0;
            var parameterCount = 0;

            foreach (var paramName in aggregatedParameters.Keys)
            {
                var aggregatedParam = aggregatedParameters[paramName];
                
                for (int i = 0; i < aggregatedParam.Length; i++)
                {
                    var values = clientUpdates.Values
                        .Where(update => update.ContainsKey(paramName))
                        .Select(update => update[paramName][i])
                        .ToList();

                    if (values.Count > 1)
                    {
                        var mean = values.Average();
                        var variance = values.Sum(v => Math.Pow(v - mean, 2)) / (values.Count - 1);
                        totalVariance += variance;
                        parameterCount++;
                    }
                }
            }

            metrics.AverageParameterVariance = parameterCount > 0 ? totalVariance / parameterCount : 0.0;
            metrics.ParticipatingClients = clientUpdates.Count;
            metrics.ParameterCount = parameterCount;
            
            return metrics;
        }
    }

}
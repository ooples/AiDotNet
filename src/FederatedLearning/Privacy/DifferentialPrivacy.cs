using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.Extensions;

namespace AiDotNet.FederatedLearning.Privacy
{
    /// <summary>
    /// Differential privacy implementation for federated learning
    /// Provides formal privacy guarantees for client data protection
    /// </summary>
    public class DifferentialPrivacy : IDifferentialPrivacy
    {
        /// <summary>
        /// Random number generator for noise generation
        /// </summary>
        private readonly Random _random = default!;

        /// <summary>
        /// Privacy accountant for tracking privacy budget
        /// </summary>
        public PrivacyAccountant PrivacyAccountant { get; private set; }

        /// <summary>
        /// Initialize differential privacy mechanism
        /// </summary>
        /// <param name="seed">Random seed for reproducibility</param>
        public DifferentialPrivacy(int? seed = null)
        {
            _random = seed.HasValue ? new Random(seed.Value) : new Random();
            PrivacyAccountant = new PrivacyAccountant();
        }

        /// <summary>
        /// Apply differential privacy to parameters using Gaussian mechanism
        /// </summary>
        /// <param name="parameters">Parameters to privatize</param>
        /// <param name="epsilon">Privacy parameter (smaller = more private)</param>
        /// <param name="delta">Privacy parameter (probability of privacy breach)</param>
        /// <param name="privacySettings">Additional privacy settings</param>
        /// <returns>Privatized parameters</returns>
        public Dictionary<string, Vector<double>> ApplyPrivacy(
            Dictionary<string, Vector<double>> parameters,
            double epsilon,
            double delta,
            PrivacySettings privacySettings)
        {
            if (parameters == null || parameters.Count == 0)
                throw new ArgumentException("Parameters cannot be null or empty");

            if (epsilon <= 0)
                throw new ArgumentException("Epsilon must be positive");

            if (delta < 0 || delta >= 1)
                throw new ArgumentException("Delta must be in [0, 1)");

            // Check privacy budget
            if (!PrivacyAccountant.CanSpend(epsilon, delta))
            {
                throw new InvalidOperationException("Insufficient privacy budget");
            }

            var privatizedParameters = new Dictionary<string, Vector<double>>();

            foreach (var kvp in parameters)
            {
                var parameterName = kvp.Key;
                var parameterValues = kvp.Value;

                Vector<double> privatizedValues;

                switch (privacySettings.MechanismType)
                {
                    case PrivacyMechanism.Gaussian:
                        privatizedValues = ApplyGaussianMechanism(parameterValues, epsilon, delta, privacySettings);
                        break;
                    case PrivacyMechanism.Laplace:
                        privatizedValues = ApplyLaplaceMechanism(parameterValues, epsilon, privacySettings);
                        break;
                    case PrivacyMechanism.ExponentialMechanism:
                        privatizedValues = ApplyExponentialMechanism(parameterValues, epsilon, privacySettings);
                        break;
                    case PrivacyMechanism.PrivateAggregationOfTeacherEnsembles:
                        privatizedValues = ApplyPATEMechanism(parameterValues, epsilon, delta, privacySettings);
                        break;
                    default:
                        privatizedValues = ApplyGaussianMechanism(parameterValues, epsilon, delta, privacySettings);
                        break;
                }

                privatizedParameters[parameterName] = privatizedValues;
            }

            // Update privacy budget
            PrivacyAccountant.Spend(epsilon, delta);

            return privatizedParameters;
        }

        /// <summary>
        /// Apply Gaussian mechanism for (ε, δ)-differential privacy
        /// </summary>
        /// <param name="values">Original values</param>
        /// <param name="epsilon">Privacy parameter</param>
        /// <param name="delta">Privacy parameter</param>
        /// <param name="settings">Privacy settings</param>
        /// <returns>Privatized values</returns>
        private Vector<double> ApplyGaussianMechanism(Vector<double> values, double epsilon, double delta, PrivacySettings settings)
        {
            // Clip values to ensure bounded sensitivity
            var clippedValues = ClipValues(values, settings.ClippingThreshold);

            // Calculate noise scale for Gaussian mechanism
            var sensitivity = settings.ClippingThreshold;
            var c = Math.Sqrt(2 * Math.Log(1.25 / delta));
            var noiseScale = c * sensitivity / epsilon;

            // Add Gaussian noise
            var noisyValues = new double[clippedValues.Length];
            for (int i = 0; i < clippedValues.Length; i++)
            {
                var noise = _random.NextGaussian(0, noiseScale * settings.NoiseMultiplier);
                noisyValues[i] = clippedValues[i] + noise;
            }

            return new Vector<double>(noisyValues);
        }

        /// <summary>
        /// Apply Laplace mechanism for ε-differential privacy
        /// </summary>
        /// <param name="values">Original values</param>
        /// <param name="epsilon">Privacy parameter</param>
        /// <param name="settings">Privacy settings</param>
        /// <returns>Privatized values</returns>
        private Vector<double> ApplyLaplaceMechanism(Vector<double> values, double epsilon, PrivacySettings settings)
        {
            // Clip values to ensure bounded sensitivity
            var clippedValues = ClipValues(values, settings.ClippingThreshold);

            // Calculate noise scale for Laplace mechanism
            var sensitivity = settings.ClippingThreshold;
            var noiseScale = sensitivity / epsilon;

            // Add Laplace noise
            var noisyValues = new double[clippedValues.Length];
            for (int i = 0; i < clippedValues.Length; i++)
            {
                var noise = GenerateLaplaceNoise(noiseScale * settings.NoiseMultiplier);
                noisyValues[i] = clippedValues[i] + noise;
            }

            return new Vector<double>(noisyValues);
        }

        /// <summary>
        /// Apply exponential mechanism for discrete outputs
        /// </summary>
        /// <param name="values">Original values</param>
        /// <param name="epsilon">Privacy parameter</param>
        /// <param name="settings">Privacy settings</param>
        /// <returns>Privatized values</returns>
        private Vector<double> ApplyExponentialMechanism(Vector<double> values, double epsilon, PrivacySettings settings)
        {
            // For simplicity, quantize values and apply exponential mechanism
            var quantizedValues = QuantizeValues(values, settings.QuantizationBits);
            
            // Apply exponential mechanism (simplified implementation)
            var privatizedValues = new double[values.Length];
            var sensitivity = 1.0; // Sensitivity for discrete values

            for (int i = 0; i < values.Length; i++)
            {
                var options = GenerateDiscreteOptions(quantizedValues[i], settings.QuantizationBits);
                var scores = options.Select(opt => -Math.Abs(opt - quantizedValues[i])).ToArray();
                
                // Sample from exponential distribution
                var probabilities = SoftmaxWithTemperature(scores, epsilon / (2 * sensitivity));
                var sampledIndex = SampleFromDistribution(probabilities);
                privatizedValues[i] = options[sampledIndex];
            }

            return new Vector<double>(privatizedValues);
        }

        /// <summary>
        /// Apply PATE (Private Aggregation of Teacher Ensembles) mechanism
        /// </summary>
        /// <param name="values">Original values</param>
        /// <param name="epsilon">Privacy parameter</param>
        /// <param name="delta">Privacy parameter</param>
        /// <param name="settings">Privacy settings</param>
        /// <returns>Privatized values</returns>
        private Vector<double> ApplyPATEMechanism(Vector<double> values, double epsilon, double delta, PrivacySettings settings)
        {
            // PATE mechanism for ensemble-based privacy
            // This is a simplified implementation - full PATE requires multiple teacher models
            
            var privatizedValues = new double[values.Length];
            var sensitivity = settings.ClippingThreshold;
            
            // Use Gaussian mechanism as base for PATE
            var noiseScale = Math.Sqrt(2 * Math.Log(1.25 / delta)) * sensitivity / epsilon;

            for (int i = 0; i < values.Length; i++)
            {
                // Add noise for teacher aggregation
                var teacherNoise = _random.NextGaussian(0, noiseScale * 0.5);
                var studentNoise = _random.NextGaussian(0, noiseScale * 0.5);
                
                privatizedValues[i] = values[i] + teacherNoise + studentNoise;
            }

            return new Vector<double>(privatizedValues);
        }

        /// <summary>
        /// Clip values to ensure bounded sensitivity
        /// </summary>
        /// <param name="values">Original values</param>
        /// <param name="clippingThreshold">Clipping threshold</param>
        /// <returns>Clipped values</returns>
        private Vector<double> ClipValues(Vector<double> values, double clippingThreshold)
        {
            var clippedValues = new double[values.Length];
            
            for (int i = 0; i < values.Length; i++)
            {
                clippedValues[i] = Math.Max(-clippingThreshold, Math.Min(clippingThreshold, values[i]));
            }

            return new Vector<double>(clippedValues);
        }

        /// <summary>
        /// Generate Laplace noise
        /// </summary>
        /// <param name="scale">Noise scale parameter</param>
        /// <returns>Laplace noise sample</returns>
        private double GenerateLaplaceNoise(double scale)
        {
            var u = _random.NextDouble() - 0.5;
            return -scale * Math.Sign(u) * Math.Log(1 - 2 * Math.Abs(u));
        }

        /// <summary>
        /// Quantize values for discrete mechanisms
        /// </summary>
        /// <param name="values">Original values</param>
        /// <param name="bits">Number of quantization bits</param>
        /// <returns>Quantized values</returns>
        private Vector<double> QuantizeValues(Vector<double> values, int bits)
        {
            var levels = (1 << bits) - 1;
            var minVal = values.Min();
            var maxVal = values.Max();
            var range = maxVal - minVal;

            if (range == 0)
                return new Vector<double>(values.ToArray());

            var quantizedValues = new double[values.Length];
            for (int i = 0; i < values.Length; i++)
            {
                var normalized = (values[i] - minVal) / range;
                var quantized = Math.Round(normalized * levels);
                quantizedValues[i] = minVal + (quantized / levels) * range;
            }

            return new Vector<double>(quantizedValues);
        }

        /// <summary>
        /// Generate discrete options for exponential mechanism
        /// </summary>
        /// <param name="value">Original value</param>
        /// <param name="bits">Number of bits</param>
        /// <returns>Discrete options</returns>
        private double[] GenerateDiscreteOptions(double value, int bits)
        {
            var numOptions = 1 << bits;
            var step = 2.0 / numOptions;
            var options = new double[numOptions];
            
            for (int i = 0; i < numOptions; i++)
            {
                options[i] = -1.0 + i * step;
            }

            return options;
        }

        /// <summary>
        /// Apply softmax with temperature for probability distribution
        /// </summary>
        /// <param name="scores">Utility scores</param>
        /// <param name="temperature">Temperature parameter</param>
        /// <returns>Probability distribution</returns>
        private double[] SoftmaxWithTemperature(double[] scores, double temperature)
        {
            var maxScore = scores.Max();
            var expScores = scores.Select(s => Math.Exp((s - maxScore) / temperature)).ToArray();
            var sumExp = expScores.Sum();
            
            return expScores.Select(exp => exp / sumExp).ToArray();
        }

        /// <summary>
        /// Sample from probability distribution
        /// </summary>
        /// <param name="probabilities">Probability distribution</param>
        /// <returns>Sampled index</returns>
        private int SampleFromDistribution(double[] probabilities)
        {
            var sample = _random.NextDouble();
            var cumulative = 0.0;
            
            for (int i = 0; i < probabilities.Length; i++)
            {
                cumulative += probabilities[i];
                if (sample <= cumulative)
                    return i;
            }

            return probabilities.Length - 1;
        }

        /// <summary>
        /// Calculate privacy cost for composition
        /// </summary>
        /// <param name="epsilons">List of epsilon values</param>
        /// <param name="deltas">List of delta values</param>
        /// <param name="compositionType">Type of composition</param>
        /// <returns>Total privacy cost</returns>
        public (double epsilon, double delta) CalculatePrivacyCost(
            List<double> epsilons,
            List<double> deltas,
            CompositionType compositionType)
        {
            switch (compositionType)
            {
                case CompositionType.Basic:
                    return (epsilons.Sum(), deltas.Sum());
                
                case CompositionType.Advanced:
                    // Advanced composition theorem
                    var k = epsilons.Count;
                    var maxEpsilon = epsilons.Max();
                    var totalDelta = deltas.Sum();
                    
                    var advancedEpsilon = Math.Sqrt(2 * k * Math.Log(1 / totalDelta)) * maxEpsilon + k * maxEpsilon * (Math.Exp(maxEpsilon) - 1);
                    return (advancedEpsilon, totalDelta);
                
                case CompositionType.RenyiDP:
                    // Rényi DP composition (simplified)
                    var renyiEpsilon = Math.Sqrt(epsilons.Sum(e => e * e));
                    return (renyiEpsilon, deltas.Max());
                
                default:
                    return (epsilons.Sum(), deltas.Sum());
            }
        }

        /// <summary>
        /// Validate privacy parameters
        /// </summary>
        /// <param name="epsilon">Privacy parameter</param>
        /// <param name="delta">Privacy parameter</param>
        /// <returns>True if valid</returns>
        public bool ValidatePrivacyParameters(double epsilon, double delta)
        {
            return epsilon > 0 && delta >= 0 && delta < 1;
        }

        /// <summary>
        /// Get privacy analysis report
        /// </summary>
        /// <returns>Privacy analysis report</returns>
        public PrivacyAnalysisReport GetPrivacyAnalysis()
        {
            return new PrivacyAnalysisReport
            {
                TotalEpsilon = PrivacyAccountant.TotalEpsilonSpent,
                TotalDelta = PrivacyAccountant.TotalDeltaSpent,
                RemainingBudget = PrivacyAccountant.RemainingBudget,
                NumberOfQueries = PrivacyAccountant.QueryCount,
                PrivacyGuarantees = PrivacyAccountant.GetPrivacyGuarantees()
            };
        }
        /// <summary>
        /// Calculates the noise scale for a given privacy budget.
        /// </summary>
        /// <param name="epsilon">Privacy budget parameter.</param>
        /// <param name="delta">Privacy failure probability.</param>
        /// <param name="sensitivity">Sensitivity of the function.</param>
        /// <returns>The noise scale to apply.</returns>
        public double CalculateNoiseScale(double epsilon, double delta, double sensitivity)
        {
            if (epsilon <= 0)
                throw new ArgumentException("Epsilon must be positive");
            
            if (delta < 0 || delta >= 1)
                throw new ArgumentException("Delta must be in [0, 1)");
                
            if (sensitivity < 0)
                throw new ArgumentException("Sensitivity must be non-negative");

            // For Gaussian mechanism
            var c = Math.Sqrt(2 * Math.Log(1.25 / delta));
            return c * sensitivity / epsilon;
        }

        /// <summary>
        /// Clips gradients to ensure bounded sensitivity.
        /// </summary>
        /// <param name="gradients">The gradients to clip.</param>
        /// <param name="maxNorm">Maximum allowed norm.</param>
        /// <returns>Clipped gradients.</returns>
        public Dictionary<string, Vector<double>> ClipGradients(
            Dictionary<string, Vector<double>> gradients,
            double maxNorm)
        {
            if (gradients == null || gradients.Count == 0)
                throw new ArgumentException("Gradients cannot be null or empty");
                
            if (maxNorm <= 0)
                throw new ArgumentException("Max norm must be positive");

            var clippedGradients = new Dictionary<string, Vector<double>>();

            // Calculate total norm
            var totalNormSquared = 0.0;
            foreach (var kvp in gradients)
            {
                var gradient = kvp.Value;
                for (int i = 0; i < gradient.Length; i++)
                {
                    totalNormSquared += gradient[i] * gradient[i];
                }
            }

            var totalNorm = Math.Sqrt(totalNormSquared);
            var scaleFactor = Math.Min(1.0, maxNorm / totalNorm);

            // Apply clipping
            foreach (var kvp in gradients)
            {
                var gradient = kvp.Value;
                var clippedValues = new double[gradient.Length];
                
                for (int i = 0; i < gradient.Length; i++)
                {
                    clippedValues[i] = gradient[i] * scaleFactor;
                }
                
                clippedGradients[kvp.Key] = new Vector<double>(clippedValues);
            }

            return clippedGradients;
        }
    }

    /// <summary>
    /// Privacy accountant for tracking privacy budget
    /// </summary>
    public class PrivacyAccountant
    {
        public double TotalEpsilonSpent { get; private set; }
        public double TotalDeltaSpent { get; private set; }
        public double EpsilonBudget { get; set; } = 10.0;
        public double DeltaBudget { get; set; } = 1e-5;
        public int QueryCount { get; private set; }
        
        private List<(double epsilon, double delta, DateTime timestamp)> _queries;

        public PrivacyAccountant()
        {
            _queries = new List<(double, double, DateTime)>();
        }

        public double RemainingBudget => Math.Max(0, EpsilonBudget - TotalEpsilonSpent);

        public bool CanSpend(double epsilon, double delta)
        {
            return TotalEpsilonSpent + epsilon <= EpsilonBudget && TotalDeltaSpent + delta <= DeltaBudget;
        }

        public void Spend(double epsilon, double delta)
        {
            TotalEpsilonSpent += epsilon;
            TotalDeltaSpent += delta;
            QueryCount++;
            _queries.Add((epsilon, delta, DateTime.UtcNow));
        }

        public List<string> GetPrivacyGuarantees()
        {
            var guarantees = new List<string>
            {
                $"(ε, δ)-DP with ε = {TotalEpsilonSpent:F6}, δ = {TotalDeltaSpent:E}",
                $"Privacy budget remaining: ε = {RemainingBudget:F6}",
                $"Total queries: {QueryCount}"
            };

            return guarantees;
        }
    }

    /// <summary>
    /// Privacy analysis report
    /// </summary>
    public class PrivacyAnalysisReport
    {
        public double TotalEpsilon { get; set; }
        public double TotalDelta { get; set; }
        public double RemainingBudget { get; set; }
        public int NumberOfQueries { get; set; }
        public List<string> PrivacyGuarantees { get; set; } = new List<string>();
    }

    /// <summary>
    /// Privacy mechanism types
    /// </summary>
    public enum PrivacyMechanism
    {
        Gaussian,
        Laplace,
        ExponentialMechanism,
        PrivateAggregationOfTeacherEnsembles
    }

    /// <summary>
    /// Composition types for privacy analysis
    /// </summary>
    public enum CompositionType
    {
        Basic,
        Advanced,
        RenyiDP
    }

    /// <summary>
    /// Extended privacy settings
    /// </summary>
    public class PrivacySettings
    {
        public bool UseDifferentialPrivacy { get; set; } = true;
        public double Epsilon { get; set; } = 1.0;
        public double Delta { get; set; } = 1e-5;
        public double ClippingThreshold { get; set; } = 1.0;
        public double NoiseMultiplier { get; set; } = 1.0;
        public PrivacyMechanism MechanismType { get; set; } = PrivacyMechanism.Gaussian;
        public CompositionType CompositionType { get; set; } = CompositionType.Advanced;
        public int QuantizationBits { get; set; } = 8;
        public bool UseAdaptiveClipping { get; set; } = false;
        public double AdaptiveClippingRate { get; set; } = 0.1;
    }
}

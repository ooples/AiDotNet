using System;
using System.Collections.Generic;
using AiDotNet.Extensions;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.ReinforcementLearning.Policies.Exploration;
using AiDotNet.Validation;

namespace AiDotNet.ReinforcementLearning.Policies
{
    /// <summary>
    /// Policy using Beta distribution for bounded continuous action spaces.
    /// Network outputs alpha and beta parameters for each action dimension.
    /// Actions are naturally bounded to [0, 1] and can be scaled to any [min, max] range.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class BetaPolicy<T> : PolicyBase<T>
    {
        private readonly NeuralNetwork<T> _policyNetwork;
        private readonly IExplorationStrategy<T> _explorationStrategy;
        private readonly int _actionSize;
        private readonly double _actionMin;
        private readonly double _actionMax;

        /// <summary>
        /// Initializes a new instance of the BetaPolicy class.
        /// </summary>
        /// <param name="policyNetwork">Network that outputs alpha and beta parameters (2 * actionSize outputs).</param>
        /// <param name="actionSize">The size of the action space.</param>
        /// <param name="explorationStrategy">The exploration strategy.</param>
        /// <param name="actionMin">Minimum action value (default: 0.0).</param>
        /// <param name="actionMax">Maximum action value (default: 1.0).</param>
        /// <param name="random">Optional random number generator.</param>
        public BetaPolicy(
            NeuralNetwork<T> policyNetwork,
            int actionSize,
            IExplorationStrategy<T> explorationStrategy,
            double actionMin = 0.0,
            double actionMax = 1.0,
            Random? random = null)
            : base(random)
        {
            Guard.NotNull(policyNetwork);
            _policyNetwork = policyNetwork;
            Guard.NotNull(explorationStrategy);
            _explorationStrategy = explorationStrategy;
            _actionSize = actionSize;
            _actionMin = actionMin;
            _actionMax = actionMax;
        }

        /// <summary>
        /// Selects an action by sampling from Beta distributions.
        /// </summary>
        public override Vector<T> SelectAction(Vector<T> state, bool training = true)
        {
            ValidateState(state, nameof(state));

            // Get alpha and beta parameters from network
            var stateTensor = Tensor<T>.FromVector(state);
            var outputTensor = _policyNetwork.Predict(stateTensor);
            var output = outputTensor.ToVector();

            // Network should output 2 * actionSize values (alpha and beta for each dimension)
            if (output.Length != 2 * _actionSize)
            {
                throw new InvalidOperationException(
                    string.Format("Network output size {0} does not match expected size {1} (2 * actionSize)",
                        output.Length, 2 * _actionSize));
            }

            var action = new Vector<T>(_actionSize);

            for (int i = 0; i < _actionSize; i++)
            {
                // Get alpha and beta (ensure positive via softplus: log(1 + exp(x)) + 1)
                double alphaRaw = NumOps.ToDouble(output[i]);
                double betaRaw = NumOps.ToDouble(output[_actionSize + i]);

                double alpha = Softplus(alphaRaw) + 1.0;  // Add 1 to ensure alpha > 1 for well-behaved Beta
                double beta = Softplus(betaRaw) + 1.0;

                // Sample from Beta distribution using transformation method
                double sample = SampleBeta(alpha, beta);

                // Scale from [0, 1] to [actionMin, actionMax]
                double scaledAction = _actionMin + sample * (_actionMax - _actionMin);
                action[i] = NumOps.FromDouble(scaledAction);
            }

            if (training)
            {
                return _explorationStrategy.GetExplorationAction(state, action, _actionSize, _random);
            }

            return action;
        }

        /// <summary>
        /// Computes the log probability of an action under the Beta distribution policy.
        /// </summary>
        public override T ComputeLogProb(Vector<T> state, Vector<T> action)
        {
            ValidateState(state, nameof(state));
            ValidateActionSize(_actionSize, action.Length, nameof(action));

            // Get alpha and beta parameters
            var stateTensor = Tensor<T>.FromVector(state);
            var outputTensor = _policyNetwork.Predict(stateTensor);
            var output = outputTensor.ToVector();

            T logProb = NumOps.Zero;

            for (int i = 0; i < _actionSize; i++)
            {
                double alphaRaw = NumOps.ToDouble(output[i]);
                double betaRaw = NumOps.ToDouble(output[_actionSize + i]);

                double alpha = Softplus(alphaRaw) + 1.0;
                double beta = Softplus(betaRaw) + 1.0;

                // Rescale action from [actionMin, actionMax] to [0, 1]
                double actionValue = NumOps.ToDouble(action[i]);
                double x = (actionValue - _actionMin) / (_actionMax - _actionMin);
                x = Math.Max(1e-7, Math.Min(1.0 - 1e-7, x));  // Clip to avoid log(0)

                // Beta distribution log probability
                // log p(x) = (α-1)log(x) + (β-1)log(1-x) - log(B(α,β))
                // where B(α,β) = Γ(α)Γ(β)/Γ(α+β)
                double logBeta = LogGamma(alpha) + LogGamma(beta) - LogGamma(alpha + beta);
                double betaLogProb = (alpha - 1) * Math.Log(x) + (beta - 1) * Math.Log(1 - x) - logBeta;

                // Account for rescaling Jacobian
                double rescaleLogProb = Math.Log(_actionMax - _actionMin);

                logProb = NumOps.Add(logProb, NumOps.FromDouble(betaLogProb - rescaleLogProb));
            }

            return logProb;
        }

        /// <summary>
        /// Gets the neural networks used by this policy.
        /// </summary>
        public override IReadOnlyList<INeuralNetwork<T>> GetNetworks()
        {
            return new List<INeuralNetwork<T>> { _policyNetwork };
        }

        /// <summary>
        /// Resets the exploration strategy.
        /// </summary>
        public override void Reset()
        {
            _explorationStrategy.Reset();
        }

        // Helper methods

        private double Softplus(double x)
        {
            // Numerically stable softplus: log(1 + exp(x))
            if (x > 20.0)
            {
                return x;  // For large x, softplus ≈ x
            }
            return Math.Log(1.0 + Math.Exp(x));
        }

        private double SampleBeta(double alpha, double beta)
        {
            // Sample from Beta using Gamma samples: if X~Gamma(α) and Y~Gamma(β), then X/(X+Y)~Beta(α,β)
            double x = SampleGamma(alpha);
            double y = SampleGamma(beta);
            return x / (x + y);
        }

        private double SampleGamma(double shape)
        {
            // Marsaglia and Tsang's method for Gamma sampling
            if (shape < 1.0)
            {
                return SampleGamma(shape + 1.0) * Math.Pow(_random.NextDouble(), 1.0 / shape);
            }

            double d = shape - 1.0 / 3.0;
            double c = 1.0 / Math.Sqrt(9.0 * d);

            while (true)
            {
                double x = 0.0;
                double v = 0.0;

                do
                {
                    // Sample from standard normal
                    x = _random.NextGaussian();
                    v = 1.0 + c * x;
                } while (v <= 0.0);

                v = v * v * v;
                double u = _random.NextDouble();

                if (u < 1.0 - 0.0331 * x * x * x * x)
                {
                    return d * v;
                }

                if (Math.Log(u) < 0.5 * x * x + d * (1.0 - v + Math.Log(v)))
                {
                    return d * v;
                }
            }
        }

        private double LogGamma(double x)
        {
            // Stirling's approximation for log-gamma function
            // Accurate for x > 1
            if (x < 1.5)
            {
                // Use recursion: Γ(x+1) = x * Γ(x)
                return Math.Log(x) + LogGamma(x + 1.0);
            }

            double logSqrt2Pi = 0.5 * Math.Log(2.0 * Math.PI);
            return logSqrt2Pi + (x - 0.5) * Math.Log(x) - x +
                   1.0 / (12.0 * x) - 1.0 / (360.0 * x * x * x);
        }
    }
}

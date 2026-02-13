using System;
using System.Collections.Generic;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.ReinforcementLearning.Policies.Exploration;
using AiDotNet.Validation;

namespace AiDotNet.ReinforcementLearning.Policies
{
    /// <summary>
    /// Policy for discrete action spaces using a neural network to output action logits.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class DiscretePolicy<T> : PolicyBase<T>
    {
        private readonly NeuralNetwork<T> _policyNetwork;
        private readonly IExplorationStrategy<T> _explorationStrategy;
        private readonly int _actionSize;

        public DiscretePolicy(
            NeuralNetwork<T> policyNetwork,
            int actionSize,
            IExplorationStrategy<T> explorationStrategy,
            Random? random = null)
            : base(random)
        {
            Guard.NotNull(policyNetwork);
            _policyNetwork = policyNetwork;
            Guard.NotNull(explorationStrategy);
            _explorationStrategy = explorationStrategy;
            _actionSize = actionSize;
        }

        public override Vector<T> SelectAction(Vector<T> state, bool training = true)
        {
            // Get action probabilities from network
            var stateTensor = Tensor<T>.FromVector(state);
            var logitsTensor = _policyNetwork.Predict(stateTensor);
            var logits = logitsTensor.ToVector();

            // Apply softmax to get probabilities
            var probabilities = Softmax(logits);

            // Sample action from distribution
            var policyAction = SampleCategorical(probabilities);

            if (training)
            {
                // Apply exploration strategy
                return _explorationStrategy.GetExplorationAction(state, policyAction, _actionSize, _random);
            }

            return policyAction;
        }

        public override T ComputeLogProb(Vector<T> state, Vector<T> action)
        {
            // Get logits from network
            var stateTensor = Tensor<T>.FromVector(state);
            var logitsTensor = _policyNetwork.Predict(stateTensor);
            var logits = logitsTensor.ToVector();

            // Apply softmax
            var probabilities = Softmax(logits);

            // Find which action was taken (one-hot encoded)
            int actionIndex = 0;
            for (int i = 0; i < action.Length; i++)
            {
                if (NumOps.ToDouble(action[i]) > 0.5)
                {
                    actionIndex = i;
                    break;
                }
            }

            // Return log probability of that action
            var prob = probabilities[actionIndex];
            var logProb = NumOps.FromDouble(Math.Log(NumOps.ToDouble(prob) + 1e-8));
            return logProb;
        }

        public override IReadOnlyList<INeuralNetwork<T>> GetNetworks()
        {
            return new List<INeuralNetwork<T>> { _policyNetwork };
        }

        public override void Reset()
        {
            _explorationStrategy.Reset();
        }

        // Helper methods
        private Vector<T> Softmax(Vector<T> logits)
        {
            var probabilities = new Vector<T>(logits.Length);
            T maxLogit = logits[0];

            for (int i = 1; i < logits.Length; i++)
            {
                if (NumOps.ToDouble(logits[i]) > NumOps.ToDouble(maxLogit))
                {
                    maxLogit = logits[i];
                }
            }

            T sumExp = NumOps.Zero;
            for (int i = 0; i < logits.Length; i++)
            {
                var expValue = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(NumOps.Subtract(logits[i], maxLogit))));
                probabilities[i] = expValue;
                sumExp = NumOps.Add(sumExp, expValue);
            }

            for (int i = 0; i < probabilities.Length; i++)
            {
                probabilities[i] = NumOps.Divide(probabilities[i], sumExp);
            }

            return probabilities;
        }

        private Vector<T> SampleCategorical(Vector<T> probabilities)
        {
            double randomValue = _random.NextDouble();
            double cumulativeProbability = 0.0;

            for (int i = 0; i < probabilities.Length; i++)
            {
                cumulativeProbability += NumOps.ToDouble(probabilities[i]);
                if (randomValue <= cumulativeProbability)
                {
                    var action = new Vector<T>(probabilities.Length);
                    action[i] = NumOps.One;
                    return action;
                }
            }

            // Fallback (should not happen)
            var fallbackAction = new Vector<T>(probabilities.Length);
            fallbackAction[probabilities.Length - 1] = NumOps.One;
            return fallbackAction;
        }
    }
}

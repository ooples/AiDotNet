using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Helpers;
using AiDotNet.ReinforcementLearning.Policies.Exploration;
using System;
using System.Collections.Generic;

namespace AiDotNet.ReinforcementLearning.Policies
{
    /// <summary>
    /// Policy for discrete action spaces using a neural network to output action logits.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class DiscretePolicy<T> : IPolicy<T>
    {
        private readonly NeuralNetwork<T> _policyNetwork;
        private readonly IExplorationStrategy<T> _explorationStrategy;
        private readonly Random _random;
        private readonly int _actionSize;

        public DiscretePolicy(
            NeuralNetwork<T> policyNetwork,
            int actionSize,
            IExplorationStrategy<T> explorationStrategy,
            Random? random = null)
        {
            _policyNetwork = policyNetwork ?? throw new ArgumentNullException(nameof(policyNetwork));
            _explorationStrategy = explorationStrategy ?? throw new ArgumentNullException(nameof(explorationStrategy));
            _random = random ?? new Random();
            _actionSize = actionSize;
        }

        public Vector<T> SelectAction(Vector<T> state, bool training = true)
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

        public T ComputeLogProb(Vector<T> state, Vector<T> action)
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
                if (NumOps<T>.ToDouble(action[i]) > 0.5)
                {
                    actionIndex = i;
                    break;
                }
            }

            // Return log probability of that action
            var prob = probabilities[actionIndex];
            var logProb = NumOps<T>.FromDouble(Math.Log(NumOps<T>.ToDouble(prob) + 1e-8));
            return logProb;
        }

        public IReadOnlyList<INeuralNetwork<T>> GetNetworks()
        {
            return new List<INeuralNetwork<T>> { _policyNetwork };
        }

        public void Reset()
        {
            _explorationStrategy.Reset();
        }

        public void Dispose()
        {
            // Cleanup if needed
        }

        // Helper methods
        private Vector<T> Softmax(Vector<T> logits)
        {
            var probabilities = new Vector<T>(logits.Length);
            T maxLogit = logits[0];

            for (int i = 1; i < logits.Length; i++)
            {
                if (NumOps<T>.ToDouble(logits[i]) > NumOps<T>.ToDouble(maxLogit))
                {
                    maxLogit = logits[i];
                }
            }

            T sumExp = NumOps<T>.Zero;
            for (int i = 0; i < logits.Length; i++)
            {
                var expValue = NumOps<T>.FromDouble(Math.Exp(NumOps<T>.ToDouble(NumOps<T>.Subtract(logits[i], maxLogit))));
                probabilities[i] = expValue;
                sumExp = NumOps<T>.Add(sumExp, expValue);
            }

            for (int i = 0; i < probabilities.Length; i++)
            {
                probabilities[i] = NumOps<T>.Divide(probabilities[i], sumExp);
            }

            return probabilities;
        }

        private Vector<T> SampleCategorical(Vector<T> probabilities)
        {
            double randomValue = _random.NextDouble();
            double cumulativeProbability = 0.0;

            for (int i = 0; i < probabilities.Length; i++)
            {
                cumulativeProbability += NumOps<T>.ToDouble(probabilities[i]);
                if (randomValue <= cumulativeProbability)
                {
                    var action = new Vector<T>(probabilities.Length);
                    action[i] = NumOps<T>.One;
                    return action;
                }
            }

            // Fallback (should not happen)
            var fallbackAction = new Vector<T>(probabilities.Length);
            fallbackAction[probabilities.Length - 1] = NumOps<T>.One;
            return fallbackAction;
        }
    }
}

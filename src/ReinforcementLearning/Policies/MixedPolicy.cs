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
    /// Policy for environments with both discrete and continuous action spaces.
    /// Outputs both categorical distribution for discrete actions and Gaussian for continuous actions.
    /// Common in robotics where you have discrete mode selection and continuous parameter control.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class MixedPolicy<T> : PolicyBase<T>
    {
        private readonly NeuralNetwork<T> _discreteNetwork;
        private readonly NeuralNetwork<T> _continuousNetwork;
        private readonly IExplorationStrategy<T> _discreteExploration;
        private readonly IExplorationStrategy<T> _continuousExploration;
        private readonly int _discreteActionSize;
        private readonly int _continuousActionSize;
        private readonly bool _sharedFeatures;

        /// <summary>
        /// Initializes a new instance of the MixedPolicy class.
        /// </summary>
        /// <param name="discreteNetwork">Network for discrete action logits.</param>
        /// <param name="continuousNetwork">Network for continuous action parameters (mean and log_std).</param>
        /// <param name="discreteActionSize">Number of discrete actions.</param>
        /// <param name="continuousActionSize">Number of continuous action dimensions.</param>
        /// <param name="discreteExploration">Exploration strategy for discrete actions.</param>
        /// <param name="continuousExploration">Exploration strategy for continuous actions.</param>
        /// <param name="sharedFeatures">Whether networks share feature extraction layers.</param>
        /// <param name="random">Optional random number generator.</param>
        public MixedPolicy(
            NeuralNetwork<T> discreteNetwork,
            NeuralNetwork<T> continuousNetwork,
            int discreteActionSize,
            int continuousActionSize,
            IExplorationStrategy<T> discreteExploration,
            IExplorationStrategy<T> continuousExploration,
            bool sharedFeatures = false,
            Random? random = null)
            : base(random)
        {
            Guard.NotNull(discreteNetwork);
            _discreteNetwork = discreteNetwork;
            Guard.NotNull(continuousNetwork);
            _continuousNetwork = continuousNetwork;
            Guard.NotNull(discreteExploration);
            _discreteExploration = discreteExploration;
            Guard.NotNull(continuousExploration);
            _continuousExploration = continuousExploration;
            _discreteActionSize = discreteActionSize;
            _continuousActionSize = continuousActionSize;
            _sharedFeatures = sharedFeatures;
        }

        /// <summary>
        /// Selects mixed action: [discrete_action, continuous_actions]
        /// </summary>
        public override Vector<T> SelectAction(Vector<T> state, bool training = true)
        {
            ValidateState(state, nameof(state));

            var stateTensor = Tensor<T>.FromVector(state);

            // Get discrete action (one-hot)
            var discreteLogitsTensor = _discreteNetwork.Predict(stateTensor);
            var discreteLogits = discreteLogitsTensor.ToVector();
            var discreteProbabilities = Softmax(discreteLogits);
            var discreteAction = SampleCategorical(discreteProbabilities);

            if (training)
            {
                discreteAction = _discreteExploration.GetExplorationAction(state, discreteAction, _discreteActionSize, _random);
            }

            // Get continuous action (Gaussian)
            var continuousOutputTensor = _continuousNetwork.Predict(stateTensor);
            var continuousOutput = continuousOutputTensor.ToVector();

            var continuousAction = new Vector<T>(_continuousActionSize);
            for (int i = 0; i < _continuousActionSize; i++)
            {
                double meanValue = NumOps.ToDouble(continuousOutput[i]);
                double logStdValue = NumOps.ToDouble(continuousOutput[_continuousActionSize + i]);
                double stdValue = Math.Exp(logStdValue);

                // Sample from Gaussian
                double sampledValue = meanValue + stdValue * _random.NextGaussian();

                continuousAction[i] = NumOps.FromDouble(sampledValue);
            }

            if (training)
            {
                continuousAction = _continuousExploration.GetExplorationAction(state, continuousAction, _continuousActionSize, _random);
            }

            // Concatenate: [discrete, continuous]
            var mixedAction = new Vector<T>(_discreteActionSize + _continuousActionSize);
            for (int i = 0; i < _discreteActionSize; i++)
            {
                mixedAction[i] = discreteAction[i];
            }
            for (int i = 0; i < _continuousActionSize; i++)
            {
                mixedAction[_discreteActionSize + i] = continuousAction[i];
            }

            return mixedAction;
        }

        /// <summary>
        /// Computes log probability of mixed action.
        /// </summary>
        public override T ComputeLogProb(Vector<T> state, Vector<T> action)
        {
            ValidateState(state, nameof(state));
            ValidateActionSize(_discreteActionSize + _continuousActionSize, action.Length, nameof(action));

            var stateTensor = Tensor<T>.FromVector(state);

            // Split action into discrete and continuous parts
            var discreteAction = new Vector<T>(_discreteActionSize);
            var continuousAction = new Vector<T>(_continuousActionSize);

            for (int i = 0; i < _discreteActionSize; i++)
            {
                discreteAction[i] = action[i];
            }
            for (int i = 0; i < _continuousActionSize; i++)
            {
                continuousAction[i] = action[_discreteActionSize + i];
            }

            // Discrete log prob
            var discreteLogitsTensor = _discreteNetwork.Predict(stateTensor);
            var discreteLogits = discreteLogitsTensor.ToVector();
            var discreteProbabilities = Softmax(discreteLogits);

            int discreteActionIndex = 0;
            for (int i = 0; i < _discreteActionSize; i++)
            {
                if (NumOps.ToDouble(discreteAction[i]) > 0.5)
                {
                    discreteActionIndex = i;
                    break;
                }
            }

            T discreteLogProb = NumOps.FromDouble(Math.Log(NumOps.ToDouble(discreteProbabilities[discreteActionIndex]) + 1e-8));

            // Continuous log prob
            var continuousOutputTensor = _continuousNetwork.Predict(stateTensor);
            var continuousOutput = continuousOutputTensor.ToVector();

            T continuousLogProb = NumOps.Zero;
            for (int i = 0; i < _continuousActionSize; i++)
            {
                double meanValue = NumOps.ToDouble(continuousOutput[i]);
                double logStdValue = NumOps.ToDouble(continuousOutput[_continuousActionSize + i]);
                double stdValue = Math.Exp(logStdValue);
                double actionValue = NumOps.ToDouble(continuousAction[i]);

                double diff = (actionValue - meanValue) / stdValue;
                double gaussianLogProb = -0.5 * diff * diff - Math.Log(stdValue) - 0.5 * Math.Log(2.0 * Math.PI);

                continuousLogProb = NumOps.Add(continuousLogProb, NumOps.FromDouble(gaussianLogProb));
            }

            // Total log prob is sum (since actions are independent)
            return NumOps.Add(discreteLogProb, continuousLogProb);
        }

        /// <summary>
        /// Gets the neural networks used by this policy.
        /// </summary>
        public override IReadOnlyList<INeuralNetwork<T>> GetNetworks()
        {
            return new List<INeuralNetwork<T>> { _discreteNetwork, _continuousNetwork };
        }

        /// <summary>
        /// Resets both exploration strategies.
        /// </summary>
        public override void Reset()
        {
            _discreteExploration.Reset();
            _continuousExploration.Reset();
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

            var fallbackAction = new Vector<T>(probabilities.Length);
            fallbackAction[probabilities.Length - 1] = NumOps.One;
            return fallbackAction;
        }
    }
}

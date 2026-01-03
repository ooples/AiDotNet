using System;
using System.Collections.Generic;
using AiDotNet.Extensions;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.ReinforcementLearning.Policies.Exploration;

namespace AiDotNet.ReinforcementLearning.Policies
{
    /// <summary>
    /// Policy for continuous action spaces using a neural network to output Gaussian parameters.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class ContinuousPolicy<T> : PolicyBase<T>
    {
        private readonly NeuralNetwork<T> _policyNetwork;
        private readonly IExplorationStrategy<T> _explorationStrategy;
        private readonly int _actionSize;
        private readonly bool _useTanhSquashing;

        public ContinuousPolicy(
            NeuralNetwork<T> policyNetwork,
            int actionSize,
            IExplorationStrategy<T> explorationStrategy,
            bool useTanhSquashing = false,
            Random? random = null)
            : base(random)
        {
            _policyNetwork = policyNetwork ?? throw new ArgumentNullException(nameof(policyNetwork));
            _explorationStrategy = explorationStrategy ?? throw new ArgumentNullException(nameof(explorationStrategy));
            _actionSize = actionSize;
            _useTanhSquashing = useTanhSquashing;
        }

        public override Vector<T> SelectAction(Vector<T> state, bool training = true)
        {
            // Get mean and log_std from network
            var stateTensor = Tensor<T>.FromVector(state);
            var outputTensor = _policyNetwork.Predict(stateTensor);
            var output = outputTensor.ToVector();

            // Split output into mean and log_std
            var mean = new Vector<T>(_actionSize);
            var logStd = new Vector<T>(_actionSize);

            for (int i = 0; i < _actionSize; i++)
            {
                mean[i] = output[i];
                logStd[i] = output[_actionSize + i];
            }

            // Sample from Gaussian distribution
            var action = new Vector<T>(_actionSize);
            for (int i = 0; i < _actionSize; i++)
            {
                double meanValue = NumOps.ToDouble(mean[i]);
                double stdValue = Math.Exp(NumOps.ToDouble(logStd[i]));

                double sampledValue = meanValue + stdValue * _random.NextGaussian();

                if (_useTanhSquashing)
                {
                    sampledValue = Math.Tanh(sampledValue);
                }

                action[i] = NumOps.FromDouble(sampledValue);
            }

            if (training)
            {
                return _explorationStrategy.GetExplorationAction(state, action, _actionSize, _random);
            }

            return action;
        }

        public override T ComputeLogProb(Vector<T> state, Vector<T> action)
        {
            // Get mean and log_std from network
            var stateTensor = Tensor<T>.FromVector(state);
            var outputTensor = _policyNetwork.Predict(stateTensor);
            var output = outputTensor.ToVector();

            T logProb = NumOps.Zero;

            for (int i = 0; i < _actionSize; i++)
            {
                double meanValue = NumOps.ToDouble(output[i]);
                double logStdValue = NumOps.ToDouble(output[_actionSize + i]);
                double stdValue = Math.Exp(logStdValue);

                double actionValue = NumOps.ToDouble(action[i]);

                // Gaussian log probability: -0.5 * ((x - mu) / sigma)^2 - log(sigma) - 0.5 * log(2*pi)
                double diff = (actionValue - meanValue) / stdValue;
                double gaussianLogProb = -0.5 * diff * diff - Math.Log(stdValue) - 0.5 * Math.Log(2.0 * Math.PI);

                if (_useTanhSquashing)
                {
                    // Correction for tanh squashing: log_prob -= log(1 - tanh^2(x))
                    double tanhCorrection = Math.Log(1.0 - Math.Tanh(actionValue) * Math.Tanh(actionValue) + 1e-6);
                    gaussianLogProb -= tanhCorrection;
                }

                logProb = NumOps.Add(logProb, NumOps.FromDouble(gaussianLogProb));
            }

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
    }
}

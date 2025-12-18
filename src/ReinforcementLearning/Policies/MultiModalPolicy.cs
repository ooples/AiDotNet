using System;
using System.Collections.Generic;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.ReinforcementLearning.Policies.Exploration;

namespace AiDotNet.ReinforcementLearning.Policies
{
    /// <summary>
    /// Multi-modal policy using mixture of Gaussians for complex action distributions.
    /// </summary>
    public class MultiModalPolicy<T> : PolicyBase<T>
    {
        private readonly NeuralNetwork<T> _policyNetwork;
        private readonly IExplorationStrategy<T> _explorationStrategy;
        private readonly int _actionSize;
        private readonly int _numComponents;

        public MultiModalPolicy(
            NeuralNetwork<T> policyNetwork,
            int actionSize,
            int numComponents,
            IExplorationStrategy<T> explorationStrategy,
            Random? random = null)
            : base(random)
        {
            _policyNetwork = policyNetwork ?? throw new ArgumentNullException(nameof(policyNetwork));
            _explorationStrategy = explorationStrategy ?? throw new ArgumentNullException(nameof(explorationStrategy));
            _actionSize = actionSize;
            _numComponents = numComponents;
        }

        public override Vector<T> SelectAction(Vector<T> state, bool training = true)
        {
            ValidateState(state, nameof(state));

            var stateTensor = Tensor<T>.FromVector(state);
            var outputTensor = _policyNetwork.Predict(stateTensor);
            var output = outputTensor.ToVector();

            int outputSize = _numComponents * (1 + 2 * _actionSize);
            if (output.Length != outputSize)
            {
                throw new InvalidOperationException(
                    string.Format("Network output size {0} does not match expected size {1}",
                        output.Length, outputSize));
            }

            var mixingCoefficients = new Vector<T>(_numComponents);
            var means = new List<Vector<T>>();
            var logStds = new List<Vector<T>>();

            int offset = 0;
            for (int k = 0; k < _numComponents; k++)
            {
                mixingCoefficients[k] = output[offset++];
            }

            mixingCoefficients = Softmax(mixingCoefficients);

            for (int k = 0; k < _numComponents; k++)
            {
                var mean = new Vector<T>(_actionSize);
                for (int i = 0; i < _actionSize; i++)
                {
                    mean[i] = output[offset++];
                }
                means.Add(mean);
            }

            for (int k = 0; k < _numComponents; k++)
            {
                var logStd = new Vector<T>(_actionSize);
                for (int i = 0; i < _actionSize; i++)
                {
                    logStd[i] = output[offset++];
                }
                logStds.Add(logStd);
            }

            int selectedComponent = SampleCategoricalIndex(mixingCoefficients);

            var action = new Vector<T>(_actionSize);
            for (int i = 0; i < _actionSize; i++)
            {
                double meanValue = NumOps.ToDouble(means[selectedComponent][i]);
                double stdValue = Math.Exp(NumOps.ToDouble(logStds[selectedComponent][i]));

                double u1 = _random.NextDouble();
                double u2 = _random.NextDouble();
                double normalSample = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);

                double sampledValue = meanValue + stdValue * normalSample;
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
            ValidateState(state, nameof(state));
            ValidateActionSize(_actionSize, action.Length, nameof(action));

            var stateTensor = Tensor<T>.FromVector(state);
            var outputTensor = _policyNetwork.Predict(stateTensor);
            var output = outputTensor.ToVector();

            var mixingCoefficients = new Vector<T>(_numComponents);
            var means = new List<Vector<T>>();
            var logStds = new List<Vector<T>>();

            int offset = 0;
            for (int k = 0; k < _numComponents; k++)
            {
                mixingCoefficients[k] = output[offset++];
            }

            mixingCoefficients = Softmax(mixingCoefficients);

            for (int k = 0; k < _numComponents; k++)
            {
                var mean = new Vector<T>(_actionSize);
                for (int i = 0; i < _actionSize; i++)
                {
                    mean[i] = output[offset++];
                }
                means.Add(mean);
            }

            for (int k = 0; k < _numComponents; k++)
            {
                var logStd = new Vector<T>(_actionSize);
                for (int i = 0; i < _actionSize; i++)
                {
                    logStd[i] = output[offset++];
                }
                logStds.Add(logStd);
            }

            double totalProb = 0.0;
            for (int k = 0; k < _numComponents; k++)
            {
                double componentWeight = NumOps.ToDouble(mixingCoefficients[k]);
                double componentLogProb = 0.0;

                for (int i = 0; i < _actionSize; i++)
                {
                    double meanValue = NumOps.ToDouble(means[k][i]);
                    double logStdValue = NumOps.ToDouble(logStds[k][i]);
                    double stdValue = Math.Exp(logStdValue);
                    double actionValue = NumOps.ToDouble(action[i]);

                    double diff = (actionValue - meanValue) / stdValue;
                    componentLogProb += -0.5 * diff * diff - Math.Log(stdValue) - 0.5 * Math.Log(2.0 * Math.PI);
                }

                totalProb += componentWeight * Math.Exp(componentLogProb);
            }

            return NumOps.FromDouble(Math.Log(totalProb + 1e-8));
        }

        public override IReadOnlyList<INeuralNetwork<T>> GetNetworks()
        {
            return new List<INeuralNetwork<T>> { _policyNetwork };
        }

        public override void Reset()
        {
            _explorationStrategy.Reset();
        }

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

        private int SampleCategoricalIndex(Vector<T> probabilities)
        {
            double randomValue = _random.NextDouble();
            double cumulativeProbability = 0.0;

            for (int i = 0; i < probabilities.Length; i++)
            {
                cumulativeProbability += NumOps.ToDouble(probabilities[i]);
                if (randomValue <= cumulativeProbability)
                {
                    return i;
                }
            }

            return probabilities.Length - 1;
        }
    }
}

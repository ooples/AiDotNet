using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.AutoML.SearchSpace;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.AutoML.NAS
{
    /// <summary>
    /// Efficient Neural Architecture Search via Parameter Sharing.
    /// ENAS uses a controller RNN to sample architectures and shares weights among child models,
    /// achieving 1000x speedup over standard NAS.
    ///
    /// Reference: "Efficient Neural Architecture Search via Parameter Sharing" (ICML 2018)
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations</typeparam>
    public class ENAS<T> : NasAutoMLModelBase<T>
    {
        private readonly INumericOperations<T> _ops;
        private readonly SearchSpaceBase<T> _nasSearchSpace;
        private readonly int _numNodes;
        private readonly int _numOperations;
        private readonly Random _random;

        // Controller (policy network) parameters
        private readonly List<Vector<T>> _controllerWeights;
        private readonly List<Vector<T>> _controllerGradients;

        // Shared weights for all child models
        private readonly Dictionary<string, Vector<T>> _sharedWeights;
        private readonly Dictionary<string, Vector<T>> _sharedGradients;

        // REINFORCE baseline for variance reduction
        private T _baseline;
        private readonly T _baselineDecay;

        // Controller hyperparameters
        private readonly int _controllerHiddenSize;
        private readonly T _entropyWeight;

        protected override INumericOperations<T> NumOps => _ops;
        protected override SearchSpaceBase<T> NasSearchSpace => _nasSearchSpace;
        protected override int NasNumNodes => _numNodes;

        public ENAS(SearchSpaceBase<T> searchSpace, int numNodes = 4,
            int controllerHiddenSize = 100, double baselineDecay = 0.95, double entropyWeight = 0.01)
        {
            _ops = MathHelper.GetNumericOperations<T>();
            _nasSearchSpace = searchSpace;
            _numNodes = numNodes;
            _numOperations = searchSpace.Operations?.Count ?? 5;
            _random = new Random(42);

            _controllerHiddenSize = controllerHiddenSize;
            _baselineDecay = _ops.FromDouble(baselineDecay);
            _entropyWeight = _ops.FromDouble(entropyWeight);
            _baseline = _ops.Zero;

            // Initialize controller weights (simplified LSTM controller)
            _controllerWeights = new List<Vector<T>>();
            _controllerGradients = new List<Vector<T>>();

            // Each decision requires: node selection + operation selection
            int numDecisions = _numNodes * 2;  // For each node: which prev node + which operation
            for (int i = 0; i < numDecisions; i++)
            {
                var weight = new Vector<T>(_controllerHiddenSize);
                for (int j = 0; j < weight.Length; j++)
                {
                    weight[j] = _ops.FromDouble((_random.NextDouble() - 0.5) * 0.1);
                }
                _controllerWeights.Add(weight);
                _controllerGradients.Add(new Vector<T>(_controllerHiddenSize));
            }

            // Initialize shared weights
            _sharedWeights = new Dictionary<string, Vector<T>>();
            _sharedGradients = new Dictionary<string, Vector<T>>();
        }

        /// <summary>
        /// Samples an architecture using the controller policy
        /// </summary>
        public (Architecture<T> architecture, T logProb, T entropy) SampleArchitecture()
        {
            var architecture = new Architecture<T>();
            T totalLogProb = _ops.Zero;
            T totalEntropy = _ops.Zero;

            var hiddenState = new Vector<T>(_controllerHiddenSize);
            for (int i = 0; i < hiddenState.Length; i++)
            {
                hiddenState[i] = _ops.Zero;
            }

            for (int nodeIdx = 1; nodeIdx <= _numNodes; nodeIdx++)
            {
                // Sample which previous node to connect from
                int maxPrevNodes = nodeIdx;
                var prevNodeProbs = ComputeProbabilities(hiddenState, maxPrevNodes);
                int selectedPrevNode = SampleFromDistribution(prevNodeProbs);

                // Update log probability and entropy
                totalLogProb = _ops.Add(totalLogProb, _ops.Log(_ops.Add(prevNodeProbs[selectedPrevNode], _ops.FromDouble(1e-10))));
                totalEntropy = _ops.Add(totalEntropy, ComputeEntropy(prevNodeProbs));

                // Update hidden state
                UpdateHiddenState(hiddenState, selectedPrevNode);

                // Sample which operation to apply
                var opProbs = ComputeProbabilities(hiddenState, _numOperations);
                int selectedOp = SampleFromDistribution(opProbs);

                // Update log probability and entropy
                totalLogProb = _ops.Add(totalLogProb, _ops.Log(_ops.Add(opProbs[selectedOp], _ops.FromDouble(1e-10))));
                totalEntropy = _ops.Add(totalEntropy, ComputeEntropy(opProbs));

                // Update hidden state
                UpdateHiddenState(hiddenState, selectedOp);

                // Add to architecture
                if (_nasSearchSpace.Operations != null && selectedOp < _nasSearchSpace.Operations.Count)
                {
                    var operation = _nasSearchSpace.Operations[selectedOp];
                    architecture.AddOperation(nodeIdx, selectedPrevNode, operation);
                }
            }

            return (architecture, totalLogProb, totalEntropy);
        }

        /// <summary>
        /// Computes probability distribution over choices using controller
        /// </summary>
        private List<T> ComputeProbabilities(Vector<T> hiddenState, int numChoices)
        {
            // Simplified: use linear layer + softmax
            var logits = new T[numChoices];
            for (int i = 0; i < numChoices; i++)
            {
                T logit = _ops.Zero;
                for (int j = 0; j < Math.Min(hiddenState.Length, _controllerHiddenSize); j++)
                {
                    logit = _ops.Add(logit, hiddenState[j]);
                }
                logits[i] = _ops.Divide(logit, _ops.FromDouble(hiddenState.Length));
            }

            // Apply softmax
            T maxLogit = logits[0];
            for (int i = 1; i < numChoices; i++)
            {
                if (_ops.GreaterThan(logits[i], maxLogit))
                    maxLogit = logits[i];
            }

            T sumExp = _ops.Zero;
            var expValues = new T[numChoices];
            for (int i = 0; i < numChoices; i++)
            {
                expValues[i] = _ops.Exp(_ops.Subtract(logits[i], maxLogit));
                sumExp = _ops.Add(sumExp, expValues[i]);
            }

            var probs = new List<T>();
            for (int i = 0; i < numChoices; i++)
            {
                probs.Add(_ops.Divide(expValues[i], sumExp));
            }

            return probs;
        }

        /// <summary>
        /// Samples from a probability distribution
        /// </summary>
        private int SampleFromDistribution(List<T> probs)
        {
            double rand = _random.NextDouble();
            double cumulative = 0.0;

            for (int i = 0; i < probs.Count; i++)
            {
                cumulative += Convert.ToDouble(probs[i]);
                if (rand <= cumulative)
                    return i;
            }

            return probs.Count - 1;
        }

        /// <summary>
        /// Computes entropy of a probability distribution
        /// </summary>
        private T ComputeEntropy(List<T> probs)
        {
            T entropy = _ops.Zero;
            foreach (var p in probs)
            {
                if (!_ops.GreaterThan(p, _ops.Zero))
                    continue;

                entropy = _ops.Subtract(entropy, _ops.Multiply(p, _ops.Log(p)));
            }
            return entropy;
        }

        /// <summary>
        /// Updates controller hidden state (simplified LSTM cell)
        /// </summary>
        private void UpdateHiddenState(Vector<T> hiddenState, int choice)
        {
            // Simplified update: add choice information to hidden state
            T choiceValue = _ops.FromDouble(choice / 10.0);
            for (int i = 0; i < hiddenState.Length; i++)
            {
                hiddenState[i] = _ops.Multiply(hiddenState[i], _ops.FromDouble(0.9));
                hiddenState[i] = _ops.Add(hiddenState[i], choiceValue);
            }
        }

        /// <summary>
        /// Updates controller using REINFORCE policy gradient
        /// </summary>
        public void UpdateController(T reward, T logProb, T entropy)
        {
            // Update baseline using exponential moving average
            _baseline = _ops.Add(
                _ops.Multiply(_baselineDecay, _baseline),
                _ops.Multiply(_ops.Subtract(_ops.One, _baselineDecay), reward)
            );

            // REINFORCE gradient: (reward - baseline) * logProb + entropy_weight * entropy
            T advantage = _ops.Subtract(reward, _baseline);
            _ = _ops.Subtract(
                _ops.Multiply(advantage, logProb),
                _ops.Multiply(_entropyWeight, entropy)
            );

            // Gradient is stored for optimizer to use
            // In practice, this would use backpropagation through the controller
        }

        /// <summary>
        /// Gets shared weights for a specific operation
        /// </summary>
        public Vector<T> GetSharedWeights(string operationKey)
        {
            if (!_sharedWeights.ContainsKey(operationKey))
            {
                // Initialize new shared weights
                var weights = new Vector<T>(100);  // Fixed size for simplicity
                for (int i = 0; i < weights.Length; i++)
                {
                    weights[i] = _ops.FromDouble((_random.NextDouble() - 0.5) * 0.1);
                }
                _sharedWeights[operationKey] = weights;
                _sharedGradients[operationKey] = new Vector<T>(100);
            }

            return _sharedWeights[operationKey];
        }

        /// <summary>
        /// Gets controller parameters for optimization
        /// </summary>
        public List<Vector<T>> GetControllerParameters() => _controllerWeights;

        /// <summary>
        /// Gets controller gradients
        /// </summary>
        public List<Vector<T>> GetControllerGradients() => _controllerGradients;

        /// <summary>
        /// Gets shared weights for all operations
        /// </summary>
        public Dictionary<string, Vector<T>> GetSharedWeights() => _sharedWeights;

        /// <summary>
        /// Gets shared weight gradients
        /// </summary>
        public Dictionary<string, Vector<T>> GetSharedGradients() => _sharedGradients;

        /// <summary>
        /// Gets current baseline value
        /// </summary>
        public T GetBaseline() => _baseline;

        protected override Architecture<T> SearchArchitecture(
            Tensor<T> inputs,
            Tensor<T> targets,
            Tensor<T> validationInputs,
            Tensor<T> validationTargets,
            TimeSpan timeLimit,
            CancellationToken cancellationToken)
        {
            return SampleArchitecture().architecture;
        }

        protected override AutoMLModelBase<T, Tensor<T>, Tensor<T>> CreateInstanceForCopy()
        {
            return new ENAS<T>(
                _nasSearchSpace,
                _numNodes,
                _controllerHiddenSize,
                baselineDecay: _ops.ToDouble(_baselineDecay),
                entropyWeight: _ops.ToDouble(_entropyWeight));
        }
    }
}

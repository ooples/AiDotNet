using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.AutoML.SearchSpace;
using AiDotNet.Enums;
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
    /// <remarks>
    /// <para><b>For Beginners:</b> ENAS makes architecture search 1000x faster by sharing
    /// weights between candidate architectures. Instead of training each design from scratch,
    /// all designs share the same trained parameters. A controller network learns to pick
    /// good architectures, like a manager who assigns existing workers to different teams
    /// rather than hiring new ones each time.</para>
    /// </remarks>
    /// <example>
    /// <code>
    /// // Use AiModelBuilder facade with ENAS for neural architecture search
    /// var trainingData = Tensor&lt;float&gt;.CreateRandom(4, 3, 32, 32);
    /// var trainingLabels = Tensor&lt;float&gt;.CreateRandom(4, 10);
    /// var inputTensor = Tensor&lt;float&gt;.CreateRandom(1, 3, 32, 32);
    /// 
    /// var builder = new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
    ///     .ConfigureModel(new ENAS&lt;float&gt;(new MobileNetSearchSpace&lt;float&gt;(), numNodes: 4));
    ///
    /// var result = builder.Build(trainingData, trainingLabels);
    /// var prediction = result.Predict(inputTensor);
    /// </code>
    /// </example>
    [ModelMetadataExempt]
    [ModelDomain(ModelDomain.MachineLearning)]
    [ModelCategory(ModelCategory.NeuralNetwork)]
    [ModelCategory(ModelCategory.Optimization)]
    [ModelTask(ModelTask.Classification)]
    [ModelTask(ModelTask.FeatureExtraction)]
    [ModelComplexity(ModelComplexity.High)]
    [ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("Efficient Neural Architecture Search via Parameter Sharing", "https://arxiv.org/abs/1802.03268")]
    public partial class ENAS<T> : NasAutoMLModelBase<T>
    {
        private readonly INumericOperations<T> _ops;
        private readonly SearchSpaceBase<T> _nasSearchSpace;
        private readonly int _numNodes;
        private readonly int _numOperations;
        private readonly Random _random;

        // Controller (policy network) parameters
        private readonly List<Vector<T>> _controllerWeights;
        private readonly List<Vector<T>> _controllerGradients;

        // d log pi(a) / d W for the most recently sampled architecture (the REINFORCE score function)
        private readonly List<Vector<T>> _sampleLogProbGradients;

        // Shared weights for all child models
        private readonly Dictionary<string, Vector<T>> _sharedWeights;
        private readonly Dictionary<string, Vector<T>> _sharedGradients;

        // REINFORCE baseline for variance reduction
        private T _baseline;
        private readonly T _baselineDecay;

        // Controller hyperparameters
        private readonly int _controllerHiddenSize;
        private readonly int _controllerMaxChoices;
        private readonly T _entropyWeight;

        /// <inheritdoc />
        protected override INumericOperations<T> NumOps => _ops;
        /// <inheritdoc />
        protected override SearchSpaceBase<T> NasSearchSpace => _nasSearchSpace;
        /// <inheritdoc />
        protected override int NasNumNodes => _numNodes;

        public ENAS(SearchSpaceBase<T> searchSpace, int numNodes = 4,
            int controllerHiddenSize = 100, double baselineDecay = 0.95, double entropyWeight = 0.01)
        {
            _ops = MathHelper.GetNumericOperations<T>();
            _nasSearchSpace = searchSpace;
            _numNodes = numNodes;
            _numOperations = searchSpace.Operations?.Count ?? 5;
            _random = RandomHelper.CreateSeededRandom(42);

            _controllerHiddenSize = controllerHiddenSize;
            _controllerMaxChoices = Math.Max(_numNodes, _numOperations);
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
                var weight = new Vector<T>(_controllerMaxChoices * _controllerHiddenSize);
                for (int j = 0; j < weight.Length; j++)
                {
                    weight[j] = _ops.FromDouble((_random.NextDouble() - 0.5) * 0.1);
                }
                _controllerWeights.Add(weight);
                _controllerGradients.Add(new Vector<T>(weight.Length));
            }

            _sampleLogProbGradients = _controllerWeights.Select(w => new Vector<T>(w.Length)).ToList();

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
            foreach (var gradient in _sampleLogProbGradients)
            {
                gradient.Fill(_ops.Zero);
            }

            for (int nodeIdx = 1; nodeIdx <= _numNodes; nodeIdx++)
            {
                int prevDecisionIdx = (nodeIdx - 1) * 2;

                // Sample previous node connection
                int selectedPrevNode = SampleAndUpdateState(
                    hiddenState, nodeIdx, prevDecisionIdx,
                    ref totalLogProb, ref totalEntropy);

                // Sample operation to apply
                int selectedOp = SampleAndUpdateState(
                    hiddenState, _numOperations, prevDecisionIdx + 1,
                    ref totalLogProb, ref totalEntropy);

                // Add to architecture
                AddOperationToArchitecture(architecture, nodeIdx, selectedPrevNode, selectedOp);
            }

            return (architecture, totalLogProb, totalEntropy);
        }

        /// <summary>
        /// Computes probability distribution over choices using controller
        /// </summary>
        private List<T> ComputeProbabilities(Vector<T> hiddenState, int numChoices, int decisionIdx)
        {
            var probabilities = ComputeSoftmaxProbabilities(hiddenState, numChoices, decisionIdx);
            return probabilities.ToList();
        }

        private Vector<T> ComputeSoftmaxProbabilities(Vector<T> hiddenState, int numChoices, int decisionIdx)
        {
            var logits = ComputeLogits(hiddenState, numChoices, decisionIdx);
            return NasSamplingHelper.Softmax(logits, _ops);
        }

        private Vector<T> ComputeLogits(Vector<T> hiddenState, int numChoices, int decisionIdx)
        {
            // logits = W[:numChoices, :hiddenSize] h, with W stored row-major as [maxChoices, controllerHiddenSize].
            var engine = AiDotNetEngine.Current;
            var weights = ControllerWeightRows(_controllerWeights[decisionIdx % _controllerWeights.Count], numChoices);
            int hiddenSize = Math.Min(hiddenState.Length, _controllerHiddenSize);
            var hidden = hiddenState.GetSubVector(0, hiddenSize).AsTensor().Reshape(new[] { hiddenSize, 1 });
            if (hiddenSize < _controllerHiddenSize)
            {
                weights = weights.Slice(1, 0, hiddenSize);
            }

            return engine.TensorMatMul(weights, hidden).Reshape(new[] { numChoices }).ToVector();
        }

        private Tensor<T> ControllerWeightRows(Vector<T> weights, int numChoices)
        {
            return weights.AsTensor().Reshape(new[] { _controllerMaxChoices, _controllerHiddenSize }).Slice(0, 0, numChoices);
        }

        /// <summary>
        /// Samples from a probability distribution
        /// </summary>
        private int SampleFromDistribution(List<T> probs)
        {
            double rand = _random.NextDouble(), cumulative = 0.0;

            for (int i = 0; i < probs.Count; i++)
            {
                cumulative += _ops.ToDouble(probs[i]);
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
            // H = -sum p log p over the support (p > 0), so 0 log 0 contributes 0.
            var engine = AiDotNetEngine.Current;
            var p = new Vector<T>(probs).AsTensor();
            var plogp = engine.TensorWhere(engine.TensorGreaterThan(p, _ops.Zero),
                engine.TensorMultiply(p, engine.TensorLog(p)), new Tensor<T>(new[] { p.Length }));
            return _ops.Negate(engine.TensorSum(plogp));
        }

        /// <summary>
        /// Updates controller hidden state (simplified LSTM cell)
        /// </summary>
        private void UpdateHiddenState(Vector<T> hiddenState, int choice)
        {
            // Simplified update: add choice information to hidden state
            var engine = AiDotNetEngine.Current;
            var updated = engine.TensorAddScalar(
                engine.TensorMultiplyScalar(hiddenState.AsTensor(), _ops.FromDouble(0.9)), _ops.FromDouble(choice / 10.0));
            updated.ToVector().AsSpan().CopyTo(hiddenState.AsWritableSpan());
        }

        /// <summary>
        /// Adds d log softmax(W h)_a / d W = (onehot(a) - p) h^T for one controller decision to the sample's
        /// score function.
        /// </summary>
        private void AccumulateLogProbGradient(Vector<T> hiddenState, List<T> probs, int selected, int decisionIdx)
        {
            var engine = AiDotNetEngine.Current;
            int weightIdx = decisionIdx % _controllerWeights.Count;
            int numChoices = probs.Count;
            int hiddenSize = Math.Min(hiddenState.Length, _controllerHiddenSize);

            var selectedIndex = new Tensor<int>(new[] { 1 });
            selectedIndex.Fill(selected);
            var oneHot = engine.TensorOneHot<T>(selectedIndex, numChoices).Reshape(new[] { numChoices });
            var dLogits = engine.TensorSubtract(oneHot, new Vector<T>(probs).AsTensor());
            var dWeights = engine.TensorOuter(dLogits, hiddenState.GetSubVector(0, hiddenSize).AsTensor());

            // Place the [numChoices, hiddenSize] block in the [maxChoices, controllerHiddenSize] weight layout.
            var full = new Tensor<T>(new[] { _controllerMaxChoices, _controllerHiddenSize });
            full.SetSubTensor(new[] { 0, 0 }, dWeights);
            var accumulated = engine.TensorAdd(_sampleLogProbGradients[weightIdx].AsTensor(), full.Reshape(new[] { full.Length }));
            accumulated.ToVector().AsSpan().CopyTo(_sampleLogProbGradients[weightIdx].AsWritableSpan());
        }

        /// <summary>
        /// Samples a choice, updates log probability, entropy, and hidden state
        /// </summary>
        private int SampleAndUpdateState(
            Vector<T> hiddenState,
            int numChoices,
            int decisionIdx,
            ref T totalLogProb,
            ref T totalEntropy)
        {
            var probs = ComputeProbabilities(hiddenState, numChoices, decisionIdx);
            int selected = SampleFromDistribution(probs);
            AccumulateLogProbGradient(hiddenState, probs, selected, decisionIdx);

            totalLogProb = _ops.Add(totalLogProb, _ops.Log(_ops.Add(probs[selected], _ops.FromDouble(1e-10))));
            totalEntropy = _ops.Add(totalEntropy, ComputeEntropy(probs));
            UpdateHiddenState(hiddenState, selected);

            return selected;
        }

        /// <summary>
        /// Adds an operation to the architecture if valid
        /// </summary>
        private void AddOperationToArchitecture(Architecture<T> architecture, int nodeIdx, int selectedPrevNode, int selectedOp)
        {
            if (_nasSearchSpace.Operations != null && selectedOp < _nasSearchSpace.Operations.Count)
            {
                var operation = _nasSearchSpace.Operations[selectedOp];
                architecture.AddOperation(nodeIdx, selectedPrevNode, operation);
            }
        }

        /// <summary>
        /// Updates controller using REINFORCE policy gradient
        /// </summary>
        public void UpdateController(T reward, T logProb, T entropy)
        {
            // ENAS (Pham et al. 2018) adds the entropy bonus to the reward, moves the baseline toward it, and
            // minimises -(R - b) log pi(a): the gradient is -(R - b) times the score function of the sample.
            T shapedReward = _ops.Add(reward, _ops.Multiply(_entropyWeight, entropy));

            // Update baseline using exponential moving average
            _baseline = _ops.Add(
                _ops.Multiply(_baselineDecay, _baseline),
                _ops.Multiply(_ops.Subtract(_ops.One, _baselineDecay), shapedReward)
            );

            T scale = _ops.Negate(_ops.Subtract(shapedReward, _baseline));
            var engine = AiDotNetEngine.Current;
            for (int i = 0; i < _controllerGradients.Count; i++)
            {
                engine.Multiply(_sampleLogProbGradients[i], scale).AsSpan().CopyTo(_controllerGradients[i].AsWritableSpan());
            }
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

        /// <inheritdoc />
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
    }
}

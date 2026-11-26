using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.AutoML.SearchSpace;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines.Autodiff;

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

        // LSTM controller parameters (Pham et al. 2018, Sec. 2.1), in the order of ControllerShapes:
        // W_lstm, b_lstm, g_emb (start input), w_emb (operation embeddings), W_soft (operation logits),
        // W_attn1 and W_attn2 and v_attn (the pointer attention that picks a previous node).
        private readonly List<Vector<T>> _controllerWeights;
        private readonly List<Vector<T>> _controllerGradients;

        // Adam state for the controller (the paper trains the policy with Adam, learning rate 0.00035)
        private readonly List<Vector<T>> _controllerMomentum;
        private readonly List<Vector<T>> _controllerVelocity;
        private int _controllerStep;

        // Decisions of the most recently sampled architecture: per node, the previous node then the operation
        private readonly List<int> _lastDecisions;

        // Samples drawn but not yet rewarded, oldest first: their log-probability (as returned to the caller) and
        // their decisions. UpdateController finds its sample here by logProb, so a batch can be sampled, evaluated
        // and then rewarded one by one.
        private readonly List<(T LogProb, List<int> Decisions)> _pendingSamples;
        private const int MaxPendingSamples = 1024;

        // Shared weights for all child models
        private readonly Dictionary<string, Vector<T>> _sharedWeights;
        private readonly Dictionary<string, Vector<T>> _sharedGradients;

        // REINFORCE baseline for variance reduction
        private T _baseline;
        private readonly T _baselineDecay;

        // Controller hyperparameters
        private readonly int _controllerHiddenSize;
        private readonly T _entropyWeight;

        // Search hyperparameters. The defaults (Pham et al. 2018, Sec. 3 and the reference implementation) are
        // uniform(-0.1, 0.1) initialisation, logits squashed as 2.5 * tanh(logits / 5), controller Adam at
        // 0.00035; shared weights trained with SGD (momentum 0.9), learning rate cosine-annealed from 0.05, l2
        // 1e-4, gradient clipping at 5; 50 search steps and 10 derive samples.
        private readonly ENASOptions _options;

        /// <inheritdoc />
        protected override INumericOperations<T> NumOps => _ops;
        /// <inheritdoc />
        protected override SearchSpaceBase<T> NasSearchSpace => _nasSearchSpace;
        /// <inheritdoc />
        protected override int NasNumNodes => _numNodes;

        public ENAS(SearchSpaceBase<T> searchSpace, int numNodes = 4,
            int controllerHiddenSize = 100, double baselineDecay = 0.95, double entropyWeight = 0.01)
            : this(searchSpace, new ENASOptions
            {
                NumNodes = numNodes,
                ControllerHiddenSize = controllerHiddenSize,
                BaselineDecay = baselineDecay,
                EntropyWeight = entropyWeight,
            })
        {
        }

        /// <summary>
        /// Creates an ENAS search with every hyperparameter taken from <paramref name="options"/>.
        /// </summary>
        /// <param name="searchSpace">The operations to search over.</param>
        /// <param name="options">The search hyperparameters; the defaults are the paper's.</param>
        public ENAS(SearchSpaceBase<T> searchSpace, ENASOptions options)
        {
            if (searchSpace is null) throw new ArgumentNullException(nameof(searchSpace));
            if (options is null) throw new ArgumentNullException(nameof(options));
            ValidateOptions(options);

            _ops = MathHelper.GetNumericOperations<T>();
            _options = options;
            _nasSearchSpace = searchSpace;
            _numNodes = options.NumNodes;
            _numOperations = searchSpace.Operations?.Count ?? 5;
            _random = RandomHelper.CreateSeededRandom(options.Seed);

            _controllerHiddenSize = options.ControllerHiddenSize;
            _baselineDecay = _ops.FromDouble(options.BaselineDecay);
            _entropyWeight = _ops.FromDouble(options.EntropyWeight);
            _baseline = _ops.Zero;

            // Every controller parameter starts uniform in [-range, range] (0.1 in the paper).
            double initRange = options.ControllerInitRange;
            _controllerWeights = ControllerShapes()
                .Select(shape => new Vector<T>(Enumerable.Range(0, shape[0] * shape[1])
                    .Select(_ => _ops.FromDouble((_random.NextDouble() * 2.0 - 1.0) * initRange))))
                .ToList();
            _controllerGradients = _controllerWeights.Select(w => new Vector<T>(w.Length)).ToList();
            _controllerMomentum = _controllerWeights.Select(w => new Vector<T>(w.Length)).ToList();
            _controllerVelocity = _controllerWeights.Select(w => new Vector<T>(w.Length)).ToList();
            _lastDecisions = new List<int>();
            _pendingSamples = new List<(T LogProb, List<int> Decisions)>();

            // Initialize shared weights
            _sharedWeights = new Dictionary<string, Vector<T>>();
            _sharedGradients = new Dictionary<string, Vector<T>>();
        }

        /// <summary>The search hyperparameters.</summary>
        public ENASOptions Options => _options;

        private static void ValidateOptions(ENASOptions options)
        {
            if (options.NumNodes < 1)
                throw new ArgumentOutOfRangeException(nameof(options), "NumNodes must be at least 1.");
            if (options.ControllerHiddenSize < 1)
                throw new ArgumentOutOfRangeException(nameof(options), "ControllerHiddenSize must be at least 1.");
            if (options.BaselineDecay < 0.0 || options.BaselineDecay > 1.0)
                throw new ArgumentOutOfRangeException(nameof(options), "BaselineDecay must be in [0, 1].");
            if (options.ControllerTemperature <= 0.0)
                throw new ArgumentOutOfRangeException(nameof(options), "ControllerTemperature must be positive.");
            if (options.ControllerLearningRate <= 0.0)
                throw new ArgumentOutOfRangeException(nameof(options), "ControllerLearningRate must be positive.");
            if (options.SharedMinLearningRate < 0.0 || options.SharedMaxLearningRate < options.SharedMinLearningRate)
                throw new ArgumentOutOfRangeException(nameof(options), "Shared learning rates must satisfy 0 <= min <= max.");
            if (options.SearchSteps < 0)
                throw new ArgumentOutOfRangeException(nameof(options), "SearchSteps must be non-negative.");
            if (options.DeriveSamples < 1)
                throw new ArgumentOutOfRangeException(nameof(options), "DeriveSamples must be at least 1.");
        }

        /// <summary>Shapes of the controller parameters, in <see cref="GetControllerParameters"/> order.</summary>
        private int[][] ControllerShapes()
        {
            int h = _controllerHiddenSize;
            return new[]
            {
                new[] { 2 * h, 4 * h },          // W_lstm: [x; h] -> gates (input, forget, cell, output)
                new[] { 1, 4 * h },              // b_lstm
                new[] { 1, h },                  // g_emb: the first input
                new[] { _numOperations, h },     // w_emb: embedding of a sampled operation
                new[] { h, _numOperations },     // W_soft: operation logits
                new[] { h, h },                  // W_attn1: applied to each anchor
                new[] { h, h },                  // W_attn2: applied to the current hidden state
                new[] { h, 1 },                  // v_attn
            };
        }

        /// <summary>
        /// Samples an architecture using the controller policy
        /// </summary>
        public (Architecture<T> architecture, T logProb, T entropy) SampleArchitecture()
        {
            var (decisions, logProb, entropy) = RunController(ControllerViews(), replay: null);
            _lastDecisions.Clear();
            _lastDecisions.AddRange(decisions);
            T sampleLogProb = logProb.GetFlat(0);
            if (_pendingSamples.Count == MaxPendingSamples)
            {
                _pendingSamples.RemoveAt(0);
            }

            _pendingSamples.Add((sampleLogProb, decisions));

            var architecture = new Architecture<T>();
            foreach (int nodeIdx in Enumerable.Range(1, _numNodes))
            {
                AddOperationToArchitecture(architecture, nodeIdx, decisions[2 * (nodeIdx - 1)], decisions[2 * (nodeIdx - 1) + 1]);
            }

            return (architecture, sampleLogProb, entropy.GetFlat(0));
        }

        /// <summary>Decisions of the most recent <see cref="SampleArchitecture"/> call (previous node, operation per node).</summary>
        internal IReadOnlyList<int> LastDecisions => _lastDecisions;

        /// <summary>
        /// Tensor views over the controller parameter vectors (writes to the vectors are visible through them).
        /// </summary>
        private List<Tensor<T>> ControllerViews()
        {
            var shapes = ControllerShapes();
            return Enumerable.Range(0, shapes.Length)
                .Select(i => Tensor<T>.FromMemory(_controllerWeights[i].AsWritableMemory(), shapes[i]))
                .ToList();
        }

        /// <summary>
        /// Runs the ENAS controller (Pham et al. 2018, Sec. 2.1): an LSTM whose start input is g_emb produces the
        /// anchor of the cell's input node. For each node it then (1) points at a previous node with attention,
        /// logits_j = v^T tanh(W_attn1 h_j + W_attn2 h_t) over the anchors h_j, and feeds that anchor back in;
        /// (2) picks an operation from W_soft h_t and feeds its embedding back in; (3) steps once more to record
        /// the node's anchor. Logits are squashed as 2.5 * tanh(logits / 5). With <paramref name="replay"/> the
        /// given decisions are scored instead of sampled. Returns sum log pi(a) and the summed entropy as [1, 1]
        /// tensors.
        /// </summary>
        private (List<int> Decisions, Tensor<T> LogProb, Tensor<T> Entropy) RunController(List<Tensor<T>> parameters, IReadOnlyList<int>? replay)
        {
            var engine = AiDotNetEngine.Current;
            int h = _controllerHiddenSize;
            var wLstm = parameters[0];
            var bLstm = parameters[1];
            var gEmb = parameters[2];
            var wEmb = parameters[3];
            var wSoft = parameters[4];
            var wAttn1 = parameters[5];
            var wAttn2 = parameters[6];
            var vAttn = parameters[7];
            T inverseTemperature = _ops.FromDouble(1.0 / _options.ControllerTemperature);
            T tanhConstant = _ops.FromDouble(_options.ControllerTanhConstant);

            var decisions = new List<int>();
            var logProb = new Tensor<T>(new[] { 1, 1 });
            var negativeEntropy = new Tensor<T>(new[] { 1, 1 });

            var hidden = new Tensor<T>(new[] { 1, h });
            var cell = new Tensor<T>(new[] { 1, h });
            (hidden, cell) = LstmStep(gEmb, hidden, cell, wLstm, bLstm, h);
            var anchors = new List<Tensor<T>> { hidden };
            var anchorProjections = new List<Tensor<T>> { engine.TensorMatMul(hidden, wAttn1) };
            var input = gEmb;

            for (int node = 1; node <= _numNodes; node++)
            {
                // (1) Previous node: pointer attention over the anchors of nodes 0 .. node - 1.
                (hidden, cell) = LstmStep(input, hidden, cell, wLstm, bLstm, h);
                var projected = anchorProjections.Count == 1
                    ? anchorProjections[0]
                    : engine.TensorConcatenate(anchorProjections.ToArray(), 0);
                var query = engine.TensorTanh(engine.TensorAdd(projected,
                    engine.TensorTile(engine.TensorMatMul(hidden, wAttn2), new[] { anchors.Count, 1 })));
                var prevLogits = engine.Reshape(engine.TensorMatMul(query, vAttn), new[] { 1, anchors.Count });
                int prev = Decide(prevLogits, replay, decisions, inverseTemperature, tanhConstant, ref logProb, ref negativeEntropy);
                input = anchors[prev];

                // (2) Operation: W_soft h_t, then the operation's embedding is the next input.
                (hidden, cell) = LstmStep(input, hidden, cell, wLstm, bLstm, h);
                int op = Decide(engine.TensorMatMul(hidden, wSoft), replay, decisions, inverseTemperature, tanhConstant,
                    ref logProb, ref negativeEntropy);
                input = engine.TensorSlice(wEmb, new[] { op, 0 }, new[] { 1, h });

                // (3) The node's anchor.
                (hidden, cell) = LstmStep(input, hidden, cell, wLstm, bLstm, h);
                anchors.Add(hidden);
                anchorProjections.Add(engine.TensorMatMul(hidden, wAttn1));
                input = gEmb;
            }

            return (decisions, logProb, engine.TensorNegate(negativeEntropy));
        }

        /// <summary>
        /// One controller decision over [1, n] raw logits: squashes them, samples (or replays) a choice, and adds
        /// its log-probability and sum p log p to the running [1, 1] totals.
        /// </summary>
        private int Decide(Tensor<T> rawLogits, IReadOnlyList<int>? replay, List<int> decisions, T inverseTemperature,
            T tanhConstant, ref Tensor<T> logProb, ref Tensor<T> negativeEntropy)
        {
            var engine = AiDotNetEngine.Current;
            var logits = engine.TensorMultiplyScalar(engine.TensorTanh(engine.TensorMultiplyScalar(rawLogits, inverseTemperature)), tanhConstant);
            var logProbabilities = engine.TensorLogSoftmax(logits, 1);
            var probabilities = engine.TensorExp(logProbabilities);

            int choice = replay is null
                ? SampleFromDistribution(probabilities.ToVector().ToList())
                : replay[decisions.Count];
            decisions.Add(choice);

            logProb = engine.TensorAdd(logProb, engine.TensorSlice(logProbabilities, new[] { 0, choice }, new[] { 1, 1 }));
            negativeEntropy = engine.TensorAdd(negativeEntropy,
                engine.ReduceSum(engine.TensorMultiply(probabilities, logProbabilities), new[] { 1 }, true));
            return choice;
        }

        /// <summary>One LSTM cell step on [1, h] row vectors (gate order: input, forget, cell, output).</summary>
        private static (Tensor<T> Hidden, Tensor<T> Cell) LstmStep(Tensor<T> input, Tensor<T> hidden, Tensor<T> cell,
            Tensor<T> wLstm, Tensor<T> bLstm, int h)
        {
            var engine = AiDotNetEngine.Current;
            var gates = engine.TensorAdd(engine.TensorMatMul(engine.TensorConcatenate(new[] { input, hidden }, 1), wLstm), bLstm);
            var inputGate = engine.TensorSigmoid(engine.TensorSlice(gates, new[] { 0, 0 }, new[] { 1, h }));
            var forgetGate = engine.TensorSigmoid(engine.TensorSlice(gates, new[] { 0, h }, new[] { 1, h }));
            var candidate = engine.TensorTanh(engine.TensorSlice(gates, new[] { 0, 2 * h }, new[] { 1, h }));
            var outputGate = engine.TensorSigmoid(engine.TensorSlice(gates, new[] { 0, 3 * h }, new[] { 1, h }));
            var nextCell = engine.TensorAdd(engine.TensorMultiply(forgetGate, cell), engine.TensorMultiply(inputGate, candidate));
            return (engine.TensorMultiply(outputGate, engine.TensorTanh(nextCell)), nextCell);
        }

        /// <summary>log pi(decisions) under the current controller.</summary>
        internal T ReplayLogProbability(IReadOnlyList<int> decisions)
        {
            return RunController(ControllerViews(), decisions).LogProb.GetFlat(0);
        }

        /// <summary>d log pi(decisions) / d theta for every controller parameter, by reverse mode on the tape.</summary>
        internal List<Vector<T>> LogProbabilityGradient(IReadOnlyList<int> decisions)
        {
            var sources = ControllerViews();
            Dictionary<Tensor<T>, Tensor<T>> gradients;
            using (var tape = new GradientTape<T>())
            {
                gradients = tape.ComputeGradients(RunController(sources, decisions).LogProb, sources, false);
            }

            return sources
                .Select(source => gradients.TryGetValue(source, out var gradient)
                    ? gradient.ToVector()
                    : new Vector<T>(source.Length))
                .ToList();
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
        /// Updates the controller with REINFORCE (Williams 1992) as in ENAS (Pham et al. 2018, Sec. 2.2): the
        /// gradient is -(R - b) * d/dtheta sum_t log pi(a_t) over the decisions a_t of the sample whose
        /// <paramref name="logProb"/> is given, where R is the reward plus the entropy bonus and b is an
        /// exponential moving average of R (updated with R first, as in the reference implementation). The
        /// controller then takes an Adam step.
        /// </summary>
        /// <param name="reward">The reward of the sampled architecture (for example its validation accuracy).</param>
        /// <param name="logProb">The log-probability <see cref="SampleArchitecture"/> returned for that architecture.</param>
        /// <param name="entropy">The entropy <see cref="SampleArchitecture"/> returned for that architecture.</param>
        /// <remarks>
        /// Several architectures can be sampled before they are rewarded: each call is matched to its own sample
        /// through <paramref name="logProb"/>, so a reward never reinforces a different architecture.
        /// </remarks>
        /// <exception cref="InvalidOperationException">
        /// <paramref name="logProb"/> is not that of a sample drawn by <see cref="SampleArchitecture"/> and not yet
        /// rewarded.
        /// </exception>
        public void UpdateController(T reward, T logProb, T entropy)
        {
            int pending = _pendingSamples.FindLastIndex(sample => _ops.Equals(sample.LogProb, logProb));
            if (pending < 0)
            {
                throw new InvalidOperationException(
                    "UpdateController must be given the logProb of an architecture drawn by SampleArchitecture that has not been rewarded yet.");
            }

            var decisions = _pendingSamples[pending].Decisions;
            _pendingSamples.RemoveAt(pending);
            ReinforceStep(reward, decisions, entropy);
        }

        /// <summary>The REINFORCE step of <see cref="UpdateController"/> for the given sampled decisions.</summary>
        private void ReinforceStep(T reward, IReadOnlyList<int> decisions, T entropy)
        {
            T shapedReward = _ops.Add(reward, _ops.Multiply(_entropyWeight, entropy));

            // Update baseline using exponential moving average
            _baseline = _ops.Add(
                _ops.Multiply(_baselineDecay, _baseline),
                _ops.Multiply(_ops.Subtract(_ops.One, _baselineDecay), shapedReward)
            );

            if (decisions.Count == 0)
            {
                return;
            }

            T scale = _ops.Negate(_ops.Subtract(shapedReward, _baseline));
            var scoreFunction = LogProbabilityGradient(decisions);
            var engine = AiDotNetEngine.Current;
            for (int i = 0; i < _controllerGradients.Count; i++)
            {
                engine.Multiply(scoreFunction[i], scale).AsSpan().CopyTo(_controllerGradients[i].AsWritableSpan());
            }

            _controllerStep++;
            T learningRate = _ops.FromDouble(_options.ControllerLearningRate);
            T beta1 = _ops.FromDouble(_options.ControllerBeta1);
            T beta2 = _ops.FromDouble(_options.ControllerBeta2);
            T epsilon = _ops.FromDouble(_options.ControllerEpsilon);
            T noWeightDecay = _ops.Zero;
            for (int i = 0; i < _controllerWeights.Count; i++)
            {
                NeuralArchitectureSearch<T>.AdamStep(_controllerWeights[i], _controllerGradients[i], _controllerMomentum[i],
                    _controllerVelocity[i], learningRate, beta1, beta2, epsilon, noWeightDecay, _controllerStep);
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
        /// <remarks>
        /// ENAS (Pham et al. 2018, Sec. 2.2) alternates two phases on one weight-sharing supernet: the shared
        /// weights take an SGD step on the training data for an architecture sampled from the controller, then the
        /// controller takes a REINFORCE step on the validation reward of a fresh sample. The result is the
        /// highest-reward architecture among several sampled from the trained policy.
        /// </remarks>
        protected override Architecture<T> SearchArchitecture(
            Tensor<T> inputs,
            Tensor<T> targets,
            Tensor<T> validationInputs,
            Tensor<T> validationTargets,
            TimeSpan timeLimit,
            CancellationToken cancellationToken)
        {
            if (_nasSearchSpace.Operations is null || _nasSearchSpace.Operations.Count == 0)
            {
                return SampleArchitecture().architecture;
            }

            var deadlineUtc = timeLimit <= TimeSpan.Zero
                ? DateTime.UtcNow
                : DateTime.UtcNow.Add(timeLimit);

            var supernet = new SuperNet<T>(_nasSearchSpace, numNodes: _numNodes);
            var warmupSample = inputs.Rank >= 1 && inputs.Shape[0] > 1
                ? inputs.Slice(axis: 0, start: 0, end: 1).Contiguous()
                : inputs;
            supernet.Predict(warmupSample);

            int searchSteps = _options.SearchSteps;
            T[] learningRates = NeuralArchitectureSearch<T>.CosineAnnealedLearningRates(searchSteps,
                _ops.FromDouble(_options.SharedMaxLearningRate), _ops.FromDouble(_options.SharedMinLearningRate));
            T momentum = _ops.FromDouble(_options.SharedMomentum);
            T weightDecay = _ops.FromDouble(_options.SharedWeightDecay);
            T clipNorm = _ops.FromDouble(_options.SharedGradientClipNorm);
            var momentumBuffers = new Dictionary<string, Vector<T>>();

            foreach (int step in Enumerable.Range(0, searchSteps))
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (DateTime.UtcNow >= deadlineUtc)
                {
                    break;
                }

                // Phase 1: shared weights, trained on the architecture the controller samples.
                ApplyArchitectureToModel(supernet, SampleArchitecture().architecture);
                supernet.BackwardWeights(inputs, targets, supernet.DefaultLossFunction);
                var weights = supernet.GetWeightParameters();
                var gradients = supernet.GetWeightGradients();
                var keys = weights.Keys.ToList();
                T clipScale = NeuralArchitectureSearch<T>.GradientClipScale(keys.Select(key => gradients[key]), clipNorm);
                foreach (var key in keys)
                {
                    if (!momentumBuffers.ContainsKey(key))
                    {
                        momentumBuffers[key] = new Vector<T>(weights[key].Length);
                    }

                    NeuralArchitectureSearch<T>.SgdMomentumStep(weights[key], gradients[key], momentumBuffers[key],
                        learningRates[step], momentum, weightDecay, clipScale);
                }

                // Phase 2: controller, rewarded by the validation quality of a fresh sample.
                var (architecture, logProb, entropy) = SampleArchitecture();
                UpdateController(ValidationReward(supernet, architecture, validationInputs, validationTargets), logProb, entropy);
            }

            var candidates = Enumerable.Range(0, _options.DeriveSamples).Select(_ => SampleArchitecture().architecture).ToList();
            var rewards = new Vector<T>(candidates
                .Select(candidate => ValidationReward(supernet, candidate, validationInputs, validationTargets))
                .ToArray());
            return candidates[NasSamplingHelper.ArgMax(rewards)];
        }

        /// <summary>The reward of an architecture with the shared weights: 1 / (1 + validation loss).</summary>
        private T ValidationReward(SuperNet<T> supernet, Architecture<T> architecture, Tensor<T> inputs, Tensor<T> targets)
        {
            ApplyArchitectureToModel(supernet, architecture);
            T loss = supernet.ComputeValidationLoss(inputs, targets);
            return _ops.Divide(_ops.One, _ops.Add(_ops.One, loss));
        }
    }
}

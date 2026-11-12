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
    /// Partial Channel Connections for Memory-Efficient Differentiable Architecture Search.
    /// PC-DARTS reduces memory consumption by sampling only a subset of channels during the search,
    /// making it more scalable to larger search spaces and datasets.
    ///
    /// Reference: "PC-DARTS: Partial Channel Connections for Memory-Efficient Architecture Search" (ICLR 2020)
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations</typeparam>
    /// <remarks>
    /// <para><b>For Beginners:</b> PC-DARTS makes architecture search more memory-efficient
    /// by only using a subset of channels during the search phase. Regular DARTS uses all
    /// channels which requires huge GPU memory. PC-DARTS samples partial channels, like
    /// tasting a few spoonfuls from a pot instead of drinking the whole thing to judge
    /// the flavor. This enables searching on larger datasets and bigger architectures.</para>
    /// </remarks>
    /// <example>
    /// <code>
    /// // Use AiModelBuilder facade with PC-DARTS for neural architecture search
    /// var trainingData = Tensor&lt;float&gt;.CreateRandom(4, 3, 32, 32);
    /// var trainingLabels = Tensor&lt;float&gt;.CreateRandom(4, 10);
    /// var inputTensor = Tensor&lt;float&gt;.CreateRandom(1, 3, 32, 32);
    /// 
    /// var builder = new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
    ///     .ConfigureModel(new PCDARTS&lt;float&gt;(new MobileNetSearchSpace&lt;float&gt;(), numNodes: 4));
    ///
    /// var result = builder.Build(trainingData, trainingLabels);
    /// var prediction = result.Predict(inputTensor);
    /// </code>
    /// </example>
    [ModelDomain(ModelDomain.MachineLearning)]
    [ModelCategory(ModelCategory.NeuralNetwork)]
    [ModelCategory(ModelCategory.Optimization)]
    [ModelTask(ModelTask.Classification)]
    [ModelTask(ModelTask.FeatureExtraction)]
    [ModelComplexity(ModelComplexity.High)]
    [ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("PC-DARTS: Partial Channel Connections for Memory-Efficient Architecture Search", "https://arxiv.org/abs/1907.05737")]
    public class PCDARTS<T> : NasAutoMLModelBase<T>
    {
        private readonly INumericOperations<T> _ops;
        private readonly SearchSpaceBase<T> _nasSearchSpace;
        private readonly int _numNodes;
        private readonly int _numOperations;
        private readonly Random _random;

        // Architecture parameters
        private readonly List<Matrix<T>> _architectureParams;
        private readonly List<Matrix<T>> _architectureGradients;

        // Edge-normalization parameters beta: one per incoming edge of each node (PC-DARTS Sec. 3.3), trained
        // jointly with alpha by the same optimizer.
        private readonly List<Vector<T>> _edgeParams;
        private readonly List<Vector<T>> _edgeGradients;

        // Adam state of the architecture step, for alpha and beta.
        private readonly List<Matrix<T>> _alphaMomentum;
        private readonly List<Matrix<T>> _alphaVelocity;
        private readonly List<Vector<T>> _edgeMomentum;
        private readonly List<Vector<T>> _edgeVelocity;
        private int _architectureStep;

        // PC-DARTS specific: channel sampling ratio
        private readonly double _channelSamplingRatio;
        private readonly bool _useEdgeNormalization;

        // Search settings (Xu et al. 2020, Sec. 4.2) and the operations derivation may choose: every operation but
        // the zero operation, which marks an absent edge (all of them when the zero operation is the only one).
        private readonly PCDARTSOptions _options;
        private readonly int[] _candidateOperations;

        /// <inheritdoc />
        protected override INumericOperations<T> NumOps => _ops;
        /// <inheritdoc />
        protected override SearchSpaceBase<T> NasSearchSpace => _nasSearchSpace;
        /// <inheritdoc />
        protected override int NasNumNodes => _numNodes;

        /// <summary>
        /// Creates a PC-DARTS search whose partial-channel factor is K = round(1 / <paramref name="channelSamplingRatio"/>)
        /// and whose other settings are the paper's defaults (<see cref="PCDARTSOptions"/>).
        /// </summary>
        /// <param name="searchSpace">The search space defining the candidate operations.</param>
        /// <param name="numNodes">The number of intermediate nodes of the searched cell.</param>
        /// <param name="channelSamplingRatio">The fraction 1/K of channels each edge's operation mixture sees, in (0, 1].</param>
        /// <param name="useEdgeNormalization">Whether to weight the edges by softmax(beta) (PC-DARTS Sec. 3.3).</param>
        public PCDARTS(SearchSpaceBase<T> searchSpace, int numNodes = 4,
            double channelSamplingRatio = 0.25, bool useEdgeNormalization = true)
            : this(searchSpace, OptionsForSamplingRatio(channelSamplingRatio), numNodes, useEdgeNormalization)
        {
        }

        /// <summary>
        /// Creates a PC-DARTS search with the given search settings.
        /// </summary>
        /// <param name="searchSpace">The search space defining the candidate operations.</param>
        /// <param name="options">The search settings; the defaults are those of Xu et al. 2020 (Sec. 4.2).</param>
        /// <param name="numNodes">The number of intermediate nodes of the searched cell.</param>
        /// <param name="useEdgeNormalization">Whether to weight the edges by softmax(beta) (PC-DARTS Sec. 3.3).</param>
        public PCDARTS(SearchSpaceBase<T> searchSpace, PCDARTSOptions options)
            : this(searchSpace, options, 4, true)
        {
        }

        /// <summary>
        /// Creates a PC-DARTS search with the given search settings, cell size and edge normalization.
        /// </summary>
        /// <param name="searchSpace">The search space defining the candidate operations.</param>
        /// <param name="options">The search settings; the defaults are those of Xu et al. 2020 (Sec. 4.2).</param>
        /// <param name="numNodes">The number of intermediate nodes of the searched cell.</param>
        /// <param name="useEdgeNormalization">Whether to weight the edges by softmax(beta) (PC-DARTS Sec. 3.3).</param>
        public PCDARTS(SearchSpaceBase<T> searchSpace, PCDARTSOptions options, int numNodes,
            bool useEdgeNormalization)
        {
            if (searchSpace is null)
                throw new ArgumentNullException(nameof(searchSpace));
            if (numNodes <= 0)
                throw new ArgumentOutOfRangeException(nameof(numNodes), numNodes, "The cell needs at least one node.");
            ValidateOptions(options);

            _ops = MathHelper.GetNumericOperations<T>();
            _nasSearchSpace = searchSpace;
            _numNodes = numNodes;

            // The same operation list the search's SuperNet uses: the search space's, or its five defaults when the
            // search space lists none.
            var operationNames = SuperNet<T>.OperationNamesOf(searchSpace);
            _numOperations = operationNames.Count;
            var nonZero = Enumerable.Range(0, _numOperations)
                .Where(op => !SuperNet<T>.IsZeroOperationName(operationNames[op])).ToArray();
            _candidateOperations = nonZero.Length > 0 ? nonZero : Enumerable.Range(0, _numOperations).ToArray();
            _random = RandomHelper.CreateSeededRandom(42);

            _options = options;
            _channelSamplingRatio = 1.0 / options.PartialChannelK;
            _useEdgeNormalization = useEdgeNormalization;

            // Initialize architecture parameters
            _architectureParams = new List<Matrix<T>>();
            _architectureGradients = new List<Matrix<T>>();

            for (int i = 0; i < _numNodes; i++)
            {
                var alpha = new Matrix<T>(i + 1, _numOperations);
                for (int j = 0; j < alpha.Rows; j++)
                {
                    for (int k = 0; k < alpha.Columns; k++)
                    {
                        alpha[j, k] = _ops.FromDouble((_random.NextDouble() - 0.5) * 0.2);
                    }
                }
                _architectureParams.Add(alpha);
                _architectureGradients.Add(new Matrix<T>(i + 1, _numOperations));
            }

            // beta starts at 1e-3 * N(0, 1) like the reference implementation; drawn after alpha so the alpha
            // stream is unchanged, one seed per node from the search's Random.
            _edgeParams = new List<Vector<T>>();
            var engine = AiDotNetEngine.Current;
            for (int i = 0; i < _numNodes; i++)
            {
                var beta = new Tensor<T>(new[] { i + 1 });
                engine.TensorNormalInPlace(beta, 0.0, 1e-3, _random.Next());
                _edgeParams.Add(beta.ToVector());
            }

            _edgeGradients = _edgeParams.Select(beta => new Vector<T>(beta.Length)).ToList();
            _alphaMomentum = _architectureParams.Select(alpha => new Matrix<T>(alpha.Rows, alpha.Columns)).ToList();
            _alphaVelocity = _architectureParams.Select(alpha => new Matrix<T>(alpha.Rows, alpha.Columns)).ToList();
            _edgeMomentum = _edgeParams.Select(beta => new Vector<T>(beta.Length)).ToList();
            _edgeVelocity = _edgeParams.Select(beta => new Vector<T>(beta.Length)).ToList();
        }

        /// <summary>The search settings.</summary>
        public PCDARTSOptions SearchOptions => _options;

        /// <summary>Default settings whose partial-channel factor is K = round(1 / ratio).</summary>
        private static PCDARTSOptions OptionsForSamplingRatio(double channelSamplingRatio)
        {
            if (!(channelSamplingRatio > 0.0 && channelSamplingRatio <= 1.0))
                throw new ArgumentOutOfRangeException(nameof(channelSamplingRatio), channelSamplingRatio,
                    "The channel sampling ratio must be in (0, 1].");

            return new PCDARTSOptions { PartialChannelK = Math.Max(1, (int)Math.Round(1.0 / channelSamplingRatio)) };
        }

        private static void ValidateOptions(PCDARTSOptions options)
        {
            if (options is null)
                throw new ArgumentNullException(nameof(options));
            if (options.Epochs < 0 || options.WarmupEpochs < 0)
                throw new ArgumentOutOfRangeException(nameof(options), "Epochs and warm-up epochs must be non-negative.");
            if (options.BatchSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(options), "The batch size must be positive.");
            if (options.PartialChannelK < 1)
                throw new ArgumentOutOfRangeException(nameof(options), "The partial-channel factor K must be at least 1.");
            if (options.ArchitectureLearningRate <= 0.0 || options.WeightMinLearningRate < 0.0
                || options.WeightMaxLearningRate < options.WeightMinLearningRate)
                throw new ArgumentOutOfRangeException(nameof(options),
                    "Learning rates must satisfy architecture > 0 and 0 <= weight min <= weight max.");
        }

        /// <summary>
        /// Samples a subset of channels for partial channel connections
        /// </summary>
        public List<int> SampleChannels(int totalChannels)
        {
            int numSampledChannels = Math.Max(1, (int)(totalChannels * _channelSamplingRatio));
            var allChannels = Enumerable.Range(0, totalChannels).ToList();

            for (int i = allChannels.Count - 1; i > 0; i--)
            {
                int j = _random.Next(i + 1);
                (allChannels[i], allChannels[j]) = (allChannels[j], allChannels[i]);
            }

            var sampledChannels = allChannels.Take(numSampledChannels).ToList();
            sampledChannels.Sort();
            return sampledChannels;
        }

        /// <summary>
        /// Applies edge normalization to prevent operation collapse: the operation mixture of edge (i, j) is
        /// weighted by softmax over the incoming edges of node j of the edge parameters beta, so the final
        /// weight is softmax(beta)_i * softmax(alpha_i)_o (PC-DARTS, Xu et al. 2020, Sec. 3.3, Eq. 4).
        /// </summary>
        public Matrix<T> ApplyEdgeNormalization(Matrix<T> alpha)
        {
            if (!_useEdgeNormalization)
                return NasSamplingHelper.SoftmaxRows(alpha, _ops);

            var result = NasSamplingHelper.SoftmaxRows(alpha, _ops);
            if (alpha.Rows == 0 || alpha.Columns == 0)
            {
                return result;
            }

            // A matrix with k rows holds the k incoming edges of node k - 1; any other shape has no learned beta
            // and gets the uniform edge weights beta = 0 starts from.
            var engine = AiDotNetEngine.Current;
            var beta = alpha.Rows <= _edgeParams.Count ? _edgeParams[alpha.Rows - 1] : new Vector<T>(alpha.Rows);
            var edgeWeights = engine.Softmax(beta).AsTensor().Reshape(new[] { alpha.Rows, 1 });
            return engine.TensorMultiply(Tensor<T>.FromMatrix(result),
                engine.TensorBroadcastTo(edgeWeights, new[] { alpha.Rows, alpha.Columns })).ToMatrix();
        }

        /// <summary>
        /// Derives the discrete architecture
        /// </summary>
        public Architecture<T> DeriveArchitecture()
        {
            var architecture = new Architecture<T>();

            for (int nodeIdx = 0; nodeIdx < _numNodes; nodeIdx++)
            {
                var alpha = _architectureParams[nodeIdx];
                var weights = ApplyEdgeNormalization(alpha);

                // Select top-2 edges for each intermediate node (following DARTS convention)
                var edgeScores = new List<(int prevNode, int opIdx, T score)>();

                // The strongest non-zero operation of each edge, scored by its softmax(beta)-weighted softmax(alpha)
                // weight (PC-DARTS reference genotype, after DARTS Sec. 2.4): the zero operation marks an absent
                // edge, so neither the operation choice nor the edge ranking considers it.
                var candidateWeights = _candidateOperations.Length == weights.Columns
                    ? Tensor<T>.FromMatrix(weights)
                    : AiDotNetEngine.Current.TensorIndexSelect(Tensor<T>.FromMatrix(weights),
                        new Tensor<int>(_candidateOperations, new[] { _candidateOperations.Length }), 1);
                int[] bestOps = NasSamplingHelper.ArgMaxRows(candidateWeights, out T[] bestWeights);

                for (int prevNodeIdx = 0; prevNodeIdx <= nodeIdx; prevNodeIdx++)
                {
                    edgeScores.Add((prevNodeIdx, _candidateOperations[bestOps[prevNodeIdx]], bestWeights[prevNodeIdx]));
                }

                // Select top-2 edges
                edgeScores.Sort((a, b) => CompareDescending(a.score, b.score));
                int numEdgesToKeep = Math.Min(2, edgeScores.Count);

                for (int i = 0; i < numEdgesToKeep; i++)
                {
                    var (prevNode, opIdx, _) = edgeScores[i];
                    if (_nasSearchSpace.Operations != null && opIdx < _nasSearchSpace.Operations.Count)
                    {
                        var operation = _nasSearchSpace.Operations[opIdx];
                        architecture.AddOperation(nodeIdx + 1, prevNode, operation);
                    }
                }
            }

            return architecture;
        }

        private int CompareDescending(T left, T right)
        {
            if (_ops.GreaterThan(left, right))
            {
                return -1;
            }

            if (_ops.LessThan(left, right))
            {
                return 1;
            }

            return 0;
        }

        /// <summary>
        /// Gets the memory savings ratio compared to standard DARTS
        /// </summary>
        public double GetMemorySavingsRatio()
        {
            return 1.0 - _channelSamplingRatio;
        }

        /// <summary>
        /// Gets architecture parameters for optimization
        /// </summary>
        public List<Matrix<T>> GetArchitectureParameters() => _architectureParams;

        /// <summary>
        /// Gets architecture gradients
        /// </summary>
        public List<Matrix<T>> GetArchitectureGradients() => _architectureGradients;

        /// <summary>
        /// Gets the edge-normalization parameters beta (one vector per node, one entry per incoming edge). They are
        /// architecture parameters: <see cref="UpdateArchitectureParameters"/> trains them together with alpha.
        /// </summary>
        public List<Vector<T>> GetEdgeParameters() => _edgeParams;

        /// <summary>
        /// Gets the accumulated gradients of the edge-normalization parameters beta.
        /// </summary>
        public List<Vector<T>> GetEdgeGradients() => _edgeGradients;

        /// <summary>
        /// Back-propagates dL/dW, the loss gradient with respect to the operation weights of one node, through the
        /// edge normalization W = softmax(beta)_i * softmax(alpha_i)_o (PC-DARTS Eq. 4; plain softmax(alpha_i)_o
        /// without edge normalization), and adds the result to the alpha and beta gradients of that node.
        /// </summary>
        /// <param name="nodeIndex">The node (0-based); its alpha has nodeIndex + 1 rows, one per incoming edge.</param>
        /// <param name="weightGradient">dL/dW, with the shape of that node's alpha.</param>
        public void AccumulateArchitectureGradients(int nodeIndex, Matrix<T> weightGradient)
        {
            if (nodeIndex < 0 || nodeIndex >= _numNodes)
                throw new ArgumentOutOfRangeException(nameof(nodeIndex));
            if (weightGradient is null)
                throw new ArgumentNullException(nameof(weightGradient));

            var alphaMatrix = _architectureParams[nodeIndex];
            if (weightGradient.Rows != alphaMatrix.Rows || weightGradient.Columns != alphaMatrix.Columns)
                throw new ArgumentException("The weight gradient must have the shape of the node's alpha.", nameof(weightGradient));

            var engine = AiDotNetEngine.Current;
            var alpha = Tensor<T>.FromMatrix(alphaMatrix);
            var beta = _edgeParams[nodeIndex].AsTensor().Reshape(new[] { alphaMatrix.Rows, 1 });
            var upstream = Tensor<T>.FromMatrix(weightGradient);
            var sources = new List<Tensor<T>> { alpha, beta };

            Dictionary<Tensor<T>, Tensor<T>> gradients;
            using (var tape = new GradientTape<T>())
            {
                var weights = engine.TensorSoftmax(alpha, 1);
                if (_useEdgeNormalization)
                {
                    weights = engine.TensorMultiply(weights, engine.TensorBroadcastTo(engine.TensorSoftmax(beta, 0),
                        new[] { alphaMatrix.Rows, alphaMatrix.Columns }));
                }

                var surrogate = engine.ReduceSum(engine.TensorMultiply(weights, upstream), new[] { 0, 1 }, true);
                gradients = tape.ComputeGradients(surrogate, sources, false);
            }

            if (gradients.TryGetValue(alpha, out var alphaGradient))
            {
                var accumulated = engine.TensorAdd(Tensor<T>.FromMatrix(_architectureGradients[nodeIndex]),
                    alphaGradient.Reshape(new[] { alphaMatrix.Rows, alphaMatrix.Columns }));
                accumulated.ToVector().AsSpan().CopyTo(_architectureGradients[nodeIndex].AsWritableSpan());
            }

            if (_useEdgeNormalization && gradients.TryGetValue(beta, out var betaGradient))
            {
                var accumulated = engine.TensorAdd(_edgeGradients[nodeIndex].AsTensor(),
                    betaGradient.Reshape(new[] { alphaMatrix.Rows }));
                accumulated.ToVector().AsSpan().CopyTo(_edgeGradients[nodeIndex].AsWritableSpan());
            }
        }

        /// <summary>
        /// One architecture step: Adam on alpha and beta together with their accumulated gradients, which are then
        /// cleared. The defaults are PC-DARTS' architecture optimizer (Xu et al. 2020, Sec. 4.2): Adam with
        /// learning rate 6e-4, betas (0.5, 0.999) and weight decay 1e-3.
        /// </summary>
        public void UpdateArchitectureParameters(double learningRate = 6e-4, double beta1 = 0.5, double beta2 = 0.999,
            double weightDecay = 1e-3, double epsilon = 1e-8)
        {
            _architectureStep++;
            T lr = _ops.FromDouble(learningRate);
            T b1 = _ops.FromDouble(beta1);
            T b2 = _ops.FromDouble(beta2);
            T eps = _ops.FromDouble(epsilon);
            T wd = _ops.FromDouble(weightDecay);
            for (int i = 0; i < _numNodes; i++)
            {
                NeuralArchitectureSearch<T>.AdamStep(_architectureParams[i], _architectureGradients[i],
                    _alphaMomentum[i], _alphaVelocity[i], lr, b1, b2, eps, wd, _architectureStep);
                _architectureGradients[i].AsWritableSpan().Clear();

                if (_useEdgeNormalization)
                {
                    NeuralArchitectureSearch<T>.AdamStep(_edgeParams[i], _edgeGradients[i],
                        _edgeMomentum[i], _edgeVelocity[i], lr, b1, b2, eps, wd, _architectureStep);
                }

                _edgeGradients[i].AsWritableSpan().Clear();
            }
        }

        /// <summary>
        /// Gets the channel sampling ratio
        /// </summary>
        public double GetChannelSamplingRatio() => _channelSamplingRatio;

        /// <inheritdoc />
        protected override Architecture<T> SearchArchitecture(
            Tensor<T> inputs,
            Tensor<T> targets,
            Tensor<T> validationInputs,
            Tensor<T> validationTargets,
            TimeSpan timeLimit,
            CancellationToken cancellationToken)
            => RunSearch(inputs, targets, validationInputs, validationTargets, timeLimit, cancellationToken);

        /// <summary>
        /// The PC-DARTS search (Xu et al. 2020, Sec. 3 and 4.2), the first-order DARTS alternation (Liu et al. 2019,
        /// Algorithm 1 with xi = 0) on a cell with partial channel connections and edge normalization: each step
        /// first updates alpha and beta on a validation batch with Adam, then the network weights on a training
        /// batch with SGD. During the first <see cref="PCDARTSOptions.WarmupEpochs"/> epochs only the weights are
        /// updated. The architecture is then derived from softmax(alpha) weighted by softmax(beta).
        /// </summary>
        /// <param name="inputs">The training inputs, one sample per row of axis 0.</param>
        /// <param name="targets">The training targets, with the cell output's shape.</param>
        /// <param name="validationInputs">The validation inputs the architecture step uses.</param>
        /// <param name="validationTargets">The validation targets.</param>
        /// <param name="timeLimit">A positive limit stops the search early; zero or negative imposes none.</param>
        /// <param name="cancellationToken">Cancels the search.</param>
        internal Architecture<T> RunSearch(
            Tensor<T> inputs,
            Tensor<T> targets,
            Tensor<T> validationInputs,
            Tensor<T> validationTargets,
            TimeSpan timeLimit,
            CancellationToken cancellationToken)
        {
            if (inputs is null) throw new ArgumentNullException(nameof(inputs));
            if (targets is null) throw new ArgumentNullException(nameof(targets));
            if (validationInputs is null) throw new ArgumentNullException(nameof(validationInputs));
            if (validationTargets is null) throw new ArgumentNullException(nameof(validationTargets));
            int trainCount = SampleCount(inputs, targets, nameof(targets));
            int validationCount = SampleCount(validationInputs, validationTargets, nameof(validationTargets));

            if (_options.Epochs == 0 || trainCount == 0)
                return DeriveArchitecture();

            var supernet = new SuperNet<T>(_nasSearchSpace, numNodes: _numNodes);
            var weights = supernet.GetWeightParameters();
            var weightKeys = weights.Keys.ToList();
            var weightSources = weightKeys.Select(supernet.GetWeightTensor).ToList();
            var weightVectors = weightKeys.Select(key => weights[key]).ToList();
            var weightBuffers = weightVectors.Select(w => new Vector<T>(w.Length)).ToList();

            // Tape views over the stored alpha and beta: the Adam step writes the stored values in place, so the
            // views always hold the current architecture.
            var alphaViews = _architectureParams
                .Select(alpha => Tensor<T>.FromMemory(alpha.AsWritableMemory(), new[] { alpha.Rows, alpha.Columns }))
                .ToList();
            var betaViews = _edgeParams
                .Select(beta => Tensor<T>.FromMemory(beta.AsWritableMemory(), new[] { beta.Length }))
                .ToList();
            IReadOnlyList<Tensor<T>>? edgeViews = _useEdgeNormalization ? betaViews : null;

            T[] learningRates = NeuralArchitectureSearch<T>.CosineAnnealedLearningRates(_options.Epochs,
                _ops.FromDouble(_options.WeightMaxLearningRate), _ops.FromDouble(_options.WeightMinLearningRate));
            T momentum = _ops.FromDouble(_options.WeightMomentum);
            T weightDecay = _ops.FromDouble(_options.WeightDecay);
            T clipNorm = _ops.FromDouble(_options.GradientClipNorm);

            int batchSize = _options.BatchSize;
            int trainBatches = (trainCount + batchSize - 1) / batchSize;
            int validationBatches = (validationCount + batchSize - 1) / batchSize;
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();

            for (int epoch = 0; epoch < _options.Epochs; epoch++)
            {
                bool updateArchitecture = epoch >= _options.WarmupEpochs && validationCount > 0;
                int[] trainOrder = Permutation(trainCount);
                int[] validationOrder = updateArchitecture ? Permutation(validationCount) : Array.Empty<int>();
                T learningRate = learningRates[epoch];

                for (int step = 0; step < trainBatches; step++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    if (timeLimit > TimeSpan.Zero && stopwatch.Elapsed >= timeLimit)
                        return DeriveArchitecture();

                    if (updateArchitecture)
                    {
                        var validationRows = BatchIndices(validationOrder, step % validationBatches, batchSize);
                        ArchitectureStep(supernet, Rows(validationInputs, validationRows),
                            Rows(validationTargets, validationRows), alphaViews, betaViews, edgeViews);
                    }

                    var trainRows = BatchIndices(trainOrder, step, batchSize);
                    WeightStep(supernet, Rows(inputs, trainRows), Rows(targets, trainRows), alphaViews, edgeViews,
                        weightSources, weightVectors, weightBuffers, learningRate, momentum, weightDecay, clipNorm);
                }
            }

            return DeriveArchitecture();
        }

        /// <summary>
        /// The architecture step: the gradient of the validation loss with respect to alpha and beta (first order,
        /// at the current weights), then Adam on both with the options' settings.
        /// </summary>
        private void ArchitectureStep(SuperNet<T> supernet, Tensor<T> input, Tensor<T> target,
            List<Tensor<T>> alphaViews, List<Tensor<T>> betaViews, IReadOnlyList<Tensor<T>>? edgeViews)
        {
            var engine = AiDotNetEngine.Current;
            var sources = edgeViews is null ? alphaViews : alphaViews.Concat(betaViews).ToList();
            using (var tape = new GradientTape<T>())
            {
                var loss = supernet.PartialChannelLoss(input, target, alphaViews, edgeViews, _options.PartialChannelK);
                var gradients = tape.ComputeGradients(loss, sources, false);
                for (int node = 0; node < _numNodes; node++)
                {
                    if (gradients.TryGetValue(alphaViews[node], out var alphaGradient))
                    {
                        var destination = _architectureGradients[node];
                        engine.TensorAdd(Tensor<T>.FromMatrix(destination),
                                alphaGradient.Reshape(new[] { destination.Rows, destination.Columns }))
                            .ToVector().AsSpan().CopyTo(destination.AsWritableSpan());
                    }

                    if (edgeViews is not null && gradients.TryGetValue(betaViews[node], out var betaGradient))
                    {
                        var destination = _edgeGradients[node];
                        engine.TensorAdd(destination.AsTensor(), betaGradient.Reshape(new[] { destination.Length }))
                            .ToVector().AsSpan().CopyTo(destination.AsWritableSpan());
                    }
                }
            }

            UpdateArchitectureParameters(_options.ArchitectureLearningRate, _options.ArchitectureBeta1,
                _options.ArchitectureBeta2, _options.ArchitectureWeightDecay, _options.ArchitectureEpsilon);
        }

        /// <summary>
        /// The weight step: the gradient of the training loss with respect to the network weights at the current
        /// alpha and beta, clipped to the options' global norm, then SGD with momentum and weight decay.
        /// </summary>
        private void WeightStep(SuperNet<T> supernet, Tensor<T> input, Tensor<T> target,
            List<Tensor<T>> alphaViews, IReadOnlyList<Tensor<T>>? edgeViews, List<Tensor<T>> weightSources,
            List<Vector<T>> weightVectors, List<Vector<T>> weightBuffers, T learningRate, T momentum, T weightDecay,
            T clipNorm)
        {
            // The update runs inside the tape's scope, while the gradients' storage is still live.
            using (var tape = new GradientTape<T>())
            {
                var loss = supernet.PartialChannelLoss(input, target, alphaViews, edgeViews, _options.PartialChannelK);
                var gradients = tape.ComputeGradients(loss, weightSources, false);
                var weightGradients = weightSources
                    .Select((source, i) => gradients.TryGetValue(source, out var gradient)
                        ? gradient.Reshape(new[] { weightVectors[i].Length }).ToVector()
                        : new Vector<T>(weightVectors[i].Length))
                    .ToList();

                T clipScale = NeuralArchitectureSearch<T>.GradientClipScale(weightGradients, clipNorm);
                for (int i = 0; i < weightVectors.Count; i++)
                {
                    NeuralArchitectureSearch<T>.SgdMomentumStep(weightVectors[i], weightGradients[i], weightBuffers[i],
                        learningRate, momentum, weightDecay, clipScale);
                }
            }
        }

        /// <summary>The number of samples (axis 0) of an input tensor, checked against its targets.</summary>
        private static int SampleCount(Tensor<T> data, Tensor<T> labels, string labelsName)
        {
            if (data.Rank < 1 || labels.Rank < 1 || data.Shape[0] != labels.Shape[0])
                throw new ArgumentException("Inputs and targets must have the same number of samples along axis 0.",
                    labelsName);

            return data.Shape[0];
        }

        /// <summary>A uniformly random order of <paramref name="count"/> samples (Fisher-Yates on the search's Random).</summary>
        private int[] Permutation(int count)
        {
            var order = Enumerable.Range(0, count).ToArray();
            for (int i = count - 1; i > 0; i--)
            {
                int j = _random.Next(i + 1);
                (order[i], order[j]) = (order[j], order[i]);
            }

            return order;
        }

        /// <summary>The sample indices of mini-batch <paramref name="batch"/> of a shuffled order.</summary>
        private static int[] BatchIndices(int[] order, int batch, int batchSize)
        {
            int start = batch * batchSize;
            int length = Math.Min(batchSize, order.Length - start);
            var indices = new int[length];
            Array.Copy(order, start, indices, 0, length);
            return indices;
        }

        /// <summary>The samples <paramref name="indices"/> of a tensor, gathered along axis 0.</summary>
        private static Tensor<T> Rows(Tensor<T> data, int[] indices)
            => AiDotNetEngine.Current.TensorIndexSelect(data, new Tensor<int>(indices, new[] { indices.Length }), 0);
    }
}

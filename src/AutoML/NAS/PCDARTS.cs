using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.AutoML.SearchSpace;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
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

        /// <inheritdoc />
        protected override INumericOperations<T> NumOps => _ops;
        /// <inheritdoc />
        protected override SearchSpaceBase<T> NasSearchSpace => _nasSearchSpace;
        /// <inheritdoc />
        protected override int NasNumNodes => _numNodes;

        public PCDARTS(SearchSpaceBase<T> searchSpace, int numNodes = 4,
            double channelSamplingRatio = 0.25, bool useEdgeNormalization = true)
        {
            _ops = MathHelper.GetNumericOperations<T>();
            _nasSearchSpace = searchSpace;
            _numNodes = numNodes;
            _numOperations = searchSpace.Operations?.Count ?? 5;
            _random = RandomHelper.CreateSeededRandom(42);

            _channelSamplingRatio = channelSamplingRatio;
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

                int[] bestOps = NasSamplingHelper.ArgMaxRows(weights, out T[] bestWeights);

                for (int prevNodeIdx = 0; prevNodeIdx <= nodeIdx; prevNodeIdx++)
                {
                    edgeScores.Add((prevNodeIdx, bestOps[prevNodeIdx], bestWeights[prevNodeIdx]));
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
        {
            return DeriveArchitecture();
        }
    }
}

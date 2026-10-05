using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// One graph-convolution step of PICK's GLCN (Yu et al., ICPR 2020; reference <c>GCNLayer</c>) over
/// node-edge-node triplets:
/// <c>H_ij = ReLU(x_i W_vi + x_j W_vj + alpha_ij + b)</c>, <c>x_i' = ReLU((sum_j A_ij H_ij) W_node)</c>,
/// <c>alpha_ij' = ReLU(H_ij W_alpha)</c>.
/// </summary>
/// <remarks>
/// <para>The single-input <see cref="LayerBase{T}.Forward(Tensor{T})"/> takes nodes <c>[N, D]</c> and uses zero
/// relation embeddings with a uniform adjacency. PICK drives <see cref="Forward(Tensor{T}, Tensor{T}, Tensor{T})"/>
/// with its relation embedding and the learned soft adjacency.</para>
/// <para><b>For Beginners:</b> Each text box updates itself from every other box, weighted by how strongly the
/// learned graph connects them and by where the boxes sit relative to each other.</para>
/// </remarks>
[LayerCategory(LayerCategory.Graph)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 2, Cost = ComputeCost.Medium, TestInputShape = "4, 8", TestConstructorArgs = "8, 8")]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input, Note = "One row per graph node (text segment).")]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class PickGcnLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _inDim;
    private readonly int _outDim;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _wVi;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _wVj;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]
    private Tensor<T> _biasH;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _wNode;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _wAlpha;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>Creates the layer. The reference initializes with torch's kaiming_uniform(a = sqrt 5) and b ~ U(0, 1).</summary>
    public PickGcnLayer([LayerState] int inDim, [LayerState] int outDim)
        : base(new[] { -1, inDim }, new[] { -1, outDim })
    {
        if (inDim <= 0) throw new ArgumentOutOfRangeException(nameof(inDim));
        if (outDim <= 0) throw new ArgumentOutOfRangeException(nameof(outDim));
        _inDim = inDim;
        _outDim = outDim;
        var random = LayerInitializationSeedScope.NextRandom();
        _wVi = KaimingUniform(inDim, inDim, random);
        _wVj = KaimingUniform(inDim, inDim, random);
        _biasH = new Tensor<T>(new[] { inDim });
        for (int i = 0; i < inDim; i++) _biasH[i] = NumOps.FromDouble(random.NextDouble());
        _wNode = KaimingUniform(inDim, outDim, random);
        _wAlpha = KaimingUniform(inDim, outDim, random);
        RegisterTrainableParameter(_wVi, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_wVj, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_biasH, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_wNode, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_wAlpha, PersistentTensorRole.Weights);
    }

    /// <inheritdoc/>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => inputRank == 2
        ? new[]
        {
            new OutputAxisContract(TensorAxis.Time, AxisRelation.Same(TensorAxis.Time)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_outDim)),
        }
        : null;

    /// <summary>
    /// torch <c>kaiming_uniform_(w, a=sqrt(5))</c> on a <c>[rows, cols]</c> parameter: torch reads fan_in as
    /// <c>size(1)</c>, so the bound is <c>1 / sqrt(cols)</c>.
    /// </summary>
    internal Tensor<T> KaimingUniform(int rows, int cols, Random random)
    {
        var tensor = new Tensor<T>(new[] { rows, cols });
        double bound = 1.0 / Math.Sqrt(cols);
        for (int i = 0; i < tensor.Length; i++) tensor[i] = NumOps.FromDouble(((random.NextDouble() * 2) - 1) * bound);
        return tensor;
    }

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        int n = input.Shape[0];
        var alpha = new Tensor<T>(new[] { n, n, _inDim });
        var adjacency = new Tensor<T>(new[] { n, n });
        adjacency.Fill(NumOps.FromDouble(1.0 / n));
        return Forward(input, alpha, adjacency).Nodes;
    }

    /// <summary>Runs the step on nodes <c>[N, D]</c>, relation embeddings <c>[N, N, D]</c> and adjacency <c>[N, N]</c>.</summary>
    internal (Tensor<T> Nodes, Tensor<T> Alpha) Forward(Tensor<T> nodes, Tensor<T> alpha, Tensor<T> adjacency)
    {
        int n = nodes.Shape[0];
        if (nodes.Rank != 2 || nodes.Shape[1] != _inDim)
            throw new ArgumentException($"PickGcnLayer expects rank-2 nodes of shape [N, {_inDim}]; got shape [{string.Join(", ", nodes.Shape.ToArray())}] (feature dimension mismatch).", nameof(nodes));
        var xi = Engine.TensorBroadcastTo(Engine.Reshape(Engine.TensorMatMul(nodes, _wVi), new[] { n, 1, _inDim }), new[] { n, n, _inDim });
        var xj = Engine.TensorBroadcastTo(Engine.Reshape(Engine.TensorMatMul(nodes, _wVj), new[] { 1, n, _inDim }), new[] { n, n, _inDim });
        var bias = Engine.TensorBroadcastTo(Engine.Reshape(_biasH, new[] { 1, 1, _inDim }), new[] { n, n, _inDim });
        var h = Engine.ReLU(Engine.TensorAdd(Engine.TensorAdd(xi, xj), Engine.TensorAdd(alpha, bias)));
        // sum_j A_ij H_ij as a batched [1, N] x [N, D] product per node i.
        var aggregated = Engine.Reshape(
            Engine.TensorBatchMatMul<T>(Engine.Reshape(adjacency, new[] { n, 1, n }), h), new[] { n, _inDim });
        var newNodes = Engine.ReLU(Engine.TensorMatMul(aggregated, _wNode));
        var newAlpha = Engine.ReLU(Engine.Reshape(Engine.TensorMatMul(Engine.Reshape(h, new[] { n * n, _inDim }), _wAlpha), new[] { n, n, _outDim }));
        return (newNodes, newAlpha);
    }

    /// <inheritdoc/>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["InDim"] = _inDim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["OutDim"] = _outDim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    /// <inheritdoc/>
    public override void ResetState() { }
}

/// <summary>
/// PICK's graph learning-convolutional network (reference <c>GLCN</c>): it learns a soft adjacency over the
/// document's text segments from their embeddings, then runs <c>numLayers</c> <see cref="PickGcnLayer{T}"/>
/// steps seeded by an embedding of the six box-relation features.
/// </summary>
/// <remarks>
/// <para>Graph learning (reference <c>GraphLearningLayer</c>):
/// <list type="bullet">
/// <item><c>x_hat = x P</c>, with no bias and <c>learningDim</c> columns.</item>
/// <item><c>A = softmax_j(LeakyReLU(w . |x_hat_i - x_hat_j|)) + 1e-10</c>.</item>
/// <item>The graph-learning loss is
/// <c>sum_ij exp(A_ij + eta ||x_hat_i - x_hat_j||) / N^2 + gamma ||A||_F</c>.</item>
/// </list>
/// The pairwise distance is computed as <c>sqrt(s_ij + I_ij) - I_ij</c>. That is the same value, and it keeps
/// the gradient finite on the diagonal, where the reference's norm of an exact zero vector has an
/// infinite derivative.</para>
/// <para><b>For Beginners:</b> This decides which text boxes on a page should talk to each other, then lets them
/// exchange information so that, for example, a "Total:" label informs the number beside it.</para>
/// </remarks>
[LayerCategory(LayerCategory.Graph)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = false, ExpectedInputRank = 2, Cost = ComputeCost.Medium, TestInputShape = "4, 8", TestConstructorArgs = "8, 4, 1, 1.0, 1.0")]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input, Note = "One row per graph node (text segment).")]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class PickGraphLayer<T> : LayerBase<T>, IShapeContract
{
    /// <summary>Box-relation features per node pair (reference: x_ij, y_ij, w_i/h_i, h_j/h_i, w_j/h_i, len_j/len_i).</summary>
    public const int RelationFeatures = 6;

    private readonly int _dim;
    private readonly int _learningDim;
    private readonly int _numLayers;
    private readonly double _eta;
    private readonly double _gamma;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _projection;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _learnW;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _alphaTransform;

    [SubLayerInput("_dim")]
    private readonly List<PickGcnLayer<T>> _gcn = new();

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>Creates the network. PICK: 512 wide, learningDim 128, 2 layers, eta 1, gamma 1.</summary>
    public PickGraphLayer([LayerState] int dim, [LayerState] int learningDim = 128, [LayerState] int numLayers = 2,
        [LayerState] double eta = 1.0, [LayerState] double gamma = 1.0)
        : base(new[] { -1, dim }, new[] { -1, dim })
    {
        if (dim <= 0) throw new ArgumentOutOfRangeException(nameof(dim));
        if (learningDim <= 0) throw new ArgumentOutOfRangeException(nameof(learningDim));
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        _dim = dim;
        _learningDim = learningDim;
        _numLayers = numLayers;
        _eta = eta;
        _gamma = gamma;
        var random = LayerInitializationSeedScope.NextRandom();
        // nn.Linear(dim, learningDim, bias=False): weight [learningDim, dim], bound 1/sqrt(dim); stored [dim, learningDim].
        _projection = Uniform(new[] { dim, learningDim }, 1.0 / Math.Sqrt(dim), random, centered: true);
        _learnW = Uniform(new[] { learningDim }, 1.0, random, centered: false); // U(0, 1)
        _alphaTransform = Uniform(new[] { RelationFeatures, dim }, 1.0 / Math.Sqrt(RelationFeatures), random, centered: true);
        RegisterTrainableParameter(_projection, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_learnW, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_alphaTransform, PersistentTensorRole.Weights);
        for (int i = 0; i < numLayers; i++)
        {
            var layer = new PickGcnLayer<T>(dim, dim);
            _gcn.Add(layer);
            RegisterSubLayer(layer);
        }
    }

    /// <inheritdoc/>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => inputRank == 2
        ? new[]
        {
            new OutputAxisContract(TensorAxis.Time, AxisRelation.Same(TensorAxis.Time)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_dim)),
        }
        : null;

    private Tensor<T> Uniform(int[] shape, double bound, Random random, bool centered)
    {
        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = NumOps.FromDouble(centered ? ((random.NextDouble() * 2) - 1) * bound : random.NextDouble() * bound);
        return tensor;
    }

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        int n = input.Shape[0];
        return Forward(input, new Tensor<T>(new[] { n, n, RelationFeatures })).Nodes;
    }

    /// <summary>
    /// Learns the soft adjacency and runs the convolutions: nodes <c>[N, D]</c> and relation features
    /// <c>[N, N, 6]</c> in; updated nodes, the soft adjacency and the graph-learning loss (a scalar tensor) out.
    /// </summary>
    internal (Tensor<T> Nodes, Tensor<T> Adjacency, Tensor<T> GraphLearningLoss) Forward(Tensor<T> nodes, Tensor<T> relations)
    {
        if (nodes.Rank != 2 || nodes.Shape[1] != _dim)
            throw new ArgumentException($"PickGraphLayer expects rank-2 nodes of shape [N, {_dim}]; got shape [{string.Join(", ", nodes.Shape.ToArray())}] (feature dimension mismatch).", nameof(nodes));
        int n = nodes.Shape[0];
        if (relations.Rank != 3 || relations.Shape[0] != n || relations.Shape[1] != n || relations.Shape[2] != RelationFeatures)
            throw new ArgumentException($"PickGraphLayer expects relation features [{n}, {n}, {RelationFeatures}].", nameof(relations));

        // Graph learning.
        var xHat = Engine.TensorMatMul(nodes, _projection);                                            // [N, L]
        var xi = Engine.TensorBroadcastTo(Engine.Reshape(xHat, new[] { n, 1, _learningDim }), new[] { n, n, _learningDim });
        var xj = Engine.TensorBroadcastTo(Engine.Reshape(xHat, new[] { 1, n, _learningDim }), new[] { n, n, _learningDim });
        var difference = Engine.TensorSubtract(xi, xj);
        var scores = Engine.Reshape(
            Engine.TensorMatMul(Engine.Reshape(Engine.TensorAbs(difference), new[] { n * n, _learningDim }), Engine.Reshape(_learnW, new[] { _learningDim, 1 })),
            new[] { n, n });
        var adjacency = Engine.TensorAddScalar(Engine.Softmax(Engine.LeakyReLU(scores, NumOps.FromDouble(0.01)), 1), NumOps.FromDouble(1e-10));

        // Graph-learning loss.
        var identity = new Tensor<T>(new[] { n, n });
        for (int i = 0; i < n; i++) identity[i, i] = NumOps.One;
        var squared = Engine.ReduceSum(Engine.TensorMultiply(difference, difference), new[] { 2 }, keepDims: false);  // [N, N]
        var distance = Engine.TensorSubtract(Engine.TensorSqrt(Engine.TensorAdd(squared, identity)), identity);
        var distanceTerm = Engine.TensorMultiplyScalar(
            Engine.ReduceSum(Engine.TensorExp(Engine.TensorAdd(adjacency, Engine.TensorMultiplyScalar(distance, NumOps.FromDouble(_eta)))), null),
            NumOps.FromDouble(1.0 / ((double)n * n)));
        var frobenius = Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorMultiply(adjacency, adjacency), null));
        var loss = Engine.TensorAdd(distanceTerm, Engine.TensorMultiplyScalar(frobenius, NumOps.FromDouble(_gamma)));

        // Convolutions, seeded by the relation embedding (Linear(6 -> D), no bias).
        var alpha = Engine.Reshape(
            Engine.TensorMatMul(Engine.Reshape(relations, new[] { n * n, RelationFeatures }), _alphaTransform), new[] { n, n, _dim });
        var x = nodes;
        foreach (var layer in _gcn) (x, alpha) = layer.Forward(x, alpha, adjacency);
        return (x, adjacency, loss);
    }

    /// <inheritdoc/>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["Dim"] = _dim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["LearningDim"] = _learningDim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["NumLayers"] = _numLayers.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["Eta"] = _eta.ToString("R", System.Globalization.CultureInfo.InvariantCulture);
        metadata["Gamma"] = _gamma.ToString("R", System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        foreach (var layer in _gcn) layer.ResetState();
    }
}

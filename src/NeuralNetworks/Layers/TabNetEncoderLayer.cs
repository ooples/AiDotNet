using System;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// TabNet encoder (Arik &amp; Pfister 2019): a sequential-attention block that produces an
/// aggregated decision representation through several decision steps, each of which selects a
/// sparse subset of features via a learnable mask.
/// </summary>
/// <remarks>
/// <para>
/// TabNet is not a feed-forward stack, so the decision-step loop is encapsulated in this single
/// composite layer (held in a model's <c>Layers</c> list like any other layer). Each decision step:
/// </para>
/// <list type="number">
///   <item>An <see cref="AttentiveTransformerLayer{T}"/> turns the previous step's feature
///   representation into a sparsemax mask over the input features, scaled by prior usage.</item>
///   <item>The mask is applied to the (batch-normalized) input features.</item>
///   <item>A <see cref="FeatureTransformerLayer{T}"/> processes the masked features; its output is
///   split into a decision part (ReLU'd and accumulated into the output) and an attention part
///   (fed to the next step's attentive transformer).</item>
///   <item>The prior scales are relaxed by <c>prior * (gamma - mask)</c> so already-used features
///   are discouraged in later steps.</item>
/// </list>
/// <para>
/// The feature count is resolved lazily from the first forward input (like <see cref="DenseLayer{T}"/>),
/// so the layer adapts to the actual fed input width. All sub-layers are registered via
/// <see cref="LayerBase{T}.RegisterSubLayer"/> so the trainable-parameter walk reaches them and the
/// optimizer updates them; the forward pass is built entirely from tape-recorded Engine ops and
/// sub-layer Forward calls, so gradients flow end-to-end. The decision/attention split uses a
/// constant selection-matrix matmul (tape-safe) rather than a tensor slice. Output: <c>[batch, decisionDim]</c>.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TabNetEncoderLayer<T> : LayerBase<T>
{
    private readonly int _decisionDim;
    private readonly int _attentionDim;
    private readonly int _numSteps;
    private readonly int _numSharedLayers;
    private readonly int _numStepSpecificLayers;
    private readonly double _relaxationFactor;
    private readonly int _virtualBatchSize;
    private readonly double _momentum;
    private readonly double _epsilon;

    private int _numFeatures = -1;
    private bool _built;

    private BatchNormalizationLayer<T>? _inputBatchNorm;
    private FeatureTransformerLayer<T>? _initialFeatureTransformer;
    private AttentiveTransformerLayer<T>[]? _attentiveTransformers;
    private FeatureTransformerLayer<T>[]? _stepFeatureTransformers;
    private Tensor<T>? _decisionSelector;
    private Tensor<T>? _attentionSelector;

    /// <summary>
    /// Initializes a new <see cref="TabNetEncoderLayer{T}"/>. The input feature count is resolved
    /// lazily from the first forward pass.
    /// </summary>
    /// <param name="decisionDim">Decision (output) dimension n_d.</param>
    /// <param name="attentionDim">Attention dimension n_a fed to the attentive transformer.</param>
    /// <param name="numSteps">Number of sequential decision steps.</param>
    /// <param name="numSharedLayers">Shared feature-transformer layers (within each transformer).</param>
    /// <param name="numStepSpecificLayers">Step-specific feature-transformer layers.</param>
    /// <param name="relaxationFactor">Gamma — prior-scale relaxation across steps.</param>
    /// <param name="virtualBatchSize">Ghost batch-norm virtual batch size.</param>
    /// <param name="momentum">Batch-norm momentum.</param>
    /// <param name="epsilon">Batch-norm epsilon.</param>
    public TabNetEncoderLayer(
        int decisionDim,
        int attentionDim,
        int numSteps,
        int numSharedLayers = 2,
        int numStepSpecificLayers = 2,
        double relaxationFactor = 1.5,
        int virtualBatchSize = 128,
        double momentum = 0.02,
        double epsilon = 1e-5)
        : base(new[] { -1 }, new[] { decisionDim })
    {
        if (decisionDim <= 0) throw new ArgumentOutOfRangeException(nameof(decisionDim));
        if (attentionDim <= 0) throw new ArgumentOutOfRangeException(nameof(attentionDim));
        if (numSteps <= 0) throw new ArgumentOutOfRangeException(nameof(numSteps));

        _decisionDim = decisionDim;
        _attentionDim = attentionDim;
        _numSteps = numSteps;
        _numSharedLayers = numSharedLayers;
        _numStepSpecificLayers = numStepSpecificLayers;
        _relaxationFactor = relaxationFactor;
        _virtualBatchSize = virtualBatchSize;
        _momentum = momentum;
        _epsilon = epsilon;
    }

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    private void BuildComponents(int numFeatures)
    {
        _numFeatures = numFeatures;
        int ftOut = _decisionDim + _attentionDim;

        _inputBatchNorm = new BatchNormalizationLayer<T>();
        _initialFeatureTransformer = new FeatureTransformerLayer<T>(
            numFeatures, ftOut, null, null, _numSharedLayers, _numStepSpecificLayers,
            _virtualBatchSize, _momentum, _epsilon);

        _attentiveTransformers = new AttentiveTransformerLayer<T>[_numSteps];
        _stepFeatureTransformers = new FeatureTransformerLayer<T>[_numSteps];
        for (int i = 0; i < _numSteps; i++)
        {
            _attentiveTransformers[i] = new AttentiveTransformerLayer<T>(
                _attentionDim, numFeatures, _relaxationFactor, _virtualBatchSize, _momentum, _epsilon);
            _stepFeatureTransformers[i] = new FeatureTransformerLayer<T>(
                numFeatures, ftOut, null, null, _numSharedLayers, _numStepSpecificLayers,
                _virtualBatchSize, _momentum, _epsilon);
        }

        RegisterSubLayer(_inputBatchNorm);
        RegisterSubLayer(_initialFeatureTransformer);
        for (int i = 0; i < _numSteps; i++)
        {
            RegisterSubLayer(_attentiveTransformers[i]);
            RegisterSubLayer(_stepFeatureTransformers[i]);
        }

        // Constant column-selection matrices: decision = ft @ S_d ([ftOut, n_d]),
        // attention = ft @ S_a ([ftOut, n_a]). A matmul against a fixed 0/1 selector
        // is a tape-safe way to take the [:, :n_d] and [:, n_d:] slices of the
        // feature-transformer output without a (tape-fragile) tensor slice op.
        _decisionSelector = new Tensor<T>(new[] { ftOut, _decisionDim });
        for (int i = 0; i < _decisionDim; i++) _decisionSelector[i, i] = NumOps.One;
        _attentionSelector = new Tensor<T>(new[] { ftOut, _attentionDim });
        for (int j = 0; j < _attentionDim; j++) _attentionSelector[_decisionDim + j, j] = NumOps.One;

        ResolveShapes(new[] { numFeatures }, new[] { _decisionDim });
        _built = true;
    }

    /// <inheritdoc/>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        int features = input.Shape[input.Rank - 1];
        if (!_built || _numFeatures != features)
        {
            BuildComponents(features);
        }

        int batch = input.Rank == 1 ? 1 : input.Shape[0];
        var x = Engine.Reshape(input, new[] { batch, _numFeatures });

        // Initial batch norm + initial feature transformer to seed the attention input.
        var xbn = _inputBatchNorm!.Forward(x);
        var ft0 = _initialFeatureTransformer!.Forward(xbn);
        var att = Engine.TensorMatMul(ft0, _attentionSelector!); // [batch, n_a]

        var prior = Tensor<T>.CreateDefault(new[] { batch, _numFeatures }, NumOps.One);
        Tensor<T>? aggregated = null;

        for (int step = 0; step < _numSteps; step++)
        {
            var mask = _attentiveTransformers![step].Forward(att, prior);      // [batch, D]
            prior = _attentiveTransformers[step].UpdatePriorScales(prior, mask);
            var masked = Engine.TensorMultiply(mask, xbn);                     // [batch, D]
            var ft = _stepFeatureTransformers![step].Forward(masked);          // [batch, ftOut]
            var decision = Engine.ReLU(Engine.TensorMatMul(ft, _decisionSelector!)); // [batch, n_d]
            aggregated = aggregated is null ? decision : Engine.TensorAdd(aggregated, decision);
            att = Engine.TensorMatMul(ft, _attentionSelector!);                // [batch, n_a]
        }

        return aggregated ?? Engine.TensorMatMul(ft0, _decisionSelector!);
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var all = new System.Collections.Generic.List<T>();
        foreach (var sub in GetSubLayers())
        {
            var p = sub.GetParameters();
            for (int i = 0; i < p.Length; i++) all.Add(p[i]);
        }
        var result = new Vector<T>(all.Count);
        for (int i = 0; i < all.Count; i++) result[i] = all[i];
        return result;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        int offset = 0;
        foreach (var sub in GetSubLayers())
        {
            int count = checked((int)sub.ParameterCount);
            if (count == 0) continue;
            var p = new Vector<T>(count);
            for (int i = 0; i < count; i++) p[i] = parameters[offset + i];
            sub.SetParameters(p);
            offset += count;
        }
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        foreach (var sub in GetSubLayers()) sub.UpdateParameters(learningRate);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        foreach (var sub in GetSubLayers()) sub.ResetState();
    }
}

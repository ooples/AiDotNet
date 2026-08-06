using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// A ConvNeXt v2 block: depth-wise convolution, layer norm, point-wise expansion, GELU,
/// Global Response Normalization (GRN), point-wise projection, and a residual connection.
/// </summary>
/// <remarks>
/// <para>ConvNeXt v2 (Woo et al., 2023) adds GRN to the v1 block. GRN normalizes each channel by
/// its global response relative to the mean response across channels, which increases contrast
/// and selectivity between channels — it is the distinguishing element of v2, so a v1 block with
/// GRN omitted is a different architecture.</para>
/// <para><b>GRN:</b> for channel <c>c</c> with per-channel L2 response <c>G_c</c> over the
/// sequence axis,</para>
/// <code>
///   N_c = G_c / mean_c(G)
///   out = gamma_c * (X_c * N_c) + beta_c + X_c
/// </code>
/// <para><b>Shapes:</b> operates on <c>[B, S, C]</c>, i.e. channels-last, so the point-wise
/// convolutions are ordinary dense projections over the channel axis.</para>
/// <para><b>Gradient tracking:</b> every operation goes through <c>IEngine</c>, so the tape
/// records the block — including GRN — without any manual backward.</para>
/// </remarks>
/// <typeparam name="T">Numeric type (float / double).</typeparam>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = false, ExpectedInputRank = 3, Cost = ComputeCost.Medium, TestInputShape = "1, 8, 16", TestConstructorArgs = "16, 48, 7")]
public partial class ConvNeXtV2Block<T> : LayerBase<T>
{
    private readonly int _channels;
    private readonly int _intermediateChannels;
    private readonly int _kernelSize;

    private readonly DepthwiseConv1DLayer<T> _depthwise;
    private readonly LayerNormalizationLayer<T> _norm;
    private readonly DenseLayer<T> _pointwiseExpand;
    private readonly DenseLayer<T> _pointwiseProject;

    /// <summary>GRN scale, one per intermediate channel.</summary>
    private Tensor<T> _grnGamma;

    /// <summary>GRN offset, one per intermediate channel.</summary>
    private Tensor<T> _grnBeta;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    public override long ParameterCount =>
        _depthwise.ParameterCount + _norm.ParameterCount +
        _pointwiseExpand.ParameterCount + _pointwiseProject.ParameterCount +
        _grnGamma.Length + _grnBeta.Length;

    /// <summary>
    /// Initializes a new ConvNeXt v2 block.
    /// </summary>
    /// <param name="channels">Block width <c>C</c>, unchanged from input to output.</param>
    /// <param name="intermediateChannels">Width of the point-wise expansion.</param>
    /// <param name="kernelSize">Depth-wise convolution kernel size.</param>
    public ConvNeXtV2Block(
        [LayerState] int channels,
        [LayerState] int intermediateChannels,
        [LayerState] int kernelSize = 7)
        : base(new[] { -1, -1, channels }, new[] { -1, -1, channels })
    {
        if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels));
        if (intermediateChannels <= 0) throw new ArgumentOutOfRangeException(nameof(intermediateChannels));
        if (kernelSize <= 0) throw new ArgumentOutOfRangeException(nameof(kernelSize));

        _channels = channels;
        _intermediateChannels = intermediateChannels;
        _kernelSize = kernelSize;

        _depthwise = new DepthwiseConv1DLayer<T>(channels, kernelSize);
        _norm = new LayerNormalizationLayer<T>();
        _pointwiseExpand = new DenseLayer<T>(intermediateChannels, (IActivationFunction<T>)new GELUActivation<T>());
        _pointwiseProject = new DenseLayer<T>(channels, (IActivationFunction<T>)new IdentityActivation<T>());

        RegisterSubLayer(_depthwise);
        RegisterSubLayer(_norm);
        RegisterSubLayer(_pointwiseExpand);
        RegisterSubLayer(_pointwiseProject);

        // GRN starts as an identity transform (gamma = 0, beta = 0), per the paper's
        // initialization, so an untrained block reduces to the residual path.
        _grnGamma = new Tensor<T>(new[] { intermediateChannels });
        _grnBeta = new Tensor<T>(new[] { intermediateChannels });

        // Register both so the engine tracks them as persistent trainable tensors. Without
        // this the GRN pair is reported by ParameterCount and serialized by GetParameters()
        // but accumulates no gradient, so gamma and beta stay pinned at their zero
        // initialization forever — and since gamma = 0 makes GRN an exact identity, the
        // block silently degrades to plain ConvNeXt v1 and the global response
        // normalization the v2 paper (Woo et al., 2023 §3.2) introduces never engages.
        RegisterTrainableParameter(_grnGamma, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_grnBeta, PersistentTensorRole.Biases);
    }

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Shape.Length != 3)
            throw new ArgumentException($"ConvNeXtV2Block expects rank-3 [B, S, C], got rank {input.Shape.Length}.", nameof(input));

        int B = input.Shape[0];
        int S = input.Shape[1];
        int C = input.Shape[2];

        if (C != _channels)
            throw new ArgumentException($"ConvNeXtV2Block was configured for channels={_channels} but got C={C}.", nameof(input));

        // Depth-wise convolution expects channels-first [B, C, S]; the rest of the block is
        // channels-last, so permute in and back out.
        var chFirst = Engine.TensorPermute(input, new[] { 0, 2, 1 });
        var conv = _depthwise.Forward(chFirst);
        var x = Engine.TensorPermute(conv, new[] { 0, 2, 1 });          // [B, S, C]

        x = _norm.Forward(x);
        x = _pointwiseExpand.Forward(x);                                 // [B, S, I]
        x = ApplyGlobalResponseNormalization(x, B, S, _intermediateChannels);
        x = _pointwiseProject.Forward(x);                                // [B, S, C]

        // Residual connection.
        return Engine.TensorAdd(x, input);
    }

    /// <summary>
    /// Global Response Normalization (ConvNeXt v2). Aggregates each channel's response over the
    /// sequence axis, divides by the mean channel response, then applies a learned scale and
    /// offset around a residual.
    /// </summary>
    /// <summary>
    /// Materializes the lazily-allocated sub-layers from this block's known geometry, without
    /// executing them.
    /// </summary>
    /// <remarks>
    /// The depth-wise convolution runs channels-first, so it resolves against <c>[1, C, 1]</c>;
    /// everything after it is channels-last, and the projection consumes the EXPANDED width
    /// rather than the block width. Each call is guarded by <c>IsShapeResolved</c>, so this is a
    /// no-op once the block has run.
    /// </remarks>
    private void ResolveChildShapes()
    {
        if (!_depthwise.IsShapeResolved) _depthwise.ResolveFromShape(new[] { 1, _channels, 1 });
        if (!_norm.IsShapeResolved) _norm.ResolveFromShape(new[] { 1, 1, _channels });
        if (!_pointwiseExpand.IsShapeResolved) _pointwiseExpand.ResolveFromShape(new[] { 1, 1, _channels });
        if (!_pointwiseProject.IsShapeResolved) _pointwiseProject.ResolveFromShape(new[] { 1, 1, _intermediateChannels });
    }

    private Tensor<T> ApplyGlobalResponseNormalization(Tensor<T> x, int B, int S, int I)
    {
        // Per-channel L2 over the sequence axis: sqrt(sum_s x^2) -> [B, 1, I].
        var squared = Engine.TensorMultiply(x, x);

        var onesRow = new Tensor<T>(new[] { B, 1, S });
        for (int i = 0; i < onesRow.Length; i++) onesRow[i] = NumOps.One;

        var sumSq = Engine.TensorBatchMatMul<T>(onesRow, squared);       // [B, 1, I]

        // Offset before the square root, not after. d/du sqrt(u) = 1/(2*sqrt(u)) is INFINITE at
        // u = 0, and a channel whose activations are all zero across the sequence gives exactly
        // u = 0 — so the backward pass produces a non-finite gradient that poisons every
        // downstream parameter on the first step. The existing epsilon below guards the DIVISION
        // by the mean response, which is a different degeneracy and does not help here.
        //
        // This only became reachable once the GRN gain and bias were registered as trainable:
        // before that the branch carried no gradient at all, so the infinite derivative was
        // never requested. APNet2 trains an 8-block ConvNeXt v2 stack and went NaN on its first
        // optimizer step.
        var sqrtEpsilon = new Tensor<T>(sumSq.Shape.ToArray());
        for (int i = 0; i < sqrtEpsilon.Length; i++) sqrtEpsilon[i] = NumOps.FromDouble(1e-12);
        var g = Engine.TensorSqrt(Engine.TensorAdd(sumSq, sqrtEpsilon));  // [B, 1, I]

        // Mean response across channels: [B, 1, 1], broadcast back over I.
        var onesChannels = new Tensor<T>(new[] { B, I, 1 });
        for (int i = 0; i < onesChannels.Length; i++) onesChannels[i] = NumOps.One;

        var gSum = Engine.TensorBatchMatMul<T>(g, onesChannels);         // [B, 1, 1]
        var gMean = Engine.TensorDivideScalar(gSum, NumOps.FromDouble(I));

        // Guard against a zero mean response (an all-zero activation map) before dividing.
        var epsilon = new Tensor<T>(new[] { B, 1, 1 });
        for (int i = 0; i < epsilon.Length; i++) epsilon[i] = NumOps.FromDouble(1e-6);
        gMean = Engine.TensorAdd(gMean, epsilon);

        var gMeanBroadcast = Engine.TensorBroadcastTo(gMean, [B, 1, I]);
        var n = Engine.TensorDivide(g, gMeanBroadcast);                  // [B, 1, I]

        var nBroadcast = Engine.TensorBroadcastTo(n, [B, S, I]);
        var scaled = Engine.TensorMultiply(x, nBroadcast);

        var gamma = Engine.TensorBroadcastTo(Engine.Reshape(_grnGamma, [1, 1, I]), [B, S, I]);
        var beta = Engine.TensorBroadcastTo(Engine.Reshape(_grnBeta, [1, 1, I]), [B, S, I]);

        // out = gamma * (x * N) + beta + x
        return Engine.TensorAdd(
            Engine.TensorAdd(Engine.TensorMultiply(gamma, scaled), beta),
            x);
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var parts = new[]
        {
            _depthwise.GetParameters(), _norm.GetParameters(),
            _pointwiseExpand.GetParameters(), _pointwiseProject.GetParameters()
        };

        int total = _grnGamma.Length + _grnBeta.Length;
        foreach (var p in parts) total += p.Length;

        var flat = new Vector<T>(total);
        int k = 0;
        foreach (var p in parts)
            for (int i = 0; i < p.Length; i++) flat[k++] = p[i];
        for (int i = 0; i < _grnGamma.Length; i++) flat[k++] = _grnGamma[i];
        for (int i = 0; i < _grnBeta.Length; i++) flat[k++] = _grnBeta[i];

        return flat;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        var sizes = new[]
        {
            _depthwise.GetParameters().Length, _norm.GetParameters().Length,
            _pointwiseExpand.GetParameters().Length, _pointwiseProject.GetParameters().Length
        };

        int expected = sizes.Sum() + _grnGamma.Length + _grnBeta.Length;

        // The sub-layers allocate their weights lazily on first Forward, so before the block has
        // ever run `expected` counts only the children that happen to be materialized already —
        // restoring a trained block into a fresh instance failed with "Expected 224 parameters,
        // got 1856".
        //
        // Resolve the children's shapes directly. This is the framework's own mechanism for
        // materializing a lazy layer without executing it: it allocates parameters from a known
        // input shape and nothing else. Running a synthetic probe Forward instead would evaluate
        // GRN at a degenerate sequence length of 1 and leave the layer in a state the caller
        // never asked for.
        //
        // Call it unconditionally. Guarding on `sizes.Sum() == 0` never fired: _depthwise is
        // sized eagerly in the constructor, so the sum is already non-zero on a fresh block and
        // the three genuinely-lazy children (_norm, _pointwiseExpand, _pointwiseProject) were
        // left unresolved. ResolveChildShapes is itself per-child IsShapeResolved-guarded, so
        // this is a no-op once the block has run.
        ResolveChildShapes();

        sizes = new[]
        {
            _depthwise.GetParameters().Length, _norm.GetParameters().Length,
            _pointwiseExpand.GetParameters().Length, _pointwiseProject.GetParameters().Length
        };
        expected = sizes.Sum() + _grnGamma.Length + _grnBeta.Length;

        if (parameters.Length != expected)
            throw new ArgumentException($"Expected {expected} parameters, got {parameters.Length}.", nameof(parameters));

        int k = 0;
        var targets = new LayerBase<T>[] { _depthwise, _norm, _pointwiseExpand, _pointwiseProject };
        for (int t = 0; t < targets.Length; t++)
        {
            var slice = new Vector<T>(sizes[t]);
            for (int i = 0; i < sizes[t]; i++) slice[i] = parameters[k++];
            targets[t].SetParameters(slice);
        }

        for (int i = 0; i < _grnGamma.Length; i++) _grnGamma[i] = parameters[k++];
        for (int i = 0; i < _grnBeta.Length; i++) _grnBeta[i] = parameters[k++];
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Includes the sub-layers' tensors explicitly: <c>LayerBase</c> does not recurse into
    /// registered children, and a composite that omits them reports an empty trainable set while
    /// still advertising a parameter count — which desynchronizes the ParameterBuffer and
    /// corrupts training rather than failing loudly.
    /// </remarks>
    public override IReadOnlyList<Tensor<T>> GetTrainableParameters()
    {
        var result = new List<Tensor<T>>();
        result.AddRange(_depthwise.GetTrainableParameters());
        result.AddRange(_norm.GetTrainableParameters());
        result.AddRange(_pointwiseExpand.GetTrainableParameters());
        result.AddRange(_pointwiseProject.GetTrainableParameters());
        result.Add(_grnGamma);
        result.Add(_grnBeta);
        return result;
    }

    /// <inheritdoc/>
    public override void SetTrainableParameters(IReadOnlyList<Tensor<T>> parameters)
    {
        var counts = new[]
        {
            _depthwise.GetTrainableParameters().Count, _norm.GetTrainableParameters().Count,
            _pointwiseExpand.GetTrainableParameters().Count, _pointwiseProject.GetTrainableParameters().Count
        };

        int expected = counts.Sum() + 2;
        if (parameters.Count != expected)
            throw new ArgumentException($"Expected {expected} trainable tensors, got {parameters.Count}.", nameof(parameters));

        int at = 0;
        _depthwise.SetTrainableParameters(parameters.Skip(at).Take(counts[0]).ToList()); at += counts[0];
        _norm.SetTrainableParameters(parameters.Skip(at).Take(counts[1]).ToList()); at += counts[1];
        _pointwiseExpand.SetTrainableParameters(parameters.Skip(at).Take(counts[2]).ToList()); at += counts[2];
        _pointwiseProject.SetTrainableParameters(parameters.Skip(at).Take(counts[3]).ToList()); at += counts[3];

        _grnGamma = parameters[at];
        _grnBeta = parameters[at + 1];
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Publishes the block geometry. The expansion ratio is NOT always 3x — APNet2's bounded CI
    /// fixtures use other ratios — so deserialization must read the real value rather than infer
    /// it, or the rebuilt block has the wrong parameter count.
    /// </remarks>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["Channels"] = _channels.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["IntermediateChannels"] = _intermediateChannels.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["KernelSize"] = _kernelSize.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _depthwise.ResetState();
        _norm.ResetState();
        _pointwiseExpand.ResetState();
        _pointwiseProject.ResetState();
    }
}

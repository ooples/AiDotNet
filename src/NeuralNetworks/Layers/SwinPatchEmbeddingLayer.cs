using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Patch embedding layer for Swin Transformer that converts images to patch sequences.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This layer divides an input image into non-overlapping patches and projects each patch
/// to an embedding vector. This is the first step in processing images with Swin Transformer.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of this layer as cutting an image into small squares (patches)
/// and converting each square into a list of numbers (embedding) that describes its content.
/// This allows the transformer to process images as sequences, similar to how it processes text.
/// </para>
/// <para>
/// Reference: Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", ICCV 2021
/// </para>
/// </remarks>
[LayerCategory(LayerCategory.Embedding)]
[LayerCategory(LayerCategory.Transformer)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerTask(LayerTask.SpatialProcessing)]
// ExpectedInputRank corrected from 3 to 4: OnFirstForward accepts s.Length == 4 and throws for
// anything else ("requires rank-4 [B,C,H,W] input; got rank {n}. Add a batch dimension"), and
// TestInputShape "1, 3, 8, 8" is itself rank 4. The 3 was copied from PatchEmbeddingLayer, which
// genuinely does take [C,H,W]; this layer does not, so declaring a rank-3 layout to match it would
// have been a false claim (and ADNSHAPE005 would have flagged the disagreement either way).
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 4, TestInputShape = "1, 3, 8, 8", TestConstructorArgs = "4, 16")]
// Rank 4 ONLY, and batch is NOT optional - see the guard above.
//
// The output is a token sequence: ForwardTraced permutes NCHW -> NHWC and reshapes to
// [batch, numPatches, _embedDim], so the flattened patch grid is Time and the embedding is Features -
// the same reading SwinPatchMergingLayer, the layer that consumes this one, declares for its input.
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class SwinPatchEmbeddingLayer<T> : LayerBase<T>, IShapeContract
{
    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// The patch grid comes from the inner projection, constructed as
    /// <c>new ConvolutionalLayer&lt;T&gt;(embedDim, kernelSize: patchSize, stride: patchSize,
    /// padding: 0)</c>, and <c>ForwardTraced</c> reads its result back as
    /// <c>patchH = projected.Shape[2]</c>, <c>patchW = projected.Shape[3]</c>,
    /// <c>numPatches = patchH * patchW</c>. So each spatial axis is that convolution's window and the
    /// token axis is their product - which is exactly <see cref="AxisRelation.ProductOf"/> over two
    /// <see cref="AxisRelation.Window"/>s. <see cref="AxisRelation.Product"/> alone multiplies RAW
    /// input axes and cannot express the per-side division first.
    /// </para>
    /// <para>
    /// NOT <c>Fixed(NumPatches)</c>. That property is <c>(_inputHeight / _patchSize) *
    /// (_inputWidth / _patchSize)</c> over fields resolved from the FIRST input seen, so freezing it
    /// would state one resolution as if it were configuration. The window form derives the same
    /// number from whatever input actually arrives.
    /// </para>
    /// <para>
    /// <c>OnFirstForward</c> additionally requires <c>inH % _patchSize == 0</c> and
    /// <c>inW % _patchSize == 0</c>, so the window's floor is exact here rather than a crop - but the
    /// window is still the right form, because it is what the convolution computes.
    /// </para>
    /// <para>
    /// The trailing <c>_norm</c> is a <see cref="LayerNormalizationLayer{T}"/> resolved at
    /// <c>[_embedDim]</c> and preserves shape, so it does not enter the contract.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank != 4 || _patchSize <= 0 || _embedDim <= 0) return null;

        return new[]
        {
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            new OutputAxisContract(
                TensorAxis.Time,
                AxisRelation.ProductOf(
                    AxisRelation.Window(TensorAxis.Height, kernel: _patchSize, stride: _patchSize, padding: 0),
                    AxisRelation.Window(TensorAxis.Width, kernel: _patchSize, stride: _patchSize, padding: 0))),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_embedDim)),
        };
    }

    private readonly int _patchSize;
    private readonly int _embedDim;
    // Non-readonly: lazy ctor leaves these = -1 until OnFirstForward.
    private int _inputChannels;
    private int _inputHeight;
    private int _inputWidth;

    /// <summary>
    /// The convolutional layer used for patch projection.
    /// Uses kernel size = stride = patch size for non-overlapping patches.
    /// </summary>
    private readonly ConvolutionalLayer<T> _projection;

    /// <summary>
    /// Layer normalization applied after patch embedding.
    /// </summary>
    private readonly LayerNormalizationLayer<T> _norm;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the number of patches produced by this layer.
    /// </summary>
    public int NumPatches => (_inputHeight / _patchSize) * (_inputWidth / _patchSize);

    /// <summary>
    /// Gets the height of the patch grid.
    /// </summary>
    public int PatchGridHeight => _inputHeight / _patchSize;

    /// <summary>
    /// Gets the width of the patch grid.
    /// </summary>
    public int PatchGridWidth => _inputWidth / _patchSize;

    /// <summary>
    /// Creates a new Swin patch embedding layer.
    /// </summary>
    /// <param name="inputHeight">Height of input images.</param>
    /// <param name="inputWidth">Width of input images.</param>
    /// <param name="inputChannels">Number of input channels (default: 3 for RGB).</param>
    /// <param name="patchSize">Size of each patch (default: 4 from Swin paper).</param>
    /// <param name="embedDim">Dimension of patch embeddings (default: 96 for Swin-Tiny).</param>
    /// <exception cref="ArgumentException">Thrown if input dimensions are not divisible by patch size.</exception>
    public SwinPatchEmbeddingLayer(
        int patchSize = 4,
        int embedDim = 96)
        : base([-1, -1, -1], [embedDim])
    {
        if (patchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(patchSize), "Patch size must be positive.");
        if (embedDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(embedDim), "Embed dim must be positive.");

        _patchSize = patchSize;
        _embedDim = embedDim;
        _inputChannels = -1; // resolved in OnFirstForward
        _inputHeight = -1;   // resolved in OnFirstForward
        _inputWidth = -1;    // resolved in OnFirstForward

        // Projection: Conv with kernel=stride=patchSize creates non-overlapping patches.
        _projection = new ConvolutionalLayer<T>(
            embedDim,
            kernelSize: patchSize,
            stride: patchSize,
            padding: 0);

        // Layer normalization over embedding dimension
        _norm = new LayerNormalizationLayer<T>();

        RegisterSubLayer(_projection);
        RegisterSubLayer(_norm);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Resolves <c>_inputChannels</c> / <c>_inputHeight</c> / <c>_inputWidth</c>
    /// from <c>input.Shape</c> and propagates the resolved channel-shape
    /// to the inner projection conv via <see cref="LayerBase{T}.ResolveFromShape"/>.
    /// Validates the patch-size divisibility constraint here instead of
    /// at construction.
    /// </remarks>
    protected override void OnFirstForward(Tensor<T> input)
    {
        var s = input._shape;
        // The Forward path below indexes axis 0 as batch, axis 1 as channels —
        // so rank-3 [C,H,W] is NOT supported. Reject it explicitly here so
        // shape resolution and Forward agree on the input rank contract.
        int inChannels, inH, inW;
        if (s.Length == 4) { inChannels = s[1]; inH = s[2]; inW = s[3]; }
        else
            throw new ArgumentException(
                $"SwinPatchEmbeddingLayer requires rank-4 [B,C,H,W] input; got rank {s.Length}. " +
                "Add a batch dimension before calling Forward.",
                nameof(input));
        if (inH % _patchSize != 0)
            throw new ArgumentException($"Input height ({inH}) must be divisible by patch size ({_patchSize}).", nameof(input));
        if (inW % _patchSize != 0)
            throw new ArgumentException($"Input width ({inW}) must be divisible by patch size ({_patchSize}).", nameof(input));

        _inputChannels = inChannels;
        _inputHeight = inH;
        _inputWidth = inW;

        _projection.ResolveFromShape(new[] { inChannels, inH, inW });
        _projection.SetTrainingMode(IsTrainingMode);
        // Resolve the inner normalization before replaying a serialized flat
        // parameter vector. Otherwise its ParameterCount is still zero, so
        // SetParameters consumes only the projection weights and silently
        // drops the trained gamma/beta values; the first Forward then creates
        // fresh normalization parameters and a clone immediately diverges.
        _norm.ResolveFromShape(new[] { _embedDim });
        _norm.SetTrainingMode(IsTrainingMode);

        ResolveShapes(
            new[] { inChannels, inH, inW },
            new[] { _embedDim });

        // Replay any Deserialize-buffered parameters now that _projection is resolved.
        if (_pendingParameters is not null)
        {
            var pending = _pendingParameters;
            _pendingParameters = null;
            SetParameters(pending);
        }
    }

    /// <summary>
    /// Performs the forward pass, converting image to patch sequence.
    /// </summary>
    /// <param name="input">Input tensor of shape [batch, channels, height, width].</param>
    /// <returns>Output tensor of shape [batch, numPatches, embedDim].</returns>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (!IsShapeResolved) OnFirstForward(input);

        // Input: [batch, channels, height, width]
        int batch = input.Shape[0];

        // Apply convolution: [batch, embedDim, H/patchSize, W/patchSize]
        var projected = _projection.Forward(input);

        int patchH = projected.Shape[2];
        int patchW = projected.Shape[3];
        int numPatches = patchH * patchW;

        // NCHW -> NHWC -> [batch, numPatches, embedDim]. Keep this conversion
        // on Engine operations so the compiled training graph retains the edge
        // from the normalized sequence back to the convolutional projection.
        var channelsLast = Engine.TensorPermute(projected, [0, 2, 3, 1]);
        var sequence = Engine.Reshape(channelsLast, [batch, numPatches, _embedDim]);

        // Apply layer normalization
        var normalized = _norm.Forward(sequence);

        return normalized;
    }

    /// <summary>
    /// Emits the constructor settings (patch size + embedding dim) that cannot be
    /// inferred from shapes alone. Without these the reflection-driven deserialization
    /// fallback rebuilds the projection conv with a default embedding dim, producing a
    /// different ParameterCount than was serialized and throwing when the buffered
    /// parameters are replayed on the first forward.
    /// </summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["PatchSize"] = _patchSize.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["EmbedDim"] = _embedDim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    [Scratch]
    private Vector<T>? _pendingParameters;

    /// <inheritdoc/>
    public override Vector<T> GetParameterGradients()
    {
        var projGrads = _projection.GetParameterGradients();
        var normGrads = _norm.GetParameterGradients();

        var result = new T[projGrads.Length + normGrads.Length];
        projGrads.AsSpan().CopyTo(result.AsSpan(0, projGrads.Length));
        normGrads.AsSpan().CopyTo(result.AsSpan(projGrads.Length, normGrads.Length));

        return new Vector<T>(result);
    }

    /// <inheritdoc/>
    public override void ClearGradients()
    {
        base.ClearGradients();
        _projection.ClearGradients();
        _norm.ClearGradients();
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _projection.ResetState();
        _norm.ResetState();
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        _projection.UpdateParameters(learningRate);
        _norm.UpdateParameters(learningRate);
    }

    /// <inheritdoc />
}

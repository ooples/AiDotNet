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
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 3, TestInputShape = "1, 3, 8, 8", TestConstructorArgs = "4, 16")]
public class SwinPatchEmbeddingLayer<T> : LayerBase<T>
{
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

    /// <inheritdoc/>
    public override long ParameterCount => _projection.ParameterCount + _norm.ParameterCount;

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
        int inChannels, inH, inW;
        if (s.Length == 3) { inChannels = s[0]; inH = s[1]; inW = s[2]; }
        else if (s.Length == 4) { inChannels = s[1]; inH = s[2]; inW = s[3]; }
        else
            throw new ArgumentException(
                $"SwinPatchEmbeddingLayer requires rank-3 [C,H,W] or rank-4 [B,C,H,W] input; got rank {s.Length}.",
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
    public override Tensor<T> Forward(Tensor<T> input)
    {
        if (!IsShapeResolved) OnFirstForward(input);

        // Input: [batch, channels, height, width]
        int batch = input.Shape[0];

        // Apply convolution: [batch, embedDim, H/patchSize, W/patchSize]
        var projected = _projection.Forward(input);

        int patchH = projected.Shape[2];
        int patchW = projected.Shape[3];
        int numPatches = patchH * patchW;

        // Reshape to sequence: [batch, numPatches, embedDim]
        var sequence = new Tensor<T>([batch, numPatches, _embedDim]);

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < patchH; h++)
            {
                for (int w = 0; w < patchW; w++)
                {
                    int seqIdx = h * patchW + w;
                    for (int c = 0; c < _embedDim; c++)
                    {
                        sequence[b, seqIdx, c] = projected[b, c, h, w];
                    }
                }
            }
        }

        // Apply layer normalization
        var normalized = _norm.Forward(sequence);

        return normalized;
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var projParams = _projection.GetParameters();
        var normParams = _norm.GetParameters();

        var result = new T[projParams.Length + normParams.Length];
        projParams.AsSpan().CopyTo(result.AsSpan(0, projParams.Length));
        normParams.AsSpan().CopyTo(result.AsSpan(projParams.Length, normParams.Length));

        return new Vector<T>(result);
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        // Pre-Forward: _projection's input channel count is unresolved.
        // Buffer and replay from OnFirstForward.
        if (!IsShapeResolved)
        {
            _pendingParameters = parameters;
            return;
        }

        int projCount = checked((int)_projection.ParameterCount);
        int normCount = checked((int)_norm.ParameterCount);

        var projParams = new T[projCount];
        var normParams = new T[normCount];

        parameters.AsSpan().Slice(0, projCount).CopyTo(projParams);
        parameters.AsSpan().Slice(projCount, normCount).CopyTo(normParams);

        _projection.SetParameters(new Vector<T>(projParams));
        _norm.SetParameters(new Vector<T>(normParams));
    }

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

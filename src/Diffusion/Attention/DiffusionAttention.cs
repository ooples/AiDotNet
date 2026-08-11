using AiDotNet.ActivationFunctions;
// File-level, deliberately: two Tensors namespaces in the project's global usings also define a
// TensorLayout, so [TensorLayout(...)] only binds when this import shadows them from a nearer scope.
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Attention;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.Attention;

/// <summary>
/// Memory-efficient attention layer for diffusion models using Flash Attention.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This attention layer automatically uses Flash Attention when the sequence length
/// exceeds a threshold, providing significant memory and performance benefits for
/// high-resolution image generation.
/// </para>
/// <para>
/// <b>For Beginners:</b> Attention in diffusion models is computationally expensive.
///
/// For a 512x512 image at 8x downsampling:
/// - Sequence length = (512/8)^2 = 4096 tokens
/// - Standard attention: 4096 x 4096 = 16 million attention weights!
///
/// This class automatically uses Flash Attention for long sequences:
/// - Under 256 tokens: Standard attention (faster for short sequences)
/// - 256+ tokens: Flash Attention (memory-efficient, scales better)
///
/// Usage:
/// ```csharp
/// var attention = new DiffusionAttention&lt;float&gt;(
///     channels: 320,
///     numHeads: 8,
///     spatialSize: 64);
///
/// var output = attention.Forward(input);
/// ```
/// </para>
/// </remarks>
// Shape-preserving at rank 3. The forward branches on rank - `isImageFormat = Shape.Length == 4`
// selects [B,C,H,W], otherwise [B,S,D] - and ONLY the rank-3 sequence form was probed, so only that
// is declared. The rank-4 image form is deliberately left undeclared rather than assumed.
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class DiffusionAttention<T> : LayerBase<T>, IShapeContract
{
    /// <summary>
    /// Number of channels.
    /// </summary>
    private readonly int _channels;

    /// <summary>
    /// Number of attention heads.
    /// </summary>
    private readonly int _numHeads;

    /// <summary>
    /// Dimension per head.
    /// </summary>
    private readonly int _headDim;

    /// <summary>
    /// Spatial size (height/width).
    /// </summary>
    private readonly int _spatialSize;

    /// <summary>
    /// Sequence length threshold for switching to Flash Attention.
    /// </summary>
    private readonly int _flashAttentionThreshold;

    /// <summary>
    /// Flash Attention layer for long sequences.
    /// </summary>
    private readonly FlashAttentionLayer<T> _flashAttention;

    /// <summary>
    /// Standard attention layer for short sequences.
    /// </summary>
    private readonly MultiHeadAttentionLayer<T> _standardAttention;

    /// <summary>
    /// Flash Attention configuration.
    /// </summary>
    private readonly FlashAttentionConfig _flashConfig;

    /// <summary>
    /// Cached input for backward pass.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Whether Flash Attention was used in the last forward pass.
    /// </summary>
    private bool _usedFlashAttention;

    /// <summary>
    /// Whether this layer supports training.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the number of channels.
    /// </summary>
    public int Channels => _channels;

    /// <summary>
    /// Gets the number of attention heads.
    /// </summary>
    public int NumHeads => _numHeads;

    /// <summary>
    /// Gets whether Flash Attention is enabled.
    /// </summary>
    public bool FlashAttentionEnabled => _flashConfig != null;

    /// <summary>Construction state: the 'useCausalMask' the layer was built with.</summary>
    private readonly bool _useCausalMask;

    /// <summary>
    /// Initializes a new diffusion attention layer.
    /// </summary>
    /// <param name="channels">Number of input/output channels.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="spatialSize">Spatial size (height = width) of input feature maps.</param>
    /// <param name="flashAttentionThreshold">Sequence length threshold for using Flash Attention (default: 256).</param>
    /// <param name="useCausalMask">Whether to use causal masking (default: false for images).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Configuration tips:
    ///
    /// - channels: Should match your UNet block channels (e.g., 320, 640, 1280)
    /// - numHeads: 8 is typical; channels must be divisible by numHeads
    /// - spatialSize: 64 for 512px images at 8x downsampling
    /// - flashAttentionThreshold: Lower = more Flash Attention usage = less memory
    /// </para>
    /// </remarks>
    public DiffusionAttention(
        int channels,
        int numHeads = 8,
        int spatialSize = 64,
        int flashAttentionThreshold = 256,
        bool useCausalMask = false)
        : base(
            CalculateInputShape(channels, spatialSize),
            CalculateOutputShape(channels, spatialSize))
    {
        _useCausalMask = useCausalMask;
        if (channels <= 0)
            throw new ArgumentOutOfRangeException(nameof(channels), "Channels must be positive.");
        if (numHeads <= 0)
            throw new ArgumentOutOfRangeException(nameof(numHeads), "Number of heads must be positive.");
        if (channels % numHeads != 0)
            throw new ArgumentException($"Channels ({channels}) must be divisible by numHeads ({numHeads}).");

        _channels = channels;
        _numHeads = numHeads;
        _headDim = channels / numHeads;
        _spatialSize = spatialSize;
        _flashAttentionThreshold = flashAttentionThreshold;

        int sequenceLength = spatialSize * spatialSize;

        // Configure Flash Attention for memory efficiency
        _flashConfig = new FlashAttentionConfig
        {
            UseCausalMask = useCausalMask,
            BlockSizeQ = 64,  // Optimal for most GPUs
            BlockSizeKV = 64,
            RecomputeInBackward = true,  // Memory-efficient backward pass
            ReturnAttentionWeights = false,  // Don't materialize full attention matrix
            Precision = FlashAttentionPrecision.Float32
        };

        // Create Flash Attention layer
        _flashAttention = new FlashAttentionLayer<T>(
            sequenceLength: sequenceLength,
            embeddingDimension: channels,
            headCount: numHeads,
            config: _flashConfig);

        // Create standard attention as fallback for short sequences
        _standardAttention = new MultiHeadAttentionLayer<T>(numHeads, (channels) / (numHeads), 
            activationFunction: new IdentityActivation<T>());
    }

    private static int[] CalculateInputShape(int channels, int spatialSize)
    {
        return new[] { 1, spatialSize * spatialSize, channels };
    }

    private static int[] CalculateOutputShape(int channels, int spatialSize)
    {
        return new[] { 1, spatialSize * spatialSize, channels };
    }

    /// <summary>
    /// Performs the forward pass through the attention layer.
    /// </summary>
    /// <param name="input">Input tensor of shape [batch, channels, height, width].</param>
    /// <returns>Output tensor of the same shape.</returns>
    /// <remarks>
    /// <para>
    /// The input is reshaped from image format [B, C, H, W] to sequence format [B, H*W, C]
    /// for attention computation, then reshaped back to image format.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        // Retain the backward-activation cache only when an eager manual Backward
        // will read it; skip in inference/under tape so the denoise-loop arena can
        // recycle scratch without aliasing a stale reference (issue #1668).
        _lastInput = ShouldCacheForBackward ? input : null;

        // Determine if input is image format [B, C, H, W] or sequence format [B, S, D]
        bool isImageFormat = input.Shape.Length == 4;
        int batchSize, sequenceLength, channels;
        Tensor<T> x;

        if (isImageFormat)
        {
            batchSize = input.Shape[0];
            channels = input.Shape[1];
            int height = input.Shape[2];
            int width = input.Shape[3];
            sequenceLength = height * width;

            // Reshape [B, C, H, W] -> [B, H*W, C]. Every shape op must go
            // through Engine so the gradient tape records the transformation —
            // direct Tensor<T>.Transpose / .Reshape bypass the tape and break
            // gradient flow through the attention block.
            var permuted = Engine.TensorPermute(input, new[] { 0, 2, 3, 1 });
            x = Engine.Reshape(permuted, new[] { batchSize, sequenceLength, channels });
        }
        else
        {
            batchSize = input.Shape[0];
            sequenceLength = input.Shape[1];
            channels = input.Shape[2];
            x = input;
        }

        // Choose attention implementation based on sequence length
        Tensor<T> output;
        if (sequenceLength >= _flashAttentionThreshold)
        {
            // Use Flash Attention for long sequences
            output = _flashAttention.Forward(x);
            _usedFlashAttention = true;
        }
        else
        {
            // Use standard attention for short sequences
            output = _standardAttention.Forward(x);
            _usedFlashAttention = false;
        }

        // Reshape back to image format if needed — via Engine for tape recording.
        if (isImageFormat)
        {
            int height = input.Shape[2];
            int width = input.Shape[3];
            var reshaped = Engine.Reshape(output, new[] { batchSize, height, width, channels });
            output = Engine.TensorPermute(reshaped, new[] { 0, 3, 1, 2 });
        }

        return output;
    }

    /// <summary>
    /// Propagates eval/training mode to the nested Flash / standard attention sublayers.
    /// Without this override the private sublayers never see model.SetTrainingMode(false),
    /// keeping their internal Dense projections on the allocating tape/training Forward
    /// branch during inference instead of the GPU-resident inference fast path.
    /// </summary>
    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
        _flashAttention.SetTrainingMode(isTraining);
        _standardAttention.SetTrainingMode(isTraining);
    }

    /// <summary>
    /// Updates parameters using computed gradients.
    /// </summary>
    public override void UpdateParameters(T learningRate)
    {
        _flashAttention.UpdateParameters(learningRate);
        _standardAttention.UpdateParameters(learningRate);
    }

    /// <summary>
    /// Resets the layer's internal state.
    /// </summary>
    public override void ResetState()
    {
        _lastInput = null;
        _usedFlashAttention = false;
        _flashAttention.ResetState();
        _standardAttention.ResetState();
    }


    /// <summary>
    /// Gets diagnostic information about the layer.
    /// </summary>
    public override Dictionary<string, string> GetDiagnostics()
    {
        var diagnostics = base.GetDiagnostics();
        diagnostics["Channels"] = _channels.ToString();
        diagnostics["NumHeads"] = _numHeads.ToString();
        diagnostics["HeadDim"] = _headDim.ToString();
        diagnostics["SpatialSize"] = _spatialSize.ToString();
        diagnostics["FlashAttentionThreshold"] = _flashAttentionThreshold.ToString();
        diagnostics["LastUsedFlashAttention"] = _usedFlashAttention.ToString();
        return diagnostics;
    }
}

/// <summary>
/// Cross-attention layer for diffusion models with Flash Attention optimization.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Cross-attention allows the model to attend to conditioning information (like text embeddings)
/// when generating images. This is how text-to-image models like Stable Diffusion work.
/// </para>
/// <para>
/// <b>For Beginners:</b> Cross-attention is how the model "looks at" text while generating images.
///
/// - Query (Q): Comes from the image features
/// - Key (K) and Value (V): Come from text embeddings
/// - Output: Image features enriched with text information
///
/// This enables the model to generate images that match the text description.
/// </para>
/// </remarks>
// SHAPE-PRESERVING on BOTH forms its own forward branches on, read off ForwardWithContext:
//
//   rank 4 - `isImageFormat` branch: permute [B,C,H,W] -> [B,H,W,C], flatten to [B, H*W, C], attend,
//            then reshape back to [B,H,W,C] and permute to [B,C,H,W]. The trailing reshape only has
//            enough elements if attention kept the feature width, which it does.
//   rank 3 - the else branch passes the tensor to _crossAttention untouched.
//
// The identity on the feature axis is the nested CrossAttentionLayer's, not an assumption: it is
// declared `[-1, queryDim] -> [-1, queryDim]` with `[LayerProperty(ChangesShape = false)]`, so both
// the sequence length and the width survive. Axis roles are therefore identical on both sides at
// both ranks and OutputAxesFor is generated as Same throughout.
//
// The context tensor is deliberately absent from this contract. It is a second input, and the output
// borrows nothing from its shape - contextDim is consumed inside the attention's key/value
// projections and never reaches the result.
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input,
    Note = "Already-flattened token form: [batch, sequence, queryDim].")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input,
    Note = "Image form: flattened to H*W tokens internally and restored before returning.")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class DiffusionCrossAttention<T> : LayerBase<T>, IShapeContract
{
    /// <summary>
    /// Query dimension (spatial channels).
    /// </summary>
    private readonly int _queryDim;

    /// <summary>
    /// Context dimension (text embedding dimension).
    /// </summary>
    private readonly int _contextDim;

    /// <summary>
    /// Number of attention heads.
    /// </summary>
    private readonly int _numHeads;

    /// <summary>
    /// Spatial size (height/width).
    /// </summary>
    private readonly int _spatialSize;

    /// <summary>
    /// Cross-attention layer.
    /// </summary>
    private readonly CrossAttentionLayer<T> _crossAttention;

    /// <summary>
    /// Cached input for backward pass.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Cached context for backward pass.
    /// </summary>
    private Tensor<T>? _lastContext;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the query dimension.
    /// </summary>
    public int QueryDim => _queryDim;

    /// <summary>
    /// Gets the context dimension.
    /// </summary>
    public int ContextDim => _contextDim;

    /// <summary>
    /// Initializes a new diffusion cross-attention layer.
    /// </summary>
    /// <param name="queryDim">Dimension of query (spatial channels).</param>
    /// <param name="contextDim">Dimension of context (text embedding).</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="spatialSize">Spatial size of input feature maps.</param>
    public DiffusionCrossAttention(
        int queryDim,
        int contextDim,
        int numHeads = 8,
        int spatialSize = 64)
        : base(
            new[] { 1, spatialSize * spatialSize, queryDim },
            new[] { 1, spatialSize * spatialSize, queryDim })
    {
        if (queryDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(queryDim), "Query dimension must be positive.");
        if (contextDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(contextDim), "Context dimension must be positive.");

        _queryDim = queryDim;
        _contextDim = contextDim;
        _numHeads = numHeads;
        _spatialSize = spatialSize;

        int sequenceLength = spatialSize * spatialSize;

        _crossAttention = new CrossAttentionLayer<T>(
            queryDim: queryDim,
            contextDim: contextDim,
            headCount: numHeads,
            sequenceLength: sequenceLength);
    }

    /// <summary>
    /// Performs the forward pass through cross-attention.
    /// </summary>
    /// <param name="input">Input tensor (query) of shape [batch, channels, height, width].</param>
    /// <returns>Output tensor of the same shape.</returns>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        // This version uses null context - use ForwardWithContext for actual cross-attention
        return ForwardWithContext(input, null);
    }

    /// <summary>
    /// Performs the forward pass with context (conditioning).
    /// </summary>
    /// <param name="input">Input tensor (query) of shape [batch, channels, height, width].</param>
    /// <param name="context">Context tensor (key/value) of shape [batch, contextLen, contextDim].</param>
    /// <returns>Output tensor of the same shape as input.</returns>
    public Tensor<T> ForwardWithContext(Tensor<T> input, Tensor<T>? context)
    {
        // See Forward: keep backward-activation caches only when a manual Backward reads them.
        bool cacheBwd = ShouldCacheForBackward;
        _lastInput = cacheBwd ? input : null;
        _lastContext = cacheBwd ? context : null;

        bool isImageFormat = input.Shape.Length == 4;
        int batchSize, sequenceLength, channels;
        Tensor<T> x;

        if (isImageFormat)
        {
            batchSize = input.Shape[0];
            channels = input.Shape[1];
            int height = input.Shape[2];
            int width = input.Shape[3];
            sequenceLength = height * width;

            // Shape ops via Engine so the gradient tape records them.
            var permuted = Engine.TensorPermute(input, new[] { 0, 2, 3, 1 });
            x = Engine.Reshape(permuted, new[] { batchSize, sequenceLength, channels });
        }
        else
        {
            batchSize = input.Shape[0];
            sequenceLength = input.Shape[1];
            channels = input.Shape[2];
            x = input;
        }

        // Perform cross-attention
        Tensor<T> output;
        if (context != null)
        {
            output = _crossAttention.Forward(x, context);
        }
        else
        {
            // Self-attention fallback when no context
            output = _crossAttention.Forward(x, x);
        }

        if (isImageFormat)
        {
            int height = input.Shape[2];
            int width = input.Shape[3];
            var reshaped = Engine.Reshape(output, new[] { batchSize, height, width, channels });
            output = Engine.TensorPermute(reshaped, new[] { 0, 3, 1, 2 });
        }

        return output;
    }

    /// <summary>
    /// Propagates eval/training mode to the nested cross-attention sublayer so inference
    /// uses the GPU-resident fast path rather than the allocating tape/training branch.
    /// </summary>
    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
        _crossAttention.SetTrainingMode(isTraining);
    }

    /// <summary>
    /// Updates parameters.
    /// </summary>
    public override void UpdateParameters(T learningRate)
    {
        _crossAttention.UpdateParameters(learningRate);
    }

    /// <summary>
    /// Resets the layer's internal state.
    /// </summary>
    public override void ResetState()
    {
        _lastInput = null;
        _lastContext = null;
        _crossAttention.ResetState();
    }


}

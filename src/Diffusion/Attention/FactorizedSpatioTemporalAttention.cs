using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Attention;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.Attention;

/// <summary>
/// Factorized spatio-temporal attention that applies spatial and temporal attention separately.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "Scalable Diffusion Models with Transformers" (Peebles and Xie, 2023)</item>
/// <item>Paper: "Video Diffusion Models" (Ho et al., 2022)</item>
/// </list></para>
/// <para><b>For Beginners:</b> Factorized Spatio-Temporal Attention processes spatial (within-frame) and temporal (across-frame) relationships separately. This is much more efficient than joint attention while still capturing both spatial detail and temporal motion.</para>
/// <para>
/// Factorized spatio-temporal attention decomposes full 3D attention into separate spatial
/// and temporal components. This reduces computational complexity from O((T*H*W)^2) to
/// O(T*(H*W)^2 + H*W*T^2), making it feasible for high-resolution long videos.
/// </para>
/// <para>
/// Architecture:
/// - Spatial attention: self-attention within each frame (across H*W positions)
/// - Temporal attention: self-attention across frames (for each spatial position)
/// - LayerNorm + residual connections around each attention block
/// </para>
/// </remarks>
// Shape-preserving at rank 3 [Batch, Time, Features]; only that rank was probed, so only it is declared.
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class FactorizedSpatioTemporalAttention<T> : LayerBase<T>, IShapeContract
{
    private readonly int _channels;
    private readonly int _numHeads;
    private readonly int _numFrames;
    private readonly int _spatialSize;
    private readonly DiffusionAttention<T> _spatialAttention;
    private readonly TemporalSelfAttention<T> _temporalAttention;
    private readonly LayerNormalizationLayer<T> _spatialNorm;
    private readonly LayerNormalizationLayer<T> _temporalNorm;
    private Tensor<T>? _lastInput;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the number of channels.
    /// </summary>
    public int Channels => _channels;

    /// <summary>
    /// Gets the number of frames.
    /// </summary>
    public int NumFrames => _numFrames;

    /// <summary>
    /// Initializes a new factorized spatio-temporal attention layer.
    /// </summary>
    /// <param name="channels">Number of feature channels.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="numFrames">Number of video frames.</param>
    /// <param name="spatialSize">Spatial size of feature maps.</param>
    public FactorizedSpatioTemporalAttention(
        int channels,
        int numHeads = 8,
        int numFrames = 16,
        int spatialSize = 64)
        : base(
            new[] { 1, numFrames * spatialSize * spatialSize, channels },
            new[] { 1, numFrames * spatialSize * spatialSize, channels })
    {
        _channels = channels;
        _numHeads = numHeads;
        _numFrames = numFrames;
        _spatialSize = spatialSize;

        _spatialAttention = new DiffusionAttention<T>(
            channels: channels,
            numHeads: numHeads,
            spatialSize: spatialSize);

        _temporalAttention = new TemporalSelfAttention<T>(
            channels: channels,
            numHeads: numHeads,
            numFrames: numFrames,
            spatialSize: spatialSize);

        _spatialNorm = new LayerNormalizationLayer<T>();
        _temporalNorm = new LayerNormalizationLayer<T>();
    }

    /// <summary>
    /// Applies spatial attention then temporal attention with residual connections.
    /// </summary>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        // #1668: skip the backward-activation cache in inference so the denoise-loop
        // arena can recycle scratch without aliasing a stale reference.
        _lastInput = ShouldCacheForBackward ? input : null;

        // Spatial attention with residual
        var spatialNormed = _spatialNorm.Forward(input);
        var spatialOut = _spatialAttention.Forward(spatialNormed);
        var afterSpatial = AddTensors(input, spatialOut);

        // Temporal attention with residual
        var temporalNormed = _temporalNorm.Forward(afterSpatial);
        var temporalOut = _temporalAttention.Forward(temporalNormed);
        var output = AddTensors(afterSpatial, temporalOut);

        return output;
    }

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        _spatialAttention.UpdateParameters(learningRate);
        _temporalAttention.UpdateParameters(learningRate);
        _spatialNorm.UpdateParameters(learningRate);
        _temporalNorm.UpdateParameters(learningRate);
    }

    private static void CopyParams(Vector<T> src, Vector<T> dst, ref int offset)
    {
        for (int i = 0; i < src.Length; i++)
            dst[offset + i] = src[i];
        offset += src.Length;
    }

    private static Vector<T> ExtractParams(Vector<T> src, int count, ref int offset)
    {
        var result = new Vector<T>(count);
        for (int i = 0; i < count; i++)
            result[i] = src[offset + i];
        offset += count;
        return result;
    }

    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        return AiDotNetEngine.Current.TensorAdd(a, b);
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _spatialAttention.ResetState();
        _temporalAttention.ResetState();
        _spatialNorm.ResetState();
        _temporalNorm.ResetState();
    }


}

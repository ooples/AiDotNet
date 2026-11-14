using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.Compression;

/// <summary>
/// Error feedback compressor: wraps any compression method with residual accumulation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Most compression methods are biased — they lose some information.
/// Error feedback fixes this by remembering what was lost (the "residual") and adding it back
/// in the next round. Over time, all information eventually gets transmitted, making the
/// compressed training converge to the same solution as uncompressed training.</para>
///
/// <para><b>How it works:</b></para>
/// <list type="number">
/// <item><description>Before compressing, add the accumulated error from previous rounds to the gradient.</description></item>
/// <item><description>Compress the combined (gradient + error) signal.</description></item>
/// <item><description>Compute new error = (gradient + old_error) - decompressed result.</description></item>
/// <item><description>Store error for next round.</description></item>
/// </list>
///
/// <para><b>Mathematical guarantee:</b> With error feedback, any contractive compressor
/// (compression preserves a fraction of the signal) converges at the same rate as
/// uncompressed SGD, up to a constant factor.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for model parameters.</typeparam>
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
public class ErrorFeedbackCompressor<T> : FederatedLearningComponentBase<T>
{
    private readonly AdvancedCompressionOptions _options;
    private readonly Dictionary<int, Tensor<T>> _clientErrors = new();

    /// <summary>
    /// Initializes a new instance of <see cref="ErrorFeedbackCompressor{T}"/>.
    /// </summary>
    /// <param name="options">Advanced compression configuration.</param>
    public ErrorFeedbackCompressor(AdvancedCompressionOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
    }

    /// <summary>
    /// Applies error feedback to a gradient before compression.
    /// Returns the gradient + accumulated error, which should be passed to the actual compressor.
    /// </summary>
    /// <param name="gradient">Original gradient from the current round.</param>
    /// <param name="clientId">Client ID for per-client error tracking.</param>
    /// <returns>Gradient with accumulated error added (ready for compression).</returns>
    public Tensor<T> ApplyErrorFeedback(Tensor<T> gradient, int clientId)
    {
        if (gradient is null) throw new ArgumentNullException(nameof(gradient));
        if (!_options.UseErrorFeedback) return gradient;

        int size = gradient.Shape[0];
        var error = GetOrResetError(clientId, size);

        // Combined = gradient + accumulated error
        return Engine.TensorAdd(gradient.Reshape(size), error);
    }

    /// <summary>
    /// Updates the error accumulator after compression.
    /// </summary>
    /// <param name="combined">The gradient + error before compression.</param>
    /// <param name="decompressed">The decompressed (lossy) version that was actually sent.</param>
    /// <param name="clientId">The client ID.</param>
    public void UpdateError(Tensor<T> combined, Tensor<T> decompressed, int clientId)
    {
        if (combined is null) throw new ArgumentNullException(nameof(combined));
        if (decompressed is null) throw new ArgumentNullException(nameof(decompressed));
        if (!_options.UseErrorFeedback) return;

        int size = combined.Shape[0];
        GetOrResetError(clientId, size);

        // New error = combined - decompressed (missing decompressed entries count as 0).
        var sent = new Tensor<T>(new[] { size });
        int overlap = Math.Min(size, decompressed.Shape[0]);
        decompressed.AsSpan().Slice(0, overlap).CopyTo(sent.AsWritableSpan());
        _clientErrors[clientId] = Engine.TensorSubtract(combined.Reshape(size), sent);
    }

    /// <summary>
    /// Compresses a gradient using 1-bit SGD with error feedback.
    /// </summary>
    /// <param name="gradient">The raw gradient to compress.</param>
    /// <param name="clientId">The client ID for error tracking.</param>
    /// <returns>Compressed (sign-encoded) gradient and the scaling factor.</returns>
    public (Tensor<T> Compressed, T Scale) CompressOneBit(Tensor<T> gradient, int clientId)
    {
        if (gradient is null) throw new ArgumentNullException(nameof(gradient));

        // Apply error feedback first
        var combined = ApplyErrorFeedback(gradient, clientId);
        int size = combined.Shape[0];

        // Mean absolute value for scaling.
        // (Formed as sum|x| / n with the sum in float for Half / BFloat16, where a running sum saturates at 2048.)
        var scale = size > 0
            ? NumOps.FromDouble(AiDotNet.FederatedLearning.Infrastructure.FederatedReductions.SumAbs(combined) / size)
            : NumOps.Zero;

        // 1-bit encode: sign * scale, with sign(x) = +1 for x >= 0 and -1 otherwise (including NaN).
        var nonNegative = Engine.TensorAdd(
            Engine.TensorGreaterThan(combined, NumOps.Zero), Engine.TensorEquals(combined, NumOps.Zero));
        var compressed = Engine.TensorWhere(
            nonNegative,
            Engine.TensorFullLike(combined, scale),
            Engine.TensorFullLike(combined, NumOps.Negate(scale)));

        // Update error feedback
        UpdateError(combined, compressed, clientId);

        return (compressed, scale);
    }

    /// <summary>
    /// Gets the current error norm for a client (useful for monitoring).
    /// </summary>
    /// <param name="clientId">The client ID.</param>
    /// <returns>L2 norm of the accumulated error, or 0 if no error exists.</returns>
    public T GetErrorNorm(int clientId)
    {
        if (!_clientErrors.TryGetValue(clientId, out var error)) return NumOps.Zero;

        return NumOps.FromDouble(Math.Sqrt(AiDotNet.FederatedLearning.Infrastructure.FederatedReductions.SumOfSquares(error)));
    }

    /// <summary>
    /// Returns the client's error accumulator, replacing it with zeros when missing or when the model size changed.
    /// </summary>
    private Tensor<T> GetOrResetError(int clientId, int size)
    {
        if (!_clientErrors.TryGetValue(clientId, out var error) || error.Shape[0] != size)
        {
            error = new Tensor<T>(new[] { size });
            _clientErrors[clientId] = error;
        }

        return error;
    }

    /// <summary>
    /// Resets error accumulation for a client.
    /// </summary>
    /// <param name="clientId">The client ID.</param>
    public void ResetError(int clientId)
    {
        if (_clientErrors.TryGetValue(clientId, out var error))
        {
            _clientErrors[clientId] = new Tensor<T>(new[] { error.Shape[0] });
        }
    }

    /// <summary>
    /// Resets all accumulated errors across all clients.
    /// </summary>
    public void ResetAllErrors()
    {
        _clientErrors.Clear();
    }
}

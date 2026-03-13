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
public class ErrorFeedbackCompressor<T> : FederatedLearningComponentBase<T>
{
    private readonly AdvancedCompressionOptions _options;
    private readonly Dictionary<int, double[]> _clientErrors = new();

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
        var combined = new Tensor<T>(new[] { size });

        // Get or create error accumulator
        if (!_clientErrors.ContainsKey(clientId))
        {
            _clientErrors[clientId] = new double[size];
        }

        var error = _clientErrors[clientId];

        // Resize error if model size changed
        if (error.Length != size)
        {
            _clientErrors[clientId] = new double[size];
            error = _clientErrors[clientId];
        }

        // Combined = gradient + accumulated error
        for (int i = 0; i < size; i++)
        {
            double grad = NumOps.ToDouble(gradient[i]);
            combined[i] = NumOps.FromDouble(grad + error[i]);
        }

        return combined;
    }

    /// <summary>
    /// Updates the error accumulator after compression.
    /// Call this after compressing and decompressing to track what was lost.
    /// </summary>
    /// <param name="combined">The combined gradient + error that was compressed.</param>
    /// <param name="decompressed">The decompressed result (what was actually sent).</param>
    /// <param name="clientId">Client ID for per-client error tracking.</param>
    public void UpdateError(Tensor<T> combined, Tensor<T> decompressed, int clientId)
    {
        if (combined is null) throw new ArgumentNullException(nameof(combined));
        if (decompressed is null) throw new ArgumentNullException(nameof(decompressed));
        if (!_options.UseErrorFeedback) return;

        int size = combined.Shape[0];

        if (!_clientErrors.ContainsKey(clientId))
        {
            _clientErrors[clientId] = new double[size];
        }

        var error = _clientErrors[clientId];
        if (error.Length != size)
        {
            _clientErrors[clientId] = new double[size];
            error = _clientErrors[clientId];
        }

        // New error = combined - decompressed
        for (int i = 0; i < size; i++)
        {
            double combinedVal = NumOps.ToDouble(combined[i]);
            double decompressedVal = i < decompressed.Shape[0] ? NumOps.ToDouble(decompressed[i]) : 0;
            error[i] = combinedVal - decompressedVal;
        }
    }

    /// <summary>
    /// Applies 1-bit SGD compression (sign-only encoding) with error feedback.
    /// Each gradient component is encoded as +1 or -1 (its sign), scaled by the mean magnitude.
    /// </summary>
    /// <param name="gradient">The gradient tensor to compress.</param>
    /// <param name="clientId">Client ID for per-client error tracking.</param>
    /// <returns>Compressed (sign-encoded) gradient and the scaling factor.</returns>
    public (Tensor<T> Compressed, double Scale) CompressOneBit(Tensor<T> gradient, int clientId)
    {
        if (gradient is null) throw new ArgumentNullException(nameof(gradient));

        // Apply error feedback first
        var combined = ApplyErrorFeedback(gradient, clientId);
        int size = combined.Shape[0];

        // Compute mean absolute value for scaling
        double sumAbs = 0;
        for (int i = 0; i < size; i++)
        {
            sumAbs += Math.Abs(NumOps.ToDouble(combined[i]));
        }

        double scale = size > 0 ? sumAbs / size : 0;

        // 1-bit encode: sign * scale
        var compressed = new Tensor<T>(new[] { size });
        for (int i = 0; i < size; i++)
        {
            double val = NumOps.ToDouble(combined[i]);
            double sign = val >= 0 ? 1.0 : -1.0;
            compressed[i] = NumOps.FromDouble(sign * scale);
        }

        // Update error feedback
        UpdateError(combined, compressed, clientId);

        return (compressed, scale);
    }

    /// <summary>
    /// Gets the accumulated error norm for a client (for monitoring convergence).
    /// </summary>
    /// <param name="clientId">Client ID.</param>
    /// <returns>L2 norm of the accumulated error, or 0 if no error exists.</returns>
    public double GetErrorNorm(int clientId)
    {
        if (!_clientErrors.ContainsKey(clientId)) return 0;

        var error = _clientErrors[clientId];
        double sumSq = 0;
        foreach (double e in error)
        {
            sumSq += e * e;
        }

        return Math.Sqrt(sumSq);
    }

    /// <summary>
    /// Resets the accumulated error for a specific client.
    /// </summary>
    /// <param name="clientId">Client ID whose error to reset.</param>
    public void ResetError(int clientId)
    {
        if (_clientErrors.ContainsKey(clientId))
        {
            Array.Clear(_clientErrors[clientId], 0, _clientErrors[clientId].Length);
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

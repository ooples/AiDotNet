namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for federated update compression (quantization, sparsification, and error feedback).
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Compression reduces the size of client updates sent to the server to save bandwidth.
/// This can speed up training in distributed settings, especially on slow or expensive networks.
/// </remarks>
public class FederatedCompressionOptions : ModelOptions
{
    /// <summary>
    /// Gets or sets the compression strategy name.
    /// </summary>
    /// <remarks>
    /// Supported built-ins:
    /// - "None"
    /// - "TopK"
    /// - "RandomK"
    /// - "Threshold"
    /// - "UniformQuantization"
    /// - "StochasticQuantization"
    /// </remarks>
    public string Strategy { get; set; } = "TopK";

    /// <summary>
    /// Gets or sets the compression ratio (0.0 to 1.0) for sparsification strategies.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> A value of 0.1 means "keep about 10% of the update entries".
    /// </remarks>
    public double Ratio { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the number of bits used for quantization strategies.
    /// </summary>
    public int QuantizationBits { get; set; } = 8;

    /// <summary>
    /// Gets or sets the absolute threshold for the "Threshold" strategy.
    /// </summary>
    public double Threshold { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets whether to use error feedback (residual accumulation) on the client.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Error feedback stores the information lost during compression and
    /// adds it into the next update, which often improves convergence.
    /// </remarks>
    public bool UseErrorFeedback { get; set; } = true;

    /// <summary>
    /// Gets or sets advanced compression options (PowerSGD, sketching, adaptive, 1-bit SGD).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When set, these advanced methods provide 100-1000x compression
    /// beyond what basic TopK/Quantization can achieve. Set <see cref="Strategy"/> to "Advanced"
    /// and configure this property for fine-grained control.</para>
    ///
    /// <para>Set to null (default) to use only the basic compression strategies.</para>
    /// </remarks>
    public AdvancedCompressionOptions? Advanced { get; set; } = null;
}


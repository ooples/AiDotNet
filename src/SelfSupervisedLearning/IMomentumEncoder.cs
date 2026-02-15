using AiDotNet.Interfaces;

namespace AiDotNet.SelfSupervisedLearning;

/// <summary>
/// Defines the contract for momentum-updated encoders used in SSL methods.
/// </summary>
/// <typeparam name="T">The numeric type used for computations (typically float or double).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A momentum encoder is a copy of the main encoder that updates
/// more slowly using exponential moving average (EMA). This provides stable, consistent targets
/// during training.</para>
///
/// <para><b>How it works:</b></para>
/// <code>
/// momentum_encoder_param = m * momentum_encoder_param + (1 - m) * encoder_param
/// </code>
/// <para>Where m is typically 0.99-0.999 (very slow updates).</para>
///
/// <para><b>Why use a momentum encoder?</b></para>
/// <list type="bullet">
/// <item>Provides stable targets that don't change rapidly</item>
/// <item>Prevents model collapse in methods like BYOL</item>
/// <item>Ensures consistent negative embeddings in MoCo</item>
/// </list>
///
/// <para><b>Used by:</b> MoCo, MoCo v2, MoCo v3, BYOL, DINO</para>
/// <para><b>Not used by:</b> SimCLR, SimSiam (uses stop-gradient instead), Barlow Twins</para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("MomentumEncoder")]
public interface IMomentumEncoder<T>
{
    /// <summary>
    /// Gets the momentum coefficient for EMA updates.
    /// </summary>
    /// <remarks>
    /// <para>Typical values: 0.99-0.9999. Higher values = slower updates = more stable targets.</para>
    /// <para>MoCo uses 0.999, BYOL uses 0.996 → 1.0 (scheduled).</para>
    /// </remarks>
    double Momentum { get; }

    /// <summary>
    /// Gets the underlying momentum-updated encoder network.
    /// </summary>
    INeuralNetwork<T> Encoder { get; }

    /// <summary>
    /// Encodes input using the momentum encoder (no gradient computation).
    /// </summary>
    /// <param name="input">The input tensor to encode.</param>
    /// <returns>The encoded representation (detached from computation graph).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The momentum encoder output is treated as a fixed target
    /// - gradients don't flow through it. Only the main encoder is trained directly.</para>
    /// </remarks>
    Tensor<T> Encode(Tensor<T> input);

    /// <summary>
    /// Updates the momentum encoder parameters using EMA from the main encoder.
    /// </summary>
    /// <param name="mainEncoder">The main encoder to copy parameters from.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Call this after each training step. The momentum encoder
    /// slowly tracks the main encoder, providing stable targets.</para>
    ///
    /// <para>Update formula:</para>
    /// <code>θ_momentum = m * θ_momentum + (1 - m) * θ_main</code>
    /// </remarks>
    void UpdateFromMainEncoder(INeuralNetwork<T> mainEncoder);

    /// <summary>
    /// Updates the momentum encoder parameters using EMA from parameter vectors.
    /// </summary>
    /// <param name="mainEncoderParams">Parameters from the main encoder.</param>
    void UpdateFromParameters(Vector<T> mainEncoderParams);

    /// <summary>
    /// Sets the momentum coefficient.
    /// </summary>
    /// <param name="momentum">New momentum value (0-1, typically 0.99-0.9999).</param>
    /// <remarks>
    /// <para>Some methods schedule momentum during training (e.g., BYOL increases from 0.996 to 1.0).</para>
    /// </remarks>
    void SetMomentum(double momentum);

    /// <summary>
    /// Copies all parameters from the main encoder (hard copy, not EMA).
    /// </summary>
    /// <param name="mainEncoder">The encoder to copy from.</param>
    /// <remarks>
    /// <para>Used for initialization at the start of training.</para>
    /// </remarks>
    void CopyFromMainEncoder(INeuralNetwork<T> mainEncoder);

    /// <summary>
    /// Gets all parameters of the momentum encoder.
    /// </summary>
    /// <returns>A vector containing all parameters.</returns>
    Vector<T> GetParameters();

    /// <summary>
    /// Sets the parameters of the momentum encoder directly.
    /// </summary>
    /// <param name="parameters">The parameters to set.</param>
    void SetParameters(Vector<T> parameters);
}

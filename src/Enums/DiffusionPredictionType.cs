namespace AiDotNet.Enums;

/// <summary>
/// Defines what the diffusion model predicts during the denoising process.
/// </summary>
/// <remarks>
/// <para>
/// Different diffusion models can be trained to predict different targets.
/// The prediction type affects how the scheduler interprets the model output
/// and computes the denoised sample.
/// </para>
/// <para>
/// <b>For Beginners:</b> When a model looks at a noisy image, it can be trained to predict:
///
/// - Epsilon (noise): "What noise was added to make this blurry?"
/// - Sample: "What did the clean image look like?"
/// - V-prediction: A mathematical blend of both (more stable training)
///
/// Most models use Epsilon prediction as it's the most common and well-studied approach.
/// </para>
/// </remarks>
public enum DiffusionPredictionType
{
    /// <summary>
    /// Model predicts the noise (epsilon) that was added to the clean sample.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is the most common prediction type, used in the original DDPM paper.
    /// The model learns: given x_t, predict epsilon such that x_t = sqrt(alpha_cumprod) * x_0 + sqrt(1-alpha_cumprod) * epsilon
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The model looks at a noisy image and guesses
    /// "what noise was added to make it look like this?" This is like asking
    /// someone to identify what static was overlaid on a TV signal.
    /// </para>
    /// </remarks>
    Epsilon,

    /// <summary>
    /// Model directly predicts the clean sample (x_0).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Instead of predicting noise, the model directly predicts what the original
    /// clean sample looked like. This can sometimes provide more direct gradients
    /// but may be harder to train.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The model looks at a noisy image and directly guesses
    /// "what was the original clear image?" This is like asking someone to
    /// imagine what a static-filled TV picture would look like if it were clear.
    /// </para>
    /// </remarks>
    Sample,

    /// <summary>
    /// Model predicts v = sqrt(alpha_cumprod) * epsilon - sqrt(1-alpha_cumprod) * x_0.
    /// </summary>
    /// <remarks>
    /// <para>
    /// V-prediction is a velocity-based parameterization that can provide more stable
    /// training, especially for continuous-time diffusion models. It's used in some
    /// advanced diffusion models like Imagen.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is a mathematical trick that combines noise and
    /// sample prediction in a way that makes training more stable. Think of it as
    /// predicting "the direction of change" rather than either endpoint.
    /// </para>
    /// </remarks>
    VPrediction
}

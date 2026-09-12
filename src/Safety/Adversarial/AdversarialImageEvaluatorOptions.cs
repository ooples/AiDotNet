using AiDotNet.Models.Options;

namespace AiDotNet.Safety.Adversarial;

/// <summary>
/// Configuration options for <see cref="AdversarialImageEvaluator{T}"/>.
/// </summary>
/// <remarks>
/// <para>
/// Introduced by issue #2090: the decision threshold was a constructor parameter, so the model
/// advertised no configuration surface and the value could not be set through an options object.
/// </para>
/// </remarks>
public class AdversarialImageEvaluatorOptions : ModelHyperparameterOptions
{
    /// <summary>
    /// Gets or sets the score above which an image is judged adversarial. Default: 0.5.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The evaluator scores each image between 0 and 1 for how likely it is
    /// to have been deliberately tampered with. This is the line it draws: above it, the image is
    /// reported as an attack. Lowering it catches more attacks but flags more harmless images;
    /// raising it does the opposite.
    /// </para>
    /// </remarks>
    public double Threshold { get; set; } = 0.5;

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when <see cref="Threshold"/> is not within [0, 1].
    /// </exception>
    /// <remarks>
    /// <para>
    /// Range-checked rather than passed to <c>Require</c>, because zero is a meaningful threshold —
    /// it means "report everything" — and <c>Require</c> rejects zero.
    /// </para>
    /// </remarks>
    public void Validate()
    {
        if (Threshold < 0.0 || Threshold > 1.0)
        {
            throw new ArgumentException(
                $"{GetType().Name}.{nameof(Threshold)} is {Threshold}, but it must be between 0 and 1.",
                OptionsParameterName);
        }
    }
}

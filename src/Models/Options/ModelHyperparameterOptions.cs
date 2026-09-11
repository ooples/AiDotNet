namespace AiDotNet.Models.Options;

/// <summary>
/// Base class for the per-model-family hyperparameter options: the sizes and shapes that
/// come from the paper a model was published in.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Every model in this library ships with the settings from the
/// research paper that introduced it, so you can use a model without configuring anything.
/// This class holds the settings that apply to nearly every model, and the family classes
/// that derive from it add the ones specific to a kind of model.
/// </para>
/// <para>
/// <b>Pattern.</b> Properties here are non-nullable, and each concrete options class assigns
/// its published defaults in its parameterless constructor. This differs deliberately from
/// the nullable + <c>GetEffectiveX()</c> pattern used for infrastructure configuration
/// (telemetry, profiling, AutoML): <c>null</c> there means "decide at runtime from the data
/// or the hardware", which is meaningful for a batch size and meaningless for a layer count.
/// </para>
/// <para>
/// <b>Every property here must be read by the model that owns it.</b> A property nobody
/// reads is worse than no property, because it advertises configurability that does not
/// exist. The options-surface ratchet test enforces this.
/// </para>
/// </remarks>
public abstract class ModelHyperparameterOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Gets or sets the maximum global gradient norm, above which gradients are rescaled
    /// during training. Zero or negative disables clipping.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> During training the model adjusts itself based on how wrong
    /// it was. Occasionally that correction is enormous and destabilises everything learned so
    /// far. Gradient clipping caps the size of a single correction. 1.0 is the usual choice.</para>
    /// </remarks>
    public double MaxGradNorm { get; set; } = 1.0;

    /// <summary>
    /// The constructor parameter name reported when validation fails.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Every model takes its configuration as a parameter called <c>options</c>, so an invalid
    /// value always reached the model that way. Naming it keeps the exception actionable and
    /// keeps the contract these constructors already had before their scalar parameters moved
    /// here.
    /// </para>
    /// </remarks>
    protected const string OptionsParameterName = "options";

    /// <summary>
    /// Throws when a required dimension has been left unset.
    /// </summary>
    /// <param name="value">The value to check.</param>
    /// <param name="propertyName">The name of the property being checked.</param>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="value"/> is zero or negative. Reported against the
    /// <c>options</c> parameter, because that is how an invalid value reaches a model.
    /// </exception>
    /// <remarks>
    /// <para>
    /// Fails loudly rather than letting a zero-width dimension through. A zero produces layers
    /// that appear to build and then misbehave a long way from the cause, which is far more
    /// expensive to diagnose than an exception naming the property.
    /// </para>
    /// </remarks>
    protected void Require(int value, string propertyName)
    {
        if (value <= 0)
        {
            throw new ArgumentException(
                $"{GetType().Name}.{propertyName} is {value}, but it must be greater than zero. "
                    + $"Set it explicitly, or ensure {GetType().Name}'s parameterless constructor "
                    + "assigns its model's published default.",
                OptionsParameterName);
        }
    }

    /// <summary>
    /// Throws when a required rate or ratio has been left unset.
    /// </summary>
    /// <param name="value">The value to check.</param>
    /// <param name="propertyName">The name of the property being checked.</param>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="value"/> is not a finite number greater than zero.
    /// </exception>
    protected void Require(double value, string propertyName)
    {
        if (double.IsNaN(value) || double.IsInfinity(value) || value <= 0.0)
        {
            throw new ArgumentException(
                $"{GetType().Name}.{propertyName} is {value.ToString(System.Globalization.CultureInfo.InvariantCulture)}, "
                    + "but it must be a finite number greater than zero. Set it explicitly, or ensure "
                    + $"{GetType().Name}'s parameterless constructor assigns its model's published default.",
                OptionsParameterName);
        }
    }
}

using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the HyperbolicNeuralNetwork.
/// </summary>
public class HyperbolicNeuralNetworkOptions : ModelHyperparameterOptions
{

    /// <summary>
    /// Gets or sets curvature. Default: <c>-1.0</c>.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How strongly the space the model works in is curved. Negative is hyperbolic.</para>
    /// </remarks>
    public double Curvature { get; set; } = -1.0;

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    /// <remarks>
    /// Curvature is range-checked rather than passed to <c>Require</c>: the base helper demands a
    /// value greater than zero, but hyperbolic space is by definition negatively curved, so the
    /// published default of -1.0 would fail its own validation. Zero (Euclidean) and positive
    /// (spherical) curvature both describe a different geometry than this model implements.
    /// </remarks>
    /// <exception cref="ArgumentException">
    /// Thrown when <see cref="Curvature"/> is not a finite number less than zero.
    /// </exception>
    public void Validate()
    {
        if (double.IsNaN(Curvature) || double.IsInfinity(Curvature) || Curvature >= 0.0)
        {
            throw new ArgumentException(
                $"{GetType().Name}.{nameof(Curvature)} is "
                    + $"{Curvature.ToString(System.Globalization.CultureInfo.InvariantCulture)}, "
                    + "but hyperbolic space requires a finite negative curvature (the published "
                    + "default is -1.0).",
                OptionsParameterName);
        }
    }
}

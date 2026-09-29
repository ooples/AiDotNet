using AiDotNet.Models.Options;

namespace AiDotNet.UncertaintyQuantification.BayesianNeuralNetworks;

/// <summary>
/// Configuration options for <see cref="BayesianNeuralNetwork{T}"/>.
/// </summary>
/// <remarks>
/// <para>
/// Introduced by issue #2090: the sample count was a constructor parameter, so the model
/// advertised no configuration surface and the value could not be set through an options object.
/// </para>
/// </remarks>
public class BayesianNeuralNetworkOptions : ModelHyperparameterOptions
{
    /// <summary>
    /// Gets or sets how many forward passes are averaged to estimate a prediction and its
    /// uncertainty. Default: 30.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Bayesian network does not have one fixed set of weights — it has a
    /// distribution over them. To predict, it draws several different weight samples, runs the
    /// input through each, and looks at the spread of the answers. More samples give a steadier
    /// estimate and a better sense of how unsure the model is, at proportionally more work.
    /// </para>
    /// <para>
    /// Deliberately NOT shared with <see cref="MCDropoutNeuralNetworkOptions"/>, which publishes 50
    /// for the same-named knob: a shared base would have to pick one of the two and silently change
    /// the other model.
    /// </para>
    /// </remarks>
    public int NumSamples { get; set; } = 30;

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when <see cref="NumSamples"/> is not at least one.
    /// </exception>
    public void Validate()
    {
        Require(NumSamples, nameof(NumSamples));
    }
}

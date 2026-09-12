using AiDotNet.Models.Options;

namespace AiDotNet.UncertaintyQuantification.BayesianNeuralNetworks;

/// <summary>
/// Configuration options for <see cref="MCDropoutNeuralNetwork{T}"/>.
/// </summary>
/// <remarks>
/// <para>
/// Introduced by issue #2090: the sample count was a constructor parameter, so the model
/// advertised no configuration surface and the value could not be set through an options object.
/// </para>
/// </remarks>
public class MCDropoutNeuralNetworkOptions : ModelHyperparameterOptions
{
    /// <summary>
    /// Gets or sets how many forward passes with dropout left ON are averaged to estimate a
    /// prediction and its uncertainty. Default: 50.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Dropout normally switches off during prediction. Monte-Carlo dropout
    /// deliberately leaves it on and predicts several times, so each pass uses a slightly different
    /// subset of the network. The spread across those passes is the model's uncertainty, and more
    /// passes make that estimate steadier at proportionally more work.
    /// </para>
    /// <para>
    /// Deliberately NOT shared with <see cref="BayesianNeuralNetworkOptions"/>, which publishes 30
    /// for the same-named knob: a shared base would have to pick one of the two and silently change
    /// the other model.
    /// </para>
    /// </remarks>
    public int NumSamples { get; set; } = 50;

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

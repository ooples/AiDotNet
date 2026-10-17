namespace AiDotNet.TextToSpeech.EndToEnd;

/// <summary>Options for VITS (Kim et al. 2021): a conditional VAE with a normalizing-flow prior, a stochastic duration
/// predictor and a HiFi-GAN decoder trained adversarially end to end.</summary>
/// <remarks>
/// <para>
/// The shared settings are in <see cref="VitsModelOptions"/>. VITS adds its stochastic duration predictor (kernel 3,
/// dropout 0.5, 4 spline flows; the reference's <c>StochasticDurationPredictor(hidden, 192, 3, 0.5, 4)</c>) sampled with
/// noise scale 0.8 at synthesis, and places blank tokens between the characters.
/// </para>
/// <para><b>For Beginners:</b> These options configure the VITS model. Default values follow the original paper settings.</para>
/// </remarks>
public class VITSOptions : VitsModelOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public VITSOptions(VITSOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        DurationPredictorFlows = other.DurationPredictorFlows;
        DurationNoiseScale = other.DurationNoiseScale;
    }

    /// <summary>Creates the paper's configuration.</summary>
    public VITSOptions()
    {
        AddBlank = true;
    }

    /// <summary>Gets or sets the stochastic duration predictor's spline flows (4).</summary>
    public int DurationPredictorFlows { get; set; } = 4;

    /// <summary>Gets or sets the duration-predictor noise scale at synthesis (0.8).</summary>
    public double DurationNoiseScale { get; set; } = 0.8;
}

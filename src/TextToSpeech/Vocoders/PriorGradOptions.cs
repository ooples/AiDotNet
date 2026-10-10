namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>Options for the PriorGrad vocoder (Lee et al. 2022): DiffWave with a data-dependent diagonal Gaussian
/// prior whose standard deviation follows the mel spectrogram's frame energy.</summary>
/// <remarks>
/// <para>
/// Defaults are the paper's vocoder (§4): the DiffWave BASE network and schedules it builds on (30 residual layers of 64
/// channels, T = 50 with β linear from 1e-4 to 0.05, the six-step fast schedule 1e-4, 1e-3, 0.01, 0.05, 0.2, 0.5);
/// 22.05 kHz audio, 80 log-mel bands from a 1024-point FFT with hop 256 between 80 Hz and 7.6 kHz; the prior's
/// standard deviation the frame energy √Σ exp(mel) normalized to (0, 1] and clipped below at 0.1; Adam at 2e-4.
/// </para>
/// <para>What the paper leaves open follows microsoft/NeuralSpeech PriorGrad-vocoder: the HiFi-GAN feature pipeline
/// (reflect padding, a 1024-sample Hann window, magnitude, librosa's Slaney bands, <c>ln(max(x, 1e-5))</c>); the energy
/// normalized by training-set extremes with the maximum overridden to 4; batch 16; 62-frame crops; no gradient
/// clipping; each sampling step clamped to [−1, 1]. Without fitted statistics (<see cref="EnergyMin"/> null) the
/// minimum is the energy of silence, √(mel bands · 1e-5).</para>
/// <para><b>For Beginners:</b> These options configure the PriorGrad model. Default values follow the original paper settings.</para>
/// </remarks>
public class PriorGradOptions : DiffWaveOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public PriorGradOptions(PriorGradOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        MelMaxFrequency = other.MelMaxFrequency;
        EnergyMin = other.EnergyMin;
        EnergyMax = other.EnergyMax;
        MinStd = other.MinStd;
    }

    /// <summary>Creates the paper's PriorGrad vocoder configuration.</summary>
    public PriorGradOptions()
    {
        MelMinFrequency = 80;
    }

    /// <summary>Gets or sets the highest mel frequency (7.6 kHz).</summary>
    public double MelMaxFrequency { get; set; } = 7600;

    /// <summary>Gets or sets the training set's lowest frame energy, or null for the energy of silence.</summary>
    public double? EnergyMin { get; set; }

    /// <summary>Gets or sets the frame energy that maps to a standard deviation of 1; higher energies are clipped to it
    /// (4, the reference's override of the training-set maximum).</summary>
    public double EnergyMax { get; set; } = 4.0;

    /// <summary>Gets or sets the prior's minimum standard deviation (0.1).</summary>
    public double MinStd { get; set; } = 0.1;
}

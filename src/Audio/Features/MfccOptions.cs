namespace AiDotNet.Audio.Features;

/// <summary>
/// Options for MFCC extraction.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Mfcc model. Default values follow the original paper settings.</para>
/// </remarks>
public class MfccOptions : AudioFeatureOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public MfccOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MfccOptions(MfccOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        NumCoefficients = other.NumCoefficients;
        NumMels = other.NumMels;
        IncludeEnergy = other.IncludeEnergy;
        AppendDelta = other.AppendDelta;
        AppendDeltaDelta = other.AppendDeltaDelta;
        FMin = other.FMin;
        FMax = other.FMax;
    }

    /// <summary>
    /// Gets or sets the number of MFCC coefficients to compute.
    /// Default is 13 (standard for speech recognition).
    /// </summary>
    public int NumCoefficients { get; set; } = 13;

    /// <summary>
    /// Gets or sets the number of Mel filterbank channels.
    /// Default is 40.
    /// </summary>
    public int NumMels { get; set; } = 40;

    /// <summary>
    /// Gets or sets whether to replace the first coefficient with log energy.
    /// </summary>
    public bool IncludeEnergy { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to append delta (velocity) coefficients.
    /// </summary>
    public bool AppendDelta { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to append delta-delta (acceleration) coefficients.
    /// Requires AppendDelta to be true.
    /// </summary>
    public bool AppendDeltaDelta { get; set; } = false;

    /// <summary>
    /// Gets or sets the minimum frequency for the mel filterbank.
    /// </summary>
    public double FMin { get; set; } = 0;

    /// <summary>
    /// Gets or sets the maximum frequency for the mel filterbank.
    /// Null means use Nyquist frequency (SampleRate / 2).
    /// </summary>
    public double? FMax { get; set; }
}

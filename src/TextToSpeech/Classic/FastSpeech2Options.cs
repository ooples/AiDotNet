namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for FastSpeech 2 (variance adaptor with pitch, energy, and duration predictors).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the FastSpeech2 model. Default values follow the original paper settings.</para>
/// </remarks>
public class FastSpeech2Options : AcousticModelOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public FastSpeech2Options(FastSpeech2Options other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        VariancePredictorFilterSize = other.VariancePredictorFilterSize;
        VariancePredictorKernelSize = other.VariancePredictorKernelSize;
        NumPitchBins = other.NumPitchBins;
        NumEnergyBins = other.NumEnergyBins;
        UsePitchPredictor = other.UsePitchPredictor;
        UseEnergyPredictor = other.UseEnergyPredictor;
    }

    public FastSpeech2Options() { EncoderDim = 256; DecoderDim = 80; HiddenDim = 256; NumEncoderLayers = 4; NumDecoderLayers = 4; NumHeads = 2; SampleRate = 22050; MelChannels = 80; HopSize = 256; FftSize = 1024; VocabSize = 256; }

    /// <summary>Gets or sets the variance predictor filter size.</summary>
    public int VariancePredictorFilterSize { get; set; } = 256;

    /// <summary>Gets or sets the variance predictor kernel size.</summary>
    public int VariancePredictorKernelSize { get; set; } = 3;

    /// <summary>Gets or sets the number of pitch bins for quantization.</summary>
    public int NumPitchBins { get; set; } = 256;

    /// <summary>Gets or sets the number of energy bins for quantization.</summary>
    public int NumEnergyBins { get; set; } = 256;

    /// <summary>Gets or sets whether to use pitch prediction.</summary>
    public bool UsePitchPredictor { get; set; } = true;

    /// <summary>Gets or sets whether to use energy prediction.</summary>
    public bool UseEnergyPredictor { get; set; } = true;
}

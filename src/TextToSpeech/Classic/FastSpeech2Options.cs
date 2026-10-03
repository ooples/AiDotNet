namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for FastSpeech 2 (variance adaptor with pitch, energy, and duration predictors).</summary>
/// <remarks>
/// <para>Defaults are the paper's LJSpeech configuration (Ren et al. 2021, App. A, Table 7).</para>
/// <para><b>For Beginners:</b> These options configure the FastSpeech2 model. Default values follow the original paper settings.</para>
/// </remarks>
public class FastSpeech2Options : AcousticModelOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public FastSpeech2Options(FastSpeech2Options other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        VariancePredictorFilterSize = other.VariancePredictorFilterSize;
        VariancePredictorKernelSize = other.VariancePredictorKernelSize;
        VariancePredictorDropout = other.VariancePredictorDropout;
        NumPitchBins = other.NumPitchBins;
        NumEnergyBins = other.NumEnergyBins;
        PitchMinHz = other.PitchMinHz;
        PitchMaxHz = other.PitchMaxHz;
        FftFilterSize = other.FftFilterSize;
        FftKernelSizes = (int[])other.FftKernelSizes.Clone();
        UsePitchPredictor = other.UsePitchPredictor;
        UseEnergyPredictor = other.UseEnergyPredictor;
    }

    public FastSpeech2Options()
    {
        EncoderDim = 256;
        DecoderDim = 80;
        HiddenDim = 256;
        NumEncoderLayers = 4;
        NumDecoderLayers = 4;
        NumHeads = 2;
        SampleRate = 22050;
        MelChannels = 80;
        HopSize = 256;
        FftSize = 1024;
        VocabSize = 256;
        UsePostnet = false; // FastSpeech 2 has no postnet.
    }

    /// <summary>Gets or sets the variance predictor filter size (256).</summary>
    public int VariancePredictorFilterSize { get; set; } = 256;

    /// <summary>Gets or sets the variance predictor kernel size (3).</summary>
    public int VariancePredictorKernelSize { get; set; } = 3;

    /// <summary>Gets or sets the variance predictor dropout (0.5).</summary>
    public double VariancePredictorDropout { get; set; } = 0.5;

    /// <summary>Gets or sets the number of pitch bins for quantization (256, log scale).</summary>
    public int NumPitchBins { get; set; } = 256;

    /// <summary>Gets or sets the number of energy bins for quantization (256, uniform).</summary>
    public int NumEnergyBins { get; set; } = 256;

    /// <summary>Gets or sets the lowest pitch bin edge in Hz (WORLD DIO's F0 floor, 71 Hz).</summary>
    public double PitchMinHz { get; set; } = 71.0;

    /// <summary>Gets or sets the highest pitch bin edge in Hz (WORLD DIO's F0 ceiling, 800 Hz).</summary>
    public double PitchMaxHz { get; set; } = 800.0;

    /// <summary>Gets or sets the filter size of the FFT blocks' first convolution (1024).</summary>
    public int FftFilterSize { get; set; } = 1024;

    /// <summary>Gets or sets the kernel sizes of the FFT blocks' two convolutions (9 and 1).</summary>
    public int[] FftKernelSizes { get; set; } = { 9, 1 };

    /// <summary>Gets or sets whether to use pitch prediction (the paper's ablation removes it).</summary>
    public bool UsePitchPredictor { get; set; } = true;

    /// <summary>Gets or sets whether to use energy prediction (the paper's ablation removes it).</summary>
    public bool UseEnergyPredictor { get; set; } = true;
}

namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Options for AdaSpeech (adaptive TTS with acoustic condition modeling and conditional layer normalization).</summary>
/// <remarks>
/// <para>
/// AdaSpeech "follows the basic structure in FastSpeech 2" and "other model configurations follow Ren et al. (2021)
/// unless otherwise stated" (Chen et al. 2021, §3), so these options extend <see cref="FastSpeech2Options"/>: 4 FFT
/// blocks in the phoneme encoder and in the mel decoder, hidden 256 (phoneme embedding, speaker embedding, attention
/// and FFN input/output), 2 heads, FFN filter 1024 with kernel 9, an 80-bin mel output. Audio is 16 kHz with a 12.5 ms
/// hop and 50 ms window. The acoustic condition encoders use 256 filters; the phoneme-level vectors are 4-dimensional.
/// </para>
/// <para><b>For Beginners:</b> These options configure the AdaSpeech model. Default values follow the original paper settings.</para>
/// </remarks>
public class AdaSpeechOptions : FastSpeech2Options
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public AdaSpeechOptions(AdaSpeechOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        NumSpeakers = other.NumSpeakers;
        AcousticConditionFilterSize = other.AcousticConditionFilterSize;
        UtteranceEncoderKernelSize = other.UtteranceEncoderKernelSize;
        UtteranceEncoderStride = other.UtteranceEncoderStride;
        PhonemeEncoderKernelSize = other.PhonemeEncoderKernelSize;
        PhonemeConditionDim = other.PhonemeConditionDim;
        AcousticConditionDropout = other.AcousticConditionDropout;
    }

    /// <summary>Creates the paper's LibriTTS configuration.</summary>
    public AdaSpeechOptions()
    {
        SampleRate = 16000;
        HopSize = 200;
        FftSize = 1024;
    }

    /// <summary>Gets or sets the number of speakers in the speaker embedding table (LibriTTS: 2456).</summary>
    public int NumSpeakers { get; set; } = 2456;

    /// <summary>Gets or sets the filters of the acoustic condition encoders and predictor (256).</summary>
    /// <remarks>The utterance-level vector is added to the hidden sequence, so the utterance-level encoder uses
    /// <see cref="TtsModelOptions.HiddenDim"/> filters; the paper's 256 is that width.</remarks>
    public int AcousticConditionFilterSize { get; set; } = 256;

    /// <summary>Gets or sets the utterance-level encoder's convolution kernel (5).</summary>
    public int UtteranceEncoderKernelSize { get; set; } = 5;

    /// <summary>Gets or sets the utterance-level encoder's convolution stride (3).</summary>
    public int UtteranceEncoderStride { get; set; } = 3;

    /// <summary>Gets or sets the phoneme-level encoder's and predictor's convolution kernel (3).</summary>
    public int PhonemeEncoderKernelSize { get; set; } = 3;

    /// <summary>Gets or sets the dimension of the phoneme-level acoustic vectors (4).</summary>
    public int PhonemeConditionDim { get; set; } = 4;

    /// <summary>Gets or sets the dropout in the acoustic condition encoders and predictor.</summary>
    /// <remarks>The paper does not state it; 0.5 is the dropout of FastSpeech 2's variance predictor, the
    /// identically structured <c>Conv1D → ReLU → LN → Dropout</c> network the backbone uses.</remarks>
    public double AcousticConditionDropout { get; set; } = 0.5;
}

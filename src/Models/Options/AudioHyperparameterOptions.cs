namespace AiDotNet.Models.Options;

/// <summary>
/// Shared configuration for audio and speech models: text-to-speech, speech recognition,
/// speech enhancement and denoising, audio classification and audio generation.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Audio models first turn a sound wave into a picture-like grid of
/// numbers (a spectrogram) and then work on that. Most of the settings here describe how that
/// conversion is done — how many samples per second the audio has, how finely it is chopped
/// up in time, and how many frequency bands are kept. Each model's own options class ships
/// the values from the paper that introduced it, so you do not normally set any of these.
/// </para>
/// <para>
/// Derived options classes assign their paper's values in their parameterless constructor.
/// See <see cref="ModelHyperparameterOptions"/> for why these properties are non-nullable.
/// </para>
/// <para>
/// This base spans <c>src/TextToSpeech</c>, <c>src/SpeechRecognition</c> and <c>src/Audio</c>
/// because they share the same signal parameters. It lives beside
/// <see cref="DocumentNeuralNetworkOptions"/> rather than in any one of those areas for the
/// same reason.
/// </para>
/// <para>
/// <b>Naming.</b> The constructors being replaced use two names for each of two concepts —
/// <c>hopLength</c> and <c>hopSize</c>, <c>fftSize</c> and <c>frameSize</c>. This base picks
/// one of each (<see cref="HopLength"/>, <see cref="FftSize"/>), which is the usual naming in
/// the audio literature and in librosa.
/// </para>
/// </remarks>
public abstract class AudioHyperparameterOptions : ModelHyperparameterOptions
{
    /// <summary>
    /// Gets or sets the audio sample rate in hertz.
    /// </summary>
    /// <value>Zero in this base. The derived model supplies the positive sample rate
    /// required by its training data and published configuration.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many measurements of the sound wave there are per
    /// second. 16000 is the speech standard, 44100 is CD quality. A model trained at one rate
    /// will produce nonsense if fed audio at another, so this has to match the model, not your
    /// source file — resample the audio instead of changing this.</para>
    /// </remarks>
    public int SampleRate { get; set; }

    /// <summary>
    /// Gets or sets the number of mel frequency bands in the spectrogram.
    /// </summary>
    /// <value>Zero in this base. Models using mel features supply their published
    /// band count; models that do not use mel features may leave this unused.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The sound is split into this many frequency bands, spaced
    /// the way human hearing perceives pitch rather than evenly. 80 is near-universal for
    /// speech models.</para>
    /// </remarks>
    public int NumMels { get; set; }

    /// <summary>
    /// Gets or sets the FFT window size, in samples, used to build each spectrogram frame.
    /// </summary>
    /// <value>Zero in this base. A spectrogram-based model supplies the window size
    /// from its signal-processing configuration.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How big a slice of audio is analysed at once. Larger sees
    /// frequency more precisely but time less precisely, and vice versa — that trade-off is
    /// unavoidable.</para>
    /// </remarks>
    public int FftSize { get; set; }

    /// <summary>
    /// Gets or sets the hop length, in samples, between consecutive spectrogram frames.
    /// </summary>
    /// <value>Zero in this base. A spectrogram-based model supplies the frame spacing
    /// matching its training configuration.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How far the analysis window slides each step. Usually a
    /// quarter of <see cref="FftSize"/>, so the windows overlap and nothing falls between
    /// them.</para>
    /// </remarks>
    public int HopLength { get; set; }

    /// <summary>
    /// Gets or sets the width of the model's main hidden representation.
    /// </summary>
    /// <value>Zero in this base. The derived model supplies its architecture's hidden width.</value>
    /// <remarks><para><b>For Beginners:</b> Each sound or text position is represented by
    /// this many numbers inside the network. A wider representation usually requires
    /// more memory and computation; use the width expected by pretrained weights.</para></remarks>
    public int HiddenDim { get; set; }

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// </summary>
    /// <value>Zero in this base. Models using attention supply their architecture's
    /// head count; models without attention may leave it unused.</value>
    /// <remarks><para><b>For Beginners:</b> Attention heads let the model compare several
    /// relationships at once. The head count must be compatible with the hidden width
    /// and the model's attention implementation.</para></remarks>
    public int NumHeads { get; set; }

    /// <summary>
    /// Gets or sets the number of encoder layers.
    /// </summary>
    /// <value>Zero in this base. Models with an encoder supply the depth from their
    /// architecture configuration.</value>
    /// <remarks><para><b>For Beginners:</b> Encoder layers progressively interpret the
    /// input sound or text. More layers add processing steps and increase computation;
    /// pretrained weights require the matching depth.</para></remarks>
    public int NumEncoderLayers { get; set; }

    /// <summary>
    /// Gets or sets the number of decoder layers, for models that generate audio or text.
    /// </summary>
    /// <value>Zero in this base. Models with a decoder supply the depth from their
    /// architecture configuration.</value>
    /// <remarks><para><b>For Beginners:</b> Decoder layers turn the encoded information
    /// into the output representation. More layers increase computation; models without
    /// a decoder do not consume this setting.</para></remarks>
    public int NumDecoderLayers { get; set; }

    /// <summary>
    /// Gets or sets the speaking-rate multiplier for synthesised speech. 1.0 is the model's
    /// natural pace.
    /// </summary>
    /// <value>1.0, the shared neutral multiplier. A derived model may override it
    /// according to its synthesis configuration.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Above 1.0 talks faster, below 1.0 slower.</para>
    /// </remarks>
    public double SpeakingRate { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the BCP-47 language tag the model is configured for, or null when the
    /// model is language-agnostic.
    /// </summary>
    /// <value>Null in this base, meaning no language tag is selected. A derived model
    /// or caller can supply a tag supported by that model.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> "en" for English, "fr" for French. Multilingual models
    /// leave this null and detect the language themselves.</para>
    /// </remarks>
    public string? Language { get; set; }

    /// <summary>
    /// Throws if a signal parameter every audio model requires has been left unset.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required value is zero or negative, which means the derived options class
    /// did not assign its model's published defaults.
    /// </exception>
    protected void ValidateCore()
    {
        Require(SampleRate, nameof(SampleRate));
    }
}

using AiDotNet.Models.Options;
using AiDotNet.Onnx;

using AiDotNet.Audio.AudioGen;

namespace AiDotNet.Audio.AudioGen;

/// <summary>
/// Configuration options for audio generation models.
/// </summary>
/// <remarks>
/// <para>
/// AudioGen models generate audio from text descriptions using a language model
/// approach with discrete audio codes (like EnCodec).
/// </para>
/// <para><b>For Beginners:</b> AudioGen works differently from TTS:
/// - TTS: Converts specific text to spoken words
/// - AudioGen: Creates sounds/music matching a description
///
/// Example prompts:
/// - "A dog barking in the distance"
/// - "Gentle piano music with rain sounds"
/// - "Crowd cheering at a sports event"
/// </para>
/// </remarks>
public class AudioGenOptions : AudioNeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public AudioGenOptions()
    {
        ModelSize = AudioGenModelSize.Medium;
        SampleRate = 16000;
        DurationSeconds = 5.0;
        MaxDurationSeconds = 30.0;
        Temperature = 1.0;
        TopK = 250;
        TopP = 0.0;
        GuidanceScale = 3.0;
        Channels = 1;
        TextHiddenDim = 0;
        LmHiddenDim = 0;
        NumLmLayers = 0;
        NumHeads = 0;
        NumCodebooks = 4;
        CodebookSize = 1024;
        MaxTextLength = 256;
    }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public AudioGenOptions(AudioGenOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        ModelSize = other.ModelSize;
        SampleRate = other.SampleRate;
        DurationSeconds = other.DurationSeconds;
        MaxDurationSeconds = other.MaxDurationSeconds;
        Temperature = other.Temperature;
        TopK = other.TopK;
        TopP = other.TopP;
        GuidanceScale = other.GuidanceScale;
        Channels = other.Channels;
        OnnxOptions = other.OnnxOptions;
        TextHiddenDim = other.TextHiddenDim;
        LmHiddenDim = other.LmHiddenDim;
        NumLmLayers = other.NumLmLayers;
        NumHeads = other.NumHeads;
        NumCodebooks = other.NumCodebooks;
        CodebookSize = other.CodebookSize;
        MaxTextLength = other.MaxTextLength;
        EncodecWeightsPath = other.EncodecWeightsPath;
    }

    /// <summary>
    /// Gets or sets the model size to use.
    /// </summary>
    public AudioGenModelSize ModelSize { get; set; } = AudioGenModelSize.Medium;

    /// <summary>
    /// Gets or sets the output sample rate in Hz.
    /// </summary>
    /// <remarks>
    /// Defaults to 16000, the rate of AudioGen (Kreuk et al. 2022) and of its released compression
    /// model: EnCodec tokens at 50 per second with upsampling ratios 8 x 5 x 4 x 2, a hop of 320
    /// samples. Earlier versions defaulted to 32000 (MusicGen's rate); set it explicitly to keep that.
    /// Other rates still work: the decoder hop is the rate divided by 50.
    /// </remarks>
    public int SampleRate { get; set; } = 16000;

    /// <summary>
    /// Gets or sets the duration of generated audio in seconds.
    /// </summary>
    public double DurationSeconds { get; set; } = 5.0;

    /// <summary>
    /// Gets or sets the maximum duration in seconds.
    /// </summary>
    public double MaxDurationSeconds { get; set; } = 30.0;

    /// <summary>
    /// Gets or sets the temperature for sampling.
    /// Higher values = more random, lower = more deterministic.
    /// </summary>
    public double Temperature { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the top-k value for sampling.
    /// Only the top k tokens are considered.
    /// </summary>
    public int TopK { get; set; } = 250;

    /// <summary>
    /// Gets or sets the top-p (nucleus) value for sampling.
    /// </summary>
    public double TopP { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the classifier-free guidance scale.
    /// Higher values = stronger prompt following.
    /// </summary>
    public double GuidanceScale { get; set; } = 3.0;

    /// <summary>
    /// Gets or sets the number of audio channels (1=mono, 2=stereo).
    /// </summary>
    public int Channels { get; set; } = 1;

    /// <summary>
    /// Gets or sets the ONNX execution options.
    /// </summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    /// <summary>
    /// Gets or sets the text hidden dim.
    /// </summary>
    public int TextHiddenDim { get; set; }

    /// <summary>
    /// Gets or sets the lm hidden dim.
    /// </summary>
    public int LmHiddenDim { get; set; }

    /// <summary>
    /// Gets or sets the num lm layers.
    /// </summary>
    public int NumLmLayers { get; set; }

    /// <summary>
    /// Gets or sets the num heads.
    /// </summary>
    public int NumHeads { get; set; }

    /// <summary>
    /// Gets or sets the num codebooks.
    /// </summary>
    public int NumCodebooks { get; set; }

    /// <summary>
    /// Gets or sets the codebook size.
    /// </summary>
    public int CodebookSize { get; set; }

    /// <summary>
    /// Gets or sets the max text length.
    /// </summary>
    public int MaxTextLength { get; set; }

    /// <summary>
    /// Gets or sets the path of a pretrained EnCodec checkpoint (safetensors, audiocraft state-dict
    /// names) whose decoder and codebooks the native model decodes with; null keeps the seeded
    /// initialization. AudioGen decodes with a pretrained, frozen EnCodec (Kreuk et al. 2022, §3.1).
    /// </summary>
    /// <remarks>
    /// The released checkpoint (facebook/audiogen-medium, <c>compression_state_dict.bin</c>) is a
    /// PyTorch pickle whose tensors sit under <c>best_state</c>. Convert it to safetensors first and
    /// strip the <c>best_state.</c> prefix from every name, for example
    /// <c>safetensors.torch.save_file(torch.load(path)["best_state"], "encodec.safetensors")</c>.
    /// It matches <see cref="SampleRate"/> = 16000, <see cref="NumCodebooks"/> = 4 and
    /// <see cref="CodebookSize"/> = 2048; other settings change the tensor shapes and the load is
    /// rejected.
    /// </remarks>
    public string? EncodecWeightsPath { get; set; }

    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    public void Validate()
    {
        ValidateCore();
    }
}

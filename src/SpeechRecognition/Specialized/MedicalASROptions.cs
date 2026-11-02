using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.Specialized;

/// <summary>Options for Medical ASR: domain-specialized medical speech recognition.</summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the MedicalASR model. Default values follow the original paper's recommended settings for optimal speech recognition accuracy.</para>
/// </remarks>
public class MedicalASROptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public MedicalASROptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MedicalASROptions(MedicalASROptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        SampleRate = other.SampleRate;
        MaxAudioLengthSeconds = other.MaxAudioLengthSeconds;
        EncoderDim = other.EncoderDim;
        NumEncoderLayers = other.NumEncoderLayers;
        NumAttentionHeads = other.NumAttentionHeads;
        NumMels = other.NumMels;
        VocabSize = other.VocabSize;
        MaxTextLength = other.MaxTextLength;
        ModelPath = other.ModelPath;
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
        DropoutRate = other.DropoutRate;
        Language = other.Language;
        DecoderType = other.DecoderType;
        PyramidalReductions = other.PyramidalReductions;
        DecoderDim = other.DecoderDim;
        NumDecoderLayers = other.NumDecoderLayers;
    }

    /// <summary>
    /// Gets or sets which end-to-end decoder to build.
    /// </summary>
    /// <remarks>
    /// The paper compares CTC and LAS and reports that "the LAS was more resilient to noisy data and
    /// CTC required more data clean up", which is why LAS is the default for spontaneous
    /// doctor-patient conversation. Switching to CTC reproduces the paper's other arm rather than
    /// degrading the model.
    /// </remarks>
    /// <value>Defaults to <see cref="MedicalAsrDecoderType.ListenAttendSpell"/>.</value>
    public MedicalAsrDecoderType DecoderType { get; set; } = MedicalAsrDecoderType.ListenAttendSpell;

    /// <summary>
    /// Gets or sets the number of pyramidal reductions in the LAS "listen" stage.
    /// </summary>
    /// <remarks>
    /// Each pyramidal layer concatenates adjacent time steps, halving the sequence length. Without
    /// it the speller must attend over every acoustic frame of a multi-minute consultation, which is
    /// the cost LAS's pyramid exists to remove. Ignored when <see cref="DecoderType"/> is CTC, whose
    /// frame-synchronous head needs the full time resolution.
    /// </remarks>
    /// <value>Defaults to 3, giving an 8x reduction.</value>
    public int PyramidalReductions { get; set; } = 3;

    /// <summary>Gets or sets the speller (attention decoder) hidden width.</summary>
    /// <value>Defaults to 256.</value>
    public int DecoderDim { get; set; } = 256;

    /// <summary>Gets or sets the number of speller layers.</summary>
    /// <value>Defaults to 2.</value>
    public int NumDecoderLayers { get; set; } = 2;

    /// <summary>Gets or sets the sample rate, in Hz, that input audio is expected at.</summary>
    public int SampleRate { get; set; } = 16000;
    /// <summary>Gets or sets the longest clip, in seconds, of the reference configuration. It is informational: input is neither trimmed nor rejected by length.</summary>
    public int MaxAudioLengthSeconds { get; set; } = 30;
    /// <summary>Gets or sets the hidden width of the encoder.</summary>
    public int EncoderDim { get; set; } = 512;
    /// <summary>Gets or sets the number of encoder layers.</summary>
    public int NumEncoderLayers { get; set; } = 18;
    /// <summary>Gets or sets the number of heads in each attention layer.</summary>
    public int NumAttentionHeads { get; set; } = 8;
    /// <summary>Gets or sets the number of mel filterbank channels in the input features.</summary>
    public int NumMels { get; set; } = 80;
    /// <summary>Gets or sets the number of output tokens the model predicts over.</summary>
    public int VocabSize { get; set; } = 10000;
    /// <summary>Gets or sets the most tokens decoded for one utterance.</summary>
    public int MaxTextLength { get; set; } = 512;
    /// <summary>Gets or sets the path of the pretrained ONNX model. The ONNX constructor sets it from its modelPath argument; setting it here does not load a model.</summary>
    public string? ModelPath { get; set; }
    /// <summary>Gets or sets the ONNX Runtime settings used when the model is loaded from an ONNX file.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    /// <summary>Gets or sets the dropout probability of the native layers during training.</summary>
    public double DropoutRate { get; set; } = 0.1;
    /// <summary>Gets or sets the language code (for example "en") reported for a transcript when none is passed or detected.</summary>
    public string Language { get; set; } = "en";
}

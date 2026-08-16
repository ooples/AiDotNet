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

    public int SampleRate { get; set; } = 16000;
    public int MaxAudioLengthSeconds { get; set; } = 30;
    public int EncoderDim { get; set; } = 512;
    public int NumEncoderLayers { get; set; } = 18;
    public int NumAttentionHeads { get; set; } = 8;
    public int NumMels { get; set; } = 80;
    public int VocabSize { get; set; } = 10000;
    public int MaxTextLength { get; set; } = 512;
    public string? ModelPath { get; set; }
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    public double DropoutRate { get; set; } = 0.1;
    public string Language { get; set; } = "en";
}

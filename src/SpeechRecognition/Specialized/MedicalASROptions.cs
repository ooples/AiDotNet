using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.Specialized;

/// <summary>Options for <see cref="MedicalASR{T}"/>, Chiu et al. 2018's CTC and LAS recognizers for medical conversations.</summary>
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
        Bidirectional = other.Bidirectional;
        OutputDelay = other.OutputDelay;
        NumAttentionHeads = other.NumAttentionHeads;
        NumMels = other.NumMels;
        MaxFrequency = other.MaxFrequency;
        StackedFrames = other.StackedFrames;
        VocabSize = other.VocabSize;
        PhoneInventory = (string[])other.PhoneInventory.Clone();
        MaxTextLength = other.MaxTextLength;
        ModelPath = other.ModelPath;
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
        Language = other.Language;
        DecoderType = other.DecoderType;
        DecoderDim = other.DecoderDim;
        NumDecoderLayers = other.NumDecoderLayers;
        ScheduledSamplingProbability = other.ScheduledSamplingProbability;
        ScheduledSamplingRampSteps = other.ScheduledSamplingRampSteps;
        Seed = other.Seed;
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

    /// <summary>Gets or sets the LAS decoder's LSTM units, also the width of its grapheme embedding and attention.</summary>
    /// <value>Defaults to 768 (section 5.2: "the decoder has 3 layers of uni-directional LSTMs with 768 units").</value>
    public int DecoderDim { get; set; } = 768;

    /// <summary>Gets or sets the LAS decoder's LSTM layers.</summary>
    /// <value>Defaults to 3 (section 5.2).</value>
    public int NumDecoderLayers { get; set; } = 3;

    /// <summary>Gets or sets the highest probability with which scheduled sampling feeds the LAS decoder its own previous
    /// prediction instead of the ground truth.</summary>
    /// <value>Defaults to 0.3 (section 4.2: "linearly ramp up the probability of sampling from model's prediction up to
    /// 30%, and then stay at this sampling rate").</value>
    public double ScheduledSamplingProbability { get; set; } = 0.3;

    /// <summary>Gets or sets the training steps over which the sampling probability ramps linearly from 0.</summary>
    /// <value>Defaults to 100000; the paper does not state the ramp length.</value>
    public int ScheduledSamplingRampSteps { get; set; } = 100000;

    /// <summary>Gets or sets the audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>Gets or sets the longest audio, in seconds, a transcription reads.</summary>
    public int MaxAudioLengthSeconds { get; set; } = 30;

    /// <summary>Gets or sets the LSTM units of each encoder layer: per direction when <see cref="Bidirectional"/>, in total
    /// otherwise.</summary>
    /// <value>Defaults to the LAS encoder's 768 per direction (section 5.2); the paper's CTC models use 350 per direction or
    /// 700 unidirectional (section 5.1), see <see cref="ForCtc"/>.</value>
    public int EncoderDim { get; set; } = 768;

    /// <summary>Gets or sets the encoder LSTM layers.</summary>
    /// <value>Defaults to the LAS encoder's 6 (section 5.2); the CTC models have 5.</value>
    public int NumEncoderLayers { get; set; } = 6;

    /// <summary>Gets or sets whether the encoder LSTMs are bidirectional.</summary>
    /// <value>Defaults to true: the LAS encoder is bidirectional, and the bidirectional CTC model was the better one
    /// (20.1% against 23.5% WER, section 5.1).</value>
    public bool Bidirectional { get; set; } = true;

    /// <summary>Gets or sets the output delay, in input steps, of a unidirectional CTC model: its first outputs are ignored
    /// and its last input step repeated as many times (section 4.1).</summary>
    /// <value>Defaults to 10 (section 5.1). Unused by bidirectional encoders.</value>
    public int OutputDelay { get; set; } = 10;

    /// <summary>Gets or sets the heads of the LAS multi-headed attention.</summary>
    /// <value>Defaults to 4, the head count of the same group's LAS (Chiu et al. 2018, "State-of-the-art speech
    /// recognition with sequence-to-sequence models"); this paper does not state it. Must divide
    /// <see cref="DecoderDim"/>.</value>
    public int NumAttentionHeads { get; set; } = 4;

    /// <summary>Gets or sets the filterbank channels per 25 ms frame.</summary>
    /// <value>Defaults to 80 (section 4).</value>
    public int NumMels { get; set; } = 80;

    /// <summary>Gets or sets the highest frequency the filterbanks cover, in Hz.</summary>
    /// <value>Defaults to 3800 (section 4: "we ignore frequencies above 3800Hz").</value>
    public double MaxFrequency { get; set; } = 3800;

    /// <summary>Gets or sets how many 10 ms frames are stacked into one input step, which is also the stride.</summary>
    /// <value>Defaults to 3 (section 4: "input is 240 filterbanks every 30 ms").</value>
    public int StackedFrames { get; set; } = 3;

    /// <summary>Gets or sets the output units. LAS predicts graphemes: Unicode code points below this, with 0 as the
    /// boundary that starts and ends a transcript. CTC predicts context-dependent phones 1 to VocabSize - 1, with 0 the
    /// blank.</summary>
    /// <value>Defaults to 128, the ASCII graphemes; the paper's CTC models predict 8K CD phones plus the blank
    /// (8001, section 4.1).</value>
    public int VocabSize { get; set; } = 128;

    /// <summary>Gets or sets the names of the CTC model's phone units by index, used to write its transcript. Units past
    /// the list are written as their index.</summary>
    public string[] PhoneInventory { get; set; } = Array.Empty<string>();

    /// <summary>Gets or sets the most graphemes a LAS transcription emits.</summary>
    public int MaxTextLength { get; set; } = 512;

    /// <summary>Gets or sets the ONNX model path, for the ONNX constructor.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    /// <summary>The paper's bidirectional CTC model (section 5.1): five bidirectional LSTM layers of 350 units per direction
    /// over 8K context-dependent phones and the blank.</summary>
    public static MedicalASROptions ForCtc() => new()
    {
        DecoderType = MedicalAsrDecoderType.Ctc, NumEncoderLayers = 5, EncoderDim = 350, VocabSize = 8001,
    };

    internal MedicalAsrConfiguration ToConfiguration() => new()
    {
        DecoderType = DecoderType, SampleRate = SampleRate, NumFilterbanks = NumMels, MaxFrequency = MaxFrequency,
        StackedFrames = StackedFrames, EncoderLayers = NumEncoderLayers, EncoderUnits = EncoderDim, Bidirectional = Bidirectional,
        OutputDelay = OutputDelay, DecoderLayers = NumDecoderLayers, DecoderUnits = DecoderDim, AttentionHeads = NumAttentionHeads,
        VocabSize = VocabSize,
    };

    /// <summary>Gets or sets the transcript language.</summary>
    public string Language { get; set; } = "en";
}

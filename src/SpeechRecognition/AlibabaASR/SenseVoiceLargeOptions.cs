using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.AlibabaASR;

/// <summary>Options for SenseVoice-Large: the autoregressive encoder-decoder SenseVoice (FunAudioLLM, 2024).</summary>
/// <remarks>
/// <para><b>For Beginners:</b> Defaults are the Whisper-large geometry SenseVoice-Large follows: 128 mel bands over a
/// 30-second window, a 32-layer encoder and a 32-layer decoder of width 1,280 with 20 heads. The report gives 1,587M
/// parameters; its vocabulary is unpublished, so Whisper large-v3's 51,866 tokens (1,550M parameters) is the default.</para>
/// </remarks>
public class SenseVoiceLargeOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public SenseVoiceLargeOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SenseVoiceLargeOptions(SenseVoiceLargeOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        // Seed is inherited from ModelOptions and easy to miss; a copy without it builds a different model.
        Seed = other.Seed;
        SampleRate = other.SampleRate;
        NumMels = other.NumMels;
        EncoderDim = other.EncoderDim;
        NumEncoderLayers = other.NumEncoderLayers;
        NumDecoderLayers = other.NumDecoderLayers;
        NumAttentionHeads = other.NumAttentionHeads;
        VocabSize = other.VocabSize;
        MaxTextLength = other.MaxTextLength;
        ModelPath = other.ModelPath;
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
        Language = other.Language;
        PredictLanguage = other.PredictLanguage;
        PredictEmotion = other.PredictEmotion;
        PredictEvents = other.PredictEvents;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
        Vocabulary = other.Vocabulary is null ? null : (string[])other.Vocabulary.Clone();
    }

    public int SampleRate { get; set; } = 16000;
    /// <summary>Log-mel bands (128).</summary>
    public int NumMels { get; set; } = 128;
    /// <summary>The encoder's and the decoder's width (1,280); the feed-forward is four times wider.</summary>
    public int EncoderDim { get; set; } = 1280;
    /// <summary>Encoder layers (32).</summary>
    public int NumEncoderLayers { get; set; } = 32;
    /// <summary>Decoder layers (32).</summary>
    public int NumDecoderLayers { get; set; } = 32;
    public int NumAttentionHeads { get; set; } = 20;
    /// <summary>The vocabulary (51,866). Its last five tokens are the special ones: end of text, start of transcript, and
    /// the LID, SER and AED task tokens; the others are text, language, emotion and event tokens.</summary>
    public int VocabSize { get; set; } = 51866;
    /// <summary>The longest decoded sequence, task outputs included (at most the decoder's 448 positions).</summary>
    public int MaxTextLength { get; set; } = 224;
    public string? ModelPath { get; set; }
    public OnnxModelOptions OnnxOptions { get; set; } = new();
    public string Language { get; set; } = "auto";
    /// <summary>Whether the prompt carries the LID task token, so the decoder first predicts the language.</summary>
    public bool PredictLanguage { get; set; } = true;
    /// <summary>Whether the prompt carries the SER task token, so the decoder predicts the speech emotion.</summary>
    public bool PredictEmotion { get; set; }
    /// <summary>Whether the prompt carries the AED task token, so the decoder predicts audio events with timestamps.</summary>
    public bool PredictEvents { get; set; }
    /// <summary>The AdamW learning rate used when no optimizer is injected (1e-4; the report states no optimizer).</summary>
    public double LearningRate { get; set; } = 1e-4;
    /// <summary>The AdamW weight decay used when no optimizer is injected (0.01).</summary>
    public double WeightDecay { get; set; } = 0.01;
    /// <summary>Optional vocabulary mapping token IDs to text.</summary>
    public string[]? Vocabulary { get; set; }
}

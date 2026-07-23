using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.SpeechRecognition;

/// <summary>
/// Configuration options for the CTC Decoder speech recognition model.
/// </summary>
/// <remarks>
/// <para>
/// CTC (Connectionist Temporal Classification, Graves et al., 2006) is a training criterion
/// and decoding algorithm that allows sequence-to-sequence mapping without requiring exact
/// input-output alignment. With a greedy or beam-search decoder, CTC-trained models
/// directly output transcriptions from encoder features. CTC is used by wav2vec 2.0,
/// Conformer-CTC, DeepSpeech 2, and many production ASR systems.
/// </para>
/// <para>
/// <b>For Beginners:</b> CTC decoding solves a fundamental problem: audio frames don't
/// neatly align with letters or words. Some frames correspond to silence, some to the
/// middle of a vowel, and the model must figure out which frames correspond to which
/// characters. CTC introduces a "blank" token that the model outputs when nothing new
/// is being said, then the decoder collapses repeated characters and removes blanks.
///
/// Example of CTC decoding:
/// Raw output:  h h - e e - l - l - l - o o
/// After collapse: h e l l o  (where - is the blank token)
/// </para>
/// </remarks>
public class CTCDecoderOptions : ModelOptions
{
    #region Audio

    /// <summary>Gets or sets the expected audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>Gets or sets the maximum audio length in seconds.</summary>
    public int MaxAudioLengthSeconds { get; set; } = 30;

    #endregion

    #region Architecture

    /// <summary>Gets or sets the model variant ("small", "medium", "large").</summary>
    public string Variant { get; set; } = "medium";

    /// <summary>Gets or sets the encoder hidden dimension.</summary>
    public int EncoderDim { get; set; } = 512;

    /// <summary>Gets or sets the number of encoder layers.</summary>
    public int NumEncoderLayers { get; set; } = 6;

    /// <summary>Gets or sets the number of attention heads.</summary>
    public int NumAttentionHeads { get; set; } = 8;

    /// <summary>Gets or sets the feed-forward dimension.</summary>
    public int FeedForwardDim { get; set; } = 2048;

    /// <summary>Gets or sets the number of mel-spectrogram frequency bins.</summary>
    public int NumMels { get; set; } = 80;

    /// <summary>Gets or sets the vocabulary size.</summary>
    public int VocabSize { get; set; } = 5000;

    #endregion

    #region Decoding

    /// <summary>Gets or sets the CTC beam width for beam search decoding.</summary>
    /// <remarks>
    /// Beam width controls the trade-off between speed and accuracy.
    /// - 1 = greedy decoding (fastest, slightly less accurate)
    /// - 10-100 = beam search (better accuracy, slower)
    /// - Higher values give diminishing returns past ~20.
    /// </remarks>
    public int BeamWidth { get; set; } = 10;

    /// <summary>Gets or sets whether to use a language model for rescoring.</summary>
    public bool UseLM { get; set; }

    /// <summary>Gets or sets the language model weight for beam search.</summary>
    /// <remarks>
    /// Controls how much the language model influences decoding vs. the acoustic model.
    /// Typical values: 0.3 to 2.0.
    /// </remarks>
    public double LMWeight { get; set; } = 0.5;

    /// <summary>Gets or sets the word insertion penalty for beam search.</summary>
    /// <remarks>
    /// Positive values encourage more words (reduce under-generation).
    /// Negative values encourage fewer words (reduce over-generation).
    /// </remarks>
    public double WordInsertionPenalty { get; set; }

    /// <summary>Gets or sets the CTC vocabulary (characters or BPE tokens).</summary>
    public string[] Vocabulary { get; set; } = GetDefaultVocabulary();

    /// <summary>Gets or sets the default language code.</summary>
    public string Language { get; set; } = "en";

    /// <summary>Gets or sets the blank token index in the vocabulary.</summary>
    public int BlankTokenIndex { get; set; }

    private static string[] GetDefaultVocabulary()
    {
        return new[]
        {
            "<blank>", "<pad>", "<s>", "</s>", "<unk>", "|",
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
            "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
            "u", "v", "w", "x", "y", "z", "'", " "
        };
    }

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 3e-4;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.1;

    #endregion
}

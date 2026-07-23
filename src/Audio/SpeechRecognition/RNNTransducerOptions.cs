using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.SpeechRecognition;

/// <summary>
/// Configuration options for the RNN-Transducer (RNN-T) speech recognition model.
/// </summary>
/// <remarks>
/// <para>
/// RNN-Transducer (Graves, 2012; He et al., 2019) combines an audio encoder with a label
/// prediction network and a joint network to produce a streaming ASR system. Unlike CTC,
/// RNN-T can model output dependencies through its prediction network, achieving strong
/// results without an external language model.
/// </para>
/// <para>
/// <b>For Beginners:</b> RNN-T is a real-time speech recognizer ideal for live transcription.
/// Unlike batch models (like Whisper), it processes audio as it arrives, making it perfect for
/// live captioning and voice assistants. It has three parts: an encoder (listens), a predictor
/// (remembers what was said), and a joiner (combines both to output the next word).
/// </para>
/// </remarks>
public class RNNTransducerOptions : ModelOptions
{
    #region Audio Settings

    /// <summary>Gets or sets the audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>Gets or sets the number of mel spectrogram channels.</summary>
    public int NumMels { get; set; } = 80;

    /// <summary>Gets or sets the language code.</summary>
    public string Language { get; set; } = "en";

    #endregion

    #region Encoder

    /// <summary>Gets or sets the encoder hidden dimension.</summary>
    public int EncoderDim { get; set; } = 512;

    /// <summary>Gets or sets the number of encoder layers.</summary>
    public int NumEncoderLayers { get; set; } = 12;

    /// <summary>Gets or sets the number of encoder attention heads.</summary>
    public int NumEncoderHeads { get; set; } = 8;

    #endregion

    #region Prediction Network

    /// <summary>Gets or sets the prediction network hidden dimension.</summary>
    public int PredictionDim { get; set; } = 512;

    /// <summary>Gets or sets the number of prediction LSTM layers.</summary>
    public int NumPredictionLayers { get; set; } = 2;

    /// <summary>Gets or sets the embedding dimension for output tokens.</summary>
    public int EmbeddingDim { get; set; } = 256;

    #endregion

    #region Joint Network

    /// <summary>Gets or sets the joint network hidden dimension.</summary>
    public int JointDim { get; set; } = 512;

    /// <summary>Gets or sets the vocabulary size (subword tokens).</summary>
    public int VocabSize { get; set; } = 5000;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 1e-3;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.1;

    #endregion

    #region Decoding

    /// <summary>Gets or sets the model variant ("small", "medium", "large").</summary>
    public string Variant { get; set; } = "medium";

    /// <summary>Gets or sets the CTC/RNN-T vocabulary (characters or BPE tokens).</summary>
    public string[] Vocabulary { get; set; } = GetDefaultVocabulary();

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
}

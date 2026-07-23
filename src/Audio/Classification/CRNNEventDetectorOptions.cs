using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Classification;

/// <summary>
/// Configuration options for the CRNN (Convolutional Recurrent Neural Network) Sound Event Detector.
/// </summary>
/// <remarks>
/// <para>
/// CRNN for SED (Cakir et al., 2017) combines convolutional layers for spectral feature extraction
/// with recurrent layers (GRU/LSTM) for temporal modeling. It is the standard baseline architecture
/// for the DCASE Sound Event Detection challenge and achieves strong results on AudioSet, ESC-50,
/// and URBAN-SED benchmarks. The model processes mel spectrograms through CNN blocks, then models
/// temporal dependencies with bidirectional GRU layers, producing frame-level event probabilities.
/// </para>
/// <para>
/// <b>For Beginners:</b> CRNN is a classic and reliable approach to detecting sounds over time.
/// Think of it in two stages:
///
/// 1. <b>CNN stage</b>: Looks at the spectrogram like a picture, finding patterns like "this
///    frequency pattern looks like a dog bark" or "this shape looks like a piano note"
/// 2. <b>RNN stage</b>: Reads the patterns over time like reading a sentence, understanding
///    "the dog bark started, continued, and then stopped"
///
/// This combination makes CRNN good at both identifying what sounds are present AND when they
/// happen. It's simpler than Transformer-based models but still very effective.
/// </para>
/// </remarks>
public class CRNNEventDetectorOptions : ModelOptions
{
    #region Audio Preprocessing

    /// <summary>Gets or sets the audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>Gets or sets the FFT window size.</summary>
    public int FftSize { get; set; } = 1024;

    /// <summary>Gets or sets the hop length between FFT frames.</summary>
    public int HopLength { get; set; } = 160;

    /// <summary>Gets or sets the number of mel filterbank channels.</summary>
    public int NumMels { get; set; } = 64;

    /// <summary>Gets or sets the minimum frequency for mel filterbank.</summary>
    public int FMin { get; set; } = 50;

    /// <summary>Gets or sets the maximum frequency for mel filterbank.</summary>
    public int FMax { get; set; } = 8000;

    #endregion

    #region CNN Architecture

    /// <summary>Gets or sets the model variant ("small", "base", "large").</summary>
    public string Variant { get; set; } = "base";

    /// <summary>Gets or sets the number of CNN channels per block.</summary>
    /// <remarks>
    /// <para>Each block doubles the channels. Default [64, 128, 256] means 3 CNN blocks
    /// with 64, 128, and 256 filters respectively.</para>
    /// </remarks>
    public int[] CNNChannels { get; set; } = [64, 128, 256];

    /// <summary>Gets or sets the CNN kernel size.</summary>
    public int CNNKernelSize { get; set; } = 3;

    /// <summary>Gets or sets the CNN pooling size for frequency dimension.</summary>
    public int PoolSize { get; set; } = 2;

    #endregion

    #region RNN Architecture

    /// <summary>Gets or sets the RNN hidden size.</summary>
    /// <remarks>
    /// <para>Size of the GRU/LSTM hidden state. Larger values capture more temporal context
    /// but increase computation. 128-256 is typical for SED tasks.</para>
    /// </remarks>
    public int RNNHiddenSize { get; set; } = 128;

    /// <summary>Gets or sets the number of RNN layers.</summary>
    public int NumRNNLayers { get; set; } = 2;

    /// <summary>Gets or sets whether to use bidirectional RNN.</summary>
    /// <remarks>
    /// <para>Bidirectional RNN processes audio forwards and backwards, capturing both
    /// past and future context. This improves accuracy but doubles computation.</para>
    /// </remarks>
    public bool Bidirectional { get; set; } = true;

    #endregion

    #region Detection

    /// <summary>Gets or sets the confidence threshold for event detection.</summary>
    public double Threshold { get; set; } = 0.3;

    /// <summary>Gets or sets the window size in seconds for detection.</summary>
    public double DetectionWindowSize { get; set; } = 10.0;

    /// <summary>Gets or sets the window overlap ratio (0-1).</summary>
    public double WindowOverlap { get; set; } = 0.5;

    /// <summary>Gets or sets custom event labels. If null, uses AudioSet-527 labels.</summary>
    public string[]? CustomLabels { get; set; }

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to a pre-trained ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 1e-3;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.2;

    /// <summary>Gets or sets the label smoothing factor.</summary>
    public double LabelSmoothing { get; set; } = 0.0;

    #endregion
}

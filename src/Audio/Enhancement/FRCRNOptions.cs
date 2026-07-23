using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Enhancement;

/// <summary>
/// Configuration options for the FRCRN (Frequency Recurrence CRN) model.
/// </summary>
/// <remarks>
/// <para>
/// FRCRN (Zhao et al., ICASSP 2022, Alibaba DAMO Academy) uses frequency recurrence to
/// model spectral correlations and complex spectral mapping. It won 1st place in the
/// ICASSP 2022 DNS Challenge non-personalized track with superior noise suppression while
/// preserving speech quality.
/// </para>
/// <para>
/// <b>For Beginners:</b> FRCRN uses a clever trick: it processes audio frequencies in
/// sequence (low to high) like reading a book, so each frequency knows about its neighbors.
/// This helps it distinguish speech from noise because speech frequencies are correlated
/// (they appear together), while noise frequencies are more random.
/// </para>
/// </remarks>
public class FRCRNOptions : ModelOptions
{
    #region Audio

    /// <summary>Gets or sets the expected audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>Gets or sets the FFT size.</summary>
    public int FFTSize { get; set; } = 512;

    /// <summary>Gets or sets the hop length.</summary>
    public int HopLength { get; set; } = 128;

    #endregion

    #region Architecture

    /// <summary>Gets or sets the model variant ("base" or "large").</summary>
    public string Variant { get; set; } = "base";

    /// <summary>Gets or sets the encoder channel dimension.</summary>
    public int EncoderChannels { get; set; } = 64;

    /// <summary>Gets or sets the number of encoder-decoder stages.</summary>
    public int NumStages { get; set; } = 5;

    /// <summary>Gets or sets the LSTM hidden size for frequency recurrence.</summary>
    public int LstmHiddenSize { get; set; } = 256;

    /// <summary>Gets or sets the number of frequency bins.</summary>
    public int NumFreqBins { get; set; } = 257;

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
    public double DropoutRate { get; set; } = 0.05;

    #endregion
}

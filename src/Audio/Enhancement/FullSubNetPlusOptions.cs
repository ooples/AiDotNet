using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Enhancement;

/// <summary>
/// Configuration options for the FullSubNet+ (Full-Band and Sub-Band Fusion Network Plus) model.
/// </summary>
/// <remarks>
/// <para>
/// FullSubNet+ (Chen et al., ICASSP 2022) improves upon FullSubNet by using a channel-attention-based
/// full-band model and redesigned sub-band inputs. It achieves state-of-the-art speech enhancement
/// on the DNS Challenge dataset with PESQ 3.25 and STOI 0.96 at 16 kHz.
/// </para>
/// <para>
/// <b>For Beginners:</b> FullSubNet+ is a two-part neural network that cleans up noisy audio:
/// <list type="number">
/// <item>The "full-band" part looks at all frequencies together to understand the overall pattern</item>
/// <item>The "sub-band" part focuses on small frequency ranges for fine detail</item>
/// <item>The two parts share information to get the best of both worlds</item>
/// </list>
/// Think of it like cleaning a painting: one person looks at the whole picture for context,
/// while another works on details, and they coordinate together.
/// </para>
/// <para>
/// <b>References:</b>
/// <list type="bullet">
/// <item>Paper: "FullSubNet+: Channel Attention FullSubNet with Complex Spectrograms" (Chen et al., ICASSP 2022)</item>
/// <item>Repository: https://github.com/hit-thusz-RookieCJ/FullSubNet-plus</item>
/// </list>
/// </para>
/// </remarks>
public class FullSubNetPlusOptions : ModelOptions
{
    #region Audio Preprocessing

    /// <summary>
    /// Gets or sets the audio sample rate in Hz.
    /// </summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>
    /// Gets or sets the FFT window size.
    /// </summary>
    public int FftSize { get; set; } = 512;

    /// <summary>
    /// Gets or sets the hop length between frames.
    /// </summary>
    public int HopLength { get; set; } = 256;

    /// <summary>
    /// Gets or sets the number of frequency bins (FftSize / 2 + 1).
    /// </summary>
    public int NumFreqBins { get; set; } = 257;

    #endregion

    #region Full-Band Model

    /// <summary>
    /// Gets or sets the number of full-band LSTM layers.
    /// </summary>
    /// <remarks>
    /// <para>The full-band model uses LSTM layers with channel attention to capture
    /// spectral patterns across all frequencies.</para>
    /// </remarks>
    public int FullBandLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the hidden size for full-band LSTM.
    /// </summary>
    public int FullBandHiddenSize { get; set; } = 512;

    /// <summary>
    /// Gets or sets whether to use channel attention in the full-band model.
    /// </summary>
    /// <remarks>
    /// <para>Channel attention helps the model focus on the most informative frequency channels.</para>
    /// </remarks>
    public bool UseChannelAttention { get; set; } = true;

    #endregion

    #region Sub-Band Model

    /// <summary>
    /// Gets or sets the number of sub-band LSTM layers.
    /// </summary>
    public int SubBandLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the hidden size for sub-band LSTM.
    /// </summary>
    public int SubBandHiddenSize { get; set; } = 384;

    /// <summary>
    /// Gets or sets the sub-band bandwidth (number of frequency bins per sub-band).
    /// </summary>
    public int SubBandWidth { get; set; } = 3;

    /// <summary>
    /// Gets or sets the number of neighboring sub-bands for context.
    /// </summary>
    public int NumNeighborSubBands { get; set; } = 15;

    #endregion

    #region Enhancement

    /// <summary>
    /// Gets or sets the enhancement strength (0.0 = no enhancement, 1.0 = maximum).
    /// </summary>
    public double EnhancementStrength { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the dropout rate.
    /// </summary>
    public double DropoutRate { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets whether to use complex spectrograms (magnitude + phase).
    /// </summary>
    /// <remarks>
    /// <para>Using complex spectrograms allows the model to enhance both magnitude and phase,
    /// leading to better perceptual quality.</para>
    /// </remarks>
    public bool UseComplexSpectrogram { get; set; } = true;

    #endregion

    #region Model Loading

    /// <summary>
    /// Gets or sets the path to a pre-trained ONNX model file.
    /// </summary>
    public string? ModelPath { get; set; }

    /// <summary>
    /// Gets or sets ONNX runtime options.
    /// </summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>
    /// Gets or sets the initial learning rate.
    /// </summary>
    public double LearningRate { get; set; } = 1e-3;

    /// <summary>
    /// Gets or sets the weight decay for regularization.
    /// </summary>
    public double WeightDecay { get; set; } = 0.0;

    #endregion
}

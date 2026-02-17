using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Fingerprinting;

/// <summary>
/// Configuration options for the Neural Audio Fingerprint (NeuralFP) model.
/// </summary>
/// <remarks>
/// <para>
/// NeuralFP (Chang et al., 2021) uses a neural network to learn compact audio fingerprints
/// for large-scale audio retrieval. It generates fixed-length embeddings from mel spectrograms
/// that are robust to noise, compression, and time-stretching. The model uses contrastive
/// learning to ensure similar audio produces similar fingerprints.
/// </para>
/// <para>
/// <b>For Beginners:</b> NeuralFP creates "audio IDs" using AI instead of traditional signal
/// processing. It converts a short audio clip into a small vector of numbers (like a barcode).
/// Two recordings of the same song will produce very similar vectors, even if one is noisy
/// or compressed. This enables Shazam-like audio identification.
/// </para>
/// </remarks>
public class NeuralFPOptions : ModelOptions
{
    #region Audio Preprocessing

    /// <summary>
    /// Gets or sets the expected audio sample rate in Hz.
    /// </summary>
    public int SampleRate { get; set; } = 8000;

    /// <summary>
    /// Gets or sets the number of mel filterbank channels.
    /// </summary>
    public int NumMels { get; set; } = 256;

    /// <summary>
    /// Gets or sets the FFT window size in samples.
    /// </summary>
    public int FftSize { get; set; } = 1024;

    /// <summary>
    /// Gets or sets the hop length between frames in samples.
    /// </summary>
    public int HopLength { get; set; } = 256;

    /// <summary>
    /// Gets or sets the segment duration in seconds for fingerprint extraction.
    /// </summary>
    public double SegmentDurationSec { get; set; } = 1.0;

    #endregion

    #region Model Architecture

    /// <summary>
    /// Gets or sets the fingerprint embedding dimension.
    /// </summary>
    public int EmbeddingDim { get; set; } = 128;

    /// <summary>
    /// Gets or sets the number of convolutional blocks in the encoder.
    /// </summary>
    public int NumConvBlocks { get; set; } = 4;

    /// <summary>
    /// Gets or sets the base filter count for convolutional layers.
    /// </summary>
    public int BaseFilters { get; set; } = 32;

    /// <summary>
    /// Gets or sets the temperature for contrastive loss during training.
    /// </summary>
    public double Temperature { get; set; } = 0.05;

    #endregion

    #region Matching

    /// <summary>
    /// Gets or sets the cosine similarity threshold for considering a fingerprint match.
    /// </summary>
    public double MatchThreshold { get; set; } = 0.7;

    #endregion

    #region Model Loading

    /// <summary>
    /// Gets or sets the path to the ONNX model file.
    /// </summary>
    public string? ModelPath { get; set; }

    /// <summary>
    /// Gets or sets the ONNX runtime options.
    /// </summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>
    /// Gets or sets the learning rate for training.
    /// </summary>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>
    /// Gets or sets the dropout rate.
    /// </summary>
    public double DropoutRate { get; set; } = 0.1;

    #endregion
}

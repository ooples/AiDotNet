using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Speaker;

/// <summary>
/// Configuration options for the pyannote 3.x speaker diarization model.
/// </summary>
/// <remarks>
/// <para>
/// pyannote.audio 3.x (Plaquet &amp; Bredin, ASRU 2023) is a state-of-the-art speaker
/// diarization pipeline using end-to-end neural segmentation with PyanNet architecture.
/// It segments audio into speaker turns and supports overlapping speech detection.
/// Achieves 11.2% DER on AMI Mix-Headset benchmark.
/// </para>
/// <para>
/// <b>For Beginners:</b> pyannote is a system that figures out "who spoke when" in a
/// recording with multiple speakers. It splits audio into segments, assigns each to a
/// speaker, and can even detect when two people talk at the same time. It's widely used
/// for meeting transcription, podcast processing, and call analytics.
/// </para>
/// </remarks>
public class PyAnnoteOptions : ModelOptions
{
    #region Audio Preprocessing

    /// <summary>
    /// Gets or sets the expected audio sample rate in Hz.
    /// </summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>
    /// Gets or sets the number of mel filterbank channels.
    /// </summary>
    public int NumMels { get; set; } = 80;

    /// <summary>
    /// Gets or sets the FFT window size in samples.
    /// </summary>
    public int FftSize { get; set; } = 512;

    /// <summary>
    /// Gets or sets the hop length between frames in samples.
    /// </summary>
    public int HopLength { get; set; } = 160;

    #endregion

    #region Segmentation Model (PyanNet)

    /// <summary>
    /// Gets or sets the number of SincNet filters in the first layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// SincNet uses parametric sinc-based filters that learn directly from raw audio.
    /// </para>
    /// </remarks>
    public int SincNetFilters { get; set; } = 80;

    /// <summary>
    /// Gets or sets the LSTM hidden size for the segmentation model.
    /// </summary>
    public int LSTMHiddenSize { get; set; } = 128;

    /// <summary>
    /// Gets or sets the number of LSTM layers.
    /// </summary>
    public int NumLSTMLayers { get; set; } = 4;

    /// <summary>
    /// Gets or sets the linear layer hidden dimension after LSTM.
    /// </summary>
    public int LinearDim { get; set; } = 128;

    /// <summary>
    /// Gets or sets the maximum number of speakers per chunk.
    /// </summary>
    public int MaxSpeakersPerChunk { get; set; } = 3;

    #endregion

    #region Embedding Model

    /// <summary>
    /// Gets or sets the embedding dimension for speaker embeddings.
    /// </summary>
    public int EmbeddingDim { get; set; } = 512;

    /// <summary>
    /// Gets or sets the path to a separate embedding model.
    /// </summary>
    public string? EmbeddingModelPath { get; set; }

    #endregion

    #region Diarization Pipeline

    /// <summary>
    /// Gets or sets the segmentation chunk duration in seconds.
    /// </summary>
    public double ChunkDurationSeconds { get; set; } = 5.0;

    /// <summary>
    /// Gets or sets the step between consecutive chunks in seconds.
    /// </summary>
    public double ChunkStepSeconds { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the clustering threshold for speaker assignment.
    /// </summary>
    public double ClusteringThreshold { get; set; } = 0.7;

    /// <summary>
    /// Gets or sets the minimum segment duration in seconds.
    /// </summary>
    public double MinSegmentDuration { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets whether overlapping speech detection is enabled.
    /// </summary>
    public bool EnableOverlapDetection { get; set; } = true;

    /// <summary>
    /// Gets or sets the maximum number of speakers (null for auto-detection).
    /// </summary>
    public int? MaxSpeakers { get; set; }

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
    public double LearningRate { get; set; } = 1e-3;

    /// <summary>
    /// Gets or sets the weight decay for regularization.
    /// </summary>
    public double WeightDecay { get; set; } = 1e-5;

    /// <summary>
    /// Gets or sets the dropout rate.
    /// </summary>
    public double DropoutRate { get; set; } = 0.1;

    #endregion
}

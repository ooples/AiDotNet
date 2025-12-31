namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for speaker embedding extraction models (d-vector/x-vector extraction).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Speaker embedding extractors convert voice audio into fixed-length vectors that
/// capture the unique characteristics of a speaker's voice. These embeddings enable
/// speaker verification, identification, and diarization tasks.
/// </para>
/// <para>
/// <b>For Beginners:</b> Speaker embeddings are like a "voiceprint" - a compact
/// representation of what makes someone's voice unique.
///
/// How speaker embeddings work:
/// 1. Audio of someone speaking is fed into the model
/// 2. The model outputs a fixed-size vector (e.g., 256 or 512 numbers)
/// 3. This vector captures voice characteristics (pitch, timbre, accent, etc.)
/// 4. Vectors from the same speaker are similar; different speakers are different
///
/// Common use cases:
/// - Voice authentication ("Is this person who they claim to be?")
/// - Speaker identification ("Who is speaking?")
/// - Voice cloning (TTS with specific voice)
/// - Meeting transcription (separating speakers)
///
/// Key concepts:
/// - d-vector: Early embedding approach using DNN
/// - x-vector: Modern approach using TDNN with statistics pooling
/// - ECAPA-TDNN: State-of-the-art speaker embedding model
/// </para>
/// <para>
/// This interface extends <see cref="IFullModel{T, TInput, TOutput}"/> for Tensor-based audio processing.
/// </para>
/// </remarks>
public interface ISpeakerEmbeddingExtractor<T> : IFullModel<T, Tensor<T>, Tensor<T>>
{
    /// <summary>
    /// Gets the expected sample rate for input audio.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Typically 16000 Hz for speaker recognition models.
    /// </para>
    /// </remarks>
    int SampleRate { get; }

    /// <summary>
    /// Gets the dimension of output speaker embeddings.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Common values: 192, 256, or 512. Higher dimensions may capture more nuance
    /// but require more storage and computation.
    /// </para>
    /// </remarks>
    int EmbeddingDimension { get; }

    /// <summary>
    /// Gets the minimum audio duration required for reliable embedding extraction.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Very short audio clips may not contain enough voice
    /// information for accurate speaker representation. This property tells you
    /// the minimum length needed for reliable results.
    /// </para>
    /// </remarks>
    double MinimumDurationSeconds { get; }

    /// <summary>
    /// Gets whether this model is running in ONNX inference mode.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When true, the model uses pre-trained ONNX weights for inference.
    /// When false, the model can be trained from scratch using the neural network infrastructure.
    /// </para>
    /// </remarks>
    bool IsOnnxMode { get; }

    /// <summary>
    /// Extracts speaker embedding from audio.
    /// </summary>
    /// <param name="audio">Audio waveform tensor [samples] or [batch, samples].</param>
    /// <returns>Speaker embedding tensor [embedding_dim] or [batch, embedding_dim].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main method for extracting a voiceprint.
    /// - Pass in audio of someone speaking
    /// - Get back a compact vector representing their voice
    /// </para>
    /// </remarks>
    Tensor<T> ExtractEmbedding(Tensor<T> audio);

    /// <summary>
    /// Extracts speaker embedding from audio asynchronously.
    /// </summary>
    /// <param name="audio">Audio waveform tensor [samples] or [batch, samples].</param>
    /// <param name="cancellationToken">Cancellation token for async operation.</param>
    /// <returns>Speaker embedding tensor [embedding_dim] or [batch, embedding_dim].</returns>
    Task<Tensor<T>> ExtractEmbeddingAsync(Tensor<T> audio, CancellationToken cancellationToken = default);

    /// <summary>
    /// Extracts embeddings from multiple audio segments.
    /// </summary>
    /// <param name="audioSegments">List of audio waveform tensors.</param>
    /// <returns>List of speaker embedding tensors.</returns>
    /// <remarks>
    /// <para>
    /// Useful for processing multiple utterances from the same recording or
    /// comparing embeddings across different audio files.
    /// </para>
    /// </remarks>
    IReadOnlyList<Tensor<T>> ExtractEmbeddings(IReadOnlyList<Tensor<T>> audioSegments);

    /// <summary>
    /// Computes similarity between two speaker embeddings.
    /// </summary>
    /// <param name="embedding1">First speaker embedding.</param>
    /// <param name="embedding2">Second speaker embedding.</param>
    /// <returns>Similarity score, typically cosine similarity (0 to 1).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This tells you how similar two voiceprints are.
    /// - Score close to 1.0: Likely same speaker
    /// - Score close to 0.0: Likely different speakers
    /// </para>
    /// </remarks>
    T ComputeSimilarity(Tensor<T> embedding1, Tensor<T> embedding2);

    /// <summary>
    /// Aggregates multiple embeddings into a single representative embedding.
    /// </summary>
    /// <param name="embeddings">Collection of embeddings from the same speaker.</param>
    /// <returns>Aggregated embedding representing the speaker.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> If you have multiple recordings of the same person,
    /// this combines them into one stronger voiceprint. More samples = better accuracy.
    /// </para>
    /// </remarks>
    Tensor<T> AggregateEmbeddings(IReadOnlyList<Tensor<T>> embeddings);

    /// <summary>
    /// Normalizes an embedding for comparison (typically L2 normalization).
    /// </summary>
    /// <param name="embedding">The embedding to normalize.</param>
    /// <returns>Normalized embedding with unit length.</returns>
    Tensor<T> NormalizeEmbedding(Tensor<T> embedding);
}

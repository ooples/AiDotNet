namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for self-supervised audio foundation models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Audio foundation models learn general-purpose audio representations through self-supervised
/// pre-training on large unlabeled datasets. These representations can be fine-tuned for
/// downstream tasks like speech recognition, speaker verification, and emotion detection.
/// </para>
/// <para><b>For Beginners:</b> A foundation model is like a student who has read millions of
/// books but hasn't been told what to look for. It develops a deep understanding of language
/// (or in this case, audio). When you need it for a specific task (like recognizing emotions),
/// you just teach it the final step - it already understands the audio.
///
/// These models provide embeddings (numerical representations) that capture:
/// - Phonetic content (what sounds are being made)
/// - Speaker characteristics (who is speaking)
/// - Prosody and emotion (how they're speaking)
/// - Acoustic environment (where the recording was made)
///
/// Examples: HuBERT, wav2vec 2.0, WavLM, data2vec
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("AudioFoundationModel")]
public interface IAudioFoundationModel<T>
{
    /// <summary>
    /// Gets the sample rate this model operates at.
    /// </summary>
    int SampleRate { get; }

    /// <summary>
    /// Gets the embedding dimension of the model.
    /// </summary>
    int EmbeddingDimension { get; }

    /// <summary>
    /// Gets the number of transformer layers in the model.
    /// </summary>
    int NumLayers { get; }

    /// <summary>
    /// Extracts the final-layer embeddings from audio.
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <returns>Embedding tensor from the last transformer layer.</returns>
    Tensor<T> ExtractEmbeddings(Tensor<T> audio);

    /// <summary>
    /// Extracts embeddings asynchronously.
    /// </summary>
    Task<Tensor<T>> ExtractEmbeddingsAsync(Tensor<T> audio, CancellationToken cancellationToken = default);

    /// <summary>
    /// Extracts features from a specific layer.
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <param name="layerIndex">Which transformer layer to extract from (0=first, -1=last).</param>
    /// <returns>Feature tensor from the specified layer.</returns>
    Tensor<T> ExtractLayerFeatures(Tensor<T> audio, int layerIndex = -1);

    /// <summary>
    /// Extracts weighted combination of features from all layers.
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <param name="layerWeights">Weights for each layer (null = uniform).</param>
    /// <returns>Weighted feature tensor.</returns>
    Tensor<T> ExtractWeightedFeatures(Tensor<T> audio, T[]? layerWeights = null);
}

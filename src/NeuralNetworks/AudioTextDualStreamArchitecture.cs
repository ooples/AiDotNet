namespace AiDotNet.NeuralNetworks;

/// <summary>
/// A neural network architecture for audio + text two-stream models
/// (CLAP-family encoders) that hosts a separate audio encoder and text encoder.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Concrete audio-language specialisation of <see cref="DualEncoderArchitecture{T}"/>.
/// The abstract base owns the two encoder stacks under modality-neutral names
/// (<c>EncoderALayers</c> / <c>EncoderBLayers</c>); this subclass exposes them
/// under their semantic aliases <c>AudioLayers</c> / <c>TextLayers</c> so
/// CLAP-family code reads naturally.
/// </para>
/// <para>
/// CLAP (Wu et al. 2023) and similar audio-language pretraining models train
/// two parallel encoders: an audio side (HTSAT / PANNs / Wav2Vec / etc.) and a
/// text side (RoBERTa / BERT / etc.). They only meet at the contrastive
/// objective, so each side gets its own customisable layer stack.
/// </para>
/// </remarks>
public class AudioTextDualStreamArchitecture<T> : DualEncoderArchitecture<T>
{
    /// <summary>
    /// Gets the layer stack for the audio encoder. Alias for
    /// <see cref="DualEncoderArchitecture{T}.EncoderALayers"/>. Read-only view
    /// so callers can inspect the encoder without mutating internals.
    /// </summary>
    public IReadOnlyList<ILayer<T>> AudioLayers => EncoderALayers;

    /// <summary>
    /// Gets the layer stack for the text encoder. Alias for
    /// <see cref="DualEncoderArchitecture{T}.EncoderBLayers"/>. Read-only view
    /// (see <see cref="AudioLayers"/> for rationale).
    /// </summary>
    public IReadOnlyList<ILayer<T>> TextLayers => EncoderBLayers;

    /// <summary>
    /// Initializes a new audio-text dual-stream architecture with explicit
    /// audio and text encoder layer stacks.
    /// </summary>
    public AudioTextDualStreamArchitecture(
        IEnumerable<ILayer<T>> audioLayers,
        IEnumerable<ILayer<T>> textLayers,
        InputType inputType = InputType.OneDimensional,
        NeuralNetworkTaskType taskType = NeuralNetworkTaskType.NaturalLanguageProcessing,
        NetworkComplexity complexity = NetworkComplexity.Medium,
        int inputSize = 0,
        int outputSize = 0,
        int audioEmbeddingDim = 0,
        int textEmbeddingDim = 0,
        bool shouldReturnFullSequence = false)
        : base(
            encoderALayers: audioLayers,
            encoderBLayers: textLayers,
            inputType: inputType,
            taskType: taskType,
            complexity: complexity,
            inputSize: inputSize,
            outputSize: outputSize,
            shouldReturnFullSequence: shouldReturnFullSequence,
            imageEmbeddingDim: audioEmbeddingDim,
            textEmbeddingDim: textEmbeddingDim)
    {
    }
}

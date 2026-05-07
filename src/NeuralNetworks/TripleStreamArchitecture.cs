namespace AiDotNet.NeuralNetworks;

/// <summary>
/// A neural network architecture for three-stream models (e.g., perceiver/abstractor/resampler-style
/// generative VLMs and cross-modality fusion encoders) that hosts a vision encoder, an auxiliary
/// stream (perceiver / abstractor / resampler / cross-modality fusion), and a text or decoder stream.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Models like IDEFICS2, MPLUGOwl3, Qwen3VL, BridgeTower, LXMERT, and METER need three independent
/// layer stacks. The flat <see cref="NeuralNetworkArchitecture{T}.Layers"/> property cannot describe
/// three graphs, so callers wanting custom layers for all streams should use this type. The
/// <c>VisionLayers</c> populates the model's primary <c>Layers</c> list, while <c>AuxiliaryLayers</c>
/// and <c>TextOrDecoderLayers</c> populate the model-specific auxiliary lists (the perceiver /
/// abstractor / resampler / cross-modality stack and the language decoder / text encoder
/// respectively).
/// </para>
/// <para><b>For Beginners:</b> Bigger vision-language models have three pieces — a vision tower, an
/// adapter or "fusion" tower in the middle, and a text or language-decoder tower — that pass data
/// to each other in sequence. If you want to swap in custom layers for all three, this architecture
/// lets you describe each piece independently instead of jamming them into one list (which drops the
/// last two pieces).</para>
/// </remarks>
public class TripleStreamArchitecture<T> : NeuralNetworkArchitecture<T>
{
    /// <summary>
    /// Gets the layer stack for the vision encoder. Populates the model's primary <c>Layers</c> list.
    /// </summary>
    public List<ILayer<T>> VisionLayers { get; } = new List<ILayer<T>>();

    /// <summary>
    /// Gets the auxiliary stream layer stack (perceiver / abstractor / resampler /
    /// cross-modality fusion / bridge depending on model).
    /// </summary>
    public List<ILayer<T>> AuxiliaryLayers { get; } = new List<ILayer<T>>();

    /// <summary>
    /// Gets the layer stack for the text encoder or language decoder, depending on model.
    /// </summary>
    public List<ILayer<T>> TextOrDecoderLayers { get; } = new List<ILayer<T>>();

    /// <summary>
    /// Initializes a new triple-stream architecture with explicit vision, auxiliary, and text/decoder layers.
    /// </summary>
    public TripleStreamArchitecture(
        IEnumerable<ILayer<T>> visionLayers,
        IEnumerable<ILayer<T>> auxiliaryLayers,
        IEnumerable<ILayer<T>> textOrDecoderLayers,
        InputType inputType = InputType.ThreeDimensional,
        NeuralNetworkTaskType taskType = NeuralNetworkTaskType.TextGeneration,
        NetworkComplexity complexity = NetworkComplexity.Medium,
        int inputSize = 0,
        int inputHeight = 0,
        int inputWidth = 0,
        int inputDepth = 1,
        int outputSize = 0,
        bool shouldReturnFullSequence = false,
        int imageEmbeddingDim = 0,
        int textEmbeddingDim = 0,
        int inputFrames = 0)
        : base(
            inputType: inputType,
            taskType: taskType,
            complexity: complexity,
            inputSize: inputSize,
            inputHeight: inputHeight,
            inputWidth: inputWidth,
            inputDepth: inputDepth,
            outputSize: outputSize,
            layers: null,
            shouldReturnFullSequence: shouldReturnFullSequence,
            imageEmbeddingDim: imageEmbeddingDim,
            textEmbeddingDim: textEmbeddingDim,
            inputFrames: inputFrames)
    {
        if (visionLayers is null) throw new ArgumentNullException(nameof(visionLayers));
        if (auxiliaryLayers is null) throw new ArgumentNullException(nameof(auxiliaryLayers));
        if (textOrDecoderLayers is null) throw new ArgumentNullException(nameof(textOrDecoderLayers));
        VisionLayers.AddRange(visionLayers);
        AuxiliaryLayers.AddRange(auxiliaryLayers);
        TextOrDecoderLayers.AddRange(textOrDecoderLayers);
    }
}

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Modality-agnostic base for two-encoder neural network architectures.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Dual-encoder networks (CLIP, ALIGN, CLAP, ImageBind, …) train two parallel
/// encoders that produce embeddings into a shared space for contrastive
/// learning. The flat <see cref="NeuralNetworkArchitecture{T}.Layers"/> list
/// describes only one graph, so when callers want to customise both encoders
/// they must reach for a richer architecture descriptor — this abstract base
/// holds the two layer stacks under modality-neutral names so consumers can
/// program against the generic dual-encoder shape, while concrete subclasses
/// expose semantically-named aliases (<c>VisionLayers</c>/<c>TextLayers</c>,
/// <c>AudioLayers</c>/<c>TextLayers</c>, etc.) for the call-site clarity each
/// modality pair deserves.
/// </para>
/// <para>
/// SOLID rationale:
/// <list type="bullet">
/// <item><description><b>SRP</b>: each concrete subclass describes exactly one
/// modality pairing.</description></item>
/// <item><description><b>OCP</b>: new modality pairs add new subclasses without
/// touching existing CLIP-family or audio-text consumers.</description></item>
/// <item><description><b>LSP</b>: tape-training and parameter-iteration code
/// that walks <c>EncoderALayers</c> + <c>EncoderBLayers</c> works against any
/// subclass.</description></item>
/// <item><description><b>ISP</b>: modality-specific aliases let consumers
/// depend only on the names that apply to them — vision-language models never
/// see <c>AudioLayers</c>.</description></item>
/// </list>
/// </para>
/// </remarks>
public abstract class DualEncoderArchitecture<T> : NeuralNetworkArchitecture<T>
{
    /// <summary>
    /// Gets the layer stack for the first encoder stream. Models that consume
    /// this architecture must explicitly read <c>EncoderALayers</c> at
    /// initialisation time — the base <see cref="NeuralNetworkArchitecture{T}.Layers"/>
    /// list is intentionally left empty (we pass <c>layers: null</c> to the
    /// base constructor) so the two encoder streams stay distinct and don't
    /// silently flatten into the single primary list.
    /// </summary>
    public List<ILayer<T>> EncoderALayers { get; } = new List<ILayer<T>>();

    /// <summary>
    /// Gets the layer stack for the second encoder stream. Same contract as
    /// <see cref="EncoderALayers"/>: models must consume this list directly
    /// rather than expecting it to be visible through <c>Layers</c>.
    /// Typically maps to <c>TextEncoderLayers</c> on the concrete model.
    /// </summary>
    public List<ILayer<T>> EncoderBLayers { get; } = new List<ILayer<T>>();

    /// <summary>
    /// Initializes a new dual-encoder architecture with the given encoder stacks.
    /// </summary>
    protected DualEncoderArchitecture(
        IEnumerable<ILayer<T>> encoderALayers,
        IEnumerable<ILayer<T>> encoderBLayers,
        InputType inputType = InputType.ThreeDimensional,
        NeuralNetworkTaskType taskType = NeuralNetworkTaskType.NaturalLanguageProcessing,
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
        if (encoderALayers is null) throw new ArgumentNullException(nameof(encoderALayers));
        if (encoderBLayers is null) throw new ArgumentNullException(nameof(encoderBLayers));
        // Validate individual layer entries rather than just the enumerable
        // references — a null inside the sequence would silently corrupt the
        // encoder stack and only fail during forward/backward traversal much
        // later (with a much less actionable stack trace). Fail fast here.
        foreach (var layer in encoderALayers)
        {
            if (layer is null)
                throw new ArgumentException(
                    $"encoderALayers contains a null element at index {EncoderALayers.Count}.",
                    nameof(encoderALayers));
            EncoderALayers.Add(layer);
        }
        foreach (var layer in encoderBLayers)
        {
            if (layer is null)
                throw new ArgumentException(
                    $"encoderBLayers contains a null element at index {EncoderBLayers.Count}.",
                    nameof(encoderBLayers));
            EncoderBLayers.Add(layer);
        }
    }
}

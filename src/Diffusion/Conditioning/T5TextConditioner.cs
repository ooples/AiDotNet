using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tokenization.HuggingFace;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Diffusion.Conditioning;

/// <summary>
/// T5 text encoder conditioning module (Raffel et al., JMLR 2020).
/// Used as the conditioning encoder for Stable Diffusion 3, FLUX.1, and
/// Imagen pipelines. Pre-LN RMSNorm stack with learned relative position
/// bias (paper-shared across all encoder layers).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
[ModelDomain(ModelDomain.NaturalLanguageProcessing)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.Embedding)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer",
    "https://arxiv.org/abs/1910.10683",
    Year = 2020,
    Authors = "Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu")]
public class T5TextConditioner<T> : TextConditioningBase<T>
{
    private readonly T5Variant _variant;
    /// <summary>Explicit transformer dimensions; null means the variant's paper value.</summary>
    private readonly int? _hiddenSizeOverride;
    private readonly int? _numLayersOverride;
    private readonly int? _numHeadsOverride;

    public override bool ProducesPooledOutput => false;

    /// <param name="hiddenSize">Transformer width. Defaults to the variant's paper value; the
    /// conditioner's embedding dimension follows it.</param>
    /// <param name="numLayers">Transformer depth. Defaults to the variant's paper value.</param>
    /// <param name="numHeads">Attention heads; must divide <paramref name="hiddenSize"/>.
    /// Defaults to the variant's paper value.</param>
    public T5TextConditioner(
        ITokenizer tokenizer,
        T5Variant variant = T5Variant.Base,
        NeuralNetworkArchitecture<T>? architecture = null,
        int? hiddenSize = null,
        int? numLayers = null,
        int? numHeads = null)
        : base(
            architecture: architecture ?? BuildDefaultArchitecture(variant),
            tokenizer: tokenizer,
            maxSequenceLength: 512,
            embeddingDimension: hiddenSize ?? GetEmbeddingDim(variant))
    {
        Guard.NotNull(tokenizer);
        _variant = variant;
        if (hiddenSize is <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenSize));
        if (numLayers is <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        if (numHeads is <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));
        int effectiveHidden = hiddenSize ?? GetHiddenSize(variant);
        int effectiveHeads = numHeads ?? GetNumHeads(variant);
        if (effectiveHidden % effectiveHeads != 0)
            throw new ArgumentException(
                $"hiddenSize ({effectiveHidden}) must be divisible by numHeads ({effectiveHeads}).",
                nameof(numHeads));
        _hiddenSizeOverride = hiddenSize;
        _numLayersOverride = numLayers;
        _numHeadsOverride = numHeads;
    
        // Build the layer stack here, where this subclass's own fields are set. The base cannot do
        // it: CreateDefaultLayers is abstract and reads subclass state (CLIP reads _variant), so a
        // call from the base constructor would run before those fields exist - which is why the
        // stack was previously deferred to the first forward instead.
        //
        // Deferring it made ParameterCount, GetParameters, named activations, serialization and
        // clone all see a model with no layers at all until someone ran a forward (#2151). The
        // saving that deferral was protecting is unaffected: AiDotNet's layers are weight-lazy
        // (InputShape[0] = -1 until resolved), so constructing the layer OBJECTS allocates no
        // weights, and a T5-XXL variant still pays for its parameters only at first forward.
        InitializeLayers();
}

    /// <summary>
    /// Loads a paper-canonical T5 conditioner with its real pretrained
    /// SentencePiece tokenizer from HuggingFace. Default
    /// <paramref name="huggingFaceModelName"/> is variant-aware:
    /// <c>google/t5-v1_1-base</c> for Base, <c>google/t5-v1_1-xxl</c> for XXL, etc.
    /// </summary>
    public static T5TextConditioner<T> FromPretrained(
        T5Variant variant = T5Variant.Base,
        string? huggingFaceModelName = null,
        string? cacheDir = null)
    {
        string modelName = huggingFaceModelName ?? variant switch
        {
            T5Variant.Small => "google/t5-v1_1-small",
            T5Variant.Base  => "google/t5-v1_1-base",
            T5Variant.Large => "google/t5-v1_1-large",
            T5Variant.XL    => "google/t5-v1_1-xl",
            T5Variant.XXL   => "google/t5-v1_1-xxl",
            _ => "google/t5-v1_1-base",
        };
        var tokenizer = AutoTokenizer.FromPretrained(modelName, cacheDir);
        return new T5TextConditioner<T>(tokenizer, variant);
    }

    protected override IEnumerable<ILayer<T>> CreateDefaultLayers() =>
        LayerHelper<T>.CreateDefaultT5TextLayers(
            vocabSize: VocabSize,
            hiddenSize: _hiddenSizeOverride ?? GetHiddenSize(_variant),
            numLayers: _numLayersOverride ?? GetNumLayers(_variant),
            numHeads: _numHeadsOverride ?? GetNumHeads(_variant));

    /// <summary>
    /// T5 pools by mean over non-pad tokens. With fixed-length padding (the
    /// SD3/FLUX/Imagen convention) the base class's <see cref="TextConditioningBase{T}.MeanPool"/>
    /// is paper-faithful.
    /// </summary>
    public override Tensor<T> GetPooledEmbedding(Tensor<T> sequenceEmbeddings) =>
        MeanPool(sequenceEmbeddings);

    private static NeuralNetworkArchitecture<T> BuildDefaultArchitecture(T5Variant variant) =>
        new NeuralNetworkArchitecture<T>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Custom,
            complexity: NetworkComplexity.Deep,
            inputSize: 1);

    private static int GetEmbeddingDim(T5Variant variant) => variant switch
    {
        T5Variant.Small => 512,
        T5Variant.Base => 768,
        T5Variant.Large => 1024,
        T5Variant.XL => 2048,
        T5Variant.XXL => 4096,
        _ => 768,
    };
    private static int GetHiddenSize(T5Variant variant) => GetEmbeddingDim(variant);
    private static int GetNumLayers(T5Variant variant) => variant switch
    {
        T5Variant.Small => 6,
        T5Variant.Base => 12,
        T5Variant.Large => 24,
        T5Variant.XL => 24,
        T5Variant.XXL => 24,
        _ => 12,
    };
    private static int GetNumHeads(T5Variant variant) => variant switch
    {
        T5Variant.Small => 8,
        T5Variant.Base => 12,
        T5Variant.Large => 16,
        T5Variant.XL => 32,
        T5Variant.XXL => 64,
        _ => 12,
    };
}

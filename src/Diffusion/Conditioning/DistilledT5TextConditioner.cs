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
/// Distilled T5 text encoder conditioning module — same architecture as T5
/// but half the layer count, per the DistilBERT-style knowledge-distillation
/// recipe (Sanh et al., 2019).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
[ModelDomain(ModelDomain.NaturalLanguageProcessing)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.Embedding)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter",
    "https://arxiv.org/abs/1910.01108",
    Year = 2019,
    Authors = "Victor Sanh, Lysandre Debut, Julien Chaumond, Thomas Wolf")]
public class DistilledT5TextConditioner<T> : TextConditioningBase<T>
{
    private readonly DistilledT5Variant _variant;
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
    public DistilledT5TextConditioner(
        ITokenizer tokenizer,
        DistilledT5Variant variant = DistilledT5Variant.Base,
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
    /// Loads a paper-canonical DistilledT5 conditioner with its real
    /// pretrained tokenizer (shares the T5 SentencePiece vocab from
    /// <c>google/t5-v1_1-base</c>).
    /// </summary>
    public static DistilledT5TextConditioner<T> FromPretrained(
        DistilledT5Variant variant = DistilledT5Variant.Base,
        string huggingFaceModelName = "google/t5-v1_1-base",
        string? cacheDir = null)
    {
        var tokenizer = AutoTokenizer.FromPretrained(huggingFaceModelName, cacheDir);
        return new DistilledT5TextConditioner<T>(tokenizer, variant);
    }

    protected override IEnumerable<ILayer<T>> CreateDefaultLayers() =>
        LayerHelper<T>.CreateDefaultDistilledT5TextLayers(
            vocabSize: VocabSize,
            hiddenSize: _hiddenSizeOverride ?? GetHiddenSize(_variant),
            numLayers: _numLayersOverride ?? GetNumLayers(_variant),
            numHeads: _numHeadsOverride ?? GetNumHeads(_variant));

    private static NeuralNetworkArchitecture<T> BuildDefaultArchitecture(DistilledT5Variant variant) =>
        new NeuralNetworkArchitecture<T>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Custom,
            complexity: NetworkComplexity.Medium,
            inputSize: 1);

    private static int GetEmbeddingDim(DistilledT5Variant variant) => variant switch
    {
        DistilledT5Variant.Small => 512,
        DistilledT5Variant.Base => 768,
        DistilledT5Variant.Large => 1024,
        _ => 768,
    };
    private static int GetHiddenSize(DistilledT5Variant variant) => GetEmbeddingDim(variant);
    // Half the layers vs full T5.
    private static int GetNumLayers(DistilledT5Variant variant) => variant switch
    {
        DistilledT5Variant.Small => 3,
        DistilledT5Variant.Base => 6,
        DistilledT5Variant.Large => 12,
        _ => 6,
    };
    private static int GetNumHeads(DistilledT5Variant variant) => variant switch
    {
        DistilledT5Variant.Small => 8,
        DistilledT5Variant.Base => 12,
        DistilledT5Variant.Large => 16,
        _ => 12,
    };
}

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
/// SigLIP text encoder conditioning module (Zhai et al., ICCV 2023).
/// Same encoder architecture as CLIP; the paper's contribution is the
/// sigmoid contrastive loss (vs CLIP's softmax), which is upstream of
/// this encoder body.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.Embedding)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Sigmoid Loss for Language Image Pre-Training",
    "https://arxiv.org/abs/2303.15343",
    Year = 2023,
    Authors = "Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, Lucas Beyer")]
public class SigLIPTextConditioner<T> : TextConditioningBase<T>
{
    private readonly SigLIPVariant _variant;
    /// <summary>Explicit transformer dimensions; null means the variant's paper value.</summary>
    private readonly int? _hiddenSizeOverride;
    private readonly int? _numLayersOverride;
    private readonly int? _numHeadsOverride;

    public override bool ProducesPooledOutput => true;

    /// <param name="hiddenSize">Transformer width. Defaults to the variant's paper value; the
    /// conditioner's embedding dimension follows it.</param>
    /// <param name="numLayers">Transformer depth. Defaults to the variant's paper value.</param>
    /// <param name="numHeads">Attention heads; must divide <paramref name="hiddenSize"/>.
    /// Defaults to the variant's paper value.</param>
    public SigLIPTextConditioner(
        ITokenizer tokenizer,
        SigLIPVariant variant = SigLIPVariant.Base,
        NeuralNetworkArchitecture<T>? architecture = null,
        int? hiddenSize = null,
        int? numLayers = null,
        int? numHeads = null)
        : base(
            architecture: architecture ?? BuildDefaultArchitecture(variant),
            tokenizer: tokenizer,
            maxSequenceLength: 64,
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
    /// Loads a paper-canonical SigLIP text conditioner with its real
    /// pretrained tokenizer from HuggingFace. Default model is
    /// <c>google/siglip-base-patch16-224</c>.
    /// </summary>
    public static SigLIPTextConditioner<T> FromPretrained(
        SigLIPVariant variant = SigLIPVariant.Base,
        string huggingFaceModelName = "google/siglip-base-patch16-224",
        string? cacheDir = null)
    {
        var tokenizer = AutoTokenizer.FromPretrained(huggingFaceModelName, cacheDir);
        return new SigLIPTextConditioner<T>(tokenizer, variant);
    }

    protected override IEnumerable<ILayer<T>> CreateDefaultLayers() =>
        LayerHelper<T>.CreateDefaultSigLIPTextLayers(
            vocabSize: VocabSize,
            maxSeqLen: MaxSequenceLength,
            hiddenSize: _hiddenSizeOverride ?? GetHiddenSize(_variant),
            numLayers: _numLayersOverride ?? GetNumLayers(_variant),
            numHeads: _numHeadsOverride ?? GetNumHeads(_variant));

    private static NeuralNetworkArchitecture<T> BuildDefaultArchitecture(SigLIPVariant variant) =>
        new NeuralNetworkArchitecture<T>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Custom,
            complexity: NetworkComplexity.Deep,
            inputSize: 1);

    private static int GetEmbeddingDim(SigLIPVariant variant) => variant switch
    {
        SigLIPVariant.Base => 768,
        SigLIPVariant.Large => 1024,
        SigLIPVariant.So400M => 1152,
        _ => 768,
    };
    private static int GetHiddenSize(SigLIPVariant variant) => GetEmbeddingDim(variant);
    private static int GetNumLayers(SigLIPVariant variant) => variant switch
    {
        SigLIPVariant.Base => 12,
        SigLIPVariant.Large => 24,
        SigLIPVariant.So400M => 27,
        _ => 12,
    };
    private static int GetNumHeads(SigLIPVariant variant) => variant switch
    {
        SigLIPVariant.Base => 12,
        SigLIPVariant.Large => 16,
        SigLIPVariant.So400M => 16,
        _ => 12,
    };
}

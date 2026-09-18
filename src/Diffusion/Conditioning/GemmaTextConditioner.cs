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
/// Gemma text encoder conditioning module (Gemma Team 2024).
/// Pre-LN RMSNorm Transformer stack with RoPE multi-head attention and
/// SiLU FFN. Used by Imagen 3 and other Google diffusion pipelines.
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
    "Gemma: Open Models Based on Gemini Research and Technology",
    "https://arxiv.org/abs/2403.08295",
    Year = 2024,
    Authors = "Gemma Team")]
public class GemmaTextConditioner<T> : TextConditioningBase<T>
{
    private readonly GemmaVariant _variant;
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
    public GemmaTextConditioner(
        ITokenizer tokenizer,
        GemmaVariant variant = GemmaVariant.TwoB,
        NeuralNetworkArchitecture<T>? architecture = null,
        int? hiddenSize = null,
        int? numLayers = null,
        int? numHeads = null)
        : base(
            architecture: architecture ?? BuildDefaultArchitecture(variant),
            tokenizer: tokenizer,
            maxSequenceLength: 8192,
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
    /// Loads a paper-canonical Gemma conditioner with its real pretrained
    /// SentencePiece tokenizer from HuggingFace.
    /// </summary>
    public static GemmaTextConditioner<T> FromPretrained(
        GemmaVariant variant = GemmaVariant.TwoB,
        string? huggingFaceModelName = null,
        string? cacheDir = null)
    {
        string modelName = huggingFaceModelName ?? variant switch
        {
            GemmaVariant.SevenB => "google/gemma-7b",
            _ => "google/gemma-2b",
        };
        var tokenizer = AutoTokenizer.FromPretrained(modelName, cacheDir);
        return new GemmaTextConditioner<T>(tokenizer, variant);
    }

    protected override IEnumerable<ILayer<T>> CreateDefaultLayers() =>
        LayerHelper<T>.CreateDefaultGemmaTextLayers(
            vocabSize: VocabSize,
            maxSeqLen: MaxSequenceLength,
            hiddenSize: _hiddenSizeOverride ?? GetHiddenSize(_variant),
            numLayers: _numLayersOverride ?? GetNumLayers(_variant),
            numHeads: _numHeadsOverride ?? GetNumHeads(_variant));

    /// <summary>
    /// Decoder-style models pool by extracting the embedding at the last
    /// non-pad token position. With fixed-length padded sequences (the
    /// diffusion-pipeline convention), the last position is canonical.
    /// </summary>
    public override Tensor<T> GetPooledEmbedding(Tensor<T> sequenceEmbeddings)
    {
        int rank = sequenceEmbeddings.Shape.Length;
        if (rank != 3)
            throw new ArgumentException(
                $"GetPooledEmbedding expects rank-3 [B, S, D]; got rank {rank}.");
        int batch = sequenceEmbeddings.Shape[0];
        int seqLen = sequenceEmbeddings.Shape[1];
        int dim = sequenceEmbeddings.Shape[2];
        var pooled = new Vector<T>(batch * dim);
        for (int b = 0; b < batch; b++)
            for (int d = 0; d < dim; d++)
                pooled[b * dim + d] = sequenceEmbeddings[b * seqLen * dim + (seqLen - 1) * dim + d];
        return new Tensor<T>(new[] { batch, dim }, pooled);
    }

    private static NeuralNetworkArchitecture<T> BuildDefaultArchitecture(GemmaVariant variant) =>
        new NeuralNetworkArchitecture<T>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Custom,
            complexity: NetworkComplexity.Deep,
            inputSize: 1);

    private static int GetEmbeddingDim(GemmaVariant variant) => variant switch
    {
        GemmaVariant.SevenB => 3072, _ => 2048,
    };
    private static int GetHiddenSize(GemmaVariant variant) => GetEmbeddingDim(variant);
    private static int GetNumLayers(GemmaVariant variant) => variant switch
    {
        GemmaVariant.SevenB => 28, _ => 18,
    };
    private static int GetNumHeads(GemmaVariant variant) => variant switch
    {
        GemmaVariant.SevenB => 16, _ => 8,
    };
}

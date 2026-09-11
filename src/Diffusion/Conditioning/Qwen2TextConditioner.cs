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
/// Qwen2 text encoder conditioning module (Yang et al. 2024).
/// Pre-LN RMSNorm Transformer stack with RoPE grouped-query attention and
/// SiLU FFN. Used in multilingual diffusion pipelines for stronger Asian-
/// language prompt understanding.
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
    "Qwen2 Technical Report",
    "https://arxiv.org/abs/2407.10671",
    Year = 2024,
    Authors = "An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, et al.")]
public class Qwen2TextConditioner<T> : TextConditioningBase<T>
{
    private readonly Qwen2Variant _variant;
    /// <summary>Explicit transformer dimensions; null means the variant's paper value.</summary>
    private readonly int? _hiddenSizeOverride;
    private readonly int? _numLayersOverride;
    private readonly int? _numHeadsOverride;
    private readonly int? _numKvHeadsOverride;

    public override bool ProducesPooledOutput => false;

    /// <param name="hiddenSize">Transformer width. Defaults to the variant's paper value; the
    /// conditioner's embedding dimension follows it.</param>
    /// <param name="numLayers">Transformer depth. Defaults to the variant's paper value.</param>
    /// <param name="numHeads">Attention heads; must divide <paramref name="hiddenSize"/>.
    /// Defaults to the variant's paper value.</param>
    /// <param name="numKvHeads">Key/value heads for grouped-query attention; must divide
    /// <paramref name="numHeads"/>. Defaults to the variant's paper value.</param>
    public Qwen2TextConditioner(
        ITokenizer tokenizer,
        Qwen2Variant variant = Qwen2Variant.OnePointFiveB,
        NeuralNetworkArchitecture<T>? architecture = null,
        int? hiddenSize = null,
        int? numLayers = null,
        int? numHeads = null,
        int? numKvHeads = null)
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
        if (numKvHeads is <= 0) throw new ArgumentOutOfRangeException(nameof(numKvHeads));
        int effectiveKvHeads = numKvHeads ?? GetNumKvHeads(variant);
        if (effectiveHeads % effectiveKvHeads != 0)
            throw new ArgumentException(
                $"numHeads ({effectiveHeads}) must be divisible by numKvHeads ({effectiveKvHeads}).",
                nameof(numKvHeads));
        _hiddenSizeOverride = hiddenSize;
        _numLayersOverride = numLayers;
        _numHeadsOverride = numHeads;
        _numKvHeadsOverride = numKvHeads;
    
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
    /// Loads a paper-canonical Qwen2 conditioner with its real pretrained
    /// BPE tokenizer from HuggingFace.
    /// </summary>
    public static Qwen2TextConditioner<T> FromPretrained(
        Qwen2Variant variant = Qwen2Variant.OnePointFiveB,
        string? huggingFaceModelName = null,
        string? cacheDir = null)
    {
        string modelName = huggingFaceModelName ?? variant switch
        {
            Qwen2Variant.SevenB => "Qwen/Qwen2-7B",
            _ => "Qwen/Qwen2-1.5B",
        };
        var tokenizer = AutoTokenizer.FromPretrained(modelName, cacheDir);
        return new Qwen2TextConditioner<T>(tokenizer, variant);
    }

    protected override IEnumerable<ILayer<T>> CreateDefaultLayers() =>
        LayerHelper<T>.CreateDefaultQwen2TextLayers(
            vocabSize: VocabSize,
            maxSeqLen: MaxSequenceLength,
            hiddenSize: _hiddenSizeOverride ?? GetHiddenSize(_variant),
            numLayers: _numLayersOverride ?? GetNumLayers(_variant),
            numHeads: _numHeadsOverride ?? GetNumHeads(_variant),
            numKvHeads: _numKvHeadsOverride ?? GetNumKvHeads(_variant));

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

    private static NeuralNetworkArchitecture<T> BuildDefaultArchitecture(Qwen2Variant variant) =>
        new NeuralNetworkArchitecture<T>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Custom,
            complexity: NetworkComplexity.Deep,
            inputSize: 1);

    private static int GetEmbeddingDim(Qwen2Variant variant) => variant switch
    {
        Qwen2Variant.OnePointFiveB => 1536,
        Qwen2Variant.SevenB => 4096,
        _ => 1536,
    };
    private static int GetHiddenSize(Qwen2Variant variant) => GetEmbeddingDim(variant);
    private static int GetNumLayers(Qwen2Variant variant) => variant switch
    {
        Qwen2Variant.OnePointFiveB => 28,
        Qwen2Variant.SevenB => 32,
        _ => 28,
    };
    private static int GetNumHeads(Qwen2Variant variant) => variant switch
    {
        Qwen2Variant.OnePointFiveB => 12,
        Qwen2Variant.SevenB => 32,
        _ => 12,
    };
    // Qwen2 GQA ratio (Yang 2024 Table 1): heads/kv_heads = 6 for 1.5B, 4 for 7B.
    private static int GetNumKvHeads(Qwen2Variant variant) => variant switch
    {
        Qwen2Variant.OnePointFiveB => 2,
        Qwen2Variant.SevenB => 8,
        _ => 2,
    };
}

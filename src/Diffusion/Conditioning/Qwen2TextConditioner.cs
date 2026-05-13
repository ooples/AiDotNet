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

    public override bool ProducesPooledOutput => false;

    public Qwen2TextConditioner(
        ITokenizer tokenizer,
        Qwen2Variant variant = Qwen2Variant.OnePointFiveB,
        NeuralNetworkArchitecture<T>? architecture = null)
        : base(
            architecture: architecture ?? BuildDefaultArchitecture(variant),
            tokenizer: tokenizer,
            maxSequenceLength: 512,
            embeddingDimension: GetEmbeddingDim(variant))
    {
        Guard.NotNull(tokenizer);
        _variant = variant;
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
            hiddenSize: GetHiddenSize(_variant),
            numLayers: GetNumLayers(_variant),
            numHeads: GetNumHeads(_variant),
            numKvHeads: GetNumKvHeads(_variant));

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() =>
        new Qwen2TextConditioner<T>(Tokenizer, _variant, Architecture);

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

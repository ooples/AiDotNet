using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;

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

    public override bool ProducesPooledOutput => false;

    public GemmaTextConditioner(
        GemmaVariant variant = GemmaVariant.TwoB,
        ITokenizer? tokenizer = null,
        NeuralNetworkArchitecture<T>? architecture = null)
        : base(
            architecture: architecture ?? BuildDefaultArchitecture(variant),
            // Gemma uses SentencePiece — we use the LLaMA-style tokenizer
            // factory as the closest SentencePiece-based default. Production
            // pipelines should pass a real Gemma tokenizer via the tokenizer
            // parameter (e.g. AutoTokenizer.FromPretrained("google/gemma-2b")).
            tokenizer: tokenizer ?? LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.LLaMA),
            maxSequenceLength: 256,
            embeddingDimension: GetEmbeddingDim(variant))
    {
        _variant = variant;
    }

    protected override IEnumerable<ILayer<T>> CreateDefaultLayers() =>
        LayerHelper<T>.CreateDefaultGemmaTextLayers(
            vocabSize: VocabSize,
            maxSeqLen: MaxSequenceLength,
            hiddenSize: GetHiddenSize(_variant),
            numLayers: GetNumLayers(_variant),
            numHeads: GetNumHeads(_variant));

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() =>
        new GemmaTextConditioner<T>(_variant, Tokenizer, Architecture);

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
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Custom,
            complexity: NetworkComplexity.Deep,
            inputSize: 256,
            outputSize: GetEmbeddingDim(variant));

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

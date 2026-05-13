using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;

namespace AiDotNet.Diffusion.Conditioning;

/// <summary>
/// ChatGLM3 text encoder conditioning module (Zeng et al. 2023).
/// Pre-LN RMSNorm Transformer stack with RoPE multi-query attention
/// (KV heads = 1) and SiLU FFN. Used in Kolors and other Chinese-language
/// diffusion pipelines.
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
    "GLM-130B: An Open Bilingual Pre-trained Model",
    "https://arxiv.org/abs/2210.02414",
    Year = 2023,
    Authors = "Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi Zheng, Xiao Xia, Weng Lam Tam, Zixuan Ma, Yufei Xue, Jidong Zhai, Wenguang Chen, Peng Zhang, Yuxiao Dong, Jie Tang")]
public class ChatGLM3TextConditioner<T> : TextConditioningBase<T>
{
    private readonly ChatGLM3Variant _variant;

    public override bool ProducesPooledOutput => false;

    public ChatGLM3TextConditioner(
        ChatGLM3Variant variant = ChatGLM3Variant.SixB,
        ITokenizer? tokenizer = null,
        NeuralNetworkArchitecture<T>? architecture = null)
        : base(
            architecture: architecture ?? BuildDefaultArchitecture(variant),
            // ChatGLM uses its own SentencePiece tokenizer; no exact match in
            // LanguageModelTokenizerFactory yet, so the closest available
            // SentencePiece default is the LLaMA-family tokenizer. Production
            // pipelines should pass a real ChatGLM tokenizer via the
            // constructor parameter (AutoTokenizer.FromPretrained("THUDM/chatglm3-6b")).
            tokenizer: tokenizer ?? LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.LLaMA),
            maxSequenceLength: 512,
            embeddingDimension: GetEmbeddingDim(variant))
    {
        _variant = variant;
    }

    protected override IEnumerable<ILayer<T>> CreateDefaultLayers() =>
        LayerHelper<T>.CreateDefaultChatGLM3TextLayers(
            vocabSize: VocabSize,
            maxSeqLen: MaxSequenceLength,
            hiddenSize: GetHiddenSize(_variant),
            numLayers: GetNumLayers(_variant),
            numHeads: GetNumHeads(_variant),
            numKvHeads: GetNumKvHeads(_variant));

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() =>
        new ChatGLM3TextConditioner<T>(_variant, Tokenizer, Architecture);

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

    private static NeuralNetworkArchitecture<T> BuildDefaultArchitecture(ChatGLM3Variant variant) =>
        new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Custom,
            complexity: NetworkComplexity.Deep,
            inputSize: 512,
            outputSize: GetEmbeddingDim(variant));

    private static int GetEmbeddingDim(ChatGLM3Variant _) => 4096;
    private static int GetHiddenSize(ChatGLM3Variant variant) => GetEmbeddingDim(variant);
    private static int GetNumLayers(ChatGLM3Variant _) => 28;
    private static int GetNumHeads(ChatGLM3Variant _) => 32;
    private static int GetNumKvHeads(ChatGLM3Variant _) => 2;  // ChatGLM3 uses 2-head GQA.
}

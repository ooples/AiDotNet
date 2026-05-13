using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;

namespace AiDotNet.Diffusion.Conditioning;

/// <summary>
/// CLIP text encoder conditioning module (Radford et al., ICML 2021).
/// Primary text conditioner for Stable Diffusion 1.x / 2.x and one of two
/// encoders in SDXL / SD3 / FLUX.1.
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
    "Learning Transferable Visual Models From Natural Language Supervision",
    "https://arxiv.org/abs/2103.00020",
    Year = 2021,
    Authors = "Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever")]
public class CLIPTextConditioner<T> : TextConditioningBase<T>
{
    private readonly CLIPVariant _variant;

    public override bool ProducesPooledOutput => true;

    public CLIPTextConditioner(
        CLIPVariant variant = CLIPVariant.ViTL14,
        ITokenizer? tokenizer = null,
        NeuralNetworkArchitecture<T>? architecture = null)
        : base(
            architecture: architecture ?? BuildDefaultArchitecture(variant),
            tokenizer: tokenizer ?? ClipTokenizerFactory.CreateSimple(),
            maxSequenceLength: 77,
            embeddingDimension: GetEmbeddingDim(variant))
    {
        _variant = variant;
    }

    protected override IEnumerable<ILayer<T>> CreateDefaultLayers() =>
        LayerHelper<T>.CreateDefaultCLIPTextLayers(
            vocabSize: VocabSize,
            maxSeqLen: MaxSequenceLength,
            hiddenSize: GetHiddenSize(_variant),
            numLayers: GetNumLayers(_variant),
            numHeads: GetNumHeads(_variant),
            projectionDim: EmbeddingDimension);

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() =>
        new CLIPTextConditioner<T>(_variant, Tokenizer, Architecture);

    /// <summary>
    /// CLIP pools by extracting the embedding at the EOS token position
    /// (Radford 2021 §3.1). Diffusion pipelines pad to fixed length so the
    /// canonical EOS placement is the last sequence position.
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
        {
            int eosPos = seqLen - 1;
            for (int d = 0; d < dim; d++)
                pooled[b * dim + d] = sequenceEmbeddings[b * seqLen * dim + eosPos * dim + d];
        }
        return new Tensor<T>(new[] { batch, dim }, pooled);
    }

    private static NeuralNetworkArchitecture<T> BuildDefaultArchitecture(CLIPVariant variant) =>
        new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Custom,
            complexity: NetworkComplexity.Deep,
            inputSize: 77,
            outputSize: GetEmbeddingDim(variant));

    private static int GetEmbeddingDim(CLIPVariant variant) => variant switch
    {
        CLIPVariant.ViTL14 => 768,
        CLIPVariant.ViTH14 => 1024,
        CLIPVariant.ViTBigG14 => 1280,
        _ => 768,
    };
    private static int GetHiddenSize(CLIPVariant variant) => GetEmbeddingDim(variant);
    private static int GetNumLayers(CLIPVariant variant) => variant switch
    {
        CLIPVariant.ViTL14 => 12,
        CLIPVariant.ViTH14 => 24,
        CLIPVariant.ViTBigG14 => 32,
        _ => 12,
    };
    private static int GetNumHeads(CLIPVariant variant) => variant switch
    {
        CLIPVariant.ViTL14 => 12,
        CLIPVariant.ViTH14 => 16,
        CLIPVariant.ViTBigG14 => 20,
        _ => 12,
    };
}

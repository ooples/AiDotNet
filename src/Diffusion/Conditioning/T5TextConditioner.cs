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

    public override bool ProducesPooledOutput => false;

    public T5TextConditioner(
        ITokenizer tokenizer,
        T5Variant variant = T5Variant.Base,
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
            hiddenSize: GetHiddenSize(_variant),
            numLayers: GetNumLayers(_variant),
            numHeads: GetNumHeads(_variant));

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() =>
        new T5TextConditioner<T>(Tokenizer, _variant, Architecture);

    /// <summary>
    /// T5 pools by mean over non-pad tokens. With fixed-length padding (the
    /// SD3/FLUX/Imagen convention) the base class's <see cref="TextConditioningBase{T}.MeanPool"/>
    /// is paper-faithful.
    /// </summary>
    public override Tensor<T> GetPooledEmbedding(Tensor<T> sequenceEmbeddings) =>
        MeanPool(sequenceEmbeddings);

    private static NeuralNetworkArchitecture<T> BuildDefaultArchitecture(T5Variant variant) =>
        new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
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

using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Finance.Base;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;

namespace AiDotNet.Finance.NLP;

/// <summary>
/// FinancialBERT neural network model for domain-specific financial language processing.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> FinancialBERT is a BERT model pre-trained on financial text
/// from corporate reports, analyst notes, and financial news. It understands financial
/// terminology and context better than general-purpose language models. Use it for tasks
/// like classifying financial documents, extracting key information from earnings reports,
/// or detecting sentiment in market commentary.</para>
/// </remarks>
/// <example>
/// <code>
/// // Define architecture for financial document classification (512 tokens, 5 document types)
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.MultiClassClassification,
///     inputHeight: 512, inputWidth: 1, inputDepth: 1, outputSize: 5);
///
/// // Training mode: BERT pre-trained on corporate reports and financial news
/// var model = new FinancialBERT&lt;double&gt;(architecture);
///
/// // ONNX inference mode: load pre-trained FinancialBERT model
/// var onnxModel = new FinancialBERT&lt;double&gt;(architecture, "financial_bert.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ResearchPaper("FinancialBERT - A Pretrained Language Model for Financial Text Mining", "https://www.researchgate.net/publication/358284785_FinancialBERT_-_A_Pretrained_Language_Model_for_Financial_Text_Mining", Year = 2022, Authors = "Ahmed Rachid Hazourli")]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 2e-5, ReferenceBatchSize = 32,
                Source = "The FinancialBERT paper's fine-tuning settings are not reproduced here, so these are BERT's: Devlin et al. 2019, Sec. A.3, fine-tune BERT with Adam at a learning "
                        + "rate chosen from 5e-5, 3e-5 and 2e-5 with a batch size of 16 or 32; the smallest "
                        + "rate and the larger batch are taken here.")]
public partial class FinancialBERT<T> : FinancialBertModelBase<T>
{
    private readonly FinancialBERTOptions<T> _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <summary>
    /// Creates a FinancialBERT that runs a pretrained ONNX model.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="onnxModelPath">Path to the ONNX model.</param>
    /// <param name="options">The model options; the paper's defaults when null.</param>
    /// <param name="optimizer">The optimizer; the paper's when null.</param>
    public FinancialBERT(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        FinancialBERTOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, onnxModelPath, OrDefault(options).MaxSequenceLength, OrDefault(options).VocabularySize,
            OrDefault(options).HiddenDimension, ResolveClassCount(architecture, OrDefault(options).NumClasses))
    {
        _options = options ?? new FinancialBERTOptions<T>();
        Options = _options;
        _options.Validate();
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a FinancialBERT built from native layers, which can be trained.
    /// </summary>
    /// <param name="architecture">The neural network architecture; custom layers replace the encoder.</param>
    /// <param name="options">The model options; the paper's defaults when null.</param>
    /// <param name="optimizer">The optimizer; the paper's when null.</param>
    /// <param name="lossFunction">The loss; cross-entropy on logits when null.</param>
    public FinancialBERT(
        NeuralNetworkArchitecture<T> architecture,
        FinancialBERTOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, OrDefault(options).MaxSequenceLength, OrDefault(options).VocabularySize,
            OrDefault(options).HiddenDimension, ResolveClassCount(architecture, OrDefault(options).NumClasses), lossFunction)
    {
        _options = options ?? new FinancialBERTOptions<T>();
        Options = _options;
        _options.Validate();
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        InitializeLayers();
    }

    private static FinancialBERTOptions<T> OrDefault(FinancialBERTOptions<T>? options) => options ?? new FinancialBERTOptions<T>();

    /// <inheritdoc/>
    protected override FinancialBertEncoderOptions<T> BertOptions => _options;

    /// <inheritdoc/>
    protected override void InitializeLayers()
        => InitializeBertLayers(
            _options.NumAttentionHeads, _options.IntermediateDimension, _options.NumLayers, NumSentimentClasses,
            _options.DropoutRate);

    /// <inheritdoc/>
    protected override void TrainCore(Tensor<T> input, Tensor<T> target, Tensor<T> output)
    {
        SetTrainingMode(true);
        _optimizer.UpdateParameters(Layers);
        SetTrainingMode(false);
    }
}

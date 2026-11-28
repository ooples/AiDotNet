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
/// SEC-BERT neural network model for domain-specific financial language processing.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// SEC-BERT is a BERT-based model specifically pretrained on SEC filings (10-K, 10-Q, etc.).
/// </para>
/// <para><b>For Beginners:</b> SEC-BERT is a language model trained exclusively on SEC
/// filings (10-K annual reports, 10-Q quarterly reports, 8-K current reports). It understands
/// the unique language and structure of regulatory documents, making it ideal for extracting
/// information from corporate disclosures, identifying risk factors, and classifying
/// financial statements.</para>
/// <para>
/// Reference: Loukas et al., "SEC-BERT: A Pre-trained Financial Language Model", 2022.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Define architecture for SEC filing classification (512 tokens, 10 filing categories)
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.MultiClassClassification,
///     inputHeight: 512, inputWidth: 1, inputDepth: 1, outputSize: 10);
///
/// // Training mode: BERT pre-trained on SEC 10-K, 10-Q, and 8-K filings
/// var model = new SECBERT&lt;double&gt;(architecture);
///
/// // ONNX inference mode: load pre-trained SEC-BERT model
/// var onnxModel = new SECBERT&lt;double&gt;(architecture, "secbert.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
// DOI corrected: 10.1145/3533271.3561753 is "Learning to simulate realistic limit order book
// markets from data as a World Agent" (ICAIF'22) — an unrelated paper. SEC-BERT is introduced in
// Loukas et al., "FiNER: Financial Numeric Entity Recognition for XBRL Tagging" (ACL 2022).
[ResearchPaper("FiNER: Financial Numeric Entity Recognition for XBRL Tagging (introduces SEC-BERT)", "https://doi.org/10.18653/v1/2022.acl-long.303", Year = 2022, Authors = "Lefteris Loukas, Manos Fergadiotis, Ion Androutsopoulos, Prodromos Malakasiotis")]
public partial class SECBERT<T> : FinancialBertModelBase<T>, ITrainingObjectiveProvider<T>
{
    private readonly SECBERTOptions<T> _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <inheritdoc/>
    public TrainingObjectiveKind TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    /// <inheritdoc/>
    public Tensor<T> ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <inheritdoc/>
    public T EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
        => LossFunction.ComputeTapeLoss(ForwardForTraining(input), target).Data.Span[0];

    /// <inheritdoc/>
    protected internal override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer => _optimizer;

    /// <summary>
    /// Creates a SECBERT that runs a pretrained ONNX model.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="onnxModelPath">Path to the ONNX model.</param>
    /// <param name="options">The model options; the paper's defaults when null.</param>
    /// <param name="optimizer">The optimizer; the paper's when null.</param>
    public SECBERT(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        SECBERTOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, onnxModelPath, OrDefault(options).MaxSequenceLength, OrDefault(options).VocabularySize,
            OrDefault(options).HiddenDimension, ResolveClassCount(architecture, OrDefault(options).NumClasses))
    {
        _options = options ?? new SECBERTOptions<T>();
        Options = _options;
        _options.Validate();
        _optimizer = optimizer ?? CreatePaperOptimizer();
        InitializeLayers();
    }

    /// <summary>
    /// Creates a SECBERT built from native layers, which can be trained.
    /// </summary>
    /// <param name="architecture">The neural network architecture; custom layers replace the encoder.</param>
    /// <param name="options">The model options; the paper's defaults when null.</param>
    /// <param name="optimizer">The optimizer; the paper's when null.</param>
    /// <param name="lossFunction">The loss; cross-entropy on logits when null.</param>
    public SECBERT(
        NeuralNetworkArchitecture<T> architecture,
        SECBERTOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, OrDefault(options).MaxSequenceLength, OrDefault(options).VocabularySize,
            OrDefault(options).HiddenDimension, ResolveClassCount(architecture, OrDefault(options).NumClasses), lossFunction)
    {
        _options = options ?? new SECBERTOptions<T>();
        Options = _options;
        _options.Validate();
        _optimizer = optimizer ?? CreatePaperOptimizer();
        InitializeLayers();
    }

    private static SECBERTOptions<T> OrDefault(SECBERTOptions<T>? options) => options ?? new SECBERTOptions<T>();

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreatePaperOptimizer()
        => new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
                UseAMSGrad = false,
            });

    /// <inheritdoc/>
    protected override FinancialBertEncoderOptions<T> BertOptions => _options;

    /// <inheritdoc/>
    protected override void InitializeLayers()
        => InitializeBertLayers(
            _options.NumAttentionHeads, _options.IntermediateDimension, _options.NumLayers, NumSentimentClasses,
            _options.DropoutRate);
}

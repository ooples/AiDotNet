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
/// InvestLM neural network model specialized for investment professionals and research.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> InvestLM is a language model fine-tuned specifically for
/// investment professionals. It can analyze investment opportunities, summarize financial
/// reports, provide market commentary, and answer questions about portfolio strategies.
/// Think of it as an AI research analyst that understands investment terminology, valuation
/// methods, and market dynamics.</para>
/// </remarks>
/// <example>
/// <code>
/// // Define architecture for investment-focused language generation (2048 tokens)
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.MultiClassClassification,
///     inputHeight: 2048, inputWidth: 1, inputDepth: 1, outputSize: 32000);
///
/// // Training mode: instruction-tuned LLM for investment analysis and portfolio commentary
/// var model = new InvestLM&lt;double&gt;(architecture);
///
/// // ONNX inference mode: load pre-trained InvestLM model
/// var onnxModel = new InvestLM&lt;double&gt;(architecture, "investlm.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("InvestLM: A Large Language Model for Investment using Financial Domain Instruction Tuning", "https://arxiv.org/abs/2309.13064", Year = 2023, Authors = "Yi Yang, Yixuan Tang, Kar Yan Tam")]
[PaperOptimizer(OptimizerKind.Unspecified, LearningRate = 3e-4, ReferenceBatchSize = 16,
                Source = "Yang et al. 2023, Sec. 3: InvestLM is the LLaMA-65B model, trained for 15 "
                        + "epochs at a learning rate of 3e-4 and a batch size of 16 examples. The 3e-3 "
                        + "and batch of 32 in the same passage belong to a LLaMA-7B model tuned for "
                        + "subsequent analysis, not to InvestLM. The optimizer is left unspecified "
                        + "because the paper names none.")]
public partial class InvestLM<T> : FinancialDecoderModelBase<T>
{
    private readonly InvestLMOptions<T> _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <summary>
    /// Creates a InvestLM that runs a pretrained ONNX model.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="onnxModelPath">Path to the ONNX model.</param>
    /// <param name="options">The model options; the paper's defaults when null.</param>
    /// <param name="optimizer">The optimizer; the paper's when null.</param>
    public InvestLM(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        InvestLMOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, onnxModelPath, OrDefault(options).MaxSequenceLength, OrDefault(options).VocabularySize,
            OrDefault(options).HiddenDimension, ResolveClassCount(architecture, OrDefault(options).NumClasses))
    {
        _options = options ?? new InvestLMOptions<T>();
        Options = _options;
        _options.Validate();
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a InvestLM built from native layers, which can be trained.
    /// </summary>
    /// <param name="architecture">The neural network architecture; custom layers replace the encoder.</param>
    /// <param name="options">The model options; the paper's defaults when null.</param>
    /// <param name="optimizer">The optimizer; the paper's when null.</param>
    /// <param name="lossFunction">The loss; cross-entropy on logits when null.</param>
    public InvestLM(
        NeuralNetworkArchitecture<T> architecture,
        InvestLMOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, OrDefault(options).MaxSequenceLength, OrDefault(options).VocabularySize,
            OrDefault(options).HiddenDimension, ResolveClassCount(architecture, OrDefault(options).NumClasses), lossFunction)
    {
        _options = options ?? new InvestLMOptions<T>();
        Options = _options;
        _options.Validate();
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        InitializeLayers();
    }

    private static InvestLMOptions<T> OrDefault(InvestLMOptions<T>? options) => options ?? new InvestLMOptions<T>();

    /// <inheritdoc/>
    protected override FinancialLanguageModelOptions<T> LanguageModelOptions => _options;

    /// <inheritdoc/>
    protected override string ReleasedTokenizerDescription => "the LLaMA SentencePiece tokenizer (32,000 pieces)";

    /// <inheritdoc/>
    protected override void InitializeLayers()
        => InitializeDecoderLayers(
            pool => LayerHelper<T>.CreateLlamaDecoderLayers(
                VocabularySize, MaxSequenceLength, HiddenDimension, _options.NumAttentionHeads,
                _options.IntermediateDimension, _options.NumLayers, pool),
            NumSentimentClasses,
            _options.DropoutRate);
}

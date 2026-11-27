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
/// BloombergGPT neural network model for comprehensive financial language processing.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> BloombergGPT is a 50-billion parameter language model
/// trained on a mix of general text and Bloomberg's massive financial data archive. Unlike
/// general-purpose LLMs, it deeply understands financial jargon, SEC filings, earnings
/// reports, and market commentary. It can analyze financial documents, answer questions
/// about markets, and generate financial text with domain-specific accuracy.</para>
/// </remarks>
/// <example>
/// <code>
/// // Define architecture for financial language generation (2048 max tokens, 50B-parameter vocab)
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.MultiClassClassification,
///     inputHeight: 2048, inputWidth: 1, inputDepth: 1, outputSize: 131072);
///
/// // Training mode: decoder-only transformer trained on Bloomberg financial data
/// var model = new BloombergGPT&lt;double&gt;(architecture);
///
/// // ONNX inference mode: load pre-trained BloombergGPT model
/// var onnxModel = new BloombergGPT&lt;double&gt;(architecture, "bloomberggpt.onnx");
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
[ResearchPaper("BloombergGPT: A Large Language Model for Finance", "https://arxiv.org/abs/2303.17564", Year = 2023, Authors = "Shijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski, Mark Dredze, Sebastian Gehrmann, Prabhanjan Kambadur, David Rosenberg, Gideon Mann")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 6e-5, WeightDecay = 0.1,
                WarmupSteps = 1800, MinLearningRate = 6e-6,
                Schedule = LearningRateSchedulerType.CosineAnnealing,
                Source = "Wu et al. 2023, Sec. 4 and Table: AdamW with a weight decay of 0.1, a maximum "
                        + "learning rate of 6e-5 and a cosine decay schedule with linear warmup over the "
                        + "first 1800 steps, reaching a final learning rate of 6e-6, which the paper "
                        + "gives as 0.1x the maximum. No batch size is declared because the run employs "
                        + "batch size warmup, so no single value holds.")]
public partial class BloombergGPT<T> : FinancialDecoderModelBase<T>
{
    private readonly BloombergGPTOptions<T> _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <summary>
    /// Creates a BloombergGPT that runs a pretrained ONNX model.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="onnxModelPath">Path to the ONNX model.</param>
    /// <param name="options">The model options; the paper's defaults when null.</param>
    /// <param name="optimizer">The optimizer; the paper's when null.</param>
    public BloombergGPT(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        BloombergGPTOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, onnxModelPath, OrDefault(options).MaxSequenceLength, OrDefault(options).VocabularySize,
            OrDefault(options).HiddenDimension, ResolveClassCount(architecture, OrDefault(options).NumClasses))
    {
        _options = options ?? new BloombergGPTOptions<T>();
        Options = _options;
        _options.Validate();
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a BloombergGPT built from native layers, which can be trained.
    /// </summary>
    /// <param name="architecture">The neural network architecture; custom layers replace the encoder.</param>
    /// <param name="options">The model options; the paper's defaults when null.</param>
    /// <param name="optimizer">The optimizer; the paper's when null.</param>
    /// <param name="lossFunction">The loss; cross-entropy on logits when null.</param>
    public BloombergGPT(
        NeuralNetworkArchitecture<T> architecture,
        BloombergGPTOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, OrDefault(options).MaxSequenceLength, OrDefault(options).VocabularySize,
            OrDefault(options).HiddenDimension, ResolveClassCount(architecture, OrDefault(options).NumClasses), lossFunction)
    {
        _options = options ?? new BloombergGPTOptions<T>();
        Options = _options;
        _options.Validate();
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        InitializeLayers();
    }

    private static BloombergGPTOptions<T> OrDefault(BloombergGPTOptions<T>? options) => options ?? new BloombergGPTOptions<T>();

    /// <inheritdoc/>
    protected override FinancialLanguageModelOptions<T> LanguageModelOptions => _options;

    /// <inheritdoc/>
    protected override string ReleasedTokenizerDescription => "BloombergGPT's unigram tokenizer (131,072 tokens)";

    /// <inheritdoc/>
    protected override void InitializeLayers()
        => InitializeDecoderLayers(
            pool => LayerHelper<T>.CreateBloomDecoderLayers(
                VocabularySize, MaxSequenceLength, HiddenDimension, _options.NumAttentionHeads,
                _options.IntermediateDimension, _options.NumLayers, _options.DropoutRate, pool),
            NumSentimentClasses,
            _options.DropoutRate);
}

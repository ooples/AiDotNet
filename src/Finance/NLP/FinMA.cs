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
/// FinMA (Financial Multi-Agent) neural network model for collaborative financial task solving.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> FinMA is a financial language model from the PIXIU project
/// that can handle a wide range of financial tasks including sentiment analysis, named
/// entity recognition in SEC filings, stock movement prediction, and financial question
/// answering. It was instruction-tuned on 136K financial task examples, making it a
/// versatile financial AI assistant.</para>
/// </remarks>
/// <example>
/// <code>
/// // Define architecture for instruction-tuned financial multi-task model (2048 tokens)
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.MultiClassClassification,
///     inputHeight: 2048, inputWidth: 1, inputDepth: 1, outputSize: 32000);
///
/// // Training mode: multi-task financial LLM for sentiment, NER, and QA
/// var model = new FinMA&lt;double&gt;(architecture);
///
/// // ONNX inference mode: load pre-trained FinMA model
/// var onnxModel = new FinMA&lt;double&gt;(architecture, "finma.onnx");
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
[ResearchPaper("PIXIU: A Large Language Model, Instruction Data and Evaluation Benchmark for Finance", "https://arxiv.org/abs/2306.05443", Year = 2023, Authors = "Qianqian Xie, Weiguang Han, Xiao Zhang, Yanzhao Lai, Min Peng, Alejandro Lopez-Lira, Jimin Huang")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 8e-6, WeightDecay = 1e-5,
                ReferenceBatchSize = 32,
                Source = "Xie et al. 2023, Sec. 4: fine-tuning based on the AdamW optimizer with a "
                        + "batch size of 32, an initial learning rate of 8e-6 and a weight decay of "
                        + "1e-5. The two variants differ only in epoch count, so those shared values "
                        + "apply to both.")]
public partial class FinMA<T> : FinancialDecoderModelBase<T>
{
    private readonly FinMAOptions<T> _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <summary>
    /// Creates a FinMA that runs a pretrained ONNX model.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="onnxModelPath">Path to the ONNX model.</param>
    /// <param name="options">The model options; the paper's defaults when null.</param>
    /// <param name="optimizer">The optimizer; the paper's when null.</param>
    public FinMA(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        FinMAOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, onnxModelPath, OrDefault(options).MaxSequenceLength, OrDefault(options).VocabularySize,
            OrDefault(options).HiddenDimension, ResolveClassCount(architecture, OrDefault(options).NumClasses))
    {
        _options = options ?? new FinMAOptions<T>();
        Options = _options;
        _options.Validate();
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a FinMA built from native layers, which can be trained.
    /// </summary>
    /// <param name="architecture">The neural network architecture; custom layers replace the encoder.</param>
    /// <param name="options">The model options; the paper's defaults when null.</param>
    /// <param name="optimizer">The optimizer; the paper's when null.</param>
    /// <param name="lossFunction">The loss; cross-entropy on logits when null.</param>
    public FinMA(
        NeuralNetworkArchitecture<T> architecture,
        FinMAOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, OrDefault(options).MaxSequenceLength, OrDefault(options).VocabularySize,
            OrDefault(options).HiddenDimension, ResolveClassCount(architecture, OrDefault(options).NumClasses), lossFunction)
    {
        _options = options ?? new FinMAOptions<T>();
        Options = _options;
        _options.Validate();
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        InitializeLayers();
    }

    private static FinMAOptions<T> OrDefault(FinMAOptions<T>? options) => options ?? new FinMAOptions<T>();

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

    /// <inheritdoc/>
    protected override void TrainCore(Tensor<T> input, Tensor<T> target, Tensor<T> output)
    {
        SetTrainingMode(true);
        _optimizer.UpdateParameters(Layers);
        SetTrainingMode(false);
    }
}

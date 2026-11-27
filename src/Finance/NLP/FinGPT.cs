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
/// FinGPT neural network model for domain-specific financial language generation and analysis.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> FinGPT is an open-source financial language model that can
/// analyze news, generate financial reports, and perform sentiment analysis. Unlike
/// proprietary models like BloombergGPT, FinGPT is designed to be accessible and
/// fine-tunable by anyone. It can be adapted to specific financial tasks like stock
/// prediction from news headlines or risk assessment from annual reports.</para>
/// </remarks>
/// <example>
/// <code>
/// // Define architecture for open-source financial language generation (2048 tokens)
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.MultiClassClassification,
///     inputHeight: 2048, inputWidth: 1, inputDepth: 1, outputSize: 32000);
///
/// // Training mode: fine-tunable decoder for financial text generation and analysis
/// var model = new FinGPT&lt;double&gt;(architecture);
///
/// // ONNX inference mode: load pre-trained FinGPT model
/// var onnxModel = new FinGPT&lt;double&gt;(architecture, "fingpt.onnx");
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
[ResearchPaper("FinGPT: Open-Source Financial Large Language Models", "https://arxiv.org/abs/2306.06031", Year = 2023, Authors = "Hongyang Yang, Xiao-Yang Liu, Christina Dan Wang")]
[PaperOptimizer(OptimizerKind.Unspecified, LearningRate = 2e-4, ReferenceBatchSize = 64,
                Source = "Yang et al. 2023, Sec. 4: the fine-tuning configuration gives a batch size of "
                        + "64, a learning rate of 2e-4 and 3 training epochs over a LoRA adapter of rank "
                        + "8. The optimizer is left unspecified because the paper names none.")]
public partial class FinGPT<T> : FinancialDecoderModelBase<T>
{
    private readonly FinGPTOptions<T> _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <summary>
    /// Creates a FinGPT that runs a pretrained ONNX model.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="onnxModelPath">Path to the ONNX model.</param>
    /// <param name="options">The model options; the paper's defaults when null.</param>
    /// <param name="optimizer">The optimizer; the paper's when null.</param>
    public FinGPT(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        FinGPTOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, onnxModelPath, OrDefault(options).MaxSequenceLength, OrDefault(options).VocabularySize,
            OrDefault(options).HiddenDimension, ResolveClassCount(architecture, OrDefault(options).NumClasses))
    {
        _options = options ?? new FinGPTOptions<T>();
        Options = _options;
        _options.Validate();
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a FinGPT built from native layers, which can be trained.
    /// </summary>
    /// <param name="architecture">The neural network architecture; custom layers replace the encoder.</param>
    /// <param name="options">The model options; the paper's defaults when null.</param>
    /// <param name="optimizer">The optimizer; the paper's when null.</param>
    /// <param name="lossFunction">The loss; cross-entropy on logits when null.</param>
    public FinGPT(
        NeuralNetworkArchitecture<T> architecture,
        FinGPTOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, OrDefault(options).MaxSequenceLength, OrDefault(options).VocabularySize,
            OrDefault(options).HiddenDimension, ResolveClassCount(architecture, OrDefault(options).NumClasses), lossFunction)
    {
        _options = options ?? new FinGPTOptions<T>();
        Options = _options;
        _options.Validate();
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        InitializeLayers();
    }

    private static FinGPTOptions<T> OrDefault(FinGPTOptions<T>? options) => options ?? new FinGPTOptions<T>();

    /// <inheritdoc/>
    protected override FinancialLanguageModelOptions<T> LanguageModelOptions => _options;

    /// <inheritdoc/>
    protected override string ReleasedTokenizerDescription => "the Llama-2 SentencePiece tokenizer (32,000 pieces)";

    /// <inheritdoc/>
    protected override void InitializeLayers()
        => InitializeDecoderLayers(
            pool => LayerHelper<T>.CreateLlamaDecoderLayers(
                VocabularySize, MaxSequenceLength, HiddenDimension, _options.NumAttentionHeads,
                _options.IntermediateDimension, _options.NumLayers, pool),
            NumSentimentClasses,
            _options.DropoutRate);
}

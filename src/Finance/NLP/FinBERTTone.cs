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
/// FinBERT-tone neural network model specialized for financial sentiment analysis.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> FinBERT-Tone is a specialized sentiment analysis model for
/// financial text. Given a sentence from an earnings call, news article, or analyst report,
/// it classifies the sentiment as positive, negative, or neutral. It understands financial
/// context, so "revenue declined less than expected" is correctly identified as positive
/// sentiment despite containing the word "declined".</para>
/// </remarks>
/// <example>
/// <code>
/// // Define architecture for financial tone classification (512 tokens, 3 sentiment classes)
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.MultiClassClassification,
///     inputHeight: 512, inputWidth: 1, inputDepth: 1, outputSize: 3);
///
/// // Training mode: fine-tune BERT for financial tone detection in earnings calls
/// var model = new FinBERTTone&lt;double&gt;(architecture);
///
/// // ONNX inference mode: load pre-trained FinBERT-Tone model
/// var onnxModel = new FinBERTTone&lt;double&gt;(architecture, "finbert_tone.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("FinBERT: A Pretrained Language Model for Financial Communications", "https://arxiv.org/abs/2006.08097", Year = 2020, Authors = "Yi Yang, Mark Christopher Siy UY, Allen Huang")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 2e-5, ReferenceBatchSize = 32,
                Source = "Yang, Uy and Huang 2020, Sec. 5.2: fine-tuning follows the optimization choices of Devlin et al. 2019; Devlin et al. 2019, Sec. A.3, fine-tune BERT with Adam at a learning "
                        + "rate chosen from 5e-5, 3e-5 and 2e-5 with a batch size of 16 or 32; the smallest "
                        + "rate and the larger batch are taken here.")]
public partial class FinBERTTone<T> : FinancialBertModelBase<T>
{
    private readonly FinBERTToneOptions<T> _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <summary>
    /// Creates a FinBERTTone that runs a pretrained ONNX model.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="onnxModelPath">Path to the ONNX model.</param>
    /// <param name="options">The model options; the paper's defaults when null.</param>
    /// <param name="optimizer">The optimizer; the paper's when null.</param>
    public FinBERTTone(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        FinBERTToneOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, onnxModelPath, OrDefault(options).MaxSequenceLength, OrDefault(options).VocabularySize,
            OrDefault(options).HiddenDimension, ResolveClassCount(architecture, OrDefault(options).NumToneClasses))
    {
        _options = options ?? new FinBERTToneOptions<T>();
        Options = _options;
        _options.Validate();
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a FinBERTTone built from native layers, which can be trained.
    /// </summary>
    /// <param name="architecture">The neural network architecture; custom layers replace the encoder.</param>
    /// <param name="options">The model options; the paper's defaults when null.</param>
    /// <param name="optimizer">The optimizer; the paper's when null.</param>
    /// <param name="lossFunction">The loss; cross-entropy on logits when null.</param>
    public FinBERTTone(
        NeuralNetworkArchitecture<T> architecture,
        FinBERTToneOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, OrDefault(options).MaxSequenceLength, OrDefault(options).VocabularySize,
            OrDefault(options).HiddenDimension, ResolveClassCount(architecture, OrDefault(options).NumToneClasses), lossFunction)
    {
        _options = options ?? new FinBERTToneOptions<T>();
        Options = _options;
        _options.Validate();
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        InitializeLayers();
    }

    private static FinBERTToneOptions<T> OrDefault(FinBERTToneOptions<T>? options) => options ?? new FinBERTToneOptions<T>();

    /// <inheritdoc/>
    protected override FinancialBertEncoderOptions<T> BertOptions => _options;

    /// <inheritdoc/>
    protected override void InitializeLayers()
        => InitializeBertLayers(
            _options.NumAttentionHeads, _options.IntermediateDimension, _options.NumLayers, NumSentimentClasses,
            _options.DropoutRate);
}

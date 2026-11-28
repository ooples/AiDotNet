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
/// FinBERT (Financial BERT) model for financial sentiment analysis.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// FinBERT is a BERT model fine-tuned on financial text for sentiment analysis.
/// It understands financial terminology and context to accurately classify
/// sentiment in financial news, SEC filings, earnings calls, and other financial text.
/// </para>
/// <para><b>For Beginners:</b> FinBERT solves a key problem in financial NLP:
///
/// <b>The Key Insight:</b>
/// General-purpose sentiment models often misinterpret financial language. For example,
/// "shares dropped 5% on earnings miss" is clearly negative for the stock, but a general
/// sentiment model might not understand this. FinBERT is trained specifically on financial
/// text to understand such nuances.
///
/// <b>What Problems Does FinBERT Solve?</b>
/// - Sentiment analysis of financial news articles
/// - Processing SEC filings (10-K, 10-Q, 8-K) for sentiment signals
/// - Analyzing earnings call transcripts for tone
/// - Social media sentiment monitoring for trading signals
/// - Document classification in financial contexts
///
/// <b>How FinBERT Works:</b>
/// 1. <b>Tokenization:</b> Text is split into WordPiece tokens (subwords)
/// 2. <b>Embedding:</b> Tokens are converted to dense vectors with position information
/// 3. <b>Transformer:</b> 12 layers of bidirectional self-attention capture context
/// 4. <b>Classification:</b> [CLS] token embedding is used for sentiment prediction
/// 5. <b>Output:</b> Softmax over classes: Positive, Negative, Neutral
///
/// <b>FinBERT Architecture:</b>
/// - Input: [CLS] token1 token2 ... tokenN [SEP]
/// - Embeddings: Token + Position + Segment embeddings (768-dim)
/// - Transformer: 12 layers, 12 heads, 768 hidden dim
/// - Output: 3-class softmax (negative, neutral, positive)
///
/// <b>Key Benefits:</b>
/// - Understands financial language and terminology
/// - Captures context across the entire input sequence
/// - Pre-trained on large financial corpora
/// - State-of-the-art accuracy on financial sentiment benchmarks
/// </para>
/// <para>
/// <b>Reference:</b> Araci, "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models", 2019.
/// https://arxiv.org/abs/1908.10063
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Define architecture for financial sentiment classification (512 tokens, 3 classes: pos/neg/neutral)
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.MultiClassClassification,
///     inputHeight: 512, inputWidth: 1, inputDepth: 1, outputSize: 3);
///
/// // Training mode: fine-tune BERT on financial text for sentiment analysis
/// var model = new FinBERT&lt;double&gt;(architecture);
///
/// // ONNX inference mode: load pre-trained FinBERT model
/// var onnxModel = new FinBERT&lt;double&gt;(architecture, "finbert.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Vector<>))]
[ResearchPaper("FinBERT: Financial Sentiment Analysis with Pre-trained Language Models", "https://arxiv.org/abs/1908.10063", Year = 2019, Authors = "Dogu Araci")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 2e-5,
                Schedule = LearningRateSchedulerType.LinearWarmup, WarmupFraction = 0.2,
                PostWarmupDecay = LinearWarmupScheduler.DecayMode.Linear, MinLearningRate = 0,
                ReferenceBatchSize = 64,
                Source = "Araci 2019, Sec. 4.3.3: learning rate 2e-5, warm-up proportion 0.2, mini-batch size 64, 6 epochs, with the slanted triangular schedule of Howard and Ruder 2018 -- linear warmup then linear decay. The optimizer is not named there; FinBERT fine-tunes BERT, and Devlin et al. 2019 state Adam for fine-tuning (Sec. A.3).")]
public partial class FinBERT<T> : FinancialBertModelBase<T>
{
    private readonly FinBERTOptions<T> _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <summary>
    /// Creates a FinBERT that runs a pretrained ONNX model.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="onnxModelPath">Path to the ONNX model.</param>
    /// <param name="options">The model options; the paper's defaults when null.</param>
    /// <param name="optimizer">The optimizer; the paper's when null.</param>
    // Both constructors carry optional parameters, as their shipped forms did; the required onnxModelPath string keeps
    // every call unambiguous.
#pragma warning disable RS0026
    public FinBERT(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        FinBERTOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, onnxModelPath, OrDefault(options).MaxSequenceLength, OrDefault(options).VocabularySize,
            OrDefault(options).HiddenDimension, ResolveClassCount(architecture, OrDefault(options).NumSentimentClasses))
#pragma warning restore RS0026
    {
        _options = options ?? new FinBERTOptions<T>();
        Options = _options;
        _options.Validate();
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a FinBERT built from native layers, which can be trained.
    /// </summary>
    /// <param name="architecture">The neural network architecture; custom layers replace the encoder.</param>
    /// <param name="options">The model options; the paper's defaults when null.</param>
    /// <param name="optimizer">The optimizer; the paper's when null.</param>
    /// <param name="lossFunction">The loss; cross-entropy on logits when null.</param>
    public FinBERT(
        NeuralNetworkArchitecture<T> architecture,
        FinBERTOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, OrDefault(options).MaxSequenceLength, OrDefault(options).VocabularySize,
            OrDefault(options).HiddenDimension, ResolveClassCount(architecture, OrDefault(options).NumSentimentClasses), lossFunction)
    {
        _options = options ?? new FinBERTOptions<T>();
        Options = _options;
        _options.Validate();
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        InitializeLayers();
    }

    private static FinBERTOptions<T> OrDefault(FinBERTOptions<T>? options) => options ?? new FinBERTOptions<T>();

    /// <inheritdoc/>
    protected override FinancialBertEncoderOptions<T> BertOptions => _options;

    /// <inheritdoc/>
    protected override void InitializeLayers()
        => InitializeBertLayers(
            _options.NumAttentionHeads, _options.IntermediateDimension, _options.NumLayers, NumSentimentClasses,
            _options.DropoutRate);
}

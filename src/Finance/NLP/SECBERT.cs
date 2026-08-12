using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Finance.Interfaces;
using ModelOptions = AiDotNet.Models.Options;
using AiDotNet.Finance.Base;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
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
///     taskType: NeuralNetworkTaskType.Classification,
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
public partial class SECBERT<T> : FinancialNLPModelBase<T>
{
    #region Shared Fields

    private readonly ModelOptions.SECBERTOptions<T> _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private double _dropout;

    /// <inheritdoc/>
    public override AiDotNet.Models.Options.ModelOptions GetOptions() => _options;

    #endregion

    #region Interface Properties

    /// <inheritdoc/>

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a SEC-BERT network using a pretrained ONNX model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the SECBERT model, SECBERT sets up the architecture and options. This prepares the model for training or inference.
    /// </para>
    /// </remarks>
    public SECBERT(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        ModelOptions.SECBERTOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, onnxModelPath, 
               options?.MaxSequenceLength ?? 512, 
               options?.VocabularySize ?? 30522,
               options?.HiddenDimension ?? 768)
    {
        options ??= new ModelOptions.SECBERTOptions<T>();
        _options = options;
        Options = _options;
        options.Validate();

        _dropout = options.DropoutRate;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a SEC-BERT network in native mode for training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the SECBERT model, SECBERT sets up the architecture and options. This prepares the model for training or inference.
    /// </para>
    /// </remarks>
    public SECBERT(
        NeuralNetworkArchitecture<T> architecture,
        ModelOptions.SECBERTOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture,
               options?.MaxSequenceLength ?? 512,
               options?.VocabularySize ?? 30522,
               options?.HiddenDimension ?? 768,
               3, // numSentimentClasses
               lossFunction)
    {
        options ??= new ModelOptions.SECBERTOptions<T>();
        _options = options;
        Options = _options;
        options.Validate();

        _dropout = options.DropoutRate;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Executes InitializeLayers for the SECBERT.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the SECBERT model, InitializeLayers builds and wires up model components. This sets up the SECBERT architecture before use.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else if (UseNativeMode)
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultSECBERTLayers(
                Architecture, MaxSequenceLength, VocabularySize, HiddenDimension,
                12, 12, _dropout)); // Default heads/layers

        }
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <summary>
    /// Executes TrainCore for the SECBERT.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the SECBERT model, TrainCore performs a training step. This updates the SECBERT architecture so it learns from data.
    /// </para>
    /// </remarks>
    protected override void TrainCore(Tensor<T> input, Tensor<T> target, Tensor<T> output)
    {
        SetTrainingMode(true);
        _optimizer.UpdateParameters(Layers);
        SetTrainingMode(false);
    }

    // UpdateParameters re-sliced the flat vector across Layers by hand -- the base walks
    // exactly the same enumeration, so this said nothing the base does not already say.

    /// <summary>
    /// Executes SerializeModelSpecificData for the SECBERT.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the SECBERT model, SerializeModelSpecificData saves or restores model-specific settings. This lets the SECBERT architecture be reused later.
    /// </para>
    /// </remarks>
    protected override void SerializeModelSpecificData(BinaryWriter writer)
    {
        writer.Write(_dropout);
    }

    /// <summary>
    /// Executes DeserializeModelSpecificData for the SECBERT.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the SECBERT model, DeserializeModelSpecificData saves or restores model-specific settings. This lets the SECBERT architecture be reused later.
    /// </para>
    /// </remarks>
    protected override void DeserializeModelSpecificData(BinaryReader reader)
    {
        _dropout = reader.ReadDouble();
    }

    /// <summary>
    /// Executes ForecastNative for the SECBERT.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the SECBERT model, ForecastNative produces predictions from input data. This is the main inference step of the SECBERT architecture.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ForecastNative(Tensor<T> input, double[]? quantiles)
    {
        SetTrainingMode(false);
        var current = input;
        foreach (var layer in Layers) current = layer.Forward(current);
        return current;
    }

    /// <summary>
    /// Executes ValidateInputShape for the SECBERT.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the SECBERT model, ValidateInputShape checks inputs and configuration. This protects the SECBERT architecture from mismatches and errors.
    /// </para>
    /// </remarks>
    protected override void ValidateInputShape(Tensor<T> input)
    {
        if (input.Rank < 2) throw new ArgumentException("Input must be at least 2D.");
    }

    #endregion
}

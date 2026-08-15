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
///     taskType: NeuralNetworkTaskType.Classification,
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
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("PIXIU: A Large Language Model, Instruction Data and Evaluation Benchmark for Finance", "https://arxiv.org/abs/2306.05443", Year = 2023, Authors = "Qianqian Xie, Weiguang Han, Xiao Zhang, Yanzhao Lai, Min Peng, Alejandro Lopez-Lira, Jimin Huang")]
public partial class FinMA<T> : FinancialNLPModelBase<T>
{
    #region Shared Fields

    private readonly ModelOptions.FinMAOptions<T> _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private double _dropout;
    private int _numAgents;

    /// <inheritdoc/>
    public override AiDotNet.Models.Options.ModelOptions GetOptions() => _options;

    #endregion

    #region Interface Properties

    /// <inheritdoc/>

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a FinMA network using a pretrained ONNX model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinMA model, FinMA sets up the architecture and options. This prepares the model for training or inference.
    /// </para>
    /// </remarks>
    public FinMA(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        ModelOptions.FinMAOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, onnxModelPath, 
               options?.MaxSequenceLength ?? 512, 
               options?.VocabularySize ?? 32000,
               options?.HiddenDimension ?? 768)
    {
        options ??= new ModelOptions.FinMAOptions<T>();
        _options = options;
        Options = _options;
        options.Validate();

        _dropout = options.DropoutRate;
        _numAgents = options.NumAgents;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a FinMA network in native mode for training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinMA model, FinMA sets up the architecture and options. This prepares the model for training or inference.
    /// </para>
    /// </remarks>
    public FinMA(
        NeuralNetworkArchitecture<T> architecture,
        ModelOptions.FinMAOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture,
               options?.MaxSequenceLength ?? 512,
               options?.VocabularySize ?? 32000,
               options?.HiddenDimension ?? 768,
               3,
               lossFunction)
    {
        options ??= new ModelOptions.FinMAOptions<T>();
        _options = options;
        Options = _options;
        options.Validate();

        _dropout = options.DropoutRate;
        _numAgents = options.NumAgents;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Executes InitializeLayers for the FinMA.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinMA model, InitializeLayers builds and wires up model components. This sets up the FinMA architecture before use.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultFinMALayers(
                Architecture,
                vocabularySize: VocabularySize,
                maxSequenceLength: MaxSequenceLength,
                hiddenDimension: HiddenDimension,
                numAttentionHeads: _options.NumAttentionHeads,
                intermediateDimension: _options.IntermediateDimension,
                numLayers: _options.NumLayers,
                numClasses: _options.NumClasses,
                dropoutRate: _dropout));

        }
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <summary>
    /// Executes TrainCore for the FinMA.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinMA model, TrainCore performs a training step. This updates the FinMA architecture so it learns from data.
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
    /// Executes CreateNewInstance for the FinMA.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinMA model, CreateNewInstance builds and wires up model components. This sets up the FinMA architecture before use.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var options = new ModelOptions.FinMAOptions<T>
        {
            MaxSequenceLength = MaxSequenceLength,
            NumAgents = _numAgents,
            VocabularySize = VocabularySize,
            HiddenDimension = HiddenDimension
        };
        return new FinMA<T>(Architecture, options, _optimizer, LossFunction);
    }

    /// <summary>
    /// Executes SerializeModelSpecificData for the FinMA.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinMA model, SerializeModelSpecificData saves or restores model-specific settings. This lets the FinMA architecture be reused later.
    /// </para>
    /// </remarks>
    protected override void SerializeModelSpecificData(BinaryWriter writer)
    {
        writer.Write(_dropout);
        writer.Write(_numAgents);
    }

    /// <summary>
    /// Executes DeserializeModelSpecificData for the FinMA.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinMA model, DeserializeModelSpecificData saves or restores model-specific settings. This lets the FinMA architecture be reused later.
    /// </para>
    /// </remarks>
    protected override void DeserializeModelSpecificData(BinaryReader reader)
    {
        _dropout = reader.ReadDouble();
        _numAgents = reader.ReadInt32();
    }

    /// <summary>
    /// Executes ForecastNative for the FinMA.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinMA model, ForecastNative produces predictions from input data. This is the main inference step of the FinMA architecture.
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
    /// Executes ValidateInputShape for the FinMA.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinMA model, ValidateInputShape checks inputs and configuration. This protects the FinMA architecture from mismatches and errors.
    /// </para>
    /// </remarks>
    protected override void ValidateInputShape(Tensor<T> input)
    {
        if (input.Rank < 2) throw new ArgumentException("Input must be at least 2D.");
    }

    #endregion
}

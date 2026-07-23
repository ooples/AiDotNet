using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks.Options;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Implements a Zamba language model: embedding + HybridBlockScheduler (Mamba + shared attention) + RMS norm + LM head.
/// </summary>
/// <remarks>
/// <para>
/// Zamba from Zyphra uses a hybrid architecture where Mamba blocks are the backbone and a single
/// shared attention layer is interleaved at regular intervals. The shared attention weights reduce
/// parameter count while retaining attention's retrieval capabilities.
/// </para>
/// <para><b>For Beginners:</b> Zamba uses mostly Mamba blocks with a single shared attention layer
/// reused at regular intervals, achieving strong quality with fewer parameters.</para>
/// <para><b>Reference:</b> Glorioso et al., "Zamba: A Compact 7B SSM Hybrid Model", 2024.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     InputType.OneDimensional, NeuralNetworkTaskType.TextGeneration,
///     inputSize: 4096, outputSize: 32000);
/// var model = new ZambaLanguageModel&lt;float&gt;(architecture);
/// var tokens = Tensor&lt;float&gt;.Random(new[] { 1, 128 });
/// var logits = model.Predict(tokens);
/// </code>
/// </example>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.RecurrentNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Zamba: A Compact 7B SSM Hybrid Model", "https://arxiv.org/abs/2405.16712", Year = 2024, Authors = "Paolo Glorioso, Quentin Anthony, Yury Tokpanov, James Whittington, Jonathan Pilault, Adam Ibrahim, Beren Millidge")]
public class ZambaLanguageModel<T> : NeuralNetworkBase<T>
{
    private readonly ZambaOptions _options;
    private readonly int _vocabSize;
    private readonly int _modelDimension;
    private readonly int _numLayers;
    private readonly int _stateDimension;
    private readonly int _attentionInterval;
    private readonly int _maxSeqLength;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public override ModelOptions GetOptions() => _options;

    /// <summary>Gets the vocabulary size.</summary>
    public int VocabSize => _vocabSize;

    /// <summary>Gets the model dimension (d_model).</summary>
    public int ModelDimension => _modelDimension;

    /// <summary>Gets the number of Zamba blocks.</summary>
    public int NumLayers => _numLayers;

    #region Constructors

    public ZambaLanguageModel(
        NeuralNetworkArchitecture<T> architecture,
        int vocabSize = 32000,
        int modelDimension = 3712,
        int numLayers = 76,
        int stateDimension = 16,
        int attentionInterval = 6,
        int maxSeqLength = 4096,
        ILossFunction<T>? lossFunction = null,
        ZambaOptions? options = null)
        : base(architecture,
            lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.TextGeneration))
    {
        _options = options ?? new ZambaOptions();
        Options = _options;
        _vocabSize = vocabSize;
        _modelDimension = modelDimension;
        _numLayers = numLayers;
        _stateDimension = stateDimension;
        _attentionInterval = attentionInterval;
        _maxSeqLength = maxSeqLength;
        InitializeLayers();
    }

    #endregion

    #region Initialization

    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateZambaLayers(
                _vocabSize, _modelDimension, _numLayers, _stateDimension, _attentionInterval, _maxSeqLength));
        }
    }

    #endregion

    #region NeuralNetworkBase Overrides

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        // Keep inference on the base funnel so unbatched token sequences are
        // promoted consistently with Train, then squeezed back for the caller.
        // Walking Layers directly made the output shape depend on whether the
        // recurrent blocks had first been materialized by a batched training
        // forward, which also broke trained-model clone parity.
        return base.PredictCore(input);
    }

    public override void UpdateParameters(Vector<T> gradients)
    {
        if (gradients.Length != ParameterCount)
        {
            throw new ArgumentException(
                $"Expected {ParameterCount} gradients, but got {gradients.Length}",
                nameof(gradients));
        }

        var currentParams = GetParameters();
        T learningRate = NumOps.FromDouble(0.001);
        currentParams = Engine.Subtract(currentParams, Engine.Multiply(gradients, learningRate));
        SetParameters(currentParams);
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "Architecture", "Zamba" },
                { "VocabSize", _vocabSize },
                { "ModelDimension", _modelDimension },
                { "NumLayers", _numLayers },
                { "StateDimension", _stateDimension },
                { "AttentionInterval", _attentionInterval },
                { "MaxSeqLength", _maxSeqLength },
                { "LayerCount", Layers.Count }
            },
            ModelData = SerializeForMetadata()
        };
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_vocabSize);
        writer.Write(_modelDimension);
        writer.Write(_numLayers);
        writer.Write(_stateDimension);
        writer.Write(_attentionInterval);
        writer.Write(_maxSeqLength);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new ZambaLanguageModel<T>(
            Architecture, _vocabSize, _modelDimension, _numLayers, _stateDimension,
            _attentionInterval, _maxSeqLength, LossFunction, _options);
    }

    #endregion
}

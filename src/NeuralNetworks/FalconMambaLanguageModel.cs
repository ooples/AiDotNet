using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks.Options;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Implements a Falcon Mamba language model: embedding + N MambaBlock blocks + RMS norm + LM head.
/// </summary>
/// <remarks>
/// <para>
/// Falcon Mamba from TII (Technology Innovation Institute) is a pure Mamba-based language model
/// trained at 7B scale, achieving competitive results with Transformer-based models while
/// maintaining constant memory during generation regardless of sequence length.
/// </para>
/// <para><b>Reference:</b> Zuo et al., "Falcon Mamba: The First Competitive Attention-free 7B Language Model", 2024.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.RecurrentNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ModelPaper("Falcon Mamba: The First Competitive Attention-free 7B Language Model", "https://arxiv.org/abs/2410.05355", Year = 2024, Authors = "Jingwei Zuo, Younes Belkada, Paul Music, Rouven Bauer, Komal Kumar Bein, Yago Gimenez")]
public class FalconMambaLanguageModel<T> : NeuralNetworkBase<T>
{
    private readonly FalconMambaOptions _options;
    private readonly int _vocabSize;
    private readonly int _modelDimension;
    private readonly int _numLayers;
    private readonly int _stateDimension;
    private readonly int _expandFactor;
    private readonly int _maxSeqLength;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public override ModelOptions GetOptions() => _options;

    /// <summary>Gets the vocabulary size.</summary>
    public int VocabSize => _vocabSize;

    /// <summary>Gets the model dimension (d_model).</summary>
    public int ModelDimension => _modelDimension;

    /// <summary>Gets the number of Falcon Mamba blocks.</summary>
    public int NumLayers => _numLayers;

    #region Constructors

    public FalconMambaLanguageModel(
        NeuralNetworkArchitecture<T> architecture,
        int vocabSize = 65024,
        int modelDimension = 256,
        int numLayers = 4,
        int stateDimension = 16,
        int expandFactor = 2,
        int maxSeqLength = 512,
        ILossFunction<T>? lossFunction = null,
        FalconMambaOptions? options = null)
        : base(architecture,
            lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.TextGeneration))
    {
        _options = options ?? new FalconMambaOptions();
        Options = _options;
        _vocabSize = vocabSize;
        _modelDimension = modelDimension;
        _numLayers = numLayers;
        _stateDimension = stateDimension;
        _expandFactor = expandFactor;
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
            Layers.AddRange(LayerHelper<T>.CreateFalconMambaLayers(
                _vocabSize, _modelDimension, _numLayers, _stateDimension, _expandFactor, _maxSeqLength));
        }
    }

    #endregion

    #region NeuralNetworkBase Overrides

    public override Tensor<T> Predict(Tensor<T> input)
    {
        SetTrainingMode(false);
        var output = input;
        for (int i = 0; i < Layers.Count; i++)
        {
            output = Layers[i].Forward(output);
        }
        return output;
    }

    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        SetTrainingMode(true);
        var predictions = Predict(input);
        LastLoss = LossFunction.CalculateLoss(predictions.ToVector(), expectedOutput.ToVector());
        var outputGradients = LossFunction.CalculateDerivative(predictions.ToVector(), expectedOutput.ToVector());
        Backpropagate(Tensor<T>.FromVector(outputGradients));
        var parameterGradients = GetParameterGradients();
        parameterGradients = ClipGradient(parameterGradients);
        UpdateParameters(parameterGradients);
        SetTrainingMode(false);
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
                { "Architecture", "FalconMamba" },
                { "VocabSize", _vocabSize },
                { "ModelDimension", _modelDimension },
                { "NumLayers", _numLayers },
                { "StateDimension", _stateDimension },
                { "ExpandFactor", _expandFactor },
                { "MaxSeqLength", _maxSeqLength },
                { "LayerCount", Layers.Count }
            },
            ModelData = this.Serialize()
        };
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_vocabSize);
        writer.Write(_modelDimension);
        writer.Write(_numLayers);
        writer.Write(_stateDimension);
        writer.Write(_expandFactor);
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
        return new FalconMambaLanguageModel<T>(
            Architecture, _vocabSize, _modelDimension, _numLayers, _stateDimension,
            _expandFactor, _maxSeqLength, LossFunction, _options);
    }

    #endregion
}

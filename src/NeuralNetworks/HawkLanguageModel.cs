using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks.Options;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Implements a Hawk language model: embedding + N pure RGLR blocks + layer norm + LM head.
/// </summary>
/// <remarks>
/// <para>
/// Hawk is the pure-recurrent variant from Google DeepMind (companion to Griffin), using only
/// Real-Gated Linear Recurrence blocks without any attention. This gives strict O(n) complexity
/// and O(1) memory per token during generation.
/// </para>
/// <para><b>For Beginners:</b> Hawk is a pure-recurrent model that uses no attention at all,
/// giving strict O(n) complexity and O(1) memory per token during generation.</para>
/// <para><b>Reference:</b> De et al., "Griffin: Mixing Gated Linear Recurrences with Local Attention", 2024.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.RecurrentNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ModelPaper("Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models", "https://arxiv.org/abs/2402.19427", Year = 2024, Authors = "Soham De, Samuel L. Smith, Anushan Fernando, Aleksandar Botev, George Cristian-Muraru, Albert Gu, Ruba Haroun, Leonard Berrada, Yutian Chen, Srivatsan Srinivasan, Guillaume Desjardins, Arnaud Doucet, David Budden, Yee Whye Teh, Razvan Pascanu, Nando De Freitas, Caglar Gulcehre")]
public class HawkLanguageModel<T> : NeuralNetworkBase<T>
{
    private readonly HawkOptions _options;
    private readonly int _vocabSize;
    private readonly int _modelDimension;
    private readonly int _numLayers;
    private readonly int _maxSeqLength;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public override ModelOptions GetOptions() => _options;

    /// <summary>Gets the vocabulary size.</summary>
    public int VocabSize => _vocabSize;

    /// <summary>Gets the model dimension (d_model).</summary>
    public int ModelDimension => _modelDimension;

    /// <summary>Gets the number of Hawk blocks.</summary>
    public int NumLayers => _numLayers;

    #region Constructors

    public HawkLanguageModel(
        NeuralNetworkArchitecture<T> architecture,
        int vocabSize = 256000,
        int modelDimension = 256,
        int numLayers = 4,
        int maxSeqLength = 512,
        ILossFunction<T>? lossFunction = null,
        HawkOptions? options = null)
        : base(architecture,
            lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.TextGeneration))
    {
        _options = options ?? new HawkOptions();
        Options = _options;
        _vocabSize = vocabSize;
        _modelDimension = modelDimension;
        _numLayers = numLayers;
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
            Layers.AddRange(LayerHelper<T>.CreateHawkLayers(
                _vocabSize, _modelDimension, _numLayers, _maxSeqLength));
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
                { "Architecture", "Hawk" },
                { "VocabSize", _vocabSize },
                { "ModelDimension", _modelDimension },
                { "NumLayers", _numLayers },
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
        writer.Write(_maxSeqLength);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new HawkLanguageModel<T>(
            Architecture, _vocabSize, _modelDimension, _numLayers, _maxSeqLength,
            LossFunction, _options);
    }

    #endregion
}

using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks.Options;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Implements a RecurrentGemma language model: embedding + N RGLR blocks + layer norm + LM head.
/// </summary>
/// <remarks>
/// <para>
/// RecurrentGemma is Google's production recurrent language model based on the Griffin architecture.
/// It uses Real-Gated Linear Recurrence (RGLR) blocks for O(n) complexity and O(1) per-token generation.
/// </para>
/// <para><b>For Beginners:</b> RecurrentGemma is Google's production model that uses recurrence instead of
/// attention, giving O(n) complexity and constant memory per token during text generation.</para>
/// <para><b>Reference:</b> Botev et al., "RecurrentGemma: Moving Past Transformers for Efficient Open Language Models", 2024.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.RecurrentNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ModelPaper("RecurrentGemma: Moving Past Transformers for Efficient Open Language Models", "https://arxiv.org/abs/2404.07839", Year = 2024, Authors = "Aleksandar Botev, Soham De, Samuel L. Smith, Anushan Fernando, George-Cristian Muraru, Ruba Haroun, Leonard Berrada, Razvan Pascanu, Pier Giuseppe Sessa, Robert Dadashi, Leonard Hussenot, Johan Ferret, Sertan Girgin, Olivier Bachem, Alek Andreev, Kathleen Kenealy, Thomas Mesnard, Cassidy Hardin, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Riviere, Mihir Sanjay Kale, Juliette Love, Pouya Tafti, Armand Joulin, Noah Fiedel, Evan Senter, Yutian Chen, Srivatsan Srinivasan, Guillaume Desjardins, David Budden, Arnaud Doucet, Koray Kavukcuoglu, Nando De Freitas")]
public class RecurrentGemmaLanguageModel<T> : NeuralNetworkBase<T>
{
    private readonly RecurrentGemmaOptions _options;
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

    /// <summary>Gets the number of RecurrentGemma blocks.</summary>
    public int NumLayers => _numLayers;

    #region Constructors

    public RecurrentGemmaLanguageModel(
        NeuralNetworkArchitecture<T> architecture,
        int vocabSize = 256000,
        int modelDimension = 256,
        int numLayers = 4,
        int maxSeqLength = 512,
        ILossFunction<T>? lossFunction = null,
        RecurrentGemmaOptions? options = null)
        : base(architecture,
            lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.TextGeneration))
    {
        _options = options ?? new RecurrentGemmaOptions();
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
            Layers.AddRange(LayerHelper<T>.CreateRecurrentGemmaLayers(
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
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "Architecture", "RecurrentGemma" },
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
        return new RecurrentGemmaLanguageModel<T>(
            Architecture, _vocabSize, _modelDimension, _numLayers, _maxSeqLength,
            LossFunction, _options);
    }

    #endregion
}

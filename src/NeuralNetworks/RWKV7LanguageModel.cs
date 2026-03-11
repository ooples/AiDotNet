using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks.Options;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Implements a full RWKV-7 "Goose" language model: token embedding + N RWKV7Blocks + RMS normalization + LM head.
/// </summary>
/// <remarks>
/// <para>
/// RWKV-7 introduces dynamic state evolution with learnable transition matrices,
/// group normalization on WKV output, and SiLU channel mixing for improved training stability.
/// </para>
/// <para><b>For Beginners:</b> RWKV-7 is the latest version of the RWKV architecture that
/// combines the best of RNNs and Transformers, achieving linear-time inference with
/// competitive quality to Transformer models.</para>
/// <para><b>Reference:</b> Peng et al., "RWKV-7 Goose with Expressive Dynamic State Evolution", 2025.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.RecurrentNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ModelPaper("RWKV: Reinventing RNNs for the Transformer Era", "https://arxiv.org/abs/2305.13048", Year = 2023, Authors = "Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman, Huanqi Cao, Xin Cheng, Michael Chung, Matteo Grella, Kranthi Kiran GV, Xuzheng He, Haowen Hou, Przemyslaw Kazienko, Jan Kocon, Jiaming Kong, Bartlomiej Koptyra, Hayden Lau, Krishna Sri Ipsit Mantri, Ferdinand Mom, Atsushi Saito, Xiangru Tang, Bolun Wang, Johan S. Wind, Stanislaw Wozniak, Ruichong Zhang, Zhenyuan Zhang, Qihang Zhao, Peng Zhou, Jian Zhu, Rui-Jie Zhu")]
public class RWKV7LanguageModel<T> : NeuralNetworkBase<T>
{
    private readonly RWKV7Options _options;
    private readonly int _vocabSize;
    private readonly int _modelDimension;
    private readonly int _numLayers;
    private readonly int _numHeads;
    private readonly double _ffnMultiplier;
    private readonly int _maxSeqLength;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public override ModelOptions GetOptions() => _options;

    /// <summary>Gets the vocabulary size.</summary>
    public int VocabSize => _vocabSize;

    /// <summary>Gets the model dimension (d_model).</summary>
    public int ModelDimension => _modelDimension;

    /// <summary>Gets the number of RWKV-7 blocks.</summary>
    public int NumLayers => _numLayers;

    #region Constructors

    public RWKV7LanguageModel(
        NeuralNetworkArchitecture<T> architecture,
        int vocabSize = 65536,
        int modelDimension = 256,
        int numLayers = 4,
        int numHeads = 4,
        double ffnMultiplier = 3.5,
        int maxSeqLength = 512,
        ILossFunction<T>? lossFunction = null,
        RWKV7Options? options = null)
        : base(architecture,
            lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.TextGeneration))
    {
        _options = options ?? new RWKV7Options();
        Options = _options;
        _vocabSize = vocabSize;
        _modelDimension = modelDimension;
        _numLayers = numLayers;
        _numHeads = numHeads;
        _ffnMultiplier = ffnMultiplier;
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
            Layers.AddRange(LayerHelper<T>.CreateRWKV7Layers(
                _vocabSize, _modelDimension, _numLayers, _numHeads, _ffnMultiplier, _maxSeqLength));
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
                { "Architecture", "RWKV-7-Goose" },
                { "VocabSize", _vocabSize },
                { "ModelDimension", _modelDimension },
                { "NumLayers", _numLayers },
                { "NumHeads", _numHeads },
                { "FFNMultiplier", _ffnMultiplier },
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
        writer.Write(_numHeads);
        writer.Write(_ffnMultiplier);
        writer.Write(_maxSeqLength);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadDouble();
        _ = reader.ReadInt32();
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new RWKV7LanguageModel<T>(
            Architecture, _vocabSize, _modelDimension, _numLayers, _numHeads,
            _ffnMultiplier, _maxSeqLength, LossFunction, _options);
    }

    #endregion
}

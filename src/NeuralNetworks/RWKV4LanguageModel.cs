using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks.Options;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Implements a full RWKV-4 language model: token embedding + N RWKVLayer blocks + layer normalization + LM head.
/// </summary>
/// <remarks>
/// <para>
/// This assembles the complete RWKV-4 architecture as described in the original paper:
/// <code>
///   1. Token Embedding: token indices -> dense vectors [batch, seqLen, modelDim]
///   2. N x RWKVLayer: time mixing (WKV attention) + channel mixing (squared ReLU) with residual connections
///   3. Layer Normalization: final normalization
///   4. LM Head: dense projection to vocabulary logits [batch, seqLen, vocabSize]
/// </code>
/// </para>
/// <para>
/// RWKV-4 is the first widely-adopted version of the RWKV architecture. It replaces standard
/// multi-head attention with a linear-complexity WKV (Weighted Key Value) mechanism that uses
/// fixed exponential decay to weight past tokens. The channel mixing sub-layer replaces the
/// standard FFN with a squared ReLU gating mechanism.
/// </para>
/// <para>
/// Key characteristics of RWKV-4:
/// <list type="bullet">
///   <item>Fixed learned time decay per channel (not data-dependent)</item>
///   <item>Single-head WKV attention (no matrix-valued states)</item>
///   <item>Squared ReLU channel mixing: sigmoid(r) * (W_v * max(k, 0)^2)</item>
///   <item>Token shift mixing with fixed learned coefficients</item>
///   <item>O(n) time complexity and O(1) memory per token during generation</item>
/// </list>
/// </para>
/// <para><b>For Beginners:</b> RWKV-4 is a language model that generates text like GPT, but runs
/// much faster because it processes text in linear time instead of quadratic time.
///
/// How it works:
/// 1. Each word is converted to a vector (embedding)
/// 2. Multiple RWKV layers process the vectors, building understanding of context
/// 3. The output is probabilities for what the next word should be
///
/// What makes it special:
/// - Processes text in linear time (twice as long text takes twice as long, not four times)
/// - Uses constant memory per token during generation
/// - First RWKV version to achieve competitive quality with Transformers
///
/// Real-world examples: RWKV-4 models from 169M to 14B parameters.
/// </para>
/// <para>
/// <b>Reference:</b> Peng et al., "RWKV: Reinventing RNNs for the Transformer Era", 2023.
/// https://arxiv.org/abs/2305.13048
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.RecurrentNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ModelPaper("RWKV: Reinventing RNNs for the Transformer Era", "https://arxiv.org/abs/2305.13048", Year = 2023, Authors = "Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman, Huanqi Cao, Xin Cheng, Michael Chung, Matteo Grella, Kranthi Kiran GV, Xuzheng He, Haowen Hou, Przemyslaw Kazienko, Jan Kocon, Jiaming Kong, Bartlomiej Koptyra, Hayden Lau, Krishna Sri Ipsit Mantri, Ferdinand Mom, Atsushi Saito, Xiangru Tang, Bolun Wang, Johan S. Wind, Stanislaw Wozniak, Ruichong Zhang, Zhenyuan Zhang, Qihang Zhao, Peng Zhou, Jian Zhu, Rui-Jie Zhu")]
public class RWKV4LanguageModel<T> : NeuralNetworkBase<T>
{
    private readonly RWKV4Options _options;
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

    /// <summary>Gets the number of RWKV-4 blocks.</summary>
    public int NumLayers => _numLayers;

    #region Constructors

    /// <summary>
    /// Creates an RWKV-4 language model using native library layers.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="vocabSize">
    /// Size of the token vocabulary. Typical: 50277 for RWKV-4 models (using the 20B tokenizer).
    /// <para><b>For Beginners:</b> How many different words/tokens the model knows.</para>
    /// </param>
    /// <param name="modelDimension">
    /// Model dimension (d_model). Default: 256.
    /// <para><b>For Beginners:</b> Width of the hidden representation. RWKV-4 169M uses 768,
    /// 1.5B uses 2048, 7B uses 4096, 14B uses 5120.</para>
    /// </param>
    /// <param name="numLayers">
    /// Number of RWKV layers. Default: 4.
    /// <para><b>For Beginners:</b> Depth of the network. RWKV-4 169M uses 12 layers,
    /// 1.5B uses 24, 7B uses 32, 14B uses 40.</para>
    /// </param>
    /// <param name="maxSeqLength">Maximum sequence length. Default: 512.</param>
    /// <param name="lossFunction">Optional loss function for training. Defaults to cross-entropy for text generation.</param>
    /// <param name="options">Optional RWKV-4 specific options.</param>
    public RWKV4LanguageModel(
        NeuralNetworkArchitecture<T> architecture,
        int vocabSize = 50277,
        int modelDimension = 256,
        int numLayers = 4,
        int maxSeqLength = 512,
        ILossFunction<T>? lossFunction = null,
        RWKV4Options? options = null)
        : base(
            architecture,
            lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.TextGeneration))
    {
        _options = options ?? new RWKV4Options();
        Options = _options;

        if (vocabSize <= 0)
            throw new ArgumentException($"Vocab size ({vocabSize}) must be positive.", nameof(vocabSize));
        if (modelDimension <= 0)
            throw new ArgumentException($"Model dimension ({modelDimension}) must be positive.", nameof(modelDimension));
        if (numLayers <= 0)
            throw new ArgumentException($"Number of layers ({numLayers}) must be positive.", nameof(numLayers));
        if (maxSeqLength <= 0)
            throw new ArgumentException($"Max sequence length ({maxSeqLength}) must be positive.", nameof(maxSeqLength));

        _vocabSize = vocabSize;
        _modelDimension = modelDimension;
        _numLayers = numLayers;
        _maxSeqLength = maxSeqLength;

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateRWKV4Layers(
                _vocabSize, _modelDimension, _numLayers, _maxSeqLength));
        }
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <inheritdoc />
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

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        SetTrainingMode(true);

        // Forward pass
        var predictions = Predict(input);

        // Calculate loss
        var flatPredictions = predictions.ToVector();
        var flatExpected = expectedOutput.ToVector();
        LastLoss = LossFunction.CalculateLoss(flatPredictions, flatExpected);

        // Backward pass through all layers in reverse
        var outputGradients = LossFunction.CalculateDerivative(flatPredictions, flatExpected);
        Backpropagate(Tensor<T>.FromVector(outputGradients));

        // Get parameter gradients and update
        var parameterGradients = GetParameterGradients();
        parameterGradients = ClipGradient(parameterGradients);
        UpdateParameters(parameterGradients);

        SetTrainingMode(false);
    }

    /// <inheritdoc />
    public override void UpdateParameters(Vector<T> gradients)
    {
        int expectedCount = ParameterCount;
        if (gradients.Length != expectedCount)
        {
            throw new ArgumentException(
                $"Expected {expectedCount} gradients, but got {gradients.Length}",
                nameof(gradients));
        }

        var currentParams = GetParameters();
        T learningRate = NumOps.FromDouble(0.001);
        currentParams = Engine.Subtract(currentParams, Engine.Multiply(gradients, learningRate));
        SetParameters(currentParams);
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "Architecture", "RWKV-4" },
                { "VocabSize", _vocabSize },
                { "ModelDimension", _modelDimension },
                { "NumLayers", _numLayers },
                { "MaxSeqLength", _maxSeqLength },
                { "TotalParameters", ParameterCount },
                { "LayerCount", Layers.Count }
            },
            ModelData = this.Serialize()
        };
    }

    /// <inheritdoc />
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_vocabSize);
        writer.Write(_modelDimension);
        writer.Write(_numLayers);
        writer.Write(_maxSeqLength);
    }

    /// <inheritdoc />
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        int vocabSize = reader.ReadInt32();
        int modelDimension = reader.ReadInt32();
        int numLayers = reader.ReadInt32();
        int maxSeqLength = reader.ReadInt32();

        if (vocabSize != _vocabSize || modelDimension != _modelDimension ||
            numLayers != _numLayers || maxSeqLength != _maxSeqLength)
        {
            throw new InvalidOperationException(
                $"Deserialized dimensions (vocab={vocabSize}, dim={modelDimension}, layers={numLayers}, seq={maxSeqLength}) " +
                $"do not match instance (vocab={_vocabSize}, dim={_modelDimension}, layers={_numLayers}, seq={_maxSeqLength}).");
        }
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new RWKV4LanguageModel<T>(
            Architecture, _vocabSize, _modelDimension, _numLayers, _maxSeqLength,
            LossFunction, _options);
    }

    #endregion
}

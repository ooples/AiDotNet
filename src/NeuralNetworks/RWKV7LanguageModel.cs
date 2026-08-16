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
/// <example>
/// <code>
/// var options = new RWKV7Options { VocabSize = 65536, ModelDim = 4096, NumLayers = 32 };
/// var model = new RWKV7LanguageModel&lt;float&gt;(options);
/// var tokens = Tensor&lt;float&gt;.Random(new[] { 1, 128 });
/// var logits = model.Predict(tokens);
/// </code>
/// </example>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.RecurrentNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("RWKV: Reinventing RNNs for the Transformer Era", "https://arxiv.org/abs/2305.13048", Year = 2023, Authors = "Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman, Huanqi Cao, Xin Cheng, Michael Chung, Matteo Grella, Kranthi Kiran GV, Xuzheng He, Haowen Hou, Przemyslaw Kazienko, Jan Kocon, Jiaming Kong, Bartlomiej Koptyra, Hayden Lau, Krishna Sri Ipsit Mantri, Ferdinand Mom, Atsushi Saito, Xiangru Tang, Bolun Wang, Johan S. Wind, Stanislaw Wozniak, Ruichong Zhang, Zhenyuan Zhang, Qihang Zhao, Peng Zhou, Jian Zhu, Rui-Jie Zhu")]
public class RWKV7LanguageModel<T> : TokenLanguageModelLayoutBase<T>
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

    /// <summary>Gets the number of attention heads.</summary>
    public int NumHeads => _numHeads;

    /// <summary>Gets the FFN dimension multiplier.</summary>
    public double FFNMultiplier => _ffnMultiplier;

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
            // RWKV-7's LM head emits RAW LOGITS (DenseLayer with no activation, see
            // LayerHelper.CreateRWKV7Layers), so the loss must be cross-entropy-with-logits (fused
            // log-softmax + NLL, == PyTorch nn.CrossEntropyLoss / RWKV-LM's F.cross_entropy) — the same
            // pairing RWKV4LanguageModel already uses. The TextGeneration DEFAULT is
            // CategoricalCrossEntropy, which expects softmax PROBABILITIES and takes log(predicted):
            // fed un-normalized logits it clamps every non-positive logit to its 1e-7 floor, and
            // TensorClamp has ZERO gradient outside [1e-7, 1], so training receives no signal at all.
            // Measured on the Generated Q-S shard before this fix: loss frozen at exactly
            // 2048*-ln(1e-7) = 33005.70 from step 1 to step 100 with ALL 5,787,136 parameters NaN,
            // under both a dense and a one-hot target. With this pairing the same probe trains
            // normally, matching healthy sibling RWKV4.
            lossFunction ?? new AiDotNet.LossFunctions.CrossEntropyWithLogitsLoss<T>())
    {
        if (vocabSize <= 0) throw new ArgumentException($"Vocabulary size ({vocabSize}) must be positive.", nameof(vocabSize));
        if (modelDimension <= 0) throw new ArgumentException($"Model dimension ({modelDimension}) must be positive.", nameof(modelDimension));
        if (numLayers <= 0) throw new ArgumentException($"Number of layers ({numLayers}) must be positive.", nameof(numLayers));
        if (numHeads <= 0) throw new ArgumentException($"Number of heads ({numHeads}) must be positive.", nameof(numHeads));
        if (modelDimension % numHeads != 0) throw new ArgumentException($"Model dimension ({modelDimension}) must be divisible by number of heads ({numHeads}).", nameof(modelDimension));

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

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        SetTrainingMode(false);
        return Accelerate(input, () =>
        {
            var output = input;
            for (int i = 0; i < Layers.Count; i++)
            {
                output = Layers[i].Forward(output);
            }
            return output;
        });
    }

    // UpdateParameters applied a GRADIENT STEP, but its one-argument form is the value setter and every caller passes values -- the override corrupted the model. Removed under AIDN082.
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
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
            ModelData = SerializeForMetadata()
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

    #endregion
}

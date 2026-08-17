using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks.Options;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Implements a Samba language model: embedding + HybridBlockScheduler (Mamba + sliding window attention) + RMS norm + LM head.
/// </summary>
/// <remarks>
/// <para>
/// Samba from Microsoft Research alternates Mamba blocks with sliding window attention blocks
/// in a regular pattern, combining Mamba's efficient long-range processing with local attention's
/// precise token interactions within a fixed window size.
/// </para>
/// <para><b>For Beginners:</b> Samba alternates between Mamba blocks (for long-range context) and
/// sliding window attention (for precise local interactions) for efficient unlimited-context modeling.</para>
/// <para><b>Reference:</b> Ren et al., "Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling", 2024.</para>
/// </remarks>
/// <example>
/// <code>
/// var options = new SambaOptions { VocabSize = 32000, ModelDim = 2048, NumLayers = 24 };
/// var model = new SambaLanguageModel&lt;float&gt;(options);
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
[ResearchPaper("Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling", "https://arxiv.org/abs/2406.07522", Year = 2024, Authors = "Liliang Ren, Yang Liu, Yadong Lu, Yelong Shen, Chen Liang, Weizhu Chen")]
public partial class SambaLanguageModel<T> : TokenLanguageModelLayoutBase<T>
{
    private readonly SambaOptions _options;
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

    /// <summary>Gets the number of Samba blocks.</summary>
    public int NumLayers => _numLayers;

    #region Constructors

    public SambaLanguageModel(
        NeuralNetworkArchitecture<T> architecture,
        int vocabSize = 32000,
        int modelDimension = 256,
        int numLayers = 8,
        int stateDimension = 16,
        int attentionInterval = 2,
        int maxSeqLength = 512,
        ILossFunction<T>? lossFunction = null,
        SambaOptions? options = null)
        : base(architecture,
            // Samba's LM head emits RAW LOGITS (DenseLayer with no activation, see
            // LayerHelper.CreateSambaLayers), so the loss must be cross-entropy-with-logits (fused
            // log-softmax + NLL, == PyTorch nn.CrossEntropyLoss) — the same pairing
            // RWKV4LanguageModel already uses. The TextGeneration DEFAULT is CategoricalCrossEntropy,
            // which expects softmax PROBABILITIES and takes log(predicted); fed un-normalized logits
            // it drives training in a degenerate direction, because the only way to reduce
            // -sum(target*log(clamp(logit,1e-7,1))) is to push every logit past the clamp CEILING.
            // Measured on the Generated Q-S shard before this fix: the dense-target probe "descended"
            // from 17677.87 to -0.0000 — i.e. it hit log(1)=0 by saturating the clamp rather than by
            // learning a distribution. With this pairing the same probe descends legitimately.
            lossFunction ?? new AiDotNet.LossFunctions.CrossEntropyWithLogitsLoss<T>())
    {
        _options = options ?? new SambaOptions();
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
            Layers.AddRange(LayerHelper<T>.CreateSambaLayers(
                _vocabSize, _modelDimension, _numLayers, _stateDimension, _attentionInterval, _maxSeqLength));
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
                { "Architecture", "Samba" },
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

    #endregion
}

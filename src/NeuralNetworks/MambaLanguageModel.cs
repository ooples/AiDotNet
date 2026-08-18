using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks.Options;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Implements a full Mamba language model: token embedding + N MambaBlocks + RMS normalization + LM head.
/// </summary>
/// <remarks>
/// <para>
/// This assembles the complete Mamba architecture as described in the original paper.
/// Mamba uses selective state spaces with input-dependent gating to achieve linear-time
/// sequence modeling with competitive quality to Transformers.
/// </para>
/// <para><b>For Beginners:</b> Mamba is an efficient alternative to Transformers that processes
/// sequences in linear time instead of quadratic time, making it much faster for long sequences
/// while maintaining competitive quality.</para>
/// <para><b>Reference:</b> Gu and Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2024.</para>
/// </remarks>
/// <example>
/// <code>
/// var options = new MambaOptions { VocabSize = 50280, ModelDim = 2560, NumLayers = 64 };
/// var model = new MambaLanguageModel&lt;float&gt;(options);
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
[ResearchPaper("Mamba: Linear-Time Sequence Modeling with Selective State Spaces", "https://arxiv.org/abs/2312.00752", Year = 2023, Authors = "Albert Gu, Tri Dao")]
public partial class MambaLanguageModel<T> : TokenLanguageModelLayoutBase<T>
{
    private readonly MambaOptions _options;
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

    /// <summary>Gets the number of Mamba blocks.</summary>
    public int NumLayers => _numLayers;

    /// <summary>Gets the SSM state dimension.</summary>
    public int StateDimension => _stateDimension;

    #region Constructors

    public MambaLanguageModel(
        NeuralNetworkArchitecture<T> architecture,
        int vocabSize = 50277,
        int modelDimension = 256,
        int numLayers = 4,
        int stateDimension = 16,
        int expandFactor = 2,
        int maxSeqLength = 512,
        ILossFunction<T>? lossFunction = null,
        MambaOptions? options = null)
        : base(architecture,
            // Mamba's LM head emits RAW LOGITS (DenseLayer with no activation, see
            // LayerHelper.CreateMambaLayers), so the loss must be cross-entropy-with-logits (fused
            // log-softmax + NLL, == PyTorch nn.CrossEntropyLoss) — the same pairing
            // RWKV4LanguageModel already uses, and the pairing the Mamba paper's LM objective assumes.
            // The TextGeneration DEFAULT is CategoricalCrossEntropy, which expects softmax
            // PROBABILITIES and takes log(predicted): feeding it un-normalized logits makes the
            // objective degenerate, because every non-positive logit is clamped to the 1e-7 floor where
            // TensorClamp has ZERO gradient, so those classes get no training signal at all.
            lossFunction ?? new AiDotNet.LossFunctions.CrossEntropyWithLogitsLoss<T>())
    {
        if (vocabSize <= 0) throw new ArgumentException($"Vocabulary size ({vocabSize}) must be positive.", nameof(vocabSize));
        if (modelDimension <= 0) throw new ArgumentException($"Model dimension ({modelDimension}) must be positive.", nameof(modelDimension));
        if (numLayers <= 0) throw new ArgumentException($"Number of layers ({numLayers}) must be positive.", nameof(numLayers));
        if (stateDimension <= 0) throw new ArgumentException($"State dimension ({stateDimension}) must be positive.", nameof(stateDimension));

        _options = options ?? new MambaOptions();
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
            Layers.AddRange(LayerHelper<T>.CreateMambaLayers(
                _vocabSize, _modelDimension, _numLayers, _stateDimension, _expandFactor, _maxSeqLength));
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
                { "Architecture", "Mamba" },
                { "VocabSize", _vocabSize },
                { "ModelDimension", _modelDimension },
                { "NumLayers", _numLayers },
                { "StateDimension", _stateDimension },
                { "ExpandFactor", _expandFactor },
                { "MaxSeqLength", _maxSeqLength },
                { "LayerCount", Layers.Count }
            },
            ModelData = SerializeForMetadata()
        };
    }





    #endregion
}

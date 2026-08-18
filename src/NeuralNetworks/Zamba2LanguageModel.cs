using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks.Options;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Implements a Zamba2 language model: embedding + HybridBlockScheduler (Mamba2 + shared attention with LoRA) + RMS norm + LM head.
/// </summary>
/// <remarks>
/// <para>
/// Zamba2 from Zyphra improves on Zamba by using Mamba2 blocks instead of Mamba1, adding multiple
/// shared attention layers with LoRA adapters for differentiation, and concatenating the original
/// shared attention output with the Mamba block output before each attention invocation.
/// </para>
/// <para><b>For Beginners:</b> Zamba2 upgrades Zamba with Mamba2 blocks and LoRA-adapted shared
/// attention layers for better efficiency and quality.</para>
/// <para><b>Reference:</b> Glorioso et al., "The Zamba2 Suite: Technical Report", 2024.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     InputType.OneDimensional, NeuralNetworkTaskType.TextGeneration,
///     inputSize: 4096, outputSize: 32000);
/// var model = new Zamba2LanguageModel&lt;float&gt;(architecture);
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
[ResearchPaper("The Zamba2 Suite: Technical Report", "https://arxiv.org/abs/2411.15242", Year = 2024, Authors = "Paolo Glorioso, Quentin Anthony, Yury Tokpanov, Anna Golubeva, Vasudev Shyam, James Whittington, Jonathan Pilault, Beren Millidge")]
public partial class Zamba2LanguageModel<T> : TokenLanguageModelLayoutBase<T>
{
    private readonly Zamba2Options _options;
    private readonly int _vocabSize;
    private readonly int _modelDimension;
    private readonly int _numLayers;
    private readonly int _stateDimension;
    private readonly int _numHeads;
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

    /// <summary>Gets the number of Zamba2 blocks.</summary>
    public int NumLayers => _numLayers;

    #region Constructors

    public Zamba2LanguageModel(
        NeuralNetworkArchitecture<T> architecture,
        int vocabSize = 32000,
        int modelDimension = 3584,
        int numLayers = 81,
        int stateDimension = 64,
        int numHeads = 32,
        int attentionInterval = 6,
        int maxSeqLength = 4096,
        ILossFunction<T>? lossFunction = null,
        Zamba2Options? options = null)
        : base(architecture,
            // Zamba-2's LM head emits RAW LOGITS (DenseLayer with no activation, see
            // LayerHelper.CreateZamba2Layers), so the loss must be cross-entropy-with-logits (fused
            // log-softmax + NLL, == PyTorch nn.CrossEntropyLoss) — the same pairing
            // RWKV4LanguageModel already uses. The TextGeneration DEFAULT is CategoricalCrossEntropy,
            // which expects softmax PROBABILITIES and takes log(predicted): feeding it un-normalized
            // logits makes the objective degenerate, because every non-positive logit is clamped to the
            // 1e-7 floor where TensorClamp has ZERO gradient, so those classes never train.
            lossFunction ?? new AiDotNet.LossFunctions.CrossEntropyWithLogitsLoss<T>())
    {
        _options = options ?? new Zamba2Options();
        Options = _options;
        _vocabSize = vocabSize;
        _modelDimension = modelDimension;
        _numLayers = numLayers;
        _stateDimension = stateDimension;
        _numHeads = numHeads;
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
            Layers.AddRange(LayerHelper<T>.CreateZamba2Layers(
                _vocabSize, _modelDimension, _numLayers, _stateDimension, _numHeads, _attentionInterval, _maxSeqLength));
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

    // UpdateParameters applied a GRADIENT STEP, but its one-argument form is the value setter and every caller passes values -- the override corrupted the model. Removed under AIDN082.
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "Architecture", "Zamba2" },
                { "VocabSize", _vocabSize },
                { "ModelDimension", _modelDimension },
                { "NumLayers", _numLayers },
                { "StateDimension", _stateDimension },
                { "NumHeads", _numHeads },
                { "AttentionInterval", _attentionInterval },
                { "MaxSeqLength", _maxSeqLength },
                { "LayerCount", Layers.Count }
            },
            ModelData = SerializeForMetadata()
        };
    }





    #endregion
}

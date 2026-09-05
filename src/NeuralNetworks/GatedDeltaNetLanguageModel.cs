using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Optimizers;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Implements a full Gated DeltaNet language model: token embedding + N GatedDeltaNetLayer blocks + RMS norm + LM head.
/// </summary>
/// <remarks>
/// <para>
/// Gated DeltaNet combines linear attention with gated delta rules for efficient sequence modeling.
/// The delta rule update allows the model to both write new associations and erase old ones in its
/// memory, unlike standard linear attention which can only accumulate.
/// </para>
/// <para><b>For Beginners:</b> Gated DeltaNet improves on Mamba2 by using delta rules that can
/// both add and remove information from memory, leading to better sequence modeling.</para>
/// <para><b>Reference:</b> Yang et al., "Gated Delta Networks: Improving Mamba2 with Delta Rule", 2024.</para>
/// </remarks>
/// <example>
/// <code>
/// var options = new GatedDeltaNetOptions { VocabSize = 32000, ModelDim = 2048, NumLayers = 24, NumHeads = 16 };
/// var model = new GatedDeltaNetLanguageModel&lt;float&gt;(options);
/// var tokens = Tensor&lt;float&gt;.CreateRandom(new[] { 1, 128 });
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
[ResearchPaper("Gated Delta Networks: Improving Mamba2 with Delta Rule", "https://arxiv.org/abs/2412.06464", Year = 2024, Authors = "Songlin Yang, Jan Kautz, Ali Hatamizadeh")]
public partial class GatedDeltaNetLanguageModel<T> : TokenLanguageModelLayoutBase<T>
{
    private readonly GatedDeltaNetOptions _options;
    private readonly int _vocabSize;
    private readonly int _modelDimension;
    private readonly int _numLayers;
    private readonly int _numHeads;
    private readonly int _maxSeqLength;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public override ModelOptions GetOptions() => _options;

    /// <summary>Gets the vocabulary size.</summary>
    public int VocabSize => _vocabSize;

    /// <summary>Gets the model dimension (d_model).</summary>
    public int ModelDimension => _modelDimension;

    /// <summary>Gets the number of Gated DeltaNet blocks.</summary>
    public int NumLayers => _numLayers;

    #region Constructors

    public GatedDeltaNetLanguageModel(
        NeuralNetworkArchitecture<T> architecture,
        int vocabSize = 50277,
        int modelDimension = 256,
        int numLayers = 4,
        int numHeads = 8,
        int maxSeqLength = 512,
        ILossFunction<T>? lossFunction = null,
        GatedDeltaNetOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture,
            lossFunction ?? new AiDotNet.LossFunctions.CrossEntropyWithLogitsLoss<T>())
    {
        _options = options ?? new GatedDeltaNetOptions();
        Options = _options;
        _vocabSize = vocabSize;
        _modelDimension = modelDimension;
        _numLayers = numLayers;
        _numHeads = numHeads;
        _maxSeqLength = maxSeqLength;
        // THE PAPER'S RATE, NOT THE LIBRARY DEFAULT. Constructing AdamWOptimizer with no options
        // silently trained at InitialLearningRate = 1e-3, which is neither the published rate nor
        // something the caller could change short of building the whole optimizer themselves.
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
            });
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
            Layers.AddRange(LayerHelper<T>.CreateGatedDeltaNetLayers(
                _vocabSize, _modelDimension, _numLayers, _numHeads, _maxSeqLength));
        }
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <summary>
    /// Gated DeltaNet carries a per-head fast-weight matrix through a timestep recurrence: each
    /// step applies the delta rule to the state, writing only the difference between the target
    /// value and what the state currently retrieves for that key. That data-dependent loop is not
    /// a static op graph, so it cannot be captured once and safely replayed by the fused
    /// compiled-training plan; the eager tape re-runs the true recurrence every step, so AdamW
    /// receives the real gradients. Same reason as the sibling recurrent models
    /// (<see cref="GriffinLanguageModel{T}"/>, <see cref="HawkLanguageModel{T}"/>) and the same
    /// root cause documented on <c>NeuralNetworkBase.SupportsFusedCompiledTraining</c> (#1643).
    /// This is a structural property of the architecture, not a temporary restriction.
    /// </summary>
    protected override bool SupportsFusedCompiledTraining => false;

    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GetOrCreateBaseOptimizer()
        => _optimizer;

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

    // UpdateParameters validated the length and distributed the vector across Layers. The base does
    // both. Its trailing "did the loop consume the whole vector" guard is not lost either -- it
    // protected against sum(layer.ParameterCount) drifting from ParameterCount, and the base derives
    // the count and the distribution from ONE enumeration, so they cannot drift apart. Removed under
    // AIDN082.

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "Architecture", "GatedDeltaNet" },
                { "VocabSize", _vocabSize },
                { "ModelDimension", _modelDimension },
                { "NumLayers", _numLayers },
                { "NumHeads", _numHeads },
                { "MaxSeqLength", _maxSeqLength },
                { "LayerCount", Layers.Count }
            },
            ModelData = SerializeForMetadata()
        };
    }





    #endregion
}

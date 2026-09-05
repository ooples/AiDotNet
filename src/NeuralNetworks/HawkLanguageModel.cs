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
/// <remarks>
/// <para>
/// <b>Do not override <c>Train</c> to call <c>TrainWithTape</c> directly.</b> Besides the
/// tape/optimizer step, the base entry point performs canonical batch promotion, first-step
/// LSUV, optimizer persistence, OOM recovery, and fused-compiled training where eligible.
/// Bypassing it skipped those contracts and made the unbatched recurrence numerically unstable
/// on its second FP32 update. This model previously carried a <c>Train</c> override whose whole
/// body was <c>base.Train(...)</c> to hold that note; the override added no behavior and is
/// gone, but the reason it existed is recorded here. <see cref="GriffinLanguageModel{T}"/> has
/// the same recurrence structure and no override, which is the shape both models want.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.TextGeneration,
///     inputSize: 2048,
///     outputSize: 256000);
/// var model = new HawkLanguageModel&lt;float&gt;(architecture);
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
[ResearchPaper("Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models", "https://arxiv.org/abs/2402.19427", Year = 2024, Authors = "Soham De, Samuel L. Smith, Anushan Fernando, Aleksandar Botev, George Cristian-Muraru, Albert Gu, Ruba Haroun, Leonard Berrada, Yutian Chen, Srivatsan Srinivasan, Guillaume Desjardins, Arnaud Doucet, David Budden, Yee Whye Teh, Razvan Pascanu, Nando De Freitas, Caglar Gulcehre")]
public partial class HawkLanguageModel<T> : TokenLanguageModelLayoutBase<T>
{
    private readonly HawkOptions _options;
    private readonly int _vocabSize;
    private readonly int _modelDimension;
    private readonly int _recurrenceDimension;
    private readonly int _numLayers;
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

    /// <summary>Gets the number of Hawk blocks.</summary>
    public int NumLayers => _numLayers;

    #region Constructors

    public HawkLanguageModel(
        NeuralNetworkArchitecture<T> architecture,
        int vocabSize = 256000,
        int modelDimension = 2048,
        int numLayers = 24,
        int maxSeqLength = 2048,
        ILossFunction<T>? lossFunction = null,
        HawkOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture,
            // Hawk trains on next-token logits. Keep softmax fused with cross-entropy so
            // very unlikely tokens never create the -target/probability gradient blow-up
            // produced by a separate Softmax + CategoricalCrossEntropy pair.
            lossFunction ?? new AiDotNet.LossFunctions.CrossEntropyWithLogitsLoss<T>())
    {
        _options = options ?? new HawkOptions();
        Options = _options;
        _vocabSize = vocabSize;
        _modelDimension = modelDimension;
        _recurrenceDimension = _options.RecurrenceDimension;
        _numLayers = numLayers;
        _maxSeqLength = maxSeqLength;
        if (_recurrenceDimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "RecurrenceDimension must be positive.");
        _optimizer = optimizer ?? CreateDefaultOptimizer();
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
                _vocabSize, _modelDimension, _numLayers, _maxSeqLength,
                _recurrenceDimension));
        }
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <summary>
    /// Uses the constructor-selected optimizer. Hawk's paper trains with
    /// AdamW; callers can supply any gradient optimizer through the constructor.
    /// </summary>
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GetOrCreateBaseOptimizer()
        => _optimizer;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreateDefaultOptimizer()
        => new AiDotNet.Optimizers.AdamWOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
                WeightDecay = _options.WeightDecay,
                Beta1 = _options.Beta1,
                Beta2 = _options.Beta2,
                Epsilon = _options.Epsilon,
                EnableGradientClipping = _options.EnableGradientClipping,
                MaxGradientNorm = _options.MaxGradientNorm
            });

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
                { "Architecture", "Hawk" },
                { "VocabSize", _vocabSize },
                { "ModelDimension", _modelDimension },
                { "RecurrenceDimension", _recurrenceDimension },
                { "NumLayers", _numLayers },
                { "MaxSeqLength", _maxSeqLength },
                { "LayerCount", Layers.Count }
            },
            ModelData = SerializeForMetadata()
        };
    }





    #endregion
}

using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks.Options;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Implements a Griffin language model: embedding + N RGLR blocks with local attention + layer norm + LM head.
/// </summary>
/// <remarks>
/// <para>
/// Griffin from Google DeepMind combines Real-Gated Linear Recurrence (RGLR) blocks with local
/// sliding window attention for a hybrid architecture achieving near-Transformer quality with
/// sub-quadratic complexity. Every Mth block uses local attention instead of RGLR.
/// </para>
/// <para><b>For Beginners:</b> Griffin combines fast linear recurrence blocks with local sliding window
/// attention for a hybrid model that processes sequences efficiently while maintaining quality.</para>
/// <para><b>Reference:</b> De et al., "Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models", 2024.</para>
/// </remarks>
/// <example>
/// <code>
/// var options = new GriffinOptions { VocabSize = 256000, ModelDim = 2560, NumLayers = 26 };
/// var model = new GriffinLanguageModel&lt;float&gt;(options);
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
[ResearchPaper("Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models", "https://arxiv.org/abs/2402.19427", Year = 2024, Authors = "Soham De, Samuel L. Smith, Anushan Fernando, Aleksandar Botev, George Cristian-Muraru, Albert Gu, Ruba Haroun, Leonard Berrada, Yutian Chen, Srivatsan Srinivasan, Guillaume Desjardins, Arnaud Doucet, David Budden, Yee Whye Teh, Razvan Pascanu, Nando De Freitas, Caglar Gulcehre")]
public class GriffinLanguageModel<T> : NeuralNetworkBase<T>
{
    private readonly GriffinOptions _options;
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

    /// <summary>Gets the number of Griffin blocks.</summary>
    public int NumLayers => _numLayers;

    #region Constructors

    public GriffinLanguageModel(
        NeuralNetworkArchitecture<T> architecture,
        int vocabSize = 256000,
        int modelDimension = 2048,
        int numLayers = 24,
        int maxSeqLength = 2048,
        ILossFunction<T>? lossFunction = null,
        GriffinOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture,
            lossFunction ?? new AiDotNet.LossFunctions.CrossEntropyWithLogitsLoss<T>())
    {
        _options = options ?? new GriffinOptions();
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
            Layers.AddRange(LayerHelper<T>.CreateGriffinLayers(
                _vocabSize, _modelDimension, _numLayers, _maxSeqLength,
                _recurrenceDimension));
        }
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <summary>
    /// Griffin's RG-LRU carries a data-dependent hidden state through a timestep
    /// recurrence. That stateful loop cannot be captured once and safely replayed
    /// by the static fused-training plan; use the eager tape so every step records
    /// the current recurrence and AdamW receives the true finite gradients.
    /// </summary>
    protected override bool SupportsFusedCompiledTraining => false;

    /// <summary>
    /// Uses the constructor-selected optimizer. Griffin's paper trains with
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

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
        {
            throw new ArgumentException(
                $"Expected {ParameterCount} parameters, but got {parameters.Length}",
                nameof(parameters));
        }

        // The optimizer supplies post-update parameter values. Treating this vector as raw
        // gradients applied a second hard-coded SGD step and materialized full-model copies,
        // causing the generated memorization trajectory to explode.
        int offset = 0;
        foreach (var layer in Layers)
        {
            int count = (int)layer.ParameterCount;
            layer.UpdateParameters(parameters.Slice(offset, count));
            offset += count;
        }
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "Architecture", "Griffin" },
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
        return new GriffinLanguageModel<T>(
            Architecture, _vocabSize, _modelDimension, _numLayers, _maxSeqLength,
            LossFunction, new GriffinOptions(_options), optimizer: null);
    }

    #endregion
}

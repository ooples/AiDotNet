using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.ReinforcementLearning.Agents.DecisionTransformer;

/// <summary>
/// The Decision Transformer network of Chen et al. 2021 (section 3): the last K timesteps become 3K tokens (return-to-go,
/// state, action), each embedded by its own linear layer plus a learned per-timestep embedding, layer-normalized, and
/// processed by a causally masked GPT; the hidden state at each state token predicts that timestep's action.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>The input packs each timestep as <c>[return-to-go, state, action, timestep]</c>: <c>[K, 1 + S + A + 1]</c> or
/// <c>[batch, K, 1 + S + A + 1]</c>. The output is one action per timestep, <c>[K, A]</c> or <c>[batch, K, A]</c>: tanh
/// means for continuous actions (the authors' Gym model), logits for discrete ones. As in the authors' code the GPT has
/// no positional embedding of its own (the timestep embedding replaces it) and its MLP uses ReLU (appendix A).</para>
/// </remarks>
// The agent's network, not a model of its own: DecisionTransformerAgent carries the model metadata.
[ModelMetadataExempt]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, BatchOptional = true, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, BatchOptional = true, Direction = TensorLayoutDirection.Output)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-4, WeightDecay = 1e-4,
                ReferenceBatchSize = 64, WarmupSteps = 100000,
                Schedule = LearningRateSchedulerType.LinearWarmup, MaxGradientNorm = 0.25,
                Source = "Chen et al. 2021, appendix A.2 and table 9: AdamW, learning rate 1e-4, weight decay 1e-4, "
                        + "batch size 64, linear warmup over the first 10^5 steps, grad norm clip 0.25.")]
internal sealed partial class DecisionTransformerNetwork<T> : NeuralNetworkBase<T>
{
    private readonly int _stateSize, _actionSize, _embeddingDim, _numLayers, _numHeads, _contextLength, _maxEpisodeLength;
    private readonly double _dropoutRate;
    private readonly bool _continuous;
    private bool _built;

    private DenseLayer<T>? _embedReturn, _embedState, _embedAction, _predictAction;
    private EmbeddingLayer<T>? _embedTimestep;
    private LayerNormalizationLayer<T>? _embedNorm, _finalNorm;
    private DropoutLayer<T>? _embedDropout;
    private readonly List<TransformerEncoderBlock<T>> _blocks = new();

    public DecisionTransformerNetwork(int stateSize, int actionSize, int embeddingDim, int numLayers, int numHeads,
        int contextLength, int maxEpisodeLength, double dropoutRate, bool continuous, int? seed = null)
        : base(new NeuralNetworkArchitecture<T>(
                inputType: InputType.TwoDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                inputHeight: contextLength,
                inputWidth: 1 + stateSize + actionSize + 1,
                outputSize: actionSize) { RandomSeed = seed ?? 42 },
            new MeanSquaredErrorLoss<T>())
    {
        if (embeddingDim % numHeads != 0)
            throw new ArgumentException($"The embedding dimension {embeddingDim} must divide into {numHeads} heads.", nameof(numHeads));
        _stateSize = stateSize;
        _actionSize = actionSize;
        _embeddingDim = embeddingDim;
        _numLayers = numLayers;
        _numHeads = numHeads;
        _contextLength = contextLength;
        _maxEpisodeLength = maxEpisodeLength;
        _dropoutRate = dropoutRate;
        _continuous = continuous;
        InitializeLayers();
    }

    /// <summary>Width of one packed timestep: return-to-go, state, action, timestep.</summary>
    public int StepWidth => 1 + _stateSize + _actionSize + 1;

    /// <summary>The GPT blocks, each a pre-LN causal self-attention block with a ReLU MLP.</summary>
    internal IReadOnlyList<TransformerEncoderBlock<T>> Blocks => _blocks;

    protected override void InitializeLayers()
    {
        // The base constructor calls this before the fields are set; the constructor calls it again.
        if (_embeddingDim == 0 || _built) return;
        _built = true;
        int h = _embeddingDim;
        _embedReturn = new DenseLayer<T>(h, (IActivationFunction<T>)new IdentityActivation<T>());
        _embedState = new DenseLayer<T>(h, (IActivationFunction<T>)new IdentityActivation<T>());
        _embedAction = new DenseLayer<T>(h, (IActivationFunction<T>)new IdentityActivation<T>());
        _embedTimestep = new EmbeddingLayer<T>(_maxEpisodeLength, h);
        _embedNorm = new LayerNormalizationLayer<T>(h);
        _embedDropout = new DropoutLayer<T>(_dropoutRate);
        Layers.Add(_embedReturn);
        Layers.Add(_embedState);
        Layers.Add(_embedAction);
        Layers.Add(_embedTimestep);
        Layers.Add(_embedNorm);
        Layers.Add(_embedDropout);
        for (int i = 0; i < _numLayers; i++)
        {
            var block = new TransformerEncoderBlock<T>(h, _numHeads, 4 * h, _dropoutRate, new ReLUActivation<T>());
            // GPT: every token attends only to itself and earlier tokens.
            if (block.AttentionLayer is MultiHeadAttentionLayer<T> attention)
            {
                attention.UseCausalMask = true;
                attention.ConfigureAttentionDropout(_dropoutRate);
            }
            _blocks.Add(block);
            Layers.Add(block);
        }
        _finalNorm = new LayerNormalizationLayer<T>(h);
        _predictAction = new DenseLayer<T>(_actionSize, (IActivationFunction<T>)new IdentityActivation<T>());
        Layers.Add(_finalNorm);
        Layers.Add(_predictAction);
    }

    /// <inheritdoc />
    /// <remarks>Every layer is sized by the first forward and <see cref="NeuralNetworkBase{T}.Layers"/> is not a chain.</remarks>
    protected override void ResolveLazyLayerShapes()
    {
        Forward(new Tensor<T>(new[] { 1, 1, StepWidth }), training: false);
    }

    protected override Tensor<T> PredictCore(Tensor<T> input) => Forward(input, training: false);

    /// <inheritdoc />
    public override Tensor<T> ForwardForTraining(Tensor<T> input) => Forward(input, training: true);

    private Tensor<T> Forward(Tensor<T> input, bool training)
    {
        if (_embedReturn is null || _embedState is null || _embedAction is null || _embedTimestep is null
            || _embedNorm is null || _embedDropout is null || _finalNorm is null || _predictAction is null)
            throw new InvalidOperationException("The network is not initialized.");
        bool single = input.Rank == 2;
        var x = single ? Engine.Reshape(input, new[] { 1, input.Shape[0], input.Shape[1] }) : input;
        if (x.Rank != 3 || x.Shape[2] != StepWidth)
            throw new ArgumentException($"Expected [batch, K, {StepWidth}] packed timesteps, got [{string.Join(", ", input.Shape)}].", nameof(input));
        int batch = x.Shape[0], k = x.Shape[1], h = _embeddingDim;

        var returns = Engine.TensorSlice(x, new[] { 0, 0, 0 }, new[] { batch, k, 1 });
        var states = Engine.TensorSlice(x, new[] { 0, 0, 1 }, new[] { batch, k, _stateSize });
        var actions = Engine.TensorSlice(x, new[] { 0, 0, 1 + _stateSize }, new[] { batch, k, _actionSize });
        var timesteps = Engine.Reshape(Engine.TensorSlice(x, new[] { 0, 0, StepWidth - 1 }, new[] { batch, k, 1 }), new[] { batch, k });

        // Algorithm 1: each modality's linear embedding plus the per-timestep (not per-token) embedding.
        var time = _embedTimestep.Forward(timesteps);
        var r = Engine.TensorAdd(_embedReturn.Forward(returns), time);
        var s = Engine.TensorAdd(_embedState.Forward(states), time);
        var a = Engine.TensorAdd(_embedAction.Forward(actions), time);

        // Interleave as (R_1, s_1, a_1, ..., R_K, s_K, a_K), then layer-normalize (section 3, "followed by layer
        // normalization").
        var tokens = Engine.Reshape(Engine.TensorStack(new[] { r, s, a }, 2), new[] { batch, 3 * k, h });
        tokens = _embedNorm.Forward(tokens);
        _embedDropout.SetTrainingMode(training);
        tokens = _embedDropout.Forward(tokens);
        foreach (var block in _blocks)
        {
            block.SetTrainingMode(training);
            tokens = block.Forward(tokens);
        }
        tokens = _finalNorm.Forward(tokens);

        // The hidden state of state token s_t predicts a_t.
        var stateTokens = Engine.Reshape(
            Engine.TensorSlice(Engine.Reshape(tokens, new[] { batch, k, 3, h }), new[] { 0, 0, 1, 0 }, new[] { batch, k, 1, h }),
            new[] { batch, k, h });
        var prediction = _predictAction.Forward(stateTokens);
        if (_continuous) prediction = Engine.Tanh(prediction);
        return single ? Engine.Reshape(prediction, new[] { k, _actionSize }) : prediction;
    }

    public override ModelMetadata<T> GetModelMetadata() => new ModelMetadata<T>
    {
        Name = "DecisionTransformer",
        AdditionalInfo = new Dictionary<string, object>
        {
            ["EmbeddingDim"] = _embeddingDim,
            ["NumLayers"] = _numLayers,
            ["NumHeads"] = _numHeads,
            ["ContextLength"] = _contextLength,
        }
    };

    public override IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters)
    {
        var copy = new DecisionTransformerNetwork<T>(_stateSize, _actionSize, _embeddingDim, _numLayers, _numHeads,
            _contextLength, _maxEpisodeLength, _dropoutRate, _continuous);
        copy.ResolveLazyLayerShapes();
        copy.UpdateParameters(parameters);
        return copy;
    }
}

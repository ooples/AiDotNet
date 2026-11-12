using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.ReinforcementLearning.Agents.QMIX;

/// <summary>
/// One step of QMIX's agent network (Rashid et al. 2018, §4 and Fig. 2c; pymarl's RNNAgent): a ReLU layer, a GRU cell,
/// and a linear layer to the action values. Every agent shares it; the agent id is part of its input.
/// </summary>
/// <remarks>
/// <para>
/// Input <c>[B, inputSize + hidden]</c>: the agent's input (its observation, its last action one-hot and its id
/// one-hot) followed by its recurrent state. Output <c>[B, actions + hidden]</c>: the action values, then the new
/// state. Carrying the state in the tensor keeps the network single-input, so it is an ordinary network to train,
/// copy and serialize.
/// </para>
/// <para>
/// The cell is PyTorch's <c>nn.GRUCell</c>, as pymarl uses: <c>r = σ(W_ir x + b_ir + W_hr h + b_hr)</c>,
/// <c>z = σ(W_iz x + b_iz + W_hz h + b_hz)</c>, <c>n = tanh(W_in x + b_in + r ⊙ (W_hn h + b_hn))</c>,
/// <c>h' = (1 − z) ⊙ n + z ⊙ h</c>, the reset gate applied after the recurrent projection (unlike
/// <see cref="GRUCellLayer{T}"/>, TensorFlow's form).
/// </para>
/// </remarks>
[LayerCategory(LayerCategory.Recurrent)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, ChangesShape = true, TestInputShape = "1, 7", TestConstructorArgs = "3, 4, 2")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
internal sealed partial class QmixAgentLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _inputSize, _hidden, _actions;
    private readonly DenseLayer<T> _fc1, _gruInput, _gruHidden, _fc2;

    public override bool SupportsTraining => true;

    public QmixAgentLayer(int inputSize, int hidden, int actions)
        : base(new[] { inputSize + hidden }, new[] { actions + hidden })
    {
        if (inputSize <= 0) throw new ArgumentOutOfRangeException(nameof(inputSize));
        if (hidden <= 0) throw new ArgumentOutOfRangeException(nameof(hidden));
        if (actions <= 0) throw new ArgumentOutOfRangeException(nameof(actions));
        _inputSize = inputSize;
        _hidden = hidden;
        _actions = actions;
        var identity = new IdentityActivation<T>() as IActivationFunction<T>;
        _fc1 = new DenseLayer<T>(hidden, new ReLUActivation<T>() as IActivationFunction<T>);
        _gruInput = new DenseLayer<T>(3 * hidden, identity);
        _gruHidden = new DenseLayer<T>(3 * hidden, identity);
        _fc2 = new DenseLayer<T>(actions, identity);
        RegisterSubLayer(_fc1);
        RegisterSubLayer(_gruInput);
        RegisterSubLayer(_gruHidden);
        RegisterSubLayer(_fc2);
    }

    public int InputSize => _inputSize;
    public int Hidden => _hidden;
    public int Actions => _actions;

    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
        => inputRank == 2
            ? new[]
            {
                new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
                new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_actions + _hidden)),
            }
            : null;

    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank != 2 || input.Shape[1] != _inputSize + _hidden)
            throw new ArgumentException(
                $"Expected [batch, {_inputSize} inputs + {_hidden} state], got [{string.Join(", ", input.Shape.ToArray())}].", nameof(input));
        int batch = input.Shape[0], n = _hidden;
        var x = _fc1.Forward(Engine.TensorSlice(input, new[] { 0, 0 }, new[] { batch, _inputSize }));
        var h = Engine.TensorSlice(input, new[] { 0, _inputSize }, new[] { batch, n });
        var gi = _gruInput.Forward(x);
        var gh = _gruHidden.Forward(h);
        Tensor<T> Gate(Tensor<T> g, int k) => Engine.TensorSlice(g, new[] { 0, k * n }, new[] { batch, n });
        var r = Engine.Sigmoid(Engine.TensorAdd(Gate(gi, 0), Gate(gh, 0)));
        var z = Engine.Sigmoid(Engine.TensorAdd(Gate(gi, 1), Gate(gh, 1)));
        var candidate = Engine.Tanh(Engine.TensorAdd(Gate(gi, 2), Engine.TensorMultiply(r, Gate(gh, 2))));
        // h' = (1 - z) n + z h = n + z (h - n)
        var next = Engine.TensorAdd(candidate, Engine.TensorMultiply(z, Engine.TensorSubtract(h, candidate)));
        return Engine.TensorConcatenate(new[] { _fc2.Forward(next), next }, 1);
    }

    /// <summary>PyTorch's default initialization: every weight and bias uniform on ±1/√fan_in.</summary>
    internal void InitializeLikeTorch(Random random)
    {
        QmixTorch.Initialize(Engine, random, _fc1, _inputSize, _hidden);
        QmixTorch.Initialize(Engine, random, _gruInput, _hidden, 3 * _hidden);
        QmixTorch.Initialize(Engine, random, _gruHidden, _hidden, 3 * _hidden);
        QmixTorch.Initialize(Engine, random, _fc2, _hidden, _actions);
    }

    /// <summary>Loads pymarl's RNNAgent state dict (fc1, rnn = nn.GRUCell, fc2), each tensor flattened row-major.</summary>
    internal void LoadTorch(IReadOnlyDictionary<string, double[]> state)
    {
        TorchParameters.Linear(Engine, _fc1, _inputSize, _hidden, state["fc1.weight"], state["fc1.bias"]);
        TorchParameters.Linear(Engine, _gruInput, _hidden, 3 * _hidden, state["rnn.weight_ih"], state["rnn.bias_ih"]);
        TorchParameters.Linear(Engine, _gruHidden, _hidden, 3 * _hidden, state["rnn.weight_hh"], state["rnn.bias_hh"]);
        TorchParameters.Linear(Engine, _fc2, _hidden, _actions, state["fc2.weight"], state["fc2.bias"]);
    }

    public override void UpdateParameters(T learningRate)
    {
        foreach (var layer in new[] { _fc1, _gruInput, _gruHidden, _fc2 }) layer.UpdateParameters(learningRate);
    }

    public override void ResetState()
    {
    }
}

/// <summary>
/// QMIX's monotonic mixing network (Rashid et al. 2018, §4 and Fig. 2a; pymarl's QMixer): the agents' chosen Q-values
/// mixed by weights that hypernetworks generate from the global state, kept non-negative by an absolute value.
/// </summary>
/// <remarks>
/// <para>
/// Input <c>[B, n + S]</c>: the n agents' Q-values, then the state. <c>W₁ = |hyper_w₁(s)|</c> <c>[n, E]</c>,
/// <c>b₁ = hyper_b₁(s)</c>, <c>hidden = ELU(Q W₁ + b₁)</c>, <c>W₂ = |hyper_w_final(s)|</c> <c>[E, 1]</c>, and the final
/// bias <c>V(s)</c> is a two-layer ReLU network. Output <c>[B, 1]</c>: Q_tot, monotonic in every agent's Q-value.
/// </para>
/// <para>
/// The weight hypernetworks are single linear layers in the paper; <c>hypernetLayers = 2</c> gives pymarl's qmix.yaml
/// (a ReLU hidden layer of <c>hypernetEmbed</c> units).
/// </para>
/// </remarks>
[LayerCategory(LayerCategory.Dense)]
[LayerTask(LayerTask.Projection)]
[LayerProperty(IsTrainable = true, ChangesShape = true, TestInputShape = "1, 5", TestConstructorArgs = "2, 3, 4, 1, 4")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
internal sealed partial class QmixMixingLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _agents, _stateSize, _embed, _hypernetLayers, _hypernetEmbed;
    private readonly DenseLayer<T>[] _hyperW1, _hyperWFinal;
    private readonly DenseLayer<T> _hyperB1, _v1, _v2;

    public override bool SupportsTraining => true;

    public QmixMixingLayer(int agents, int stateSize, int embed, int hypernetLayers, int hypernetEmbed)
        : base(new[] { agents + stateSize }, new[] { 1 })
    {
        if (agents <= 0) throw new ArgumentOutOfRangeException(nameof(agents));
        if (stateSize <= 0) throw new ArgumentOutOfRangeException(nameof(stateSize));
        if (embed <= 0) throw new ArgumentOutOfRangeException(nameof(embed));
        if (hypernetLayers is not (1 or 2))
            throw new ArgumentOutOfRangeException(nameof(hypernetLayers), hypernetLayers, "QMIX's hypernetworks have one or two layers.");
        if (hypernetLayers == 2 && hypernetEmbed <= 0) throw new ArgumentOutOfRangeException(nameof(hypernetEmbed));
        _agents = agents;
        _stateSize = stateSize;
        _embed = embed;
        _hypernetLayers = hypernetLayers;
        _hypernetEmbed = hypernetEmbed;
        var identity = new IdentityActivation<T>() as IActivationFunction<T>;
        var relu = new ReLUActivation<T>() as IActivationFunction<T>;
        DenseLayer<T> Dense(int outputs, IActivationFunction<T> activation)
        {
            var layer = new DenseLayer<T>(outputs, activation);
            RegisterSubLayer(layer);
            return layer;
        }
        _hyperW1 = hypernetLayers == 1
            ? new[] { Dense(embed * agents, identity) }
            : new[] { Dense(hypernetEmbed, relu), Dense(embed * agents, identity) };
        _hyperWFinal = hypernetLayers == 1
            ? new[] { Dense(embed, identity) }
            : new[] { Dense(hypernetEmbed, relu), Dense(embed, identity) };
        _hyperB1 = Dense(embed, identity);
        _v1 = Dense(embed, relu);
        _v2 = Dense(1, identity);
    }

    public int Agents => _agents;
    public int StateSize => _stateSize;
    public int Embed => _embed;
    public int HypernetLayers => _hypernetLayers;

    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
        => inputRank == 2
            ? new[]
            {
                new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
                new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(1)),
            }
            : null;

    private static Tensor<T> Chain(DenseLayer<T>[] layers, Tensor<T> x)
    {
        foreach (var layer in layers) x = layer.Forward(x);
        return x;
    }

    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank != 2 || input.Shape[1] != _agents + _stateSize)
            throw new ArgumentException(
                $"Expected [batch, {_agents} agent Q-values + {_stateSize} state], got [{string.Join(", ", input.Shape.ToArray())}].", nameof(input));
        int batch = input.Shape[0];
        var qs = Engine.Reshape(Engine.TensorSlice(input, new[] { 0, 0 }, new[] { batch, _agents }), new[] { batch, 1, _agents });
        var s = Engine.TensorSlice(input, new[] { 0, _agents }, new[] { batch, _stateSize });
        var w1 = Engine.Reshape(Engine.TensorAbs(Chain(_hyperW1, s)), new[] { batch, _agents, _embed });
        var b1 = Engine.Reshape(_hyperB1.Forward(s), new[] { batch, 1, _embed });
        var hidden = Engine.ELU(Engine.TensorAdd(Engine.TensorBatchMatMul(qs, w1), b1), 1.0);
        var wFinal = Engine.Reshape(Engine.TensorAbs(Chain(_hyperWFinal, s)), new[] { batch, _embed, 1 });
        var v = Engine.Reshape(_v2.Forward(_v1.Forward(s)), new[] { batch, 1, 1 });
        return Engine.Reshape(Engine.TensorAdd(Engine.TensorBatchMatMul(hidden, wFinal), v), new[] { batch, 1 });
    }

    // (layer, fan-in, outputs, pymarl's name) for every linear layer.
    private IEnumerable<(DenseLayer<T> Layer, int Inputs, int Outputs, string Name)> Linears()
    {
        if (_hypernetLayers == 1)
        {
            yield return (_hyperW1[0], _stateSize, _embed * _agents, "hyper_w_1");
            yield return (_hyperWFinal[0], _stateSize, _embed, "hyper_w_final");
        }
        else
        {
            yield return (_hyperW1[0], _stateSize, _hypernetEmbed, "hyper_w_1.0");
            yield return (_hyperW1[1], _hypernetEmbed, _embed * _agents, "hyper_w_1.2");
            yield return (_hyperWFinal[0], _stateSize, _hypernetEmbed, "hyper_w_final.0");
            yield return (_hyperWFinal[1], _hypernetEmbed, _embed, "hyper_w_final.2");
        }
        yield return (_hyperB1, _stateSize, _embed, "hyper_b_1");
        yield return (_v1, _stateSize, _embed, "V.0");
        yield return (_v2, _embed, 1, "V.2");
    }

    /// <summary>PyTorch's default initialization: every weight and bias uniform on ±1/√fan_in.</summary>
    internal void InitializeLikeTorch(Random random)
    {
        foreach (var (layer, inputs, outputs, _) in Linears()) QmixTorch.Initialize(Engine, random, layer, inputs, outputs);
    }

    /// <summary>Loads pymarl's QMixer state dict, each tensor flattened row-major.</summary>
    internal void LoadTorch(IReadOnlyDictionary<string, double[]> state)
    {
        foreach (var (layer, inputs, outputs, name) in Linears())
            TorchParameters.Linear(Engine, layer, inputs, outputs, state[name + ".weight"], state[name + ".bias"]);
    }

    public override void UpdateParameters(T learningRate)
    {
        foreach (var (layer, _, _, _) in Linears()) layer.UpdateParameters(learningRate);
    }

    public override void ResetState()
    {
    }
}

internal static class QmixTorch
{
    /// <summary><c>nn.Linear</c>'s and <c>nn.GRUCell</c>'s default: weight and bias uniform on ±1/√fan_in.</summary>
    public static void Initialize<T>(IEngine engine, Random random, DenseLayer<T> layer, int inputs, int outputs)
    {
        double bound = 1.0 / Math.Sqrt(inputs);
        double Draw() => (random.NextDouble() * 2.0 - 1.0) * bound;
        var weight = new double[outputs * inputs];
        for (int i = 0; i < weight.Length; i++) weight[i] = Draw();
        var bias = new double[outputs];
        for (int i = 0; i < bias.Length; i++) bias[i] = Draw();
        TorchParameters.Linear(engine, layer, inputs, outputs, weight, bias);
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Initialization;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Finance.Forecasting.Foundation;

/// <summary>
/// TimeGrad's trainable graph (Rasul et al. 2021, arXiv:2101.12072): an RNN over the series that conditions a
/// denoising network epsilon_theta(x_k, h_{t-1}, k) on its hidden state.
/// </summary>
/// <remarks>
/// <para>
/// epsilon_theta follows the authors' reference implementation (pytorch-ts <c>EpsilonTheta</c>, after DiffWave,
/// Kong et al. 2021). The target vector x_k (length D) is treated as a one-channel signal: a 1x1 input projection,
/// a sinusoidal diffusion-step embedding passed through two SiLU projections, the RNN state upsampled to length D,
/// then gated residual blocks whose dilated convolutions double their dilation every block within a cycle, each
/// adding a step projection before and a conditioner projection after the dilated convolution. Skip outputs are
/// summed, scaled by 1/sqrt(blocks), projected and mapped back to one channel. The signal is padded circularly by
/// two on entry and the two closing 3-tap convolutions are unpadded, so the output has length D again.
/// </para>
/// <para>
/// The layers are published in a fixed order and every role is bound to a position. A deserialize or clone
/// replaces the owning model's layer instances, so the model calls <see cref="BindTo"/> before each forward.
/// </para>
/// </remarks>
internal sealed class TimeGradNetwork<T>
{
    private const double LeakySlope = 0.4;

    private readonly int _numRnnLayers;
    private readonly int _rnnHidden;
    private readonly double _rnnDropout;
    private readonly int _targetDim;
    private readonly int _residualLayers;
    private readonly int _residualChannels;
    private readonly int _dilationCycle;
    private readonly int _timeEmbeddingDim;
    private readonly int _residualHidden;
    private readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    private readonly List<ILayer<T>> _layers = new();
    private IReadOnlyList<ILayer<T>>? _bindSource;

    private readonly List<ILayer<T>> _encoder = new();
    private Conv1DLayer<T> _inputProjection;
    private DenseLayer<T> _stepProjection1;
    private DenseLayer<T> _stepProjection2;
    private DenseLayer<T> _conditionUpsampler1;
    private DenseLayer<T> _conditionUpsampler2;
    private readonly List<ResidualBlock> _blocks = new();
    private Conv1DLayer<T> _skipProjection;
    private Conv1DLayer<T> _outputProjection;

    /// <summary>Builds the graph for the given hyperparameters.</summary>
    public TimeGradNetwork(
        int numRnnLayers, int rnnHidden, double rnnDropout, int targetDim,
        int residualLayers, int residualChannels, int dilationCycle, int timeEmbeddingDim, int residualHidden)
    {
        if (numRnnLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numRnnLayers));
        if (rnnHidden <= 0) throw new ArgumentOutOfRangeException(nameof(rnnHidden));
        if (double.IsNaN(rnnDropout) || rnnDropout < 0 || rnnDropout >= 1) throw new ArgumentOutOfRangeException(nameof(rnnDropout));
        if (targetDim <= 0) throw new ArgumentOutOfRangeException(nameof(targetDim));
        if (residualLayers <= 0) throw new ArgumentOutOfRangeException(nameof(residualLayers));
        if (residualChannels <= 0) throw new ArgumentOutOfRangeException(nameof(residualChannels));
        if (dilationCycle <= 0) throw new ArgumentOutOfRangeException(nameof(dilationCycle));
        if (timeEmbeddingDim <= 0) throw new ArgumentOutOfRangeException(nameof(timeEmbeddingDim));
        if (residualHidden <= 0) throw new ArgumentOutOfRangeException(nameof(residualHidden));

        _numRnnLayers = numRnnLayers;
        _rnnHidden = rnnHidden;
        _rnnDropout = rnnDropout;
        _targetDim = targetDim;
        _residualLayers = residualLayers;
        _residualChannels = residualChannels;
        _dilationCycle = dilationCycle;
        _timeEmbeddingDim = timeEmbeddingDim;
        _residualHidden = residualHidden;
        Build();
    }

    /// <summary>The graph's layers in their fixed order, for the owning model to publish.</summary>
    public IReadOnlyList<ILayer<T>> Layers => _layers;

    /// <summary>
    /// Points every role at the layer at the same position of <paramref name="layers"/>, refusing a list whose types
    /// or convolution settings differ from this layout. Does nothing when the roles already point there.
    /// </summary>
    public void BindTo(IReadOnlyList<ILayer<T>> layers)
    {
        if (layers is null) throw new ArgumentNullException(nameof(layers));
        bool same = layers.Count == _layers.Count;
        for (int i = 0; same && i < _layers.Count; i++) same = ReferenceEquals(layers[i], _layers[i]);
        if (same) return;
        if (layers.Count != _layers.Count)
            throw new InvalidOperationException(
                $"TimeGrad's graph has {_layers.Count} layers; the supplied list has {layers.Count}.");

        var previous = _layers.ToList();
        _bindSource = layers;
        try
        {
            Build();
        }
        catch
        {
            _bindSource = previous;
            Build();
            throw;
        }
        finally
        {
            _bindSource = null;
        }
    }

    [System.Diagnostics.CodeAnalysis.MemberNotNull(
        nameof(_inputProjection), nameof(_stepProjection1), nameof(_stepProjection2), nameof(_conditionUpsampler1),
        nameof(_conditionUpsampler2), nameof(_skipProjection), nameof(_outputProjection))]
    private void Build()
    {
        _layers.Clear();
        _encoder.Clear();
        _blocks.Clear();

        // History encoder: stacked LSTMs, dropout between them as in a multi-layer torch LSTM.
        for (int i = 0; i < _numRnnLayers; i++)
        {
            _encoder.Add(Add(new LSTMLayer<T>(_rnnHidden)));
            if (_rnnDropout > 0 && i < _numRnnLayers - 1) _encoder.Add(Add(new DropoutLayer<T>(_rnnDropout)));
        }

        _inputProjection = Add(new Conv1DLayer<T>(_residualChannels, 1, padding: 0));
        _stepProjection1 = Add(new DenseLayer<T>(_residualHidden, (IActivationFunction<T>)new SiLUActivation<T>()));
        _stepProjection2 = Add(new DenseLayer<T>(_residualHidden, (IActivationFunction<T>)new SiLUActivation<T>()));
        _conditionUpsampler1 = Add(new DenseLayer<T>(Math.Max(1, _targetDim / 2)));
        _conditionUpsampler2 = Add(new DenseLayer<T>(_targetDim));
        for (int i = 0; i < _residualLayers; i++)
        {
            int dilation = 1 << (i % _dilationCycle);
            _blocks.Add(new ResidualBlock(
                dilation,
                Add(new Conv1DLayer<T>(2 * _residualChannels, 3, dilation: dilation, padding: 0)),
                Add(new DenseLayer<T>(_residualChannels)),
                Add(new Conv1DLayer<T>(2 * _residualChannels, 1, padding: 0)),
                Add(new Conv1DLayer<T>(2 * _residualChannels, 1, padding: 0))));
        }

        _skipProjection = Add(new Conv1DLayer<T>(_residualChannels, 3, padding: 0));
        // Zero-initialised, as in the reference: the untrained network predicts no noise.
        _outputProjection = Add(new Conv1DLayer<T>(1, 3, padding: 0, initializationStrategy: new ZeroInitializationStrategy<T>()));
    }

    private TLayer Add<TLayer>(TLayer layer) where TLayer : class, ILayer<T>
    {
        if (_bindSource is not null)
        {
            int index = _layers.Count;
            if (index >= _bindSource.Count || _bindSource[index] is not TLayer existing)
                throw new InvalidOperationException(
                    $"The layer list does not match TimeGrad's layout at position {index}: expected {typeof(TLayer).Name}, " +
                    $"found {(index < _bindSource.Count ? _bindSource[index].GetType().Name : "the end")}.");
            if (existing is Conv1DLayer<T> boundConvolution && layer is Conv1DLayer<T> layoutConvolution)
                RequireSameConvolution(index, layoutConvolution, boundConvolution);
            layer = existing;
        }

        _layers.Add(layer);
        return layer;
    }

    private static void RequireSameConvolution(int index, Conv1DLayer<T> layout, Conv1DLayer<T> bound)
    {
        var expected = layout.GetMetadata();
        var found = bound.GetMetadata();
        foreach (var key in new[] { "OutputChannels", "KernelSize", "Dilation", "Stride", "Padding", "Groups" })
        {
            expected.TryGetValue(key, out var want);
            found.TryGetValue(key, out var got);
            if (!string.Equals(want, got, StringComparison.Ordinal))
                throw new InvalidOperationException(
                    $"The layer list does not match TimeGrad's layout at position {index}: the Conv1DLayer's {key} is " +
                    $"{got ?? "unset"}, but the layout needs {want ?? "unset"}.");
        }
    }

    /// <summary>
    /// Runs the history encoder over a normalized sequence <c>[B, L, D]</c>, returning the hidden state after every
    /// step, <c>[B, L, rnnHidden]</c>; position t conditions the prediction of step t + 1.
    /// </summary>
    public Tensor<T> EncodeHistory(Tensor<T> sequence)
    {
        if (sequence is null) throw new ArgumentNullException(nameof(sequence));
        if (sequence.Rank != 3 || sequence.Shape[2] != _targetDim)
            throw new ArgumentException(
                $"TimeGrad's encoder takes [B, L, {_targetDim}]; got [{string.Join(", ", sequence.Shape.ToArray())}].",
                nameof(sequence));
        var x = sequence;
        foreach (var layer in _encoder) x = layer.Forward(x);
        return x;
    }

    /// <summary>The number of stacked LSTMs, the length of the state arrays <see cref="AdvanceHistory"/> keeps.</summary>
    public int RecurrentLayerCount => _numRnnLayers;

    /// <summary>
    /// Extends the encoder over <paramref name="steps"/> <c>[B, L, D]</c> from the per-LSTM states in
    /// <paramref name="hidden"/> and <paramref name="cell"/> (null entries start from zeros), updating them in place,
    /// and returns the top layer's state after the last step, <c>[B, rnnHidden]</c>. Reading a history in pieces gives
    /// the same state as reading it at once, so a sampler appends each new value instead of re-reading the context.
    /// </summary>
    public Tensor<T> AdvanceHistory(IEngine engine, Tensor<T> steps, Tensor<T>?[] hidden, Tensor<T>?[] cell)
    {
        if (steps is null) throw new ArgumentNullException(nameof(steps));
        if (hidden is null || cell is null || hidden.Length != _numRnnLayers || cell.Length != _numRnnLayers)
            throw new ArgumentException($"Pass one hidden and one cell state slot per LSTM ({_numRnnLayers}).", nameof(hidden));
        var x = steps;
        int recurrent = 0;
        foreach (var layer in _encoder)
        {
            if (layer is LSTMLayer<T> lstm)
            {
                x = lstm.ForwardFromState(x, hidden[recurrent], cell[recurrent], out var finalHidden, out var finalCell);
                hidden[recurrent] = finalHidden;
                cell[recurrent] = finalCell;
                recurrent++;
            }
            else
            {
                x = layer.Forward(x);
            }
        }

        return engine.Reshape(engine.TensorNarrow(x, 1, x.Shape[1] - 1, 1), new[] { x.Shape[0], x.Shape[2] });
    }

    /// <summary>
    /// The sinusoidal diffusion-step embedding of the reference DiffusionEmbedding, computed with engine operations
    /// from a <c>[N, 1]</c> column of step indices so a compiled replay recomputes it from the replayed steps:
    /// <c>[sin(k 10^(4i/E)), cos(k 10^(4i/E))]</c> for i = 0..E-1, shape <c>[N, 2E]</c>.
    /// </summary>
    public Tensor<T> StepEmbedding(IEngine engine, Tensor<T> steps)
    {
        if (steps is null) throw new ArgumentNullException(nameof(steps));
        int n = steps.Shape[0];
        var frequencies = new Tensor<T>(new[] { 1, _timeEmbeddingDim });
        for (int i = 0; i < _timeEmbeddingDim; i++)
            frequencies[i] = _numOps.FromDouble(Math.Pow(10.0, i * 4.0 / _timeEmbeddingDim));
        var shape = new[] { n, _timeEmbeddingDim };
        var phase = engine.TensorMultiply(
            engine.TensorBroadcastTo(engine.Reshape(steps, new[] { n, 1 }), shape),
            engine.TensorBroadcastTo(frequencies, shape));
        return engine.TensorConcatenate(new[] { engine.TensorSin(phase), engine.TensorCos(phase) }, axis: 1);
    }

    /// <summary>
    /// epsilon_theta(x_k, h, k): predicts the noise in <paramref name="noisy"/> <c>[N, D]</c> given the RNN state
    /// <paramref name="condition"/> <c>[N, rnnHidden]</c> and the step embedding <c>[N, 2E]</c>. Returns <c>[N, D]</c>.
    /// </summary>
    public Tensor<T> PredictNoise(IEngine engine, Tensor<T> noisy, Tensor<T> condition, Tensor<T> stepEmbedding)
    {
        if (noisy is null) throw new ArgumentNullException(nameof(noisy));
        if (condition is null) throw new ArgumentNullException(nameof(condition));
        if (stepEmbedding is null) throw new ArgumentNullException(nameof(stepEmbedding));
        int n = noisy.Shape[0];
        T slope = _numOps.FromDouble(LeakySlope);

        var x = CircularPad(engine, engine.Reshape(noisy, new[] { n, 1, _targetDim }), 2);
        x = engine.TensorLeakyReLU(_inputProjection.Forward(x), slope);
        int length = x.Shape[2];

        var step = _stepProjection2.Forward(_stepProjection1.Forward(stepEmbedding));
        var conditioner = engine.TensorLeakyReLU(_conditionUpsampler1.Forward(condition), slope);
        conditioner = engine.TensorLeakyReLU(_conditionUpsampler2.Forward(conditioner), slope);
        conditioner = CircularPad(engine, engine.Reshape(conditioner, new[] { n, 1, _targetDim }), 2);

        var channelShape = new[] { n, _residualChannels, length };
        T residualScale = _numOps.FromDouble(1.0 / Math.Sqrt(2.0));
        Tensor<T>? skipSum = null;
        foreach (var block in _blocks)
        {
            var stepBias = engine.TensorBroadcastTo(
                engine.Reshape(block.StepProjection.Forward(step), new[] { n, _residualChannels, 1 }), channelShape);
            var y = engine.TensorAdd(x, stepBias);
            y = engine.TensorAdd(
                block.DilatedConvolution.Forward(CircularPad(engine, y, block.Dilation)),
                block.ConditionerProjection.Forward(conditioner));
            var gate = engine.TensorNarrow(y, 1, 0, _residualChannels);
            var filter = engine.TensorNarrow(y, 1, _residualChannels, _residualChannels);
            y = engine.TensorMultiply(engine.TensorSigmoid(gate), engine.TensorTanh(filter));
            y = engine.TensorLeakyReLU(block.OutputProjection.Forward(y), slope);
            x = engine.TensorMultiplyScalar(
                engine.TensorAdd(x, engine.TensorNarrow(y, 1, 0, _residualChannels)), residualScale);
            var skip = engine.TensorNarrow(y, 1, _residualChannels, _residualChannels);
            skipSum = skipSum is null ? skip : engine.TensorAdd(skipSum, skip);
        }

        var summed = engine.TensorMultiplyScalar(
            skipSum ?? throw new InvalidOperationException("TimeGrad's denoiser has no residual blocks."),
            _numOps.FromDouble(1.0 / Math.Sqrt(_blocks.Count)));
        var output = _outputProjection.Forward(engine.TensorLeakyReLU(_skipProjection.Forward(summed), slope));
        return engine.Reshape(output, new[] { n, _targetDim });
    }

    /// <summary>
    /// Pads the last axis of <c>[N, C, L]</c> circularly by <paramref name="padding"/> on each side, as the reference
    /// convolutions' <c>padding_mode="circular"</c> does; works when the padding exceeds L by tiling first.
    /// </summary>
    private static Tensor<T> CircularPad(IEngine engine, Tensor<T> x, int padding)
    {
        if (padding == 0) return x;
        int length = x.Shape[2];
        int wraps = (padding + length - 1) / length;
        var copies = new Tensor<T>[2 * wraps + 1];
        for (int i = 0; i < copies.Length; i++) copies[i] = x;
        return engine.TensorNarrow(engine.TensorConcatenate(copies, axis: 2), 2, wraps * length - padding, length + 2 * padding);
    }

    /// <summary>One gated residual block of epsilon_theta.</summary>
    private sealed class ResidualBlock
    {
        public ResidualBlock(int dilation, Conv1DLayer<T> dilatedConvolution, DenseLayer<T> stepProjection,
            Conv1DLayer<T> conditionerProjection, Conv1DLayer<T> outputProjection)
        {
            Dilation = dilation;
            DilatedConvolution = dilatedConvolution;
            StepProjection = stepProjection;
            ConditionerProjection = conditionerProjection;
            OutputProjection = outputProjection;
        }

        public int Dilation { get; }
        public Conv1DLayer<T> DilatedConvolution { get; }
        public DenseLayer<T> StepProjection { get; }
        public Conv1DLayer<T> ConditionerProjection { get; }
        public Conv1DLayer<T> OutputProjection { get; }
    }
}

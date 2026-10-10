using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>
/// The WaveRNN (Kalchbrenner et al. 2018, §2, Eq. 2, Fig. 1): a single recurrent layer of a GRU variant whose state is
/// split into a coarse and a fine half, reading <c>x_t = [c_{t−1}, f_{t−1}, c_t]</c> (the 8 high and 8 low bits of the
/// 16-bit samples scaled to [−1, 1]) through a masked input matrix — the current coarse value reaches only the fine half —
/// with a dual softmax over the coarse and the fine bits.
/// </summary>
/// <remarks>
/// <para><c>u = σ(R_u h + I_u x)</c>, <c>r = σ(R_r h + I_r x)</c>, <c>e = tanh(r ⊙ (R_e h) + I_e x)</c>,
/// <c>h = u ⊙ h + (1 − u) ⊙ e</c>; <c>P(c) = softmax(O₂ relu(O₁ y_c))</c>, <c>P(f) = softmax(O₄ relu(O₃ y_f))</c>.
/// The mask is structural: the previous sample's coarse and fine values feed every unit; the current coarse value feeds
/// a projection into the fine halves only.</para>
/// <para>The paper conditions the WaveRNN without saying how; the mel spectrogram is upsampled by a transposed
/// convolutional network as in WaveNet and projected into all three gates. O₁ and O₃ keep the halves' width.</para>
/// </remarks>
internal sealed class WaveRnnCore<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly List<LayerBase<T>> _layers = new();
    private readonly int _hidden;
    private readonly int _half;
    private readonly TransposedUpsampler<T> _upsample;
    private readonly NormedConv1DLayer<T> _recurrent;      // R = [R_u; R_r; R_e]: H → 3H
    private readonly NormedConv1DLayer<T> _previousInput;  // I applied to [c_{t−1}, f_{t−1}]: 2 → 3H
    private readonly NormedConv1DLayer<T> _currentInput;   // I applied to c_t, fine halves only: 1 → 3 · H/2
    private readonly NormedConv1DLayer<T> _condition;      // the upsampled mel into the gates: mel → 3H
    private readonly NormedConv1DLayer<T> _o1, _o2, _o3, _o4;

    public WaveRnnCore(IEngine engine, int hidden, int melChannels, int[] upsampleScales, int classes)
    {
        if (hidden % 2 != 0) throw new ArgumentException("The state splits into two halves.", nameof(hidden));
        _engine = engine;
        _hidden = hidden;
        _half = hidden / 2;
        _upsample = new TransposedUpsampler<T>(melChannels, upsampleScales, _layers);
        _recurrent = Linear(hidden, 3 * hidden);
        _previousInput = Linear(2, 3 * hidden);
        _currentInput = Linear(1, 3 * _half);
        _condition = Linear(melChannels, 3 * hidden);
        _o1 = Linear(_half, _half);
        _o2 = Linear(_half, classes);
        _o3 = Linear(_half, _half);
        _o4 = Linear(_half, classes);
    }

    private NormedConv1DLayer<T> Linear(int input, int output)
    {
        var layer = new NormedConv1DLayer<T>(input, output, 1, 1, 1, 1, 0, false, ConvolutionNormalization.None);
        _layers.Add(layer);
        return layer;
    }

    public IReadOnlyList<LayerBase<T>> Layers => _layers;

    /// <summary>The recurrent matrix R, the one the paper prunes (§3).</summary>
    public NormedConv1DLayer<T> Recurrent => _recurrent;

    public int Hidden => _hidden;

    /// <summary>The upsampled mel spectrogram <c>[1, mel, frames · hop]</c>.</summary>
    public Tensor<T> Upsample(Tensor<T> mel) => _upsample.Forward(mel);

    private Tensor<T> Rows(Tensor<T> x, int from, int count) => _engine.TensorSlice(x, new[] { 0, from, 0 }, new[] { 1, count, x.Shape[2] });

    // [1, 3 · H/2, T] fine-half contributions placed into [1, 3H, T] with zeros in the coarse halves.
    private Tensor<T> FineOnly(Tensor<T> fine)
    {
        int t = fine.Shape[2];
        var zeros = new Tensor<T>(new[] { 1, _half, t });
        var parts = new List<Tensor<T>>();
        for (int g = 0; g < 3; g++)
        {
            parts.Add(zeros);
            parts.Add(Rows(fine, g * _half, _half));
        }
        return _engine.TensorConcatenate(parts.ToArray(), 1);
    }

    /// <summary>The input contributions <c>[1, 3H, T]</c> of the previous samples' coarse and fine values
    /// <c>[1, 2, T]</c>, the current coarse values <c>[1, 1, T]</c> (fine halves only) and the condition.</summary>
    public Tensor<T> Inputs(Tensor<T> previous, Tensor<T> currentCoarse, Tensor<T> condition)
        => _engine.TensorAdd(_engine.TensorAdd(_previousInput.Forward(previous), FineOnly(_currentInput.Forward(currentCoarse))),
            _condition.Forward(condition));

    /// <summary>One step of Eq. 2: the next state <c>[1, H, 1]</c> from the state and the input contributions
    /// <c>[1, 3H, 1]</c>.</summary>
    public Tensor<T> Step(Tensor<T> h, Tensor<T> inputs)
    {
        var rh = _recurrent.Forward(h);
        var u = _engine.Sigmoid(_engine.TensorAdd(Rows(rh, 0, _hidden), Rows(inputs, 0, _hidden)));
        var r = _engine.Sigmoid(_engine.TensorAdd(Rows(rh, _hidden, _hidden), Rows(inputs, _hidden, _hidden)));
        var e = _engine.Tanh(_engine.TensorAdd(_engine.TensorMultiply(r, Rows(rh, 2 * _hidden, _hidden)), Rows(inputs, 2 * _hidden, _hidden)));
        var keep = _engine.TensorMultiply(u, h);
        var update = _engine.TensorMultiply(_engine.TensorAddScalar(_engine.TensorNegate(u), NumOps.One), e);
        return _engine.TensorAdd(keep, update);
    }

    /// <summary>The coarse logits <c>[1, classes, T]</c> of states <c>[1, H, T]</c>.</summary>
    public Tensor<T> CoarseLogits(Tensor<T> states) => _o2.Forward(_engine.ReLU(_o1.Forward(Rows(states, 0, _half))));

    /// <summary>The fine logits <c>[1, classes, T]</c> of states <c>[1, H, T]</c>.</summary>
    public Tensor<T> FineLogits(Tensor<T> states) => _o4.Forward(_engine.ReLU(_o3.Forward(Rows(states, _half, _half))));
}

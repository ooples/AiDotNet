using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.FlowDiffusion;

namespace AiDotNet.TextToSpeech.Classic;

/// <summary>
/// ProDiff's spectrogram denoiser f_θ(x_t | t, c) (Huang et al. 2022 §4.4, Fig. 1c, Table 4; reference
/// <c>usr/diff/net.py</c> DiffNet, after DiffSpeech): a 1×1 input convolution with ReLU, a sinusoidal step embedding
/// through Linear–Mish–Linear, and N residual blocks — a kernel-3 convolution of <c>x + step</c> plus a 1×1 projection of
/// the condition, the gate <c>σ(a) ⊙ tanh(b)</c>, a 1×1 output convolution split into residual (<c>(x + r)/√2</c>) and
/// skip — whose skips are summed, scaled by <c>1/√N</c>, projected with ReLU and mapped to mel bins by a zero-initialized
/// 1×1 convolution. Convolutions are Kaiming-normal initialized.
/// </summary>
internal sealed class ProDiffDenoiser<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly List<LayerBase<T>> _layers = new();
    private readonly int _channels;
    private readonly KaimingConv1DLayer<T> _inputProjection;
    private readonly DenseLayer<T> _stepMlp1;
    private readonly DenseLayer<T> _stepMlp2;
    private readonly List<(KaimingConv1DLayer<T> Dilated, DenseLayer<T> Step, KaimingConv1DLayer<T> Condition, KaimingConv1DLayer<T> Output)> _blocks = new();
    private readonly KaimingConv1DLayer<T> _skipProjection;
    private readonly Conv1DLayer<T> _outputProjection;

    public ProDiffDenoiser(IEngine engine, int melChannels, int conditionChannels, int channels, int layers, int dilationCycle)
    {
        _engine = engine;
        _channels = channels;
        IActivationFunction<T> identity = new IdentityActivation<T>();
        _inputProjection = Add(new KaimingConv1DLayer<T>(melChannels, channels, 1, 1, 0, false));
        _stepMlp1 = Add(new DenseLayer<T>(4 * channels, identity));
        _stepMlp2 = Add(new DenseLayer<T>(channels, identity));
        for (int i = 0; i < layers; i++)
        {
            int dilation = 1 << (i % Math.Max(1, dilationCycle));
            _blocks.Add((
                Add(new KaimingConv1DLayer<T>(channels, 2 * channels, 3, 1, dilation, false, dilation)),
                Add(new DenseLayer<T>(channels, identity)),
                Add(new KaimingConv1DLayer<T>(conditionChannels, 2 * channels, 1, 1, 0, false)),
                Add(new KaimingConv1DLayer<T>(channels, 2 * channels, 1, 1, 0, false))));
        }
        _skipProjection = Add(new KaimingConv1DLayer<T>(channels, channels, 1, 1, 0, false));
        // nn.init.zeros_(output_projection.weight): the denoiser starts by predicting a constant.
        _outputProjection = Add(new Conv1DLayer<T>(channels, melChannels, 1, 1, 1, 0, null,
            new AiDotNet.Initialization.ZeroInitializationStrategy<T>()));
    }

    public IReadOnlyList<LayerBase<T>> Layers => _layers;

    private TLayer Add<TLayer>(TLayer layer) where TLayer : LayerBase<T>
    {
        _layers.Add(layer);
        return layer;
    }

    /// <summary>Predicts the clean spectrogram <c>[frames, mel]</c> from <paramref name="x"/> <c>[frames, mel]</c> at
    /// step <paramref name="step"/> given the condition <c>[frames, hidden]</c>.</summary>
    public Tensor<T> Predict(Tensor<T> x, int step, Tensor<T> condition)
    {
        int frames = x.Shape[0];
        var h = _engine.ReLU(_inputProjection.Forward(ChannelsFirst(x)));
        var c = ChannelsFirst(condition);
        var embedding = _stepMlp2.Forward(_engine.Mish(_stepMlp1.Forward(StepEmbedding(step))));   // [1, channels]

        Tensor<T>? skip = null;
        double residualScale = 1.0 / Math.Sqrt(2.0);
        foreach (var (dilated, stepProjection, conditionProjection, output) in _blocks)
        {
            var shift = _engine.TensorTile(_engine.Reshape(stepProjection.Forward(embedding), new[] { 1, _channels, 1 }), new[] { 1, 1, frames });
            var y = _engine.TensorAdd(dilated.Forward(_engine.TensorAdd(h, shift)), conditionProjection.Forward(c));
            var gate = _engine.TensorSlice(y, new[] { 0, 0, 0 }, new[] { 1, _channels, frames });
            var filter = _engine.TensorSlice(y, new[] { 0, _channels, 0 }, new[] { 1, _channels, frames });
            var z = output.Forward(_engine.TensorMultiply(_engine.Sigmoid(gate), _engine.Tanh(filter)));
            h = _engine.TensorMultiplyScalar(_engine.TensorAdd(h, _engine.TensorSlice(z, new[] { 0, 0, 0 }, new[] { 1, _channels, frames })),
                NumOps.FromDouble(residualScale));
            var s = _engine.TensorSlice(z, new[] { 0, _channels, 0 }, new[] { 1, _channels, frames });
            skip = skip is null ? s : _engine.TensorAdd(skip, s);
        }
        var summed = _engine.TensorMultiplyScalar(skip!, NumOps.FromDouble(1.0 / Math.Sqrt(_blocks.Count)));
        var result = _outputProjection.Forward(_engine.ReLU(_skipProjection.Forward(summed)));      // [1, mel, frames]
        return _engine.TensorTranspose(_engine.Reshape(result, new[] { result.Shape[1], frames }));
    }

    private Tensor<T> ChannelsFirst(Tensor<T> rows)
        => _engine.Reshape(_engine.TensorTranspose(rows), new[] { 1, rows.Shape[1], rows.Shape[0] });

    // SinusoidalPosEmb(channels) of the integer step: [sin, cos] over channels / 2 log-spaced frequencies.
    private Tensor<T> StepEmbedding(int step)
    {
        int half = _channels / 2;
        double scale = Math.Log(10000) / (half - 1);
        var embedding = new Tensor<T>(new[] { 1, _channels });
        for (int i = 0; i < half; i++)
        {
            double angle = step * Math.Exp(-scale * i);
            embedding[0, i] = NumOps.FromDouble(Math.Sin(angle));
            embedding[0, half + i] = NumOps.FromDouble(Math.Cos(angle));
        }
        return embedding;
    }
}

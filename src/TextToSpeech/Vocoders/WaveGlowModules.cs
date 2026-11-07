using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>
/// WaveGlow's flow (Prenger et al. 2019, §2, Fig. 1; reference NVIDIA/waveglow <c>glow.py</c>): audio squeezed into
/// groups of samples, then steps of an invertible 1×1 convolution and an affine coupling whose WN network (dilated
/// non-causal convolutions with gated-tanh units, residual and skip connections, weight-normalized) is conditioned on the
/// upsampled mel spectrogram; a few channels leave for the output early every few steps.
/// </summary>
internal sealed class WaveGlowFlow<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly List<LayerBase<T>> _layers = new();
    private readonly int _group;
    private readonly int _earlyEvery;
    private readonly int _earlySize;
    private readonly NormedConv1DLayer<T> _upsample;
    private readonly List<(GroupedInvertibleConvFlowLayer<T> Mix, Coupling Couple)> _steps = new();

    /// <summary>The WN network of one coupling: a weight-normalized 1×1 start, a 1×1 projection of the condition for
    /// every layer, per layer a dilated convolution with a gated-tanh unit and a 1×1 projection to residual and skip
    /// channels (skip only in the last layer), and a zero-initialized 1×1 end to the shift and log-scale.</summary>
    private sealed record Coupling(int Half, int Residual, int Gate, int Skip, NormedConv1DLayer<T> Start, NormedConv1DLayer<T> Condition,
        NormedConv1DLayer<T>[] Dilated, NormedConv1DLayer<T>[] ResidualSkip, NormedConv1DLayer<T> End);

    /// <param name="engine">The tensor engine.</param>
    /// <param name="melChannels">Mel bands (80).</param>
    /// <param name="hop">Samples per mel frame, the upsampler's stride (256).</param>
    /// <param name="upsampleKernel">The upsampler's kernel (1024).</param>
    /// <param name="group">Samples per squeezed vector (8).</param>
    /// <param name="flows">Steps of flow (12).</param>
    /// <param name="earlyEvery">Steps between early outputs (4).</param>
    /// <param name="earlySize">Channels output early each time (2).</param>
    /// <param name="layers">WN layers per coupling (8).</param>
    /// <param name="residual">WN residual channels (512).</param>
    /// <param name="gate">WN gated-unit channels (512).</param>
    /// <param name="skip">WN skip channels (256).</param>
    /// <param name="kernel">WN dilated-convolution taps (3).</param>
    public WaveGlowFlow(IEngine engine, int melChannels, int hop, int upsampleKernel, int group, int flows, int earlyEvery,
        int earlySize, int layers, int residual, int gate, int skip, int kernel)
    {
        if (group % 2 != 0) throw new ArgumentException("The group size must be even.", nameof(group));
        if (kernel % 2 != 1) throw new ArgumentException("The WN kernel must be odd.", nameof(kernel));
        _engine = engine;
        _group = group;
        _earlyEvery = earlyEvery;
        _earlySize = earlySize;
        // ConvTranspose1d(mel, mel, 1024, stride = 256) with PyTorch's default initialization.
        _upsample = Add(new NormedConv1DLayer<T>(melChannels, melChannels, upsampleKernel, hop, 1, 1, 0, true, ConvolutionNormalization.None));
        int remaining = group;
        for (int k = 0; k < flows; k++)
        {
            if (k % earlyEvery == 0 && k > 0)
            {
                remaining -= earlySize;
                if (remaining < 2) throw new ArgumentException("The early outputs leave too few channels for a coupling.");
            }
            var mix = Add(new GroupedInvertibleConvFlowLayer<T>(remaining, remaining));
            int half = remaining / 2;
            var start = Add(WeightNormed(half, residual, 1, 1));
            var condition = Add(WeightNormed(melChannels * group, 2 * gate * layers, 1, 1));
            var dilated = new NormedConv1DLayer<T>[layers];
            var residualSkip = new NormedConv1DLayer<T>[layers];
            for (int i = 0; i < layers; i++)
            {
                dilated[i] = Add(WeightNormed(residual, 2 * gate, kernel, 1 << i));
                residualSkip[i] = Add(WeightNormed(gate, i < layers - 1 ? residual + skip : skip, 1, 1));
            }
            var end = Add(new NormedConv1DLayer<T>(skip, 2 * (remaining - half), 1, 1, 1, 1, 0, false, ConvolutionNormalization.None));
            end.Reinitialize(() => 0.0);
            _steps.Add((mix, new Coupling(half, residual, gate, skip, start, condition, dilated, residualSkip, end)));
        }
        RemainingChannels = remaining;
    }

    /// <summary>The channels left after every early output (4 by default).</summary>
    public int RemainingChannels { get; }

    public IReadOnlyList<LayerBase<T>> Layers => _layers;

    private TLayer Add<TLayer>(TLayer layer) where TLayer : LayerBase<T>
    {
        _layers.Add(layer);
        return layer;
    }

    private static NormedConv1DLayer<T> WeightNormed(int input, int output, int kernel, int dilation)
        => new(input, output, kernel, 1, dilation, 1, (kernel * dilation - dilation) / 2, false, ConvolutionNormalization.Weight);

    private Tensor<T> Slice(Tensor<T> x, int from, int count)
        => _engine.TensorSlice(x, new[] { 0, from, 0 }, new[] { 1, count, x.Shape[2] });

    /// <summary>The upsampled condition <c>[1, mel · group, samples / group]</c> of a mel spectrogram
    /// <c>[1, mel, frames]</c>: the transposed convolution trimmed to <paramref name="samples"/> (the reference trims to
    /// the audio in training and cuts kernel − stride samples at inference: both leave frames · hop), each group of
    /// samples' mel vectors stacked (channel <c>m · group + j</c>).</summary>
    public Tensor<T> Condition(Tensor<T> mel, int samples)
    {
        var up = _upsample.Forward(mel);
        if (up.Shape[2] < samples)
            throw new ArgumentException($"The upsampled mel ({up.Shape[2]} samples) is shorter than the audio ({samples}).");
        int m = mel.Shape[1], t = samples / _group;
        var trimmed = _engine.TensorSlice(up, new[] { 0, 0, 0 }, new[] { 1, m, samples });
        var grouped = _engine.TensorPermute(_engine.Reshape(trimmed, new[] { m, t, _group }), new[] { 0, 2, 1 }).Contiguous();
        return _engine.Reshape(grouped, new[] { 1, m * _group, t });
    }

    /// <summary>The audio <c>[samples]</c> squeezed to <c>[1, group, samples / group]</c> (channel j holds sample
    /// <c>group · i + j</c>).</summary>
    public Tensor<T> Squeeze(Tensor<T> audio)
    {
        int t = audio.Length / _group;
        return _engine.Reshape(_engine.TensorTranspose(_engine.Reshape(audio, new[] { t, _group })), new[] { 1, _group, t });
    }

    /// <summary>The audio <c>[1, 1, samples]</c> of squeezed vectors <c>[1, group, samples / group]</c>.</summary>
    public Tensor<T> Unsqueeze(Tensor<T> x)
    {
        int t = x.Shape[2];
        return _engine.Reshape(_engine.TensorTranspose(_engine.Reshape(x, new[] { _group, t })), new[] { 1, 1, _group * t });
    }

    // WN(x_a, condition): the shift (first half of End's output) and log-scale (second half).
    private (Tensor<T> Shift, Tensor<T> LogScale) Network(Coupling c, Tensor<T> xa, Tensor<T> condition, int otherChannels)
    {
        var audio = c.Start.Forward(xa);
        var spect = c.Condition.Forward(condition);
        Tensor<T>? output = null;
        int layers = c.Dilated.Length;
        for (int i = 0; i < layers; i++)
        {
            var a = _engine.TensorAdd(c.Dilated[i].Forward(audio), Slice(spect, i * 2 * c.Gate, 2 * c.Gate));
            var acts = _engine.TensorMultiply(_engine.Tanh(Slice(a, 0, c.Gate)), _engine.Sigmoid(Slice(a, c.Gate, c.Gate)));
            var rs = c.ResidualSkip[i].Forward(acts);
            Tensor<T> skip;
            if (i < layers - 1)
            {
                audio = _engine.TensorAdd(audio, Slice(rs, 0, c.Residual));
                skip = Slice(rs, c.Residual, c.Skip);
            }
            else
            {
                skip = rs;
            }
            output = output is null ? skip : _engine.TensorAdd(output, skip);
        }
        var stats = c.End.Forward(output!);
        return (Slice(stats, 0, otherChannels), Slice(stats, otherChannels, otherChannels));
    }

    /// <summary>The latent z <c>[1, group, T]</c> (early outputs first, as the reference concatenates them) and the
    /// log-determinant terms Σ log s + Σ log|det W| of squeezed audio <c>[1, group, T]</c> under the condition.</summary>
    public (Tensor<T> Z, Tensor<T> LogDeterminant) Forward(Tensor<T> x, Tensor<T> condition)
    {
        var outputs = new List<Tensor<T>>();
        Tensor<T>? logDet = null;
        for (int k = 0; k < _steps.Count; k++)
        {
            if (k % _earlyEvery == 0 && k > 0)
            {
                outputs.Add(Slice(x, 0, _earlySize));
                x = Slice(x, _earlySize, x.Shape[1] - _earlySize);
            }
            var (mix, couple) = _steps[k];
            var (mixed, mixLogDet) = mix.Transform(x, reverse: false);
            int other = mixed.Shape[1] - couple.Half;
            var xa = Slice(mixed, 0, couple.Half);
            var xb = Slice(mixed, couple.Half, other);
            var (shift, logScale) = Network(couple, xa, condition, other);
            xb = _engine.TensorAdd(_engine.TensorMultiply(_engine.TensorExp(logScale), xb), shift);
            x = _engine.TensorConcatenate(new[] { xa, xb }, 1);
            var stepLogDet = _engine.TensorAdd(mixLogDet!, _engine.ReduceSum(logScale, new[] { 0, 1, 2 }, keepDims: false));
            logDet = logDet is null ? stepLogDet : _engine.TensorAdd(logDet, stepLogDet);
        }
        outputs.Add(x);
        return (_engine.TensorConcatenate(outputs.ToArray(), 1), logDet!);
    }

    /// <summary>The squeezed audio <c>[1, group, T]</c> of the latent drawn by <paramref name="draw"/> (a standard-normal
    /// tensor of the requested shape, scaled by σ by the caller) under the condition.</summary>
    public Tensor<T> Inverse(Func<int[], Tensor<T>> draw, Tensor<T> condition)
    {
        int t = condition.Shape[2];
        var x = draw(new[] { 1, RemainingChannels, t });
        for (int k = _steps.Count - 1; k >= 0; k--)
        {
            var (mix, couple) = _steps[k];
            int other = x.Shape[1] - couple.Half;
            var xa = Slice(x, 0, couple.Half);
            var xb = Slice(x, couple.Half, other);
            var (shift, logScale) = Network(couple, xa, condition, other);
            xb = _engine.TensorMultiply(_engine.TensorSubtract(xb, shift), _engine.TensorExp(_engine.TensorNegate(logScale)));
            x = mix.Transform(_engine.TensorConcatenate(new[] { xa, xb }, 1), reverse: true).Output;
            if (k % _earlyEvery == 0 && k > 0)
                x = _engine.TensorConcatenate(new[] { draw(new[] { 1, _earlySize, t }), x }, 1);
        }
        return x;
    }
}

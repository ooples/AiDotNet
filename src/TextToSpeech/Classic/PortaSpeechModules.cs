using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Shared helpers for PortaSpeech's channels-first <c>[1, channels, time]</c> modules.</summary>
internal static class PortaSpeechOps
{
    /// <summary>LayerNorm over channels of <c>[1, C, T]</c>.</summary>
    public static Tensor<T> ChannelNorm<T>(IEngine engine, LayerNormalizationLayer<T> norm, Tensor<T> x)
    {
        int channels = x.Shape[1], time = x.Shape[2];
        var rows = norm.Forward(engine.TensorTranspose(engine.Reshape(x, new[] { channels, time })));
        return engine.Reshape(engine.TensorTranspose(rows), new[] { 1, channels, time });
    }

    /// <summary><c>[T, C]</c> → <c>[1, C, T]</c>.</summary>
    public static Tensor<T> ChannelsFirst<T>(IEngine engine, Tensor<T> rows)
        => engine.Reshape(engine.TensorTranspose(rows), new[] { 1, rows.Shape[1], rows.Shape[0] });

    /// <summary><c>[1, C, T]</c> → <c>[T, C]</c>.</summary>
    public static Tensor<T> Rows<T>(IEngine engine, Tensor<T> x)
        => engine.TensorTranspose(engine.Reshape(x, new[] { x.Shape[1], x.Shape[2] }));

    public static Tensor<T> Slice<T>(IEngine engine, Tensor<T> x, int start, int count)
        => engine.TensorSlice(x, new[] { 0, start, 0 }, new[] { 1, count, x.Shape[2] });
}

/// <summary>
/// The non-causal WaveNet of PortaSpeech's reference (<c>modules/commons/wavenet.py</c>, <c>WN</c>): per layer a
/// weight-normalized convolution to <c>2h</c> channels plus that layer's slice of a weight-normalized 1×1 condition
/// projection, the gate <c>tanh(a) ⊙ σ(b)</c>, and a weight-normalized 1×1 residual/skip convolution (the last layer
/// skip only); the output is the sum of the skips.
/// </summary>
/// <remarks>A stack built with <c>shareFrom</c> reuses that stack's gated and residual/skip convolutions (the
/// post-net's grouped parameter sharing, Ren et al. 2021 §3.3) and owns only its condition projection.</remarks>
internal sealed class PortaSpeechWaveNet<T>
{
    private readonly IEngine _engine;
    private readonly int _hidden;
    private readonly int _layers;
    private readonly List<WeightNormConv1DLayer<T>> _inLayers;
    private readonly List<WeightNormConv1DLayer<T>> _resSkipLayers;
    private readonly WeightNormConv1DLayer<T>? _condition;

    public PortaSpeechWaveNet(IEngine engine, List<LayerBase<T>> owned, int hidden, int kernelSize, int layers,
        int conditionChannels, PortaSpeechWaveNet<T>? shareFrom = null)
    {
        _engine = engine;
        _hidden = hidden;
        _layers = layers;
        if (shareFrom is not null)
        {
            _inLayers = shareFrom._inLayers;
            _resSkipLayers = shareFrom._resSkipLayers;
        }
        else
        {
            _inLayers = new List<WeightNormConv1DLayer<T>>();
            _resSkipLayers = new List<WeightNormConv1DLayer<T>>();
            for (int i = 0; i < layers; i++)
            {
                _inLayers.Add(Own(owned, new WeightNormConv1DLayer<T>(hidden, 2 * hidden, kernelSize, 1, kernelSize / 2)));
                _resSkipLayers.Add(Own(owned, new WeightNormConv1DLayer<T>(hidden, i < layers - 1 ? 2 * hidden : hidden, 1, 1, 0)));
            }
        }
        if (conditionChannels > 0)
            _condition = Own(owned, new WeightNormConv1DLayer<T>(conditionChannels, 2 * hidden * layers, 1, 1, 0));
    }

    private static TLayer Own<TLayer>(List<LayerBase<T>> owned, TLayer layer) where TLayer : LayerBase<T>
    {
        owned.Add(layer);
        return layer;
    }

    /// <summary>Runs the stack on <paramref name="x"/> <c>[1, h, T]</c> with an optional condition <c>[1, c, T]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> x, Tensor<T>? condition)
    {
        var projected = condition is not null && _condition is not null ? _condition.Forward(condition) : null;
        Tensor<T>? output = null;
        for (int i = 0; i < _layers; i++)
        {
            var a = _inLayers[i].Forward(x);
            if (projected is not null)
                a = _engine.TensorAdd(a, PortaSpeechOps.Slice(_engine, projected, i * 2 * _hidden, 2 * _hidden));
            var acts = _engine.TensorMultiply(_engine.Tanh(PortaSpeechOps.Slice(_engine, a, 0, _hidden)),
                _engine.Sigmoid(PortaSpeechOps.Slice(_engine, a, _hidden, _hidden)));
            var resSkip = _resSkipLayers[i].Forward(acts);
            if (i < _layers - 1)
            {
                x = _engine.TensorAdd(x, PortaSpeechOps.Slice(_engine, resSkip, 0, _hidden));
                var skip = PortaSpeechOps.Slice(_engine, resSkip, _hidden, _hidden);
                output = output is null ? skip : _engine.TensorAdd(output, skip);
            }
            else
            {
                output = output is null ? resSkip : _engine.TensorAdd(output, resSkip);
            }
        }
        return output!;
    }
}

/// <summary>
/// PortaSpeech's variational generator (Ren et al. 2021 §3.2, App. A.2; reference <c>fvae.py</c>): a stride-4 VAE over
/// the mel spectrogram conditioned on the linguistic features, whose prior is a volume-preserving flow.
/// </summary>
internal sealed class PortaSpeechVariationalGenerator<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly List<LayerBase<T>> _layers = new();
    private readonly int _latent;
    private readonly int _stride;

    private readonly Conv1DLayer<T> _conditionSqueeze;
    private readonly Conv1DLayer<T> _encoderPre;
    private readonly LayerNormalizationLayer<T> _encoderNorm;
    private readonly PortaSpeechWaveNet<T> _encoderNet;
    private readonly Conv1DLayer<T> _encoderOut;
    private readonly List<(Conv1DLayer<T> Pre, PortaSpeechWaveNet<T> Net, Conv1DLayer<T> Post)> _prior = new();
    private readonly Conv1DTransposeLayer<T> _decoderPre;
    private readonly LayerNormalizationLayer<T> _decoderNorm;
    private readonly PortaSpeechWaveNet<T> _decoderNet;
    private readonly Conv1DLayer<T> _decoderOut;

    public PortaSpeechVariationalGenerator(IEngine engine, int melChannels, int conditionChannels, int hidden, int latent,
        int kernelSize, int encoderLayers, int decoderLayers, int stride, int priorSteps, int priorLayers, int priorHidden,
        int priorKernelSize)
    {
        _engine = engine;
        _latent = latent;
        _stride = stride;
        _conditionSqueeze = Add(new Conv1DLayer<T>(conditionChannels, conditionChannels, 2 * stride, 1, stride, stride / 2));
        _encoderPre = Add(new Conv1DLayer<T>(melChannels, hidden, 2 * stride, 1, stride, stride / 2));
        _encoderNorm = Add(new LayerNormalizationLayer<T>(hidden));
        _encoderNet = new PortaSpeechWaveNet<T>(engine, _layers, hidden, kernelSize, encoderLayers, conditionChannels);
        _encoderOut = Add(new Conv1DLayer<T>(hidden, 2 * latent, 1, 1, 1, 0));
        for (int i = 0; i < priorSteps; i++)
            _prior.Add((Add(new Conv1DLayer<T>(latent / 2, priorHidden, 1, 1, 1, 0)),
                new PortaSpeechWaveNet<T>(engine, _layers, priorHidden, priorKernelSize, priorLayers, conditionChannels),
                Add(new Conv1DLayer<T>(priorHidden, latent - latent / 2, 1, 1, 1, 0))));
        _decoderPre = Add(new Conv1DTransposeLayer<T>(inputChannels: latent, outputChannels: hidden, kernelSize: stride, stride: stride, padding: 0));
        _decoderNorm = Add(new LayerNormalizationLayer<T>(hidden));
        _decoderNet = new PortaSpeechWaveNet<T>(engine, _layers, hidden, kernelSize, decoderLayers, conditionChannels);
        _decoderOut = Add(new Conv1DLayer<T>(hidden, melChannels, 1, 1, 1, 0));
    }

    public IReadOnlyList<LayerBase<T>> Layers => _layers;

    private TLayer Add<TLayer>(TLayer layer) where TLayer : LayerBase<T>
    {
        _layers.Add(layer);
        return layer;
    }

    /// <summary>The stride-reduced condition <c>[1, c, T / stride]</c>.</summary>
    public Tensor<T> SqueezeCondition(Tensor<T> condition) => _conditionSqueeze.Forward(condition);

    /// <summary>
    /// Posterior sample and Monte-Carlo KL (Eq. 3): <c>z = m + ε e^{logs}</c>,
    /// <c>KL = mean(log q(z | x, c) − log N(f(z); 0, I))</c> with the volume-preserving prior flow f.
    /// </summary>
    public (Tensor<T> Latent, Tensor<T> Kl) Encode(Tensor<T> mel, Tensor<T> squeezedCondition, Tensor<T> noise)
    {
        var h = _engine.ReLU(_encoderPre.Forward(mel));
        h = PortaSpeechOps.ChannelNorm(_engine, _encoderNorm, h);
        h = _encoderOut.Forward(_encoderNet.Forward(h, squeezedCondition));
        var mean = PortaSpeechOps.Slice(_engine, h, 0, _latent);
        var logScale = PortaSpeechOps.Slice(_engine, h, _latent, _latent);
        var z = _engine.TensorAdd(mean, _engine.TensorMultiply(noise, _engine.TensorExp(logScale)));
        var prior = PriorFlow(z, squeezedCondition, reverse: false);
        // log q(z) = −½ε² − logs − ½ log 2π and log N(f(z)) = −½ f(z)² − ½ log 2π: the constants cancel.
        var logQ = _engine.TensorSubtract(_engine.TensorMultiplyScalar(_engine.TensorMultiply(noise, noise), NumOps.FromDouble(-0.5)), logScale);
        var logP = _engine.TensorMultiplyScalar(_engine.TensorMultiply(prior, prior), NumOps.FromDouble(-0.5));
        var kl = _engine.ReduceMean(_engine.TensorSubtract(logQ, logP), new[] { 0, 1, 2 }, keepDims: false);
        return (z, kl);
    }

    /// <summary>A prior sample <c>f⁻¹(ε)</c>, ε ~ N(0, I).</summary>
    public Tensor<T> SamplePrior(Tensor<T> squeezedCondition, Tensor<T> noise) => PriorFlow(noise, squeezedCondition, reverse: true);

    /// <summary>The coarse mel spectrogram <c>[1, mel, T]</c> from a latent <c>[1, latent, T / stride]</c>.</summary>
    public Tensor<T> Decode(Tensor<T> latent, Tensor<T> condition)
    {
        var h = _engine.ReLU(_decoderPre.Forward(latent));
        h = PortaSpeechOps.ChannelNorm(_engine, _decoderNorm, h);
        return _decoderOut.Forward(_decoderNet.Forward(h, condition));
    }

    // Residual couplings x1 ± post(WN(pre(x0), c)), each followed by a channel flip (reference ResFlow).
    private Tensor<T> PriorFlow(Tensor<T> x, Tensor<T> condition, bool reverse)
    {
        int half = _latent / 2;
        if (!reverse)
        {
            foreach (var (pre, net, post) in _prior)
                x = Flip(Couple(x, condition, pre, net, post, half, reverse: false));
            return x;
        }
        for (int i = _prior.Count - 1; i >= 0; i--)
            x = Couple(Flip(x), condition, _prior[i].Pre, _prior[i].Net, _prior[i].Post, half, reverse: true);
        return x;
    }

    private Tensor<T> Couple(Tensor<T> x, Tensor<T> condition, Conv1DLayer<T> pre, PortaSpeechWaveNet<T> net,
        Conv1DLayer<T> post, int half, bool reverse)
    {
        var x0 = PortaSpeechOps.Slice(_engine, x, 0, half);
        var x1 = PortaSpeechOps.Slice(_engine, x, half, x.Shape[1] - half);
        var shift = post.Forward(net.Forward(pre.Forward(x0), condition));
        x1 = reverse ? _engine.TensorSubtract(x1, shift) : _engine.TensorAdd(x1, shift);
        return _engine.TensorConcatenate(new[] { x0, x1 }, 1);
    }

    private Tensor<T> Flip(Tensor<T> x)
    {
        int channels = x.Shape[1], time = x.Shape[2];
        var index = new Tensor<int>(new[] { channels });
        for (int c = 0; c < channels; c++) index[c] = channels - 1 - c;
        var flipped = _engine.TensorIndexSelect(_engine.Reshape(x, new[] { channels, time }), index, 0);
        return _engine.Reshape(flipped, new[] { 1, channels, time });
    }
}

/// <summary>
/// PortaSpeech's flow-based post-net (§3.3, App. A.3; reference Glow in <c>glow_modules.py</c>): a squeezed Glow of
/// ActNorm, grouped invertible 1×1 convolutions and conditional affine couplings whose WaveNet gated and residual/skip
/// convolutions are shared within groups of steps while each step keeps its own condition projection.
/// </summary>
internal sealed class PortaSpeechPostNet<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly List<LayerBase<T>> _layers = new();
    private readonly int _channels;
    private readonly int _hidden;
    private readonly List<Step> _steps = new();

    private sealed record Step(ActNormFlowLayer<T> Norm, GroupedInvertibleConvFlowLayer<T> Mix, WeightNormConv1DLayer<T> Start,
        PortaSpeechWaveNet<T> Net, Conv1DLayer<T> End);

    public PortaSpeechPostNet(IEngine engine, int melChannels, int conditionChannels, int hidden, int kernelSize, int blocks,
        int blockLayers, int split, int shareGroupSize)
    {
        _engine = engine;
        _channels = 2 * melChannels;
        _hidden = hidden;
        PortaSpeechWaveNet<T>? shared = null;
        for (int b = 0; b < blocks; b++)
        {
            var norm = Add(new ActNormFlowLayer<T>(_channels, initialized: true));
            var mix = Add(new GroupedInvertibleConvFlowLayer<T>(_channels, split));
            var start = Add(new WeightNormConv1DLayer<T>(_channels / 2, hidden, 1, 1, 0));
            bool newGroup = shareGroupSize <= 0 || b % shareGroupSize == 0;
            var net = new PortaSpeechWaveNet<T>(engine, _layers, hidden, kernelSize, blockLayers, 2 * conditionChannels,
                newGroup ? null : shared);
            if (newGroup) shared = net;
            // Zero-initialized so each coupling starts as the identity (reference CouplingBlock.end).
            var end = Add(new Conv1DLayer<T>(hidden, _channels, 1, 1, 1, 0, null, new AiDotNet.Initialization.ZeroInitializationStrategy<T>()));
            _steps.Add(new Step(norm, mix, start, net, end));
        }
    }

    public IReadOnlyList<LayerBase<T>> Layers => _layers;

    private TLayer Add<TLayer>(TLayer layer) where TLayer : LayerBase<T>
    {
        _layers.Add(layer);
        return layer;
    }

    /// <summary>The negative log-likelihood <c>mean ½(z² + log 2π) − log|det| / (T · mel)</c> of <paramref name="mel"/>
    /// <c>[1, mel, T]</c> (T even) under the post-net given the condition <c>[1, c, T]</c>.</summary>
    public Tensor<T> NegativeLogLikelihood(Tensor<T> mel, Tensor<T> condition)
    {
        var x = Squeeze(mel);
        var g = Squeeze(condition);
        Tensor<T>? logDet = null;
        foreach (var step in _steps)
        {
            var (normed, d1) = step.Norm.Transform(x, reverse: false);
            var (mixed, d2) = step.Mix.Transform(normed, reverse: false);
            var (coupled, d3) = Couple(step, mixed, g, reverse: false);
            x = coupled;
            foreach (var d in new[] { d1, d2, d3 })
                if (d is not null) logDet = logDet is null ? d : _engine.TensorAdd(logDet, d);
        }
        double elements = mel.Shape[1] * (double)mel.Shape[2];
        var nll = _engine.TensorAddScalar(
            _engine.TensorMultiplyScalar(_engine.ReduceSum(_engine.TensorMultiply(x, x), new[] { 0, 1, 2 }, keepDims: false),
                NumOps.FromDouble(0.5 / elements)),
            NumOps.FromDouble(0.5 * Math.Log(2 * Math.PI)));
        return _engine.TensorSubtract(nll, _engine.TensorMultiplyScalar(logDet!, NumOps.FromDouble(1.0 / elements)));
    }

    /// <summary>Generates <c>[1, mel, T]</c> from <paramref name="noise"/> of that shape given the condition.</summary>
    public Tensor<T> Generate(Tensor<T> noise, Tensor<T> condition)
    {
        var x = Squeeze(noise);
        var g = Squeeze(condition);
        for (int i = _steps.Count - 1; i >= 0; i--)
        {
            var step = _steps[i];
            x = Couple(step, x, g, reverse: true).Output;
            x = step.Mix.Transform(x, reverse: true).Output;
            x = step.Norm.Transform(x, reverse: true).Output;
        }
        return Unsqueeze(x);
    }

    private (Tensor<T> Output, Tensor<T>? LogDeterminant) Couple(Step step, Tensor<T> x, Tensor<T> condition, bool reverse)
    {
        int half = _channels / 2;
        var x0 = PortaSpeechOps.Slice(_engine, x, 0, half);
        var x1 = PortaSpeechOps.Slice(_engine, x, half, half);
        var stats = step.End.Forward(step.Net.Forward(step.Start.Forward(x0), condition));
        var shift = PortaSpeechOps.Slice(_engine, stats, 0, half);
        var logScale = PortaSpeechOps.Slice(_engine, stats, half, half);
        if (reverse)
        {
            var z = _engine.TensorMultiply(_engine.TensorSubtract(x1, shift), _engine.TensorExp(_engine.TensorNegate(logScale)));
            return (_engine.TensorConcatenate(new[] { x0, z }, 1), null);
        }
        var z1 = _engine.TensorAdd(shift, _engine.TensorMultiply(_engine.TensorExp(logScale), x1));
        return (_engine.TensorConcatenate(new[] { x0, z1 }, 1), _engine.ReduceSum(logScale, new[] { 0, 1, 2 }, keepDims: false));
    }

    // [1, C, T] -> [1, 2C, T/2] with channel p·C + c holding x[c, 2j + p] (reference utils.squeeze).
    private Tensor<T> Squeeze(Tensor<T> x)
    {
        int c = x.Shape[1], t = x.Shape[2] / 2;
        var y = _engine.TensorPermute(_engine.Reshape(x, new[] { c, t, 2 }), new[] { 2, 0, 1 }).Contiguous();
        return _engine.Reshape(y, new[] { 1, 2 * c, t });
    }

    private Tensor<T> Unsqueeze(Tensor<T> x)
    {
        int c = x.Shape[1] / 2, t = x.Shape[2];
        var y = _engine.TensorPermute(_engine.Reshape(x, new[] { 2, c, t }), new[] { 1, 2, 0 }).Contiguous();
        return _engine.Reshape(y, new[] { 1, c, 2 * t });
    }
}

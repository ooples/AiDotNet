using System.Collections.Generic;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>The released OpenVoice/V2 converter: posterior encoder, source-to-prior flow,
/// target inverse flow, and speaker-conditioned VITS HiFi-GAN. Noise is an explicit input.</summary>
internal sealed class OpenVoiceConverterCore<T>
{
    private readonly IEngine _engine;
    private readonly int _spectrogramChannels, _latent, _hidden;
    private readonly bool _zeroCondition;
    private readonly NormedConv1DLayer<T> _posteriorPre, _posteriorProjection;
    private readonly OpenVoiceWaveNet<T> _posterior;
    private readonly List<(NormedConv1DLayer<T> Pre, OpenVoiceWaveNet<T> Encoder, NormedConv1DLayer<T> Post)> _flows = new();
    private readonly HiFiGanGenerator<T> _decoder;
    private readonly List<LayerBase<T>> _layers = new();
    public IReadOnlyList<LayerBase<T>> Layers => _layers;

    public OpenVoiceConverterCore(IEngine engine, int spectrogramChannels = 513, int latentChannels = 192,
        int hiddenChannels = 192, int speakerChannels = 256, bool zeroCondition = false,
        int[]? upsampleRates = null, int[]? upsampleKernels = null, int initialChannels = 512,
        int[]? residualKernels = null, int[][]? residualDilations = null)
    {
        if (spectrogramChannels <= 0 || latentChannels <= 0 || latentChannels % 2 != 0 || hiddenChannels <= 0 || speakerChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(latentChannels));
        _engine = engine; _spectrogramChannels = spectrogramChannels; _latent = latentChannels;
        _hidden = hiddenChannels; _zeroCondition = zeroCondition;
        _posteriorPre = Plain(spectrogramChannels, hiddenChannels);
        _posterior = new OpenVoiceWaveNet<T>(engine, hiddenChannels, 5, 16, speakerChannels);
        _layers.AddRange(_posterior.Layers);
        _posteriorProjection = Plain(hiddenChannels, 2 * latentChannels);
        for (int i = 0; i < 4; i++)
        {
            var pre = Plain(latentChannels / 2, hiddenChannels);
            var encoder = new OpenVoiceWaveNet<T>(engine, hiddenChannels, 5, 4, speakerChannels);
            _layers.AddRange(encoder.Layers);
            var post = Plain(hiddenChannels, latentChannels / 2);
            // The author's mean-only coupling is initialized to the identity, before its channel flip.
            post.LoadTorchWeights(new double[hiddenChannels * latentChannels / 2], null, new double[latentChannels / 2]);
            _flows.Add((pre, encoder, post));
        }
        _decoder = new HiFiGanGenerator<T>(engine, latentChannels, initialChannels,
            upsampleRates ?? new[] { 8, 8, 2, 2 }, upsampleKernels ?? new[] { 16, 16, 4, 4 },
            residualKernels ?? new[] { 3, 7, 11 }, residualDilations ?? new[] { new[] { 1, 3, 5 }, new[] { 1, 3, 5 }, new[] { 1, 3, 5 } },
            true, speakerChannels, plainEnds: true);
        _layers.AddRange(_decoder.Layers);
    }

    private NormedConv1DLayer<T> Plain(int input, int output)
    {
        var layer = new NormedConv1DLayer<T>(input, output, 1, 1, 1, 1, 0, false, ConvolutionNormalization.None);
        _layers.Add(layer); return layer;
    }

    private Tensor<T> Mask(Tensor<T> x, Tensor<T> mask)
        => _engine.TensorMultiply(x, _engine.TensorTile(mask, new[] { 1, x.Shape[1], 1 }));

    private Tensor<T> Slice(Tensor<T> x, int offset, int channels)
        => _engine.TensorSlice(x, new[] { 0, offset, 0 }, new[] { x.Shape[0], channels, x.Shape[2] });

    private Tensor<T> Flip(Tensor<T> x)
    {
        var channels = new Tensor<T>[x.Shape[1]];
        for (int i = 0; i < channels.Length; i++) channels[i] = Slice(x, channels.Length - 1 - i, 1);
        return _engine.TensorConcatenate(channels, 1);
    }

    internal Tensor<T> Flow(Tensor<T> latent, Tensor<T> mask, Tensor<T> speaker, bool reverse)
    {
        var x = latent;
        for (int step = 0; step < _flows.Count; step++)
        {
            int index = reverse ? _flows.Count - 1 - step : step;
            if (reverse) x = Flip(x);
            var block = _flows[index];
            var first = Slice(x, 0, _latent / 2);
            var second = Slice(x, _latent / 2, _latent / 2);
            var mean = Mask(block.Post.Forward(block.Encoder.Forward(Mask(block.Pre.Forward(first), mask), mask, speaker)), mask);
            second = reverse ? Mask(_engine.TensorSubtract(second, mean), mask)
                : _engine.TensorAdd(mean, Mask(second, mask));
            x = _engine.TensorConcatenate(new[] { first, second }, 1);
            if (!reverse) x = Flip(x);
        }
        return x;
    }

    public (Tensor<T> Waveform, Tensor<T> Posterior, Tensor<T> Prior, Tensor<T> Converted) Convert(
        Tensor<T> spectrogram, Tensor<T> mask, Tensor<T> sourceSpeaker, Tensor<T> targetSpeaker, Tensor<T> noise, double tau = .3)
    {
        if (spectrogram.Rank != 3 || spectrogram.Shape[1] != _spectrogramChannels)
            throw new ArgumentException("Expected [batch, linear-frequency, frames].", nameof(spectrogram));
        if (double.IsNaN(tau) || double.IsInfinity(tau) || tau < 0) throw new ArgumentOutOfRangeException(nameof(tau));
        // V2 retains the condition projections and their learned biases even when g is zero.
        var decoderCondition = _zeroCondition ? new Tensor<T>(targetSpeaker.Shape.ToArray()) : targetSpeaker;
        var posterior = EncodePosterior(spectrogram, mask, sourceSpeaker, noise, tau);
        var sampled = posterior.Sampled;
        var prior = Flow(sampled, mask, sourceSpeaker, false);
        var converted = Flow(prior, mask, targetSpeaker, true);
        return (_decoder.Forward(Mask(converted, mask), decoderCondition), sampled, prior, converted);
    }

    private (Tensor<T> Sampled, Tensor<T> Mean, Tensor<T> LogScale) EncodePosterior(
        Tensor<T> spectrogram, Tensor<T> mask, Tensor<T> speaker, Tensor<T> noise, double tau)
    {
        var posteriorCondition = _zeroCondition ? new Tensor<T>(speaker.Shape.ToArray()) : speaker;
        var stats = Mask(_posteriorProjection.Forward(_posterior.Forward(
            Mask(_posteriorPre.Forward(spectrogram), mask), mask, posteriorCondition)), mask);
        var mean = Slice(stats, 0, _latent);
        var logs = Slice(stats, _latent, _latent);
        var sampled = Mask(_engine.TensorAdd(mean, _engine.TensorMultiplyScalar(
            _engine.TensorMultiply(noise, _engine.TensorExp(logs)), MathHelper.GetNumericOperations<T>().FromDouble(tau))), mask);
        return (sampled, mean, logs);
    }

    /// <summary>Paper section 2.3: reconstruct directly from the posterior; use the
    /// forward flow only for the separate IPA-aligned KL objective, never inverse-convert the training audio.</summary>
    internal (Tensor<T> Waveform, Tensor<T> Posterior, Tensor<T> Prior, Tensor<T> Mean, Tensor<T> LogScale) TrainingForward(
        Tensor<T> spectrogram, Tensor<T> mask, Tensor<T> speaker, Tensor<T> noise, int segmentStart, int segmentFrames)
    {
        if (spectrogram.Rank != 3 || spectrogram.Shape[1] != _spectrogramChannels || segmentStart < 0 ||
            segmentFrames <= 0 || segmentStart + segmentFrames > spectrogram.Shape[2])
            throw new ArgumentException("Invalid posterior spectrogram or decoder training segment.");
        var posterior = EncodePosterior(spectrogram, mask, speaker, noise, 1);
        var prior = Flow(posterior.Sampled, mask, speaker, false);
        var segment = _engine.TensorSlice(posterior.Sampled, new[] { 0, 0, segmentStart },
            new[] { spectrogram.Shape[0], _latent, segmentFrames });
        var condition = _zeroCondition ? new Tensor<T>(speaker.Shape.ToArray()) : speaker;
        return (_decoder.Forward(segment, condition), posterior.Sampled, prior, posterior.Mean, posterior.LogScale);
    }

    public void LoadTorch(Func<string, int[], double[]> read, Func<string, double[]?> optionalRead)
    {
        void PlainLoad(NormedConv1DLayer<T> layer, string prefix, int input, int output)
            => layer.LoadTorchWeights(read(prefix + ".weight", new[] { output, input, 1 }), null,
                read(prefix + ".bias", new[] { output }));
        PlainLoad(_posteriorPre, "enc_q.pre", _spectrogramChannels, _hidden);
        _posterior.LoadTorch(read, "enc_q.enc.");
        PlainLoad(_posteriorProjection, "enc_q.proj", _hidden, 2 * _latent);
        for (int i = 0; i < _flows.Count; i++)
        {
            string prefix = "flow.flows." + (i * 2);
            PlainLoad(_flows[i].Pre, prefix + ".pre", _latent / 2, _hidden);
            _flows[i].Encoder.LoadTorch(read, prefix + ".enc.");
            PlainLoad(_flows[i].Post, prefix + ".post", _hidden, _latent / 2);
        }
        _decoder.LoadTorch(optionalRead, "dec.");
    }
}

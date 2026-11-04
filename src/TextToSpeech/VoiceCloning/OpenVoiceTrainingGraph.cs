using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>OpenVoice section 2.3: direct posterior waveform reconstruction,
/// speaker-free IPA Transformer prior, monotonic Gaussian alignment, flow KL and VITS-style HiFi-GAN losses.</summary>
internal sealed class OpenVoiceTrainingGraph<T>
{
    internal sealed record Utterance(Tensor<T> Tokens, Tensor<T> Audio, Tensor<T> Spectrum, Tensor<T> ReferenceSpectrum,
        int Frames, int SegmentFrames);
    internal sealed record Draw(Tensor<T> Noise, int SegmentStart);
    private readonly IEngine _engine;
    private readonly OpenVoiceNative<T> _native;
    private readonly OpenVoiceTrainingOptions _options;
    private readonly OpenVoiceIpaPrior<T> _prior;
    private readonly HiFiGanDiscriminators<T> _discriminators;
    private readonly DifferentiableMel<T> _mel;
    private static readonly INumericOperations<T> Numeric = MathHelper.GetNumericOperations<T>();
    internal IReadOnlyList<LayerBase<T>> AdditionalLayers { get; }
    internal IReadOnlyList<LayerBase<T>> GeneratorLayers { get; }
    internal IReadOnlyList<LayerBase<T>> DiscriminatorLayers => _discriminators.Layers;
    internal Func<int[], Tensor<T>>? NoiseForAudit { get; set; }
    internal int? SegmentStartForAudit { get; set; }
    internal Func<string, int[], double[]>? DropoutMasksForAudit { get; set; }

    internal void LoadAuxiliaryTorch(Func<string, int[], double[]> prior, Func<string, double[]?> discriminator)
    {
        _prior.LoadTorch(prior);
        _discriminators.LoadTorch(discriminator);
    }

    internal void RestoreAuxiliaryParameters(IReadOnlyList<LayerBase<T>> restored)
    {
        if (restored.Count != AdditionalLayers.Count)
            throw new InvalidDataException("The restored OpenVoice auxiliary graph differs from its training configuration.");
        var parameters = restored.Select(layer => layer.GetParameters()).ToArray();
        for (int i = 0; i < restored.Count; i++)
            if (restored[i].GetType().FullName != AdditionalLayers[i].GetType().FullName ||
                parameters[i].Length != AdditionalLayers[i].GetParameters().Length)
                throw new InvalidDataException("The restored OpenVoice auxiliary layer differs at index " + i + ".");
        // These modules have learned tensors but no running-statistic buffers:
        // the IPA prior uses layer norms and the critics use weight norms.
        for (int i = 0; i < restored.Count; i++) AdditionalLayers[i].SetParameters(parameters[i]);
    }

    internal OpenVoiceTrainingGraph(IEngine engine, OpenVoiceNative<T> native, OpenVoiceTrainingOptions options, int hidden, int sampleRate)
    {
        if (native.HopSize != native.DecoderScaleFactor)
            throw new ArgumentException("Waveform training requires the actual decoder upsample factor to equal the analysis hop.", nameof(native));
        if (options.SegmentFrames <= 0 || options.MelChannels <= 0 || options.DiscriminatorWidthDivisor <= 0 ||
            options.MelMaxFrequency <= 0 || options.MelMaxFrequency > sampleRate / 2.0 || options.MelLossWeight < 0 ||
            options.FeatureMatchingWeight < 0 || options.KlWeight <= 0)
            throw new ArgumentException("Invalid OpenVoice waveform training configuration.", nameof(options));
        _engine = engine; _native = native; _options = options;
        _prior = new OpenVoiceIpaPrior<T>(engine, options.IpaVocabularySize, hidden, native.LatentChannels,
            options.IpaFilterChannels, options.IpaHeads, options.IpaLayers, options.IpaKernel, options.IpaDropout);
        _discriminators = new HiFiGanDiscriminators<T>(engine, new[] { 2, 3, 5, 7, 11 }, 1,
            widthDivisor: options.DiscriminatorWidthDivisor, spectralFirstScale: false, vitsScale: true);
        _mel = new DifferentiableMel<T>(engine, sampleRate, native.FftSize, native.HopSize, native.FftSize,
            options.MelChannels, 0, options.MelMaxFrequency);
        AdditionalLayers = _prior.Layers.Concat(_discriminators.Layers).ToArray();
        GeneratorLayers = native.Layers.Concat(_prior.Layers).ToArray();
    }

    internal Utterance Prepare(TtsTrainingSample<T> sample)
    {
        if (sample.Audio is null || sample.Audio.Rank != 1 || sample.Tokens.Rank != 1 || sample.Tokens.Length == 0)
            throw new ArgumentException("OpenVoice training requires IPA-token IDs and a mono target recording.", nameof(sample));
        using var noGrad = new NoGradScope<T>();
        var spectrum = _native.TrainingSpectrogram(sample.Audio);
        var reference = _native.TrainingSpectrogram(sample.SpeakerReference ?? sample.Audio);
        int frames = spectrum.Shape[0];
        if (sample.Tokens.Length > frames) throw new ArgumentException("The recording has fewer frames than IPA tokens.", nameof(sample));
        int segment = Math.Min(frames, _options.SegmentFrames);
        if (segment * _native.HopSize <= (_native.FftSize - _native.HopSize) / 2)
            throw new ArgumentException("The reconstruction segment is too short for the original reflection-padded mel analysis.", nameof(sample));
        return new Utterance(sample.Tokens, sample.Audio, spectrum, reference, frames, segment);
    }

    internal Draw Sample(Utterance data)
    {
        int seed = unchecked(_options.SamplingSeed + (int)_options.TrainingSteps * 7919);
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var shape = new[] { 1, _native.LatentChannels, data.Frames };
        var noise = NoiseForAudit?.Invoke(shape) ?? new Tensor<T>(shape,
            TorchParameters.Gaussian<T>(random, _native.LatentChannels * data.Frames, 1));
        int start = SegmentStartForAudit ?? random.Next(data.Frames - data.SegmentFrames + 1);
        if (noise.Rank != 3 || noise.Shape[0] != 1 || noise.Shape[1] != _native.LatentChannels ||
            noise.Shape[2] != data.Frames || start < 0 || start + data.SegmentFrames > data.Frames)
            throw new ArgumentException("The posterior draw or decoder crop differs from the utterance.");
        return new Draw(noise, start);
    }

    internal Tensor<T> RealSegment(Utterance data, Draw draw) => _engine.TensorSlice(data.Audio,
        new[] { draw.SegmentStart * _native.HopSize }, new[] { data.SegmentFrames * _native.HopSize });

    internal Tensor<T> Generate(Utterance data, Draw draw) => Flat(_native.TrainingForward(data.Spectrum,
        data.ReferenceSpectrum, draw.Noise, draw.SegmentStart, data.SegmentFrames).Waveform);

    private Tensor<T> Flat(Tensor<T> value) => _engine.Reshape(value, new[] { value.Length });
    private Tensor<T> Mean(Tensor<T> value) => _engine.Reshape(_engine.ReduceMean(value,
        Enumerable.Range(0, value.Rank).ToArray(), true), new[] { 1 });
    private Tensor<T> Scale(Tensor<T> value, double scale) => _engine.TensorMultiplyScalar(value, Numeric.FromDouble(scale));
    private Tensor<T> Square(Tensor<T> value) => _engine.TensorMultiply(value, value);
    private Tensor<T> Sum(IEnumerable<Tensor<T>> values)
    {
        Tensor<T>? result = null;
        foreach (var value in values) result = result is null ? value : _engine.TensorAdd(result, value);
        return result ?? new Tensor<T>(new[] { 1 });
    }

    internal Tensor<T> DiscriminatorLoss(Tensor<T> real, Tensor<T> generated)
    {
        var realScores = _discriminators.Forward(real); var fakeScores = _discriminators.Forward(generated);
        return Sum(Enumerable.Range(0, realScores.Count).Select(index => _engine.TensorAdd(
            Mean(Square(_engine.TensorAddScalar(realScores[index].Score, Numeric.FromDouble(-1)))), Mean(Square(fakeScores[index].Score)))));
    }

    internal Tensor<T> GeneratorLoss(Utterance data, Draw draw, bool adversarial, bool training)
    {
        var posterior = _native.TrainingForward(data.Spectrum, data.ReferenceSpectrum, draw.Noise, draw.SegmentStart, data.SegmentFrames);
        var prior = _prior.Forward(data.Tokens, training, DropoutMasksForAudit);
        int[] durations = Align(posterior.Prior, prior.Mean, prior.LogScale);
        var alignedMean = Expand(prior.Mean, durations); var alignedLogs = Expand(prior.LogScale, durations);
        var mask = new Tensor<T>(new[] { 1, 1, data.Frames });
        for (int i = 0; i < data.Frames; i++) mask[i] = Numeric.One;
        var kl = OpenVoiceTrainingLoss<T>.AlignedKl(_engine, posterior.Prior, posterior.LogScale, alignedMean, alignedLogs, mask);
        var generated = Flat(posterior.Waveform); var real = RealSegment(data, draw);
        Tensor<T> targetMel;
        // OpenVoice's published mel_processing uses 1e-6 inside the magnitude
        // square root; the shared HiFi-GAN helper defaults to 1e-9.
        using (new NoGradScope<T>()) targetMel = _mel.LogMel(_mel.Magnitude(real, 1e-6));
        var generatedMel = _mel.LogMel(_mel.Magnitude(generated, 1e-6));
        var mel = Mean(_engine.TensorAbs(_engine.TensorSubtract(generatedMel, targetMel)));
        var result = _engine.TensorAdd(Scale(mel, _options.MelLossWeight), Scale(kl, _options.KlWeight));
        if (!adversarial) return result;
        var fake = _discriminators.Forward(generated);
        List<(Tensor<T> Score, List<Tensor<T>> Features)> realFeatures;
        using (new NoGradScope<T>()) realFeatures = _discriminators.Forward(real);
        var gan = Sum(fake.Select(value => Mean(Square(_engine.TensorAddScalar(value.Score, Numeric.FromDouble(-1))))));
        var featureLosses = new List<Tensor<T>>();
        for (int index = 0; index < fake.Count; index++)
            for (int layer = 0; layer < fake[index].Features.Count; layer++)
                featureLosses.Add(Mean(_engine.TensorAbs(_engine.TensorSubtract(fake[index].Features[layer], realFeatures[index].Features[layer]))));
        return _engine.TensorAdd(result, _engine.TensorAdd(gan, Scale(Sum(featureLosses), _options.FeatureMatchingWeight)));
    }

    private int[] Align(Tensor<T> transformed, Tensor<T> mean, Tensor<T> logs)
    {
        // MAS is a discrete, detached choice, as in the paper's permitted VITS alternative.
        int tokens = mean.Shape[2], frames = transformed.Shape[2], channels = transformed.Shape[1];
        var scores = new double[tokens, frames];
        for (int token = 0; token < tokens; token++) for (int frame = 0; frame < frames; frame++)
        {
            double score = 0;
            for (int channel = 0; channel < channels; channel++)
            {
                double log = Numeric.ToDouble(logs[0, channel, token]);
                double delta = Numeric.ToDouble(transformed[0, channel, frame]) - Numeric.ToDouble(mean[0, channel, token]);
                score += -.5 * Math.Log(2 * Math.PI) - log - .5 * delta * delta * Math.Exp(-2 * log);
            }
            scores[token, frame] = score;
        }
        return MonotonicAlignment.MaximumPath(scores);
    }

    private Tensor<T> Expand(Tensor<T> value, int[] durations)
    {
        var frames = new List<Tensor<T>>();
        for (int token = 0; token < durations.Length; token++)
        {
            var row = _engine.TensorSlice(value, new[] { 0, 0, token }, new[] { 1, value.Shape[1], 1 });
            frames.Add(_engine.TensorTile(row, new[] { 1, 1, durations[token] }));
        }
        return _engine.TensorConcatenate(frames.ToArray(), 2);
    }
}

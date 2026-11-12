using AiDotNet.Audio.Features;
using AiDotNet.Helpers;

namespace AiDotNet.Audio.Classification;

/// <summary>
/// What both AST models do around <see cref="AstNetwork{T}"/>: the official data loader's front end (the waveform's mean
/// removed, a Kaldi filterbank, zero-padding or cutting to the target length, normalization by the dataset statistics)
/// and the paper's AudioSet training step (mixup, SpecAugment, binary cross-entropy).
/// </summary>
internal sealed class AstPipeline<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly ASTOptions _options;
    private readonly KaldiFbank<T> _fbank;

    private readonly AstNetwork<T>? _network;
    private readonly int _classes;

    /// <param name="layers">Where the network's layers go, or null for the front end alone (ONNX mode).</param>
    public AstPipeline(IEngine engine, ASTOptions options, int classes, List<LayerBase<T>>? layers)
    {
        Validate(options);
        _engine = engine;
        _options = options;
        _classes = classes;
        _fbank = new KaldiFbank<T>(engine, options.SampleRate, options.NumMels, options.Window);
        if (layers is not null) _network = new AstNetwork<T>(engine, layers, Configuration(options, classes));
    }

    public AstNetwork<T> Network => _network ?? throw new InvalidOperationException("This AST runs an ONNX graph; it has no native network.");

    /// <summary>The network shape the options describe.</summary>
    public static AstConfiguration Configuration(ASTOptions o, int classes) => new()
    {
        MelBins = o.NumMels, Frames = o.TargetLength, PatchSize = o.PatchSize, FrequencyStride = o.FrequencyStride,
        TimeStride = o.TimeStride, Dim = o.EmbeddingDim, Depth = o.NumEncoderLayers, Heads = o.NumAttentionHeads,
        MlpRatio = (double)o.FeedForwardDim / o.EmbeddingDim, Classes = classes,
    };

    private static void Validate(ASTOptions o)
    {
        if (o.SampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(o.SampleRate), o.SampleRate, "The sample rate must be positive.");
        if (o.PatchSize <= 0 || o.FrequencyStride <= 0 || o.TimeStride <= 0)
            throw new ArgumentOutOfRangeException(nameof(o.PatchSize), "The patch size and strides must be positive.");
        if (o.NumMels < o.PatchSize || o.TargetLength < o.PatchSize)
            throw new ArgumentOutOfRangeException(nameof(o.NumMels), $"The filterbank ({o.TargetLength} frames, {o.NumMels} bins) must hold one {o.PatchSize}x{o.PatchSize} patch.");
        if (o.EmbeddingDim <= 0 || o.NumAttentionHeads <= 0 || o.EmbeddingDim % o.NumAttentionHeads != 0)
            throw new ArgumentOutOfRangeException(nameof(o.EmbeddingDim), "The width must be a positive multiple of the head count.");
        if (o.NumEncoderLayers <= 0 || o.FeedForwardDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(o.NumEncoderLayers), "The depth and MLP width must be positive.");
        if (o.DatasetStd <= 0) throw new ArgumentOutOfRangeException(nameof(o.DatasetStd), o.DatasetStd, "The dataset std must be positive.");
        if (o.MixupRate < 0 || o.MixupRate > 1) throw new ArgumentOutOfRangeException(nameof(o.MixupRate), o.MixupRate, "The mixup rate is a probability.");
        if (o.FrequencyMaskLength < 0 || o.TimeMaskLength < 0) throw new ArgumentOutOfRangeException(nameof(o.FrequencyMaskLength), "Mask lengths cannot be negative.");
    }

    /// <summary>Waveforms <c>[samples]</c> or <c>[B, samples]</c> as rows <c>[B, samples]</c>.</summary>
    public Tensor<T> Rows(Tensor<T> waveforms) => waveforms.Rank switch
    {
        1 => _engine.Reshape(waveforms, new[] { 1, waveforms.Length }),
        2 => waveforms,
        _ => throw new ArgumentException($"AST takes waveforms [samples] or [batch, samples], got rank {waveforms.Rank}.", nameof(waveforms)),
    };

    /// <summary>The network input <c>[B, TargetLength, NumMels]</c> of waveforms <c>[samples]</c> or <c>[B, samples]</c>.</summary>
    public Tensor<T> Features(Tensor<T> waveforms) => Features(Rows(waveforms), masks: null);

    private Tensor<T> Features(Tensor<T> rows, Random? masks)
    {
        int b = rows.Shape[0], s = rows.Shape[1];
        var batch = new Tensor<T>[b];
        for (int i = 0; i < b; i++)
        {
            var wave = _engine.Reshape(_engine.TensorSlice(rows, new[] { i, 0 }, new[] { 1, s }), new[] { s });
            // waveform − mean: the official loader removes the clip's DC before the filterbank.
            var mean = _engine.ReduceMean(wave, new[] { 0 }, true);
            var fbank = _engine.Reshape(Fit(_fbank.Forward(_engine.TensorSubtract(wave, _engine.TensorBroadcastTo(mean, new[] { s })))),
                new[] { 1, _options.TargetLength, _options.NumMels });
            if (masks is not null) fbank = SpecAugment(fbank, masks);
            batch[i] = fbank;
        }
        var stacked = b == 1 ? batch[0] : _engine.TensorConcatenate(batch, 0);
        // (x − mean) / (2·std).
        var scaled = _engine.TensorAddScalar(stacked, NumOps.FromDouble(-_options.DatasetMean));
        return _engine.TensorMultiplyScalar(scaled, NumOps.FromDouble(1.0 / (2.0 * _options.DatasetStd)));
    }

    /// <summary>Zero-pads the end of, or cuts, a <c>[frames, mel]</c> filterbank to the target length.</summary>
    private Tensor<T> Fit(Tensor<T> fbank)
    {
        int frames = fbank.Shape[0], target = _options.TargetLength, mel = _options.NumMels;
        if (frames == target) return fbank;
        if (frames > target) return _engine.TensorSlice(fbank, new[] { 0, 0 }, new[] { target, mel });
        return _engine.TensorConcatenate(new[] { fbank, new Tensor<T>(new[] { target - frames, mel }) }, 0);
    }

    /// <summary>torchaudio's FrequencyMasking then TimeMasking on one <c>[1, frames, mel]</c> filterbank: a length
    /// drawn uniformly below the maximum and a start drawn uniformly in what remains, the band set to zero.</summary>
    private Tensor<T> SpecAugment(Tensor<T> fbank, Random random)
    {
        int frames = _options.TargetLength, mel = _options.NumMels;
        var (f0, f1) = MaskBand(_options.FrequencyMaskLength, mel, random);
        var (t0, t1) = MaskBand(_options.TimeMaskLength, frames, random);
        var keep = new Tensor<T>(new[] { 1, frames, mel });
        for (int t = 0; t < frames; t++)
            for (int m = 0; m < mel; m++)
                keep[0, t, m] = (m >= f0 && m < f1) || (t >= t0 && t < t1) ? NumOps.Zero : NumOps.One;
        return _engine.TensorMultiply(fbank, keep);
    }

    // torchaudio mask_along_axis: value = U·param; start = U·(size − value); masked [⌊start⌋, ⌊start + value⌋).
    private static (int Start, int End) MaskBand(int maximum, int size, Random random)
    {
        if (maximum <= 0) return (0, 0);
        double value = random.NextDouble() * maximum, start = random.NextDouble() * (size - value);
        return ((int)Math.Truncate(start), (int)Math.Truncate(start + value));
    }

    /// <summary>
    /// One training batch as the official loader builds it: each clip is, with probability <see cref="ASTOptions.MixupRate"/>,
    /// mixed with a clip drawn from the batch (waveforms and labels both by λ ~ Beta(10, 10)), then filterbanked, fitted,
    /// SpecAugmented and normalized. Returns the network input and the (mixed) targets.
    /// </summary>
    public (Tensor<T> Features, Tensor<T> Targets) TrainingBatch(Tensor<T> waveforms, Tensor<T> labels, Random random)
    {
        var rows = Rows(waveforms);
        int b = rows.Shape[0], s = rows.Shape[1], classes = _classes;
        var targets = labels.Rank == 1 ? _engine.Reshape(labels, new[] { 1, labels.Length }) : labels;
        if (targets.Shape[0] != b || targets.Shape[1] != classes)
            throw new ArgumentException($"The labels must be [{b}, {classes}] multi-hot rows, got [{string.Join(", ", targets.Shape.ToArray())}].", nameof(labels));
        var mixedRows = new Tensor<T>[b];
        var mixedTargets = new Tensor<T>[b];
        for (int i = 0; i < b; i++)
        {
            var wave = _engine.TensorSlice(rows, new[] { i, 0 }, new[] { 1, s });
            var label = _engine.TensorSlice(targets, new[] { i, 0 }, new[] { 1, classes });
            if (random.NextDouble() < _options.MixupRate)
            {
                int j = random.Next(b);
                double lambda = Beta(10, random);
                var other = _engine.TensorSlice(rows, new[] { j, 0 }, new[] { 1, s });
                var otherLabel = _engine.TensorSlice(targets, new[] { j, 0 }, new[] { 1, classes });
                wave = _engine.TensorAdd(_engine.TensorMultiplyScalar(wave, NumOps.FromDouble(lambda)),
                    _engine.TensorMultiplyScalar(other, NumOps.FromDouble(1 - lambda)));
                label = _engine.TensorAdd(_engine.TensorMultiplyScalar(label, NumOps.FromDouble(lambda)),
                    _engine.TensorMultiplyScalar(otherLabel, NumOps.FromDouble(1 - lambda)));
            }
            mixedRows[i] = wave;
            mixedTargets[i] = label;
        }
        var mixed = b == 1 ? mixedRows[0] : _engine.TensorConcatenate(mixedRows, 0);
        var mixedLabels = b == 1 ? mixedTargets[0] : _engine.TensorConcatenate(mixedTargets, 0);
        return (Features(mixed, random), mixedLabels);
    }

    /// <summary>The mean binary cross-entropy of logits against (possibly mixed) targets: softplus(x) − x·t.</summary>
    public Tensor<T> Loss(Tensor<T> logits, Tensor<T> targets)
        => _engine.ReduceMean(_engine.TensorSubtract(_engine.Softplus(logits), _engine.TensorMultiply(logits, targets)),
            Enumerable.Range(0, logits.Rank).ToArray(), false);

    /// <summary>Beta(a, a) as G₁ / (G₁ + G₂) with Gamma(a) draws; an integer shape is a sum of exponentials.</summary>
    private static double Beta(int a, Random random)
    {
        double Gamma()
        {
            double sum = 0;
            for (int k = 0; k < a; k++) sum -= Math.Log(1.0 - random.NextDouble());
            return sum;
        }
        double g1 = Gamma(), g2 = Gamma();
        return g1 / (g1 + g2);
    }
}

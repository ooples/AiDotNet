using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Audio.Codecs;

/// <summary>Options shared by the neural audio codecs (encoder → residual vector quantizer → decoder).</summary>
public class NeuralAudioCodecOptions : ModelOptions
{
    /// <summary>Gets or sets the audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 24000;

    /// <summary>Gets or sets the number of audio channels.</summary>
    public int Channels { get; set; } = 1;

    /// <summary>Gets or sets the most residual codebooks the quantizer holds.</summary>
    public int NumQuantizers { get; set; } = 32;

    /// <summary>Gets or sets the entries per codebook.</summary>
    public int CodebookSize { get; set; } = 1024;

    /// <summary>Gets or sets the bandwidth (kbps) <see cref="IAudioCodec{T}.Encode"/> and prediction use.</summary>
    public double TargetBandwidthKbps { get; set; } = 6.0;

    /// <summary>Gets or sets the training segment length in samples.</summary>
    public int SegmentSize { get; set; } = 24000;

    /// <summary>Gets or sets the seed of the training draws (segment offsets, bandwidths, discriminator updates).</summary>
    public int SamplingSeed { get; set; }

    /// <summary>
    /// Gets or sets whether the codec builds the discriminators its adversarial training needs (true). A codec used only
    /// to encode and decode, such as the tokenizer and vocoder inside a codec language model, sets this to false: it
    /// then allocates only its encoder, quantizer and decoder, and refuses to train.
    /// </summary>
    public bool IncludeDiscriminators { get; set; } = true;

    /// <summary>Gets or sets the path of an ONNX model with the encoder and decoder in one graph.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the path of an ONNX encoder.</summary>
    public string? EncoderModelPath { get; set; }

    /// <summary>Gets or sets the path of an ONNX decoder.</summary>
    public string? DecoderModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
}

/// <summary>
/// The shared pipeline of the neural audio codecs (SoundStream, EnCodec, DAC, SpeechTokenizer): an encoder to a latent
/// sequence, a residual vector quantizer whose number of codebooks sets the bandwidth, and a decoder back to audio,
/// trained adversarially on random segments with one bandwidth drawn per step.
/// </summary>
/// <remarks>
/// <para>A model supplies its encoder, decoder and quantizer (<see cref="CreateCodec"/>), its discriminators, its generator
/// and discriminator objectives and its optimizers. The base owns segment and bandwidth sampling, the alternating
/// discriminator and generator steps (each updating only its own layers), the <see cref="IAudioCodec{T}"/> surface and
/// a deterministic reconstruction objective for evaluation.</para>
/// <para>Each step trains the discriminator first (with <see cref="DiscriminatorUpdateProbability"/>) on a generation
/// computed without updating the codebooks, then the generator; the generation is the same, as the weights have not
/// changed in between.</para>
/// </remarks>
public abstract class NeuralAudioCodecBase<T> : AudioNeuralNetworkBase<T>, IAudioCodec<T>, ITrainingObjectiveProvider<T>
{
    private readonly NeuralAudioCodecOptions _codec;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _suppliedOptimizer;
    private readonly Dictionary<string, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>> _optimizers = new();
    private readonly bool _useNativeMode;
    private readonly List<LayerBase<T>> _generatorLayers = new();
    private readonly List<LayerBase<T>> _discriminatorLayers = new();
    private IReadOnlyList<ILayer<T>>? _trainingLayers;
    private Random _trainingRandom;
    private long _trainingSteps;
    private bool _disposed;

    /// <summary>Creates a trainable codec.</summary>
    protected NeuralAudioCodecBase(NeuralNetworkArchitecture<T> architecture, NeuralAudioCodecOptions options,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer)
        : base(architecture)
    {
        _codec = options ?? throw new ArgumentNullException(nameof(options));
        // The clone plan replays the constructor from members named after its parameters.
        Options = _codec;
        _useNativeMode = true;
        _suppliedOptimizer = optimizer;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_codec.SamplingSeed);
        base.SampleRate = _codec.SampleRate;
        InitializeLayers();
    }

    /// <summary>Creates a codec that runs exported ONNX graphs (one model, or an encoder and a decoder).</summary>
    protected NeuralAudioCodecBase(NeuralNetworkArchitecture<T> architecture, string modelPath, NeuralAudioCodecOptions options)
        : base(architecture)
    {
        _codec = options ?? throw new ArgumentNullException(nameof(options));
        Options = _codec;
        _useNativeMode = false;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_codec.SamplingSeed);
        base.SampleRate = _codec.SampleRate;
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _codec.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(_codec.EncoderModelPath ?? modelPath, _codec.OnnxOptions);
        if (_codec.DecoderModelPath is not null) OnnxDecoder = new OnnxModel<T>(_codec.DecoderModelPath, _codec.OnnxOptions);
        InitializeLayers();
    }

    /// <inheritdoc />
    public override ModelOptions GetOptions() => _codec;

    /// <summary>The codec options.</summary>
    protected NeuralAudioCodecOptions CodecSettings => _codec;

    /// <summary>Whether the paper layers were built (false when the architecture supplied its own layers).</summary>
    protected bool HasPaperLayers => _generatorLayers.Count > 0;

    /// <summary>Generator steps taken.</summary>
    protected long TrainingSteps => _trainingSteps;

    // ---------------------------------------------------------------- hooks

    /// <summary>Builds the encoder, quantizer and decoder, adding their layers to <paramref name="layers"/>; returns the
    /// quantizer.</summary>
    protected abstract IResidualVectorQuantizer<T> CreateCodec(List<LayerBase<T>> layers);

    /// <summary>Builds the discriminators' layers.</summary>
    protected abstract IReadOnlyList<LayerBase<T>> CreateDiscriminators();

    /// <summary>The latent <c>[1, dim, frames]</c> of audio <c>[1, channels, samples]</c>.</summary>
    protected abstract Tensor<T> EncodeLatent(Tensor<T> audio);

    /// <summary>The audio <c>[1, channels, samples]</c> of a latent <c>[1, dim, frames]</c>.</summary>
    protected abstract Tensor<T> DecodeLatent(Tensor<T> latent);

    /// <summary>Samples per latent frame.</summary>
    public abstract int HopLength { get; }

    /// <summary>Draws the number of codebooks for one training step and the index of its bandwidth.</summary>
    protected abstract (int Quantizers, int Bandwidth) SampleTrainingBandwidth(Random random);

    /// <summary>The generator's objective for a real segment and its reconstruction (a tape-connected scalar).</summary>
    protected abstract Tensor<T> GeneratorObjective(Tensor<T> real, Tensor<T> generated, int bandwidth);

    /// <summary>The discriminator's objective for a real segment and a detached reconstruction.</summary>
    protected abstract Tensor<T> DiscriminatorObjective(Tensor<T> real, Tensor<T> generated, int bandwidth);

    /// <summary>A deterministic reconstruction measure (lower is better) for evaluation.</summary>
    protected abstract Tensor<T> ReconstructionObjective(Tensor<T> real, Tensor<T> generated);

    /// <summary>The paper optimizer of a parameter group ("generator" or "discriminator").</summary>
    protected abstract IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreatePaperOptimizer(string group);

    /// <summary>The discriminator layers a step at <paramref name="bandwidth"/> trains (all by default; EnCodec keeps one
    /// discriminator per bandwidth and updates only that one).</summary>
    protected virtual IReadOnlyList<LayerBase<T>> DiscriminatorLayersFor(int bandwidth) => _discriminatorLayers;

    /// <summary>The optimizer group of the discriminator trained at <paramref name="bandwidth"/> ("discriminator" by
    /// default). Discriminators with their own group keep their own optimizer state.</summary>
    protected virtual string DiscriminatorGroup(int bandwidth) => "discriminator";

    /// <summary>The probability that a step also updates the discriminator (1).</summary>
    protected virtual double DiscriminatorUpdateProbability => 1.0;

    /// <summary>The loss reported for a generator step (the default reports the objective itself).</summary>
    protected virtual double? ReportedGeneratorLoss => null;

    /// <summary>The quantizer.</summary>
    protected IResidualVectorQuantizer<T>? Quantizer { get; private set; }

    // ---------------------------------------------------------------- layers

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            return;
        }
        Quantizer = CreateCodec(_generatorLayers);
        if (_codec.IncludeDiscriminators)
            _discriminatorLayers.AddRange(CreateDiscriminators());
        Layers.AddRange(_generatorLayers);
    }

    /// <inheritdoc />
    /// <remarks>The discriminators run outside the encoder–quantizer–decoder stack; surfacing them here makes them take
    /// part in parameter counting, training, serialization and cloning as the stack's layers do.</remarks>
    protected override IEnumerable<LayerBase<T>?> GetExtraTrainableLayers()
    {
        foreach (var layer in base.GetExtraTrainableLayers())
            yield return layer;
        foreach (var layer in _discriminatorLayers)
            yield return layer;
    }

    // ---------------------------------------------------------------- shapes

    /// <summary>Audio as <c>[1, channels, samples]</c> from <c>[samples]</c>, <c>[channels, samples]</c> or itself.</summary>
    protected Tensor<T> AsBatch(Tensor<T> audio) => audio.Rank switch
    {
        1 => Engine.Reshape(audio, new[] { 1, 1, audio.Length }),
        2 => Engine.Reshape(audio, new[] { 1, audio.Shape[0], audio.Shape[1] }),
        3 => audio,
        _ => throw new ArgumentException($"Expected audio [samples], [channels, samples] or [1, channels, samples], got rank {audio.Rank}.", nameof(audio)),
    };

    /// <summary>The number of codebooks for a bandwidth in kbps (each carries frame rate × log2(bins) bits per second).</summary>
    public int QuantizersForBandwidth(double kbps)
    {
        double perCodebook = (double)_codec.SampleRate / HopLength * Math.Log(_codec.CodebookSize, 2) / 1000.0;
        return Math.Max(1, Math.Min(_codec.NumQuantizers, (int)Math.Floor(kbps / perCodebook + 1e-9)));
    }

    /// <summary>Encodes, quantizes with <paramref name="quantizers"/> codebooks and decodes <c>[1, channels, samples]</c>.</summary>
    protected virtual Tensor<T> Reconstruct(Tensor<T> audio, int quantizers)
    {
        var latent = EncodeLatent(audio);
        Quantizer!.ActiveQuantizers = quantizers;
        return DecodeLatent(Quantizer.Forward(latent));
    }

    // ---------------------------------------------------------------- IAudioCodec

    /// <inheritdoc />
    int IAudioCodec<T>.SampleRate => _codec.SampleRate;

    /// <inheritdoc />
    public int NumQuantizers => _codec.NumQuantizers;

    /// <inheritdoc />
    public int CodebookSize => _codec.CodebookSize;

    /// <inheritdoc />
    public int TokenFrameRate => _codec.SampleRate / HopLength;

    /// <inheritdoc />
    public virtual int[,] Encode(Tensor<T> audio)
    {
        ThrowIfDisposed();
        RequireNative();
        SetTrainingMode(false);
        using var _ = new NoGradScope<T>();
        return Quantizer!.Encode(EncodeLatent(AsBatch(audio)), QuantizersForBandwidth(_codec.TargetBandwidthKbps));
    }

    /// <inheritdoc />
    public Task<int[,]> EncodeAsync(Tensor<T> audio, CancellationToken cancellationToken = default)
        => Task.Run(() => Encode(audio), cancellationToken);

    /// <inheritdoc />
    public virtual Tensor<T> Decode(int[,] tokens)
    {
        ThrowIfDisposed();
        RequireNative();
        if (tokens.GetLength(0) > _codec.NumQuantizers)
            throw new ArgumentException($"The codec has {_codec.NumQuantizers} codebooks; got codes for {tokens.GetLength(0)}.", nameof(tokens));
        SetTrainingMode(false);
        using var _ = new NoGradScope<T>();
        return DecodeLatent(Quantizer!.Decode(tokens));
    }

    /// <inheritdoc />
    public Task<Tensor<T>> DecodeAsync(int[,] tokens, CancellationToken cancellationToken = default)
        => Task.Run(() => Decode(tokens), cancellationToken);

    /// <inheritdoc />
    public Tensor<T> EncodeEmbeddings(Tensor<T> audio)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(audio);
        SetTrainingMode(false);
        using var _ = new NoGradScope<T>();
        return EncodeLatent(AsBatch(audio));
    }

    /// <inheritdoc />
    public Tensor<T> DecodeEmbeddings(Tensor<T> embeddings)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxDecoder is not null) return OnnxDecoder.Run(embeddings);
        RequireNative();
        SetTrainingMode(false);
        using var _ = new NoGradScope<T>();
        return DecodeLatent(embeddings.Rank == 2 ? Engine.Reshape(embeddings, new[] { 1, embeddings.Shape[0], embeddings.Shape[1] }) : embeddings);
    }

    /// <inheritdoc />
    public double GetBitrate(int? numQuantizers = null)
        => (numQuantizers ?? QuantizersForBandwidth(_codec.TargetBandwidthKbps)) * (double)_codec.SampleRate / HopLength
           * Math.Log(_codec.CodebookSize, 2);

    private void RequireNative()
    {
        if (!HasPaperLayers)
            throw new NotSupportedException($"{GetType().Name} needs its native encoder, quantizer and decoder for this operation.");
    }

    // ---------------------------------------------------------------- prediction

    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        SetTrainingMode(false);
        if (!HasPaperLayers)
        {
            var c = input;
            foreach (var l in Layers) c = l.Forward(c);
            return c;
        }
        using var _ = new NoGradScope<T>();
        var output = Reconstruct(AsBatch(input), QuantizersForBandwidth(_codec.TargetBandwidthKbps));
        return Engine.Reshape(output, input.Rank == 3 ? output._shape : input.Rank == 2 ? new[] { output.Shape[1], output.Shape[2] } : new[] { output.Length });
    }

    /// <inheritdoc />
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) => rawAudio;

    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput) => modelOutput;

    // ---------------------------------------------------------------- training

    /// <inheritdoc />
    /// <remarks>The codec reconstructs its input: <paramref name="input"/> is the audio and <paramref name="expectedOutput"/>
    /// the target it should reproduce (normally the same recording). One step draws a segment, a bandwidth, and updates the
    /// discriminator (with <see cref="DiscriminatorUpdateProbability"/>) and then the generator.</remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperLayers)
        {
            TrainWithTape(input, expectedOutput, _suppliedOptimizer);
            return;
        }
        var (source, target) = Segments(AsBatch(input), AsBatch(expectedOutput));
        TrainOnSegment(source, target);
    }

    /// <summary>The seeded source of the training draws (segment offsets, bandwidths, discriminator updates).</summary>
    protected Random TrainingRandom => _trainingRandom;

    /// <summary>
    /// One training step on a prepared segment: draws the bandwidth, updates the discriminator (with
    /// <see cref="DiscriminatorUpdateProbability"/>) on a generation computed without updating the codebooks, then the
    /// generator.
    /// </summary>
    protected void TrainOnSegment(Tensor<T> source, Tensor<T> target)
    {
        if (!_codec.IncludeDiscriminators)
            throw new InvalidOperationException("This codec was built without discriminators (IncludeDiscriminators = false), " +
                "so it can encode and decode but not train; build it with IncludeDiscriminators = true to train it.");
        var (quantizers, bandwidth) = SampleTrainingBandwidth(_trainingRandom);
        bool updateDiscriminator = _trainingRandom.NextDouble() < DiscriminatorUpdateProbability;
        if (updateDiscriminator && _discriminatorLayers.Count > 0)
        {
            Tensor<T> generated;
            using (new NoGradScope<T>())
            {
                SetTrainingMode(false);
                var g = Reconstruct(source, quantizers);
                generated = new Tensor<T>(g._shape, g.ToVector());
            }
            TrainLayers(DiscriminatorLayersFor(bandwidth), source, target, () => DiscriminatorObjective(target, generated, bandwidth),
                DiscriminatorGroup(bandwidth));
        }
        TrainLayers(_generatorLayers, source, target, () => GeneratorObjective(target, Reconstruct(source, quantizers), bandwidth), "generator");
        if (ReportedGeneratorLoss is double reported) LastLoss = NumOps.FromDouble(reported);
        _trainingSteps++;
    }

    // Matching segments of SegmentSize samples (the whole recording, zero-padded, when shorter).
    private (Tensor<T> Source, Tensor<T> Target) Segments(Tensor<T> source, Tensor<T> target)
    {
        int length = Math.Min(source.Shape[2], target.Shape[2]), size = _codec.SegmentSize;
        if (length >= size)
        {
            int start = _trainingRandom.Next(0, length - size + 1);
            return (Engine.TensorSlice(source, new[] { 0, 0, start }, new[] { 1, source.Shape[1], size }),
                Engine.TensorSlice(target, new[] { 0, 0, start }, new[] { 1, target.Shape[1], size }));
        }
        return (Seanet.Pad(Engine, Engine.TensorSlice(source, new[] { 0, 0, 0 }, new[] { 1, source.Shape[1], length }), 0, size - length, reflect: false),
            Seanet.Pad(Engine, Engine.TensorSlice(target, new[] { 0, 0, 0 }, new[] { 1, target.Shape[1], length }), 0, size - length, reflect: false));
    }

    private void TrainLayers(IReadOnlyList<LayerBase<T>> layers, Tensor<T> source, Tensor<T> target, Func<Tensor<T>> objective, string group)
    {
        _trainingLayers = layers.Cast<ILayer<T>>().ToList();
        try
        {
            TrainWithCustomObjective(source, target, (_, _) => objective(), OptimizerFor(group));
        }
        finally
        {
            _trainingLayers = null;
        }
    }

    /// <inheritdoc />
    /// <remarks>Each step updates only the generator or only the discriminators.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers || _trainingLayers is null) return parameters;
        var selected = new HashSet<Tensor<T>>(AiDotNet.Training.TapeTrainingStep<T>.CollectParameters(_trainingLayers.ToList(), -1),
            AiDotNet.Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(selected.Contains).ToList();
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> OptimizerFor(string group)
    {
        if (_suppliedOptimizer is not null) return _suppliedOptimizer;
        if (!_optimizers.TryGetValue(group, out var optimizer))
            _optimizers[group] = optimizer = CreatePaperOptimizer(group);
        return optimizer;
    }

    /// <inheritdoc />
    protected override void OnMutableConstructorConfigurationRestored()
    {
        base.OnMutableConstructorConfigurationRestored();
        _optimizers.Clear();
    }

    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => input;

    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        ThrowIfDisposed();
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var real = AsBatch(target);
            return ReconstructionObjective(real, Reconstruct(AsBatch(input), QuantizersForBandwidth(_codec.TargetBandwidthKbps)))[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    // ---------------------------------------------------------------- helpers

    /// <summary>The mean of every element, as a scalar tensor.</summary>
    protected Tensor<T> Mean(Tensor<T> x) => Engine.ReduceMean(x, Enumerable.Range(0, x.Rank).ToArray(), keepDims: false);

    /// <summary>The sum of every element, as a scalar tensor.</summary>
    protected Tensor<T> Total(Tensor<T> x) => Engine.ReduceSum(x, Enumerable.Range(0, x.Rank).ToArray(), keepDims: false);

    /// <summary>The sum of <paramref name="terms"/>.</summary>
    protected Tensor<T> Sum(IEnumerable<Tensor<T>> terms)
    {
        Tensor<T>? total = null;
        foreach (var t in terms) total = total is null ? t : Engine.TensorAdd(total, t);
        return total ?? throw new ArgumentException("No terms to sum.", nameof(terms));
    }

    /// <summary>A copy of <paramref name="x"/> that carries no gradient.</summary>
    protected static Tensor<T> Detached(Tensor<T> x) => new(x._shape, x.ToVector());

    /// <summary>A waveform as <c>[samples]</c> (mono) — the discriminators' and spectral losses' input.</summary>
    protected Tensor<T> Flat(Tensor<T> audio) => Engine.Reshape(audio, new[] { audio.Length });

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <summary>Throws when the codec was disposed.</summary>
    protected void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? GetType().Name);
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}

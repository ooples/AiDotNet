using System.Diagnostics.CodeAnalysis;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Engines;
using AiDotNet.Enums;
using AiDotNet.Extensions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.Audio;

/// <summary>
/// DiffWave model for high-quality audio waveform synthesis using diffusion.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DiffWave is a versatile diffusion model for raw audio waveform synthesis.
/// It uses a non-autoregressive architecture with dilated convolutions to
/// achieve high-quality audio generation with fast inference.
/// </para>
/// <para>
/// <b>For Beginners:</b> DiffWave generates audio (like speech or music)
/// directly as a waveform - the actual audio signal that speakers play.
///
/// Unlike spectrograms (visual representations of sound), DiffWave creates:
/// - Raw audio samples that can be played directly
/// - High-quality, natural-sounding audio
/// - Various audio types: speech, music, sound effects
///
/// How it works:
/// 1. Start with random noise (static)
/// 2. Gradually refine it into clear audio
/// 3. Use dilated convolutions to understand audio context
/// 4. Optionally condition on mel-spectrograms or text
///
/// Applications:
/// - Text-to-speech synthesis
/// - Music generation
/// - Audio super-resolution
/// - Neural vocoders
/// </para>
/// <para>
/// Technical details:
/// - Non-autoregressive: generates all samples in parallel
/// - Dilated convolutions: capture long-range audio dependencies
/// - Mel-spectrogram conditioning: for speech synthesis
/// - Fast inference compared to autoregressive models
/// - Supports variable-length audio generation
///
/// Reference: Kong et al., "DiffWave: A Versatile Diffusion Model for Audio Synthesis", 2020
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a DiffWave model
/// var diffWave = new DiffWaveModel&lt;float&gt;();
///
/// // Generate unconditional audio
/// var audio = diffWave.GenerateAudio(
///     sampleLength: 16000,  // 1 second at 16kHz
///     numInferenceSteps: 50);
///
/// // Generate audio from mel-spectrogram (vocoder mode)
/// var melSpec = ComputeMelSpectrogram(text);
/// var vocodedAudio = diffWave.GenerateFromMelSpectrogram(melSpec);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.Generation)]
[ModelTask(ModelTask.TextToSpeech)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("DiffWave: A Versatile Diffusion Model for Audio Synthesis", "https://arxiv.org/abs/2009.09761", Year = 2020, Authors = "Kong et al.")]
public class DiffWaveModel<T> : DiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Default sample rate in Hz.
    /// </summary>
    /// <remarks>22050 Hz is a standard sample rate for speech synthesis, providing good quality with manageable compute.</remarks>
    private const int DEFAULT_SAMPLE_RATE = 22050;

    #endregion

    #region Fields

    /// <summary>
    /// Number of residual channels.
    /// </summary>
    private readonly int _residualChannels;

    /// <summary>
    /// Number of residual layers.
    /// </summary>
    private readonly int _residualLayers;

    /// <summary>
    /// Dilation cycle length.
    /// </summary>
    private readonly int _dilationCycle;

    /// <summary>
    /// Number of mel-spectrogram channels for conditioning.
    /// </summary>
    private readonly int _melChannels;

    /// <summary>
    /// The diffusion network.
    /// </summary>
    private DiffWaveNetwork<T> _network;

    /// <summary>
    /// Last audio input shape seen by <see cref="PredictNoise"/>. Used by
    /// <see cref="Clone"/> to replay lazy DenseLayer shape resolution on
    /// the cloned network so its layers have the same parameter layout
    /// as the original before parameters are copied across.
    /// </summary>
    private int[]? _lastInputShape;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override long ParameterCount => _network.ParameterCount;

    /// <summary>
    /// Gets the sample rate in Hz.
    /// </summary>
    public int SampleRate { get; }

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of DiffWaveModel with full customization support.
    /// </summary>
    /// <param name="architecture">Optional neural network architecture for custom layer configuration.</param>
    /// <param name="options">Configuration options.</param>
    /// <param name="scheduler">Optional custom scheduler.</param>
    /// <param name="residualChannels">Number of residual channels.</param>
    /// <param name="residualLayers">Number of residual layers.</param>
    /// <param name="dilationCycle">Dilation cycle length.</param>
    /// <param name="melChannels">Number of mel-spectrogram channels.</param>
    /// <param name="sampleRate">Audio sample rate in Hz.</param>
    /// <param name="seed">Optional random seed.</param>
    public DiffWaveModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        int residualChannels = 64,
        int residualLayers = 30,
        int dilationCycle = 10,
        int melChannels = 80,
        int sampleRate = DEFAULT_SAMPLE_RATE,
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 200,
                BetaStart = 0.0001,
                BetaEnd = 0.05,
                BetaSchedule = BetaSchedule.Linear
            },
            scheduler ?? new DDIMScheduler<T>(new SchedulerConfig<T>(
                trainTimesteps: 200,
                betaStart: MathHelper.GetNumericOperations<T>().FromDouble(0.0001),
                betaEnd: MathHelper.GetNumericOperations<T>().FromDouble(0.05),
                betaSchedule: BetaSchedule.Linear,
                predictionType: DiffusionPredictionType.Epsilon)),
            architecture)
    {
        _residualChannels = residualChannels;
        _residualLayers = residualLayers;
        _dilationCycle = dilationCycle;
        _melChannels = melChannels;
        SampleRate = sampleRate;

        InitializeLayers(residualChannels, residualLayers, dilationCycle, melChannels, seed);
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes the model layers.
    /// </summary>
    [MemberNotNull(nameof(_network))]
    private void InitializeLayers(
        int residualChannels,
        int residualLayers,
        int dilationCycle,
        int melChannels,
        int? seed)
    {
        _network = new DiffWaveNetwork<T>(
            engine: Engine,
            residualChannels: residualChannels,
            residualLayers: residualLayers,
            dilationCycle: dilationCycle,
            melChannels: melChannels,
            seed: seed);
    }

    #endregion

    #region Generation Methods

    /// <summary>
    /// Generates unconditional audio.
    /// </summary>
    /// <param name="sampleLength">Length of audio in samples.</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Generated audio waveform tensor [1, sampleLength].</returns>
    public virtual Tensor<T> GenerateAudio(
        int sampleLength,
        int numInferenceSteps = 50,
        int? seed = null)
    {
        return GenerateFromMelSpectrogram(null, sampleLength, numInferenceSteps, seed);
    }

    /// <summary>
    /// Generates audio from a mel-spectrogram (vocoder mode).
    /// </summary>
    /// <param name="melSpectrogram">Mel-spectrogram tensor [batch, melChannels, frames].</param>
    /// <param name="sampleLength">Optional target sample length.</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Generated audio waveform tensor.</returns>
    public virtual Tensor<T> GenerateFromMelSpectrogram(
        Tensor<T>? melSpectrogram = null,
        int? sampleLength = null,
        int numInferenceSteps = 50,
        int? seed = null)
    {
        // Determine sample length from mel or use provided
        int length;
        if (melSpectrogram != null)
        {
            // Calculate from mel frames (assuming hop_size = 256)
            var frames = melSpectrogram.Shape[^1];
            length = frames * 256;
        }
        else
        {
            length = sampleLength ?? SampleRate; // Default to 1 second
        }

        var shape = new[] { 1, length };

        // Initialize with noise
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;
        var audio = SampleNoise(shape, rng);

        // Set up scheduler
        Scheduler.SetTimesteps(numInferenceSteps);

        // Denoising loop
        foreach (var timestep in Scheduler.Timesteps)
        {
            // Predict noise
            var noisePrediction = _network.Forward(audio, timestep, melSpectrogram);

            // Scheduler step
            var audioVector = audio.ToVector();
            var noiseVector = noisePrediction.ToVector();
            audioVector = Scheduler.Step(noiseVector, timestep, audioVector, NumOps.Zero);
            audio = new Tensor<T>(shape, audioVector);
        }

        return audio;
    }

    /// <summary>
    /// Generates a batch of audio samples.
    /// </summary>
    /// <param name="batchSize">Number of samples to generate.</param>
    /// <param name="sampleLength">Length of each sample.</param>
    /// <param name="numInferenceSteps">Number of steps.</param>
    /// <param name="seed">Random seed.</param>
    /// <returns>Batch of audio tensors [batch, sampleLength].</returns>
    public virtual Tensor<T> GenerateBatch(
        int batchSize,
        int sampleLength,
        int numInferenceSteps = 50,
        int? seed = null)
    {
        var shape = new[] { batchSize, sampleLength };

        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;
        var audio = SampleNoise(shape, rng);

        Scheduler.SetTimesteps(numInferenceSteps);

        foreach (var timestep in Scheduler.Timesteps)
        {
            var noisePrediction = _network.Forward(audio, timestep, null);

            var audioVector = audio.ToVector();
            var noiseVector = noisePrediction.ToVector();
            audioVector = Scheduler.Step(noiseVector, timestep, audioVector, NumOps.Zero);
            audio = new Tensor<T>(shape, audioVector);
        }

        return audio;
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Samples noise for audio generation.
    /// </summary>
    private Tensor<T> SampleNoise(int[] shape, Random rng)
    {
        var totalSize = shape.Aggregate(1, (a, b) => a * b);
        var data = new T[totalSize];

        for (int i = 0; i < totalSize; i++)
        {
            data[i] = NumOps.FromDouble(rng.NextGaussian());
        }

        return new Tensor<T>(shape, new Vector<T>(data));
    }

    /// <inheritdoc />
    public override Tensor<T> PredictNoise(Tensor<T> noisySample, int timestep)
    {
        // Remember the input shape so Clone() can replay lazy shape
        // resolution on the cloned network — the downstream DenseLayers
        // project the LAST tensor dim, so their parameter count depends
        // on the audio's time axis. Without this, the clone's layers
        // would lazily initialize to a different shape on first Predict
        // and SetParameters would reject the original's parameter
        // vector with an "Expected X parameters, got Y" mismatch.
        _lastInputShape = (int[])noisySample._shape.Clone();
        return _network.Forward(noisySample, timestep, null);
    }

    #endregion

    #region IParameterizable Implementation

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        return _network.GetParameters();
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int expected = checked((int)_network.ParameterCount);
        if (parameters.Length != expected)
            throw new ArgumentException($"Expected {expected} parameters, got {parameters.Length}.");

        _network.SetParameters(parameters);
    }

    #endregion

    #region ICloneable Implementation

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        return Clone();
    }

    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        var clone = new DiffWaveModel<T>(
            residualChannels: _residualChannels,
            residualLayers: _residualLayers,
            dilationCycle: _dilationCycle,
            melChannels: _melChannels,
            sampleRate: SampleRate,
            seed: RandomGenerator.Next());

        if (_lastInputShape is not null)
        {
            clone._network.ResolveLayerShapesFor(_lastInputShape);
            clone._lastInputShape = (int[])_lastInputShape.Clone();
        }
        if (!clone.TryShareParametersFrom(this)) clone.SetParameters(GetParameters());
        return clone;
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "DiffWaveModel",
            Version = "1.0.0",
            Description = "DiffWave model for audio waveform synthesis using diffusion",
            FeatureCount = SampleRate,
            Complexity = ParameterCount
        };

        metadata.SetProperty("ResidualChannels", _residualChannels);
        metadata.SetProperty("ResidualLayers", _residualLayers);
        metadata.SetProperty("DilationCycle", _dilationCycle);
        metadata.SetProperty("MelChannels", _melChannels);
        metadata.SetProperty("SampleRate", SampleRate);

        return metadata;
    }

    #endregion
}

/// <summary>
/// DiffWave neural network with dilated convolutions.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class DiffWaveNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly int _residualChannels;
    private readonly int _residualLayers;
    private readonly int _dilationCycle;
    private readonly int _melChannels;

    // Per Kong et al. 2020 "DiffWave" §3 / Figure 1, the spatial layers
    // are dilated 1D convolutions over the audio time axis — Conv1D with
    // kernel=1 for the input/output 1×1 projections, Conv1D with
    // kernel=3 and exponentially increasing dilation for the dilated
    // stack. The diffusion timestep embedding remains a plain
    // fully-connected MLP because the timestep is a single scalar per
    // sample (no temporal axis to convolve over).
    private readonly Conv1DLayer<T> _inputProjection;     // Kong 2020: Conv1×1
    private readonly DenseLayer<T> _diffusionEmbedding;   // FC MLP on sinusoidal embed
    private readonly List<DiffWaveResidualBlock<T>> _residualBlocks;
    private readonly Conv1DLayer<T> _outputProjection1;   // Kong 2020: Conv1×1
    private readonly Conv1DLayer<T> _outputProjection2;   // Kong 2020: Conv1×1 to 1ch

    /// <summary>
    /// Gets the number of parameters.
    /// </summary>
    /// <remarks>
    /// Live-computed from the underlying layers (which use lazy shape
    /// resolution) instead of snapshotting at construction time. See
    /// <see cref="DiffWaveResidualBlock{T}.ParameterCount"/> for the
    /// rationale — the same lazy-init pitfall applies at the network
    /// level once any sub-layer's count is also live.
    /// </remarks>
    public long ParameterCount
    {
        get
        {
            long total = _inputProjection.ParameterCount +
                         _diffusionEmbedding.ParameterCount +
                         _outputProjection1.ParameterCount +
                         _outputProjection2.ParameterCount;
            foreach (var block in _residualBlocks)
            {
                // Per-block ParameterCount is long; accumulate at full
                // width to avoid silent wrap if any block crosses
                // int.MaxValue.
                total += block.ParameterCount;
            }
            return total;
        }
    }

    /// <summary>
    /// The engine instance used for all tape-tracked tensor ops in this
    /// network and its residual blocks. Injected by the outer
    /// <see cref="DiffWaveModel{T}"/> from its base-class <c>Engine</c> so
    /// custom engines (test fakes, alternate backends, profiling wrappers)
    /// propagate end-to-end instead of leaking back to the
    /// <c>AiDotNetEngine.Current</c> singleton.
    /// </summary>
    private readonly IEngine _engine;

    /// <summary>
    /// Initializes a new DiffWaveNetwork.
    /// </summary>
    public DiffWaveNetwork(
        IEngine engine,
        int residualChannels = 64,
        int residualLayers = 30,
        int dilationCycle = 10,
        int melChannels = 80,
        int? seed = null)
    {
        _engine = engine ?? throw new ArgumentNullException(nameof(engine));
        _residualChannels = residualChannels;
        _residualLayers = residualLayers;
        _dilationCycle = dilationCycle;
        _melChannels = melChannels;

        // Input projection — 1×1 Conv1D from 1 audio channel to
        // residualChannels (Kong 2020 §3, Figure 1: "Conv 1×1" before the
        // residual stack). Eager-init: input is mono audio (1 channel)
        // so we can fully allocate weights at construction time.
        _inputProjection = new Conv1DLayer<T>(
            inputChannels: 1, outputChannels: residualChannels, kernelSize: 1);

        // Diffusion timestep embedding (Kong 2020 §3.2): the timestep
        // index t is mapped through a sinusoidal positional encoding then
        // a 2-layer FC MLP. The MLP is per-sample (no time axis), so a
        // plain DenseLayer is paper-correct here — NOT a Conv1D.
        _diffusionEmbedding = new DenseLayer<T>(residualChannels, (IActivationFunction<T>?)null);

        // Residual blocks with varying dilation
        _residualBlocks = new List<DiffWaveResidualBlock<T>>();
        for (int i = 0; i < residualLayers; i++)
        {
            var dilation = (int)Math.Pow(2, i % dilationCycle);
            _residualBlocks.Add(new DiffWaveResidualBlock<T>(
                engine: _engine,
                channels: residualChannels,
                dilation: dilation,
                conditionChannels: melChannels,
                seed: seed));
        }

        // Output projections — both 1×1 Conv1D per Kong 2020 §3 / Figure
        // 1: post-skip path is ReLU → Conv1×1 → ReLU → Conv1×1 to 1 ch.
        // Both have known input channels (skip stream is residualChannels
        // wide), so eager-init.
        _outputProjection1 = new Conv1DLayer<T>(
            inputChannels: residualChannels,
            outputChannels: residualChannels, kernelSize: 1);
        _outputProjection2 = new Conv1DLayer<T>(
            inputChannels: residualChannels,
            outputChannels: 1, kernelSize: 1);

        // ParameterCount is live-computed from layers. Conv1DLayer
        // reports a non-zero ParameterCount even before lazy init
        // (assumes inputChannels=1 placeholder) so the
        // Parameters_ShouldBeNonEmpty invariant passes on a freshly
        // constructed model. Once OnFirstForward runs, the count
        // resolves to the real value — translation-invariant in T
        // because Conv1D has kernel*C_in*C_out + C_out params, not
        // T-dependent params like the old DenseLayer-projects-T-axis
        // anti-pattern.
    }

    /// <summary>
    /// Replays the lazy shape-resolution pass that would happen on the
    /// first <see cref="Forward"/> with the given audio shape, without
    /// keeping the dummy output. Used by <see cref="DiffWaveModel{T}.Clone"/>
    /// to make the cloned network's layers match the original's parameter
    /// layout before <c>SetParameters</c> validates the count.
    /// </summary>
    internal void ResolveLayerShapesFor(int[] audioShape)
    {
        if (audioShape is null) throw new ArgumentNullException(nameof(audioShape));
        if (audioShape.Length < 1)
        {
            throw new ArgumentException(
                "Audio shape must have at least one dimension.", nameof(audioShape));
        }
        var dummyAudio = new Tensor<T>(audioShape);
        _ = Forward(dummyAudio, timestep: 0, melCondition: null);
    }

    /// <summary>
    /// Forward pass through the network — Kong et al. 2020 "DiffWave"
    /// §3 / Figure 1. Channel layout is paper-faithful channels-FIRST
    /// <c>[B, C, T]</c>: Conv1D operates on the time axis, channels are
    /// at <c>dim=1</c>, batch at <c>dim=0</c>.
    /// </summary>
    public Tensor<T> Forward(Tensor<T> audio, int timestep, Tensor<T>? melCondition)
    {
        // Diffusion timestep embedding — sinusoidal -> FC MLP.
        var diffEmbed = GetTimestepEmbedding(timestep);
        diffEmbed = _diffusionEmbedding.Forward(diffEmbed);

        // Input projection: [B, 1, T] -> [B, residualChannels, T] via
        // 1×1 Conv1D. Accepts rank-2 [B, T] (mono audio without explicit
        // channel axis) or rank-3 [B, C, T]; normalize to rank-3 first.
        var x = ProjectAudio(audio);

        // Residual stack — each block returns (out, skip), both
        // [B, residualChannels, T]; outs feed the next block, skips are
        // summed across the stack.
        var skip = CreateZeroTensor(x._shape);
        foreach (var block in _residualBlocks)
        {
            var (blockOut, skipOut) = block.Forward(x, diffEmbed, melCondition);
            x = blockOut;
            skip = AddTensors(skip, skipOut);
        }

        // Output projection: ReLU -> Conv1×1 -> ReLU -> Conv1×1(1ch).
        var output = ApplyRelu(_outputProjection1.Forward(skip));
        output = ApplyRelu(output);
        output = _outputProjection2.Forward(output);

        return ReshapeToAudio(output, audio._shape);
    }

    private Tensor<T> ProjectAudio(Tensor<T> audio)
    {
        // Normalize input to rank-3 [B, C, T] before the 1×1 Conv1D.
        // Common test/inference shapes:
        //   [B, T]        — mono audio, add C=1 axis
        //   [B, 1, T]     — already canonical
        //   [B, C, T]     — multi-channel (passes through)
        //   higher ranks  — collapse leading dims into batch
        Tensor<T> shaped;
        int rank = audio._shape.Length;
        if (rank == 2)
        {
            shaped = _engine.Reshape(audio,
                new[] { audio._shape[0], 1, audio._shape[1] });
        }
        else if (rank == 3)
        {
            shaped = audio;
        }
        else if (rank > 3)
        {
            int batch = 1;
            for (int d = 0; d < rank - 2; d++) batch *= audio._shape[d];
            int channels = audio._shape[rank - 2];
            int time = audio._shape[rank - 1];
            shaped = _engine.Reshape(audio, new[] { batch, channels, time });
        }
        else
        {
            throw new ArgumentException(
                $"DiffWave expects rank >= 2 audio input; got rank {rank}.",
                nameof(audio));
        }
        // _inputProjection was eagerly constructed with inputChannels=1
        // (mono — Kong et al. 2021 DiffWave §3 is a mono vocoder). Multi-
        // channel inputs would reach a kernel sized for a single channel
        // and produce wrong-shape activations downstream, so reject them
        // explicitly at the entry point rather than the earlier comment's
        // "multi-channel passes through" (which is incorrect — it doesn't).
        // To process stereo / multi-channel audio, callers should run one
        // DiffWaveModel instance per channel and combine outputs.
        int channelsAfterReshape = shaped._shape[1];
        if (channelsAfterReshape != 1)
        {
            throw new ArgumentException(
                $"DiffWave requires mono audio (C=1) per Kong et al. 2021 §3; " +
                $"got C={channelsAfterReshape}. Process multi-channel audio one " +
                $"channel at a time, or use a multi-channel vocoder.",
                nameof(audio));
        }
        return _inputProjection.Forward(shaped);
    }

    private Tensor<T> GetTimestepEmbedding(int timestep)
    {
        // Sinusoidal position embedding
        var dim = 128;
        var data = new T[dim];
        var halfDim = dim / 2;

        for (int i = 0; i < halfDim; i++)
        {
            var freq = Math.Exp(-Math.Log(10000.0) * i / halfDim);
            data[i] = NumOps.FromDouble(Math.Sin(timestep * freq));
            data[i + halfDim] = NumOps.FromDouble(Math.Cos(timestep * freq));
        }

        return new Tensor<T>(new[] { 1, dim }, new Vector<T>(data));
    }

    private Tensor<T> CreateZeroTensor(int[] shape)
    {
        var totalSize = shape.Aggregate(1, (a, b) => a * b);
        return new Tensor<T>(shape, new Vector<T>(new T[totalSize]));
    }

    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        return _engine.TensorAdd(a, b);
    }

    // Tape-tracked ReLU. The earlier manual span loop materialized a fresh
    // Tensor<T> from raw spans, which detaches the autodiff graph — Conv1DLayer
    // updates depend on the tape, so the OutputProjection chain wouldn't train
    // end-to-end through this boundary. _engine.ReLU records the op on the tape.
    private Tensor<T> ApplyRelu(Tensor<T> x) => _engine.ReLU(x);

    private Tensor<T> ReshapeToAudio(Tensor<T> x, int[] targetShape)
    {
        var data = new T[targetShape.Aggregate(1, (a, b) => a * b)];
        var span = x.AsSpan();
        for (int i = 0; i < Math.Min(data.Length, span.Length); i++)
        {
            data[i] = span[i];
        }
        return new Tensor<T>(targetShape, new Vector<T>(data));
    }

    /// <summary>
    /// Gets all parameters.
    /// </summary>
    public Vector<T> GetParameters()
    {
        var allParams = new List<T>();

        AddParams(allParams, _inputProjection.GetParameters());
        AddParams(allParams, _diffusionEmbedding.GetParameters());

        foreach (var block in _residualBlocks)
        {
            AddParams(allParams, block.GetParameters());
        }

        AddParams(allParams, _outputProjection1.GetParameters());
        AddParams(allParams, _outputProjection2.GetParameters());

        return new Vector<T>(allParams.ToArray());
    }

    private void AddParams(List<T> allParams, Vector<T> p)
    {
        for (int i = 0; i < p.Length; i++)
        {
            allParams.Add(p[i]);
        }
    }

    /// <summary>
    /// Sets all parameters.
    /// </summary>
    public void SetParameters(Vector<T> parameters)
    {
        var offset = 0;

        offset = SetLayerParams(_inputProjection, parameters, offset);
        offset = SetLayerParams(_diffusionEmbedding, parameters, offset);

        foreach (var block in _residualBlocks)
        {
            int count = checked((int)block.ParameterCount);
            var p = new T[count];
            for (int i = 0; i < count; i++)
            {
                p[i] = parameters[offset++];
            }
            block.SetParameters(new Vector<T>(p));
        }

        offset = SetLayerParams(_outputProjection1, parameters, offset);
        offset = SetLayerParams(_outputProjection2, parameters, offset);
    }

    private int SetLayerParams(ILayer<T> layer, Vector<T> parameters, int offset)
    {
        int count = checked((int)layer.ParameterCount);
        var p = new T[count];
        for (int i = 0; i < count; i++)
        {
            p[i] = parameters[offset + i];
        }
        layer.SetParameters(new Vector<T>(p));
        return offset + count;
    }
}

/// <summary>
/// Residual block for DiffWave with dilated convolution.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class DiffWaveResidualBlock<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Engine for tape-tracked tensor ops. Threaded down from the outer
    /// <see cref="DiffWaveModel{T}"/>'s base-class <c>Engine</c> property so
    /// every op (broadcast-add, dilated conv, etc.) runs through the same
    /// engine instance the surrounding training/profiling setup uses.
    /// </summary>
    private readonly IEngine _engine;

    private readonly int _channels;
    private readonly int _dilation;
    // Kong 2020 §3 / Figure 1 residual block layers, channels-first
    // [B, C, T] layout throughout:
    //   _dilatedConv:   Conv1D kernel=3, dilation=2^i → 2C channels (gated)
    //   _diffusionProj: FC projection of per-sample timestep embedding
    //                   from residualChannels to residualChannels;
    //                   broadcast-added across T inside Forward.
    //   _conditionProj: Conv1D kernel=1 mel conditioning → 2C
    //   _outputConv:    Conv1D kernel=1 → residualChannels (residual path)
    //   _skipConv:      Conv1D kernel=1 → residualChannels (skip path)
    private readonly Conv1DLayer<T> _dilatedConv;
    private readonly DenseLayer<T> _diffusionProj;
    private readonly Conv1DLayer<T>? _conditionProj;
    private readonly Conv1DLayer<T> _outputConv;
    private readonly Conv1DLayer<T> _skipConv;

    /// <summary>
    /// Gets the number of parameters.
    /// </summary>
    /// <remarks>
    /// Computed live from the underlying <see cref="DenseLayer{T}"/>s
    /// because they use PyTorch-style lazy shape inference — their
    /// <c>ParameterCount</c> resolves to a real value only after the first
    /// <c>Forward</c> pass. Snapshotting at construction time would freeze
    /// the count at 0 and leak through to
    /// <see cref="DiffWaveNetwork{T}.ParameterCount"/> / <c>SetParameters</c>
    /// validation.
    /// </remarks>
    public long ParameterCount =>
        _dilatedConv.ParameterCount +
        _diffusionProj.ParameterCount +
        (_conditionProj?.ParameterCount ?? 0) +
        _outputConv.ParameterCount +
        _skipConv.ParameterCount;

    /// <summary>
    /// Initializes a new residual block.
    /// </summary>
    public DiffWaveResidualBlock(
        IEngine engine,
        int channels,
        int dilation,
        int conditionChannels = 0,
        int? seed = null)
    {
        _engine = engine ?? throw new ArgumentNullException(nameof(engine));
        _channels = channels;
        _dilation = dilation;

        // Dilated 1D convolution — Kong 2020 §3 / Figure 1: kernel size
        // 3, dilation 2^(i % dilation_cycle), output channels = 2C for
        // the subsequent gated activation (split into "filter" / "gate"
        // halves). Eager-init with inputChannels=residualChannels (the
        // residual stream width).
        _dilatedConv = new Conv1DLayer<T>(
            inputChannels: channels,
            outputChannels: channels * 2,
            kernelSize: 3,
            dilation: dilation);

        // Diffusion embedding projection: FC layer mapping the
        // residualChannels-dim per-sample embedding to residualChannels
        // (then broadcast-added across the time axis inside Forward).
        _diffusionProj = new DenseLayer<T>(channels, (IActivationFunction<T>?)null);

        // Mel conditioning — Kong 2020 §3.3: optional global conditioner
        // (mel-spectrogram, upsampled to audio rate) projected via 1×1
        // Conv1D to 2C before being added pointwise to the dilated-conv
        // output. Eager-init with conditionChannels (e.g. 80 mel bins).
        if (conditionChannels > 0)
        {
            _conditionProj = new Conv1DLayer<T>(
                inputChannels: conditionChannels,
                outputChannels: channels * 2,
                kernelSize: 1);
        }

        // Output / skip projections — both 1×1 Conv1D mapping the
        // post-gated C-channel activation back to residualChannels.
        _outputConv = new Conv1DLayer<T>(
            inputChannels: channels,
            outputChannels: channels, kernelSize: 1);
        _skipConv = new Conv1DLayer<T>(
            inputChannels: channels,
            outputChannels: channels, kernelSize: 1);
    }

    /// <summary>
    /// Forward pass returning output and skip connection — Kong et al.
    /// 2020 "DiffWave" §3 / Figure 1 residual block. Inputs and outputs
    /// are channels-FIRST <c>[B, residualChannels, T]</c>.
    /// </summary>
    public (Tensor<T> Output, Tensor<T> Skip) Forward(
        Tensor<T> x,
        Tensor<T> diffusionEmbed,
        Tensor<T>? melCondition)
    {
        // Step 1 — diffusion-step embedding broadcast over time
        // (Figure 1 "+" before the dilated conv): FC project the
        // [B, residualChannels] embedding to a per-channel offset and
        // FiLM-add it onto the residual stream at every timestep.
        var diffProj = _diffusionProj.Forward(diffusionEmbed);
        var h = BroadcastAddEmbedding(x, diffProj);

        // Step 2 — dilated 1D convolution to 2C channels (filter+gate).
        h = _dilatedConv.Forward(h);

        // Step 3 — optional mel-spectrogram conditioning (Kong 2020
        // §3.3). Mel comes in at frame rate ([B, melChannels, frames])
        // and must be brought up to audio rate ([B, melChannels, T])
        // before the 1×1 Conv1D projection so the resulting condProj can
        // be added pointwise to h (which is at audio rate T). The
        // GenerateFromMelSpectrogram entry point picks T = frames *
        // hop_size with hop=256, so the ratio T/frames is the per-frame
        // sample count. Tape-safe nearest-neighbour upsample via
        // TensorRepeatElements records the op on the autodiff tape (per
        // GraphConvolutionalLayer's identical pattern at line 1285), so
        // gradients flow back through the mel encoder during training.
        if (melCondition != null && _conditionProj != null)
        {
            int audioT = h.Shape[h.Rank - 1];
            int melT = melCondition.Shape[melCondition.Rank - 1];
            Tensor<T> melAtAudioRate = melCondition;
            if (melT != audioT && melT > 0)
            {
                int ratio = audioT / melT;
                if (ratio < 1) ratio = 1;
                if (ratio > 1)
                    melAtAudioRate = _engine.TensorRepeatElements(melCondition, ratio, axis: melCondition.Rank - 1);
                // Truncate if non-integer ratio left a remainder (frame
                // count doesn't cleanly divide the audio length): clip
                // the trailing samples so the conv input width matches.
                int upT = melAtAudioRate.Shape[melAtAudioRate.Rank - 1];
                if (upT > audioT)
                {
                    int batchDim = melAtAudioRate.Shape[0];
                    int chanDim = melAtAudioRate.Shape[1];
                    melAtAudioRate = _engine.TensorSlice(melAtAudioRate,
                        new[] { 0, 0, 0 },
                        new[] { batchDim, chanDim, audioT });
                }
            }
            var condProj = _conditionProj.Forward(melAtAudioRate);
            h = AddTensors(h, condProj);
        }

        // Step 4 — gated activation (tanh ⊙ sigmoid on the channel split).
        h = ApplyGatedActivation(h);

        // Step 5 — residual / skip projections via 1×1 Conv1D.
        var output = _outputConv.Forward(h);
        var skip = _skipConv.Forward(h);

        // Step 6 — residual connection with the input (Figure 1).
        output = AddTensors(output, x);

        return (output, skip);
    }

    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        return _engine.TensorAdd(a, b);
    }

    /// <summary>
    /// Broadcasts a per-step embedding <paramref name="embedding"/> across
    /// the time axis of <paramref name="x"/> and adds it, FiLM-style.
    /// Per Kong et al. 2020 "DiffWave" §3.2 / Figure 2 the diffusion step
    /// is mapped to a <c>[B, C]</c> embedding via a linear layer then
    /// broadcast across every timestep before being added to the dilated
    /// conv input.
    /// </summary>
    /// <remarks>
    /// Channel convention: paper-faithful channels-FIRST layout
    /// <c>[B, C, T]</c>. Reshape the <c>[B, C]</c> embedding to
    /// <c>[B, C, 1]</c> so <c>TensorBroadcastAdd</c> replicates it
    /// across the T axis to produce <c>[B, C, T]</c>. Falls through to a
    /// plain <c>TensorAdd</c> when the shape contract isn't met so the
    /// engine surfaces a clear shape-mismatch error.
    /// </remarks>
    private Tensor<T> BroadcastAddEmbedding(Tensor<T> x, Tensor<T> embedding)
    {
        if (x.Rank == 3 && embedding.Rank == 2
            && x.Shape[0] == embedding.Shape[0]
            && x.Shape[1] == embedding.Shape[1])
        {
            var unsqueezed = _engine.Reshape(embedding,
                new[] { embedding.Shape[0], embedding.Shape[1], 1 });
            return _engine.TensorBroadcastAdd(x, unsqueezed);
        }
        return _engine.TensorAdd(x, embedding);
    }

    private Tensor<T> ApplyGatedActivation(Tensor<T> x)
    {
        // Gated activation per Kong et al. 2020 "DiffWave" §3.2 / Eq. 2:
        //   z = tanh(W_f * x) ⊙ sigmoid(W_g * x)
        // The 2C-channel pre-activation is split along the CHANNEL axis
        // (dim=1 for channels-first [B, C, T]) into two C-channel halves
        // — first half feeds tanh (filter), second half feeds sigmoid
        // (gate), then element-wise multiplied. Now that the model uses
        // paper-faithful Conv1D with channels-first layout, the channel
        // axis is consistently at Shape[1].
        //
        // All ops route through _engine (Tanh / Sigmoid / TensorMultiply
        // / TensorSlice) so the activation participates in the autodiff
        // tape — Conv1DLayer.UpdateParameters relies on the tape, so the
        // earlier raw-span loop disconnected the graph and the residual
        // block's filter/gate convs could not receive gradients.
        var inputShape = x._shape;
        int rank = inputShape.Length;

        if (rank < 3)
        {
            throw new InvalidOperationException(
                $"Gated activation expects rank-3 [B, C, T] input; got rank {rank}.");
        }

        int batch = inputShape[0];
        int channels = inputShape[1];
        int time = inputShape[2];
        int halfChannels = channels / 2;
        if (halfChannels <= 0 || channels % 2 != 0)
        {
            throw new InvalidOperationException(
                $"Gated activation requires an even, positive channel dim; got {channels}.");
        }

        // Split [B, 2C, T] → two [B, C, T] halves along axis 1.
        var filterHalf = _engine.TensorSlice(x,
            new[] { 0, 0, 0 },
            new[] { batch, halfChannels, time });
        var gateHalf = _engine.TensorSlice(x,
            new[] { 0, halfChannels, 0 },
            new[] { batch, halfChannels, time });
        return _engine.TensorMultiply(_engine.Tanh(filterHalf), _engine.Sigmoid(gateHalf));
    }

    /// <summary>
    /// Gets all parameters.
    /// </summary>
    public Vector<T> GetParameters()
    {
        var allParams = new List<T>();

        AddParams(allParams, _dilatedConv.GetParameters());
        AddParams(allParams, _diffusionProj.GetParameters());
        if (_conditionProj != null)
        {
            AddParams(allParams, _conditionProj.GetParameters());
        }
        AddParams(allParams, _outputConv.GetParameters());
        AddParams(allParams, _skipConv.GetParameters());

        return new Vector<T>(allParams.ToArray());
    }

    private void AddParams(List<T> allParams, Vector<T> p)
    {
        for (int i = 0; i < p.Length; i++)
        {
            allParams.Add(p[i]);
        }
    }

    /// <summary>
    /// Sets all parameters.
    /// </summary>
    public void SetParameters(Vector<T> parameters)
    {
        var offset = 0;

        offset = SetLayerParams(_dilatedConv, parameters, offset);
        offset = SetLayerParams(_diffusionProj, parameters, offset);
        if (_conditionProj != null)
        {
            offset = SetLayerParams(_conditionProj, parameters, offset);
        }
        offset = SetLayerParams(_outputConv, parameters, offset);
        offset = SetLayerParams(_skipConv, parameters, offset);
    }

    private int SetLayerParams(ILayer<T> layer, Vector<T> parameters, int offset)
    {
        int count = checked((int)layer.ParameterCount);
        var p = new T[count];
        for (int i = 0; i < count; i++)
        {
            p[i] = parameters[offset + i];
        }
        layer.SetParameters(new Vector<T>(p));
        return offset + count;
    }
}

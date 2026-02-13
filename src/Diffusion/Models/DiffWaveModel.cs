using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Extensions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Diffusion;
using AiDotNet.NeuralNetworks.Diffusion.Schedulers;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.Models;

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
public class DiffWaveModel<T> : DiffusionModelBase<T>
{
    /// <summary>
    /// Default sample rate in Hz.
    /// </summary>
    private const int DEFAULT_SAMPLE_RATE = 22050;

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
    private readonly DiffWaveNetwork<T> _network;

    /// <inheritdoc />
    public override int ParameterCount => _network.ParameterCount;

    /// <summary>
    /// Gets the sample rate in Hz.
    /// </summary>
    public int SampleRate { get; }

    /// <summary>
    /// Initializes a new instance of DiffWaveModel with full customization support.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    /// <param name="scheduler">Optional custom scheduler.</param>
    /// <param name="residualChannels">Number of residual channels.</param>
    /// <param name="residualLayers">Number of residual layers.</param>
    /// <param name="dilationCycle">Dilation cycle length.</param>
    /// <param name="melChannels">Number of mel-spectrogram channels.</param>
    /// <param name="sampleRate">Audio sample rate in Hz.</param>
    /// <param name="seed">Optional random seed.</param>
    public DiffWaveModel(
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        int residualChannels = 64,
        int residualLayers = 30,
        int dilationCycle = 10,
        int melChannels = 80,
        int sampleRate = DEFAULT_SAMPLE_RATE,
        int? seed = null)
        : base(options ?? CreateDefaultOptions(), scheduler ?? CreateDefaultScheduler(seed))
    {
        _residualChannels = residualChannels;
        _residualLayers = residualLayers;
        _dilationCycle = dilationCycle;
        _melChannels = melChannels;
        SampleRate = sampleRate;

        _network = new DiffWaveNetwork<T>(
            residualChannels: residualChannels,
            residualLayers: residualLayers,
            dilationCycle: dilationCycle,
            melChannels: melChannels,
            seed: seed);
    }

    /// <summary>
    /// Creates the default options.
    /// </summary>
    private static DiffusionModelOptions<T> CreateDefaultOptions()
    {
        return new DiffusionModelOptions<T>
        {
            TrainTimesteps = 200,
            BetaStart = 0.0001,
            BetaEnd = 0.05,
            BetaSchedule = BetaSchedule.Linear
        };
    }

    /// <summary>
    /// Creates the default scheduler.
    /// </summary>
    private static INoiseScheduler<T> CreateDefaultScheduler(int? seed)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var config = new SchedulerConfig<T>(
            trainTimesteps: 200,
            betaStart: ops.FromDouble(0.0001),
            betaEnd: ops.FromDouble(0.05),
            betaSchedule: BetaSchedule.Linear,
            predictionType: DiffusionPredictionType.Epsilon);
        return new DDIMScheduler<T>(config);
    }

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
        return _network.Forward(noisySample, timestep, null);
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        return _network.GetParameters();
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        _network.SetParameters(parameters);
    }

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

        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "DiffWaveModel",
            Version = "1.0.0",
            ModelType = ModelType.NeuralNetwork,
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

    private readonly DenseLayer<T> _inputProjection;
    private readonly DenseLayer<T> _diffusionEmbedding;
    private readonly List<DiffWaveResidualBlock<T>> _residualBlocks;
    private readonly DenseLayer<T> _outputProjection1;
    private readonly DenseLayer<T> _outputProjection2;

    /// <summary>
    /// Gets the number of parameters.
    /// </summary>
    public int ParameterCount { get; private set; }

    /// <summary>
    /// Initializes a new DiffWaveNetwork.
    /// </summary>
    public DiffWaveNetwork(
        int residualChannels = 64,
        int residualLayers = 30,
        int dilationCycle = 10,
        int melChannels = 80,
        int? seed = null)
    {
        _residualChannels = residualChannels;
        _residualLayers = residualLayers;
        _dilationCycle = dilationCycle;
        _melChannels = melChannels;

        // Input projection (1 channel audio to residual channels)
        _inputProjection = new DenseLayer<T>(1, residualChannels, (IActivationFunction<T>?)null);

        // Diffusion timestep embedding
        _diffusionEmbedding = new DenseLayer<T>(128, residualChannels, (IActivationFunction<T>?)null);

        // Residual blocks with varying dilation
        _residualBlocks = new List<DiffWaveResidualBlock<T>>();
        for (int i = 0; i < residualLayers; i++)
        {
            var dilation = (int)Math.Pow(2, i % dilationCycle);
            _residualBlocks.Add(new DiffWaveResidualBlock<T>(
                channels: residualChannels,
                dilation: dilation,
                conditionChannels: melChannels,
                seed: seed));
        }

        // Output projections
        _outputProjection1 = new DenseLayer<T>(residualChannels, residualChannels, (IActivationFunction<T>?)null);
        _outputProjection2 = new DenseLayer<T>(residualChannels, 1, (IActivationFunction<T>?)null);

        CalculateParameterCount();
    }

    private void CalculateParameterCount()
    {
        ParameterCount = _inputProjection.ParameterCount +
                         _diffusionEmbedding.ParameterCount +
                         _outputProjection1.ParameterCount +
                         _outputProjection2.ParameterCount;

        foreach (var block in _residualBlocks)
        {
            ParameterCount += block.ParameterCount;
        }
    }

    /// <summary>
    /// Forward pass through the network.
    /// </summary>
    public Tensor<T> Forward(Tensor<T> audio, int timestep, Tensor<T>? melCondition)
    {
        // Get diffusion embedding
        var diffEmbed = GetTimestepEmbedding(timestep);
        diffEmbed = _diffusionEmbedding.Forward(diffEmbed);

        // Project input
        var x = ProjectAudio(audio);

        // Apply residual blocks
        var skip = CreateZeroTensor(x.Shape);
        foreach (var block in _residualBlocks)
        {
            var (blockOut, skipOut) = block.Forward(x, diffEmbed, melCondition);
            x = blockOut;
            skip = AddTensors(skip, skipOut);
        }

        // Output projection
        var output = ApplyRelu(_outputProjection1.Forward(skip));
        output = _outputProjection2.Forward(output);

        return ReshapeToAudio(output, audio.Shape);
    }

    private Tensor<T> ProjectAudio(Tensor<T> audio)
    {
        // Simplified: flatten and project each sample
        return _inputProjection.Forward(audio);
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
        var result = new Tensor<T>(a.Shape);
        var aSpan = a.AsSpan();
        var bSpan = b.AsSpan();
        var resultSpan = result.AsWritableSpan();

        var minLen = Math.Min(aSpan.Length, bSpan.Length);
        for (int i = 0; i < minLen; i++)
        {
            resultSpan[i] = NumOps.Add(aSpan[i], bSpan[i]);
        }

        return result;
    }

    private Tensor<T> ApplyRelu(Tensor<T> x)
    {
        var result = new Tensor<T>(x.Shape);
        var span = x.AsSpan();
        var resultSpan = result.AsWritableSpan();

        for (int i = 0; i < span.Length; i++)
        {
            var val = NumOps.ToDouble(span[i]);
            resultSpan[i] = NumOps.FromDouble(Math.Max(0, val));
        }

        return result;
    }

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
            var count = block.ParameterCount;
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

    private int SetLayerParams(DenseLayer<T> layer, Vector<T> parameters, int offset)
    {
        var count = layer.ParameterCount;
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

    private readonly int _channels;
    private readonly int _dilation;
    private readonly DenseLayer<T> _dilatedConv;
    private readonly DenseLayer<T> _diffusionProj;
    private readonly DenseLayer<T>? _conditionProj;
    private readonly DenseLayer<T> _outputConv;
    private readonly DenseLayer<T> _skipConv;

    /// <summary>
    /// Gets the number of parameters.
    /// </summary>
    public int ParameterCount { get; }

    /// <summary>
    /// Initializes a new residual block.
    /// </summary>
    public DiffWaveResidualBlock(
        int channels,
        int dilation,
        int conditionChannels = 0,
        int? seed = null)
    {
        _channels = channels;
        _dilation = dilation;

        // Dilated convolution (simplified as dense layer)
        _dilatedConv = new DenseLayer<T>(channels, channels * 2, (IActivationFunction<T>?)null);

        // Diffusion embedding projection
        _diffusionProj = new DenseLayer<T>(channels, channels, (IActivationFunction<T>?)null);

        // Optional mel conditioning projection
        if (conditionChannels > 0)
        {
            _conditionProj = new DenseLayer<T>(conditionChannels, channels * 2, (IActivationFunction<T>?)null);
        }

        // Output projections
        _outputConv = new DenseLayer<T>(channels, channels, (IActivationFunction<T>?)null);
        _skipConv = new DenseLayer<T>(channels, channels, (IActivationFunction<T>?)null);

        ParameterCount = _dilatedConv.ParameterCount +
                         _diffusionProj.ParameterCount +
                         (_conditionProj?.ParameterCount ?? 0) +
                         _outputConv.ParameterCount +
                         _skipConv.ParameterCount;
    }

    /// <summary>
    /// Forward pass returning output and skip connection.
    /// </summary>
    public (Tensor<T> Output, Tensor<T> Skip) Forward(
        Tensor<T> x,
        Tensor<T> diffusionEmbed,
        Tensor<T>? melCondition)
    {
        // Add diffusion embedding
        var diffProj = _diffusionProj.Forward(diffusionEmbed);
        var h = AddTensors(x, diffProj);

        // Dilated convolution
        h = _dilatedConv.Forward(h);

        // Add mel conditioning if provided
        if (melCondition != null && _conditionProj != null)
        {
            var condProj = _conditionProj.Forward(melCondition);
            h = AddTensors(h, condProj);
        }

        // Gated activation (simplified sigmoid * tanh)
        h = ApplyGatedActivation(h);

        // Output and skip projections
        var output = _outputConv.Forward(h);
        var skip = _skipConv.Forward(h);

        // Residual connection
        output = AddTensors(output, x);

        return (output, skip);
    }

    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        var result = new Tensor<T>(a.Shape);
        var aSpan = a.AsSpan();
        var bSpan = b.AsSpan();
        var resultSpan = result.AsWritableSpan();

        var minLen = Math.Min(aSpan.Length, bSpan.Length);
        for (int i = 0; i < minLen; i++)
        {
            resultSpan[i] = NumOps.Add(aSpan[i], bSpan[i]);
        }

        return result;
    }

    private Tensor<T> ApplyGatedActivation(Tensor<T> x)
    {
        // Gated activation: tanh(x[:halfChannels]) * sigmoid(x[halfChannels:])
        // Input shape: [batch, channels, length] for 1D waveform, or [batch, channels] for embedding
        var inputShape = x.Shape;
        var span = x.AsSpan();

        // Compute output shape: same as input but with half the channel dimension
        int[] outputShape;
        int batchSize = 1;
        int halfChannels;
        int elementsPerSample;

        if (inputShape.Length == 3)
        {
            // Shape: [batch, channels, length]
            batchSize = inputShape[0];
            halfChannels = inputShape[1] / 2;
            int length = inputShape[2];
            outputShape = new[] { batchSize, halfChannels, length };
            elementsPerSample = inputShape[1] * length;
        }
        else if (inputShape.Length == 2)
        {
            // Shape: [batch, features]
            batchSize = inputShape[0];
            halfChannels = inputShape[1] / 2;
            outputShape = new[] { batchSize, halfChannels };
            elementsPerSample = inputShape[1];
        }
        else
        {
            // Fallback: treat as flat array
            halfChannels = span.Length / 2;
            outputShape = new[] { halfChannels };
            elementsPerSample = span.Length;
        }

        var result = new Tensor<T>(outputShape);
        var resultSpan = result.AsWritableSpan();
        int halfElementsPerSample = elementsPerSample / 2;

        for (int b = 0; b < batchSize; b++)
        {
            int inputOffset = b * elementsPerSample;
            int outputOffset = b * halfElementsPerSample;

            for (int i = 0; i < halfElementsPerSample; i++)
            {
                var t = NumOps.ToDouble(span[inputOffset + i]);
                var s = NumOps.ToDouble(span[inputOffset + halfElementsPerSample + i]);

                // tanh(t) * sigmoid(s)
                var tanhVal = Math.Tanh(t);
                var sigmoidVal = 1.0 / (1.0 + Math.Exp(-s));
                resultSpan[outputOffset + i] = NumOps.FromDouble(tanhVal * sigmoidVal);
            }
        }

        return result;
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

    private int SetLayerParams(DenseLayer<T> layer, Vector<T> parameters, int offset)
    {
        var count = layer.ParameterCount;
        var p = new T[count];
        for (int i = 0; i < count; i++)
        {
            p[i] = parameters[offset + i];
        }
        layer.SetParameters(new Vector<T>(p));
        return offset + count;
    }
}

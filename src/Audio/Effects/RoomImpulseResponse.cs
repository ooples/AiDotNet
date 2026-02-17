using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Effects;

/// <summary>
/// Neural Room Impulse Response estimation model for acoustic analysis and dereverberation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Neural Room Impulse Response (RIR) estimation (2023-2024) uses deep learning to predict
/// acoustic characteristics of a room from audio recordings. The model estimates the RIR
/// encoding how sound propagates, reflects, and decays, enabling dereverberation, room
/// simulation, and acoustic environment matching.
/// </para>
/// <para>
/// <b>For Beginners:</b> When you clap in a big room, you hear echoes. This model learns to
/// understand those echoes. Given a recording, it figures out the room's acoustic "fingerprint"
/// and can use it to remove room effects (dereverberation) or apply one room's sound to
/// another recording.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 257, outputSize: 16000);
/// var model = new RoomImpulseResponse&lt;float&gt;(arch, "rir_estimator.onnx");
/// var clean = model.Enhance(reverbAudio);
/// </code>
/// </para>
/// </remarks>
public class RoomImpulseResponse<T> : AudioNeuralNetworkBase<T>, IAudioEnhancer<T>
{
    #region Fields

    private readonly RoomImpulseResponseOptions _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private bool _useNativeMode;
    private bool _disposed;
    private List<T>? _streamingBuffer;

    #endregion

    #region Constructors

    /// <summary>Creates an RIR estimation model in ONNX inference mode.</summary>
    public RoomImpulseResponse(NeuralNetworkArchitecture<T> architecture, string modelPath, RoomImpulseResponseOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new RoomImpulseResponseOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        InitializeLayers();
    }

    /// <summary>Creates an RIR estimation model in native training mode.</summary>
    public RoomImpulseResponse(NeuralNetworkArchitecture<T> architecture, RoomImpulseResponseOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new RoomImpulseResponseOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        InitializeLayers();
    }

    internal static async Task<RoomImpulseResponse<T>> CreateAsync(RoomImpulseResponseOptions? options = null,
        IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new RoomImpulseResponseOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("rir_estimator", "rir_estimator.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.NumFrequencyBins, outputSize: options.RIRLength);
        return new RoomImpulseResponse<T>(arch, mp, options);
    }

    #endregion

    #region IAudioEnhancer Properties

    /// <inheritdoc />
    public int NumChannels { get; } = 1;

    /// <inheritdoc />
    public double EnhancementStrength { get; set; } = 1.0;

    /// <inheritdoc />
    public int LatencySamples => _options.RIRLength;

    #endregion

    #region IAudioEnhancer Methods

    /// <inheritdoc />
    public Tensor<T> Enhance(Tensor<T> audio)
    {
        ThrowIfDisposed();
        EnhancementStrength = _options.DereverberationStrength;
        // Estimate RIR from audio
        var features = PreprocessAudio(audio);
        Tensor<T> estimatedRIR;
        if (IsOnnxMode && OnnxEncoder is not null) estimatedRIR = OnnxEncoder.Run(features);
        else estimatedRIR = Predict(features);
        // Apply dereverberation using estimated RIR
        return ApplyDereverberation(audio, estimatedRIR);
    }

    /// <inheritdoc />
    public Tensor<T> EnhanceWithReference(Tensor<T> audio, Tensor<T> reference)
        => Enhance(audio);

    /// <inheritdoc />
    public Tensor<T> ProcessChunk(Tensor<T> audioChunk)
    {
        ThrowIfDisposed();
        _streamingBuffer ??= [];
        for (int i = 0; i < audioChunk.Length; i++) _streamingBuffer.Add(audioChunk[i]);

        int frameSize = _options.RIRLength;
        if (_streamingBuffer.Count < frameSize)
            return new Tensor<T>([0]);

        var frame = new Tensor<T>([frameSize]);
        for (int i = 0; i < frameSize; i++) frame[i] = _streamingBuffer[i];
        _streamingBuffer.RemoveRange(0, frameSize / 2);
        return Enhance(frame);
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        base.ResetState();
        _streamingBuffer = null;
    }

    /// <inheritdoc />
    public void EstimateNoiseProfile(Tensor<T> noiseOnlyAudio) { }

    #endregion

    #region NeuralNetworkBase Implementation

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultRoomImpulseResponseLayers(
            encoderDim: _options.EncoderDim, numEncoderLayers: _options.NumEncoderLayers,
            numHeads: _options.NumHeads, numFrequencyBins: _options.NumFrequencyBins,
            rirLength: _options.RIRLength, dropoutRate: _options.DropoutRate));
    }

    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        var c = input; foreach (var l in Layers) c = l.Forward(c); return c;
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode.");
        SetTrainingMode(true);
        var output = Predict(input);
        var grad = LossFunction.CalculateDerivative(output.ToVector(), expected.ToVector());
        var gt = Tensor<T>.FromVector(grad);
        for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt);
        _optimizer.UpdateParameters(Layers);
        SetTrainingMode(false);
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode) throw new NotSupportedException("ONNX mode.");
        int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; }
    }

    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        if (MelSpec is not null) return MelSpec.Forward(rawAudio);
        return rawAudio;
    }

    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "RoomImpulseResponse-Native" : "RoomImpulseResponse-ONNX",
            Description = "Neural Room Impulse Response estimation for dereverberation (2023-2024)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.NumFrequencyBins,
            Complexity = _options.NumEncoderLayers
        };
        m.AdditionalInfo["RIRLength"] = _options.RIRLength.ToString();
        m.AdditionalInfo["EncoderDim"] = _options.EncoderDim.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.EncoderDim);
        w.Write(_options.NumEncoderLayers); w.Write(_options.RIRLength);
        w.Write(_options.NumFrequencyBins); w.Write(_options.NumHeads);
        w.Write(_options.DereverberationStrength); w.Write(_options.RT60WindowSeconds);
        w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.EncoderDim = r.ReadInt32();
        _options.NumEncoderLayers = r.ReadInt32(); _options.RIRLength = r.ReadInt32();
        _options.NumFrequencyBins = r.ReadInt32(); _options.NumHeads = r.ReadInt32();
        _options.DereverberationStrength = r.ReadDouble(); _options.RT60WindowSeconds = r.ReadDouble();
        _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        => new RoomImpulseResponse<T>(Architecture, _options);

    #endregion

    #region Private Helpers

    private Tensor<T> ApplyDereverberation(Tensor<T> audio, Tensor<T> estimatedRIR)
    {
        // Simplified spectral dereverberation using estimated RIR
        var output = new Tensor<T>([audio.Length]);
        double strength = EnhancementStrength;
        for (int i = 0; i < audio.Length; i++)
        {
            double clean = NumOps.ToDouble(audio[i]);
            // Subtract estimated reverb contribution
            double reverbContrib = 0;
            for (int j = 1; j < Math.Min(estimatedRIR.Length, i); j++)
            {
                double rirVal = NumOps.ToDouble(estimatedRIR[j]);
                double audioVal = NumOps.ToDouble(audio[i - j]);
                reverbContrib += rirVal * audioVal;
            }
            double dereverbed = clean - strength * reverbContrib;
            output[i] = NumOps.FromDouble(Math.Max(-1.0, Math.Min(1.0, dereverbed)));
        }
        return output;
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(RoomImpulseResponse<T>)); }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}

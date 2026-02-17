using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.Audio.Generation;

/// <summary>
/// SoundStream neural audio codec from Google for efficient audio compression.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SoundStream (Zeghidour et al., 2021, Google) is a fully convolutional encoder-decoder
/// with residual vector quantization for audio compression at 3-18 kbps. It pioneered the
/// RVQ approach later adopted by EnCodec. SoundStream powers Google's AudioLM and MusicLM.
/// </para>
/// <para>
/// <b>For Beginners:</b> SoundStream compresses audio using AI, like a smart zip file for
/// sound. It can compress a song to just 3-6 kbps (versus 128 kbps for MP3) while keeping
/// good quality. It uses "residual vector quantization" which is like describing a painting
/// with increasingly fine details: the first pass captures the rough shape, each additional
/// pass adds more nuance.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 1, outputSize: 128);
/// var model = new SoundStream&lt;float&gt;(arch, "soundstream.onnx");
/// int[,] tokens = model.Encode(audioWaveform);
/// var reconstructed = model.Decode(tokens);
/// double bitrate = model.GetBitrate(numQuantizers: 4);
/// </code>
/// </para>
/// </remarks>
public class SoundStream<T> : AudioNeuralNetworkBase<T>, IAudioCodec<T>
{
    #region Fields

    private readonly SoundStreamOptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region IAudioCodec Properties

    /// <inheritdoc />
    public int NumQuantizers => _options.NumQuantizers;

    /// <inheritdoc />
    public int CodebookSize => _options.CodebookSize;

    /// <inheritdoc />
    public int TokenFrameRate
    {
        get
        {
            int totalDownsample = 1;
            foreach (int r in _options.DownsampleRatios) totalDownsample *= r;
            return _options.SampleRate / totalDownsample;
        }
    }

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a SoundStream model in ONNX inference mode.
    /// </summary>
    public SoundStream(NeuralNetworkArchitecture<T> architecture, string modelPath, SoundStreamOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new SoundStreamOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a SoundStream model in native training mode.
    /// </summary>
    public SoundStream(NeuralNetworkArchitecture<T> architecture, SoundStreamOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new SoundStreamOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        InitializeLayers();
    }

    internal static async Task<SoundStream<T>> CreateAsync(SoundStreamOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new SoundStreamOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("soundstream", "soundstream.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: 1, outputSize: options.EncoderDim);
        return new SoundStream<T>(arch, mp, options);
    }

    #endregion

    #region IAudioCodec

    /// <inheritdoc />
    public int[,] Encode(Tensor<T> audio)
    {
        ThrowIfDisposed();
        Tensor<T> embeddings = EncodeEmbeddings(audio);

        int numFrames = Math.Max(1, embeddings.Length / _options.EncoderDim);
        int nq = _options.NumQuantizers;
        var tokens = new int[nq, numFrames];

        for (int f = 0; f < numFrames; f++)
        {
            for (int q = 0; q < nq; q++)
            {
                int idx = f * _options.EncoderDim + q;
                if (idx < embeddings.Length)
                {
                    double val = NumOps.ToDouble(embeddings[idx]);
                    tokens[q, f] = Math.Max(0, Math.Min(_options.CodebookSize - 1,
                        (int)(((val + 1.0) / 2.0) * _options.CodebookSize)));
                }
            }
        }
        return tokens;
    }

    /// <inheritdoc />
    public Task<int[,]> EncodeAsync(Tensor<T> audio, CancellationToken cancellationToken = default)
    {
        return Task.Run(() => Encode(audio), cancellationToken);
    }

    /// <inheritdoc />
    public Tensor<T> Decode(int[,] tokens)
    {
        ThrowIfDisposed();
        int nq = tokens.GetLength(0);
        int numFrames = tokens.GetLength(1);

        var embeddings = new Tensor<T>([numFrames * _options.EncoderDim]);
        for (int f = 0; f < numFrames; f++)
        {
            for (int q = 0; q < nq && q < _options.EncoderDim; q++)
            {
                double val = (tokens[q, f] / (double)_options.CodebookSize) * 2.0 - 1.0;
                embeddings[f * _options.EncoderDim + q] = NumOps.FromDouble(val);
            }
        }

        return DecodeEmbeddings(embeddings);
    }

    /// <inheritdoc />
    public Task<Tensor<T>> DecodeAsync(int[,] tokens, CancellationToken cancellationToken = default)
    {
        return Task.Run(() => Decode(tokens), cancellationToken);
    }

    /// <inheritdoc />
    public Tensor<T> EncodeEmbeddings(Tensor<T> audio)
    {
        ThrowIfDisposed();
        return IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(audio) : Predict(audio);
    }

    /// <inheritdoc />
    public Tensor<T> DecodeEmbeddings(Tensor<T> embeddings)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxDecoder is not null) return OnnxDecoder.Run(embeddings);
        var c = embeddings;
        int decoderStart = Layers.Count / 2;
        for (int i = decoderStart; i < Layers.Count; i++) c = Layers[i].Forward(c);
        return c;
    }

    /// <inheritdoc />
    public double GetBitrate(int? numQuantizers = null)
    {
        int nq = numQuantizers ?? _options.NumQuantizers;
        double bitsPerToken = Math.Log(_options.CodebookSize) / Math.Log(2.0);
        return nq * TokenFrameRate * bitsPerToken;
    }

    #endregion

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultSoundStreamLayers(
            encoderChannels: _options.EncoderChannels, encoderDim: _options.EncoderDim,
            numResBlocks: _options.NumResBlocks, dropoutRate: _options.DropoutRate));
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
        _optimizer?.UpdateParameters(Layers);
        SetTrainingMode(false);
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode) throw new NotSupportedException("ONNX mode.");
        int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; }
    }

    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) => rawAudio;
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "SoundStream-Native" : "SoundStream-ONNX",
            Description = "SoundStream neural audio codec (Zeghidour et al., 2021, Google)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = 1,
            Complexity = _options.EncoderChannels.Length
        };
        m.AdditionalInfo["NumQuantizers"] = _options.NumQuantizers.ToString();
        m.AdditionalInfo["CodebookSize"] = _options.CodebookSize.ToString();
        m.AdditionalInfo["TargetBitrateKbps"] = _options.TargetBitrateKbps.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.Channels);
        w.Write(_options.EncoderChannels.Length);
        foreach (int ch in _options.EncoderChannels) w.Write(ch);
        w.Write(_options.DownsampleRatios.Length);
        foreach (int r in _options.DownsampleRatios) w.Write(r);
        w.Write(_options.EncoderDim); w.Write(_options.NumResBlocks);
        w.Write(_options.NumQuantizers); w.Write(_options.CodebookSize);
        w.Write(_options.CodebookDim); w.Write(_options.TargetBitrateKbps);
        w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.Channels = r.ReadInt32();
        int nch = r.ReadInt32(); _options.EncoderChannels = new int[nch];
        for (int i = 0; i < nch; i++) _options.EncoderChannels[i] = r.ReadInt32();
        int ndr = r.ReadInt32(); _options.DownsampleRatios = new int[ndr];
        for (int i = 0; i < ndr; i++) _options.DownsampleRatios[i] = r.ReadInt32();
        _options.EncoderDim = r.ReadInt32(); _options.NumResBlocks = r.ReadInt32();
        _options.NumQuantizers = r.ReadInt32(); _options.CodebookSize = r.ReadInt32();
        _options.CodebookDim = r.ReadInt32(); _options.TargetBitrateKbps = r.ReadDouble();
        _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new SoundStream<T>(Architecture, _options);

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(SoundStream<T>)); }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) { OnnxEncoder?.Dispose(); OnnxDecoder?.Dispose(); }
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}

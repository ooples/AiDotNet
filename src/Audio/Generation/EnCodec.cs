using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.Audio.Generation;

/// <summary>
/// EnCodec neural audio codec from Meta for high-fidelity audio compression.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// EnCodec (Defossez et al., 2022) compresses audio to 1.5-24 kbps using an encoder-decoder
/// with residual vector quantization (RVQ). At 6 kbps it achieves near-transparent quality.
/// EnCodec tokens serve as the audio representation for language models like MusicGen and VALL-E.
/// </para>
/// <para>
/// <b>For Beginners:</b> EnCodec is like a super-efficient audio compressor. Regular MP3 needs
/// 128 kbps for good quality; EnCodec achieves similar quality at just 6 kbps. It works by:
/// 1. Encoding audio into a compact representation
/// 2. Quantizing that into discrete tokens (like words)
/// 3. Decoding those tokens back into audio
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 1, outputSize: 128);
/// var model = new EnCodec&lt;float&gt;(arch, "encodec.onnx");
/// int[,] tokens = model.Encode(audioWaveform);
/// var reconstructed = model.Decode(tokens);
/// </code>
/// </para>
/// </remarks>
public class EnCodec<T> : AudioNeuralNetworkBase<T>, IAudioCodec<T>
{
    #region Fields

    private readonly EnCodecOptions _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
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
    /// Creates an EnCodec model in ONNX inference mode.
    /// </summary>
    public EnCodec(NeuralNetworkArchitecture<T> architecture, string modelPath, EnCodecOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new EnCodecOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        InitializeLayers();
    }

    /// <summary>
    /// Creates an EnCodec model in native training mode.
    /// </summary>
    public EnCodec(NeuralNetworkArchitecture<T> architecture, EnCodecOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new EnCodecOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        InitializeLayers();
    }

    internal static async Task<EnCodec<T>> CreateAsync(EnCodecOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new EnCodecOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("encodec", "encodec_24khz.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: 1, outputSize: options.EncoderDim);
        return new EnCodec<T>(arch, mp, options);
    }

    #endregion

    #region IAudioCodec

    /// <inheritdoc />
    public int[,] Encode(Tensor<T> audio)
    {
        ThrowIfDisposed();
        Tensor<T> embeddings = EncodeEmbeddings(audio);

        // Simulate RVQ: quantize embeddings to discrete tokens
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

        // Convert tokens to embeddings
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
        // In native mode, use the decoder layers (second half of the network)
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
        else Layers.AddRange(LayerHelper<T>.CreateDefaultEnCodecLayers(
            encoderChannels: _options.EncoderChannels, encoderDim: _options.EncoderDim,
            dropoutRate: _options.DropoutRate));
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

    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) => rawAudio;
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "EnCodec-Native" : "EnCodec-ONNX",
            Description = $"EnCodec neural audio codec at {_options.SampleRate / 1000} kHz (Defossez et al., 2022, Meta)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = 1,
            Complexity = _options.EncoderChannels.Length
        };
        m.AdditionalInfo["NumQuantizers"] = _options.NumQuantizers.ToString();
        m.AdditionalInfo["CodebookSize"] = _options.CodebookSize.ToString();
        m.AdditionalInfo["TargetBandwidthKbps"] = _options.TargetBandwidthKbps.ToString();
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
        w.Write(_options.EncoderDim); w.Write(_options.NumQuantizers);
        w.Write(_options.CodebookSize); w.Write(_options.CodebookDim);
        w.Write(_options.TargetBandwidthKbps); w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.Channels = r.ReadInt32();
        int nch = r.ReadInt32(); _options.EncoderChannels = new int[nch];
        for (int i = 0; i < nch; i++) _options.EncoderChannels[i] = r.ReadInt32();
        int ndr = r.ReadInt32(); _options.DownsampleRatios = new int[ndr];
        for (int i = 0; i < ndr; i++) _options.DownsampleRatios[i] = r.ReadInt32();
        _options.EncoderDim = r.ReadInt32(); _options.NumQuantizers = r.ReadInt32();
        _options.CodebookSize = r.ReadInt32(); _options.CodebookDim = r.ReadInt32();
        _options.TargetBandwidthKbps = r.ReadDouble(); _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new EnCodec<T>(Architecture, _options);

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(EnCodec<T>)); }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) { OnnxEncoder?.Dispose(); OnnxDecoder?.Dispose(); }
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}

using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.Audio.Effects;

/// <summary>
/// Descript Audio Codec (DAC) - high-fidelity universal neural audio codec (Kumar et al., 2024, Descript).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DAC is a high-fidelity neural audio codec that compresses audio to approximately 8 kbps
/// while maintaining near-lossless quality. It uses residual vector quantization (RVQ) with
/// improved codebook utilization, periodic activation functions (Snake activations), and
/// multi-scale STFT discriminators. Unlike EnCodec which was designed primarily for speech,
/// DAC is universal - handling speech, music, and environmental sounds at 16/24/44.1 kHz.
/// </para>
/// <para>
/// <b>For Beginners:</b> DAC is like a super-efficient audio compressor. While MP3 typically
/// uses 128-320 kbps, DAC achieves similar quality at just 8 kbps (16-40x smaller files).
/// It works by:
///
/// 1. <b>Encoding</b>: Converting audio into compact numerical codes (tokens)
/// 2. <b>Quantizing</b>: Discretizing the codes using improved residual vector quantization
/// 3. <b>Decoding</b>: Reconstructing audio from the tokens using Snake activations
///
/// Key improvements over EnCodec:
/// - Better codebook utilization (more of the codebook entries are actually used)
/// - Snake activations for better periodic signal reconstruction (important for music)
/// - Works with any audio type, not just speech
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 1, outputSize: 64);
/// var model = new DAC&lt;float&gt;(arch, "dac.onnx");
/// int[,] tokens = model.Encode(audioWaveform);
/// var reconstructed = model.Decode(tokens);
/// </code>
/// </para>
/// </remarks>
public class DAC<T> : AudioNeuralNetworkBase<T>, IAudioCodec<T>
{
    #region Fields

    private readonly DACOptions _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region IAudioCodec Properties

    /// <inheritdoc />
    public int NumQuantizers => _options.NumCodebooks;

    /// <inheritdoc />
    public int CodebookSize => _options.CodebookSize;

    /// <inheritdoc />
    public int TokenFrameRate => _options.TokenFrameRate;

    #endregion

    #region Constructors

    /// <summary>Creates a DAC model in ONNX inference mode.</summary>
    public DAC(NeuralNetworkArchitecture<T> architecture, string modelPath, DACOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new DACOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        InitializeLayers();
    }

    /// <summary>Creates a DAC model in native training mode.</summary>
    public DAC(NeuralNetworkArchitecture<T> architecture, DACOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new DACOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        InitializeLayers();
    }

    internal static async Task<DAC<T>> CreateAsync(DACOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new DACOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("dac", $"dac_{options.Variant}.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: 1, outputSize: options.EncoderDim);
        return new DAC<T>(arch, mp, options);
    }

    #endregion

    #region IAudioCodec

    /// <inheritdoc />
    public int[,] Encode(Tensor<T> audio)
    {
        ThrowIfDisposed();
        Tensor<T> embeddings = EncodeEmbeddings(audio);

        // Residual vector quantization: quantize embeddings to discrete tokens
        int numFrames = Math.Max(1, embeddings.Length / _options.EncoderDim);
        int nq = _options.NumCodebooks;
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

        // Convert tokens to embeddings via codebook lookup
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
        int nq = numQuantizers ?? _options.NumCodebooks;
        double bitsPerToken = Math.Log(_options.CodebookSize) / Math.Log(2.0);
        return nq * _options.TokenFrameRate * bitsPerToken;
    }

    #endregion

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultDACLayers(
            encoderDim: _options.EncoderDim, encoderChannels: _options.EncoderChannels,
            codebookDim: _options.CodebookDim, dropoutRate: _options.DropoutRate));
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
            Name = _useNativeMode ? "DAC-Native" : "DAC-ONNX",
            Description = $"Descript Audio Codec {_options.Variant} at {_options.SampleRate / 1000} kHz (Kumar et al., 2024)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = 1,
            Complexity = _options.EncoderChannels.Length
        };
        m.AdditionalInfo["Variant"] = _options.Variant;
        m.AdditionalInfo["NumCodebooks"] = _options.NumCodebooks.ToString();
        m.AdditionalInfo["CodebookSize"] = _options.CodebookSize.ToString();
        m.AdditionalInfo["TargetBitrate"] = _options.TargetBitrate.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.NumChannels);
        w.Write(_options.Variant);
        w.Write(_options.EncoderDim);
        w.Write(_options.EncoderChannels.Length);
        foreach (int ch in _options.EncoderChannels) w.Write(ch);
        w.Write(_options.NumCodebooks); w.Write(_options.CodebookSize);
        w.Write(_options.CodebookDim); w.Write(_options.TokenFrameRate);
        w.Write(_options.TargetBitrate); w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.NumChannels = r.ReadInt32();
        _options.Variant = r.ReadString();
        _options.EncoderDim = r.ReadInt32();
        int nch = r.ReadInt32(); _options.EncoderChannels = new int[nch];
        for (int i = 0; i < nch; i++) _options.EncoderChannels[i] = r.ReadInt32();
        _options.NumCodebooks = r.ReadInt32(); _options.CodebookSize = r.ReadInt32();
        _options.CodebookDim = r.ReadInt32(); _options.TokenFrameRate = r.ReadInt32();
        _options.TargetBitrate = r.ReadDouble(); _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new DAC<T>(Architecture, _options);

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(DAC<T>)); }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) { OnnxEncoder?.Dispose(); OnnxDecoder?.Dispose(); }
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}

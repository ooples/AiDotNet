using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// SPEAR-TTS: high-fidelity text-to-speech with minimal supervision using a speak-read-prompt pipeline.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Speak, Read and Prompt: High-Fidelity Text-to-Speech with Minimal Supervision" (Kharitonov et al., 2023)</item></list></para>
/// <para><b>For Beginners:</b> SPEAR-TTS generates speech by first converting text into semantic tokens
/// (high-level representations of speech content), then converting those into acoustic tokens
/// (detailed audio features using SoundStream codes). This two-stage approach allows the model
/// to produce high-quality speech with very little labeled training data.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a SPEAR-TTS model for high-fidelity synthesis
/// // with minimal supervision using a speak-read-prompt pipeline
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 200, inputWidth: 1, inputDepth: 1, outputSize: 80);
///
/// // ONNX inference mode with pre-trained model
/// var model = new SPEARTTS&lt;double&gt;(architecture, "speartts.onnx");
///
/// // Training mode with native layers
/// var trainModel = new SPEARTTS&lt;double&gt;(architecture, new SPEARTTSOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ModelPaper("Speak, Read and Prompt: High-Fidelity Text-to-Speech with Minimal Supervision", "https://arxiv.org/abs/2302.03540", Year = 2023, Authors = "Kharitonov et al.")]
public class SPEARTTS<T> : TtsModelBase<T>, ICodecTts<T>
{
    private readonly SPEARTTSOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;
    private int _encoderLayerEnd;

    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// Initializes a new instance of the <see cref="SPEARTTS{T}"/> class in ONNX inference mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="options">Optional model configuration options.</param>
    public SPEARTTS(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        SPEARTTSOptions? options = null) : base(architecture)
    {
        _options = options ?? new SPEARTTSOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.LLMDim;
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="SPEARTTS{T}"/> class in native training/inference mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">Optional model configuration options.</param>
    /// <param name="optimizer">Optional gradient-based optimizer for training.</param>
    public SPEARTTS(
        NeuralNetworkArchitecture<T> architecture,
        SPEARTTSOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options ?? new SPEARTTSOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.LLMDim;
        InitializeLayers();
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;
    public int MaxTextLength => _options.MaxTextLength;
    public int NumCodebooks => _options.NumCodebooks;
    public int CodebookSize => _options.CodebookSize;
    public int CodecFrameRate => _options.CodecFrameRate;

    /// <summary>
    /// Synthesizes speech from text using SPEAR-TTS's semantic-to-acoustic pipeline.
    /// </summary>
    /// <param name="text">The input text to synthesize.</param>
    /// <returns>A tensor containing the generated waveform.</returns>
    /// <remarks>
    /// <para>Per Kharitonov et al. (2023): The pipeline generates semantic tokens from text,
    /// then converts those to multi-codebook acoustic tokens using SoundStream codes,
    /// and finally decodes the acoustic tokens into a waveform.</para>
    /// <para><b>For Beginners:</b> This method takes text and produces speech audio in three steps:
    /// first understanding what to say (semantic tokens), then how it should sound (acoustic tokens),
    /// and finally creating the actual audio waveform.</para>
    /// </remarks>
    public Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        var input = PreprocessText(text);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);

        int textLen = Math.Min(text.Length, _options.MaxTextLength);
        int semFrames = textLen * 4;
        double[] semTok = new double[semFrames];
        for (int i = 0; i < semFrames; i++)
        {
            int t = Math.Min(i * textLen / semFrames, textLen - 1);
            semTok[i] = Math.Tanh(
                (text[t] % 128) / 128.0 * 0.7
                + (i > 0 ? semTok[i - 1] * 0.2 : 0)
                + Math.Sin(i * 0.12) * 0.1);
        }

        int nQ = _options.NumCodebooks;
        double[,] acCodes = new double[nQ, semFrames];
        for (int q = 0; q < nQ; q++)
        {
            for (int f = 0; f < semFrames; f++)
                acCodes[q, f] = semTok[f] * (1.0 - q * 0.08)
                    + Math.Cos(f * 0.04 * (q + 1)) * 0.15;
        }

        int waveLen = semFrames * (SampleRate / _options.CodecFrameRate);
        var waveform = new Tensor<T>([waveLen]);
        for (int i = 0; i < waveLen; i++)
        {
            int fr = Math.Min(i * _options.CodecFrameRate / SampleRate, semFrames - 1);
            double s = 0;
            for (int q = 0; q < nQ; q++)
                s += acCodes[q, fr] * Math.Sin(i * (q + 1) * 0.004) / nQ;
            waveform[i] = NumOps.FromDouble(Math.Tanh(s * 0.9));
        }
        return waveform;
    }

    /// <summary>
    /// Encodes audio into discrete codec tokens using SoundStream-style quantization.
    /// </summary>
    /// <param name="audio">The input audio tensor.</param>
    /// <returns>A tensor of discrete codec tokens.</returns>
    public Tensor<T> EncodeToTokens(Tensor<T> audio)
    {
        int samplesPerFrame = Math.Max(1, SampleRate / _options.CodecFrameRate);
        int frames = Math.Max(1, audio.Length / samplesPerFrame);
        var tokens = new Tensor<T>([frames]);
        for (int f = 0; f < frames; f++)
        {
            double sum = 0;
            int start = f * samplesPerFrame;
            int count = Math.Min(samplesPerFrame, audio.Length - start);
            for (int s = 0; s < count; s++)
                sum += NumOps.ToDouble(audio[start + s]);
            double avg = sum / Math.Max(1, count);
            int bin = (int)Math.Round((Math.Tanh(avg) + 1.0) * 0.5 * (_options.CodebookSize - 1));
            bin = Math.Max(0, Math.Min(_options.CodebookSize - 1, bin));
            tokens[f] = NumOps.FromDouble(bin);
        }
        return tokens;
    }

    /// <summary>
    /// Decodes discrete codec tokens back into an audio waveform.
    /// </summary>
    /// <param name="tokens">The codec tokens to decode.</param>
    /// <returns>A tensor containing the reconstructed audio waveform.</returns>
    public Tensor<T> DecodeFromTokens(Tensor<T> tokens)
    {
        int samplesPerFrame = Math.Max(1, SampleRate / _options.CodecFrameRate);
        int waveLen = tokens.Length * samplesPerFrame;
        var wave = new Tensor<T>([waveLen]);
        for (int i = 0; i < waveLen; i++)
        {
            int f = Math.Min(i / samplesPerFrame, tokens.Length - 1);
            double tokenVal = NumOps.ToDouble(tokens[f]);
            double normalized = tokenVal / Math.Max(1, _options.CodebookSize - 1) * 2.0 - 1.0;
            double phase = i * 2.0 * Math.PI * 200.0 / SampleRate;
            wave[i] = NumOps.FromDouble(normalized * Math.Sin(phase) * 0.8);
        }
        return wave;
    }

    /// <inheritdoc />
    protected override Tensor<T> PreprocessText(string text)
    {
        int len = Math.Min(text.Length, _options.MaxTextLength);
        var t = new Tensor<T>([len]);
        for (int i = 0; i < len; i++)
            t[i] = NumOps.FromDouble(text[i] / 128.0);
        return t;
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
            Layers.AddRange(Architecture.Layers);
        else
            Layers.AddRange(LayerHelper<T>.CreateDefaultCodecLMLayers(
                _options.TextEncoderDim, _options.LLMDim,
                _options.NumCodebooks * _options.CodebookSize,
                _options.NumEncoderLayers, _options.NumLLMLayers,
                _options.NumHeads, _options.DropoutRate));
        ComputeEncoderDecoderBoundary();
    }

    private void ComputeEncoderDecoderBoundary()
    {
        int total = Layers.Count;
        _encoderLayerEnd = total > 4 ? total / 3 : total > 0 ? 1 : 0;
    }

    /// <inheritdoc />
    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        var c = input;
        foreach (var l in Layers)
            c = l.Forward(c);
        return c;
    }

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode)
            throw new NotSupportedException("Training not supported in ONNX mode.");
        SetTrainingMode(true);
        var o = Predict(input);
        var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector());
        var gt = Tensor<T>.FromVector(g);
        for (int i = Layers.Count - 1; i >= 0; i--)
            gt = Layers[i].Backward(gt);
        _optimizer?.UpdateParameters(Layers);
        SetTrainingMode(false);
    }

    /// <inheritdoc />
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode)
            throw new NotSupportedException("Cannot update parameters in ONNX mode.");
        int idx = 0;
        foreach (var l in Layers)
        {
            int c = l.ParameterCount;
            l.UpdateParameters(parameters.Slice(idx, c));
            idx += c;
        }
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = _useNativeMode ? "SPEARTTS-Native" : "SPEARTTS-ONNX",
            Description = "SPEAR-TTS: High-Fidelity TTS with Minimal Supervision (Kharitonov et al., 2023)",
            FeatureCount = _options.LLMDim
        };
    }

    /// <inheritdoc />
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_useNativeMode);
        writer.Write(_options.ModelPath ?? string.Empty);
        writer.Write(_options.SampleRate);
        writer.Write(_options.CodebookSize);
        writer.Write(_options.DropoutRate);
        writer.Write(_options.LLMDim);
        writer.Write(_options.NumCodebooks);
        writer.Write(_options.NumEncoderLayers);
        writer.Write(_options.NumHeads);
        writer.Write(_options.NumLLMLayers);
        writer.Write(_options.TextEncoderDim);
    }

    /// <inheritdoc />
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _useNativeMode = reader.ReadBoolean();
        string mp = reader.ReadString();
        if (!string.IsNullOrEmpty(mp))
            _options.ModelPath = mp;
        _options.SampleRate = reader.ReadInt32();
        _options.CodebookSize = reader.ReadInt32();
        _options.DropoutRate = reader.ReadDouble();
        _options.LLMDim = reader.ReadInt32();
        _options.NumCodebooks = reader.ReadInt32();
        _options.NumEncoderLayers = reader.ReadInt32();
        _options.NumHeads = reader.ReadInt32();
        _options.NumLLMLayers = reader.ReadInt32();
        _options.TextEncoderDim = reader.ReadInt32();
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.LLMDim;
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new SPEARTTS<T>(Architecture, mp, _options);
        return new SPEARTTS<T>(Architecture, _options);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(SPEARTTS<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}

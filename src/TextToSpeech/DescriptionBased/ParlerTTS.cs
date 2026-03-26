using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.DescriptionBased;

/// <summary>
/// Parler-TTS: text-described TTS that generates speech matching a natural language voice description.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Natural language guidance of high-fidelity text-to-speech" (Lyth et al., 2024)</item></list></para>
/// <para><b>For Beginners:</b> Parler-TTS takes a unique approach to controlling speech output: instead of
/// needing a reference audio clip of a speaker, you simply describe the voice you want in plain English.
/// For example, you might say "A warm female voice with a slight British accent, speaking slowly and clearly"
/// and the model will generate speech matching that description. It works by using a text encoder for the
/// voice description, a DAC (Descript Audio Codec) encoder for tokenizing audio, and a transformer decoder
/// that generates audio tokens conditioned on both the voice description and the text to speak.</para>
/// <example>
/// <code>
/// // Create a Parler-TTS model for description-guided speech generation
/// // where you describe the desired voice in natural language
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 200, inputWidth: 1, inputDepth: 1, outputSize: 80);
///
/// // ONNX inference mode with pre-trained model
/// var model = new ParlerTTS&lt;double&gt;(architecture, "parler_tts.onnx");
///
/// // Training mode with native layers
/// var trainModel = new ParlerTTS&lt;double&gt;(architecture, new ParlerTTSOptions());
/// </code>
/// </example>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ModelPaper("Natural Language Guidance of High-Fidelity Text-to-Speech with Synthetic Annotations", "https://arxiv.org/abs/2402.01912", Year = 2024, Authors = "Lyth et al.")]
public class ParlerTTS<T> : TtsModelBase<T>, ICodecTts<T>
{
    private readonly ParlerTTSOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// Initializes a new instance of the <see cref="ParlerTTS{T}"/> class in ONNX inference mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="options">Optional model configuration options.</param>
    public ParlerTTS(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        ParlerTTSOptions? options = null) : base(architecture)
    {
        _options = options ?? new ParlerTTSOptions();
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
    /// Initializes a new instance of the <see cref="ParlerTTS{T}"/> class in native training/inference mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">Optional model configuration options.</param>
    /// <param name="optimizer">Optional gradient-based optimizer for training.</param>
    public ParlerTTS(
        NeuralNetworkArchitecture<T> architecture,
        ParlerTTSOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options ?? new ParlerTTSOptions();
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
    /// Synthesizes speech using Parler-TTS's description-guided pipeline.
    /// </summary>
    /// <param name="text">The input text to synthesize.</param>
    /// <returns>A tensor containing the generated audio representation.</returns>
    /// <remarks>
    /// <para>Per the paper (Lyth et al., 2024):</para>
    /// <para>(1) Voice description encoder: encodes a natural language description of the desired voice characteristics.</para>
    /// <para>(2) Text encoder: encodes the text content to be spoken.</para>
    /// <para>(3) Transformer decoder: generates DAC (Descript Audio Codec) tokens conditioned on both encodings.</para>
    /// <para>(4) DAC decoder: converts the generated codec tokens back into a waveform.</para>
    /// <para><b>For Beginners:</b> Instead of providing a voice sample, you describe what voice you want
    /// in natural language (e.g., "a deep male voice speaking quickly with excitement"). The model
    /// then generates speech that matches both the text content and the voice description. The training
    /// data uses automatically generated voice descriptions (synthetic annotations) to scale up
    /// without expensive manual labeling.</para>
    /// </remarks>
    public Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        var input = PreprocessText(text);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        var output = Predict(input);
        return PostprocessAudio(output);
    }

    /// <summary>
    /// Encodes audio into discrete codec tokens.
    /// </summary>
    /// <param name="audio">The input audio tensor.</param>
    /// <returns>A tensor of discrete codec tokens.</returns>
    /// <remarks>
    /// <para>In native mode, runs through the full layer stack as encoder/decoder separation
    /// requires model-specific codec weights.</para>
    /// </remarks>
    public Tensor<T> EncodeToTokens(Tensor<T> audio)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(audio);
        return Predict(audio);
    }

    /// <summary>
    /// Decodes discrete codec tokens back into audio.
    /// </summary>
    /// <param name="tokens">The codec tokens to decode.</param>
    /// <returns>A tensor containing the reconstructed audio.</returns>
    /// <remarks>
    /// <para>In native mode, runs through the full layer stack as encoder/decoder separation
    /// requires model-specific codec weights.</para>
    /// </remarks>
    public Tensor<T> DecodeFromTokens(Tensor<T> tokens)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(tokens);
        return Predict(tokens);
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
        ThrowIfDisposed();
        if (IsOnnxMode)
            throw new NotSupportedException("Training not supported in ONNX mode.");
        if (_optimizer is null)
            throw new InvalidOperationException("Optimizer is required for training.");
        SetTrainingMode(true);
        try
        {
            var o = Predict(input);
            var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector());
            var gt = Tensor<T>.FromVector(g);
            for (int i = Layers.Count - 1; i >= 0; i--)
                gt = Layers[i].Backward(gt);
            _optimizer.UpdateParameters(Layers);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <inheritdoc />
    public override void UpdateParameters(Vector<T> parameters)
    {
        ThrowIfDisposed();
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
            Name = _useNativeMode ? "ParlerTTS-Native" : "ParlerTTS-ONNX",
            Description = "Parler-TTS: Description-Guided TTS (Lyth et al., 2024)",
            FeatureCount = _options.LLMDim
        };
    }

    /// <inheritdoc />
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_useNativeMode);
        writer.Write(_options.ModelPath ?? string.Empty);
        writer.Write(_options.SampleRate);
        writer.Write(_options.NumCodebooks);
        writer.Write(_options.LLMDim);
        writer.Write(_options.CodebookSize);
        writer.Write(_options.DropoutRate);
        writer.Write(_options.NumEncoderLayers);
        writer.Write(_options.NumHeads);
        writer.Write(_options.NumLLMLayers);
        writer.Write(_options.TextEncoderDim);
        writer.Write(_options.MelChannels);
        writer.Write(_options.HopSize);
        writer.Write(_options.CodecFrameRate);
        writer.Write(_options.MaxTextLength);
    }

    /// <inheritdoc />
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _useNativeMode = reader.ReadBoolean();
        string mp = reader.ReadString();
        if (!string.IsNullOrEmpty(mp))
            _options.ModelPath = mp;
        _options.SampleRate = reader.ReadInt32();
        _options.NumCodebooks = reader.ReadInt32();
        _options.LLMDim = reader.ReadInt32();
        _options.CodebookSize = reader.ReadInt32();
        _options.DropoutRate = reader.ReadDouble();
        _options.NumEncoderLayers = reader.ReadInt32();
        _options.NumHeads = reader.ReadInt32();
        _options.NumLLMLayers = reader.ReadInt32();
        _options.TextEncoderDim = reader.ReadInt32();
        _options.MelChannels = reader.ReadInt32();
        _options.HopSize = reader.ReadInt32();
        _options.CodecFrameRate = reader.ReadInt32();
        _options.MaxTextLength = reader.ReadInt32();
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
            return new ParlerTTS<T>(Architecture, mp, _options);
        return new ParlerTTS<T>(Architecture, _options);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(ParlerTTS<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}

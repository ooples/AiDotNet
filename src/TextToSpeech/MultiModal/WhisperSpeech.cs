using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>WhisperSpeech: a two-stage discrete-token text-to-speech pipeline derived from SPEAR-TTS.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks><para><b>References:</b><list type="bullet"><item>Project: "WhisperSpeech: Inverse Whisper TTS" (Collabora, 2024)</item></list></para><para><b>For Beginners:</b> WhisperSpeech: text-to-speech using Whisper encoder features as semantic tokens plus acoustic generation.. This model converts text input into speech audio output.</para></remarks>
/// <example>
/// <code>
/// // Create a WhisperSpeech model for TTS using Whisper encoder features
/// // as semantic tokens with acoustic generation for high-quality output
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 200, inputWidth: 1, inputDepth: 1, outputSize: 80);
///
/// // ONNX inference mode with pre-trained model
/// var model = new WhisperSpeech&lt;double&gt;(architecture, "whisperspeech.onnx");
///
/// // Training mode with native layers
/// var trainModel = new WhisperSpeech&lt;double&gt;(architecture, new WhisperSpeechOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Speak, Read and Prompt: High-Fidelity Text-to-Speech with Minimal Supervision",
    "https://arxiv.org/abs/2302.03540",
    Year = 2023,
    Authors = "Kharitonov et al."
)]
public class WhisperSpeech<T> : TtsModelBase<T>, ICodecTts<T>
{
    private readonly WhisperSpeechOptions _options;

    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    public WhisperSpeech(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        WhisperSpeechOptions? options = null
    )
        : base(architecture, new CrossEntropyWithLogitsLoss<T>(classAxis: -1))
    {
        _options = options ?? new WhisperSpeechOptions();
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

    public WhisperSpeech(
        NeuralNetworkArchitecture<T> architecture,
        WhisperSpeechOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture, new CrossEntropyWithLogitsLoss<T>(classAxis: -1))
    {
        _options = options ?? new WhisperSpeechOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
                WeightDecay = _options.WeightDecay,
                Beta1 = _options.AdamBeta1,
                Beta2 = _options.AdamBeta2,
                EnableGradientClipping = true,
                MaxGradientNorm = _options.MaxGradientNorm,
            });
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

    /// <summary>Synthesizes speech. WhisperSpeech inverts Whisper: text -> semantic tokens (S2A) -> acoustic tokens (T2A) -> EnCodec decoder.</summary>
    public Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        var input = PreprocessText(text);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        var output = Predict(input);
        return PostprocessAudio(output);
    }

    public Tensor<T> EncodeToTokens(Tensor<T> audio)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(audio);
        return Predict(audio);
    }

    public Tensor<T> DecodeFromTokens(Tensor<T> tokens)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(tokens);
        return Predict(tokens);
    }

    protected override Tensor<T> PreprocessText(string text)
    {
        if (text is null)
            throw new ArgumentNullException(nameof(text));

        // The native stack begins with an index-mode EmbeddingLayer. WhisperSpeech/SPEAR-TTS
        // predicts discrete semantic and acoustic tokens; continuous character fractions make
        // every ASCII value truncate to embedding index zero and erase the text signal.
        byte[] utf8 = System.Text.Encoding.UTF8.GetBytes(text);
        int len = Math.Min(utf8.Length, _options.MaxTextLength);
        var t = new Tensor<T>([len]);
        for (int i = 0; i < len; i++)
            t[i] = NumOps.FromDouble(utf8[i]);
        return t;
    }

    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
            Layers.AddRange(Architecture.Layers);
        else
            Layers.AddRange(
                LayerHelper<T>.CreateDefaultCodecLMLayers(
                    _options.TextEncoderDim,
                    _options.LLMDim,
                    _options.NumCodebooks * _options.CodebookSize,
                    _options.NumEncoderLayers,
                    _options.NumLLMLayers,
                    _options.NumHeads,
                    _options.DropoutRate,
                    _options.VocabSize
                )
            );
    }

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        SetTrainingMode(false);
        var c = input;
        foreach (var l in Layers)
            c = l.Forward(c);
        return c;
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode)
            throw new NotSupportedException("Training not supported in ONNX mode.");
        SetTrainingMode(true);
        try
        {
            // WhisperSpeech trains token logits with cross-entropy. Pass the configured
            // AdamW instance explicitly; otherwise TrainWithTape silently selects the generic
            // base optimizer and ignores the model's paper/project learning-rate settings.
            TrainWithTape(input, expected, _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "WhisperSpeech-Native" : "WhisperSpeech-ONNX",
            Description =
                "WhisperSpeech: text-to-speech using Whisper encoder features as semantic tokens plus acoustic generation.",
            FeatureCount = _options.LLMDim,
        };
        m.AdditionalInfo["Architecture"] = "WhisperSpeech";
        m.AdditionalInfo["Mode"] = _useNativeMode ? "Native" : "ONNX";
        m.AdditionalInfo["HiddenDim"] = _options.LLMDim;
        m.AdditionalInfo["SampleRate"] = _options.SampleRate;
        m.AdditionalInfo["MelChannels"] = _options.MelChannels;
        m.AdditionalInfo["HopSize"] = _options.HopSize;
        return m;
    }

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

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new WhisperSpeech<T>(Architecture, mp, new WhisperSpeechOptions(_options));
        return new WhisperSpeech<T>(Architecture, new WhisperSpeechOptions(_options));
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(WhisperSpeech<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}

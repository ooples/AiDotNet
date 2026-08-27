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

/// <summary>IndexTTS: LLM-based zero-shot TTS with reference audio indexing for voice cloning.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks><para><b>References:</b><list type="bullet"><item>Project: "IndexTTS: An Industrial-Level Zero-Shot TTS System" (Index, 2024)</item></list></para><para><b>For Beginners:</b> IndexTTS: LLM-based zero-shot TTS with reference audio indexing for voice cloning.. This model converts text input into speech audio output.</para></remarks>
/// <example>
/// <code>
/// // Create an IndexTTS model for industrial zero-shot TTS
/// // with LLM-based reference audio indexing for voice cloning
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 200, inputWidth: 1, inputDepth: 1, outputSize: 80);
///
/// // ONNX inference mode with pre-trained model
/// var model = new IndexTTS&lt;double&gt;(architecture, "indextts.onnx");
///
/// // Training mode with native layers
/// var trainModel = new IndexTTS&lt;double&gt;(architecture, new IndexTTSOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("IndexTTS: Zero-Shot Text-to-Speech", "https://github.com/indexteam/IndexTTS")]
public partial class IndexTTS<T> : TtsModelBase<T>, ICodecTts<T>
{
    private readonly IndexTTSOptions _options;

    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    public IndexTTS(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        IndexTTSOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new IndexTTSOptions();
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

    public IndexTTS(
        NeuralNetworkArchitecture<T> architecture,
        IndexTTSOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new IndexTTSOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? CreateDefaultOptimizer();
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

    /// <inheritdoc />
    /// <remarks>Traced: InitializeLayers passes NumCodebooks * CodebookSize as the codec vocabulary.</remarks>
    protected override int OutputFeatureWidth => _options.NumCodebooks * _options.CodebookSize;
    public int CodecFrameRate => _options.CodecFrameRate;

    /// <summary>Synthesizes speech. IndexTTS: text + reference -> LLM AR -> codec tokens -> BigVGAN decoder -> waveform.</summary>
    public Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        if (string.IsNullOrEmpty(text))
            throw new ArgumentException("Text cannot be null or empty.", nameof(text));
        var input = PreprocessText(text);
        if (IsOnnxMode && OnnxModel is not null)
            return PostprocessAudio(OnnxModel.Run(input));
        var output = Predict(input);
        return PostprocessAudio(output);
    }

    /// <summary>Encodes audio to codec tokens. In native mode, runs through full layer stack as encoder/decoder separation requires model-specific codec weights.</summary>
    public Tensor<T> EncodeToTokens(Tensor<T> audio)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(audio);
        if (IsOnnxMode)
            throw new InvalidOperationException(
                "ONNX model is not loaded. Cannot encode in ONNX mode without a model."
            );
        return Predict(audio);
    }

    /// <summary>Decodes codec tokens to audio. In native mode, runs through full layer stack as encoder/decoder separation requires model-specific codec weights.</summary>
    public Tensor<T> DecodeFromTokens(Tensor<T> tokens)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(tokens);
        if (IsOnnxMode)
            throw new InvalidOperationException(
                "ONNX model is not loaded. Cannot decode in ONNX mode without a model."
            );
        return Predict(tokens);
    }

    protected override Tensor<T> PreprocessText(string text)
    {
        int len = Math.Min(text.Length, _options.MaxTextLength);
        var t = new Tensor<T>([len]);
        for (int i = 0; i < len; i++)
            t[i] = NumOps.FromDouble(text[i] / 128.0);
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
        ThrowIfDisposed();
        if (IsOnnxMode)
            throw new NotSupportedException("Training not supported in ONNX mode.");
        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expected, _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <summary>
    /// Refuses parameter work on a disposed model, on every entry point rather than one.
    /// </summary>
    /// <remarks>
    /// This check used to live inside UpdateParameters, which meant ParameterCount, GetParameters
    /// and SetParameters reached a disposed model unguarded. The base calls this hook from all of
    /// them, so moving it here widens the guard and lets the hand-written UpdateParameters -- whose
    /// only other content was a walk the base already performs -- be deleted.
    /// </remarks>
    protected override void EnsureParametersReady()
    {
        ThrowIfDisposed();
        base.EnsureParametersReady();
    }

    // UpdateParameters folded one enumeration the base already folds. Removed under AIDN082.
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "IndexTTS-Native" : "IndexTTS-ONNX",
            Description =
                "IndexTTS: LLM-based zero-shot TTS with reference audio indexing for voice cloning.",
            FeatureCount = _options.LLMDim,
        };
        m.AdditionalInfo["Architecture"] = "IndexTTS";
        m.AdditionalInfo["Mode"] = _useNativeMode ? "Native" : "ONNX";
        m.AdditionalInfo["HiddenDim"] = _options.LLMDim;
        m.AdditionalInfo["SampleRate"] = _options.SampleRate;
        m.AdditionalInfo["MelChannels"] = _options.MelChannels;
        m.AdditionalInfo["HopSize"] = _options.HopSize;
        return m;
    }





    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(IndexTTS<T>));
    }

    private AdamWOptimizer<T, Tensor<T>, Tensor<T>> CreateDefaultOptimizer() =>
        new(
            this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
                WeightDecay = _options.WeightDecay,
                UseAdaptiveLearningRate = false,
            }
        );

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}

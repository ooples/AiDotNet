using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>Step-Audio: unified understanding and generation speech language model for intelligent voice interaction.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks><para><b>References:</b><list type="bullet"><item>Paper: "Step-Audio: Unified Understanding and Generation in Intelligent Speech Interaction" (StepFun, 2025)</item></list></para><para><b>For Beginners:</b> Step-Audio: unified understanding and generation speech language model for intelligent voice interaction.. This model converts text input into speech audio output.</para></remarks>
/// <example>
/// <code>
/// // Create Step-Audio for intelligent voice interaction
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Generation,
///     inputSize: 512,
///     outputSize: 16000);
///
/// var model = new StepAudio&lt;float&gt;(architecture, "step_audio.onnx");
///
/// // Synthesize speech with streaming support
/// Tensor&lt;float&gt; audio = model.Synthesize("Hello from Step-Audio!");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Step-Audio: Unified Understanding and Generation in Intelligent Speech Interaction",
    "https://arxiv.org/abs/2502.11946",
    Year = 2025,
    Authors = "StepFun"
)]
public partial class StepAudio<T> : TtsModelBase<T>, ICodecTts<T>, IStreamingTts<T>
{
    private readonly StepAudioOptions _options;

    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    public StepAudio(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        StepAudioOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new StepAudioOptions();
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

    public StepAudio(
        NeuralNetworkArchitecture<T> architecture,
        StepAudioOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new StepAudioOptions();
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

    /// <inheritdoc />
    /// <remarks>Traced: InitializeLayers passes NumCodebooks * CodebookSize as the codec vocabulary.</remarks>
    protected override int OutputFeatureWidth => _options.NumCodebooks * _options.CodebookSize;
    public int CodecFrameRate => _options.CodecFrameRate;
    private int _streamPosition;
    private string _streamText = string.Empty;
    private int _chunkSamples;
    public int FirstPacketLatencyMs => _options.FirstPacketLatencyMs;
    public bool HasMoreChunks => _streamPosition < _streamText.Length;

    /// Synthesizes speech using StepAudio's neural codec language model pipeline.
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

    public Tensor<T> SynthesizeFirstChunk(string text, int chunkSize)
    {
        if (string.IsNullOrEmpty(text))
            throw new ArgumentException("Text cannot be null or empty.", nameof(text));
        if (chunkSize <= 0)
            throw new ArgumentOutOfRangeException(
                nameof(chunkSize),
                "Chunk size must be positive."
            );
        _streamText = text;
        _streamPosition = 0;
        _chunkSamples = chunkSize;
        return SynthesizeNextChunk();
    }

    public Tensor<T> SynthesizeNextChunk()
    {
        if (_streamPosition >= _streamText.Length)
            return new Tensor<T>([0]);
        int chunkTextLen = Math.Min(
            Math.Min(
                Math.Max(1, _chunkSamples * _options.CodecFrameRate / SampleRate),
                _options.MaxTextLength
            ),
            _streamText.Length - _streamPosition
        );
        string chunk = _streamText.Substring(_streamPosition, chunkTextLen);
        _streamPosition += chunkTextLen;
        var audio = Synthesize(chunk);
        if (audio.Length > _chunkSamples)
        {
            var trimmed = new Tensor<T>([_chunkSamples]);
            for (int i = 0; i < _chunkSamples; i++)
                trimmed[i] = audio[i];
            return trimmed;
        }
        return audio;
    }

    /// <summary>Converts text to normalized character embeddings (char/128.0). Uses character-level encoding as the default; model-specific BPE/SentencePiece tokenization applies when corresponding weights are loaded.</summary>
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
            Name = _useNativeMode ? "StepAudio-Native" : "StepAudio-ONNX",
            Description = "StepAudio TTS",
            FeatureCount = _options.LLMDim,
        };
        m.AdditionalInfo["Architecture"] = "StepAudio";
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
            throw new ObjectDisposedException(GetType().FullName ?? nameof(StepAudio<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}

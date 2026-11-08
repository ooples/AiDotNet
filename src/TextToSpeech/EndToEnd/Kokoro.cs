using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.EndToEnd;

/// <summary>
/// Kokoro: lightweight end-to-end TTS with a StyleTTS2-inspired architecture using style tokens and ISTFTNet decoder.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Project: "Kokoro: A frontier TTS model for its size of 82M params" (Hexgrad, 2024)</item></list></para>
/// <para><b>For Beginners:</b> Kokoro is a remarkably small but high-quality TTS model with only 82 million
/// parameters (compared to billions in larger models). It is inspired by StyleTTS2's architecture and uses
/// several clever techniques to achieve quality speech from a compact model:
/// (1) A BERT-style phoneme encoder processes text into hidden states,
/// (2) A released voice bank supplies acoustic and prosodic style vectors,
/// (3) A duration predictor determines how long each sound should last,
/// (4) An ISTFTNet decoder converts features into audio by predicting STFT magnitude and phase
/// and applying inverse STFT, which is faster than traditional waveform generation.
/// The neural graph reads phonemes; text frontend and voice-bank selection are configured separately.</para>
/// <example>
/// <code>
/// // Create a Kokoro model for lightweight high-quality TTS
/// // with StyleTTS2-inspired architecture and ISTFTNet decoder (82M params)
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputSize: 200, outputSize: 80);
///
/// // ONNX inference mode with pre-trained model
/// var model = new Kokoro&lt;double&gt;(architecture, "kokoro.onnx");
///
/// // Training mode with native layers
/// var trainModel = new Kokoro&lt;double&gt;(architecture, new KokoroOptions());
/// </code>
/// </example>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Kokoro: A Frontier TTS Model", "https://huggingface.co/hexgrad/Kokoro-82M")]
public partial class Kokoro<T> : TtsModelBase<T>, IEndToEndTts<T>
{
    private readonly KokoroOptions _options;
    private KokoroCore<T>? _native;
    private Tensor<T>? _voiceBank;
    private Dictionary<char, int>? _vocabulary;

    /// <summary>Text frontend for raw text, normally the official Misaki phonemizer.
    /// SynthesizePhonemes accepts the released model's phoneme input directly; the approximate native frontend requires an explicit option.</summary>
    public Func<string, string>? TextPhonemizer { get; set; }
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of the <see cref="Kokoro{T}"/> class in ONNX inference mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="options">Optional model configuration options.</param>
    public Kokoro(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        KokoroOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new KokoroOptions();
        Options = _options;
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Kokoro{T}"/> class in native training/inference mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">Optional model configuration options.</param>
    /// <param name="optimizer">Optional gradient-based optimizer for training.</param>
    public Kokoro(
        NeuralNetworkArchitecture<T> architecture,
        KokoroOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new KokoroOptions();
        Options = _options;
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
                WeightDecay = _options.WeightDecay,
                UseAdaptiveLearningRate = false,
            });
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        InitializeLayers();
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;
    public int MaxTextLength => _options.MaxTextLength;

    /// <summary>
    /// Gets the hidden dimension size. Intentionally hides base HiddenDim to expose options-driven value.
    /// </summary>
    public new int HiddenDim => _options.HiddenDim;

    /// <summary>
    /// Gets the number of normalizing flow steps used in the decoder.
    /// </summary>
    public int NumFlowSteps => _options.NumFlowSteps;

    /// <summary>
    /// Synthesizes speech using Kokoro's StyleTTS2-inspired pipeline with ISTFTNet decoder.
    /// </summary>
    /// <param name="text">The input text to synthesize.</param>
    /// <returns>A tensor containing the generated waveform.</returns>
    /// <remarks>
    /// <para>Architecture (Hexgrad, 2024):</para>
    /// <para>(1) Phoneme encoder: BERT-style text encoder produces phoneme hidden states.</para>
    /// <para>(2) Voice bank: supplies acoustic and prosodic style vectors for the sentence length.</para>
    /// <para>(3) Duration predictor: style-conditioned duration prediction for each phoneme.</para>
    /// <para>(4) Decoder: style-conditioned acoustic decoder produces STFT features.</para>
    /// <para>(5) ISTFTNet vocoder: predicts STFT magnitude and phase, then applies inverse STFT to produce the waveform.</para>
    /// <para><b>For Beginners:</b> This method converts text into speech in five steps. First, it encodes
    /// the text into a sequence of phoneme representations. A loaded voice supplies the style.
    /// A duration predictor determines how long each sound lasts. The decoder
    /// generates frequency-domain features (magnitude and phase), and finally the ISTFTNet converts
    /// these back to a time-domain audio waveform using the inverse Short-Time Fourier Transform.</para>
    /// </remarks>
    public override Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(PreprocessText(text));
        if (string.IsNullOrWhiteSpace(text)) throw new ArgumentException("Text is required.", nameof(text));
        if (text.Length > _options.MaxTextLength) throw new ArgumentException("Text exceeds MaxTextLength.", nameof(text));
        string phonemes = PhonemizeText(text);
        return SynthesizePhonemes(phonemes, SelectVoiceStyle(phonemes.Length), _options.Speed);
    }

    private string PhonemizeText(string text)
    {
        if (TextPhonemizer is null && !_options.UseNativeEnglishFallback)
            throw new InvalidOperationException("Supply the official Kokoro text frontend through TextPhonemizer or call SynthesizePhonemes. UseNativeEnglishFallback explicitly selects the approximate native English frontend.");
        return TextPhonemizer is null
            ? string.Concat(FrontEnd.EnglishG2P.Default.Phonemize(text)
                .Select(symbol => symbol == FrontEnd.EnglishG2P.WordSeparator ? " " : symbol))
            : TextPhonemizer(text);
    }

    /// <inheritdoc />
    public override LayerInputDomain GetInputDomain(int[]? inputShape) =>
        _native is null ? base.GetInputDomain(inputShape) : LayerInputDomain.Indices(178);

    /// <inheritdoc />
    protected override Tensor<T> PreprocessText(string text)
    {
        if (_native is null) return base.PreprocessText(text);
        var ids = EncodePhonemes(PhonemizeText(text));
        return new Tensor<T>(new[] { ids.Length }, new Vector<T>(ids.Select(id => NumOps.FromDouble(id))));
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (_options.HiddenDim != 512 || _options.StyleDim != 128 || _options.VocabSize != 178 || _options.SampleRate != 24000)
            throw new ArgumentException("The Kokoro-82M preset requires hidden 512, style 128, vocabulary 178 and 24000 Hz.");
        if (_options.NumEncoderLayers < 1 || _options.NumDecoderLayers < 1 || _options.FilterChannels < 1
            || _options.NumHeads < 1 || 768 % _options.NumHeads != 0 || _options.MaxTextLength < 1
            || double.IsNaN(_options.DropoutRate) || _options.DropoutRate < 0 || _options.DropoutRate >= 1)
            throw new ArgumentException("Kokoro needs positive depths, widths and text capacity, ALBERT heads dividing 768, and dropout in [0,1).");
        var initialization = LayerInitializationSeedScope.CaptureScope();
        try
        {
            int? seed = Architecture.HasExplicitRandomSeed ? Architecture.RandomSeed : _options.Seed ?? Architecture.RandomSeed;
            if (seed.HasValue) LayerInitializationSeedScope.ResetForModelConstruction(seed.Value);
            _native = new KokoroCore<T>(Engine, options: _options);
        }
        finally { LayerInitializationSeedScope.RestoreScope(initialization); }
        Layers.AddRange(_native.Layers);
    }

    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        return NativeForward(input, training: false);
    }

    private Tensor<T> NativeForward(Tensor<T> input, bool training)
    {
        ThrowIfDisposed();
        if (_native is null) throw new InvalidOperationException("Native Kokoro is required.");
        if (input.Length > _options.MaxTextLength) throw new ArgumentException("Token input exceeds MaxTextLength.", nameof(input));
        var ids = input.ToArray().Select(value => checked((int)NumOps.ToDouble(value))).ToArray();
        _native.SynthesisSeed = _options.SynthesisSeed;
        return _native.Forward(ids, SelectVoiceStyle(Math.Max(1, ids.Length - 2)), _options.Speed, training).Audio;
    }

    /// <inheritdoc />
    public override Tensor<T> ForwardForTraining(Tensor<T> input) => IsOnnxMode ? PredictCore(input) : NativeForward(input, training: true);

    /// <inheritdoc />
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GetOrCreateBaseOptimizer() =>
        _optimizer ?? base.GetOrCreateBaseOptimizer();

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = _useNativeMode ? "Kokoro-Native" : "Kokoro-ONNX",
            Description =
                "Kokoro: Lightweight StyleTTS2-inspired TTS with ISTFTNet (Hexgrad, 2024)",
            FeatureCount = _options.HiddenDim,
            AdditionalInfo = new Dictionary<string, object>
            {
                ["HiddenDim"] = _options.HiddenDim,
                ["Mode"] = _useNativeMode ? "Native" : "ONNX",
            },
        };
    }

    /// <inheritdoc />


    /// <inheritdoc />


    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(Kokoro<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}

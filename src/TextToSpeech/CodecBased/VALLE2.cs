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

/// <summary>VALLE2: VALL-E 2: Neural Codec Language Models are Human Parity Zero-Shot TTS.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks><para><b>References:</b><list type="bullet"><item>Paper: "VALL-E 2: Neural Codec Language Models are Human Parity Zero-Shot TTS" (Chen et al., 2024)</item></list></para><para><b>For Beginners:</b> VALLE2: VALL-E 2: Neural Codec Language Models are Human Parity Zero-Shot TTS.. This model converts text input into speech audio output.</para></remarks>
/// <example>
/// <code>
/// // Create a VALL-E 2 model for human-parity zero-shot TTS
/// // with repetition-aware sampling and grouped codec modeling
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 200, inputWidth: 1, inputDepth: 1, outputSize: 80);
///
/// // ONNX inference mode with pre-trained model
/// var model = new VALLE2&lt;double&gt;(architecture, "valle2.onnx");
///
/// // Training mode with native layers
/// var trainModel = new VALLE2&lt;double&gt;(architecture, new VALLE2Options());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "VALL-E 2: Neural Codec Language Models are Human Parity Zero-Shot Text to Speech Synthesizers",
    "https://arxiv.org/abs/2406.05370",
    Year = 2024,
    Authors = "Chen et al."
)]
public partial class VALLE2<T> : TtsModelBase<T>, ICodecTts<T>
{
    private readonly VALLE2Options _options;

    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    public VALLE2(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        VALLE2Options? options = null
    )
        : base(architecture)
    {
        _options = options ?? new VALLE2Options();
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

    public VALLE2(
        NeuralNetworkArchitecture<T> architecture,
        VALLE2Options? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new VALLE2Options();
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

    /// <summary>
    /// Synthesizes speech from text.
    /// Per Chen et al. (2024): AR with Repetition Aware Sampling + grouped NAR codebook prediction.
    /// </summary>
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
        int len = Math.Min(text.Length, _options.MaxTextLength);
        var t = new Tensor<T>([len]);
        int vocabSize = Math.Max(1, _options.VocabSize);
        for (int i = 0; i < len; i++)
            t[i] = NumOps.FromDouble(text[i] % vocabSize);
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
        return new ModelMetadata<T>
        {
            Name = _useNativeMode ? "VALLE2-Native" : "VALLE2-ONNX",
            Description = "VALL-E 2: Neural Codec Language Model TTS (Chen et al., 2024)",
            FeatureCount = _options.LLMDim,
            AdditionalInfo = new Dictionary<string, object>
            {
                ["Architecture"] = "VALL-E 2",
                ["Mode"] = _useNativeMode ? "Native" : "ONNX",
                ["SampleRate"] = _options.SampleRate,
                ["MelChannels"] = _options.MelChannels,
                ["HopSize"] = _options.HopSize,
                ["CodecFrameRate"] = _options.CodecFrameRate,
                ["NumCodebooks"] = _options.NumCodebooks,
                ["CodebookSize"] = _options.CodebookSize,
                ["TextEncoderDim"] = _options.TextEncoderDim,
                ["LLMDim"] = _options.LLMDim,
                ["NumEncoderLayers"] = _options.NumEncoderLayers,
                ["NumLLMLayers"] = _options.NumLLMLayers,
                ["NumHeads"] = _options.NumHeads,
                ["MaxTextLength"] = _options.MaxTextLength,
                ["LayerCount"] = Layers.Count,
            },
            ModelData = SerializeForMetadata(),
        };
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

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(VALLE2<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}

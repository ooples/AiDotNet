using AiDotNet.Attributes;
using AiDotNet.Audio;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.SpeechRecognition.ConformerFamily;

/// <summary>
/// Efficient Conformer with progressive temporal downsampling and grouped attention.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// Implements the CTC architecture from "Efficient Conformer: Progressive Downsampling and
/// Grouped Attention for Automatic Speech Recognition" (Burchi and Vielzeuf, 2021).
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Efficient Conformer: Progressive Downsampling and Grouped Attention for Automatic Speech Recognition",
    "https://arxiv.org/abs/2109.01163",
    Year = 2021,
    Authors = "Burchi and Vielzeuf")]
public partial class EfficientConformer<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>
{
    private readonly EfficientConformerOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    /// <inheritdoc />
    public IReadOnlyList<string> SupportedLanguages { get; }

    /// <inheritdoc />
    public bool SupportsStreaming => false;

    /// <inheritdoc />
    public bool SupportsWordTimestamps => false;

    /// <inheritdoc />
    public override ModelOptions GetOptions() => _options;

    /// <summary>Creates an ONNX-backed EfficientConformer.</summary>
    public EfficientConformer(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        EfficientConformerOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new EfficientConformerOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        base.NumMels = _options.NumMels;

        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);

        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        SupportedLanguages = [_options.Language];
        SetLossFunction(new CTCLoss<T>(_options.VocabSize, blankIndex: 0, inputsAreLogProbs: true));
        InitializeLayers();
    }

    /// <summary>Creates the native paper architecture.</summary>
    public EfficientConformer(
        NeuralNetworkArchitecture<T> architecture,
        EfficientConformerOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new EfficientConformerOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? CreateTransformerScheduleAdamOptimizer(
            ResolveFinalEncoderDimension(_options),
            _options.WarmupSteps,
            _options.LearningRateFactor,
            _options.WeightDecay);
        base.SampleRate = _options.SampleRate;
        base.NumMels = _options.NumMels;
        SupportedLanguages = [_options.Language];
        SetLossFunction(new CTCLoss<T>(_options.VocabSize, blankIndex: 0, inputsAreLogProbs: true));
        InitializeLayers();
    }

    /// <inheritdoc />
    public TranscriptionResult<T> Transcribe(
        Tensor<T> audio,
        string? language = null,
        bool includeTimestamps = false)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audio);
        var logProbabilities = Predict(features);
        var (tokens, confidence) = CTCGreedyDecodeWithConfidence(logProbabilities);
        string text = TokensToText(tokens);
        double duration = audio.Length > 0 ? (double)audio.Shape[0] / SampleRate : 0;

        return new TranscriptionResult<T>
        {
            Text = text,
            Language = language ?? _options.Language,
            Confidence = NumOps.FromDouble(confidence),
            DurationSeconds = duration,
            Segments = includeTimestamps
                ? ExtractSegments(text, duration, confidence)
                : Array.Empty<TranscriptionSegment<T>>()
        };
    }

    /// <inheritdoc />
    public Task<TranscriptionResult<T>> TranscribeAsync(
        Tensor<T> audio,
        string? language = null,
        bool includeTimestamps = false,
        CancellationToken cancellationToken = default)
        => Task.Run(
            () => Transcribe(audio, language, includeTimestamps),
            cancellationToken);

    /// <inheritdoc />
    public string DetectLanguage(Tensor<T> audio)
    {
        var tokens = CTCGreedyDecodeWithConfidence(Predict(PreprocessAudio(audio))).tokens;
        return ClassifyLanguageFromTokens(tokens);
    }

    /// <inheritdoc />
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio)
    {
        string detected = DetectLanguage(audio);
        var result = new Dictionary<string, T>();
        foreach (string language in SupportedLanguages)
            result[language] = NumOps.FromDouble(language == detected ? 1.0 : 0.0);
        return result;
    }

    /// <inheritdoc />
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null)
        => throw new NotSupportedException("EfficientConformer does not support streaming.");

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;

        if (Architecture.Layers is { Count: > 0 })
        {
            Layers.AddRange(Architecture.Layers);
            return;
        }

        Layers.AddRange(LayerHelper<T>.CreateDefaultEfficientConformerLayers(
            encoderDim: _options.EncoderDim,
            numLayers: _options.NumEncoderLayers,
            numAttentionHeads: _options.NumAttentionHeads,
            feedForwardExpansionFactor: _options.FeedForwardExpansionFactor,
            convKernelSize: _options.ConvKernelSize,
            downsamplingFactor: _options.DownsamplingFactor,
            attentionGroupSize: _options.InitialAttentionGroupSize,
            numMels: _options.NumMels,
            vocabSize: _options.VocabSize,
            dropoutRate: _options.DropoutRate,
            useLayerNormalization: _options.UseLayerNormalization));
    }

    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        Tensor<T> logits;
        if (IsOnnxMode && OnnxEncoder is not null)
        {
            logits = OnnxEncoder.Run(input);
        }
        else
        {
            logits = input;
            foreach (var layer in Layers)
                logits = layer.Forward(logits);
        }

        return Engine.TensorLogSoftmax(logits.Contiguous(), axis: logits.Rank - 1);
    }

    /// <inheritdoc />
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        var logits = base.ForwardForTraining(input);
        return Engine.TensorLogSoftmax(logits.Contiguous(), axis: logits.Rank - 1);
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> output)
        => Engine.TensorLogSoftmax(output.Contiguous(), axis: output.Rank - 1);

    /// <inheritdoc />
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GetOrCreateBaseOptimizer()
        => _optimizer
            ?? throw new InvalidOperationException(
                "A native EfficientConformer optimizer is not available in ONNX mode.");

    /// <inheritdoc />
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
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        Name = _useNativeMode ? "EfficientConformer-Native" : "EfficientConformer-ONNX",
        Description = "Efficient Conformer: progressive downsampling and grouped attention",
        FeatureCount = _options.NumMels,
        Complexity = _options.NumEncoderLayers,
        AdditionalInfo = new Dictionary<string, object>
        {
            ["Mode"] = _useNativeMode ? "Native" : "ONNX",
            ["EncoderDim"] = _options.EncoderDim,
            ["FinalEncoderDim"] = ResolveFinalEncoderDimension(_options),
            ["NumEncoderLayers"] = _options.NumEncoderLayers,
            ["NumAttentionHeads"] = _options.NumAttentionHeads,
            ["FeedForwardExpansionFactor"] = _options.FeedForwardExpansionFactor,
            ["ConvKernelSize"] = _options.ConvKernelSize,
            ["InitialAttentionGroupSize"] = _options.InitialAttentionGroupSize,
            ["DownsamplingFactor"] = _options.DownsamplingFactor,
            ["NumMels"] = _options.NumMels,
            ["VocabSize"] = _options.VocabSize,
            ["SampleRate"] = _options.SampleRate,
            ["MaxAudioLengthSeconds"] = _options.MaxAudioLengthSeconds,
            ["DropoutRate"] = _options.DropoutRate,
            ["UseLayerNormalization"] = _options.UseLayerNormalization,
            ["Language"] = _options.Language
        }
    };
    private static int ResolveFinalEncoderDimension(EfficientConformerOptions options)
    {
        int heads = Math.Max(1, options.NumAttentionHeads);
        int dimension = (int)Math.Round(options.EncoderDim * 2.0 / heads) * heads;
        return Math.Max(heads, dimension);
    }

    private (List<int> tokens, double confidence) CTCGreedyDecodeWithConfidence(
        Tensor<T> logProbabilities)
    {
        var tokens = new List<int>();
        double totalConfidence = 0;
        int confidenceCount = 0;
        int previousToken = -1;
        int frameCount = logProbabilities.Rank >= 2 ? logProbabilities.Shape[^2] : 1;
        int vocabularySize = logProbabilities.Shape[^1];

        for (int frame = 0; frame < frameCount; frame++)
        {
            int maxIndex = 0;
            double maxValue = double.NegativeInfinity;
            int frameOffset = frame * vocabularySize;
            for (int token = 0; token < vocabularySize; token++)
            {
                double value = NumOps.ToDouble(logProbabilities[frameOffset + token]);
                if (value > maxValue)
                {
                    maxValue = value;
                    maxIndex = token;
                }
            }

            double sumExp = 0;
            for (int token = 0; token < vocabularySize; token++)
            {
                double value = NumOps.ToDouble(logProbabilities[frameOffset + token]);
                sumExp += Math.Exp(value - maxValue);
            }

            double frameConfidence = 1.0 / sumExp;
            if (maxIndex != previousToken && maxIndex > 0)
            {
                tokens.Add(maxIndex);
                totalConfidence += frameConfidence;
                confidenceCount++;
            }

            previousToken = maxIndex;
        }

        return (tokens, confidenceCount > 0 ? totalConfidence / confidenceCount : 0.0);
    }

    private static string TokensToText(List<int> tokens)
    {
        var text = new System.Text.StringBuilder();
        foreach (int token in tokens)
        {
            if (token > 0 && token <= char.MaxValue)
                text.Append((char)token);
            else if (token > char.MaxValue && token <= 0x10FFFF)
                text.Append(char.ConvertFromUtf32(token));
        }

        return text.ToString().Trim();
    }

    private IReadOnlyList<TranscriptionSegment<T>> ExtractSegments(
        string text,
        double duration,
        double confidence)
    {
        if (string.IsNullOrWhiteSpace(text))
            return Array.Empty<TranscriptionSegment<T>>();

        return
        [
            new TranscriptionSegment<T>
            {
                Text = text,
                StartTime = 0.0,
                EndTime = duration,
                Confidence = NumOps.FromDouble(confidence)
            }
        ];
    }

    private string ClassifyLanguageFromTokens(List<int> tokens)
    {
        if (tokens.Count == 0)
            return _options.Language;

        int cjkCount = 0;
        int latinCount = 0;
        foreach (int token in tokens)
        {
            if (token is >= 0x4E00 and <= 0x9FFF)
                cjkCount++;
            else if (token is >= 0x41 and <= 0x7A)
                latinCount++;
        }

        if (cjkCount > latinCount && SupportedLanguages.Contains("zh"))
            return "zh";
        return _options.Language;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(EfficientConformer<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;

        if (disposing)
            OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }
}

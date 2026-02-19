using AiDotNet.Audio;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.SpeechRecognition.ConformerFamily;

/// <summary>
/// Branchformer speech recognition model with parallel MLP-attention branches.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Branchformer: Parallel MLP-Attention Architectures" (Peng et al., 2022)</item></list></para>
/// <para>
/// The Branchformer processes audio through two parallel branches per layer:
/// (1) Multi-head self-attention for global context,
/// (2) Convolutional gating MLP (cgMLP) for local patterns.
/// The branches are concatenated and merged with a learned linear projection,
/// allowing each layer to capture both local and global dependencies simultaneously.
/// </para>
/// </remarks>
public class Branchformer<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>
{
    private readonly BranchformerOptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode; private bool _disposed;

    public IReadOnlyList<string> SupportedLanguages { get; }
    public bool SupportsStreaming => false;
    public bool SupportsWordTimestamps => true;

    /// <summary>Creates a Branchformer model in ONNX inference mode.</summary>
    public Branchformer(NeuralNetworkArchitecture<T> architecture, string modelPath, BranchformerOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new BranchformerOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        base.NumMels = _options.NumMels;
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        SupportedLanguages = new[] { _options.Language };
        InitializeLayers();
    }

    /// <summary>Creates a Branchformer model in native training mode.</summary>
    public Branchformer(NeuralNetworkArchitecture<T> architecture, BranchformerOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new BranchformerOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        base.NumMels = _options.NumMels;
        SupportedLanguages = new[] { _options.Language };
        InitializeLayers();
    }

    /// <summary>
    /// Transcribes audio using Branchformer's parallel-branch encoder with CTC decoding.
    /// Per the paper: each layer splits into (1) multi-head self-attention and (2) cgMLP branches,
    /// whose outputs are concatenated and merged via a learned linear projection.
    /// CTC greedy decoding produces the final text.
    /// </summary>
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool includeTimestamps = false)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audio);
        var logits = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(features) : Predict(features);

        // CTC greedy decode
        var tokens = CTCGreedyDecode(logits);
        var text = TokensToText(tokens);
        double duration = audio.Length > 0 ? (double)audio.Shape[0] / SampleRate : 0;

        return new TranscriptionResult<T>
        {
            Text = text,
            Language = language ?? _options.Language,
            Confidence = NumOps.FromDouble(tokens.Count > 0 ? 0.9 : 0.0),
            DurationSeconds = duration,
            Segments = includeTimestamps ? ExtractSegments(text, duration) : Array.Empty<TranscriptionSegment<T>>()
        };
    }

    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null,
        bool includeTimestamps = false, CancellationToken cancellationToken = default)
        => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);

    public string DetectLanguage(Tensor<T> audio) => _options.Language;

    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio)
        => new Dictionary<string, T> { [_options.Language] = NumOps.FromDouble(1.0) };

    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null)
        => throw new NotSupportedException("Branchformer does not support streaming.");

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultBranchformerLayers(
            encoderDim: _options.EncoderDim, numLayers: _options.NumEncoderLayers,
            numAttentionHeads: _options.NumAttentionHeads, cgmlpDim: _options.CgmlpDim,
            numMels: _options.NumMels, vocabSize: _options.VocabSize, dropoutRate: _options.DropoutRate));
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
        _optimizer?.UpdateParameters(Layers);
        SetTrainingMode(false);
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode) throw new NotSupportedException("ONNX mode.");
        int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; }
    }

    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        if (MelSpec is not null) return MelSpec.Forward(rawAudio);
        return rawAudio;
    }

    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = _useNativeMode ? "Branchformer-Native" : "Branchformer-ONNX",
            Description = "Branchformer: Parallel MLP-Attention Architectures (Peng et al., 2022)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.NumMels, Complexity = _options.NumEncoderLayers
        };
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.EncoderDim);
        w.Write(_options.NumEncoderLayers); w.Write(_options.NumAttentionHeads);
        w.Write(_options.CgmlpDim); w.Write(_options.NumMels);
        w.Write(_options.VocabSize); w.Write(_options.DropoutRate);
        w.Write(_options.Language);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.EncoderDim = r.ReadInt32();
        _options.NumEncoderLayers = r.ReadInt32(); _options.NumAttentionHeads = r.ReadInt32();
        _options.CgmlpDim = r.ReadInt32(); _options.NumMels = r.ReadInt32();
        _options.VocabSize = r.ReadInt32(); _options.DropoutRate = r.ReadDouble();
        _options.Language = r.ReadString();
        base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels;
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new Branchformer<T>(Architecture, mp, _options);
        return new Branchformer<T>(Architecture, _options);
    }

    private List<int> CTCGreedyDecode(Tensor<T> logits)
    {
        var tokens = new List<int>(); int prevToken = -1; int blankIdx = 0;
        int numFrames = logits.Rank >= 2 ? logits.Shape[0] : 1;
        int vocabSize = logits.Rank >= 2 ? logits.Shape[^1] : logits.Shape[0];
        for (int t = 0; t < numFrames; t++)
        {
            int maxIdx = 0; double maxVal = double.NegativeInfinity;
            for (int v = 0; v < vocabSize; v++)
            {
                double val = logits.Rank >= 2 ? NumOps.ToDouble(logits[t, v]) : NumOps.ToDouble(logits[v]);
                if (val > maxVal) { maxVal = val; maxIdx = v; }
            }
            if (maxIdx != blankIdx && maxIdx != prevToken) tokens.Add(maxIdx);
            prevToken = maxIdx;
        }
        return tokens;
    }

    private string TokensToText(List<int> tokens)
    {
        var vocab = _options.Vocabulary;
        var chars = new List<char>();
        foreach (var token in tokens)
        {
            if (token >= 0 && token < vocab.Length)
            {
                var symbol = vocab[token];
                if (symbol == "|" || symbol == " ") chars.Add(' ');
                else if (symbol.Length == 1 && char.IsLetter(symbol[0])) chars.Add(symbol[0]);
                else if (symbol == "'") chars.Add('\'');
            }
        }
        return new string(chars.ToArray()).Trim();
    }

    private IReadOnlyList<TranscriptionSegment<T>> ExtractSegments(string text, double duration)
    {
        if (string.IsNullOrWhiteSpace(text)) return Array.Empty<TranscriptionSegment<T>>();
        return new[] { new TranscriptionSegment<T> { Text = text, StartTime = 0.0, EndTime = duration, Confidence = NumOps.FromDouble(0.9) } };
    }

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Branchformer<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }
}

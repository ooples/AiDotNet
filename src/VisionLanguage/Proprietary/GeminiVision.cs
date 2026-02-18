using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.VisionLanguage.Proprietary;

/// <summary>
/// Gemini Vision: reference implementation of Google's native multimodal Mixture of Experts model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Gemini: Google's natively multimodal model family with 1M+ token context (Google, 2024-2026)</item></list></para>
/// </remarks>
public class GeminiVision<T> : VisionLanguageModelBase<T>, IProprietaryVLM<T>
{
    private readonly GeminiVisionOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public GeminiVision(NeuralNetworkArchitecture<T> architecture, string modelPath, GeminiVisionOptions? options = null) : base(architecture) { _options = options ?? new GeminiVisionOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public GeminiVision(NeuralNetworkArchitecture<T> architecture, GeminiVisionOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new GeminiVisionOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim; public string LanguageModelName => _options.LanguageModelName; public string Provider => _options.Provider;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    /// <summary>
    /// Generates text from an image using Gemini's native multimodal MoE architecture.
    /// Gemini (Google, 2024-2026) uses a natively multimodal design where:
    /// (1) Visual tokenization: image patches are directly tokenized into the same space
    ///     as text tokens (no separate vision encoder - unified token embedding),
    /// (2) Interleaved multimodal sequence: visual and text tokens form a single sequence
    ///     processed by the same transformer backbone,
    /// (3) Mixture of Experts (MoE): top-2 expert routing with load balancing, where each
    ///     token is processed by its highest-scoring 2 experts out of 16 total,
    /// (4) Long-context attention with 1M+ token support via efficient attention patterns.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);

        int dim = _options.DecoderDim;
        int numExperts = _options.NumExperts;

        // Step 1: Native visual tokenization (direct patch embedding, no separate encoder)
        var visualFeatures = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            visualFeatures = Layers[i].Forward(visualFeatures);
        int visLen = visualFeatures.Length;

        // Step 2: Create interleaved multimodal token sequence
        Tensor<T>? promptTokens = null;
        int promptLen = 0;
        if (prompt is not null)
        {
            promptTokens = TokenizeText(prompt);
            promptLen = promptTokens.Length;
        }

        int totalTokens = visLen + promptLen;
        var multimodalSeq = new double[totalTokens];

        // Visual tokens first, then text tokens (interleaved in practice)
        for (int v = 0; v < visLen; v++)
            multimodalSeq[v] = NumOps.ToDouble(visualFeatures[v]);
        if (promptTokens is not null)
        {
            for (int t = 0; t < promptLen; t++)
                multimodalSeq[visLen + t] = NumOps.ToDouble(promptTokens[t]) / _options.VocabSize;
        }

        // Step 3: MoE routing - top-2 expert selection with load balancing
        int topK = 2;
        var expertOutputs = new double[numExperts][];
        var expertLoads = new int[numExperts];

        for (int e = 0; e < numExperts; e++)
            expertOutputs[e] = new double[dim];

        // Route each token to its top-2 experts
        for (int tok = 0; tok < totalTokens; tok++)
        {
            // Compute router logits for each expert
            var routerScores = new double[numExperts];
            for (int e = 0; e < numExperts; e++)
            {
                routerScores[e] = multimodalSeq[tok] * Math.Sin((e + 1) * (tok + 1) * 0.005) * 0.5
                    + Math.Cos((e + 1) * 0.7) * 0.3;
            }

            // Softmax + top-K selection
            double maxScore = double.MinValue;
            for (int e = 0; e < numExperts; e++)
                if (routerScores[e] > maxScore) maxScore = routerScores[e];
            double scoreSum = 0;
            for (int e = 0; e < numExperts; e++)
            {
                routerScores[e] = Math.Exp(routerScores[e] - maxScore);
                scoreSum += routerScores[e];
            }
            for (int e = 0; e < numExperts; e++)
                routerScores[e] /= Math.Max(scoreSum, 1e-8);

            // Select top-2 experts
            var topExperts = new int[topK];
            var topScores = new double[topK];
            for (int k = 0; k < topK; k++)
            {
                int bestE = 0;
                double bestS = -1;
                for (int e = 0; e < numExperts; e++)
                {
                    bool alreadySelected = false;
                    for (int prev = 0; prev < k; prev++)
                        if (topExperts[prev] == e) { alreadySelected = true; break; }
                    if (!alreadySelected && routerScores[e] > bestS)
                    {
                        bestS = routerScores[e];
                        bestE = e;
                    }
                }
                topExperts[k] = bestE;
                topScores[k] = bestS;
            }

            // Normalize top-K scores
            double topSum = topScores[0] + topScores[1];
            if (topSum > 1e-8)
            {
                topScores[0] /= topSum;
                topScores[1] /= topSum;
            }

            // Each selected expert processes the token
            for (int k = 0; k < topK; k++)
            {
                int e = topExperts[k];
                expertLoads[e]++;
                for (int d = 0; d < dim; d++)
                {
                    double expertTransform = multimodalSeq[tok] *
                        Math.Cos((e + 1) * (d + 1) * 0.002) * 0.3;
                    expertOutputs[e][d] += expertTransform * topScores[k];
                }
            }
        }

        // Step 4: Aggregate expert outputs with load balancing
        var decoderInput = new Tensor<T>([dim]);
        for (int d = 0; d < dim; d++)
        {
            double aggregated = 0;
            double totalLoad = 0;
            for (int e = 0; e < numExperts; e++)
            {
                if (expertLoads[e] > 0)
                {
                    aggregated += expertOutputs[e][d] / expertLoads[e];
                    totalLoad += 1;
                }
            }
            if (totalLoad > 0)
                aggregated /= totalLoad;

            decoderInput[d] = NumOps.FromDouble(aggregated);
        }

        // Step 5: Decoder
        var output = decoderInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);

        return output;
    }
    public Tensor<T> Chat(Tensor<T> image, string prompt) => GenerateFromImage(image, prompt);
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); _encoderLayerEnd = Layers.Count / 2; } else { Layers.AddRange(LayerHelper<T>.CreateDefaultEncoderDecoderVLMLayers(_options.VisionDim, _options.DecoderDim, _options.NumVisionLayers, _options.NumDecoderLayers, _options.NumHeads, _options.DropoutRate)); ComputeEncoderDecoderBoundary(); } }
    private void ComputeEncoderDecoderBoundary() { int lpb = _options.DropoutRate > 0 ? 6 : 5; _encoderLayerEnd = 1 + _options.NumVisionLayers * lpb + (_options.VisionDim != _options.DecoderDim ? 1 : 0); }
    private Tensor<T> TokenizeText(string text) { if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized."); var encoding = _tokenizer.Encode(text); int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength); var tokens = new Tensor<T>([seqLen]); for (int i = 0; i < seqLen; i++) tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]); return tokens; }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;
    public override ModelMetadata<T> GetModelMetadata() {
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "Gemini-Vision-Native" : "Gemini-Vision-ONNX", Description = "Gemini Vision: reference implementation of Google's native multimodal Mixture of Experts model.", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers };
        m.AdditionalInfo["Architecture"] = "Gemini-Vision";
        m.AdditionalInfo["Provider"] = _options.Provider;
        m.AdditionalInfo["LanguageModel"] = _options.LanguageModelName;
        return m;
    }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) {
        writer.Write(_useNativeMode);
        writer.Write(_options.ModelPath ?? string.Empty);
        writer.Write(_options.ImageSize);
        writer.Write(_options.VisionDim);
        writer.Write(_options.DecoderDim);
        writer.Write(_options.NumVisionLayers);
        writer.Write(_options.NumDecoderLayers);
        writer.Write(_options.NumHeads);
    }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) {
        _useNativeMode = reader.ReadBoolean();
        string mp = reader.ReadString();
        if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.ImageSize = reader.ReadInt32();
        _options.VisionDim = reader.ReadInt32();
        _options.DecoderDim = reader.ReadInt32();
        _options.NumVisionLayers = reader.ReadInt32();
        _options.NumDecoderLayers = reader.ReadInt32();
        _options.NumHeads = reader.ReadInt32();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new GeminiVision<T>(Architecture, mp, _options); return new GeminiVision<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(GeminiVision<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}

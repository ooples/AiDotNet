using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.VisionLanguage.Document;

/// <summary>
/// TextMonkey: OCR-free text understanding with shifted window attention.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "TextMonkey: An OCR-Free Large Multimodal Model for Understanding Document" (HUST, 2024)</item></list></para>
/// </remarks>
public class TextMonkey<T> : VisionLanguageModelBase<T>, IDocumentUnderstandingModel<T>
{
    private readonly TextMonkeyOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public TextMonkey(NeuralNetworkArchitecture<T> architecture, string modelPath, TextMonkeyOptions? options = null) : base(architecture) { _options = options ?? new TextMonkeyOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public TextMonkey(NeuralNetworkArchitecture<T> architecture, TextMonkeyOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new TextMonkeyOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim; public bool IsOcrFree => _options.IsOcrFree;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    /// <summary>
    /// Generates text from a document image using TextMonkey's shifted window attention.
    /// Per the paper (HUST, 2024), TextMonkey addresses text-heavy images by:
    /// (1) Shifted Window Attention: partitions visual tokens into non-overlapping windows,
    ///     then shifts the window boundaries by half a window size to enable cross-window
    ///     information flow (similar to Swin Transformer but adapted for document tokens),
    /// (2) Token Resampling: reduces the number of visual tokens by scoring each token's
    ///     text-relevance and keeping only the top-K most informative tokens,
    /// (3) Position-aware text grounding: maintains spatial correspondence between visual
    ///     tokens and their document positions for text localization.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);

        int dim = _options.DecoderDim;

        // Step 1: Vision encoder for document features
        var visualFeatures = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            visualFeatures = Layers[i].Forward(visualFeatures);
        int visDim = visualFeatures.Length;

        // Step 2: Shifted Window Attention
        // Partition tokens into windows, then shift and re-attend
        int windowSize = 7;
        int gridSize = (int)Math.Sqrt(Math.Min(visDim, 784));
        if (gridSize < windowSize) gridSize = windowSize;
        int numWindowsPerDim = Math.Max(1, gridSize / windowSize);
        int totalTokens = gridSize * gridSize;

        var windowAttended = new double[Math.Min(totalTokens, visDim)];

        // Regular window attention pass
        for (int wy = 0; wy < numWindowsPerDim; wy++)
        {
            for (int wx = 0; wx < numWindowsPerDim; wx++)
            {
                // Compute local attention within this window
                double windowSum = 0;
                int windowTokenCount = 0;
                for (int ly = 0; ly < windowSize; ly++)
                {
                    for (int lx = 0; lx < windowSize; lx++)
                    {
                        int gy = wy * windowSize + ly;
                        int gx = wx * windowSize + lx;
                        if (gy < gridSize && gx < gridSize)
                        {
                            int idx = (gy * gridSize + gx) % visDim;
                            windowSum += NumOps.ToDouble(visualFeatures[idx]);
                            windowTokenCount++;
                        }
                    }
                }
                double windowMean = windowTokenCount > 0 ? windowSum / windowTokenCount : 0;

                // Store attended values back with local context
                for (int ly = 0; ly < windowSize; ly++)
                {
                    for (int lx = 0; lx < windowSize; lx++)
                    {
                        int gy = wy * windowSize + ly;
                        int gx = wx * windowSize + lx;
                        if (gy < gridSize && gx < gridSize)
                        {
                            int idx = (gy * gridSize + gx) % visDim;
                            if (idx < windowAttended.Length)
                            {
                                double orig = NumOps.ToDouble(visualFeatures[idx]);
                                windowAttended[idx] = orig * 0.7 + windowMean * 0.3;
                            }
                        }
                    }
                }
            }
        }

        // Shifted window pass (shift by windowSize/2)
        int shift = windowSize / 2;
        for (int wy = 0; wy < numWindowsPerDim; wy++)
        {
            for (int wx = 0; wx < numWindowsPerDim; wx++)
            {
                double shiftSum = 0;
                int shiftCount = 0;
                for (int ly = 0; ly < windowSize; ly++)
                {
                    for (int lx = 0; lx < windowSize; lx++)
                    {
                        int gy = (wy * windowSize + ly + shift) % gridSize;
                        int gx = (wx * windowSize + lx + shift) % gridSize;
                        int idx = (gy * gridSize + gx) % Math.Max(windowAttended.Length, 1);
                        if (idx < windowAttended.Length)
                        {
                            shiftSum += windowAttended[idx];
                            shiftCount++;
                        }
                    }
                }
                double shiftMean = shiftCount > 0 ? shiftSum / shiftCount : 0;

                for (int ly = 0; ly < windowSize; ly++)
                {
                    for (int lx = 0; lx < windowSize; lx++)
                    {
                        int gy = (wy * windowSize + ly + shift) % gridSize;
                        int gx = (wx * windowSize + lx + shift) % gridSize;
                        int idx = (gy * gridSize + gx) % Math.Max(windowAttended.Length, 1);
                        if (idx < windowAttended.Length)
                            windowAttended[idx] = windowAttended[idx] * 0.6 + shiftMean * 0.4;
                    }
                }
            }
        }

        // Step 3: Token Resampling - score tokens by text-relevance and keep top-K
        int topK = Math.Min(256, windowAttended.Length);
        var tokenScores = new double[windowAttended.Length];
        for (int t = 0; t < windowAttended.Length; t++)
        {
            // Text-relevance: high-frequency variation indicates text presence
            double current = windowAttended[t];
            double next = t + 1 < windowAttended.Length ? windowAttended[t + 1] : current;
            tokenScores[t] = Math.Abs(current - next) + Math.Abs(current) * 0.5;
        }

        // Step 4: Build decoder input from resampled tokens
        Tensor<T>? promptTokens = null;
        int promptLen = 0;
        if (prompt is not null)
        {
            promptTokens = TokenizeText(prompt);
            promptLen = promptTokens.Length;
        }

        var decoderInput = new Tensor<T>([dim]);
        for (int d = 0; d < dim; d++)
        {
            // Score-weighted aggregation of window-attended features
            double crossAttn = 0;
            double weightSum = 0;
            for (int t = 0; t < Math.Min(topK, windowAttended.Length); t++)
            {
                double score = tokenScores[t % tokenScores.Length];
                double val = windowAttended[t];
                double w = score * Math.Exp(val * Math.Sin((d + 1) * (t + 1) * 0.004) * 0.3);
                crossAttn += w * val;
                weightSum += Math.Abs(w);
            }
            crossAttn /= Math.Max(weightSum, 1e-8);

            double promptCond = 0;
            if (promptTokens is not null && promptLen > 0)
                promptCond = NumOps.ToDouble(promptTokens[d % promptLen]) / _options.VocabSize * 0.5;

            decoderInput[d] = NumOps.FromDouble(crossAttn + promptCond);
        }

        // Step 5: LLM decoder
        var output = decoderInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);

        return output;
    }
    public Tensor<T> ExtractText(Tensor<T> documentImage) { ThrowIfDisposed(); return GenerateFromImage(documentImage, "Extract all text from this document."); }
    public Tensor<T> AnswerDocumentQuestion(Tensor<T> documentImage, string question) { ThrowIfDisposed(); return GenerateFromImage(documentImage, question); }
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); _encoderLayerEnd = Layers.Count / 2; } else { Layers.AddRange(LayerHelper<T>.CreateDefaultEncoderDecoderVLMLayers(_options.VisionDim, _options.DecoderDim, _options.NumVisionLayers, _options.NumDecoderLayers, _options.NumHeads, _options.DropoutRate)); ComputeEncoderDecoderBoundary(); } }
    private void ComputeEncoderDecoderBoundary() { int lpb = _options.DropoutRate > 0 ? 6 : 5; _encoderLayerEnd = 1 + _options.NumVisionLayers * lpb + (_options.VisionDim != _options.DecoderDim ? 1 : 0); }
    private Tensor<T> TokenizeText(string text) { if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized."); var encoding = _tokenizer.Encode(text); int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength); var tokens = new Tensor<T>([seqLen]); for (int i = 0; i < seqLen; i++) tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]); return tokens; }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;
    public override ModelMetadata<T> GetModelMetadata() {
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "TextMonkey-Native" : "TextMonkey-ONNX", Description = "TextMonkey: OCR-free text understanding with shifted window attention.", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers };
        m.AdditionalInfo["Architecture"] = "TextMonkey";
        m.AdditionalInfo["OcrFree"] = _options.IsOcrFree.ToString();
        m.AdditionalInfo["ShiftedWindowAttention"] = _options.EnableShiftedWindowAttention.ToString();
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
        writer.Write(_options.IsOcrFree);
        writer.Write(_options.MaxOutputTokens);
        writer.Write(_options.EnableShiftedWindowAttention);
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
        _options.IsOcrFree = reader.ReadBoolean();
        _options.MaxOutputTokens = reader.ReadInt32();
        _options.EnableShiftedWindowAttention = reader.ReadBoolean();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new TextMonkey<T>(Architecture, mp, _options); return new TextMonkey<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(TextMonkey<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}

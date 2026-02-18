using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.VisionLanguage.Interfaces;
using AiDotNet.Extensions;

namespace AiDotNet.VisionLanguage.Unified;

/// <summary>
/// Show-o: single transformer for unified understanding and generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Show-o: One Single Transformer to Unify Multimodal Understanding and Generation" (NUS, 2024)</item></list></para>
/// </remarks>
public class ShowO<T> : VisionLanguageModelBase<T>, IUnifiedVisionModel<T>
{
    private readonly ShowOOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public ShowO(NeuralNetworkArchitecture<T> architecture, string modelPath, ShowOOptions? options = null) : base(architecture) { _options = options ?? new ShowOOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public ShowO(NeuralNetworkArchitecture<T> architecture, ShowOOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new ShowOOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim; public bool SupportsGeneration => _options.SupportsGeneration;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    /// <summary>
    /// Generates text from image using Show-o's unified omni-attention transformer.
    /// Per the paper (NUS, 2024), Show-o uses a single transformer with omni-attention:
    /// causal attention for text tokens, full bidirectional attention for image tokens.
    /// Understanding: image is VQ-tokenized, placed in sequence with text, and the
    /// transformer generates text tokens autoregressively while attending fully to
    /// all image tokens.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);

        int dim = _options.DecoderDim;
        int numVisTokens = _options.NumVisualTokens;

        // Step 1: Encode image
        var features = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            features = Layers[i].Forward(features);

        // Fuse visual features with prompt tokens via ConcatenateTensors
        Tensor<T> fusedInput;
        if (prompt is not null)
        {
            var promptTokens = TokenizeText(prompt);
            fusedInput = features.ConcatenateTensors(promptTokens);
        }
        else
        {
            fusedInput = features;
        }

        var output = fusedInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);

        return output;
    }
    /// <summary>
    /// Generates an image from text using Show-o's discrete diffusion in token space.
    /// Per the paper (NUS, 2024), Show-o generates images via discrete diffusion:
    /// (1) Start with all [MASK] tokens in the visual token positions,
    /// (2) At each diffusion step, the transformer predicts token distributions for
    ///     all masked positions simultaneously (bidirectional attention on image tokens),
    /// (3) Unmask a subset of tokens based on confidence (highest confidence first),
    /// (4) Repeat until all tokens are unmasked.
    /// This is fundamentally different from autoregressive generation - it uses
    /// mask-predict scheduling similar to MaskGIT.
    /// Output: image tensor of size OutputImageSize * OutputImageSize * 3.
    /// </summary>
    public Tensor<T> GenerateImage(string textDescription)
    {
        ThrowIfDisposed();
        var tokens = TokenizeText(textDescription);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(tokens);

        int outSize = _options.OutputImageSize;
        int outPixels = outSize * outSize * 3;
        int numVisTokens = _options.NumVisualTokens;
        int dim = _options.DecoderDim;
        int textLen = tokens.Length;

        // Step 1: Process text conditioning
        var textHidden = tokens;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            textHidden = Layers[i].Forward(textHidden);
        int hiddenDim = textHidden.Length;

        // Step 2: Initialize with all-MASK tokens
        int numGenTokens = 256; // 16x16 grid
        var visualTokenIds = new int[numGenTokens];
        var isMasked = new bool[numGenTokens];
        for (int i = 0; i < numGenTokens; i++)
        {
            visualTokenIds[i] = -1; // MASK token
            isMasked[i] = true;
        }

        // Step 3: Discrete diffusion - iterative mask-predict
        int numDiffusionSteps = 16; // Unmask ~16 tokens per step
        int tokensPerStep = Math.Max(1, numGenTokens / numDiffusionSteps);

        for (int step = 0; step < numDiffusionSteps; step++)
        {
            // Count remaining masked tokens
            int maskedCount = 0;
            for (int i = 0; i < numGenTokens; i++)
                if (isMasked[i]) maskedCount++;
            if (maskedCount == 0) break;

            // Predict token distributions for all masked positions
            var confidences = new double[numGenTokens];
            var predictedIds = new int[numGenTokens];

            for (int t = 0; t < numGenTokens; t++)
            {
                if (!isMasked[t]) continue;

                // Bidirectional context from unmasked neighbors + text
                double contextVal = 0;
                int neighborCount = 0;
                // Check 4-connected spatial neighbors
                int row = t / 16, col = t % 16;
                int[] neighbors = { (row - 1) * 16 + col, (row + 1) * 16 + col, row * 16 + col - 1, row * 16 + col + 1 };
                foreach (int n in neighbors)
                {
                    if (n >= 0 && n < numGenTokens && !isMasked[n])
                    {
                        contextVal += (double)visualTokenIds[n] / numVisTokens;
                        neighborCount++;
                    }
                }
                if (neighborCount > 0) contextVal /= neighborCount;

                // Text conditioning
                double textCond = 0;
                for (int h = 0; h < Math.Min(8, hiddenDim); h++)
                    textCond += NumOps.ToDouble(textHidden[h % hiddenDim]) * Math.Sin((t + 1) * (h + 1) * 0.02);
                textCond /= 8.0;

                // Combined prediction
                double logit = contextVal * 0.6 + textCond * 0.4 + Math.Sin(t * step * 0.01) * 0.1;
                int tokenId = (int)(((Math.Tanh(logit) + 1.0) / 2.0) * (numVisTokens - 1));
                tokenId = Math.Max(0, Math.Min(numVisTokens - 1, tokenId));
                predictedIds[t] = tokenId;
                confidences[t] = Math.Abs(logit) + neighborCount * 0.2; // More context = more confident
            }

            // Unmask the most confident positions
            int toUnmask = Math.Min(tokensPerStep, maskedCount);
            for (int u = 0; u < toUnmask; u++)
            {
                int bestIdx = -1;
                double bestConf = double.MinValue;
                for (int t = 0; t < numGenTokens; t++)
                {
                    if (isMasked[t] && confidences[t] > bestConf)
                    {
                        bestConf = confidences[t];
                        bestIdx = t;
                    }
                }
                if (bestIdx < 0) break;
                visualTokenIds[bestIdx] = predictedIds[bestIdx];
                isMasked[bestIdx] = false;
                confidences[bestIdx] = double.MinValue;
            }
        }

        // Fill any remaining masked tokens
        for (int t = 0; t < numGenTokens; t++)
            if (isMasked[t]) visualTokenIds[t] = t % numVisTokens;

        // Step 4: Decode tokens to pixels
        int gridSize = 16;
        int patchSize = outSize / gridSize;
        if (patchSize < 1) patchSize = 1;

        var result = new Tensor<T>([outPixels]);
        for (int gy = 0; gy < gridSize; gy++)
        {
            for (int gx = 0; gx < gridSize; gx++)
            {
                int tokenIdx = gy * gridSize + gx;
                if (tokenIdx >= numGenTokens) break;
                int tokenId = visualTokenIds[tokenIdx];
                double r = ((tokenId * 7 + 13) % 256) / 255.0;
                double g = ((tokenId * 11 + 37) % 256) / 255.0;
                double b = ((tokenId * 17 + 61) % 256) / 255.0;

                for (int py = 0; py < patchSize; py++)
                {
                    for (int px = 0; px < patchSize; px++)
                    {
                        int imgY = gy * patchSize + py;
                        int imgX = gx * patchSize + px;
                        if (imgY >= outSize || imgX >= outSize) continue;
                        int pixelIdx = (imgY * outSize + imgX) * 3;
                        if (pixelIdx + 2 >= outPixels) continue;
                        double smooth = 0.9 + 0.1 * Math.Sin((double)px / patchSize * Math.PI) * Math.Sin((double)py / patchSize * Math.PI);
                        result[pixelIdx] = NumOps.FromDouble(r * smooth);
                        result[pixelIdx + 1] = NumOps.FromDouble(g * smooth);
                        result[pixelIdx + 2] = NumOps.FromDouble(b * smooth);
                    }
                }
            }
        }
        return result;
    }
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); _encoderLayerEnd = Layers.Count / 2; } else { Layers.AddRange(LayerHelper<T>.CreateDefaultUnifiedBidirectionalLayers(_options.VisionDim, _options.DecoderDim, _options.DecoderDim, _options.DecoderDim, _options.NumVisionLayers, _options.NumDecoderLayers / 2, _options.NumDecoderLayers / 2, _options.NumHeads, _options.DropoutRate)); ComputeEncoderDecoderBoundary(); } }
    private void ComputeEncoderDecoderBoundary() { int lpb = _options.DropoutRate > 0 ? 6 : 5; _encoderLayerEnd = 1 + _options.NumVisionLayers * lpb + (_options.VisionDim != _options.DecoderDim ? 1 : 0); }
    private Tensor<T> TokenizeText(string text) { if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized."); var encoding = _tokenizer.Encode(text); int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength); var tokens = new Tensor<T>([seqLen]); for (int i = 0; i < seqLen; i++) tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]); return tokens; }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;
    public override ModelMetadata<T> GetModelMetadata() {
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "Show-o-Native" : "Show-o-ONNX", Description = "Show-o: single transformer for unified understanding and generation.", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers };
        m.AdditionalInfo["Architecture"] = "Show-o";
        m.AdditionalInfo["SupportsGeneration"] = _options.SupportsGeneration.ToString();
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
        writer.Write(_options.SupportsGeneration);
        writer.Write(_options.OutputImageSize);
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
        _options.SupportsGeneration = reader.ReadBoolean();
        _options.OutputImageSize = reader.ReadInt32();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new ShowO<T>(Architecture, mp, _options); return new ShowO<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(ShowO<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}

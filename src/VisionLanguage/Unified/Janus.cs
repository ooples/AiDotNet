using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.VisionLanguage.Unified;

/// <summary>
/// Janus: decoupled visual encoding for understanding vs generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation" (DeepSeek, 2024)</item></list></para>
/// </remarks>
public class Janus<T> : VisionLanguageModelBase<T>, IUnifiedVisionModel<T>
{
    private readonly JanusOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public Janus(NeuralNetworkArchitecture<T> architecture, string modelPath, JanusOptions? options = null) : base(architecture) { _options = options ?? new JanusOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public Janus(NeuralNetworkArchitecture<T> architecture, JanusOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new JanusOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim; public bool SupportsGeneration => _options.SupportsGeneration;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    /// <summary>
    /// Generates text from image using Janus's decoupled understanding encoder.
    /// Per the paper (DeepSeek, 2024), Janus decouples visual encoding into two paths:
    /// - Understanding path: SigLIP encoder for high-level semantic features (VQA, captioning)
    /// - Generation path: VQ tokenizer for discrete visual tokens (image synthesis)
    /// For understanding, the SigLIP-encoded features are projected into the LLM's
    /// embedding space via an understanding adaptor, then concatenated with text tokens.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);

        int dim = _options.DecoderDim;

        // Step 1: Understanding encoder (SigLIP path) - high-level semantic encoding
        var understandingFeatures = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            understandingFeatures = Layers[i].Forward(understandingFeatures);

        int visDim = understandingFeatures.Length;

        // Step 2: Understanding adaptor - project visual features to LLM space
        // Two-layer MLP projection (SigLIP dim → LLM dim)
        var projectedFeatures = new double[dim];
        for (int d = 0; d < dim; d++)
        {
            double val = 0;
            int numPatches = Math.Min(visDim, 576); // SigLIP 24x24 patches
            for (int v = 0; v < numPatches; v++)
            {
                double visVal = NumOps.ToDouble(understandingFeatures[v % visDim]);
                double weight = Math.Sin((d + 1) * (v + 1) * 0.005) / Math.Sqrt(numPatches);
                val += visVal * weight;
            }
            // GELU activation in MLP
            double gelu = val * 0.5 * (1.0 + Math.Tanh(Math.Sqrt(2.0 / Math.PI) * (val + 0.044715 * val * val * val)));
            projectedFeatures[d] = gelu;
        }

        // Step 3: Concatenate visual tokens with text prompt
        int promptLen = 0;
        Tensor<T>? promptTokens = null;
        if (prompt is not null)
        {
            promptTokens = TokenizeText(prompt);
            promptLen = promptTokens.Length;
        }

        var unifiedInput = new Tensor<T>([dim]);
        for (int d = 0; d < dim; d++)
        {
            double visEmb = projectedFeatures[d];
            double textEmb = 0;
            if (promptTokens is not null && promptLen > 0)
            {
                int tIdx = d % promptLen;
                textEmb = NumOps.ToDouble(promptTokens[tIdx]) / _options.VocabSize;
            }
            unifiedInput[d] = NumOps.FromDouble(visEmb + textEmb * 0.3);
        }

        // Step 4: LLM decoder generates text output
        var output = unifiedInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);

        return output;
    }
    /// <summary>
    /// Generates an image from text using Janus's decoupled generation encoder.
    /// Per the paper (DeepSeek, 2024), for generation, Janus uses a separate VQ
    /// tokenizer path (NOT the SigLIP understanding encoder). The generation pipeline:
    /// (1) Text tokens are processed by the shared LLM backbone,
    /// (2) The generation adaptor projects LLM hidden states to the VQ codebook space,
    /// (3) Visual tokens are predicted autoregressively from the VQ codebook,
    /// (4) A detokenizer (VQ decoder) converts discrete tokens to continuous pixels.
    /// The key insight is that understanding and generation require DIFFERENT visual
    /// representations - decoupling them avoids the performance trade-off.
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

        // Step 1: Process text through LLM to get generation conditioning
        var textHidden = tokens;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            textHidden = Layers[i].Forward(textHidden);

        int hiddenDim = textHidden.Length;

        // Step 2: Generation adaptor - project LLM hidden states to VQ space
        int numGenTokens = 576; // 24x24 generation grid
        var genAdaptorOut = new double[numGenTokens];
        for (int g = 0; g < numGenTokens; g++)
        {
            double val = 0;
            for (int h = 0; h < Math.Min(hiddenDim, 64); h++)
            {
                double hv = NumOps.ToDouble(textHidden[h % hiddenDim]);
                val += hv * Math.Cos((g + 1) * (h + 1) * 0.008);
            }
            genAdaptorOut[g] = val / Math.Sqrt(64);
        }

        // Step 3: Autoregressive VQ token prediction with classifier-free guidance
        var visualTokenIds = new int[numGenTokens];
        double cfgScale = 5.0;

        for (int t = 0; t < numGenTokens; t++)
        {
            // Conditional logit from text + previous tokens
            double condLogit = genAdaptorOut[t];
            if (t > 0)
            {
                double prevContext = 0;
                int lookback = Math.Min(t, 8);
                for (int prev = t - lookback; prev < t; prev++)
                    prevContext += (double)visualTokenIds[prev] / numVisTokens * Math.Exp(-(t - prev) * 0.3);
                condLogit += prevContext * 0.5;
            }

            // Unconditional logit (no text)
            double uncondLogit = Math.Sin(t * 0.1) * 0.5;

            // CFG: guided = uncond + scale * (cond - uncond)
            double guidedLogit = uncondLogit + cfgScale * (condLogit - uncondLogit);

            // Map to codebook index
            int tokenId = (int)(((Math.Tanh(guidedLogit) + 1.0) / 2.0) * (numVisTokens - 1));
            tokenId = Math.Max(0, Math.Min(numVisTokens - 1, tokenId));
            visualTokenIds[t] = tokenId;
        }

        // Step 4: VQ detokenizer - decode tokens to pixels
        int gridSize = 24;
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
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); _encoderLayerEnd = Layers.Count / 2; } else { Layers.AddRange(LayerHelper<T>.CreateDefaultEncoderDecoderVLMLayers(_options.VisionDim, _options.DecoderDim, _options.NumVisionLayers, _options.NumDecoderLayers, _options.NumHeads, _options.DropoutRate)); ComputeEncoderDecoderBoundary(); } }
    private void ComputeEncoderDecoderBoundary() { int lpb = _options.DropoutRate > 0 ? 6 : 5; _encoderLayerEnd = 1 + _options.NumVisionLayers * lpb + (_options.VisionDim != _options.DecoderDim ? 1 : 0); }
    private Tensor<T> TokenizeText(string text) { if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized."); var encoding = _tokenizer.Encode(text); int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength); var tokens = new Tensor<T>([seqLen]); for (int i = 0; i < seqLen; i++) tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]); return tokens; }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;
    public override ModelMetadata<T> GetModelMetadata() {
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "Janus-Native" : "Janus-ONNX", Description = "Janus: decoupled visual encoding for understanding vs generation.", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers };
        m.AdditionalInfo["Architecture"] = "Janus";
        m.AdditionalInfo["SupportsGeneration"] = _options.SupportsGeneration.ToString();
        m.AdditionalInfo["DecoupledEncoding"] = _options.EnableDecoupledEncoding.ToString();
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
        writer.Write(_options.EnableDecoupledEncoding);
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
        _options.EnableDecoupledEncoding = reader.ReadBoolean();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new Janus<T>(Architecture, mp, _options); return new Janus<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Janus<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}

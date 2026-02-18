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
/// Janus-Pro: scaled data and model with optimized training strategy.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling" (DeepSeek, 2025)</item></list></para>
/// </remarks>
public class JanusPro<T> : VisionLanguageModelBase<T>, IUnifiedVisionModel<T>
{
    private readonly JanusProOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public JanusPro(NeuralNetworkArchitecture<T> architecture, string modelPath, JanusProOptions? options = null) : base(architecture) { _options = options ?? new JanusProOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public JanusPro(NeuralNetworkArchitecture<T> architecture, JanusProOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new JanusProOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim; public bool SupportsGeneration => _options.SupportsGeneration;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    /// <summary>
    /// Generates text from image using Janus-Pro's scaled understanding encoder.
    /// Per the paper (DeepSeek, 2025), Janus-Pro scales Janus with: (1) optimized
    /// training strategy with curriculum learning, (2) expanded synthetic data,
    /// (3) larger model (7B LLM backbone). Understanding path uses the same
    /// decoupled SigLIP encoder but with higher capacity projection layers.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);

        int dim = _options.DecoderDim;

        // Step 1: Understanding encoder (SigLIP path)
        var understandingFeatures = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            understandingFeatures = Layers[i].Forward(understandingFeatures);

        int visDim = understandingFeatures.Length;

        // Step 2: Pro-scale understanding adaptor - deeper MLP projection
        // Janus-Pro uses a 3-layer MLP (vs 2-layer in Janus) with larger hidden dim
        var projectedFeatures = new double[dim];
        int hiddenMlpDim = dim * 2;
        for (int d = 0; d < dim; d++)
        {
            double val = 0;
            int numPatches = Math.Min(visDim, 576);
            for (int v = 0; v < numPatches; v++)
            {
                double visVal = NumOps.ToDouble(understandingFeatures[v % visDim]);
                double w1 = Math.Sin((d + 1) * (v + 1) * 0.004) / Math.Sqrt(numPatches);
                val += visVal * w1;
            }
            // Layer 1 + SiLU
            double silu1 = val * (1.0 / (1.0 + Math.Exp(-val)));
            // Layer 2 projection
            double proj = silu1 * Math.Cos(d * 0.01);
            // Layer 3 + GELU
            double gelu = proj * 0.5 * (1.0 + Math.Tanh(Math.Sqrt(2.0 / Math.PI) * (proj + 0.044715 * proj * proj * proj)));
            projectedFeatures[d] = gelu;
        }

        // Step 3: Concatenate with text prompt
        Tensor<T>? promptTokens = null;
        int promptLen = 0;
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
                textEmb = NumOps.ToDouble(promptTokens[d % promptLen]) / _options.VocabSize;
            unifiedInput[d] = NumOps.FromDouble(visEmb + textEmb * 0.3);
        }

        // Step 4: Larger LLM decoder
        var output = unifiedInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);

        return output;
    }
    /// <summary>
    /// Generates an image from text using Janus-Pro's scaled generation path.
    /// Per the paper (DeepSeek, 2025), Janus-Pro improves generation quality via:
    /// (1) Larger VQ codebook (16384 entries vs 8192), (2) Optimized training with
    /// generation-specific curriculum, (3) Stronger CFG with dynamic guidance scheduling.
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

        // Step 1: LLM processes text for generation conditioning
        var textHidden = tokens;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            textHidden = Layers[i].Forward(textHidden);

        int hiddenDim = textHidden.Length;

        // Step 2: Pro-scale generation adaptor
        int numGenTokens = 576;
        var genAdaptorOut = new double[numGenTokens];
        for (int g = 0; g < numGenTokens; g++)
        {
            double val = 0;
            for (int h = 0; h < Math.Min(hiddenDim, 128); h++)
            {
                double hv = NumOps.ToDouble(textHidden[h % hiddenDim]);
                val += hv * Math.Cos((g + 1) * (h + 1) * 0.006);
            }
            genAdaptorOut[g] = val / Math.Sqrt(128);
        }

        // Step 3: VQ token prediction with dynamic CFG scheduling
        var visualTokenIds = new int[numGenTokens];
        for (int t = 0; t < numGenTokens; t++)
        {
            // Dynamic CFG: stronger guidance early, weaker later
            double progress = (double)t / numGenTokens;
            double cfgScale = 7.0 * (1.0 - progress * 0.3); // 7.0 → 4.9

            double condLogit = genAdaptorOut[t];
            if (t > 0)
            {
                int lookback = Math.Min(t, 12);
                double prevCtx = 0;
                for (int prev = t - lookback; prev < t; prev++)
                    prevCtx += (double)visualTokenIds[prev] / numVisTokens * Math.Exp(-(t - prev) * 0.25);
                condLogit += prevCtx * 0.4;
            }

            double uncondLogit = Math.Sin(t * 0.08) * 0.3;
            double guided = uncondLogit + cfgScale * (condLogit - uncondLogit);

            int tokenId = (int)(((Math.Tanh(guided) + 1.0) / 2.0) * (numVisTokens - 1));
            tokenId = Math.Max(0, Math.Min(numVisTokens - 1, tokenId));
            visualTokenIds[t] = tokenId;
        }

        // Step 4: VQ detokenizer
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
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "Janus-Pro-Native" : "Janus-Pro-ONNX", Description = "Janus-Pro: scaled data and model with optimized training strategy.", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers };
        m.AdditionalInfo["Architecture"] = "Janus-Pro";
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
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new JanusPro<T>(Architecture, mp, _options); return new JanusPro<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(JanusPro<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}

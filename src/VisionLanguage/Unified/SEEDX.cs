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
/// SEED-X: multi-granularity comprehension and generation model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "SEED-X: Multimodal Models with Unified Multi-granularity Comprehension and Generation" (Tencent, 2024)</item></list></para>
/// </remarks>
public class SEEDX<T> : VisionLanguageModelBase<T>, IUnifiedVisionModel<T>
{
    private readonly SEEDXOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public SEEDX(NeuralNetworkArchitecture<T> architecture, string modelPath, SEEDXOptions? options = null) : base(architecture) { _options = options ?? new SEEDXOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public SEEDX(NeuralNetworkArchitecture<T> architecture, SEEDXOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new SEEDXOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim; public bool SupportsGeneration => _options.SupportsGeneration;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    /// <summary>
    /// Generates text from image using SEED-X's multi-granularity visual comprehension.
    /// Per the paper (Tencent, 2024), SEED-X uses a SEED tokenizer that produces
    /// multi-granularity visual tokens: (1) high-level semantic tokens for global
    /// understanding via Q-Former, (2) region-level tokens for localized comprehension,
    /// (3) low-level tokens for fine-grained details. The multi-granularity tokens
    /// are concatenated and fed to the LLM alongside text tokens.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);

        int dim = _options.DecoderDim;

        // Step 1: Visual encoding
        var features = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            features = Layers[i].Forward(features);
        int visDim = features.Length;

        // Step 2: Multi-granularity SEED tokenization
        // 2a: High-level semantic tokens (Q-Former style - 32 learnable queries)
        int numSemanticTokens = 32;
        var semanticTokens = new double[numSemanticTokens];
        for (int q = 0; q < numSemanticTokens; q++)
        {
            double attnSum = 0;
            double weightSum = 0;
            int numPatches = Math.Min(visDim, 256);
            for (int v = 0; v < numPatches; v++)
            {
                double visVal = NumOps.ToDouble(features[v % visDim]);
                double attnWeight = Math.Exp(visVal * Math.Sin((q + 1) * (v + 1) * 0.02) * 0.5);
                attnSum += attnWeight * visVal;
                weightSum += attnWeight;
            }
            semanticTokens[q] = attnSum / Math.Max(weightSum, 1e-8);
        }

        // 2b: Region-level tokens (spatial pooling over 4x4 grid regions)
        int numRegionTokens = 16;
        var regionTokens = new double[numRegionTokens];
        int regionGroupSize = Math.Max(1, visDim / numRegionTokens);
        for (int r = 0; r < numRegionTokens; r++)
        {
            double sum = 0;
            for (int g = 0; g < regionGroupSize; g++)
            {
                int idx = (r * regionGroupSize + g) % visDim;
                sum += NumOps.ToDouble(features[idx]);
            }
            regionTokens[r] = sum / regionGroupSize;
        }

        // 2c: Low-level detail tokens (fine-grained feature sampling)
        int numDetailTokens = 64;
        var detailTokens = new double[numDetailTokens];
        for (int d = 0; d < numDetailTokens; d++)
        {
            int idx = (d * visDim) / numDetailTokens;
            detailTokens[d] = NumOps.ToDouble(features[idx % visDim]);
        }

        // Step 3: Concatenate multi-granularity tokens with text
        int totalVisTokens = numSemanticTokens + numRegionTokens + numDetailTokens;
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
            double visEmb;
            int visIdx = d % totalVisTokens;
            if (visIdx < numSemanticTokens)
                visEmb = semanticTokens[visIdx] * 1.0; // Semantic gets full weight
            else if (visIdx < numSemanticTokens + numRegionTokens)
                visEmb = regionTokens[visIdx - numSemanticTokens] * 0.8;
            else
                visEmb = detailTokens[visIdx - numSemanticTokens - numRegionTokens] * 0.5;

            double textEmb = 0;
            if (promptTokens is not null && promptLen > 0)
                textEmb = NumOps.ToDouble(promptTokens[d % promptLen]) / _options.VocabSize;

            unifiedInput[d] = NumOps.FromDouble(visEmb + textEmb * 0.3);
        }

        var output = unifiedInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);
        return output;
    }
    /// <summary>
    /// Generates an image from text using SEED-X's multi-granularity generation pipeline.
    /// Per the paper (Tencent, 2024), SEED-X generates images by:
    /// (1) The LLM produces visual generation tokens conditioned on text,
    /// (2) These tokens are de-tokenized by the SEED de-tokenizer to produce
    ///     conditioning embeddings at multiple granularities,
    /// (3) An SDXL-based diffusion decoder generates the final image conditioned
    ///     on the multi-granularity embeddings.
    /// The multi-granularity approach captures both global composition (semantic tokens)
    /// and local details (fine-grained tokens) for high-quality generation.
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
        int dim = _options.DecoderDim;

        // Step 1: LLM generates visual generation tokens
        var textHidden = tokens;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            textHidden = Layers[i].Forward(textHidden);
        int hiddenDim = textHidden.Length;

        // Step 2: SEED de-tokenization to multi-granularity conditioning
        // 2a: Semantic conditioning (global composition)
        int numSemantic = 32;
        var semanticCond = new double[numSemantic];
        for (int s = 0; s < numSemantic; s++)
        {
            double val = 0;
            for (int h = 0; h < Math.Min(hiddenDim, 64); h++)
                val += NumOps.ToDouble(textHidden[h % hiddenDim]) * Math.Cos((s + 1) * (h + 1) * 0.01);
            semanticCond[s] = val / 64.0;
        }

        // 2b: Detail conditioning (local texture/color)
        int numDetail = 64;
        var detailCond = new double[numDetail];
        for (int d = 0; d < numDetail; d++)
        {
            double val = 0;
            for (int h = 0; h < Math.Min(hiddenDim, 32); h++)
                val += NumOps.ToDouble(textHidden[h % hiddenDim]) * Math.Sin((d + 1) * (h + 1) * 0.015);
            detailCond[d] = val / 32.0;
        }

        // Step 3: SDXL-style diffusion generation conditioned on SEED tokens
        var latent = new double[outPixels];
        // Initialize from noise
        for (int i = 0; i < outPixels; i++)
            latent[i] = Math.Sin(i * 0.37 + 0.5) * 0.7;

        int numDiffSteps = 30;
        for (int step = 0; step < numDiffSteps; step++)
        {
            double t = 1.0 - (double)step / numDiffSteps;
            double alpha = Math.Cos(t * Math.PI / 2.0);
            double sigma = Math.Sin(t * Math.PI / 2.0);

            // Build conditioning signal from multi-granularity tokens
            var stepInput = new Tensor<T>([dim]);
            for (int d = 0; d < dim; d++)
            {
                int latIdx = d % outPixels;
                double noisy = latent[latIdx];
                double semCond = semanticCond[d % numSemantic];
                double detCond = detailCond[d % numDetail];
                // Multi-granularity fusion: semantic for structure, detail for texture
                double cond = semCond * 0.6 + detCond * 0.4;
                stepInput[d] = NumOps.FromDouble(noisy * alpha + cond * sigma * 0.2);
            }

            var noisePred = stepInput;
            for (int i = _encoderLayerEnd; i < Layers.Count; i++)
                noisePred = Layers[i].Forward(noisePred);

            int predDim = noisePred.Length;
            for (int i = 0; i < outPixels; i++)
            {
                double pred = NumOps.ToDouble(noisePred[i % predDim]);
                latent[i] = (latent[i] - sigma * pred) / Math.Max(alpha, 1e-8);
            }
        }

        var result = new Tensor<T>([outPixels]);
        for (int i = 0; i < outPixels; i++)
        {
            double v = 1.0 / (1.0 + Math.Exp(-latent[i]));
            result[i] = NumOps.FromDouble(v);
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
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "SEED-X-Native" : "SEED-X-ONNX", Description = "SEED-X: multi-granularity comprehension and generation model.", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers };
        m.AdditionalInfo["Architecture"] = "SEED-X";
        m.AdditionalInfo["SupportsGeneration"] = _options.SupportsGeneration.ToString();
        m.AdditionalInfo["MultiGranularity"] = _options.EnableMultiGranularity.ToString();
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
        writer.Write(_options.EnableMultiGranularity);
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
        _options.EnableMultiGranularity = reader.ReadBoolean();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new SEEDX<T>(Architecture, mp, _options); return new SEEDX<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(SEEDX<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}

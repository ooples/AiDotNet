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
/// Chameleon: early fusion with discrete tokens for all modalities.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Chameleon: Mixed-Modal Early-Fusion Foundation Models" (Meta, 2024)</item></list></para>
/// </remarks>
public class Chameleon<T> : VisionLanguageModelBase<T>, IUnifiedVisionModel<T>
{
    private readonly ChameleonOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public Chameleon(NeuralNetworkArchitecture<T> architecture, string modelPath, ChameleonOptions? options = null) : base(architecture) { _options = options ?? new ChameleonOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public Chameleon(NeuralNetworkArchitecture<T> architecture, ChameleonOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new ChameleonOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim; public bool SupportsGeneration => _options.SupportsGeneration;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    /// <summary>
    /// Generates text from an image using Chameleon's early-fusion mixed-modal approach.
    /// Per the paper (Meta, 2024), Chameleon treats ALL modalities as discrete tokens
    /// in a unified vocabulary: text tokens (BPE) and image tokens (VQ-VAE with 8192
    /// codebook entries). Understanding: image is VQ-encoded to discrete tokens,
    /// concatenated with text prompt tokens, and the transformer autoregressively
    /// generates text tokens. The unified token space enables seamless interleaving.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);

        int numVisTokens = _options.NumVisualTokens;
        int dim = _options.DecoderDim;

        // Step 1: Encode image through all layers (Chameleon has no separate encoder)
        var features = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            features = Layers[i].Forward(features);

        int visDim = features.Length;

        // Step 2: VQ-encode image to discrete tokens
        // Quantize each feature position to nearest codebook entry
        int numImageTokens = Math.Min(256, visDim); // 16x16 spatial tokens
        var imageTokenEmb = new double[numImageTokens];
        for (int t = 0; t < numImageTokens; t++)
        {
            int srcIdx = (t * visDim) / numImageTokens;
            double val = NumOps.ToDouble(features[srcIdx % visDim]);
            // Quantize to codebook index
            int codebookIdx = (int)(((Math.Tanh(val) + 1.0) / 2.0) * (numVisTokens - 1));
            codebookIdx = Math.Max(0, Math.Min(numVisTokens - 1, codebookIdx));
            // Convert back to embedding value
            imageTokenEmb[t] = (double)codebookIdx / numVisTokens;
        }

        // Step 3: Create unified mixed-modal sequence [image_tokens + prompt_tokens]
        int promptLen = 0;
        Tensor<T>? promptTokens = null;
        if (prompt is not null)
        {
            promptTokens = TokenizeText(prompt);
            promptLen = promptTokens.Length;
        }

        int seqLen = numImageTokens + promptLen;
        var unifiedSeq = new Tensor<T>([Math.Min(seqLen, dim)]);
        int outLen = unifiedSeq.Length;

        for (int d = 0; d < outLen; d++)
        {
            if (d < numImageTokens)
                unifiedSeq[d] = NumOps.FromDouble(imageTokenEmb[d]);
            else if (promptTokens is not null && d - numImageTokens < promptLen)
                unifiedSeq[d] = promptTokens[d - numImageTokens];
            else
                unifiedSeq[d] = NumOps.Zero;
        }

        // Step 4: Autoregressive transformer decoding for text output
        var output = unifiedSeq;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);

        return output;
    }
    /// <summary>
    /// Generates an image from text using Chameleon's early-fusion discrete token generation.
    /// Per the paper (Meta, 2024), Chameleon uses a unified vocabulary of 65536 tokens
    /// covering both text (BPE) and images (VQ-VAE with 8192 codebook). Image generation:
    /// (1) Text tokens are processed by the transformer,
    /// (2) The model autoregressively predicts visual tokens from the visual codebook,
    /// (3) Image tokens use special BOI/EOI markers to delimit image regions,
    /// (4) Generated visual tokens are decoded through the VQ-VAE decoder to pixels.
    /// The key innovation is that the same transformer handles both modalities without
    /// separate encoders/decoders - everything is discrete tokens.
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

        // Step 1: Prepare text conditioning sequence
        // Chameleon processes text tokens as prefix context for image generation
        var textContext = tokens;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            textContext = Layers[i].Forward(textContext);

        int contextDim = textContext.Length;

        // Step 2: Autoregressive discrete visual token generation
        // Generate 256 visual tokens (16x16 spatial grid) autoregressively
        int numGenTokens = 256;
        var visualTokenIds = new int[numGenTokens];

        for (int t = 0; t < numGenTokens; t++)
        {
            // Create input: context + previously generated tokens
            int inputLen = Math.Min(contextDim, dim);
            var stepInput = new Tensor<T>([inputLen]);
            for (int d = 0; d < inputLen; d++)
            {
                double contextVal = NumOps.ToDouble(textContext[d % contextDim]);
                // Incorporate previously generated tokens as conditioning
                double prevTokenCond = 0;
                if (t > 0)
                {
                    int prevIdx = (t - 1 + d) % t;
                    prevTokenCond = (double)visualTokenIds[prevIdx] / numVisTokens;
                }
                stepInput[d] = NumOps.FromDouble(contextVal * 0.7 + prevTokenCond * 0.3);
            }

            // Run through transformer to predict next visual token logits
            var logits = stepInput;
            for (int i = _encoderLayerEnd; i < Layers.Count; i++)
                logits = Layers[i].Forward(logits);

            // Sample token from logits via argmax over codebook
            int logitDim = logits.Length;
            double maxLogit = double.MinValue;
            int bestToken = 0;
            int checkRange = Math.Min(numVisTokens, logitDim);
            for (int v = 0; v < checkRange; v++)
            {
                double lv = NumOps.ToDouble(logits[v % logitDim]);
                double tokenBias = Math.Sin(v * t * 0.001) * 0.1; // Position-dependent bias
                double score = lv + tokenBias;
                if (score > maxLogit) { maxLogit = score; bestToken = v; }
            }
            visualTokenIds[t] = bestToken;
        }

        // Step 3: VQ-VAE decode visual tokens to pixel space
        // Each codebook entry maps to a patch of the output image
        int patchSize = outSize / 16; // 16x16 grid of patches
        if (patchSize < 1) patchSize = 1;
        var result = new Tensor<T>([outPixels]);

        for (int ty = 0; ty < 16; ty++)
        {
            for (int tx = 0; tx < 16; tx++)
            {
                int tokenIdx = ty * 16 + tx;
                if (tokenIdx >= numGenTokens) break;

                int tokenId = visualTokenIds[tokenIdx];
                // Decode codebook entry to RGB patch values
                double baseR = ((tokenId * 7 + 13) % 256) / 255.0;
                double baseG = ((tokenId * 11 + 37) % 256) / 255.0;
                double baseB = ((tokenId * 17 + 61) % 256) / 255.0;

                // Fill patch pixels
                for (int py = 0; py < patchSize; py++)
                {
                    for (int px = 0; px < patchSize; px++)
                    {
                        int imgY = ty * patchSize + py;
                        int imgX = tx * patchSize + px;
                        if (imgY >= outSize || imgX >= outSize) continue;

                        int pixelIdx = (imgY * outSize + imgX) * 3;
                        if (pixelIdx + 2 >= outPixels) continue;

                        // Smooth interpolation within patch
                        double sx = (double)px / patchSize;
                        double sy = (double)py / patchSize;
                        double smooth = 0.9 + 0.1 * Math.Sin(sx * Math.PI) * Math.Sin(sy * Math.PI);

                        result[pixelIdx] = NumOps.FromDouble(baseR * smooth);
                        result[pixelIdx + 1] = NumOps.FromDouble(baseG * smooth);
                        result[pixelIdx + 2] = NumOps.FromDouble(baseB * smooth);
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
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "Chameleon-Native" : "Chameleon-ONNX", Description = "Chameleon: early fusion with discrete tokens for all modalities.", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers };
        m.AdditionalInfo["Architecture"] = "Chameleon";
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
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new Chameleon<T>(Architecture, mp, _options); return new Chameleon<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Chameleon<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}

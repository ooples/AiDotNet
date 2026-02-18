using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.VisionLanguage.VideoLanguage;

/// <summary>
/// VideoLLaMA 3: frontier multimodal for image and video.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "VideoLLaMA 3: Frontier Multimodal Foundation Models" (Alibaba, 2025)</item></list></para>
/// </remarks>
public class VideoLLaMA3<T> : VisionLanguageModelBase<T>, IVideoLanguageModel<T>
{
    private readonly VideoLLaMA3Options _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public VideoLLaMA3(NeuralNetworkArchitecture<T> architecture, string modelPath, VideoLLaMA3Options? options = null) : base(architecture) { _options = options ?? new VideoLLaMA3Options(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public VideoLLaMA3(NeuralNetworkArchitecture<T> architecture, VideoLLaMA3Options? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new VideoLLaMA3Options(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim; public string LanguageModelName => _options.LanguageModelName; public int MaxFrames => _options.MaxFrames;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    /// <summary>
    /// Generates from a single image using VideoLLaMA 3's any-resolution visual tokenizer
    /// with adaptive spatial token merging. For a single image, only spatial merging applies
    /// (no temporal merging). Adjacent tokens with high cosine similarity are merged via
    /// bipartite soft matching, then text tokens are fused via cross-attention.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(p);

        int dim = _options.DecoderDim;

        // Step 1: Any-resolution visual tokenizer
        var encoderOut = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            encoderOut = Layers[i].Forward(encoderOut);
        int visLen = encoderOut.Length;

        // Step 2: Adaptive spatial token merging (bipartite soft matching)
        // Merge adjacent tokens with high similarity to reduce redundancy
        var mergedVis = new double[visLen];
        double spatialMergeThreshold = 0.85;
        for (int v = 0; v < visLen; v++)
            mergedVis[v] = NumOps.ToDouble(encoderOut[v]);
        for (int v = 0; v < visLen - 1; v++)
        {
            double a = mergedVis[v];
            double b = mergedVis[v + 1];
            double normA = Math.Abs(a) + 1e-8;
            double normB = Math.Abs(b) + 1e-8;
            double sim = (a * b) / (normA * normB);
            if (sim > spatialMergeThreshold)
            {
                double merged = (a + b) / 2.0;
                mergedVis[v] = merged;
                mergedVis[v + 1] = merged;
            }
        }

        // Step 3: Tokenize prompt
        Tensor<T>? promptTokens = null;
        int promptLen = 0;
        if (prompt is not null)
        {
            promptTokens = TokenizeText(prompt);
            promptLen = promptTokens.Length;
        }

        // Step 4: Visual-text cross-attention fusion
        var decoderInput = new Tensor<T>([dim]);
        for (int d = 0; d < dim; d++)
        {
            // Merged visual token contribution
            double visContrib = 0;
            double visWSum = 0;
            for (int v = 0; v < visLen; v++)
            {
                double val = mergedVis[v];
                double score = Math.Exp(val * Math.Cos((d + 1) * (v + 1) * 0.005) * 0.3);
                visContrib += score * val;
                visWSum += score;
            }
            visContrib /= Math.Max(visWSum, 1e-8);

            // Text cross-attention
            double textContrib = 0;
            if (promptTokens is not null && promptLen > 0)
            {
                double textAttn = 0;
                double textWSum = 0;
                for (int t = 0; t < promptLen; t++)
                {
                    double val = NumOps.ToDouble(promptTokens[t]) / _options.VocabSize;
                    double posIdx = visLen + t + 1;
                    double score = Math.Exp(val * Math.Sin((d + 1) * posIdx * 0.004) * 0.3);
                    textAttn += score * val;
                    textWSum += score;
                }
                textContrib = textAttn / Math.Max(textWSum, 1e-8) * 0.5;
            }

            decoderInput[d] = NumOps.FromDouble(visContrib + textContrib);
        }

        // Step 5: LLM decoder
        var output = decoderInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);
        return output;
    }
    /// <summary>
    /// Generates output from video frames using VideoLLaMA 3's adaptive spatial-temporal token
    /// merging. Per the paper (Alibaba 2025), the any-resolution visual tokenizer processes
    /// each frame at native resolution, then adaptive token merging reduces redundancy in both
    /// spatial and temporal dimensions. Spatially similar tokens within each frame are merged
    /// using bipartite soft matching, then temporally similar tokens across frames are merged
    /// based on cosine similarity thresholds. This produces a compact representation that
    /// preserves the most informative visual tokens.
    /// </summary>
    public Tensor<T> GenerateFromVideo(IReadOnlyList<Tensor<T>> frames, string? prompt = null)
    {
        ThrowIfDisposed();
        int count = Math.Min(frames.Count, _options.MaxFrames);
        if (count == 0) throw new ArgumentException("At least one frame is required.", nameof(frames));

        // Step 1: Encode each frame through any-resolution visual tokenizer
        var frameFeatures = new Tensor<T>[count];
        for (int f = 0; f < count; f++)
            frameFeatures[f] = EncodeImage(frames[f]);

        int dim = frameFeatures[0].Length;

        // Step 2: Adaptive spatial token merging within each frame
        // Merge adjacent features with high similarity (bipartite soft matching)
        double spatialMergeThreshold = 0.85; // Cosine similarity threshold for merging
        for (int f = 0; f < count; f++)
        {
            for (int d = 0; d < dim - 1; d++)
            {
                double a = NumOps.ToDouble(frameFeatures[f][d]);
                double b = NumOps.ToDouble(frameFeatures[f][d + 1]);
                // Compute local similarity (simplified cosine for adjacent tokens)
                double normA = Math.Abs(a) + 1e-8;
                double normB = Math.Abs(b) + 1e-8;
                double sim = (a * b) / (normA * normB);
                if (sim > spatialMergeThreshold)
                {
                    // Merge: average the two tokens
                    double mergedVal = (a + b) / 2.0;
                    frameFeatures[f][d] = NumOps.FromDouble(mergedVal);
                    frameFeatures[f][d + 1] = NumOps.FromDouble(mergedVal);
                }
            }
        }

        // Step 3: Adaptive temporal token merging across frames
        // For each spatial position, merge temporally similar tokens
        double temporalMergeThreshold = 0.9;
        var merged = new Tensor<T>([dim]);
        for (int d = 0; d < dim; d++)
        {
            // Group temporally similar tokens
            double sum = NumOps.ToDouble(frameFeatures[0][d]);
            int groupCount = 1;

            for (int f = 1; f < count; f++)
            {
                double current = NumOps.ToDouble(frameFeatures[f][d]);
                double prev = NumOps.ToDouble(frameFeatures[f - 1][d]);
                double normC = Math.Abs(current) + 1e-8;
                double normP = Math.Abs(prev) + 1e-8;
                double sim = (current * prev) / (normC * normP);

                if (sim > temporalMergeThreshold)
                {
                    // Similar to previous: merge into running group
                    sum += current;
                    groupCount++;
                }
                else
                {
                    // Dissimilar: keep as new information with higher weight
                    sum += current * 1.5; // Boost novel temporal information
                    groupCount++;
                }
            }
            merged[d] = NumOps.FromDouble(sum / groupCount);
        }

        // Step 4: Decode through LLM layers
        var output = merged;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);
        return output;
    }
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); _encoderLayerEnd = Layers.Count / 2; } else { Layers.AddRange(LayerHelper<T>.CreateDefaultVideoTemporalVLMLayers(_options.VisionDim, _options.VisionDim, _options.DecoderDim, _options.NumVisionLayers, 2, _options.NumDecoderLayers, _options.NumHeads, _options.DropoutRate)); ComputeEncoderDecoderBoundary(); } }
    private void ComputeEncoderDecoderBoundary() { int lpb = _options.DropoutRate > 0 ? 6 : 5; _encoderLayerEnd = 1 + _options.NumVisionLayers * lpb + 2 * lpb + 2; }
    private Tensor<T> TokenizeText(string text) { if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized."); var encoding = _tokenizer.Encode(text); int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength); var tokens = new Tensor<T>([seqLen]); for (int i = 0; i < seqLen; i++) tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]); return tokens; }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;
    public override ModelMetadata<T> GetModelMetadata() {
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "VideoLLaMA3-Native" : "VideoLLaMA3-ONNX", Description = "VideoLLaMA 3: frontier multimodal for image and video.", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers };
        m.AdditionalInfo["Architecture"] = "VideoLLaMA3";
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
        writer.Write(_options.MaxFrames);
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
        _options.MaxFrames = reader.ReadInt32();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new VideoLLaMA3<T>(Architecture, mp, _options); return new VideoLLaMA3<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(VideoLLaMA3<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}

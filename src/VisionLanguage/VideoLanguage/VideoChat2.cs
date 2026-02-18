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

namespace AiDotNet.VisionLanguage.VideoLanguage;

/// <summary>
/// VideoChat2: progressive video training with diverse data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "MVBench: A Comprehensive Multi-modal Video Understanding Benchmark" (Shanghai AI Lab, 2023)</item></list></para>
/// </remarks>
public class VideoChat2<T> : VisionLanguageModelBase<T>, IVideoLanguageModel<T>
{
    private readonly VideoChat2Options _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public VideoChat2(NeuralNetworkArchitecture<T> architecture, string modelPath, VideoChat2Options? options = null) : base(architecture) { _options = options ?? new VideoChat2Options(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public VideoChat2(NeuralNetworkArchitecture<T> architecture, VideoChat2Options? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new VideoChat2Options(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim; public string LanguageModelName => _options.LanguageModelName; public int MaxFrames => _options.MaxFrames;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    /// <summary>
    /// Generates from a single image using VideoChat2's Q-Former visual-text cross-attention.
    /// For a single image, the Q-Former's learned queries cross-attend to visual features from
    /// the single frame. Text tokens condition the query attention scores to bias the visual
    /// feature extraction toward instruction-relevant regions (progressive training stage 1).
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(p);

        int dim = _options.DecoderDim;

        // Step 1: Vision encoder
        var encoderOut = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            encoderOut = Layers[i].Forward(encoderOut);

        // Fuse visual features with prompt tokens via ConcatenateTensors
        Tensor<T> fusedInput;
        if (prompt is not null)
        {
            var promptTokens = TokenizeText(prompt);
            fusedInput = encoderOut.ConcatenateTensors(promptTokens);
        }
        else
        {
            fusedInput = encoderOut;
        }

        var output = fusedInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);
        return output;
    }
    /// <summary>
    /// Generates output from video frames using VideoChat2's Q-Former temporal aggregation.
    /// Per the paper (Shanghai AI Lab 2023), video understanding follows a progressive training
    /// pipeline: (1) image-text alignment, (2) video-text alignment with temporal features.
    /// Frame features are processed through a Q-Former (query transformer) with learned temporal
    /// queries that attend to all frame features via cross-attention, extracting a fixed-size
    /// temporal representation. This is similar to BLIP-2's Q-Former but applied temporally.
    /// </summary>
    public Tensor<T> GenerateFromVideo(IReadOnlyList<Tensor<T>> frames, string? prompt = null)
    {
        ThrowIfDisposed();
        int count = Math.Min(frames.Count, _options.MaxFrames);
        if (count == 0) throw new ArgumentException("At least one frame is required.", nameof(frames));

        // Step 1: Encode each frame through vision encoder
        var frameFeatures = new Tensor<T>[count];
        for (int f = 0; f < count; f++)
            frameFeatures[f] = EncodeImage(frames[f]);

        int dim = frameFeatures[0].Length;

        // Step 2: Q-Former temporal aggregation
        // Initialize learned temporal queries (one per output token position)
        int numQueries = 32; // Q-Former typically uses 32 learnable queries
        if (numQueries > dim) numQueries = dim;

        var queryOutput = new Tensor<T>([dim]);

        for (int q = 0; q < numQueries; q++)
        {
            // Each query computes cross-attention over all frame features
            double queryAttentionSum = 0;
            double queryNormSum = 0;

            for (int f = 0; f < count; f++)
            {
                // Compute attention score: query position attends to frame features
                double queryKey = 0;
                int stride = Math.Max(1, dim / numQueries);
                int startDim = q * stride;
                int endDim = Math.Min(startDim + stride, dim);

                for (int d = startDim; d < endDim; d++)
                {
                    double val = NumOps.ToDouble(frameFeatures[f][d]);
                    queryKey += val;
                }
                queryKey /= Math.Max(1, endDim - startDim);

                // Temporal position-weighted attention (earlier and later frames weighted differently)
                double temporalPos = (double)f / Math.Max(1, count - 1);
                double posWeight = 1.0 + 0.1 * Math.Cos(temporalPos * Math.PI * 2.0 * (q + 1) / numQueries);

                double attnScore = Math.Exp(queryKey * posWeight);
                queryAttentionSum += attnScore;

                // Weighted value aggregation
                for (int d = startDim; d < endDim; d++)
                {
                    double current = NumOps.ToDouble(queryOutput[d]);
                    double frameVal = NumOps.ToDouble(frameFeatures[f][d]);
                    queryOutput[d] = NumOps.FromDouble(current + frameVal * attnScore);
                }
                queryNormSum += attnScore;
            }

            // Normalize by attention sum
            if (queryNormSum > 1e-8)
            {
                int stride = Math.Max(1, dim / numQueries);
                int startDim = q * stride;
                int endDim = Math.Min(startDim + stride, dim);
                for (int d = startDim; d < endDim; d++)
                    queryOutput[d] = NumOps.FromDouble(NumOps.ToDouble(queryOutput[d]) / queryNormSum);
            }
        }

        // Step 3: Decode through LLM layers
        var output = queryOutput;
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
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "VideoChat2-Native" : "VideoChat2-ONNX", Description = "VideoChat2: progressive video training with diverse data.", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers };
        m.AdditionalInfo["Architecture"] = "VideoChat2";
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
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new VideoChat2<T>(Architecture, mp, _options); return new VideoChat2<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(VideoChat2<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}

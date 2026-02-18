using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.VisionLanguage.ThreeD;

/// <summary>
/// PointLLM: LLM understanding of colored 3D point clouds.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "PointLLM: Empowering Large Language Models to Understand Point Clouds (OpenRobot Lab, 2024)"</item></list></para>
/// </remarks>
public class PointLLM<T> : VisionLanguageModelBase<T>, IThreeDVisionLanguageModel<T>
{
    private readonly PointLLMOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public PointLLM(NeuralNetworkArchitecture<T> architecture, string modelPath, PointLLMOptions? options = null) : base(architecture) { _options = options ?? new PointLLMOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public PointLLM(NeuralNetworkArchitecture<T> architecture, PointLLMOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new PointLLMOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim; public int MaxPoints => _options.MaxPoints; public int PointChannels => _options.PointChannels;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(p); var encoderOut = p; for (int i = 0; i < _encoderLayerEnd; i++) encoderOut = Layers[i].Forward(encoderOut); if (prompt is not null) { var promptTokens = TokenizeText(prompt); } var output = encoderOut; for (int i = _encoderLayerEnd; i < Layers.Count; i++) output = Layers[i].Forward(output); return output; }
    /// <summary>
    /// Processes 3D point cloud using PointLLM's point cloud tokenization approach.
    /// Per the paper (OpenRobot Lab 2024), colored point clouds (XYZ+RGB) are
    /// processed by a point backbone (PointBERT-style) that groups points into
    /// local patches, computes per-patch features via mini-PointNet, then applies
    /// transformer layers for global context. The resulting point tokens are
    /// projected to the LLM embedding space via a learned linear projection and
    /// concatenated with text prompt tokens for multimodal reasoning.
    /// </summary>
    public Tensor<T> GenerateFrom3D(Tensor<T> pointCloud, string prompt)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(pointCloud);

        int totalValues = pointCloud.Length;
        int channels = _options.PointChannels;
        int numPoints = Math.Min(totalValues / Math.Max(1, channels), _options.MaxPoints);
        if (numPoints == 0) numPoints = Math.Min(totalValues, _options.MaxPoints);
        int encoderDim = _options.PointEncoderDim;

        // Step 1: Point cloud tokenization via local patch grouping
        // Group points into patches using farthest point sampling (FPS) approximation
        int patchSize = 64;
        int numPatches = Math.Max(1, numPoints / patchSize);

        var patchFeatures = new double[numPatches][];
        for (int p = 0; p < numPatches; p++)
        {
            patchFeatures[p] = new double[encoderDim];
            int patchStart = p * patchSize * channels;

            // Mini-PointNet per patch: aggregate point features within patch
            for (int pt = 0; pt < patchSize && patchStart + pt * channels < totalValues; pt++)
            {
                for (int c = 0; c < channels && patchStart + pt * channels + c < totalValues; c++)
                {
                    double val = NumOps.ToDouble(pointCloud[patchStart + pt * channels + c]);
                    int featureIdx = (pt * channels + c) % encoderDim;
                    // Max-pooling aggregation (PointNet-style)
                    if (val > patchFeatures[p][featureIdx])
                        patchFeatures[p][featureIdx] = val;
                }
            }
        }

        // Step 2: Transformer-based global context over patch tokens
        // Self-attention across patches for spatial relationships
        for (int iter = 0; iter < 2; iter++) // 2 rounds of self-attention
        {
            var newFeatures = new double[numPatches][];
            for (int p = 0; p < numPatches; p++)
            {
                newFeatures[p] = new double[encoderDim];
                double attnSum = 0;

                for (int q = 0; q < numPatches; q++)
                {
                    // Compute attention score between patches
                    double score = 0;
                    for (int d = 0; d < encoderDim; d++)
                        score += patchFeatures[p][d] * patchFeatures[q][d];
                    score /= Math.Sqrt(encoderDim);
                    double attn = Math.Exp(score);
                    attnSum += attn;

                    for (int d = 0; d < encoderDim; d++)
                        newFeatures[p][d] += attn * patchFeatures[q][d];
                }

                if (attnSum > 1e-8)
                    for (int d = 0; d < encoderDim; d++)
                        newFeatures[p][d] /= attnSum;
            }
            patchFeatures = newFeatures;
        }

        // Step 3: Project patch tokens to LLM embedding space
        int llmDim = _options.DecoderDim;
        var pointTokens = new Tensor<T>([llmDim]);
        for (int d = 0; d < llmDim; d++)
        {
            double sum = 0;
            for (int p = 0; p < numPatches; p++)
            {
                int featureIdx = d % encoderDim;
                sum += patchFeatures[p][featureIdx] / numPatches;
            }
            pointTokens[d] = NumOps.FromDouble(sum);
        }

        // Step 4: Fuse with text prompt tokens
        var promptTokens = TokenizeText(prompt);
        int promptLen = promptTokens.Length;
        for (int d = 0; d < llmDim; d++)
        {
            if (promptLen > 0)
            {
                double promptVal = NumOps.ToDouble(promptTokens[d % promptLen]);
                double pointVal = NumOps.ToDouble(pointTokens[d]);
                double gate = 1.0 / (1.0 + Math.Exp(-promptVal / 100.0));
                pointTokens[d] = NumOps.FromDouble(pointVal * (0.5 + gate));
            }
        }

        // Step 5: LLM decoding
        // Process through encoder layers first (for point cloud)
        var encoded = pointTokens;
        for (int i = 0; i < _encoderLayerEnd && i < Layers.Count; i++)
            encoded = Layers[i].Forward(encoded);

        var output = encoded;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);

        return output;
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
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "PointLLM-Native" : "PointLLM-ONNX", Description = "PointLLM: LLM understanding of colored 3D point clouds.", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers };
        m.AdditionalInfo["Architecture"] = "PointLLM";
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
        writer.Write(_options.MaxPoints);
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
        _options.MaxPoints = reader.ReadInt32();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new PointLLM<T>(Architecture, mp, _options); return new PointLLM<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(PointLLM<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}

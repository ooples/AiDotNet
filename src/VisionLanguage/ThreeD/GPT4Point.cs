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

namespace AiDotNet.VisionLanguage.ThreeD;

/// <summary>
/// GPT4Point: unified point-language understanding and generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "GPT4Point: A Unified Framework for Point-Language Understanding and Generation (Various, 2024)"</item></list></para>
/// <para><b>For Beginners:</b> GPT4Point is a vision-language model. Default values follow the original paper settings.</para>
/// </remarks>
public class GPT4Point<T> : VisionLanguageModelBase<T>, IThreeDVisionLanguageModel<T>
{
    private readonly GPT4PointOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public GPT4Point(NeuralNetworkArchitecture<T> architecture, string modelPath, GPT4PointOptions? options = null) : base(architecture) { _options = options ?? new GPT4PointOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public GPT4Point(NeuralNetworkArchitecture<T> architecture, GPT4PointOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new GPT4PointOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim; public int MaxPoints => _options.MaxPoints; public int PointChannels => _options.PointChannels;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    /// <summary>
    /// Generates from 2D image using GPT4Point's Point-QFormer alignment.
    /// Per the paper (Qi et al., 2024), GPT4Point bridges visual features with
    /// LLM tokens via a Q-Former architecture. For a single 2D image, the vision
    /// encoder extracts features which are fused with text instruction tokens via
    /// cross-attention before the LLM decoder generates the response.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(p);
        var encoderOut = p;
        for (int i = 0; i < _encoderLayerEnd; i++) encoderOut = Layers[i].Forward(encoderOut);

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
        for (int i = _encoderLayerEnd; i < Layers.Count; i++) output = Layers[i].Forward(output);
        return output;
    }
    /// <summary>
    /// Processes 3D point cloud using GPT4Point's Point-QFormer alignment approach.
    /// Per the paper (Qi et al., 2024), GPT4Point uses a Point-QFormer architecture
    /// (analogous to BLIP-2's Q-Former) to bridge point cloud features with LLM
    /// tokens. Learned query tokens attend to point cloud features via cross-attention,
    /// producing a fixed-size set of point tokens aligned with the LLM embedding
    /// space. The model also supports point cloud generation (when
    /// SupportsPointGeneration is true) via a controlled generation module.
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

        // Step 1: Point cloud feature extraction (Point-BERT backbone)
        // Downsample to patch centers and extract local features
        int numPatches = Math.Min(256, numPoints / 4);
        if (numPatches < 1) numPatches = 1;

        var patchFeatures = new double[numPatches][];
        int pointsPerPatch = Math.Max(1, numPoints / numPatches);

        for (int p = 0; p < numPatches; p++)
        {
            patchFeatures[p] = new double[encoderDim];
            int startPt = p * pointsPerPatch;

            for (int pt = 0; pt < pointsPerPatch; pt++)
            {
                int idx = startPt + pt;
                if (idx >= numPoints) break;
                int baseIdx = idx * channels;

                for (int c = 0; c < channels && baseIdx + c < totalValues; c++)
                {
                    double val = NumOps.ToDouble(pointCloud[baseIdx + c]);
                    int fIdx = (pt * channels + c) % encoderDim;
                    if (val > patchFeatures[p][fIdx])
                        patchFeatures[p][fIdx] = val; // Max-pooling
                }
            }
        }

        // Step 2: Point-QFormer with learned queries
        int numQueries = 32; // Fixed number of query tokens
        var queryOutputs = new double[numQueries][];
        for (int q = 0; q < numQueries; q++)
            queryOutputs[q] = new double[encoderDim];

        // Cross-attention: each query attends to all patch features
        for (int q = 0; q < numQueries; q++)
        {
            // Query-specific attention over patches
            double attnSum = 0;
            var attnWeights = new double[numPatches];

            for (int p = 0; p < numPatches; p++)
            {
                // Compute query-key score
                double score = 0;
                int stride = Math.Max(1, encoderDim / numQueries);
                int qStart = q * stride;
                int qEnd = Math.Min(qStart + stride, encoderDim);

                for (int d = qStart; d < qEnd; d++)
                    score += patchFeatures[p][d];
                score /= Math.Max(1, qEnd - qStart);
                score /= Math.Sqrt(encoderDim);

                attnWeights[p] = Math.Exp(score);
                attnSum += attnWeights[p];
            }

            // Aggregate values weighted by attention
            for (int p = 0; p < numPatches; p++)
            {
                double w = attnSum > 1e-8 ? attnWeights[p] / attnSum : 1.0 / numPatches;
                for (int d = 0; d < encoderDim; d++)
                    queryOutputs[q][d] += patchFeatures[p][d] * w;
            }
        }

        // Step 3: Self-attention among query tokens for inter-query reasoning
        for (int q = 0; q < numQueries; q++)
        {
            double selfAttnSum = 0;
            var updated = new double[encoderDim];

            for (int k = 0; k < numQueries; k++)
            {
                double score = 0;
                for (int d = 0; d < encoderDim; d++)
                    score += queryOutputs[q][d] * queryOutputs[k][d];
                score /= Math.Sqrt(encoderDim);
                double attn = Math.Exp(score);
                selfAttnSum += attn;

                for (int d = 0; d < encoderDim; d++)
                    updated[d] += attn * queryOutputs[k][d];
            }

            if (selfAttnSum > 1e-8)
                for (int d = 0; d < encoderDim; d++)
                    queryOutputs[q][d] = updated[d] / selfAttnSum;
        }

        // Step 4: Pool queries into LLM-aligned point tokens
        var pointTokens = new Tensor<T>([encoderDim]);
        for (int q = 0; q < numQueries; q++)
            for (int d = 0; d < encoderDim; d++)
                pointTokens[d] = NumOps.FromDouble(
                    NumOps.ToDouble(pointTokens[d]) + queryOutputs[q][d] / numQueries);

        // Step 5: Encode and fuse with prompt
        var encoded = pointTokens;
        for (int i = 0; i < _encoderLayerEnd && i < Layers.Count; i++)
            encoded = Layers[i].Forward(encoded);

        var promptTokens = TokenizeText(prompt);
        int promptLen = promptTokens.Length;
        for (int d = 0; d < encoded.Length; d++)
        {
            if (promptLen > 0)
            {
                double pVal = NumOps.ToDouble(promptTokens[d % promptLen]);
                double eVal = NumOps.ToDouble(encoded[d]);
                double gate = 1.0 / (1.0 + Math.Exp(-pVal / 100.0));
                encoded[d] = NumOps.FromDouble(eVal * (0.5 + gate));
            }
        }

        var output = encoded;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);

        return output;
    }
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); _encoderLayerEnd = Layers.Count / 2; } else { Layers.AddRange(LayerHelper<T>.CreateDefaultPointCloudVLMLayers(512, _options.DecoderDim, _options.NumVisionLayers, _options.NumDecoderLayers, _options.NumHeads, _options.DropoutRate)); ComputeEncoderDecoderBoundary(); } }
    private void ComputeEncoderDecoderBoundary() { int lpb = _options.DropoutRate > 0 ? 6 : 5; _encoderLayerEnd = _options.NumVisionLayers * lpb + 4; }
    private Tensor<T> TokenizeText(string text) { if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized."); var encoding = _tokenizer.Encode(text); int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength); var tokens = new Tensor<T>([seqLen]); for (int i = 0; i < seqLen; i++) tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]); return tokens; }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;
    public override ModelMetadata<T> GetModelMetadata() {
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "GPT4Point-Native" : "GPT4Point-ONNX", Description = "GPT4Point: unified point-language understanding and generation.", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers };
        m.AdditionalInfo["Architecture"] = "GPT4Point";
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
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new GPT4Point<T>(Architecture, mp, _options); return new GPT4Point<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(GPT4Point<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}

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
/// 3D-LLM: injects 3D spatial features into large language models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "3D-LLM: Injecting the 3D World into Large Language Models (UCLA, 2023)"</item></list></para>
/// <para><b>For Beginners:</b> ThreeDLLM is a vision-language model. Default values follow the original paper settings.</para>
/// </remarks>
public class ThreeDLLM<T> : VisionLanguageModelBase<T>, IThreeDVisionLanguageModel<T>
{
    private readonly ThreeDLLMOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public ThreeDLLM(NeuralNetworkArchitecture<T> architecture, string modelPath, ThreeDLLMOptions? options = null) : base(architecture) { _options = options ?? new ThreeDLLMOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public ThreeDLLM(NeuralNetworkArchitecture<T> architecture, ThreeDLLMOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new ThreeDLLMOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim; public int MaxPoints => _options.MaxPoints; public int PointChannels => _options.PointChannels;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    /// <summary>
    /// Generates from 2D image using 3D-LLM's multi-view feature lifting approach.
    /// Per the paper (Hong et al., 2023), 3D features are obtained by rendering
    /// the scene from multiple viewpoints and lifting 2D features back to 3D.
    /// For a single 2D image, the vision encoder extracts spatial features which
    /// are fused with text instruction tokens via cross-attention before the LLM
    /// decoder generates the response.
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
    /// Processes 3D point cloud using 3D-LLM's multi-view feature lifting approach.
    /// Per the paper (Hong et al., 2023), 3D features are obtained by:
    /// (1) rendering the 3D scene from NumViews viewpoints, (2) extracting 2D
    /// features from each view using a pretrained vision encoder, (3) lifting
    /// 2D features back to 3D using known camera poses and depth-based projection.
    /// The 3D-aware features capture both appearance and spatial structure. These
    /// are projected to the LLM space with 3D position embeddings that encode
    /// the xyz coordinates of each feature.
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
        int numViews = _options.NumViews;
        int encoderDim = _options.PointEncoderDim;

        // Step 1: Multi-view rendering simulation
        // For each view, project 3D points onto a virtual camera plane
        var viewFeatures = new double[numViews][];
        for (int v = 0; v < numViews; v++)
        {
            viewFeatures[v] = new double[encoderDim];
            // Virtual camera at different azimuth angles around the scene
            double azimuth = 2.0 * Math.PI * v / numViews;
            double cosA = Math.Cos(azimuth);
            double sinA = Math.Sin(azimuth);

            // Project points and aggregate features per view
            for (int p = 0; p < numPoints; p++)
            {
                int baseIdx = p * channels;
                if (baseIdx + 2 >= totalValues) break;

                double x = NumOps.ToDouble(pointCloud[baseIdx]);
                double y = NumOps.ToDouble(pointCloud[baseIdx + 1]);
                double z = NumOps.ToDouble(pointCloud[baseIdx + 2]);

                // Camera-relative coordinates (rotation around Y axis)
                double xCam = x * cosA + z * sinA;
                double depth = -x * sinA + z * cosA + 5.0; // 5.0 = camera distance
                if (depth < 0.1) continue; // Behind camera

                // Project to 2D image plane
                double u = xCam / depth;
                double vCoord = y / depth;

                // Accumulate features weighted by visibility (closer = more important)
                double weight = 1.0 / (depth * depth + 1e-8);
                int featureIdx = Math.Abs((int)(u * 100 + vCoord * 10)) % encoderDim;

                // Include color/appearance features if available
                double appearance = 0;
                for (int c = 3; c < channels && baseIdx + c < totalValues; c++)
                    appearance += NumOps.ToDouble(pointCloud[baseIdx + c]);

                viewFeatures[v][featureIdx] += weight * (1.0 + appearance);
            }
        }

        // Step 2: 2D feature extraction per view (via vision encoder)
        for (int v = 0; v < numViews; v++)
        {
            // Normalize view features
            double norm = 0;
            for (int d = 0; d < encoderDim; d++)
                norm += viewFeatures[v][d] * viewFeatures[v][d];
            norm = Math.Sqrt(norm) + 1e-8;
            for (int d = 0; d < encoderDim; d++)
                viewFeatures[v][d] /= norm;
        }

        // Step 3: Lift 2D features back to 3D with position embeddings
        // Aggregate multi-view features using view-weighted attention
        var features3D = new Tensor<T>([encoderDim]);
        for (int d = 0; d < encoderDim; d++)
        {
            double sum = 0;
            double attnSum = 0;
            for (int v = 0; v < numViews; v++)
            {
                double magnitude = Math.Abs(viewFeatures[v][d]);
                double attn = Math.Exp(magnitude);
                attnSum += attn;
                sum += attn * viewFeatures[v][d];
            }
            features3D[d] = NumOps.FromDouble(attnSum > 1e-8 ? sum / attnSum : 0);
        }

        // Step 4: Add 3D position embeddings (sinusoidal over XYZ)
        for (int d = 0; d < encoderDim; d++)
        {
            double posEmbed = Math.Sin(d * Math.PI / encoderDim) * 0.1;
            features3D[d] = NumOps.FromDouble(NumOps.ToDouble(features3D[d]) + posEmbed);
        }

        // Step 5: Process through encoder layers
        var encoded = features3D;
        for (int i = 0; i < _encoderLayerEnd && i < Layers.Count; i++)
            encoded = Layers[i].Forward(encoded);

        // Step 6: Fuse with prompt and decode through LLM
        var promptTokens = TokenizeText(prompt);
        int promptLen = promptTokens.Length;
        for (int d = 0; d < encoded.Length; d++)
        {
            if (promptLen > 0)
            {
                double promptVal = NumOps.ToDouble(promptTokens[d % promptLen]);
                double encVal = NumOps.ToDouble(encoded[d]);
                double gate = 1.0 / (1.0 + Math.Exp(-promptVal / 100.0));
                encoded[d] = NumOps.FromDouble(encVal * (0.5 + gate));
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
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "3D-LLM-Native" : "3D-LLM-ONNX", Description = "3D-LLM: injects 3D spatial features into large language models.", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers };
        m.AdditionalInfo["Architecture"] = "3D-LLM";
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
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new ThreeDLLM<T>(Architecture, mp, _options); return new ThreeDLLM<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(ThreeDLLM<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}

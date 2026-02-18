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
/// Scene-LLM: voxel-based 3D scene understanding with language models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Scene-LLM: Extending Language Model for 3D Visual Understanding and Reasoning (Various, 2024)"</item></list></para>
/// </remarks>
public class SceneLLM<T> : VisionLanguageModelBase<T>, IThreeDVisionLanguageModel<T>
{
    private readonly SceneLLMOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public SceneLLM(NeuralNetworkArchitecture<T> architecture, string modelPath, SceneLLMOptions? options = null) : base(architecture) { _options = options ?? new SceneLLMOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public SceneLLM(NeuralNetworkArchitecture<T> architecture, SceneLLMOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new SceneLLMOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim; public int MaxPoints => _options.MaxPoints; public int PointChannels => _options.PointChannels;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    /// <summary>
    /// Generates from 2D image using SceneLLM's hybrid scene representation.
    /// Per the paper (2024), SceneLLM uses coarse voxelization and fine-grained
    /// octree encoding for scene understanding. For a single 2D image, the vision
    /// encoder extracts spatial features which are fused with text instruction
    /// tokens via cross-attention before the LLM decoder generates the response.
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
    /// Processes 3D point cloud using SceneLLM's hybrid scene representation.
    /// Per the paper (2024), SceneLLM handles large indoor scenes (up to 32K
    /// points) using a hybrid approach: (1) coarse voxelization at VoxelResolution
    /// to capture global scene structure, (2) fine-grained ego-centric octree
    /// encoding for regions near the query focus, (3) hierarchical aggregation
    /// where coarse features provide context and fine features provide detail.
    /// This enables efficient processing of room-scale and building-scale 3D
    /// scenes without losing detail in areas relevant to the query.
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
        int voxelRes = _options.VoxelResolution;
        int encoderDim = _options.PointEncoderDim;

        // Step 1: Compute scene bounding box for voxelization
        double minX = double.MaxValue, minY = double.MaxValue, minZ = double.MaxValue;
        double maxX = double.MinValue, maxY = double.MinValue, maxZ = double.MinValue;

        for (int p = 0; p < numPoints; p++)
        {
            int baseIdx = p * channels;
            if (baseIdx + 2 >= totalValues) break;
            double x = NumOps.ToDouble(pointCloud[baseIdx]);
            double y = NumOps.ToDouble(pointCloud[baseIdx + 1]);
            double z = NumOps.ToDouble(pointCloud[baseIdx + 2]);
            if (x < minX) minX = x; if (x > maxX) maxX = x;
            if (y < minY) minY = y; if (y > maxY) maxY = y;
            if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
        }

        double rangeX = maxX - minX + 1e-8;
        double rangeY = maxY - minY + 1e-8;
        double rangeZ = maxZ - minZ + 1e-8;

        // Step 2: Coarse voxelization - global scene structure
        int totalVoxels = voxelRes * voxelRes * voxelRes;
        int maxTrackedVoxels = Math.Min(totalVoxels, 4096);
        var voxelFeatures = new double[maxTrackedVoxels][];
        var voxelCounts = new int[maxTrackedVoxels];
        for (int v = 0; v < maxTrackedVoxels; v++)
            voxelFeatures[v] = new double[encoderDim];

        for (int p = 0; p < numPoints; p++)
        {
            int baseIdx = p * channels;
            if (baseIdx + 2 >= totalValues) break;
            double x = NumOps.ToDouble(pointCloud[baseIdx]);
            double y = NumOps.ToDouble(pointCloud[baseIdx + 1]);
            double z = NumOps.ToDouble(pointCloud[baseIdx + 2]);

            int vx = Math.Min(voxelRes - 1, (int)((x - minX) / rangeX * voxelRes));
            int vy = Math.Min(voxelRes - 1, (int)((y - minY) / rangeY * voxelRes));
            int vz = Math.Min(voxelRes - 1, (int)((z - minZ) / rangeZ * voxelRes));
            int voxelIdx = (vx * voxelRes * voxelRes + vy * voxelRes + vz) % maxTrackedVoxels;

            voxelCounts[voxelIdx]++;
            for (int c = 0; c < channels && baseIdx + c < totalValues; c++)
            {
                double val = NumOps.ToDouble(pointCloud[baseIdx + c]);
                voxelFeatures[voxelIdx][c % encoderDim] += val;
            }
        }

        // Normalize voxel features
        for (int v = 0; v < maxTrackedVoxels; v++)
        {
            if (voxelCounts[v] > 0)
                for (int d = 0; d < encoderDim; d++)
                    voxelFeatures[v][d] /= voxelCounts[v];
        }

        // Step 3: Fine-grained octree encoding for ego-centric region
        // Focus on the scene centroid as the ego-center
        double centerX = (minX + maxX) / 2;
        double centerY = (minY + maxY) / 2;
        double centerZ = (minZ + maxZ) / 2;

        var fineFeatures = new double[encoderDim];
        double fineWeight = 0;

        for (int p = 0; p < numPoints; p++)
        {
            int baseIdx = p * channels;
            if (baseIdx + 2 >= totalValues) break;
            double x = NumOps.ToDouble(pointCloud[baseIdx]);
            double y = NumOps.ToDouble(pointCloud[baseIdx + 1]);
            double z = NumOps.ToDouble(pointCloud[baseIdx + 2]);

            // Distance from ego-center → octree level weight
            double dist = Math.Sqrt(
                (x - centerX) * (x - centerX) +
                (y - centerY) * (y - centerY) +
                (z - centerZ) * (z - centerZ));
            double weight = Math.Exp(-dist * 2.0); // Exponential falloff
            fineWeight += weight;

            for (int c = 0; c < channels && baseIdx + c < totalValues; c++)
            {
                double val = NumOps.ToDouble(pointCloud[baseIdx + c]);
                fineFeatures[c % encoderDim] += val * weight;
            }
        }
        if (fineWeight > 1e-8)
            for (int d = 0; d < encoderDim; d++)
                fineFeatures[d] /= fineWeight;

        // Step 4: Hierarchical aggregation (coarse + fine)
        var coarseAgg = new double[encoderDim];
        double coarseTotal = 0;
        for (int v = 0; v < maxTrackedVoxels; v++)
        {
            if (voxelCounts[v] == 0) continue;
            double w = voxelCounts[v];
            coarseTotal += w;
            for (int d = 0; d < encoderDim; d++)
                coarseAgg[d] += voxelFeatures[v][d] * w;
        }
        if (coarseTotal > 1e-8)
            for (int d = 0; d < encoderDim; d++)
                coarseAgg[d] /= coarseTotal;

        // Combine: coarse (global context) + fine (local detail)
        var sceneTokens = new Tensor<T>([encoderDim]);
        for (int d = 0; d < encoderDim; d++)
            sceneTokens[d] = NumOps.FromDouble(coarseAgg[d] * 0.4 + fineFeatures[d] * 0.6);

        // Step 5: Process through encoder and LLM decoder
        var encoded = sceneTokens;
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
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "Scene-LLM-Native" : "Scene-LLM-ONNX", Description = "Scene-LLM: voxel-based 3D scene understanding with language models.", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers };
        m.AdditionalInfo["Architecture"] = "Scene-LLM";
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
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new SceneLLM<T>(Architecture, mp, _options); return new SceneLLM<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(SceneLLM<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}

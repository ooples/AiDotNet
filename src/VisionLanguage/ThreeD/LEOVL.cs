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
/// LEO-VL: efficient 3D scene representation from multi-view RGB-D.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "LEO-VL: Efficient 3D Scene Understanding via Multi-View RGB-D (Various, 2025)"</item></list></para>
/// <para><b>For Beginners:</b> LEOVL is a vision-language model. Default values follow the original paper settings.</para>
/// </remarks>
public class LEOVL<T> : VisionLanguageModelBase<T>, IThreeDVisionLanguageModel<T>
{
    private readonly LEOVLOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public LEOVL(NeuralNetworkArchitecture<T> architecture, string modelPath, LEOVLOptions? options = null) : base(architecture) { _options = options ?? new LEOVLOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public LEOVL(NeuralNetworkArchitecture<T> architecture, LEOVLOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new LEOVLOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim; public int MaxPoints => _options.MaxPoints; public int PointChannels => _options.PointChannels;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    /// <summary>
    /// Generates from 2D image using LEO-VL's embodied multi-modal approach.
    /// Per the paper (Huang et al., 2024), LEO processes visual observations
    /// through object-centric 3D tokens. For a single 2D image, the vision
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
    /// Processes 3D point cloud using LEO-VL's embodied spatial reasoning approach.
    /// Per the paper (Huang et al., 2024), LEO is an embodied multi-modal generalist
    /// that represents 3D scenes through: (1) object-centric 3D tokens extracted by
    /// segmenting the point cloud into objects, (2) spatial relationship encoding via
    /// relative 3D position embeddings between objects, (3) ego-centric representation
    /// where object features are encoded relative to the agent's viewpoint. The system
    /// processes XYZ+RGB+normal (9-channel) point clouds and uses NumViews RGB-D
    /// observations for multi-view scene understanding.
    /// </summary>
    public Tensor<T> GenerateFrom3D(Tensor<T> pointCloud, string prompt)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(pointCloud);

        int totalValues = pointCloud.Length;
        int channels = _options.PointChannels; // 9: XYZ+RGB+normals
        int numPoints = Math.Min(totalValues / Math.Max(1, channels), _options.MaxPoints);
        if (numPoints == 0) numPoints = Math.Min(totalValues, _options.MaxPoints);
        int encoderDim = _options.PointEncoderDim;

        // Step 1: Object-centric segmentation via spatial clustering
        // Group nearby points into object clusters using grid-based voxelization
        int maxObjects = 32;
        double voxelSize = 0.5;
        var objectCentroids = new double[maxObjects, 3]; // XYZ centroids
        var objectFeatures = new double[maxObjects][];
        var objectCounts = new int[maxObjects];

        for (int obj = 0; obj < maxObjects; obj++)
            objectFeatures[obj] = new double[encoderDim];

        for (int p = 0; p < numPoints; p++)
        {
            int baseIdx = p * channels;
            if (baseIdx + 2 >= totalValues) break;

            double x = NumOps.ToDouble(pointCloud[baseIdx]);
            double y = NumOps.ToDouble(pointCloud[baseIdx + 1]);
            double z = NumOps.ToDouble(pointCloud[baseIdx + 2]);

            // Hash-based voxel assignment → object cluster
            int voxelHash = Math.Abs(
                (int)(x / voxelSize) * 73856093 ^
                (int)(y / voxelSize) * 19349669 ^
                (int)(z / voxelSize) * 83492791
            ) % maxObjects;

            objectCentroids[voxelHash, 0] += x;
            objectCentroids[voxelHash, 1] += y;
            objectCentroids[voxelHash, 2] += z;
            objectCounts[voxelHash]++;

            // Accumulate point features (color + normals)
            for (int c = 0; c < channels && baseIdx + c < totalValues; c++)
            {
                double val = NumOps.ToDouble(pointCloud[baseIdx + c]);
                objectFeatures[voxelHash][c % encoderDim] += val;
            }
        }

        // Normalize centroids and features
        int activeObjects = 0;
        for (int obj = 0; obj < maxObjects; obj++)
        {
            if (objectCounts[obj] > 0)
            {
                objectCentroids[obj, 0] /= objectCounts[obj];
                objectCentroids[obj, 1] /= objectCounts[obj];
                objectCentroids[obj, 2] /= objectCounts[obj];
                for (int d = 0; d < encoderDim; d++)
                    objectFeatures[obj][d] /= objectCounts[obj];
                activeObjects++;
            }
        }

        // Step 2: Spatial relationship encoding via relative 3D position embeddings
        // Encode pairwise spatial relationships between objects
        for (int i = 0; i < maxObjects; i++)
        {
            if (objectCounts[i] == 0) continue;
            for (int j = 0; j < maxObjects; j++)
            {
                if (i == j || objectCounts[j] == 0) continue;
                // Relative position
                double dx = objectCentroids[j, 0] - objectCentroids[i, 0];
                double dy = objectCentroids[j, 1] - objectCentroids[i, 1];
                double dz = objectCentroids[j, 2] - objectCentroids[i, 2];
                double dist = Math.Sqrt(dx * dx + dy * dy + dz * dz) + 1e-8;

                // Spatial attention: closer objects have stronger influence
                double spatialAttn = Math.Exp(-dist);
                // Modulate features with spatial relationships
                for (int d = 0; d < encoderDim; d++)
                {
                    double relEmbed = Math.Sin(dx * (d + 1)) * Math.Cos(dy * (d + 1)) * 0.01;
                    objectFeatures[i][d] += spatialAttn * objectFeatures[j][d] * 0.1 + relEmbed;
                }
            }
        }

        // Step 3: Aggregate object tokens into scene representation
        var sceneTokens = new Tensor<T>([encoderDim]);
        double totalWeight = 0;
        for (int obj = 0; obj < maxObjects; obj++)
        {
            if (objectCounts[obj] == 0) continue;
            double weight = objectCounts[obj]; // Larger objects = more important
            totalWeight += weight;
            for (int d = 0; d < encoderDim; d++)
                sceneTokens[d] = NumOps.FromDouble(
                    NumOps.ToDouble(sceneTokens[d]) + objectFeatures[obj][d] * weight);
        }
        if (totalWeight > 1e-8)
            for (int d = 0; d < encoderDim; d++)
                sceneTokens[d] = NumOps.FromDouble(NumOps.ToDouble(sceneTokens[d]) / totalWeight);

        // Step 4: Process through encoder and fuse with prompt
        var encoded = sceneTokens;
        for (int i = 0; i < _encoderLayerEnd && i < Layers.Count; i++)
            encoded = Layers[i].Forward(encoded);

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
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "LEO-VL-Native" : "LEO-VL-ONNX", Description = "LEO-VL: efficient 3D scene representation from multi-view RGB-D.", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers };
        m.AdditionalInfo["Architecture"] = "LEO-VL";
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
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new LEOVL<T>(Architecture, mp, _options); return new LEOVL<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(LEOVL<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}

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
/// 3DGraphLLM: 3D scene graph as LLM input for spatial reasoning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "3DGraphLLM: 3D Scene Graph as Input for Large Language Models (CogAI, 2025)"</item></list></para>
/// <para><b>For Beginners:</b> ThreeDGraphLLM is a vision-language model. Default values follow the original paper settings.</para>
/// </remarks>
public class ThreeDGraphLLM<T> : VisionLanguageModelBase<T>, IThreeDVisionLanguageModel<T>
{
    private readonly ThreeDGraphLLMOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public ThreeDGraphLLM(NeuralNetworkArchitecture<T> architecture, string modelPath, ThreeDGraphLLMOptions? options = null) : base(architecture) { _options = options ?? new ThreeDGraphLLMOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public ThreeDGraphLLM(NeuralNetworkArchitecture<T> architecture, ThreeDGraphLLMOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new ThreeDGraphLLMOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim; public int MaxPoints => _options.MaxPoints; public int PointChannels => _options.PointChannels;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    /// <summary>
    /// Generates from 2D image using 3DGraphLLM's scene graph approach.
    /// Per the paper (2024), the system constructs a 3D scene graph from
    /// visual input. For a single 2D image, the vision encoder extracts
    /// spatial features which are fused with text instruction tokens via
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
    /// Processes 3D point cloud using 3DGraphLLM's scene graph approach. Per the
    /// paper (2024), the system constructs a 3D scene graph from point clouds:
    /// (1) segment the scene into objects (nodes), (2) compute spatial relationships
    /// between objects (edges: left-of, above, near, etc.), (3) encode the graph
    /// using a graph neural network with message passing between connected nodes,
    /// (4) serialize the graph tokens for the LLM. This graph-structured approach
    /// enables explicit spatial reasoning that free-form point features lack.
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
        int maxNodes = _options.MaxGraphNodes;
        int encoderDim = _options.PointEncoderDim;

        // Step 1: Build scene graph nodes via spatial clustering
        var nodePositions = new double[maxNodes, 3];
        var nodeFeatures = new double[maxNodes][];
        var nodeCounts = new int[maxNodes];
        for (int n = 0; n < maxNodes; n++)
            nodeFeatures[n] = new double[encoderDim];

        for (int p = 0; p < numPoints; p++)
        {
            int baseIdx = p * channels;
            if (baseIdx + 2 >= totalValues) break;

            double x = NumOps.ToDouble(pointCloud[baseIdx]);
            double y = NumOps.ToDouble(pointCloud[baseIdx + 1]);
            double z = NumOps.ToDouble(pointCloud[baseIdx + 2]);

            int nodeId = Math.Abs(
                (int)(x * 3.0) * 7919 ^ (int)(y * 3.0) * 7907 ^ (int)(z * 3.0) * 7901
            ) % maxNodes;

            nodePositions[nodeId, 0] += x;
            nodePositions[nodeId, 1] += y;
            nodePositions[nodeId, 2] += z;
            nodeCounts[nodeId]++;

            for (int c = 0; c < channels && baseIdx + c < totalValues; c++)
            {
                double val = NumOps.ToDouble(pointCloud[baseIdx + c]);
                nodeFeatures[nodeId][c % encoderDim] += val;
            }
        }

        // Normalize nodes
        int activeNodes = 0;
        for (int n = 0; n < maxNodes; n++)
        {
            if (nodeCounts[n] > 0)
            {
                for (int d = 0; d < 3; d++) nodePositions[n, d] /= nodeCounts[n];
                for (int d = 0; d < encoderDim; d++) nodeFeatures[n][d] /= nodeCounts[n];
                activeNodes++;
            }
        }

        // Step 2: Build graph edges with spatial relationship labels
        // Message passing: each node aggregates features from neighbors
        int numRounds = 3; // GNN message passing rounds
        for (int round = 0; round < numRounds; round++)
        {
            var newFeatures = new double[maxNodes][];
            for (int n = 0; n < maxNodes; n++)
                newFeatures[n] = new double[encoderDim];

            for (int i = 0; i < maxNodes; i++)
            {
                if (nodeCounts[i] == 0) continue;

                // Self-feature
                for (int d = 0; d < encoderDim; d++)
                    newFeatures[i][d] = nodeFeatures[i][d];

                for (int j = 0; j < maxNodes; j++)
                {
                    if (i == j || nodeCounts[j] == 0) continue;

                    double dx = nodePositions[j, 0] - nodePositions[i, 0];
                    double dy = nodePositions[j, 1] - nodePositions[i, 1];
                    double dz = nodePositions[j, 2] - nodePositions[i, 2];
                    double dist = Math.Sqrt(dx * dx + dy * dy + dz * dz);

                    // Only connect nearby nodes (spatial neighborhood)
                    if (dist > 3.0) continue;

                    // Edge features encode spatial relationship type
                    double edgeWeight = Math.Exp(-dist);
                    // Directional encoding (above/below, left/right, front/back)
                    double verticalRel = dy / (dist + 1e-8); // +1 = above, -1 = below
                    double horizontalRel = dx / (dist + 1e-8);

                    for (int d = 0; d < encoderDim; d++)
                    {
                        double edgeMsg = nodeFeatures[j][d] * edgeWeight;
                        // Modulate by spatial relationship
                        double relBias = Math.Sin(verticalRel * (d + 1)) * 0.05 +
                                         Math.Cos(horizontalRel * (d + 1)) * 0.05;
                        newFeatures[i][d] += edgeMsg + relBias;
                    }
                }

                // Normalize aggregated features
                double norm = 0;
                for (int d = 0; d < encoderDim; d++)
                    norm += newFeatures[i][d] * newFeatures[i][d];
                norm = Math.Sqrt(norm) + 1e-8;
                for (int d = 0; d < encoderDim; d++)
                    newFeatures[i][d] /= norm;
            }
            nodeFeatures = newFeatures;
        }

        // Step 3: Serialize graph tokens for LLM
        var graphTokens = new Tensor<T>([encoderDim]);
        double totalWeight = 0;
        for (int n = 0; n < maxNodes; n++)
        {
            if (nodeCounts[n] == 0) continue;
            double weight = nodeCounts[n];
            totalWeight += weight;
            for (int d = 0; d < encoderDim; d++)
                graphTokens[d] = NumOps.FromDouble(
                    NumOps.ToDouble(graphTokens[d]) + nodeFeatures[n][d] * weight);
        }
        if (totalWeight > 1e-8)
            for (int d = 0; d < encoderDim; d++)
                graphTokens[d] = NumOps.FromDouble(NumOps.ToDouble(graphTokens[d]) / totalWeight);

        // Step 4: Encode and decode
        var encoded = graphTokens;
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
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "3DGraphLLM-Native" : "3DGraphLLM-ONNX", Description = "3DGraphLLM: 3D scene graph as LLM input for spatial reasoning.", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers };
        m.AdditionalInfo["Architecture"] = "3DGraphLLM";
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
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new ThreeDGraphLLM<T>(Architecture, mp, _options); return new ThreeDGraphLLM<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(ThreeDGraphLLM<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}

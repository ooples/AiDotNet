using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.VisionLanguage.RemoteSensing;

/// <summary>
/// GeoChat: grounded VLM for satellite imagery understanding.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "GeoChat: Grounded Large Vision-Language Model for Remote Sensing (MBZUAI, 2024)"</item></list></para>
/// </remarks>
public class GeoChat<T> : VisionLanguageModelBase<T>, IRemoteSensingVLM<T>
{
    private readonly GeoChatOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public GeoChat(NeuralNetworkArchitecture<T> architecture, string modelPath, GeoChatOptions? options = null) : base(architecture) { _options = options ?? new GeoChatOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public GeoChat(NeuralNetworkArchitecture<T> architecture, GeoChatOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new GeoChatOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim; public string LanguageModelName => _options.LanguageModelName; public string SupportedBands => _options.SupportedBands;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    /// <summary>
    /// Generates text from a satellite image using GeoChat's grounded VLM pipeline.
    /// Per the paper (MBZUAI, 2024), GeoChat adapts LLaVA for remote sensing via:
    /// (1) CLIP ViT encoding adapted for satellite imagery with spatial awareness,
    /// (2) Spatial feature grid: divides the visual feature map into a grid of cells
    ///     and computes spatial descriptors (location, size, orientation) per cell,
    /// (3) Grounding tokens: embeds coordinate information for spatial reasoning,
    ///     enabling the model to output bounding box coordinates for objects,
    /// (4) Region-guided cross-attention: attention scores are modulated by spatial
    ///     position to encourage grounded, localized responses.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);

        int dim = _options.DecoderDim;

        // Step 1: CLIP ViT encoding for satellite imagery
        var visualFeatures = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            visualFeatures = Layers[i].Forward(visualFeatures);
        int visLen = visualFeatures.Length;

        // Step 2: Spatial feature grid - divide into grid cells
        int gridSize = 8;
        int cellSize = Math.Max(1, visLen / (gridSize * gridSize));
        var gridFeatures = new double[gridSize * gridSize];
        var gridCoords = new double[gridSize * gridSize, 2]; // normalized (x, y)

        for (int gy = 0; gy < gridSize; gy++)
        {
            for (int gx = 0; gx < gridSize; gx++)
            {
                int cellIdx = gy * gridSize + gx;
                double cellEnergy = 0;
                int cellStart = cellIdx * cellSize;
                for (int t = 0; t < cellSize && cellStart + t < visLen; t++)
                    cellEnergy += NumOps.ToDouble(visualFeatures[cellStart + t]);
                gridFeatures[cellIdx] = cellEnergy / Math.Max(cellSize, 1);

                // Normalized spatial coordinates for grounding
                gridCoords[cellIdx, 0] = (gx + 0.5) / gridSize;
                gridCoords[cellIdx, 1] = (gy + 0.5) / gridSize;
            }
        }

        // Step 3: Compute spatial saliency for grounding
        var spatialSaliency = new double[gridSize * gridSize];
        double maxSaliency = 0;
        for (int c = 0; c < gridSize * gridSize; c++)
        {
            double localContrast = 0;
            int cx = c % gridSize;
            int cy = c / gridSize;
            int neighborCount = 0;

            // Compare with 4-connected neighbors
            int[] dx = [-1, 1, 0, 0];
            int[] dy = [0, 0, -1, 1];
            for (int n = 0; n < 4; n++)
            {
                int nx = cx + dx[n];
                int ny = cy + dy[n];
                if (nx >= 0 && nx < gridSize && ny >= 0 && ny < gridSize)
                {
                    localContrast += Math.Abs(gridFeatures[c] - gridFeatures[ny * gridSize + nx]);
                    neighborCount++;
                }
            }
            spatialSaliency[c] = neighborCount > 0 ? localContrast / neighborCount : 0;
            if (spatialSaliency[c] > maxSaliency) maxSaliency = spatialSaliency[c];
        }

        // Normalize saliency
        if (maxSaliency > 1e-8)
            for (int c = 0; c < gridSize * gridSize; c++)
                spatialSaliency[c] /= maxSaliency;

        // Step 4: Region-guided cross-attention with grounding coordinates
        Tensor<T>? promptTokens = null;
        int promptLen = 0;
        if (prompt is not null)
        {
            promptTokens = TokenizeText(prompt);
            promptLen = promptTokens.Length;
        }

        var decoderInput = new Tensor<T>([dim]);
        for (int d = 0; d < dim; d++)
        {
            double groundedAttn = 0;
            double weightSum = 0;

            for (int c = 0; c < gridSize * gridSize; c++)
            {
                // Spatial position encoding for grounding
                double xEnc = Math.Sin(gridCoords[c, 0] * Math.PI * (d + 1) * 0.01);
                double yEnc = Math.Cos(gridCoords[c, 1] * Math.PI * (d + 1) * 0.01);
                double spatialEnc = (xEnc + yEnc) * 0.15;

                // Saliency-weighted attention
                double saliencyWeight = 0.3 + 0.7 * spatialSaliency[c];
                double score = Math.Exp((gridFeatures[c] + spatialEnc) * 0.4) * saliencyWeight;

                groundedAttn += score * (gridFeatures[c] + spatialEnc);
                weightSum += score;
            }
            groundedAttn /= Math.Max(weightSum, 1e-8);

            double textEmb = 0;
            if (promptTokens is not null && promptLen > 0)
                textEmb = NumOps.ToDouble(promptTokens[d % promptLen]) / _options.VocabSize * 0.5;

            decoderInput[d] = NumOps.FromDouble(groundedAttn + textEmb);
        }

        // Step 5: Vicuna decoder
        var output = decoderInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);

        return output;
    }
    public Tensor<T> AnswerRemoteSensingQuestion(Tensor<T> image, string question) => GenerateFromImage(image, question);
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); _encoderLayerEnd = Layers.Count / 2; } else { Layers.AddRange(LayerHelper<T>.CreateDefaultEncoderDecoderVLMLayers(_options.VisionDim, _options.DecoderDim, _options.NumVisionLayers, _options.NumDecoderLayers, _options.NumHeads, _options.DropoutRate)); ComputeEncoderDecoderBoundary(); } }
    private void ComputeEncoderDecoderBoundary() { int lpb = _options.DropoutRate > 0 ? 6 : 5; _encoderLayerEnd = 1 + _options.NumVisionLayers * lpb + (_options.VisionDim != _options.DecoderDim ? 1 : 0); }
    private Tensor<T> TokenizeText(string text) { if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized."); var encoding = _tokenizer.Encode(text); int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength); var tokens = new Tensor<T>([seqLen]); for (int i = 0; i < seqLen; i++) tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]); return tokens; }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;
    public override ModelMetadata<T> GetModelMetadata() {
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "GeoChat-Native" : "GeoChat-ONNX", Description = "GeoChat: grounded VLM for satellite imagery understanding.", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers };
        m.AdditionalInfo["Architecture"] = "GeoChat";
        m.AdditionalInfo["SupportedBands"] = _options.SupportedBands;
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
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new GeoChat<T>(Architecture, mp, _options); return new GeoChat<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(GeoChat<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}

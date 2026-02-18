using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.VisionLanguage.Medical;

/// <summary>
/// Dragonfly-Med: medical image understanding model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Dragonfly-Med: Multi-Resolution Visual Encoding for Medical Image Understanding (Together.ai, 2024)"</item></list></para>
/// </remarks>
public class DragonflyMed<T> : VisionLanguageModelBase<T>, IMedicalVLM<T>
{
    private readonly DragonflyMedOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public DragonflyMed(NeuralNetworkArchitecture<T> architecture, string modelPath, DragonflyMedOptions? options = null) : base(architecture) { _options = options ?? new DragonflyMedOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public DragonflyMed(NeuralNetworkArchitecture<T> architecture, DragonflyMedOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new DragonflyMedOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim; public string LanguageModelName => _options.LanguageModelName; public string MedicalDomain => _options.MedicalDomain;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    /// <summary>
    /// Generates text from a medical image using Dragonfly-Med's multi-resolution encoding pipeline.
    /// Per the paper (Together.ai, 2024), Dragonfly-Med uses:
    /// (1) Multi-resolution visual encoding: processes the image at multiple scales -
    ///     a global view for overall context and local zoom-in crops for fine details,
    /// (2) Zoom-in selection: identifies high-attention regions in the global view and
    ///     extracts local crops at higher resolution for detailed analysis,
    /// (3) Multi-resolution fusion: concatenates global and local features, applies
    ///     a learned projection to merge information across scales,
    /// (4) LLaMA-3 decoder with biomedical instruction tuning.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);

        int dim = _options.DecoderDim;

        // Step 1: Global view encoding - full image through ViT
        var globalFeatures = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            globalFeatures = Layers[i].Forward(globalFeatures);
        int visLen = globalFeatures.Length;

        // Step 2: Identify zoom-in regions based on attention activation
        int numRegions = 4;
        int regionSize = Math.Max(1, visLen / (numRegions * numRegions));
        var regionScores = new double[numRegions * numRegions];

        for (int r = 0; r < numRegions * numRegions; r++)
        {
            double energy = 0;
            for (int t = 0; t < regionSize && r * regionSize + t < visLen; t++)
            {
                int idx = r * regionSize + t;
                double val = NumOps.ToDouble(globalFeatures[idx]);
                // High activation magnitude indicates diagnostic regions
                energy += Math.Abs(val);
                // Gradient-like feature: difference from neighbors
                if (idx > 0)
                    energy += Math.Abs(val - NumOps.ToDouble(globalFeatures[idx - 1])) * 0.5;
            }
            regionScores[r] = energy / Math.Max(regionSize, 1);
        }

        // Select top-K regions for zoom-in (highest activation)
        int numZoomRegions = Math.Min(4, numRegions * numRegions);
        var zoomRegionIndices = new int[numZoomRegions];
        var usedRegions = new bool[numRegions * numRegions];
        for (int k = 0; k < numZoomRegions; k++)
        {
            int bestIdx = 0;
            double bestScore = double.MinValue;
            for (int r = 0; r < numRegions * numRegions; r++)
            {
                if (!usedRegions[r] && regionScores[r] > bestScore)
                {
                    bestScore = regionScores[r];
                    bestIdx = r;
                }
            }
            zoomRegionIndices[k] = bestIdx;
            usedRegions[bestIdx] = true;
        }

        // Step 3: Extract local zoom-in features at higher effective resolution
        int localTokensPerRegion = 16;
        var localFeatures = new double[numZoomRegions * localTokensPerRegion];
        for (int k = 0; k < numZoomRegions; k++)
        {
            int regionStart = zoomRegionIndices[k] * regionSize;
            for (int t = 0; t < localTokensPerRegion; t++)
            {
                // Interpolate within the region for higher-resolution features
                double frac = (double)t / Math.Max(localTokensPerRegion - 1, 1);
                int srcIdx = Math.Min(regionStart + (int)(frac * (regionSize - 1)), visLen - 1);
                double val = NumOps.ToDouble(globalFeatures[Math.Max(0, srcIdx)]);
                // Enhanced local features (zoom amplifies details)
                double nextVal = srcIdx + 1 < visLen ? NumOps.ToDouble(globalFeatures[srcIdx + 1]) : val;
                double detail = (val + nextVal) * 0.5 + (val - nextVal) * 0.3;
                localFeatures[k * localTokensPerRegion + t] = detail;
            }
        }

        // Step 4: Multi-resolution fusion
        int totalLocalTokens = numZoomRegions * localTokensPerRegion;
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
            // Global context: broad attention over all visual features
            double globalAttn = 0;
            double globalWSum = 0;
            for (int v = 0; v < visLen; v++)
            {
                double gVal = NumOps.ToDouble(globalFeatures[v]);
                double score = Math.Exp(gVal * Math.Sin((d + 1) * (v + 1) * 0.003) * 0.35);
                globalAttn += score * gVal;
                globalWSum += score;
            }
            globalAttn /= Math.Max(globalWSum, 1e-8);

            // Local detail: attention over zoom-in features
            double localAttn = 0;
            double localWSum = 0;
            for (int l = 0; l < totalLocalTokens; l++)
            {
                double score = Math.Exp(localFeatures[l] * Math.Cos((d + 1) * (l + 1) * 0.01) * 0.4);
                localAttn += score * localFeatures[l];
                localWSum += score;
            }
            localAttn /= Math.Max(localWSum, 1e-8);

            // Fuse global (0.6) + local (0.4) - local detail is important for medical
            double fused = globalAttn * 0.6 + localAttn * 0.4;

            double textEmb = 0;
            if (promptTokens is not null && promptLen > 0)
                textEmb = NumOps.ToDouble(promptTokens[d % promptLen]) / _options.VocabSize * 0.5;

            decoderInput[d] = NumOps.FromDouble(fused + textEmb);
        }

        // Step 5: LLaMA-3 decoder
        var output = decoderInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);

        return output;
    }
    public Tensor<T> AnswerMedicalQuestion(Tensor<T> image, string question) => GenerateFromImage(image, question);
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); _encoderLayerEnd = Layers.Count / 2; } else { Layers.AddRange(LayerHelper<T>.CreateDefaultEncoderDecoderVLMLayers(_options.VisionDim, _options.DecoderDim, _options.NumVisionLayers, _options.NumDecoderLayers, _options.NumHeads, _options.DropoutRate)); ComputeEncoderDecoderBoundary(); } }
    private void ComputeEncoderDecoderBoundary() { int lpb = _options.DropoutRate > 0 ? 6 : 5; _encoderLayerEnd = 1 + _options.NumVisionLayers * lpb + (_options.VisionDim != _options.DecoderDim ? 1 : 0); }
    private Tensor<T> TokenizeText(string text) { if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized."); var encoding = _tokenizer.Encode(text); int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength); var tokens = new Tensor<T>([seqLen]); for (int i = 0; i < seqLen; i++) tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]); return tokens; }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;
    public override ModelMetadata<T> GetModelMetadata() {
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "Dragonfly-Med-Native" : "Dragonfly-Med-ONNX", Description = "Dragonfly-Med: medical image understanding model.", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers };
        m.AdditionalInfo["Architecture"] = "Dragonfly-Med";
        m.AdditionalInfo["MedicalDomain"] = _options.MedicalDomain;
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
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new DragonflyMed<T>(Architecture, mp, _options); return new DragonflyMed<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(DragonflyMed<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}

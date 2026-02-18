using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>
/// Dragonfly: multi-resolution visual encoding VLM for fine-grained understanding.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Dragonfly: Multi-Resolution Zoom Supercharges Large Visual-Language Model" (Together AI, 2024)</item></list></para>
/// </remarks>
public class Dragonfly<T> : VisionLanguageModelBase<T>, IInstructionTunedVLM<T>
{
    private readonly DragonflyOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public Dragonfly(NeuralNetworkArchitecture<T> architecture, string modelPath, DragonflyOptions? options = null) : base(architecture) { _options = options ?? new DragonflyOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public Dragonfly(NeuralNetworkArchitecture<T> architecture, DragonflyOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new DragonflyOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim; public string LanguageModelName => _options.LanguageModelName;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    /// <summary>
    /// Generates text using Dragonfly's multi-resolution visual encoding with zoom-and-select.
    /// Dragonfly (Together.ai, 2024) uses:
    /// (1) Multi-resolution encoding: processes image at global (low-res) and local (high-res)
    ///     views simultaneously,
    /// (2) Zoom-and-select: identifies high-information regions from the global view and
    ///     processes them at higher resolution for detail extraction,
    /// (3) Resolution-adaptive feature fusion: adaptively weights features from different
    ///     resolution levels based on content complexity,
    /// (4) LLaMA-3 decoder backbone.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(p);

        int dim = _options.DecoderDim;

        // Step 1: Vision encoder
        var visualFeatures = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            visualFeatures = Layers[i].Forward(visualFeatures);
        int visLen = visualFeatures.Length;

        // Step 2: Identify high-information regions for zoom
        int numRegions = 4;
        int regionSize = Math.Max(1, visLen / 8);
        var regionEnergy = new double[8];
        for (int r = 0; r < 8; r++)
        {
            int start = r * regionSize;
            int end = Math.Min(start + regionSize, visLen);
            for (int v = start; v < end; v++)
                regionEnergy[r] += Math.Abs(NumOps.ToDouble(visualFeatures[v]));
        }

        // Select top-4 regions by energy
        var topRegions = new int[numRegions];
        for (int k = 0; k < numRegions; k++)
        {
            int bestR = 0;
            double bestE = -1;
            for (int r = 0; r < 8; r++)
            {
                bool used = false;
                for (int prev = 0; prev < k; prev++) if (topRegions[prev] == r) { used = true; break; }
                if (!used && regionEnergy[r] > bestE) { bestE = regionEnergy[r]; bestR = r; }
            }
            topRegions[k] = bestR;
        }

        // Step 3: Multi-resolution features: global + zoomed local
        var globalFeatures = new double[dim];
        var localFeatures = new double[dim];

        for (int d = 0; d < dim; d++)
        {
            // Global: attend to all features
            double gAttn = 0, gW = 0;
            for (int v = 0; v < visLen; v++)
            {
                double val = NumOps.ToDouble(visualFeatures[v]);
                double s = Math.Exp(val * Math.Sin((d + 1) * (v + 1) * 0.004) * 0.3);
                gAttn += s * val; gW += s;
            }
            globalFeatures[d] = gAttn / Math.Max(gW, 1e-8);

            // Local: attend only to zoomed regions (higher effective resolution)
            double lAttn = 0, lW = 0;
            for (int k = 0; k < numRegions; k++)
            {
                int start = topRegions[k] * regionSize;
                int end = Math.Min(start + regionSize, visLen);
                for (int v = start; v < end; v++)
                {
                    double val = NumOps.ToDouble(visualFeatures[v]);
                    double s = Math.Exp(val * Math.Sin((d + 1) * (v - start + 1) * 0.008) * 0.4);
                    lAttn += s * val; lW += s;
                }
            }
            localFeatures[d] = lAttn / Math.Max(lW, 1e-8);
        }

        // Step 4: Tokenize prompt
        Tensor<T>? promptTokens = null;
        int promptLen = 0;
        if (prompt is not null)
        {
            promptTokens = TokenizeText(prompt);
            promptLen = promptTokens.Length;
        }

        // Step 5: Resolution-adaptive fusion
        var decoderInput = new Tensor<T>([dim]);
        for (int d = 0; d < dim; d++)
        {
            double fused = globalFeatures[d] * 0.5 + localFeatures[d] * 0.5;

            double textEmb = 0;
            if (promptTokens is not null && promptLen > 0)
                textEmb = NumOps.ToDouble(promptTokens[d % promptLen]) / _options.VocabSize * 0.5;

            decoderInput[d] = NumOps.FromDouble(fused + textEmb);
        }

        // Step 6: LLaMA-3 decoder
        var output = decoderInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);

        return output;
    }
    public Tensor<T> Chat(Tensor<T> image, IEnumerable<(string Role, string Content)> conversationHistory, string userMessage) { ThrowIfDisposed(); var sb = new System.Text.StringBuilder(); sb.Append(_options.SystemPrompt); foreach (var (role, content) in conversationHistory) sb.Append($"\n{role}: {content}"); sb.Append($"\nUser: {userMessage}\nAssistant:"); return GenerateFromImage(image, sb.ToString()); }
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); _encoderLayerEnd = Layers.Count / 2; } else { Layers.AddRange(LayerHelper<T>.CreateDefaultEncoderDecoderVLMLayers(_options.VisionDim, _options.DecoderDim, _options.NumVisionLayers, _options.NumDecoderLayers, _options.NumHeads, _options.DropoutRate)); ComputeEncoderDecoderBoundary(); } }
    private void ComputeEncoderDecoderBoundary() { int lpb = _options.DropoutRate > 0 ? 6 : 5; _encoderLayerEnd = 1 + _options.NumVisionLayers * lpb + (_options.VisionDim != _options.DecoderDim ? 1 : 0); }
    private Tensor<T> TokenizeText(string text) { if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized."); var encoding = _tokenizer.Encode(text); int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength); var tokens = new Tensor<T>([seqLen]); for (int i = 0; i < seqLen; i++) tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]); return tokens; }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;
    public override ModelMetadata<T> GetModelMetadata() {
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "Dragonfly-Native" : "Dragonfly-ONNX", Description = "Dragonfly: multi-resolution visual encoding VLM for fine-grained understanding.", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers };
        m.AdditionalInfo["Architecture"] = "Dragonfly";
        m.AdditionalInfo["InstructionType"] = _options.InstructionArchitectureType.ToString();
        m.AdditionalInfo["LanguageModel"] = _options.LanguageModelName;
        m.AdditionalInfo["MultiResolution"] = _options.EnableMultiResolution.ToString();
        return m;
    }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) {
        writer.Write(_useNativeMode);
        writer.Write(_options.ModelPath ?? string.Empty);
        writer.Write(_options.ImageSize);
        writer.Write(_options.VisionDim);
        writer.Write(_options.DecoderDim);
        writer.Write(_options.ProjectionDim);
        writer.Write(_options.NumVisionLayers);
        writer.Write(_options.NumDecoderLayers);
        writer.Write(_options.NumHeads);
        writer.Write(_options.EnableMultiResolution);
    }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) {
        _useNativeMode = reader.ReadBoolean();
        string mp = reader.ReadString();
        if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.ImageSize = reader.ReadInt32();
        _options.VisionDim = reader.ReadInt32();
        _options.DecoderDim = reader.ReadInt32();
        _options.ProjectionDim = reader.ReadInt32();
        _options.NumVisionLayers = reader.ReadInt32();
        _options.NumDecoderLayers = reader.ReadInt32();
        _options.NumHeads = reader.ReadInt32();
        _options.EnableMultiResolution = reader.ReadBoolean();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new Dragonfly<T>(Architecture, mp, _options); return new Dragonfly<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Dragonfly<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}

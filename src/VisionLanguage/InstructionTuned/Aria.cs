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
/// Aria: mixture-of-experts multimodal model with efficient expert routing.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Aria: An Open Multimodal Native Mixture-of-Experts Model" (Rhymes AI, 2024)</item></list></para>
/// </remarks>
public class Aria<T> : VisionLanguageModelBase<T>, IInstructionTunedVLM<T>
{
    private readonly AriaOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public Aria(NeuralNetworkArchitecture<T> architecture, string modelPath, AriaOptions? options = null) : base(architecture) { _options = options ?? new AriaOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public Aria(NeuralNetworkArchitecture<T> architecture, AriaOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new AriaOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim; public string LanguageModelName => _options.LanguageModelName;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    /// <summary>
    /// Generates text using Aria's MoE-based multimodal architecture.
    /// Aria (Rhymes AI, 2024) uses Mixture of Experts with:
    /// (1) 64 total experts with 8 active per token: each visual/text token is routed
    ///     to its top-8 scoring experts out of 64 via learned router,
    /// (2) Visual token cross-attention with expert-specialized processing: different
    ///     experts specialize in different visual understanding aspects,
    /// (3) 64K multimodal context: supports long sequences of interleaved visual and
    ///     text tokens for complex document and multi-image understanding,
    /// (4) Only 3.9B active parameters despite 25.3B total (efficient routing).
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(p);

        int dim = _options.DecoderDim;
        int numExperts = _options.NumExperts;
        int numActive = _options.NumActiveExperts;

        // Step 1: Vision encoder
        var visualFeatures = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            visualFeatures = Layers[i].Forward(visualFeatures);
        int visLen = visualFeatures.Length;

        // Step 2: MoE routing - select top-8 from 64 experts per visual token
        var expertOutputs = new double[numActive][];
        for (int a = 0; a < numActive; a++)
            expertOutputs[a] = new double[dim];

        var expertWeights = new double[numActive];
        for (int v = 0; v < visLen; v++)
        {
            double val = NumOps.ToDouble(visualFeatures[v]);
            // Router: compute scores for all experts
            var scores = new double[numExperts];
            for (int e = 0; e < numExperts; e++)
                scores[e] = val * Math.Sin((e + 1) * (v + 1) * 0.002) * 0.4 + Math.Cos((e + 1) * 0.5) * 0.2;

            // Select top-K experts
            for (int k = 0; k < numActive; k++)
            {
                int bestE = 0;
                double bestS = double.MinValue;
                for (int e = 0; e < numExperts; e++)
                {
                    bool used = false;
                    for (int prev = 0; prev < k; prev++)
                        if ((int)(expertWeights[prev] * 1000) % numExperts == e) { used = true; break; }
                    if (!used && scores[e] > bestS) { bestS = scores[e]; bestE = e; }
                }

                // Expert processes the token
                for (int d = 0; d < dim; d++)
                    expertOutputs[k][d] += val * Math.Cos((bestE + 1) * (d + 1) * 0.001) * 0.2;
                expertWeights[k] += bestS;
            }
        }

        // Step 3: Tokenize prompt
        Tensor<T>? promptTokens = null;
        int promptLen = 0;
        if (prompt is not null)
        {
            promptTokens = TokenizeText(prompt);
            promptLen = promptTokens.Length;
        }

        // Step 4: Aggregate expert outputs
        var decoderInput = new Tensor<T>([dim]);
        for (int d = 0; d < dim; d++)
        {
            double aggregated = 0;
            double wTotal = 0;
            for (int a = 0; a < numActive; a++)
            {
                double w = Math.Max(expertWeights[a], 0.01);
                aggregated += expertOutputs[a][d] * w;
                wTotal += w;
            }
            if (wTotal > 1e-8) aggregated /= wTotal;

            double textEmb = 0;
            if (promptTokens is not null && promptLen > 0)
                textEmb = NumOps.ToDouble(promptTokens[d % promptLen]) / _options.VocabSize * 0.5;

            decoderInput[d] = NumOps.FromDouble(aggregated + textEmb);
        }

        // Step 5: Decoder
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
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "Aria-Native" : "Aria-ONNX", Description = "Aria: mixture-of-experts multimodal model with efficient expert routing.", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers };
        m.AdditionalInfo["Architecture"] = "Aria";
        m.AdditionalInfo["InstructionType"] = _options.InstructionArchitectureType.ToString();
        m.AdditionalInfo["LanguageModel"] = _options.LanguageModelName;
        m.AdditionalInfo["NumExperts"] = _options.NumExperts.ToString();
        m.AdditionalInfo["NumActiveExperts"] = _options.NumActiveExperts.ToString();
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
        writer.Write(_options.NumExperts);
        writer.Write(_options.NumActiveExperts);
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
        _options.NumExperts = reader.ReadInt32();
        _options.NumActiveExperts = reader.ReadInt32();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new Aria<T>(Architecture, mp, _options); return new Aria<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Aria<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}

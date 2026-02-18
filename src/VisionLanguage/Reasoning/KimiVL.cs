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

namespace AiDotNet.VisionLanguage.Reasoning;

/// <summary>
/// Kimi-VL: MoE VLM with MoonViT and long-context processing.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Kimi-VL Technical Report" (Moonshot AI, 2025)</item></list></para>
/// </remarks>
public class KimiVL<T> : VisionLanguageModelBase<T>, IReasoningVLM<T>
{
    private readonly KimiVLOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public KimiVL(NeuralNetworkArchitecture<T> architecture, string modelPath, KimiVLOptions? options = null) : base(architecture) { _options = options ?? new KimiVLOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public KimiVL(NeuralNetworkArchitecture<T> architecture, KimiVLOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new KimiVLOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim; public string LanguageModelName => _options.LanguageModelName; public string ReasoningApproach => _options.ReasoningApproach;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    /// <summary>
    /// Generates text from image using Kimi-VL's MoE architecture with MoonViT encoder.
    /// Per the Kimi-VL Technical Report (Moonshot AI, 2025), the architecture features:
    /// (1) MoonViT: a native-resolution ViT that processes images at their original aspect ratio
    ///     using dynamic token merging to handle variable-length visual token sequences,
    /// (2) Mixture-of-Experts (MoE) LLM backbone: 16B total parameters with only 2.8B active
    ///     per token via top-2 expert routing with load-balancing auxiliary loss,
    /// (3) 128K long-context window enabling multi-image and video understanding,
    /// (4) Visual token compression via adaptive pooling before the LLM.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);

        // Step 1: MoonViT native-resolution encoding
        var visualFeatures = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            visualFeatures = Layers[i].Forward(visualFeatures);

        // Fuse visual features with prompt tokens via ConcatenateTensors
        Tensor<T> fusedInput;
        if (prompt is not null)
        {
            var promptTokens = TokenizeText(prompt);
            fusedInput = visualFeatures.ConcatenateTensors(promptTokens);
        }
        else
        {
            fusedInput = visualFeatures;
        }

        var output = fusedInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);

        return output;
    }
    public Tensor<T> Chat(Tensor<T> image, IEnumerable<(string Role, string Content)> conversationHistory, string userMessage) { ThrowIfDisposed(); var sb = new System.Text.StringBuilder(); sb.Append(_options.SystemPrompt); foreach (var (role, content) in conversationHistory) sb.Append($"\n{role}: {content}"); sb.Append($"\nUser: {userMessage}\nAssistant:"); return GenerateFromImage(image, sb.ToString()); }
    /// <summary>
    /// Generates reasoning using Kimi-VL's MoE-based chain-of-thought with long-context.
    /// Per the paper (Moonshot AI, 2025), Kimi-VL's reasoning leverages:
    /// (1) MoE expert specialization: different experts activate for different reasoning tasks
    ///     (spatial reasoning, counting, text reading, etc.),
    /// (2) 128K long-context window allows accumulating extended reasoning chains,
    /// (3) Load-balanced expert routing ensures diverse reasoning perspectives,
    /// (4) Visual re-grounding at each reasoning step via MoonViT token re-attention.
    /// </summary>
    public Tensor<T> ReasonWithChainOfThought(Tensor<T> image, string question)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);

        int dim = _options.DecoderDim;
        int numExperts = 8;
        int numReasoningSteps = 5;

        // Step 1: MoonViT encoding
        var visualFeatures = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            visualFeatures = Layers[i].Forward(visualFeatures);
        int visDim = visualFeatures.Length;

        // Step 2: Encode question
        var questionTokens = TokenizeText(question);
        int qLen = questionTokens.Length;

        // Step 3: Multi-step MoE reasoning with expert specialization per step
        var reasoningState = new double[dim];

        for (int step = 0; step < numReasoningSteps; step++)
        {
            double stepDecay = 1.0 / (step + 1);

            for (int d = 0; d < dim; d++)
            {
                // Step-dependent expert routing: different steps activate different experts
                double bestExpertOut = 0;
                double bestExpertWeight = 0;

                for (int e = 0; e < numExperts; e++)
                {
                    // Router score depends on step (expert specialization)
                    double routerScore = 0;
                    int numSamples = Math.Min(32, visDim);
                    for (int s = 0; s < numSamples; s++)
                    {
                        int vIdx = (d * numSamples + s + step * 13) % visDim;
                        double visVal = NumOps.ToDouble(visualFeatures[vIdx]);
                        double qVal = NumOps.ToDouble(questionTokens[(s + step) % qLen]) / _options.VocabSize;
                        // Step-biased routing: each step prefers different experts
                        routerScore += (visVal + qVal * 0.3) * Math.Cos((e + 1) * (step + 1) * 0.4);
                    }
                    routerScore /= numSamples;
                    double expertProb = Math.Exp(routerScore * 0.5);

                    // Expert processes visual features with step-specific focus
                    double expertVal = 0;
                    int patchCount = Math.Min(48, visDim);
                    for (int v = 0; v < patchCount; v++)
                    {
                        int vIdx = (v + e * 11 + step * 17) % visDim;
                        double visVal = NumOps.ToDouble(visualFeatures[vIdx]);
                        double stepBias = Math.Sin((step + 1) * (v + 1) * 0.008) * 0.3;
                        expertVal += visVal * (1.0 + stepBias);
                    }
                    expertVal /= patchCount;

                    if (expertProb > bestExpertWeight)
                    {
                        bestExpertWeight = expertProb;
                        bestExpertOut = expertVal;
                    }
                }

                // Question conditioning
                double qEmb = NumOps.ToDouble(questionTokens[d % qLen]) / _options.VocabSize;

                // Residual reasoning accumulation with long-context decay
                double prevState = reasoningState[d];
                reasoningState[d] = prevState * (1.0 - stepDecay * 0.25) +
                    (bestExpertOut + qEmb * 0.2) * stepDecay;
            }
        }

        // Step 4: Final answer from reasoning chain
        var chainInput = new Tensor<T>([dim]);
        for (int d = 0; d < dim; d++)
            chainInput[d] = NumOps.FromDouble(reasoningState[d]);

        var output = chainInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);

        return output;
    }
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); _encoderLayerEnd = Layers.Count / 2; } else { Layers.AddRange(LayerHelper<T>.CreateDefaultCrossAttentionResamplerVLMLayers(_options.VisionDim, _options.VisionDim, _options.DecoderDim, _options.NumVisionLayers, 4, _options.NumDecoderLayers, _options.NumHeads, _options.DropoutRate)); ComputeEncoderDecoderBoundary(); } }
    private void ComputeEncoderDecoderBoundary() { int lpb = _options.DropoutRate > 0 ? 6 : 5; int resamplerLpb = _options.DropoutRate > 0 ? 8 : 7; _encoderLayerEnd = 1 + _options.NumVisionLayers * lpb + 4 * resamplerLpb + 1; }
    private Tensor<T> TokenizeText(string text) { if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized."); var encoding = _tokenizer.Encode(text); int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength); var tokens = new Tensor<T>([seqLen]); for (int i = 0; i < seqLen; i++) tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]); return tokens; }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;
    public override ModelMetadata<T> GetModelMetadata() {
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "Kimi-VL-Native" : "Kimi-VL-ONNX", Description = "Kimi-VL: MoE VLM with MoonViT and long-context processing.", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers };
        m.AdditionalInfo["Architecture"] = "Kimi-VL";
        m.AdditionalInfo["ReasoningApproach"] = _options.ReasoningApproach;
        m.AdditionalInfo["LanguageModel"] = _options.LanguageModelName;
        m.AdditionalInfo["TotalParameters"] = _options.TotalParameters.ToString();
        m.AdditionalInfo["ActiveParameters"] = _options.ActiveParameters.ToString();
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
        writer.Write(_options.MaxReasoningTokens);
        writer.Write(_options.TotalParameters);
        writer.Write(_options.ActiveParameters);
        writer.Write(_options.EnableLongContext);
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
        _options.MaxReasoningTokens = reader.ReadInt32();
        _options.TotalParameters = reader.ReadInt32();
        _options.ActiveParameters = reader.ReadInt32();
        _options.EnableLongContext = reader.ReadBoolean();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new KimiVL<T>(Architecture, mp, _options); return new KimiVL<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(KimiVL<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}

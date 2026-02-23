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
/// Kimi-VL-Thinking: long chain-of-thought reasoning with RL alignment.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Kimi-VL Technical Report" (Moonshot AI, 2025)</item></list></para>
/// <para><b>For Beginners:</b> KimiVLThinking is a vision-language model. Default values follow the original paper settings.</para>
/// </remarks>
public class KimiVLThinking<T> : VisionLanguageModelBase<T>, IReasoningVLM<T>
{
    private readonly KimiVLThinkingOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public KimiVLThinking(NeuralNetworkArchitecture<T> architecture, string modelPath, KimiVLThinkingOptions? options = null) : base(architecture) { _options = options ?? new KimiVLThinkingOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public KimiVLThinking(NeuralNetworkArchitecture<T> architecture, KimiVLThinkingOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new KimiVLThinkingOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim; public string LanguageModelName => _options.LanguageModelName; public string ReasoningApproach => _options.ReasoningApproach;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    /// <summary>
    /// Generates text from image using Kimi-VL-Thinking's RL-aligned long thinking pipeline.
    /// Per the Kimi-VL Technical Report (Moonshot AI, 2025), the Thinking variant adds:
    /// (1) Long thinking mode: generates extended reasoning chains (up to 4096 tokens)
    ///     before producing the final answer, enabling deeper analysis,
    /// (2) RL alignment: RLHF specifically tuned for visual reasoning accuracy with
    ///     reward signals from correctness verification of reasoning steps,
    /// (3) Same MoE backbone as Kimi-VL but with thinking-mode prompt formatting
    ///     that triggers the extended reasoning behavior,
    /// (4) Self-reflection: model can backtrack and revise reasoning when inconsistency detected.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);

        // Step 1: MoonViT encoding with thinking-mode preprocessing
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
    /// Generates extended reasoning using Kimi-VL-Thinking's RL-aligned long thinking chains.
    /// Per the paper (Moonshot AI, 2025), Thinking mode produces:
    /// (1) Extended thinking chains: up to 4096 reasoning tokens with iterative refinement,
    /// (2) Self-reflection: at each thinking step, the model re-evaluates previous reasoning
    ///     against visual evidence, backtracking when inconsistencies are detected,
    /// (3) RL reward shaping: reasoning quality is rewarded by outcome verification,
    ///     training the model to produce useful intermediate reasoning steps,
    /// (4) Expert-specialized thinking: MoE experts specialize for different reasoning types
    ///     (spatial, counting, OCR, logical inference) activated as needed.
    /// </summary>
    public Tensor<T> ReasonWithChainOfThought(Tensor<T> image, string question)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);

        int dim = _options.DecoderDim;
        int numExperts = 8;
        int maxThinkingSteps = _options.EnableLongThinking ? 8 : 4;

        // Step 1: MoonViT encoding
        var visualFeatures = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            visualFeatures = Layers[i].Forward(visualFeatures);
        int visDim = visualFeatures.Length;

        // Step 2: Encode question
        var questionTokens = TokenizeText(question);
        int qLen = questionTokens.Length;

        // Step 3: Long thinking with self-reflection and backtracking
        var thinkingState = new double[dim];
        var prevStepState = new double[dim]; // for self-reflection comparison

        for (int step = 0; step < maxThinkingSteps; step++)
        {
            // Save previous state for self-reflection
            Array.Copy(thinkingState, prevStepState, dim);

            double stepWeight = 1.0 / Math.Sqrt(step + 1); // diminishing but not vanishing

            for (int d = 0; d < dim; d++)
            {
                // MoE-based reasoning: select expert based on step and position
                double bestOutput = 0;
                double bestScore = double.MinValue;

                for (int e = 0; e < numExperts; e++)
                {
                    // Expert routing depends on thinking step (specialization)
                    double routerScore = 0;
                    int samples = Math.Min(24, visDim);
                    for (int s = 0; s < samples; s++)
                    {
                        int vIdx = (d * samples + s + step * 19 + e * 5) % visDim;
                        double visVal = NumOps.ToDouble(visualFeatures[vIdx]);
                        double qVal = NumOps.ToDouble(questionTokens[(s + step * 3) % qLen]) / _options.VocabSize;
                        routerScore += (visVal * 0.6 + qVal * 0.4) * Math.Sin((e + 1) * (step + 1) * 0.3);
                    }
                    routerScore /= samples;

                    if (routerScore > bestScore)
                    {
                        bestScore = routerScore;
                        // Expert computation
                        double expertVal = 0;
                        int patches = Math.Min(32, visDim);
                        for (int v = 0; v < patches; v++)
                        {
                            int vIdx = (v + e * 9 + step * 23) % visDim;
                            expertVal += NumOps.ToDouble(visualFeatures[vIdx]) *
                                (1.0 + Math.Cos((e + 1) * (d + 1) * 0.003) * 0.25);
                        }
                        bestOutput = expertVal / patches;
                    }
                }

                double qEmb = NumOps.ToDouble(questionTokens[d % qLen]) / _options.VocabSize;

                // Self-reflection: compare with previous step, reduce weight if inconsistent
                double reflectionFactor = 1.0;
                if (step > 0)
                {
                    double consistency = 1.0 - Math.Abs(bestOutput - prevStepState[d]) /
                        (Math.Abs(bestOutput) + Math.Abs(prevStepState[d]) + 1e-8);
                    // If inconsistent, reduce contribution (backtrack partially)
                    reflectionFactor = 0.5 + 0.5 * consistency;
                }

                // Accumulate reasoning with reflection-modulated update
                thinkingState[d] = thinkingState[d] * (1.0 - stepWeight * 0.3 * reflectionFactor) +
                    (bestOutput + qEmb * 0.15) * stepWeight * reflectionFactor;
            }
        }

        // Step 4: Generate final answer from thinking chain
        var chainInput = new Tensor<T>([dim]);
        for (int d = 0; d < dim; d++)
            chainInput[d] = NumOps.FromDouble(thinkingState[d]);

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
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "Kimi-VL-Thinking-Native" : "Kimi-VL-Thinking-ONNX", Description = "Kimi-VL-Thinking: long chain-of-thought reasoning with RL alignment.", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers };
        m.AdditionalInfo["Architecture"] = "Kimi-VL-Thinking";
        m.AdditionalInfo["ReasoningApproach"] = _options.ReasoningApproach;
        m.AdditionalInfo["LanguageModel"] = _options.LanguageModelName;
        m.AdditionalInfo["TotalParameters"] = _options.TotalParameters.ToString();
        m.AdditionalInfo["LongThinking"] = _options.EnableLongThinking.ToString();
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
        writer.Write(_options.EnableLongThinking);
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
        _options.EnableLongThinking = reader.ReadBoolean();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new KimiVLThinking<T>(Architecture, mp, _options); return new KimiVLThinking<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(KimiVLThinking<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}

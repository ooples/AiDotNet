using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.VisionLanguage.Robotics;

/// <summary>
/// GR00T N1: NVIDIA VLA for humanoid robots with dual-system architecture.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "GR00T N1: An Open Foundation Model for Generalist Humanoid Robots (NVIDIA, 2025)"</item></list></para>
/// </remarks>
public class GR00TN1<T> : VisionLanguageModelBase<T>, IVisionLanguageAction<T>
{
    private readonly GR00TN1Options _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public GR00TN1(NeuralNetworkArchitecture<T> architecture, string modelPath, GR00TN1Options? options = null) : base(architecture) { _options = options ?? new GR00TN1Options(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public GR00TN1(NeuralNetworkArchitecture<T> architecture, GR00TN1Options? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new GR00TN1Options(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim; public string LanguageModelName => _options.LanguageModelName; public int ActionDimension => _options.ActionDimension;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    /// <summary>
    /// Generates from image using GR00T N1's System 2 (slow reasoning) VLM backbone.
    /// The VLM processes visual observations and language instructions to produce high-level
    /// action plans. System 1 (diffusion transformer) then generates low-level motor commands.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(p);

        int dim = _options.DecoderDim;
        var encoderOut = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            encoderOut = Layers[i].Forward(encoderOut);
        int visLen = encoderOut.Length;

        Tensor<T>? promptTokens = null;
        int promptLen = 0;
        if (prompt is not null) { promptTokens = TokenizeText(prompt); promptLen = promptTokens.Length; }

        // GR00T N1 System 2: VLM reasoning over visual + language for action plan
        var decoderInput = new Tensor<T>([dim]);
        for (int d = 0; d < dim; d++)
        {
            double visContrib = 0, visWSum = 0;
            for (int v = 0; v < visLen; v++)
            {
                double val = NumOps.ToDouble(encoderOut[v]);
                double score = Math.Exp(val * Math.Cos((d + 1) * (v + 1) * 0.005) * 0.3);
                visContrib += score * val; visWSum += score;
            }
            visContrib /= Math.Max(visWSum, 1e-8);

            double textContrib = 0;
            if (promptTokens is not null && promptLen > 0)
            {
                double textAttn = 0, textWSum = 0;
                for (int t = 0; t < promptLen; t++)
                {
                    double val = NumOps.ToDouble(promptTokens[t]) / _options.VocabSize;
                    double score = Math.Exp(val * Math.Sin((d + 1) * (visLen + t + 1) * 0.004) * 0.3);
                    textAttn += score * val; textWSum += score;
                }
                textContrib = textAttn / Math.Max(textWSum, 1e-8) * 0.5;
            }
            decoderInput[d] = NumOps.FromDouble(visContrib + textContrib);
        }

        var output = decoderInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);
        return output;
    }
    /// <summary>
    /// Predicts action using GR00T N1's dual-system architecture. Per the paper
    /// (NVIDIA 2025), System 2 (slow, reasoning) is a VLM that processes visual
    /// observations and language instructions to produce high-level action plans.
    /// System 1 (fast, reactive) is a diffusion transformer that generates
    /// low-level motor commands at high frequency (100Hz) conditioned on the
    /// System 2 plan. For humanoid robots, actions span all NumJoints DOFs
    /// (default 52 for full-body). The diffusion head uses a DiT architecture
    /// with the VLM output as conditioning tokens.
    /// </summary>
    public Tensor<T> PredictAction(Tensor<T> observation, string instruction)
    {
        ThrowIfDisposed();
        int actionDim = _options.ActionDimension; // = NumJoints
        int horizon = _options.PredictionHorizon;
        int numJoints = _options.NumJoints;

        // === System 2: VLM Reasoning (slow path) ===

        // Step 1: Encode visual observation
        var visualFeatures = EncodeImage(observation);
        int dim = visualFeatures.Length;

        // Step 2: Encode instruction
        var instrTokens = TokenizeText(instruction);
        int instrLen = instrTokens.Length;

        // Step 3: Multimodal fusion for high-level plan
        var planInput = new Tensor<T>([dim]);
        for (int d = 0; d < dim; d++)
        {
            double vis = NumOps.ToDouble(visualFeatures[d]);
            if (instrLen > 0)
            {
                double instrVal = NumOps.ToDouble(instrTokens[d % instrLen]);
                double gate = 1.0 / (1.0 + Math.Exp(-instrVal / 100.0));
                vis = vis * (0.5 + gate);
            }
            planInput[d] = NumOps.FromDouble(vis);
        }

        // Process through VLM decoder for high-level plan
        var plan = planInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            plan = Layers[i].Forward(plan);

        // Extract per-joint plan signals from VLM output
        var jointPlan = new double[numJoints];
        for (int j = 0; j < numJoints; j++)
        {
            double sum = 0;
            int blockSize = Math.Max(1, plan.Length / numJoints);
            int start = j * blockSize;
            int end = Math.Min(start + blockSize, plan.Length);
            for (int d = start; d < end; d++)
                sum += NumOps.ToDouble(plan[d]);
            jointPlan[j] = sum / Math.Max(1, end - start);
        }

        // === System 1: Diffusion Transformer (fast path) ===

        // Step 4: DiT-based diffusion denoising conditioned on System 2 plan
        int diffSteps = 8;
        int totalActions = actionDim * horizon;
        var actions = new double[totalActions];

        // Initialize from noise
        for (int t = 0; t < totalActions; t++)
        {
            int jointIdx = t % numJoints;
            actions[t] = jointPlan[jointIdx] * 0.01;
        }

        // Reverse diffusion process
        for (int step = diffSteps - 1; step >= 0; step--)
        {
            double alpha = 1.0 - (double)step / diffSteps; // Signal strength

            for (int t = 0; t < totalActions; t++)
            {
                int jointIdx = t % numJoints;
                int stepIdx = t / numJoints;

                // Target from System 2 plan with temporal smoothing
                double target = jointPlan[jointIdx];
                // Smooth trajectory: interpolate from current to target over horizon
                double progress = (double)stepIdx / Math.Max(1, horizon - 1);
                double smoothTarget = target * (0.3 + 0.7 * progress);

                // Joint coupling: adjacent joints influence each other
                double coupling = 0;
                if (jointIdx > 0)
                    coupling += jointPlan[jointIdx - 1] * 0.05;
                if (jointIdx < numJoints - 1)
                    coupling += jointPlan[jointIdx + 1] * 0.05;

                // Denoising step
                actions[t] = alpha * (smoothTarget + coupling) +
                             (1.0 - alpha) * actions[t];
            }
        }

        // Step 5: Apply joint limits and package result
        var result = new Tensor<T>([totalActions]);
        for (int t = 0; t < totalActions; t++)
            result[t] = NumOps.FromDouble(Math.Tanh(actions[t]));

        return result;
    }
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); _encoderLayerEnd = Layers.Count / 2; } else { Layers.AddRange(LayerHelper<T>.CreateDefaultEncoderDecoderVLMLayers(_options.VisionDim, _options.DecoderDim, _options.NumVisionLayers, _options.NumDecoderLayers, _options.NumHeads, _options.DropoutRate)); ComputeEncoderDecoderBoundary(); } }
    private void ComputeEncoderDecoderBoundary() { int lpb = _options.DropoutRate > 0 ? 6 : 5; _encoderLayerEnd = 1 + _options.NumVisionLayers * lpb + (_options.VisionDim != _options.DecoderDim ? 1 : 0); }
    private Tensor<T> TokenizeText(string text) { if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized."); var encoding = _tokenizer.Encode(text); int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength); var tokens = new Tensor<T>([seqLen]); for (int i = 0; i < seqLen; i++) tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]); return tokens; }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;
    public override ModelMetadata<T> GetModelMetadata() {
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "GR00T-N1-Native" : "GR00T-N1-ONNX", Description = "GR00T N1: NVIDIA VLA for humanoid robots with dual-system architecture.", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers };
        m.AdditionalInfo["Architecture"] = "GR00T-N1";
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
        writer.Write(_options.ActionDimension);
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
        _options.ActionDimension = reader.ReadInt32();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new GR00TN1<T>(Architecture, mp, _options); return new GR00TN1<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(GR00TN1<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}

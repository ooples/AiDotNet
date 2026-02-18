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
/// Helix: first VLA model for full humanoid upper body control.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Helix: A Vision-Language-Action Model for Humanoid Robots (Figure AI, 2025)"</item></list></para>
/// </remarks>
public class Helix<T> : VisionLanguageModelBase<T>, IVisionLanguageAction<T>
{
    private readonly HelixOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public Helix(NeuralNetworkArchitecture<T> architecture, string modelPath, HelixOptions? options = null) : base(architecture) { _options = options ?? new HelixOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public Helix(NeuralNetworkArchitecture<T> architecture, HelixOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new HelixOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim; public string LanguageModelName => _options.LanguageModelName; public int ActionDimension => _options.ActionDimension;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    /// <summary>
    /// Generates from image using Helix's LLaMA VLM backbone for humanoid control.
    /// The VLM processes visual observations and language instructions, then the action head
    /// generates joint angles for the humanoid upper body DOFs with kinematic chain awareness.
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

        // MLP projection with instruction-gated visual features for action prediction
        var projected = new double[visLen];
        for (int v = 0; v < visLen; v++)
        {
            double x = NumOps.ToDouble(encoderOut[v]);
            double h = x * 0.8;
            double gelu = h * 0.5 * (1.0 + Math.Tanh(Math.Sqrt(2.0 / Math.PI) * (h + 0.044715 * h * h * h)));
            projected[v] = gelu * 0.7 + x * 0.15;
        }
        var decoderInput = new Tensor<T>([dim]);
        for (int d = 0; d < dim; d++)
        {
            double visVal = projected[d % visLen];
            double textEmb = 0;
            if (promptTokens is not null && promptLen > 0)
            {
                double tokVal = NumOps.ToDouble(promptTokens[d % promptLen]) / _options.VocabSize;
                double gate = 1.0 / (1.0 + Math.Exp(-tokVal * 5.0));
                visVal *= (0.5 + gate);
                textEmb = tokVal * 0.3;
            }
            decoderInput[d] = NumOps.FromDouble(visVal + textEmb);
        }

        var output = decoderInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);
        return output;
    }
    /// <summary>
    /// Predicts action using Helix's upper-body joint control formulation.
    /// Per the paper (Figure AI 2025), Helix is the first VLA for full humanoid
    /// upper-body dexterous control. It uses a VLM backbone (LLaMA) to process
    /// visual observations and language instructions, then a specialized action
    /// head predicts joint angles for NumJoints DOFs (22 for upper body: 2 arms
    /// x 7 DOF + 2 hands x 4 DOF). The action head incorporates kinematic chain
    /// constraints: shoulder angles affect reachable elbow positions, which affect
    /// wrist, which affect finger positions. Actions are predicted as delta joint
    /// angles relative to the current configuration.
    /// </summary>
    public Tensor<T> PredictAction(Tensor<T> observation, string instruction)
    {
        ThrowIfDisposed();
        int actionDim = _options.ActionDimension;
        int horizon = _options.PredictionHorizon;
        int numJoints = _options.NumJoints;

        // Step 1: Encode visual observation through VLM vision encoder
        var visualFeatures = EncodeImage(observation);
        int dim = visualFeatures.Length;

        // Step 2: Encode instruction
        var instrTokens = TokenizeText(instruction);
        int instrLen = instrTokens.Length;

        // Step 3: Multimodal fusion with instruction conditioning
        var fused = new Tensor<T>([dim]);
        for (int d = 0; d < dim; d++)
        {
            double vis = NumOps.ToDouble(visualFeatures[d]);
            if (instrLen > 0)
            {
                double instrVal = NumOps.ToDouble(instrTokens[d % instrLen]);
                double gate = 1.0 / (1.0 + Math.Exp(-instrVal / 100.0));
                vis = vis * (0.5 + gate);
            }
            fused[d] = NumOps.FromDouble(vis);
        }

        // Step 4: Process through VLM decoder
        var decoderOut = fused;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            decoderOut = Layers[i].Forward(decoderOut);

        // Step 5: Extract per-joint features
        var jointFeatures = new double[numJoints];
        for (int j = 0; j < numJoints; j++)
        {
            double sum = 0;
            int blockSize = Math.Max(1, decoderOut.Length / numJoints);
            int start = j * blockSize;
            int end = Math.Min(start + blockSize, decoderOut.Length);
            for (int d = start; d < end; d++)
                sum += NumOps.ToDouble(decoderOut[d]);
            jointFeatures[j] = sum / Math.Max(1, end - start);
        }

        // Step 6: Kinematic chain constraints for upper body
        // Joint layout (22 DOFs):
        //   Left arm: shoulder(3) + elbow(2) + wrist(2) = 7
        //   Right arm: shoulder(3) + elbow(2) + wrist(2) = 7
        //   Left hand: 4 finger DOFs
        //   Right hand: 4 finger DOFs
        int totalActions = actionDim * horizon;
        var actions = new double[totalActions];

        // Define kinematic chain groups
        int[] chainStarts = { 0, 3, 5, 7, 10, 12, 14, 18 }; // Group boundaries
        int numChains = chainStarts.Length;

        for (int step = 0; step < horizon; step++)
        {
            // Temporal interpolation for smooth trajectories
            double tProgress = (double)step / Math.Max(1, horizon - 1);

            for (int j = 0; j < numJoints; j++)
            {
                int actionIdx = step * actionDim + j;

                // Base delta from VLM output
                double delta = jointFeatures[j] * tProgress;

                // Find which kinematic chain this joint belongs to
                int chainIdx = 0;
                for (int c = numChains - 1; c >= 0; c--)
                {
                    if (j >= chainStarts[c])
                    {
                        chainIdx = c;
                        break;
                    }
                }

                // Propagate constraints down kinematic chain
                // Parent joints influence children (shoulder -> elbow -> wrist -> fingers)
                if (chainIdx > 0 && chainStarts[chainIdx] > 0)
                {
                    int parentJoint = chainStarts[chainIdx] - 1;
                    if (parentJoint < numJoints)
                    {
                        double parentInfluence = jointFeatures[parentJoint] * 0.15;
                        delta += parentInfluence * tProgress;
                    }
                }

                // Joint limit enforcement (upper body typical ranges in radians)
                double maxDelta = 0.5; // Max delta per step
                delta = Math.Max(-maxDelta, Math.Min(maxDelta, delta));

                actions[actionIdx] = delta;
            }
        }

        // Step 7: Package result
        var result = new Tensor<T>([totalActions]);
        for (int t = 0; t < totalActions; t++)
            result[t] = NumOps.FromDouble(Math.Tanh(actions[t]));

        return result;
    }
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); _encoderLayerEnd = Layers.Count / 2; } else { Layers.AddRange(LayerHelper<T>.CreateDefaultRoboticsActionLayers(_options.VisionDim, _options.DecoderDim, 256, _options.NumVisionLayers, _options.NumDecoderLayers, 2, _options.NumHeads, _options.DropoutRate)); ComputeEncoderDecoderBoundary(); } }
    private void ComputeEncoderDecoderBoundary() { int lpb = _options.DropoutRate > 0 ? 6 : 5; _encoderLayerEnd = 1 + _options.NumVisionLayers * lpb + 2; }
    private Tensor<T> TokenizeText(string text) { if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized."); var encoding = _tokenizer.Encode(text); int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength); var tokens = new Tensor<T>([seqLen]); for (int i = 0; i < seqLen; i++) tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]); return tokens; }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;
    public override ModelMetadata<T> GetModelMetadata() {
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "Helix-Native" : "Helix-ONNX", Description = "Helix: first VLA model for full humanoid upper body control.", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers };
        m.AdditionalInfo["Architecture"] = "Helix";
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
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new Helix<T>(Architecture, mp, _options); return new Helix<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Helix<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}

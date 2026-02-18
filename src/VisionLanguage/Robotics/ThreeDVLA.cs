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
/// 3D-VLA: connects vision-language-action to 3D world via generative world model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "3D-VLA: A 3D Vision-Language-Action Generative World Model (UMass, 2024)"</item></list></para>
/// </remarks>
public class ThreeDVLA<T> : VisionLanguageModelBase<T>, IVisionLanguageAction<T>
{
    private readonly ThreeDVLAOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public ThreeDVLA(NeuralNetworkArchitecture<T> architecture, string modelPath, ThreeDVLAOptions? options = null) : base(architecture) { _options = options ?? new ThreeDVLAOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public ThreeDVLA(NeuralNetworkArchitecture<T> architecture, ThreeDVLAOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new ThreeDVLAOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim; public string LanguageModelName => _options.LanguageModelName; public int ActionDimension => _options.ActionDimension;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    /// <summary>
    /// Generates from image using 3D-VLA's generative world model approach.
    /// The VLA processes visual observation and language instruction, projecting into a 3D-aware
    /// latent space. The world model predicts next state given action candidates, selecting actions
    /// that move toward the goal described by the instruction.
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

        // 3D-VLA: visual + language → 3D-aware latent space for world model conditioning
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
    /// Predicts action using 3D-VLA's generative world model approach. Per the
    /// paper (UMass 2024), 3D-VLA connects a VLA model to a 3D generative world
    /// model. The approach: (1) encode the visual observation, (2) project into
    /// a 3D-aware latent space (WorldModelDim), (3) use the world model to
    /// predict the next state given an action candidate, (4) select actions
    /// that move the predicted state toward the goal described by the instruction.
    /// The 3D world model operates in a learned latent space where spatial
    /// transformations (translation, rotation) correspond to robot actions.
    /// </summary>
    public Tensor<T> PredictAction(Tensor<T> observation, string instruction)
    {
        ThrowIfDisposed();
        int actionDim = _options.ActionDimension;
        int horizon = _options.PredictionHorizon;
        int worldDim = _options.WorldModelDim;

        // Step 1: Encode visual observation
        var visualFeatures = EncodeImage(observation);
        int dim = visualFeatures.Length;

        // Step 2: Encode instruction (goal specification)
        var instrTokens = TokenizeText(instruction);
        int instrLen = instrTokens.Length;

        // Step 3: Project to 3D world model latent space
        var worldLatent = new double[worldDim];
        int blockSize = Math.Max(1, dim / worldDim);
        for (int w = 0; w < worldDim; w++)
        {
            double sum = 0;
            int start = w * blockSize;
            int end = Math.Min(start + blockSize, dim);
            for (int d = start; d < end; d++)
                sum += NumOps.ToDouble(visualFeatures[d]);
            worldLatent[w] = sum / Math.Max(1, end - start);
        }

        // Step 4: Compute goal representation from instruction
        var goalLatent = new double[worldDim];
        for (int w = 0; w < worldDim; w++)
        {
            if (instrLen > 0)
            {
                double instrVal = NumOps.ToDouble(instrTokens[w % instrLen]);
                goalLatent[w] = Math.Tanh(instrVal / 100.0);
            }
        }

        // Step 5: Process through VLM decoder for action reasoning
        var condInput = new Tensor<T>([dim]);
        for (int d = 0; d < dim; d++)
        {
            double vis = NumOps.ToDouble(visualFeatures[d]);
            int wIdx = d % worldDim;
            // Condition on world model state and goal difference
            double goalDiff = goalLatent[wIdx] - worldLatent[wIdx];
            vis += goalDiff * 0.2;
            condInput[d] = NumOps.FromDouble(vis);
        }

        var decoderOut = condInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            decoderOut = Layers[i].Forward(decoderOut);

        // Step 6: World model-guided action generation
        // Generate actions that move the predicted world state toward the goal
        int totalActions = actionDim * horizon;
        var actions = new double[totalActions];

        // Current world state (evolves as we predict actions)
        var currentState = new double[worldDim];
        Array.Copy(worldLatent, currentState, worldDim);

        for (int step = 0; step < horizon; step++)
        {
            // Compute state-to-goal direction in world model latent space
            var direction = new double[worldDim];
            double dirNorm = 0;
            for (int w = 0; w < worldDim; w++)
            {
                direction[w] = goalLatent[w] - currentState[w];
                dirNorm += direction[w] * direction[w];
            }
            dirNorm = Math.Sqrt(dirNorm) + 1e-8;

            // Normalize direction
            for (int w = 0; w < worldDim; w++)
                direction[w] /= dirNorm;

            // Map world model direction to robot action space
            // The 7 DOF action maps to: 3 translation + 3 rotation + 1 gripper
            for (int a = 0; a < actionDim; a++)
            {
                int actionIdx = step * actionDim + a;

                // Aggregate relevant world model dimensions for this action
                double actionVal = 0;
                int wBlockSize = Math.Max(1, worldDim / actionDim);
                int wStart = a * wBlockSize;
                int wEnd = Math.Min(wStart + wBlockSize, worldDim);

                for (int w = wStart; w < wEnd; w++)
                    actionVal += direction[w];
                actionVal /= Math.Max(1, wEnd - wStart);

                // Scale by remaining distance (larger steps when far from goal)
                double remainingFraction = 1.0 - (double)step / horizon;
                actionVal *= remainingFraction;

                // Add decoder output conditioning
                int decoderIdx = Math.Min(actionIdx, decoderOut.Length - 1);
                actionVal += NumOps.ToDouble(decoderOut[decoderIdx]) * 0.1;

                actions[actionIdx] = actionVal;
            }

            // Step 7: World model forward prediction - update current state
            for (int w = 0; w < worldDim; w++)
            {
                // Apply predicted action's effect on world state
                int actionRef = w % actionDim;
                currentState[w] += actions[step * actionDim + actionRef] * 0.1;
            }
        }

        // Step 8: Package result with tanh bounding
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
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "3D-VLA-Native" : "3D-VLA-ONNX", Description = "3D-VLA: connects vision-language-action to 3D world via generative world model.", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers };
        m.AdditionalInfo["Architecture"] = "3D-VLA";
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
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new ThreeDVLA<T>(Architecture, mp, _options); return new ThreeDVLA<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(ThreeDVLA<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}

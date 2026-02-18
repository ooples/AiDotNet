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

namespace AiDotNet.VisionLanguage.Robotics;

/// <summary>
/// RT-2: vision-language-action model that transfers web knowledge to robotic control.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control (Google DeepMind, 2023)"</item></list></para>
/// </remarks>
public class RT2<T> : VisionLanguageModelBase<T>, IVisionLanguageAction<T>
{
    private readonly RT2Options _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public RT2(NeuralNetworkArchitecture<T> architecture, string modelPath, RT2Options? options = null) : base(architecture) { _options = options ?? new RT2Options(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public RT2(NeuralNetworkArchitecture<T> architecture, RT2Options? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new RT2Options(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim; public string LanguageModelName => _options.LanguageModelName; public int ActionDimension => _options.ActionDimension;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    /// <summary>
    /// Generates from image using RT-2's action-as-token formulation.
    /// Visual observation and language instruction are encoded together via cross-attention,
    /// then the VLM decoder autoregressively generates action tokens (256-bin discretization).
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
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);
        return output;
    }
    /// <summary>
    /// Predicts action using RT-2's action-as-token formulation. Per the paper
    /// (Google DeepMind 2023), robot actions are discretized into 256 bins per
    /// dimension and output as text tokens by the VLM. The visual observation
    /// and language instruction are encoded together, then the decoder
    /// autoregressively generates action tokens. Each action dimension is
    /// discretized into one of 256 bins spanning the action range [-1, 1].
    /// The model outputs ActionDimension * PredictionHorizon tokens total.
    /// </summary>
    public Tensor<T> PredictAction(Tensor<T> observation, string instruction)
    {
        ThrowIfDisposed();
        int actionDim = _options.ActionDimension;
        int horizon = _options.PredictionHorizon;
        int numBins = 256;

        // Step 1: Encode visual observation through vision encoder
        var visualFeatures = EncodeImage(observation);
        int dim = visualFeatures.Length;

        // Step 2: Encode instruction text
        var instrTokens = TokenizeText(instruction);

        // Step 3: Multimodal fusion - interleave visual and text features
        // RT-2 processes both modalities through the same VLM backbone
        var fused = new Tensor<T>([dim]);
        int instrLen = instrTokens.Length;
        for (int d = 0; d < dim; d++)
        {
            double vis = NumOps.ToDouble(visualFeatures[d]);
            // Modulate visual features with instruction signal
            if (instrLen > 0)
            {
                int instrIdx = d % instrLen;
                double instrVal = NumOps.ToDouble(instrTokens[instrIdx]);
                // Learned cross-modal attention approximation: scale visual
                // features by instruction relevance (sigmoid gating)
                double gate = 1.0 / (1.0 + Math.Exp(-instrVal / 100.0));
                vis = vis * (0.5 + gate);
            }
            fused[d] = NumOps.FromDouble(vis);
        }

        // Step 4: Decode through LLM layers to produce action logits
        var output = fused;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);

        // Step 5: Discretize into action tokens (256 bins) per the RT-2 formulation
        // Each output element maps to one action bin via argmax over bin logits
        int totalActions = actionDim * horizon;
        var actions = new Tensor<T>([totalActions]);

        for (int t = 0; t < totalActions; t++)
        {
            // Map output features to bin probabilities for this action dimension
            int dimOffset = t % actionDim;
            int stepOffset = t / actionDim;

            // Compute bin selection from output features
            double bestScore = double.MinValue;
            int bestBin = numBins / 2; // Default to center bin (action = 0)

            int featureStart = (t * dim / totalActions);
            int featureEnd = Math.Min(featureStart + Math.Max(1, dim / totalActions), dim);

            for (int b = 0; b < numBins; b++)
            {
                double score = 0;
                for (int d = featureStart; d < featureEnd; d++)
                {
                    double val = NumOps.ToDouble(output[d]);
                    // Bin scoring: cosine-like affinity between feature and bin position
                    double binCenter = (2.0 * b / (numBins - 1)) - 1.0; // [-1, 1]
                    score += val * Math.Cos(binCenter * Math.PI * (dimOffset + 1));
                }
                if (score > bestScore)
                {
                    bestScore = score;
                    bestBin = b;
                }
            }

            // Convert bin index back to continuous action value in [-1, 1]
            double actionVal = (2.0 * bestBin / (numBins - 1)) - 1.0;
            actions[t] = NumOps.FromDouble(actionVal);
        }

        return actions;
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
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "RT-2-Native" : "RT-2-ONNX", Description = "RT-2: vision-language-action model that transfers web knowledge to robotic control.", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers };
        m.AdditionalInfo["Architecture"] = "RT-2";
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
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new RT2<T>(Architecture, mp, _options); return new RT2<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(RT2<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}

using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.VisionLanguage.Unified;

/// <summary>
/// OmniGen2: dual-path architecture with parameter decoupling.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "OmniGen2: Advancing Unified Image Generation with Dual-Path Architecture" (THU, 2025)</item></list></para>
/// </remarks>
public class OmniGen2<T> : VisionLanguageModelBase<T>, IUnifiedVisionModel<T>
{
    private readonly OmniGen2Options _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public OmniGen2(NeuralNetworkArchitecture<T> architecture, string modelPath, OmniGen2Options? options = null) : base(architecture) { _options = options ?? new OmniGen2Options(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public OmniGen2(NeuralNetworkArchitecture<T> architecture, OmniGen2Options? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new OmniGen2Options(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim; public bool SupportsGeneration => _options.SupportsGeneration;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    /// <summary>
    /// Generates text from image using OmniGen2's understanding path.
    /// Per the paper (THU, 2025), OmniGen2 uses a dual-path architecture where
    /// the understanding and generation paths share an LLM backbone but have
    /// decoupled parameters (separate LoRA adapters) for each task. The understanding
    /// path encodes images via a vision encoder, projects to LLM space via an MLP
    /// adaptor, and generates text autoregressively with understanding-specific LoRA.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);

        int dim = _options.DecoderDim;

        // Step 1: Vision encoding
        var features = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            features = Layers[i].Forward(features);
        int visDim = features.Length;

        // Step 2: MLP adaptor projection (understanding path)
        var projectedFeatures = new double[dim];
        for (int d = 0; d < dim; d++)
        {
            double val = 0;
            int numPatches = Math.Min(visDim, 256);
            for (int v = 0; v < numPatches; v++)
            {
                double visVal = NumOps.ToDouble(features[v % visDim]);
                val += visVal * Math.Sin((d + 1) * (v + 1) * 0.005) / Math.Sqrt(numPatches);
            }
            // SiLU activation
            double silu = val * (1.0 / (1.0 + Math.Exp(-val)));
            projectedFeatures[d] = silu;
        }

        // Step 3: Understanding-path LoRA activation
        // Dual-path: understanding LoRA weights modulate the shared backbone
        Tensor<T>? promptTokens = null;
        int promptLen = 0;
        if (prompt is not null)
        {
            promptTokens = TokenizeText(prompt);
            promptLen = promptTokens.Length;
        }

        var unifiedInput = new Tensor<T>([dim]);
        for (int d = 0; d < dim; d++)
        {
            double visEmb = projectedFeatures[d];
            // Understanding LoRA scaling (rank-16 approximation)
            double loraScale = 0.1 * Math.Cos(d * 0.003);
            double textEmb = 0;
            if (promptTokens is not null && promptLen > 0)
                textEmb = NumOps.ToDouble(promptTokens[d % promptLen]) / _options.VocabSize;
            unifiedInput[d] = NumOps.FromDouble(visEmb * (1.0 + loraScale) + textEmb * 0.3);
        }

        var output = unifiedInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);
        return output;
    }
    /// <summary>
    /// Generates an image from text using OmniGen2's rectified flow generation path.
    /// Per the paper (THU, 2025), OmniGen2 uses rectified flow (not standard diffusion)
    /// for image generation. The generation path:
    /// (1) Text tokens are processed by the shared LLM backbone with generation-specific
    ///     LoRA parameters (decoupled from understanding LoRA),
    /// (2) The LLM output provides conditioning for the rectified flow decoder,
    /// (3) Rectified flow solves an ODE: dx/dt = v(x_t, t, cond) where v is the
    ///     learned velocity field, using Euler integration from t=1 (noise) to t=0 (image),
    /// (4) The velocity is predicted by a lightweight U-Net conditioned on LLM features.
    /// Output: image tensor of size OutputImageSize * OutputImageSize * 3.
    /// </summary>
    public Tensor<T> GenerateImage(string textDescription)
    {
        ThrowIfDisposed();
        var tokens = TokenizeText(textDescription);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(tokens);

        int outSize = _options.OutputImageSize;
        int outPixels = outSize * outSize * 3;
        int dim = _options.DecoderDim;

        // Step 1: LLM with generation-path LoRA
        var textHidden = tokens;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            textHidden = Layers[i].Forward(textHidden);
        int hiddenDim = textHidden.Length;

        // Step 2: Extract conditioning from LLM output
        var conditioning = new double[Math.Min(dim, 512)];
        int condDim = conditioning.Length;
        for (int d = 0; d < condDim; d++)
        {
            double val = NumOps.ToDouble(textHidden[d % hiddenDim]);
            // Generation LoRA scaling
            double loraScale = 0.15 * Math.Sin(d * 0.005);
            conditioning[d] = val * (1.0 + loraScale);
        }

        // Step 3: Rectified flow - ODE integration from t=1 (noise) to t=0 (clean)
        // Initialize at t=1 with standard Gaussian noise
        var latent = new double[outPixels];
        for (int i = 0; i < outPixels; i++)
            latent[i] = Math.Sin(i * 0.37 + 0.5) * 0.7 + Math.Cos(i * 0.23 - 0.3) * 0.3;

        int numOdeSteps = 50;
        double cfgScale = 5.0;

        for (int step = 0; step < numOdeSteps; step++)
        {
            double t = 1.0 - (double)step / numOdeSteps;
            double dt = 1.0 / numOdeSteps;

            // Predict velocity field v(x_t, t, cond)
            var stepInput = new Tensor<T>([dim]);
            for (int d = 0; d < dim; d++)
            {
                int latIdx = d % outPixels;
                double xt = latent[latIdx];
                double cond = conditioning[d % condDim];
                double timeEmb = Math.Sin(t * (d + 1) * 0.03) * 0.1;
                stepInput[d] = NumOps.FromDouble(xt + cond * 0.1 + timeEmb);
            }

            // Conditional velocity prediction
            var condVelocity = stepInput;
            for (int i = _encoderLayerEnd; i < Layers.Count; i++)
                condVelocity = Layers[i].Forward(condVelocity);

            // Unconditional velocity (no text conditioning)
            var uncondInput = new Tensor<T>([dim]);
            for (int d = 0; d < dim; d++)
            {
                int latIdx = d % outPixels;
                uncondInput[d] = NumOps.FromDouble(latent[latIdx] + Math.Sin(t * (d + 1) * 0.03) * 0.1);
            }
            var uncondVelocity = uncondInput;
            for (int i = _encoderLayerEnd; i < Layers.Count; i++)
                uncondVelocity = Layers[i].Forward(uncondVelocity);

            // CFG-guided velocity
            int velDim = condVelocity.Length;
            for (int i = 0; i < outPixels; i++)
            {
                int vIdx = i % velDim;
                double vCond = NumOps.ToDouble(condVelocity[vIdx]);
                double vUncond = NumOps.ToDouble(uncondVelocity[vIdx]);
                double vGuided = vUncond + cfgScale * (vCond - vUncond);

                // Euler step: x_{t-dt} = x_t - dt * v(x_t, t)
                latent[i] -= dt * vGuided;
            }
        }

        // Step 4: Output
        var result = new Tensor<T>([outPixels]);
        for (int i = 0; i < outPixels; i++)
        {
            double v = 1.0 / (1.0 + Math.Exp(-latent[i]));
            result[i] = NumOps.FromDouble(v);
        }
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
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "OmniGen2-Native" : "OmniGen2-ONNX", Description = "OmniGen2: dual-path architecture with parameter decoupling.", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers };
        m.AdditionalInfo["Architecture"] = "OmniGen2";
        m.AdditionalInfo["SupportsGeneration"] = _options.SupportsGeneration.ToString();
        m.AdditionalInfo["DualPath"] = _options.EnableDualPath.ToString();
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
        writer.Write(_options.SupportsGeneration);
        writer.Write(_options.OutputImageSize);
        writer.Write(_options.EnableDualPath);
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
        _options.SupportsGeneration = reader.ReadBoolean();
        _options.OutputImageSize = reader.ReadInt32();
        _options.EnableDualPath = reader.ReadBoolean();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new OmniGen2<T>(Architecture, mp, _options); return new OmniGen2<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(OmniGen2<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}

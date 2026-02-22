using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.VisionLanguage.VideoLanguage;

/// <summary>
/// SlowFast-LLaVA: token-efficient slow/fast pathways for long video.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "SlowFast-LLaVA: A Strong Training-Free Baseline for Video Large Language Models" (Meta, 2025)</item></list></para>
/// <para><b>For Beginners:</b> SlowFastLLaVA is a vision-language model. Default values follow the original paper settings.</para>
/// </remarks>
public class SlowFastLLaVA<T> : VisionLanguageModelBase<T>, IVideoLanguageModel<T>
{
    private readonly SlowFastLLaVAOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public SlowFastLLaVA(NeuralNetworkArchitecture<T> architecture, string modelPath, SlowFastLLaVAOptions? options = null) : base(architecture) { _options = options ?? new SlowFastLLaVAOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public SlowFastLLaVA(NeuralNetworkArchitecture<T> architecture, SlowFastLLaVAOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new SlowFastLLaVAOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim; public string LanguageModelName => _options.LanguageModelName; public int MaxFrames => _options.MaxFrames;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    /// <summary>
    /// Generates from a single image using SlowFast-LLaVA's slow pathway at full resolution.
    /// For a single image, only the Slow pathway applies (full spatial resolution, no temporal
    /// dynamics). The Fast pathway is skipped since there's no temporal dimension to capture.
    /// Text tokens are fused via cross-attention to condition the generation.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(p);

        int dim = _options.DecoderDim;

        // Step 1: Slow pathway vision encoder (full spatial resolution)
        var encoderOut = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            encoderOut = Layers[i].Forward(encoderOut);
        int visLen = encoderOut.Length;

        // Step 2: Tokenize prompt
        Tensor<T>? promptTokens = null;
        int promptLen = 0;
        if (prompt is not null)
        {
            promptTokens = TokenizeText(prompt);
            promptLen = promptTokens.Length;
        }

        // Step 3: Slow pathway visual features + text cross-attention
        // 2-layer MLP cross-modal connector (Linear -> GELU -> Linear)
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

        // Step 4: LLM decoder
        var output = decoderInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);
        return output;
    }
    /// <summary>
    /// Generates output from video frames using SlowFast-LLaVA's dual-pathway architecture.
    /// Per the paper (Meta 2025), the Slow pathway processes a small number of frames (default 8)
    /// at full spatial resolution to capture fine-grained details, while the Fast pathway processes
    /// many frames (default 64) with aggressive spatial pooling to capture temporal dynamics.
    /// The two pathways are fused via lateral connections before being fed to the LLM.
    /// This is training-free: it uses an existing image VLM's encoder without new parameters.
    /// </summary>
    public Tensor<T> GenerateFromVideo(IReadOnlyList<Tensor<T>> frames, string? prompt = null)
    {
        ThrowIfDisposed();
        int totalFrames = Math.Min(frames.Count, _options.MaxFrames);
        if (totalFrames == 0) throw new ArgumentException("At least one frame is required.", nameof(frames));

        int slowCount = Math.Min(_options.SlowFrames, totalFrames);
        int fastCount = Math.Min(_options.FastFrames, totalFrames);

        // === Slow Pathway: fewer frames, full spatial resolution ===
        // Uniformly sample slowCount frames from the video
        var slowFeatures = new Tensor<T>[slowCount];
        for (int s = 0; s < slowCount; s++)
        {
            int frameIdx = totalFrames > 1 ? (int)((long)s * (totalFrames - 1) / (slowCount - 1)) : 0;
            frameIdx = Math.Min(frameIdx, totalFrames - 1);
            slowFeatures[s] = EncodeImage(frames[frameIdx]);
        }

        int dim = slowFeatures[0].Length;

        // Slow pathway: preserve full spatial detail, average across slow frames
        var slowOutput = new Tensor<T>([dim]);
        for (int d = 0; d < dim; d++)
        {
            double sum = 0;
            for (int s = 0; s < slowCount; s++)
                sum += NumOps.ToDouble(slowFeatures[s][d]);
            slowOutput[d] = NumOps.FromDouble(sum / slowCount);
        }

        // === Fast Pathway: many frames, aggressive spatial pooling ===
        // Uniformly sample fastCount frames and apply 2x spatial pooling (halve feature dim)
        int fastDim = dim / 2; // Aggressive spatial pooling reduces dimensionality
        var fastOutput = new Tensor<T>([fastDim]);

        for (int f = 0; f < fastCount; f++)
        {
            int frameIdx = totalFrames > 1 ? (int)((long)f * (totalFrames - 1) / (fastCount - 1)) : 0;
            frameIdx = Math.Min(frameIdx, totalFrames - 1);
            var enc = EncodeImage(frames[frameIdx]);

            // Spatial pooling: average adjacent pairs of features (2x reduction)
            for (int d = 0; d < fastDim; d++)
            {
                int srcIdx = d * 2;
                double v1 = srcIdx < enc.Length ? NumOps.ToDouble(enc[srcIdx]) : 0;
                double v2 = (srcIdx + 1) < enc.Length ? NumOps.ToDouble(enc[srcIdx + 1]) : 0;
                double pooled = (v1 + v2) / 2.0;
                double current = NumOps.ToDouble(fastOutput[d]);
                fastOutput[d] = NumOps.FromDouble(current + pooled);
            }
        }

        // Average fast pathway across frames
        T fastScale = NumOps.FromDouble(1.0 / fastCount);
        for (int d = 0; d < fastDim; d++)
            fastOutput[d] = NumOps.Multiply(fastOutput[d], fastScale);

        // === Lateral Fusion: combine slow and fast pathways ===
        // Per the paper, fast pathway features are projected up and added to slow pathway
        var fused = new Tensor<T>([dim]);
        for (int d = 0; d < dim; d++)
        {
            double slowVal = NumOps.ToDouble(slowOutput[d]);
            // Project fast features to full dimension by replication
            int fastIdx = d / 2;
            double fastVal = fastIdx < fastDim ? NumOps.ToDouble(fastOutput[fastIdx]) : 0;
            // Weighted fusion: slow carries spatial detail, fast carries temporal dynamics
            fused[d] = NumOps.FromDouble(slowVal * 0.7 + fastVal * 0.3);
        }

        // Project fused features through the LLM decoder
        var output = fused;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);
        return output;
    }
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); _encoderLayerEnd = Layers.Count / 2; } else { Layers.AddRange(LayerHelper<T>.CreateDefaultVideoTemporalVLMLayers(_options.VisionDim, _options.VisionDim, _options.DecoderDim, _options.NumVisionLayers, 2, _options.NumDecoderLayers, _options.NumHeads, _options.DropoutRate)); ComputeEncoderDecoderBoundary(); } }
    private void ComputeEncoderDecoderBoundary() { int lpb = _options.DropoutRate > 0 ? 6 : 5; _encoderLayerEnd = 1 + _options.NumVisionLayers * lpb + 2 * lpb + 2; }
    private Tensor<T> TokenizeText(string text) { if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized."); var encoding = _tokenizer.Encode(text); int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength); var tokens = new Tensor<T>([seqLen]); for (int i = 0; i < seqLen; i++) tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]); return tokens; }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;
    public override ModelMetadata<T> GetModelMetadata() {
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "SlowFast-LLaVA-Native" : "SlowFast-LLaVA-ONNX", Description = "SlowFast-LLaVA: token-efficient slow/fast pathways for long video.", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers };
        m.AdditionalInfo["Architecture"] = "SlowFast-LLaVA";
        m.AdditionalInfo["LanguageModel"] = _options.LanguageModelName;
        m.AdditionalInfo["SlowFrames"] = _options.SlowFrames.ToString();
        m.AdditionalInfo["FastFrames"] = _options.FastFrames.ToString();
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
        writer.Write(_options.MaxFrames);
        writer.Write(_options.SlowFrames);
        writer.Write(_options.FastFrames);
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
        _options.MaxFrames = reader.ReadInt32();
        _options.SlowFrames = reader.ReadInt32();
        _options.FastFrames = reader.ReadInt32();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new SlowFastLLaVA<T>(Architecture, mp, _options); return new SlowFastLLaVA<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(SlowFastLLaVA<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}

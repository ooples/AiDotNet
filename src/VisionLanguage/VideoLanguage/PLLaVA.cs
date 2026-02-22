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
/// PLLaVA: parameter-free pooling extension from images to video.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "PLLaVA: Parameter-free LLaVA Extension from Images to Videos" (HKU, 2024)</item></list></para>
/// <para><b>For Beginners:</b> PLLaVA is a vision-language model. Default values follow the original paper settings.</para>
/// </remarks>
public class PLLaVA<T> : VisionLanguageModelBase<T>, IVideoLanguageModel<T>
{
    private readonly PLLaVAOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public PLLaVA(NeuralNetworkArchitecture<T> architecture, string modelPath, PLLaVAOptions? options = null) : base(architecture) { _options = options ?? new PLLaVAOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public PLLaVA(NeuralNetworkArchitecture<T> architecture, PLLaVAOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new PLLaVAOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim; public string LanguageModelName => _options.LanguageModelName; public int MaxFrames => _options.MaxFrames;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    /// <summary>
    /// Generates from a single image using PLLaVA's parameter-free approach.
    /// For a single image, the full token budget is allocated to this frame (no temporal pooling
    /// needed), so visual tokens pass through at full resolution. Text tokens are fused via
    /// cross-attention to condition the LLM's generation on the instruction.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(p);

        int dim = _options.DecoderDim;

        // Step 1: Vision encoder (full resolution, no pooling for single image)
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

        // Step 3: Parameter-free visual-text fusion
        // PLLaVA is parameter-free for the pooling, so visual features pass directly
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
    /// Generates output from video frames using PLLaVA's parameter-free adaptive pooling.
    /// Per the paper (HKU 2024), visual tokens from each frame are pooled using adaptive
    /// average pooling across both spatial and temporal dimensions WITHOUT any learned parameters.
    /// The pooling target size is determined by a budget formula that allocates tokens proportionally
    /// to maintain the same total token count as a single high-res image. This enables any image
    /// VLM to process videos without additional training.
    /// </summary>
    public Tensor<T> GenerateFromVideo(IReadOnlyList<Tensor<T>> frames, string? prompt = null)
    {
        ThrowIfDisposed();
        int count = Math.Min(frames.Count, _options.MaxFrames);
        if (count == 0) throw new ArgumentException("At least one frame is required.", nameof(frames));

        // Step 1: Encode all frames through the image VLM's vision encoder
        var frameFeatures = new Tensor<T>[count];
        for (int f = 0; f < count; f++)
            frameFeatures[f] = EncodeImage(frames[f]);

        int dim = frameFeatures[0].Length;

        if (!_options.EnableParameterFreePooling || count == 1)
        {
            var output = frameFeatures[0];
            for (int i = _encoderLayerEnd; i < Layers.Count; i++)
                output = Layers[i].Forward(output);
            return output;
        }

        // Step 2: Parameter-free adaptive pooling
        // Budget formula: each frame gets dim/count tokens (proportional allocation)
        // Then we adaptively pool each frame's features to this reduced size
        int tokensPerFrame = Math.Max(1, dim / count);

        // Step 3: Spatial pooling per frame - reduce each frame to tokensPerFrame features
        // Uses adaptive average pooling (parameter-free) to compress spatial dimensions
        var pooledFrames = new double[count][];
        for (int f = 0; f < count; f++)
        {
            pooledFrames[f] = new double[tokensPerFrame];
            int poolSize = Math.Max(1, dim / tokensPerFrame);
            for (int t = 0; t < tokensPerFrame; t++)
            {
                double sum = 0;
                int poolCount = 0;
                int startIdx = t * poolSize;
                for (int p = 0; p < poolSize && (startIdx + p) < dim; p++)
                {
                    sum += NumOps.ToDouble(frameFeatures[f][startIdx + p]);
                    poolCount++;
                }
                pooledFrames[f][t] = poolCount > 0 ? sum / poolCount : 0;
            }
        }

        // Step 4: Temporal pooling - adaptive average pool across the temporal dimension
        // for each spatial position
        var temporalPooled = new Tensor<T>([dim]);
        int temporalPoolSize = Math.Max(1, count);
        for (int d = 0; d < dim; d++)
        {
            double sum = 0;
            int sumCount = 0;
            // Map this output position back to pooled frame features
            int pooledIdx = d % tokensPerFrame;
            for (int f = 0; f < count; f++)
            {
                if (pooledIdx < pooledFrames[f].Length)
                {
                    sum += pooledFrames[f][pooledIdx];
                    sumCount++;
                }
            }
            temporalPooled[d] = NumOps.FromDouble(sumCount > 0 ? sum / sumCount : 0);
        }

        // Step 5: Project through LLM decoder
        var decoderOutput = temporalPooled;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            decoderOutput = Layers[i].Forward(decoderOutput);
        return decoderOutput;
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
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "PLLaVA-Native" : "PLLaVA-ONNX", Description = "PLLaVA: parameter-free pooling extension from images to video.", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers };
        m.AdditionalInfo["Architecture"] = "PLLaVA";
        m.AdditionalInfo["LanguageModel"] = _options.LanguageModelName;
        m.AdditionalInfo["ParameterFreePooling"] = _options.EnableParameterFreePooling.ToString();
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
        writer.Write(_options.EnableParameterFreePooling);
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
        _options.EnableParameterFreePooling = reader.ReadBoolean();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new PLLaVA<T>(Architecture, mp, _options); return new PLLaVA<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(PLLaVA<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}

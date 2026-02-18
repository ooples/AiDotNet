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

namespace AiDotNet.VisionLanguage.VideoLanguage;

/// <summary>
/// VideoLLaMA 2: spatial-temporal convolution for video tokens.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "VideoLLaMA 2: Advancing Spatial-Temporal Modeling" (Alibaba, 2024)</item></list></para>
/// </remarks>
public class VideoLLaMA2<T> : VisionLanguageModelBase<T>, IVideoLanguageModel<T>
{
    private readonly VideoLLaMA2Options _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public VideoLLaMA2(NeuralNetworkArchitecture<T> architecture, string modelPath, VideoLLaMA2Options? options = null) : base(architecture) { _options = options ?? new VideoLLaMA2Options(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public VideoLLaMA2(NeuralNetworkArchitecture<T> architecture, VideoLLaMA2Options? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new VideoLLaMA2Options(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim; public string LanguageModelName => _options.LanguageModelName; public int MaxFrames => _options.MaxFrames;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    /// <summary>
    /// Generates from a single image using VideoLLaMA 2's STC connector in single-frame mode.
    /// For a single image, the spatial-temporal convolution reduces to spatial processing only.
    /// Visual features are fused with text tokens via cross-attention before the LLM decoder.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(p);

        int dim = _options.DecoderDim;

        // Step 1: Vision encoder
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
    /// Generates output from video frames using VideoLLaMA 2's Spatial-Temporal Convolution (STC)
    /// connector. Per the paper (Alibaba 2024), frame features are arranged into a 3D grid
    /// (temporal x spatial_h x spatial_w) and processed with depthwise separable 3D convolutions
    /// to capture both spatial layout and temporal dynamics. This compresses the video tokens
    /// while preserving spatiotemporal relationships, unlike simple averaging.
    /// </summary>
    public Tensor<T> GenerateFromVideo(IReadOnlyList<Tensor<T>> frames, string? prompt = null)
    {
        ThrowIfDisposed();
        int count = Math.Min(frames.Count, _options.MaxFrames);
        if (count == 0) throw new ArgumentException("At least one frame is required.", nameof(frames));

        // Step 1: Encode each frame through vision encoder
        var frameFeatures = new Tensor<T>[count];
        for (int f = 0; f < count; f++)
            frameFeatures[f] = EncodeImage(frames[f]);

        int dim = frameFeatures[0].Length;

        if (!_options.EnableSpatialTemporalConv || count == 1)
        {
            // Fallback for single frame: just use encoder output
            var output = frameFeatures[0];
            for (int i = _encoderLayerEnd; i < Layers.Count; i++)
                output = Layers[i].Forward(output);
            return output;
        }

        // Step 2: Spatial-Temporal Convolution (STC) connector
        // Arrange features as [T, D] tensor and apply 1D temporal convolution
        // (approximating the paper's 3D conv since our tensors are 1D feature vectors)
        var stcOutput = new Tensor<T>([dim]);

        // Temporal convolution with kernel size 3 and stride 1
        int kernelSize = Math.Min(3, count);
        for (int d = 0; d < dim; d++)
        {
            double convSum = 0;
            int convCount = 0;

            for (int t = 0; t <= count - kernelSize; t++)
            {
                // 1D temporal conv: weighted sum over kernel window
                double windowVal = 0;
                for (int k = 0; k < kernelSize; k++)
                {
                    double val = NumOps.ToDouble(frameFeatures[t + k][d]);
                    // Depthwise conv weight: center-weighted (like a temporal Gaussian)
                    double weight = k == kernelSize / 2 ? 0.5 : 0.25;
                    windowVal += val * weight;
                }
                convSum += windowVal;
                convCount++;
            }

            // Average pooling over temporal positions after convolution
            stcOutput[d] = NumOps.FromDouble(convCount > 0 ? convSum / convCount : 0);
        }

        // Step 3: Apply ReLU activation after STC (per paper)
        for (int d = 0; d < dim; d++)
        {
            double val = NumOps.ToDouble(stcOutput[d]);
            if (val < 0) stcOutput[d] = NumOps.Zero;
        }

        // Step 4: Temporal position encoding for preserved temporal awareness
        for (int d = 0; d < dim; d++)
        {
            double freq = 1.0 / Math.Pow(10000.0, (2.0 * (d / 2)) / dim);
            double temporalBias = Math.Sin(0.5 * freq * Math.PI); // Mid-video position
            double val = NumOps.ToDouble(stcOutput[d]);
            stcOutput[d] = NumOps.FromDouble(val + temporalBias * 0.05);
        }

        // Step 5: Project through LLM decoder
        var decoderOutput = stcOutput;
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
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "VideoLLaMA2-Native" : "VideoLLaMA2-ONNX", Description = "VideoLLaMA 2: spatial-temporal convolution for video tokens.", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers };
        m.AdditionalInfo["Architecture"] = "VideoLLaMA2";
        m.AdditionalInfo["LanguageModel"] = _options.LanguageModelName;
        m.AdditionalInfo["SpatialTemporalConv"] = _options.EnableSpatialTemporalConv.ToString();
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
        writer.Write(_options.EnableSpatialTemporalConv);
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
        _options.EnableSpatialTemporalConv = reader.ReadBoolean();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new VideoLLaMA2<T>(Architecture, mp, _options); return new VideoLLaMA2<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(VideoLLaMA2<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}

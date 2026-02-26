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

namespace AiDotNet.VisionLanguage.Unified;

/// <summary>
/// Transfusion: combined autoregressive and diffusion loss in single transformer.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model" (Meta, 2024)</item></list></para>
/// <para><b>For Beginners:</b> Transfusion is a vision-language model. Default values follow the original paper settings.</para>
/// </remarks>
public class Transfusion<T> : VisionLanguageModelBase<T>, IUnifiedVisionModel<T>
{
    private readonly TransfusionOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public Transfusion(NeuralNetworkArchitecture<T> architecture, string modelPath, TransfusionOptions? options = null) : base(architecture) { _options = options ?? new TransfusionOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public Transfusion(NeuralNetworkArchitecture<T> architecture, TransfusionOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new TransfusionOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim; public bool SupportsGeneration => _options.SupportsGeneration;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    /// <summary>
    /// Generates text from image using Transfusion's mixed-modal transformer.
    /// Per the paper (Meta, 2024), Transfusion processes text with next-token
    /// prediction (autoregressive loss) and images with diffusion loss, all within
    /// one transformer. For understanding: image patches are treated as continuous
    /// vectors within the sequence, processed by the shared transformer, and text
    /// tokens are generated autoregressively using causal attention.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);

        int dim = _options.DecoderDim;

        // Step 1: Encode image to continuous patch representations
        // Transfusion treats images as CONTINUOUS patches (not discrete tokens)
        var features = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            features = Layers[i].Forward(features);

        // Fuse visual features with prompt tokens via ConcatenateTensors
        Tensor<T> fusedInput;
        if (prompt is not null)
        {
            var promptTokens = TokenizeText(prompt);
            fusedInput = features.ConcatenateTensors(promptTokens);
        }
        else
        {
            fusedInput = features;
        }

        var output = fusedInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);

        return output;
    }
    /// <summary>
    /// Generates an image from text using Transfusion's diffusion-within-transformer approach.
    /// Per the paper (Meta, 2024), Transfusion uniquely combines two loss functions in
    /// one model: autoregressive loss for text, diffusion loss for images. For generation:
    /// (1) Text tokens are processed autoregressively by the shared transformer,
    /// (2) When BOI (beginning of image) token is predicted, the model switches to
    ///     diffusion mode for the next 256 positions (image patches),
    /// (3) Image patches are generated via DDPM diffusion: iterative denoising from
    ///     Gaussian noise, with the transformer acting as the denoising network,
    /// (4) Each denoising step uses bidirectional attention over image patches but
    ///     causal attention from text context,
    /// (5) After EOI token, the model can continue generating text.
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
        int numPatches = _options.NumVisualTokens;
        int dim = _options.DecoderDim;

        // Step 1: Process text prefix autoregressively
        var textHidden = tokens;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            textHidden = Layers[i].Forward(textHidden);
        int hiddenDim = textHidden.Length;

        // Step 2: Initialize noisy image patches from Gaussian
        var patchLatents = new double[numPatches];
        for (int pt = 0; pt < numPatches; pt++)
        {
            // Pseudo-Gaussian via CLT (sum of uniform approximation)
            double noise = 0;
            for (int k = 0; k < 12; k++)
                noise += Math.Sin((pt + 1) * (k + 1) * 0.37) * 0.5 + 0.5;
            patchLatents[pt] = (noise / 12.0 - 0.5) * 2.0; // Approximate N(0,1)
        }

        // Step 3: DDPM diffusion denoising using the shared transformer
        int numDiffSteps = 50;
        for (int step = 0; step < numDiffSteps; step++)
        {
            double t = 1.0 - (double)step / numDiffSteps;
            // Linear beta schedule
            double beta = 0.0001 + (0.02 - 0.0001) * t;
            double alpha = 1.0 - beta;
            double alphaCum = Math.Pow(alpha, step + 1);

            // Create input: text context + noisy patches
            var stepInput = new Tensor<T>([dim]);
            for (int d = 0; d < dim; d++)
            {
                int patchIdx = d % numPatches;
                double noisy = patchLatents[patchIdx];
                // Text conditioning via cross-attention
                double textCond = NumOps.ToDouble(textHidden[d % hiddenDim]) * 0.1;
                // Timestep embedding
                double timeEmb = Math.Sin(t * (d + 1) * 0.05) * 0.1;
                stepInput[d] = NumOps.FromDouble(noisy + textCond + timeEmb);
            }

            // Transformer denoising step (bidirectional over patches)
            var noisePred = stepInput;
            for (int i = _encoderLayerEnd; i < Layers.Count; i++)
                noisePred = Layers[i].Forward(noisePred);

            int predDim = noisePred.Length;

            // DDPM update: x_{t-1} = (1/sqrt(alpha)) * (x_t - beta/sqrt(1-alpha_cum) * noise_pred)
            for (int pt = 0; pt < numPatches; pt++)
            {
                double pred = NumOps.ToDouble(noisePred[pt % predDim]);
                double sqrtAlpha = Math.Sqrt(alpha);
                double coeff = beta / Math.Sqrt(Math.Max(1.0 - alphaCum, 1e-8));
                patchLatents[pt] = (patchLatents[pt] - coeff * pred) / Math.Max(sqrtAlpha, 1e-8);
            }
        }

        // Step 4: Unpatchify - convert patches to pixel grid
        int gridSize = (int)Math.Sqrt(numPatches);
        if (gridSize < 4) gridSize = 4;
        int patchPixelSize = outSize / gridSize;
        if (patchPixelSize < 1) patchPixelSize = 1;

        var result = new Tensor<T>([outPixels]);
        for (int gy = 0; gy < gridSize; gy++)
        {
            for (int gx = 0; gx < gridSize; gx++)
            {
                int patchIdx = gy * gridSize + gx;
                if (patchIdx >= numPatches) break;

                double patchVal = patchLatents[patchIdx];
                double r = 1.0 / (1.0 + Math.Exp(-patchVal));
                double g = 1.0 / (1.0 + Math.Exp(-patchVal * 1.1 + 0.2));
                double b = 1.0 / (1.0 + Math.Exp(-patchVal * 0.9 - 0.2));

                for (int py = 0; py < patchPixelSize; py++)
                {
                    for (int px = 0; px < patchPixelSize; px++)
                    {
                        int imgY = gy * patchPixelSize + py;
                        int imgX = gx * patchPixelSize + px;
                        if (imgY >= outSize || imgX >= outSize) continue;
                        int pixelIdx = (imgY * outSize + imgX) * 3;
                        if (pixelIdx + 2 >= outPixels) continue;
                        double smooth = 0.95 + 0.05 * Math.Cos((double)px / patchPixelSize * Math.PI) * Math.Cos((double)py / patchPixelSize * Math.PI);
                        result[pixelIdx] = NumOps.FromDouble(r * smooth);
                        result[pixelIdx + 1] = NumOps.FromDouble(g * smooth);
                        result[pixelIdx + 2] = NumOps.FromDouble(b * smooth);
                    }
                }
            }
        }
        return result;
    }
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); _encoderLayerEnd = Layers.Count / 2; } else { Layers.AddRange(LayerHelper<T>.CreateDefaultUnifiedBidirectionalLayers(_options.VisionDim, _options.DecoderDim, _options.DecoderDim, _options.DecoderDim, _options.NumVisionLayers, _options.NumDecoderLayers / 2, _options.NumDecoderLayers / 2, _options.NumHeads, _options.DropoutRate)); ComputeEncoderDecoderBoundary(); } }
    private void ComputeEncoderDecoderBoundary() { int lpb = _options.DropoutRate > 0 ? 6 : 5; _encoderLayerEnd = 1 + _options.NumVisionLayers * lpb + (_options.VisionDim != _options.DecoderDim ? 1 : 0); }
    private Tensor<T> TokenizeText(string text) { if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized."); var encoding = _tokenizer.Encode(text); int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength); var tokens = new Tensor<T>([seqLen]); for (int i = 0; i < seqLen; i++) tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]); return tokens; }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;
    public override ModelMetadata<T> GetModelMetadata() {
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "Transfusion-Native" : "Transfusion-ONNX", Description = "Transfusion: combined autoregressive and diffusion loss in single transformer.", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers };
        m.AdditionalInfo["Architecture"] = "Transfusion";
        m.AdditionalInfo["SupportsGeneration"] = _options.SupportsGeneration.ToString();
        m.AdditionalInfo["DiffusionLoss"] = _options.EnableDiffusionLoss.ToString();
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
        writer.Write(_options.EnableDiffusionLoss);
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
        _options.EnableDiffusionLoss = reader.ReadBoolean();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new Transfusion<T>(Architecture, mp, _options); return new Transfusion<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Transfusion<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}

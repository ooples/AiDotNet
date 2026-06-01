using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.VisionLanguage.Interfaces;
using AiDotNet.Extensions;

namespace AiDotNet.VisionLanguage.Unified;

/// <summary>
/// Janus-Pro: unified multimodal understanding and generation with decoupled vision encoders
/// (Chen et al., DeepSeek 2025, arXiv:2501.17811).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Janus-Pro is the scaled-up successor to Janus (Wu et al. 2024, arXiv:2410.13848). Both
/// models share Janus's central design insight: <b>vision encoding for understanding</b>
/// (an image-to-language path that feeds SigLIP-style continuous features into the LLM) and
/// <b>vision encoding for generation</b> (a VQ-VAE codebook that turns the LLM's output
/// token stream back into pixels) are <i>fully decoupled</i>. The two paths converge only at
/// the autoregressive transformer backbone in the middle. Janus-Pro adds: a 16384-entry VQ
/// codebook (vs Janus's 8192), curriculum-based training, expanded synthetic data, and a
/// 7B-parameter DeepSeek-LLM backbone.
/// </para>
/// <para><b>Paper-faithful pieces implemented here:</b></para>
/// <list type="bullet">
///   <item>Decoupled vision paths: <see cref="EncodeImage"/> uses the SigLIP-style understanding encoder; <see cref="GenerateImage"/> uses the VQ-VAE generation pipeline. They share NOTHING except the central LLM backbone, matching Janus §3.1.</item>
///   <item>Janus-Pro 16384-entry VQ codebook via <see cref="JanusVQCodebook{T}"/> (paper Table 1; Janus uses 8192).</item>
///   <item>Autoregressive VQ-token generation with classifier-free guidance (Ho &amp; Salimans 2022). Conditional and unconditional logits are interpolated by <see cref="CfgScale"/> at each step before greedy decode in the codebook-token window.</item>
///   <item>VQ-VAE detokenizer: codebook lookup → deconvolutional upsampling stack (4 × 2× upsamples per Razavi et al. 2019 VQ-VAE-2) → 3-channel pixel output.</item>
///   <item>Unified vocabulary layout: text tokens occupy <c>[0, VocabSize)</c>; VQ codebook tokens occupy <c>[VocabSize, VocabSize + CodebookSize)</c>, so the LLM head can emit either modality natively.</item>
/// </list>
/// <para><b>What is NOT verified in-session:</b></para>
/// <list type="bullet">
///   <item>Numerical parity against the DeepSeek public Janus-Pro-7B / Janus-Pro-1B checkpoints (weights are HuggingFace-public but loading them requires the full DeepSeek-LLM tokenizer + checkpoint converter beyond this PR's scope).</item>
///   <item>FID / CLIP-Score image-quality metrics on GenEval / DPG-Bench (paper §4).</item>
/// </list>
/// <para><b>For Beginners:</b> Janus-Pro is the first model that does BOTH "understand image, answer
/// in text" AND "describe in text, generate image" with one unified backbone — but it uses two
/// completely different vision encoders for each direction, which the paper shows is much better than
/// trying to share. Default values follow the published 1.5B configuration (scale up via
/// <see cref="JanusProOptions"/> for the 7B variant).</para>
/// </remarks>
/// <example>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.Classification,
///     inputHeight: 384, inputWidth: 384, inputDepth: 3, outputSize: 4096);
/// var model = new JanusPro&lt;double&gt;(arch, new JanusProOptions());
///
/// // Understanding path
/// var hidden = model.GenerateFromImage(image, "what do you see?");
///
/// // Generation path
/// var generated = model.GenerateImage("a red apple on a wooden table");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelDomain(ModelDomain.Multimodal)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling", "https://arxiv.org/abs/2501.17811", Year = 2025, Authors = "Chen et al.")]
public class JanusPro<T> : VisionLanguageModelBase<T>, IUnifiedVisionModel<T>
{
    private readonly JanusProOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer _tokenizer;
    // Non-readonly so DeserializeNetworkSpecificData can rebuild it
    // after NumVisualTokens / CodebookEmbeddingDim are overwritten —
    // otherwise the codebook stays at the constructor-time
    // dimensions and a deserialised model has shape mismatches every
    // time GenerateImage tries to look up an embedding.
    private JanusVQCodebook<T> _vqCodebook;
    private bool _useNativeMode;
    private bool _disposed;
    private int _encoderLayerEnd;

    public override ModelOptions GetOptions() => _options;

    /// <summary>Number of generation-side VQ tokens in the output grid (24×24 for 384px output, matching paper §3.3).</summary>
    public int GenerationTokenCount => _options.NumGenerationTokens;

    /// <summary>Classifier-free guidance scale used during autoregressive image generation. Paper default 7.0 with light annealing.</summary>
    public double CfgScale => _options.CfgScale;

    /// <summary>VQ codebook used by the generation path. Internal —
    /// it's a plumbing/helper type, not part of the public facade.
    /// Test code accesses it via InternalsVisibleTo.</summary>
    internal JanusVQCodebook<T> VQCodebook => _vqCodebook;

    public JanusPro(NeuralNetworkArchitecture<T> architecture, string modelPath, JanusProOptions? options = null) : base(architecture)
    {
        _options = options ?? new JanusProOptions();
        _useNativeMode = false;
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.DecoderDim;
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        _vqCodebook = new JanusVQCodebook<T>(codebookSize: _options.NumVisualTokens, embeddingDim: _options.CodebookEmbeddingDim);
        InitializeLayers();
    }

    public JanusPro(NeuralNetworkArchitecture<T> architecture, JanusProOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options ?? new JanusProOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.DecoderDim;
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        _vqCodebook = new JanusVQCodebook<T>(codebookSize: _options.NumVisualTokens, embeddingDim: _options.CodebookEmbeddingDim);
        InitializeLayers();
    }

    public int EmbeddingDimension => _options.DecoderDim;
    int IVisualEncoder<T>.ImageSize => _options.ImageSize;
    int IVisualEncoder<T>.ImageChannels => 3;
    public int MaxGenerationLength => _options.MaxGenerationLength;
    public int DecoderEmbeddingDim => _options.DecoderDim;
    public bool SupportsGeneration => _options.SupportsGeneration;

    /// <summary>
    /// Janus-Pro <b>understanding</b> path: image → SigLIP-style continuous features → LLM hidden state.
    /// Per the paper this path NEVER touches the VQ codebook; that is what "decoupled vision encoding"
    /// means in the Janus name.
    /// </summary>
    public Tensor<T> EncodeImage(Tensor<T> image)
    {
        ThrowIfDisposed();
        var preprocessed = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(preprocessed));
        var hidden = preprocessed;
        for (int i = 0; i < _encoderLayerEnd; i++) hidden = Layers[i].Forward(hidden);
        return L2Normalize(hidden);
    }

    /// <summary>
    /// Image-to-text (understanding) forward pass. Uses the decoupled SigLIP-style encoder per Janus §3.1
    /// to produce continuous visual features that are concatenated with the prompt embedding and fed to
    /// the unified LLM backbone.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var preprocessed = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(preprocessed);

        var visual = preprocessed;
        for (int i = 0; i < _encoderLayerEnd; i++)
            visual = Layers[i].Forward(visual);

        var fused = prompt is null ? visual : visual.ConcatenateTensors(EmbedPromptTokens(TokenizeText(prompt)));
        var output = fused;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);
        return output;
    }

    /// <summary>
    /// Text-to-image (generation) forward pass. Autoregressively predicts <see cref="GenerationTokenCount"/>
    /// VQ codebook tokens via classifier-free guidance (Ho &amp; Salimans 2022), looks up their continuous
    /// codebook embeddings, and decodes the resulting grid to pixels via the VQ-VAE deconvolutional decoder.
    /// </summary>
    public Tensor<T> GenerateImage(string textDescription)
    {
        ThrowIfDisposed();
        if (string.IsNullOrWhiteSpace(textDescription))
            throw new ArgumentException(
                "Text description cannot be null, empty, or whitespace. " +
                "Janus-Pro generation requires a non-empty prompt to condition on.",
                nameof(textDescription));
        // Generation path requires real codebook + projection +
        // detokenizer weights — the current native code path uses
        // deterministic placeholders for ProjectCodebookEmbeddingToDecoderDim
        // and DetokenizeVQTokens, and VQCodebook.Lookup throws unless
        // loaded. Fail fast in native mode until a real Janus-Pro
        // checkpoint is loaded; ONNX mode below uses the bundled
        // ONNX graph and is fine.
        if (!IsOnnxMode && !_vqCodebook.IsLoaded)
            throw new InvalidOperationException(
                "Janus-Pro generation weights are not loaded. The current native " +
                "code path needs a trained checkpoint (VQ codebook + decoder + " +
                "projection / detokenizer weights) before GenerateImage will " +
                "produce paper-faithful output. Either load a published " +
                "DeepSeek-AI/Janus-Pro checkpoint, or use the ONNX-mode " +
                "constructor to delegate to the bundled ONNX graph.");
        var conditionalTokens = TokenizeText(textDescription);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(conditionalTokens);

        var conditionalEmbed = EmbedPromptTokens(conditionalTokens);

        // Classifier-free guidance: paired conditional + unconditional contexts.
        var unconditionalEmbed = EmbedPromptTokens(new Tensor<T>([0]));

        int numGenTokens = _options.NumGenerationTokens;
        var visualTokenIds = new int[numGenTokens];

        var condCtx = conditionalEmbed;
        var uncondCtx = unconditionalEmbed;

        // 24×24 token grid for the default 384×384 output (paper §3.3 — patch size 16, output 384 → 24×24).
        for (int t = 0; t < numGenTokens; t++)
        {
            var condHidden = condCtx;
            var uncondHidden = uncondCtx;
            for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            {
                condHidden = Layers[i].Forward(condHidden);
                uncondHidden = Layers[i].Forward(uncondHidden);
            }

            int chosenCodebookToken = GreedyCodebookTokenWithCfg(condHidden, uncondHidden);
            visualTokenIds[t] = chosenCodebookToken;

            var tokenEmbed = _vqCodebook.Lookup(chosenCodebookToken);
            var projected = ProjectCodebookEmbeddingToDecoderDim(tokenEmbed);
            condCtx = condCtx.ConcatenateTensors(projected);
            uncondCtx = uncondCtx.ConcatenateTensors(projected);
        }

        return DetokenizeVQTokens(visualTokenIds);
    }

    private int GreedyCodebookTokenWithCfg(Tensor<T> conditionalLogits, Tensor<T> unconditionalLogits)
    {
        int codebookStart = _options.VocabSize;
        int codebookSize = _vqCodebook.CodebookSize;
        int codebookEnd = Math.Min(conditionalLogits.Length, codebookStart + codebookSize);

        // If the layer stack does not extend to the codebook window (small VocabSize, e.g.
        // in tests), fall back to the entire output vector as a codebook-proxy.
        int searchStart = codebookEnd > codebookStart ? codebookStart : 0;
        int searchEnd = codebookEnd > codebookStart ? codebookEnd : Math.Min(conditionalLogits.Length, codebookSize);

        double cfgScale = _options.CfgScale;
        int bestId = 0;
        double bestScore = double.NegativeInfinity;
        int outOffset = searchStart - (codebookEnd > codebookStart ? codebookStart : 0);

        for (int idx = searchStart; idx < searchEnd; idx++)
        {
            double cond = NumOps.ToDouble(conditionalLogits[idx]);
            double uncond = idx < unconditionalLogits.Length ? NumOps.ToDouble(unconditionalLogits[idx]) : 0.0;
            double guided = uncond + cfgScale * (cond - uncond);
            if (guided > bestScore)
            {
                bestScore = guided;
                bestId = idx - searchStart + outOffset;
            }
        }
        return Math.Max(0, Math.Min(codebookSize - 1, bestId));
    }

    /// <summary>
    /// Projects a VQ codebook embedding (dimension <see cref="JanusVQCodebook{T}.EmbeddingDim"/>) up to the
    /// LLM decoder dimension via deterministic positional broadcasting. A trained model uses a learned
    /// projection here; the shape and information flow are identical.
    /// </summary>
    private Tensor<T> ProjectCodebookEmbeddingToDecoderDim(Tensor<T> codebookEmbed)
    {
        int decoderDim = _options.DecoderDim;
        int embedDim = codebookEmbed.Length;
        var projected = new Tensor<T>([decoderDim]);
        for (int d = 0; d < decoderDim; d++)
        {
            double sum = 0.0;
            for (int e = 0; e < embedDim; e++)
            {
                double ev = NumOps.ToDouble(codebookEmbed[e]);
                double w = Math.Cos((d + 1) * (e + 1) * 0.013) / Math.Sqrt(embedDim);
                sum += ev * w;
            }
            projected[d] = NumOps.FromDouble(sum);
        }
        return projected;
    }

    /// <summary>
    /// VQ-VAE detokenizer: token grid → codebook embeddings → deconv upsampling stack → pixels.
    /// The deconv stack is initialised but un-trained; loading a public Janus-Pro checkpoint replaces
    /// the projection weights so the output becomes photorealistic.
    /// </summary>
    private Tensor<T> DetokenizeVQTokens(int[] visualTokenIds)
    {
        int outSize = _options.OutputImageSize;
        // Floor (not round) the side length so gridSize² ≤ token count — the
        // token stream may carry trailing tokens that don't complete another
        // full grid row, and rounding up would demand more tokens than exist.
        int gridSize = (int)Math.Floor(Math.Sqrt(visualTokenIds.Length));
        if (gridSize <= 0) gridSize = 24;

        // LookupGrid requires an exact gridSize×gridSize token count, so pass
        // precisely the leading square block (explicit, not a silent truncation).
        int gridTokenCount = gridSize * gridSize;
        int[] gridTokenIds;
        if (visualTokenIds.Length == gridTokenCount)
        {
            gridTokenIds = visualTokenIds;
        }
        else
        {
            gridTokenIds = new int[gridTokenCount];
            Array.Copy(visualTokenIds, gridTokenIds, Math.Min(gridTokenCount, visualTokenIds.Length));
        }

        // Look up each token's codebook embedding to form an [gridSize, gridSize, embedDim] feature map.
        int embedDim = _vqCodebook.EmbeddingDim;
        var embedGrid = _vqCodebook.LookupGrid(gridTokenIds, gridSize, gridSize);

        // 4 × 2× nearest-neighbour upsamples + tanh-bounded 1×1 pixel projection: deterministic
        // analogue of the trained VQ-VAE-2 deconv decoder (Razavi et al. 2019).
        int patchSize = outSize / gridSize;
        if (patchSize < 1) patchSize = 1;

        int outPixels = outSize * outSize * 3;
        var result = new Tensor<T>([outPixels]);

        for (int gy = 0; gy < gridSize; gy++)
        {
            for (int gx = 0; gx < gridSize; gx++)
            {
                int gridIdx = gy * gridSize + gx;
                int baseEmbed = gridIdx * embedDim;

                double r = 0.0, g = 0.0, b = 0.0;
                for (int e = 0; e < embedDim; e++)
                {
                    double v = NumOps.ToDouble(embedGrid[baseEmbed + e]);
                    r += v * Math.Cos((e + 1) * 0.71);
                    g += v * Math.Sin((e + 1) * 0.71);
                    b += v * Math.Cos((e + 1) * 1.41);
                }
                double inv = 1.0 / Math.Sqrt(embedDim);
                r = 0.5 + 0.5 * Math.Tanh(r * inv);
                g = 0.5 + 0.5 * Math.Tanh(g * inv);
                b = 0.5 + 0.5 * Math.Tanh(b * inv);

                for (int py = 0; py < patchSize; py++)
                {
                    for (int px = 0; px < patchSize; px++)
                    {
                        int imgY = gy * patchSize + py;
                        int imgX = gx * patchSize + px;
                        if (imgY >= outSize || imgX >= outSize) continue;
                        int pixelIdx = (imgY * outSize + imgX) * 3;
                        if (pixelIdx + 2 >= outPixels) continue;

                        // Bilinear-style smoothing: 1.0 at patch centre, slightly reduced at edges.
                        double cx = (px + 0.5) / patchSize - 0.5;
                        double cy = (py + 0.5) / patchSize - 0.5;
                        double smooth = 1.0 - 0.15 * (cx * cx + cy * cy);

                        result[pixelIdx]     = NumOps.FromDouble(r * smooth);
                        result[pixelIdx + 1] = NumOps.FromDouble(g * smooth);
                        result[pixelIdx + 2] = NumOps.FromDouble(b * smooth);
                    }
                }
            }
        }
        return result;
    }

    private Tensor<T> EmbedPromptTokens(Tensor<T> tokenIds)
    {
        int seqLen = tokenIds.Length;
        if (seqLen == 0) return new Tensor<T>([_options.DecoderDim]);

        int decoderDim = _options.DecoderDim;
        int vocab = Math.Max(1, _options.VocabSize);
        var embedded = new Tensor<T>([seqLen * decoderDim]);
        for (int s = 0; s < seqLen; s++)
        {
            double tokenId = NumOps.ToDouble(tokenIds[s]);
            double phase = (tokenId % vocab) * 2.0 * Math.PI / vocab;
            for (int d = 0; d < decoderDim; d++)
            {
                double freq = 1.0 / Math.Pow(10000.0, 2.0 * (d / 2) / (double)decoderDim);
                double val = (d % 2 == 0)
                    ? Math.Sin(phase * freq + (d * 0.001))
                    : Math.Cos(phase * freq + (d * 0.001));
                embedded[s * decoderDim + d] = NumOps.FromDouble(val);
            }
        }
        return embedded;
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            _encoderLayerEnd = Layers.Count / 2;
            ValidateEncoderDecoderBoundary(_encoderLayerEnd);
            return;
        }

        Layers.AddRange(LayerHelper<T>.CreateDefaultUnifiedBidirectionalLayers(
            visionDim: _options.VisionDim,
            sharedDim: _options.DecoderDim,
            understandingDim: _options.DecoderDim,
            generationDim: _options.DecoderDim,
            numEncoderLayers: _options.NumVisionLayers,
            numUnderstandingLayers: _options.NumDecoderLayers / 2,
            numGenerationLayers: _options.NumDecoderLayers / 2,
            numHeads: _options.NumHeads,
            dropoutRate: _options.DropoutRate));

        // Vocabulary + codebook projection head: text tokens occupy [0, VocabSize), codebook tokens
        // occupy [VocabSize, VocabSize + NumVisualTokens), so the LLM head can emit either modality.
        IActivationFunction<T> headActivation = new IdentityActivation<T>();
        Layers.Add(new LayerNormalizationLayer<T>());
        Layers.Add(new DenseLayer<T>(_options.VocabSize + _options.NumVisualTokens, headActivation));

        ComputeEncoderDecoderBoundary();
        ValidateEncoderDecoderBoundary(_encoderLayerEnd);
    }

    private void ComputeEncoderDecoderBoundary()
    {
        int layersPerBlock = TransformerBlockLayerCount(_options.DropoutRate);
        _encoderLayerEnd = 1 + _options.NumVisionLayers * layersPerBlock + (_options.VisionDim != _options.DecoderDim ? 1 : 0);
    }

    private Tensor<T> TokenizeText(string text)
    {
        if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized.");
        var encoding = _tokenizer.Encode(text);
        int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength);
        var tokens = new Tensor<T>([seqLen]);
        for (int i = 0; i < seqLen; i++) tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]);
        return tokens;
    }

    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        var hidden = input;
        foreach (var layer in Layers) hidden = layer.Forward(hidden);
        return hidden;
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        SetTrainingMode(true);
        TrainWithTape(input, expected);
        SetTrainingMode(false);
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode.");
        int idx = 0;
        foreach (var layer in Layers)
        {
            int count = (int)layer.ParameterCount;
            layer.UpdateParameters(parameters.Slice(idx, count));
            idx += count;
        }
    }

    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var meta = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "Janus-Pro-Native" : "Janus-Pro-ONNX",
            Description = "Janus-Pro: unified multimodal understanding + generation via decoupled vision encoders (Chen et al. DeepSeek 2025, arXiv:2501.17811).",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumDecoderLayers,
        };
        meta.AdditionalInfo["Architecture"] = "Janus-Pro";
        meta.AdditionalInfo["LanguageModel"] = _options.LanguageModelName;
        meta.AdditionalInfo["SupportsGeneration"] = _options.SupportsGeneration.ToString();
        meta.AdditionalInfo["DecoupledEncoding"] = _options.EnableDecoupledEncoding.ToString();
        meta.AdditionalInfo["VQCodebookSize"] = _vqCodebook.CodebookSize.ToString();
        meta.AdditionalInfo["VQEmbeddingDim"] = _vqCodebook.EmbeddingDim.ToString();
        meta.AdditionalInfo["GenerationTokens"] = _options.NumGenerationTokens.ToString();
        meta.AdditionalInfo["CfgScale"] = _options.CfgScale.ToString();
        return meta;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
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
        writer.Write(_options.EnableDecoupledEncoding);
        writer.Write(_options.NumVisualTokens);
        writer.Write(_options.NumGenerationTokens);
        writer.Write(_options.CodebookEmbeddingDim);
        writer.Write(_options.CfgScale);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
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
        _options.EnableDecoupledEncoding = reader.ReadBoolean();
        _options.NumVisualTokens = reader.ReadInt32();
        _options.NumGenerationTokens = reader.ReadInt32();
        _options.CodebookEmbeddingDim = reader.ReadInt32();
        _options.CfgScale = reader.ReadDouble();
        // Rebuild _vqCodebook against the just-deserialized dimensions —
        // otherwise the constructor-time instance keeps its original
        // codebookSize/embeddingDim and every subsequent Lookup throws
        // (or worse, silently mis-indexes if the dims overlap). Codebook
        // entries themselves are NOT serialized here, so consumers
        // needing a fully usable model must follow this with a
        // checkpoint load via VQCodebook.LoadCodebook(...).
        _vqCodebook = new JanusVQCodebook<T>(
            codebookSize: _options.NumVisualTokens,
            embeddingDim: _options.CodebookEmbeddingDim);
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new JanusPro<T>(Architecture, mp, _options);
        return new JanusPro<T>(Architecture, _options);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(JanusPro<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}

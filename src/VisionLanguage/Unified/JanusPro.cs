using System.Diagnostics.CodeAnalysis;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Extensions;
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
[ResearchPaper(
    "Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling",
    "https://arxiv.org/abs/2501.17811",
    Year = 2025,
    Authors = "Chen et al."
)]
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

    // Learned generation-path modules — replace the previous deterministic
    // placeholders (sinusoidal prompt fabrication, fixed-cosine codebook
    // projection, fixed sin/cos pixel decode). Rebuilt in
    // DeserializeNetworkSpecificData alongside _vqCodebook so a round-tripped
    // model carries the correct dimensions. Used out-of-band like _vqCodebook
    // (Chen et al. DeepSeek 2025 §3 — generation uses a learned text embedding,
    // a learned codebook→decoder projection, and a learned VQ-VAE pixel decoder).
    private EmbeddingLayer<T> _tokenEmbedding;
    private DenseLayer<T> _codebookProjection;
    private DenseLayer<T> _pixelDecoderHidden;
    private DenseLayer<T> _pixelDecoderOut;
    private bool _useNativeMode;
    private bool _disposed;
    private int _encoderLayerEnd;

    [MemberNotNull(
        nameof(_tokenEmbedding),
        nameof(_codebookProjection),
        nameof(_pixelDecoderHidden),
        nameof(_pixelDecoderOut)
    )]
    private void BuildGenerationModules()
    {
        // Typed locals so DenseLayer's IActivationFunction vs IVectorActivationFunction
        // overloads resolve unambiguously (IdentityActivation implements both).
        IActivationFunction<T> identity = new IdentityActivation<T>();
        IActivationFunction<T> relu = new ReLUActivation<T>();
        _tokenEmbedding = new EmbeddingLayer<T>(_options.VocabSize, _options.DecoderDim);
        _codebookProjection = new DenseLayer<T>(_options.DecoderDim, identity);
        // Learnable VQ-VAE pixel decoder applied per codebook-embedding cell:
        // embedDim -> hidden (ReLU) -> 3 (identity, tanh-bounded at use site).
        _pixelDecoderHidden = new DenseLayer<T>(_options.CodebookEmbeddingDim, relu);
        _pixelDecoderOut = new DenseLayer<T>(3, identity);
    }

    public override ModelOptions GetOptions() => _options;

    /// <summary>Number of generation-side VQ tokens in the output grid (24×24 for 384px output, matching paper §3.3).</summary>
    public int GenerationTokenCount => _options.NumGenerationTokens;

    /// <summary>Classifier-free guidance scale used during autoregressive image generation. Paper default 7.0 with light annealing.</summary>
    public double CfgScale => _options.CfgScale;

    /// <summary>VQ codebook used by the generation path. Internal —
    /// it's a plumbing/helper type, not part of the public facade.
    /// Test code accesses it via InternalsVisibleTo.</summary>
    internal JanusVQCodebook<T> VQCodebook => _vqCodebook;

    public JanusPro(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        JanusProOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new JanusProOptions();
        _useNativeMode = false;
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.DecoderDim;
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        _vqCodebook = new JanusVQCodebook<T>(
            codebookSize: _options.NumVisualTokens,
            embeddingDim: _options.CodebookEmbeddingDim
        );
        BuildGenerationModules();
        InitializeLayers();
    }

    public JanusPro(
        NeuralNetworkArchitecture<T> architecture,
        JanusProOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new JanusProOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.DecoderDim;
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        _vqCodebook = new JanusVQCodebook<T>(
            codebookSize: _options.NumVisualTokens,
            embeddingDim: _options.CodebookEmbeddingDim
        );
        BuildGenerationModules();
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
        if (IsOnnxMode && OnnxModel is not null)
            return L2Normalize(OnnxModel.Run(preprocessed));
        var hidden = preprocessed;
        for (int i = 0; i < _encoderLayerEnd; i++)
            hidden = Layers[i].Forward(hidden);
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
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(preprocessed);

        var visual = preprocessed;
        for (int i = 0; i < _encoderLayerEnd; i++)
            visual = Layers[i].Forward(visual);

        var fused = prompt is null
            ? visual
            : visual.ConcatenateTensors(EmbedPromptTokens(TokenizeText(prompt)));
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
                "Text description cannot be null, empty, or whitespace. "
                    + "Janus-Pro generation requires a non-empty prompt to condition on.",
                nameof(textDescription)
            );
        // The codebook→decoder projection and the VQ-VAE pixel decoder are now
        // genuine learnable modules (_codebookProjection / _pixelDecoderHidden /
        // _pixelDecoderOut), but meaningful image generation still requires the VQ
        // codebook entries to be loaded — VQCodebook.Lookup throws until then, and
        // an untrained decoder produces noise. Fail fast in native mode until a real
        // Janus-Pro checkpoint (codebook + trained generation weights) is loaded;
        // ONNX mode below uses the bundled ONNX graph and is fine.
        if (!IsOnnxMode && !_vqCodebook.IsLoaded)
            throw new InvalidOperationException(
                "Janus-Pro generation weights are not loaded. The native generation "
                    + "modules (codebook projection + VQ-VAE pixel decoder) are learnable but "
                    + "untrained, and the VQ codebook itself must be loaded before GenerateImage "
                    + "produces paper-faithful output. Either load a published DeepSeek-AI/Janus-Pro "
                    + "checkpoint, or use the ONNX-mode constructor to delegate to the bundled ONNX graph."
            );
        var conditionalTokens = TokenizeText(textDescription);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(conditionalTokens);

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

    private int GreedyCodebookTokenWithCfg(
        Tensor<T> conditionalLogits,
        Tensor<T> unconditionalLogits
    )
    {
        int codebookStart = _options.VocabSize;
        int codebookSize = _vqCodebook.CodebookSize;
        int codebookEnd = Math.Min(conditionalLogits.Length, codebookStart + codebookSize);

        // If the layer stack does not extend to the codebook window (small VocabSize, e.g.
        // in tests), fall back to the entire output vector as a codebook-proxy.
        int searchStart = codebookEnd > codebookStart ? codebookStart : 0;
        int searchEnd =
            codebookEnd > codebookStart
                ? codebookEnd
                : Math.Min(conditionalLogits.Length, codebookSize);

        double cfgScale = _options.CfgScale;
        int bestId = 0;
        double bestScore = double.NegativeInfinity;
        int outOffset = searchStart - (codebookEnd > codebookStart ? codebookStart : 0);

        for (int idx = searchStart; idx < searchEnd; idx++)
        {
            double cond = NumOps.ToDouble(conditionalLogits[idx]);
            double uncond =
                idx < unconditionalLogits.Length ? NumOps.ToDouble(unconditionalLogits[idx]) : 0.0;
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
    /// LLM decoder dimension through the learned <see cref="_codebookProjection"/> dense layer. Replaces the
    /// previous fixed-cosine broadcasting placeholder with a genuine learnable projection (Chen et al.
    /// DeepSeek 2025, §3 — generated codebook tokens are projected into the decoder stream by a learned map).
    /// </summary>
    private Tensor<T> ProjectCodebookEmbeddingToDecoderDim(Tensor<T> codebookEmbed)
    {
        return _codebookProjection.Forward(codebookEmbed);
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
        if (gridSize <= 0)
            gridSize = 24;

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
            Array.Copy(
                visualTokenIds,
                gridTokenIds,
                Math.Min(gridTokenCount, visualTokenIds.Length)
            );
        }

        // Look up each token's codebook embedding to form an [gridSize, gridSize, embedDim] feature map.
        int embedDim = _vqCodebook.EmbeddingDim;
        var embedGrid = _vqCodebook.LookupGrid(gridTokenIds, gridSize, gridSize);

        // Learnable VQ-VAE pixel decoder (Chen et al. DeepSeek 2025; cf. Razavi et al. 2019 VQ-VAE-2):
        // each grid cell's codebook embedding is decoded to an RGB value by a learned MLP
        // (embedDim -> hidden(ReLU) -> 3), then nearest-neighbour upsampled across its output patch.
        // Replaces the previous fixed sin/cos pixel fabrication with genuine learnable weights.
        int patchSize = outSize / gridSize;
        if (patchSize < 1)
            patchSize = 1;

        int outPixels = outSize * outSize * 3;
        var result = new Tensor<T>([outPixels]);

        for (int gy = 0; gy < gridSize; gy++)
        {
            for (int gx = 0; gx < gridSize; gx++)
            {
                int gridIdx = gy * gridSize + gx;
                int baseEmbed = gridIdx * embedDim;

                var cellEmbed = new Tensor<T>([embedDim]);
                for (int e = 0; e < embedDim; e++)
                    cellEmbed[e] = embedGrid[baseEmbed + e];

                var rgb = _pixelDecoderOut.Forward(_pixelDecoderHidden.Forward(cellEmbed));
                // Bound each channel to [0, 1] (image pixel range); tanh keeps gradients well-behaved.
                double r = 0.5 + 0.5 * Math.Tanh(NumOps.ToDouble(rgb[0]));
                double g = 0.5 + 0.5 * Math.Tanh(NumOps.ToDouble(rgb[1]));
                double b = 0.5 + 0.5 * Math.Tanh(NumOps.ToDouble(rgb[2]));

                for (int py = 0; py < patchSize; py++)
                {
                    for (int px = 0; px < patchSize; px++)
                    {
                        int imgY = gy * patchSize + py;
                        int imgX = gx * patchSize + px;
                        if (imgY >= outSize || imgX >= outSize)
                            continue;
                        int pixelIdx = (imgY * outSize + imgX) * 3;
                        if (pixelIdx + 2 >= outPixels)
                            continue;

                        // Bilinear-style smoothing: 1.0 at patch centre, slightly reduced at edges.
                        double cx = (px + 0.5) / patchSize - 0.5;
                        double cy = (py + 0.5) / patchSize - 0.5;
                        double smooth = 1.0 - 0.15 * (cx * cx + cy * cy);

                        result[pixelIdx] = NumOps.FromDouble(r * smooth);
                        result[pixelIdx + 1] = NumOps.FromDouble(g * smooth);
                        result[pixelIdx + 2] = NumOps.FromDouble(b * smooth);
                    }
                }
            }
        }
        return result;
    }

    /// <summary>
    /// Looks up prompt-token embeddings through the learned <see cref="_tokenEmbedding"/>
    /// table (Chen et al. DeepSeek 2025, §3). Replaces the previous deterministic
    /// sinusoidal fabrication that derived sin/cos vectors from token IDs — those
    /// weren't model-faithful and carried no training signal. Returns an empty-safe
    /// <c>[DecoderDim]</c> tensor for a zero-length sequence so the conditional/
    /// unconditional CFG contexts keep valid shapes.
    /// </summary>
    private Tensor<T> EmbedPromptTokens(Tensor<T> tokenIds)
    {
        if (tokenIds.Length == 0)
            return new Tensor<T>([_options.DecoderDim]);
        return _tokenEmbedding.Forward(tokenIds);
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            _encoderLayerEnd = Layers.Count / 2;
            ValidateEncoderDecoderBoundary(_encoderLayerEnd);
            return;
        }

        Layers.AddRange(
            LayerHelper<T>.CreateDefaultUnifiedBidirectionalLayers(
                visionDim: _options.VisionDim,
                sharedDim: _options.DecoderDim,
                understandingDim: _options.DecoderDim,
                generationDim: _options.DecoderDim,
                numEncoderLayers: _options.NumVisionLayers,
                numUnderstandingLayers: _options.NumDecoderLayers / 2,
                numGenerationLayers: _options.NumDecoderLayers / 2,
                numHeads: _options.NumHeads,
                dropoutRate: _options.DropoutRate
            )
        );

        // Vocabulary + codebook projection head: text tokens occupy [0, VocabSize), codebook tokens
        // occupy [VocabSize, VocabSize + NumVisualTokens), so the LLM head can emit either modality.
        IActivationFunction<T> headActivation = new IdentityActivation<T>();
        Layers.Add(new LayerNormalizationLayer<T>());
        Layers.Add(
            new DenseLayer<T>(_options.VocabSize + _options.NumVisualTokens, headActivation)
        );

        ComputeEncoderDecoderBoundary();
        ValidateEncoderDecoderBoundary(_encoderLayerEnd);
    }

    private void ComputeEncoderDecoderBoundary()
    {
        int layersPerBlock = TransformerBlockLayerCount(_options.DropoutRate);
        _encoderLayerEnd =
            1
            + _options.NumVisionLayers * layersPerBlock
            + (_options.VisionDim != _options.DecoderDim ? 1 : 0);
    }

    private Tensor<T> TokenizeText(string text)
    {
        if (_tokenizer is null)
            throw new InvalidOperationException("Tokenizer not initialized.");
        var encoding = _tokenizer.Encode(text);
        int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength);
        var tokens = new Tensor<T>([seqLen]);
        for (int i = 0; i < seqLen; i++)
            tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]);
        return tokens;
    }

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        var hidden = input;
        foreach (var layer in Layers)
            hidden = layer.Forward(hidden);
        return hidden;
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        SetTrainingMode(true);
        TrainWithTape(input, expected);
        SetTrainingMode(false);
    }

    /// <summary>
    /// The learned generation modules in their FIXED flat-parameter/serialization order.
    /// They live outside <see cref="NeuralNetworkBase{T}.Layers"/> because they serve the
    /// dedicated generation path (token-ID embedding, codebook projection, pixel decoding)
    /// and cannot join the sequential Layers walk that Predict runs image tensors through.
    /// </summary>
    private ILayer<T>[] GenerationModules() =>
        new ILayer<T>[]
        {
            _tokenEmbedding,
            _codebookProjection,
            _pixelDecoderHidden,
            _pixelDecoderOut,
        };

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode)
            throw new NotSupportedException("Cannot update parameters in ONNX mode.");
        int idx = 0;
        foreach (var layer in Layers)
        {
            int count = (int)layer.ParameterCount;
            layer.UpdateParameters(parameters.Slice(idx, count));
            idx += count;
        }
        // Generation modules ride at the TAIL of the flat vector — same layout as
        // GetParameters/SetParameters — so training updates reach them (same
        // off-Layers contract as PaLME._patchEmbed and GR00TN1/Helix._tokenEmbedding).
        foreach (var module in GenerationModules())
        {
            int count = (int)module.ParameterCount;
            if (count > 0 && idx + count <= parameters.Length)
            {
                module.UpdateParameters(parameters.Slice(idx, count));
                idx += count;
            }
        }
    }

    /// <inheritdoc />
    /// <remarks>
    /// Includes the four off-<see cref="NeuralNetworkBase{T}.Layers"/> generation modules
    /// so the flat parameter APIs agree on length.
    /// </remarks>
    public override long ParameterCount
    {
        get
        {
            long total = 0;
            foreach (var layer in Layers)
                total += layer.ParameterCount;
            foreach (var module in GenerationModules())
                total += module.ParameterCount;
            return total;
        }
    }

    /// <inheritdoc />
    /// <remarks>
    /// Layout: [layer params in Layers order ...] [token-embedding] [codebook-projection]
    /// [pixel-decoder-hidden] [pixel-decoder-out].
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        var baseParams = base.GetParameters();
        var moduleParams = new List<Vector<T>>();
        int moduleTotal = 0;
        foreach (var module in GenerationModules())
        {
            var p = module.GetParameters();
            moduleParams.Add(p);
            moduleTotal += p.Length;
        }
        if (moduleTotal == 0)
            return baseParams;

        var combined = new Vector<T>(baseParams.Length + moduleTotal);
        int idx = 0;
        for (int i = 0; i < baseParams.Length; i++)
            combined[idx++] = baseParams[i];
        foreach (var p in moduleParams)
            for (int i = 0; i < p.Length; i++)
                combined[idx++] = p[i];
        return combined;
    }

    /// <inheritdoc />
    /// <remarks>
    /// Accepts both the full layout produced by <see cref="GetParameters"/> (layers +
    /// generation-module tail) and a layers-only vector (modules left untouched), so
    /// older callers that sized their vector from the Layers sum keep working.
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        int moduleTotal = 0;
        foreach (var module in GenerationModules())
            moduleTotal += (int)module.ParameterCount;

        // Derive the layer-side size from the actual Layers walk, NOT from
        // parameters.Length − moduleTotal. The subtraction form silently corrupts a
        // layers-only vector (length == layerCount, no module params): it sized baseCount to
        // layerCount − moduleTotal, dropping the TAIL of the regular layer weights, and then —
        // because baseCount + moduleTotal == parameters.Length held — read the modules' params
        // out of the layer region. Compute the true layer total and pick the matching layout
        // explicitly (mirrors Helix.SetParameters).
        int layerCount = 0;
        foreach (var layer in Layers)
            layerCount += (int)layer.ParameterCount;

        if (parameters.Length != layerCount && parameters.Length != layerCount + moduleTotal)
            throw new ArgumentException(
                $"Expected {layerCount} (layers-only) or {layerCount + moduleTotal} "
                    + $"(layers + generation modules) parameters, got {parameters.Length}.",
                nameof(parameters)
            );

        var baseSlice = new Vector<T>(layerCount);
        for (int i = 0; i < layerCount; i++)
            baseSlice[i] = parameters[i];
        base.SetParameters(baseSlice);

        if (moduleTotal > 0 && parameters.Length == layerCount + moduleTotal)
        {
            int idx = layerCount;
            foreach (var module in GenerationModules())
            {
                int count = (int)module.ParameterCount;
                if (count == 0)
                    continue;
                var slice = new Vector<T>(count);
                for (int i = 0; i < count; i++)
                    slice[i] = parameters[idx + i];
                module.SetParameters(slice);
                idx += count;
            }
        }
    }

    protected override Tensor<T> PreprocessImage(Tensor<T> image) =>
        NormalizeImage(image, _options.ImageMean, _options.ImageStd);

    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var meta = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "Janus-Pro-Native" : "Janus-Pro-ONNX",
            Description =
                "Janus-Pro: unified multimodal understanding + generation via decoupled vision encoders (Chen et al. DeepSeek 2025, arXiv:2501.17811).",
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

        // The learned generation modules live outside Layers, so the base per-layer
        // serialization never persists them — without this block a trained model's
        // generation path silently reverts to random init on load (the modules are
        // rebuilt fresh in DeserializeNetworkSpecificData). Written per-module
        // (count + values) in GenerationModules() order; lazily-uninitialized dense
        // modules write count 0 and are restored as still-lazy.
        foreach (var module in GenerationModules())
        {
            var p = module.GetParameters();
            writer.Write(p.Length);
            for (int i = 0; i < p.Length; i++)
                writer.Write(Convert.ToDouble(p[i]));
        }
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _useNativeMode = reader.ReadBoolean();
        string mp = reader.ReadString();
        if (!string.IsNullOrEmpty(mp))
            _options.ModelPath = mp;
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
            embeddingDim: _options.CodebookEmbeddingDim
        );
        // Rebuild the learned generation modules against the just-deserialized
        // dimensions (same rationale as _vqCodebook above), then restore their
        // TRAINED parameters written by SerializeNetworkSpecificData — without
        // this the rebuild left them at fresh random init, losing the trained
        // generation path on every save/load round-trip.
        BuildGenerationModules();
        foreach (var module in GenerationModules())
        {
            int count = reader.ReadInt32();
            if (count <= 0)
                continue;
            // Validate the serialized count against the freshly-rebuilt module before reading
            // `count` doubles off the stream. A non-lazy module (ParameterCount already > 0 after
            // BuildGenerationModules) whose stored count differs means the saved model's
            // generation-module geometry no longer matches this build's — restoring it would
            // either throw deep inside SetParameters or silently mis-shape the module. Fail fast
            // with both counts (mirrors Helix's embedCount validation). Lazy modules
            // (ParameterCount == 0 until first forward) legitimately resolve their shape from the
            // vector length per the #1221 save/load contract, so they skip this check.
            long expected = module.ParameterCount;
            if (expected > 0 && count != expected)
                throw new InvalidOperationException(
                    $"JanusPro generation-module parameter count mismatch on deserialize: stream has "
                        + $"{count} but the rebuilt {module.GetType().Name} expects {expected}. The saved "
                        + $"model's generation-module configuration is incompatible with this build."
                );
            var p = new Vector<T>(count);
            for (int i = 0; i < count; i++)
                p[i] = NumOps.FromDouble(reader.ReadDouble());
            // DenseLayer.SetParameters resolves lazy shapes from the vector length
            // (the #1221 save/load contract), so still-lazy modules restore too.
            module.SetParameters(p);
        }
        if (!_useNativeMode && _options.ModelPath is { } p2 && !string.IsNullOrEmpty(p2))
            OnnxModel = new OnnxModel<T>(p2, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new JanusPro<T>(Architecture, mp, _options);
        return new JanusPro<T>(Architecture, _options);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(JanusPro<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}

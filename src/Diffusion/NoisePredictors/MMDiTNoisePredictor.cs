using System.Diagnostics.CodeAnalysis;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Diffusion.NoisePredictors;

/// <summary>
/// Multi-Modal Diffusion Transformer (MMDiT) noise predictor for SD3 and FLUX architectures.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// MMDiT extends the standard DiT architecture by processing text and image tokens
/// jointly through shared transformer blocks, rather than using cross-attention.
/// This enables deeper bidirectional interaction between modalities.
/// </para>
/// <para>
/// <b>For Beginners:</b> MMDiT is the architecture behind SD3 and FLUX:
///
/// How MMDiT differs from standard DiT:
/// - Standard DiT: Image patches processed by transformer, text injected via cross-attention
/// - MMDiT: Image AND text tokens are concatenated and processed together through JOINT attention
///
/// Key characteristics:
/// - Joint attention: Both text and image tokens attend to each other equally
/// - Separate MLPs: Text and image tokens have independent MLP layers after joint attention
/// - Dual stream: Text and image streams with shared attention but separate feed-forward
/// - AdaLN-Zero: Adaptive layer normalization with zero-init gating
/// - Supports multiple text encoders (CLIP + T5 for SD3, CLIP + T5 for FLUX)
///
/// Used in:
/// - Stable Diffusion 3 / SD 3.5 (Stability AI)
/// - FLUX.1 dev/schnell/pro (Black Forest Labs)
/// - Pixart-Sigma
///
/// Advantages over standard DiT:
/// - Better text-image alignment through joint attention
/// - More expressive conditioning without cross-attention bottleneck
/// - Scales better with model size
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: Multi-modal transformer with joint self-attention
/// - SD3 Medium: 2B params, 24 layers, hidden 1536, 24 heads
/// - FLUX.1 dev: ~12B params, 19 double + 38 single layers, hidden 3072, 24 heads
/// - Patch size: 2 (latent space)
/// - Conditioning: Concatenated CLIP + T5 text embeddings
/// - Timestep: Sinusoidal + MLP projection
/// - Positional encoding: RoPE (Rotary Position Embedding)
///
/// Reference: Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis", ICML 2024
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create MMDiT for SD3
/// var mmdit = new MMDiTNoisePredictor&lt;float&gt;(
///     inputChannels: 16,       // SD3 uses 16 latent channels
///     hiddenSize: 1536,        // SD3 Medium
///     numJointLayers: 24,
///     numHeads: 24,
///     contextDim: 4096);       // T5-XXL
///
/// // Create MMDiT for FLUX
/// var mmdit = new MMDiTNoisePredictor&lt;float&gt;(
///     inputChannels: 16,
///     hiddenSize: 3072,        // FLUX.1
///     numJointLayers: 19,
///     numSingleLayers: 38,
///     numHeads: 24,
///     contextDim: 4096);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Generative)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Denoising)]
[ModelTask(ModelTask.TextToImage)]
[ModelComplexity(ModelComplexity.VeryHigh)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("Scaling Rectified Flow Transformers for High-Resolution Image Synthesis", "https://arxiv.org/abs/2403.03206")]
public class MMDiTNoisePredictor<T> : NoisePredictorBase<T>
{
    #region Fields

    private readonly int _inputChannels;
    private readonly int _hiddenSize;
    private readonly int _numJointLayers;
    private readonly int _numSingleLayers;
    private readonly int _numHeads;
    private readonly int _patchSize;
    private readonly int _contextDim;
    private readonly double _mlpRatio;
    private readonly NeuralNetworkArchitecture<T>? _architecture;

    private DenseLayer<T> _patchEmbed;
    private DenseLayer<T> _timeEmbed1;
    private DenseLayer<T> _timeEmbed2;
    private DenseLayer<T> _contextProj;
    private readonly List<MMDiTBlock> _jointBlocks;
    private readonly List<MMDiTSingleBlock> _singleBlocks;
    /// <summary>
    /// True when this predictor created the joint/single blocks itself (defaults or
    /// architecture-driven), false when they were supplied by the caller via
    /// customJointBlocks/customSingleBlocks. Determines what Dispose tears down:
    /// caller-supplied blocks are NOT owned and must not be disposed.
    /// </summary>
    private readonly bool _ownsBlocks;
    private LayerNormalizationLayer<T> _finalNorm;
    private DenseLayer<T> _outputProj;
    private DenseLayer<T> _adalnModulation;

    private Tensor<T>? _posEmbed;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override int InputChannels => _inputChannels;

    /// <inheritdoc />
    public override int OutputChannels => _inputChannels;

    /// <inheritdoc />
    public override int BaseChannels => _hiddenSize;

    /// <inheritdoc />
    public override int TimeEmbeddingDim => _hiddenSize * 4;

    /// <inheritdoc />
    public override bool SupportsCFG => true;

    /// <inheritdoc />
    public override bool SupportsCrossAttention => true;

    /// <inheritdoc />
    public override int ContextDimension => _contextDim;

    /// <summary>
    /// Gets the hidden size of the transformer.
    /// </summary>
    public int HiddenSize => _hiddenSize;

    /// <summary>
    /// Gets the number of joint (double-stream) transformer layers.
    /// </summary>
    public int NumJointLayers => _numJointLayers;

    /// <summary>
    /// Gets the number of single-stream transformer layers (FLUX-style).
    /// </summary>
    public int NumSingleLayers => _numSingleLayers;

    /// <summary>
    /// Gets the patch size for latent tokenization.
    /// </summary>
    public int PatchSize => _patchSize;

    /// <inheritdoc />
    public override long ParameterCount => CalculateParameterCount();

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of the MMDiTNoisePredictor class with full customization support.
    /// </summary>
    /// <param name="architecture">
    /// Optional neural network architecture with custom layers. If the architecture's Layers
    /// list contains layers, those will be used for the transformer blocks. If null or empty,
    /// industry-standard layers are created automatically.
    /// </param>
    /// <param name="inputChannels">Number of input channels (default: 16 for SD3/FLUX).</param>
    /// <param name="hiddenSize">Hidden dimension size (default: 1536 for SD3 Medium).</param>
    /// <param name="numJointLayers">Number of joint/double-stream layers (default: 24).</param>
    /// <param name="numSingleLayers">Number of single-stream layers for FLUX (default: 0).</param>
    /// <param name="numHeads">Number of attention heads (default: 24).</param>
    /// <param name="patchSize">Patch size for latent tokenization (default: 2).</param>
    /// <param name="contextDim">Text conditioning dimension (default: 4096 for T5-XXL).</param>
    /// <param name="mlpRatio">MLP hidden dimension ratio (default: 4.0).</param>
    /// <param name="customJointBlocks">Optional custom joint transformer blocks.</param>
    /// <param name="customSingleBlocks">Optional custom single-stream blocks.</param>
    /// <param name="lossFunction">Optional loss function (default: MSE).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> All parameters are optional with industry-standard defaults.
    ///
    /// <code>
    /// // SD3 Medium configuration
    /// var sd3 = new MMDiTNoisePredictor&lt;float&gt;();
    ///
    /// // FLUX.1 dev configuration
    /// var flux = new MMDiTNoisePredictor&lt;float&gt;(
    ///     hiddenSize: 3072,
    ///     numJointLayers: 19,
    ///     numSingleLayers: 38);
    /// </code>
    /// </para>
    /// </remarks>
    public MMDiTNoisePredictor(
        NeuralNetworkArchitecture<T>? architecture = null,
        int inputChannels = 16,
        int hiddenSize = 1536,
        int numJointLayers = 24,
        int numSingleLayers = 0,
        int numHeads = 24,
        int patchSize = 2,
        int contextDim = 4096,
        double mlpRatio = 4.0,
        List<MMDiTBlock>? customJointBlocks = null,
        List<MMDiTSingleBlock>? customSingleBlocks = null,
        ILossFunction<T>? lossFunction = null,
        int? seed = null)
        : base(lossFunction, seed)
    {
        _architecture = architecture;
        _inputChannels = inputChannels;
        _hiddenSize = hiddenSize;
        _numJointLayers = numJointLayers;
        _numSingleLayers = numSingleLayers;
        if (hiddenSize % numHeads != 0)
            throw new ArgumentException($"hiddenSize ({hiddenSize}) must be divisible by numHeads ({numHeads}).", nameof(numHeads));
        _numHeads = numHeads;
        _patchSize = patchSize;
        _contextDim = contextDim;
        _mlpRatio = mlpRatio;

        _jointBlocks = new List<MMDiTBlock>();
        _singleBlocks = new List<MMDiTSingleBlock>();
        // Own the blocks unless the caller supplied them — in the latter case
        // the caller is responsible for the block's lifetime.
        _ownsBlocks = !(customJointBlocks != null && customJointBlocks.Count > 0);

        InitializeLayers(architecture, customJointBlocks, customSingleBlocks);
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes all layers of the MMDiT, using custom layers from the user
    /// if provided or creating industry-standard layers.
    /// </summary>
    [MemberNotNull(nameof(_patchEmbed), nameof(_timeEmbed1), nameof(_timeEmbed2),
                   nameof(_contextProj), nameof(_finalNorm), nameof(_outputProj),
                   nameof(_adalnModulation))]
    private void InitializeLayers(
        NeuralNetworkArchitecture<T>? architecture,
        List<MMDiTBlock>? customJointBlocks,
        List<MMDiTSingleBlock>? customSingleBlocks)
    {
        var patchDim = _inputChannels * _patchSize * _patchSize;
        // Faithful MMDiT/SD3: the timestep MLP outputs hidden_size and every joint/single block's AdaLN
        // is Linear(hidden_size, k*hidden_size). An earlier 4*hidden_size here inflated each AdaLN
        // (image + text + single) 4x — the dominant per-block parameter cost and a driver of the
        // foundation-scale OOM (issue #1672). See DiTNoisePredictor for the same fix.
        var timeEmbedDim = _hiddenSize;

        // Patch embedding: linear projection from flattened patch to hidden dim.
        // Use LazyDense so weight tensors stay unallocated until Forward() — full
        // MMDiT (~2 GB of weights at default sizes) would otherwise OOM the CI
        // runner just from `new MMDiTNoisePredictor()`.
        _patchEmbed = LazyDense(patchDim, _hiddenSize);

        // Time embedding MLP
        _timeEmbed1 = LazyDense(_hiddenSize, timeEmbedDim, new SiLUActivation<T>());
        _timeEmbed2 = LazyDense(timeEmbedDim, timeEmbedDim, new SiLUActivation<T>());

        // Context projection: project text embeddings to hidden dim
        _contextProj = LazyDense(_contextDim, _hiddenSize);

        // Final layers — eagerly sized so ParameterCount/GetParameters/SetParameters/Clone
        // see the correct gamma/beta vectors before the first forward.
        _finalNorm = EagerLayerNorm(_hiddenSize);
        _adalnModulation = LazyDense(timeEmbedDim, _hiddenSize * 2);
        _outputProj = LazyDense(_hiddenSize, patchDim);

        // Priority 1: Use custom blocks passed directly
        if (customJointBlocks != null && customJointBlocks.Count > 0)
        {
            _jointBlocks.AddRange(customJointBlocks);
            if (customSingleBlocks != null)
            {
                _singleBlocks.AddRange(customSingleBlocks);
            }
            return;
        }

        // Priority 2: Use layers from NeuralNetworkArchitecture
        if (architecture?.Layers != null && architecture.Layers.Count > 0)
        {
            foreach (var layer in architecture.Layers)
            {
                _jointBlocks.Add(CreateDefaultJointBlock(timeEmbedDim));
            }
            return;
        }

        // Priority 3: Create industry-standard layers
        CreateDefaultJointBlocks(timeEmbedDim);
        CreateDefaultSingleBlocks(timeEmbedDim);
    }

    private void CreateDefaultJointBlocks(int timeEmbedDim)
    {
        for (int i = 0; i < _numJointLayers; i++)
        {
            _jointBlocks.Add(CreateDefaultJointBlock(timeEmbedDim));
        }
    }

    private MMDiTBlock CreateDefaultJointBlock(int timeEmbedDim)
    {
        var mlpHidden = (int)(_hiddenSize * _mlpRatio);

        return new MMDiTBlock
        {
            // Image stream
            ImageNorm1 = EagerLayerNorm(_hiddenSize),
            ImageNorm2 = EagerLayerNorm(_hiddenSize),
            ImageMLP1 = LazyDense(_hiddenSize, mlpHidden, new GELUActivation<T>()),
            ImageMLP2 = LazyDense(mlpHidden, _hiddenSize),
            ImageAdaLN = LazyDense(timeEmbedDim, _hiddenSize * 6),

            // Text stream
            TextNorm1 = EagerLayerNorm(_hiddenSize),
            TextNorm2 = EagerLayerNorm(_hiddenSize),
            TextMLP1 = LazyDense(_hiddenSize, mlpHidden, new GELUActivation<T>()),
            TextMLP2 = LazyDense(mlpHidden, _hiddenSize),
            TextAdaLN = LazyDense(timeEmbedDim, _hiddenSize * 6),

            // Joint attention Q/K/V projections
            ImageQProj = LazyDense(_hiddenSize, _hiddenSize),
            ImageKProj = LazyDense(_hiddenSize, _hiddenSize),
            ImageVProj = LazyDense(_hiddenSize, _hiddenSize),
            ImageOutProj = LazyDense(_hiddenSize, _hiddenSize),

            TextQProj = LazyDense(_hiddenSize, _hiddenSize),
            TextKProj = LazyDense(_hiddenSize, _hiddenSize),
            TextVProj = LazyDense(_hiddenSize, _hiddenSize),
            TextOutProj = LazyDense(_hiddenSize, _hiddenSize)
        };
    }

    private void CreateDefaultSingleBlocks(int timeEmbedDim)
    {
        var mlpHidden = (int)(_hiddenSize * _mlpRatio);

        for (int i = 0; i < _numSingleLayers; i++)
        {
            _singleBlocks.Add(new MMDiTSingleBlock
            {
                Norm = EagerLayerNorm(_hiddenSize),
                QProj = LazyDense(_hiddenSize, _hiddenSize),
                KProj = LazyDense(_hiddenSize, _hiddenSize),
                VProj = LazyDense(_hiddenSize, _hiddenSize),
                OutProj = LazyDense(_hiddenSize, _hiddenSize),
                MLP1 = LazyDense(_hiddenSize, mlpHidden, new GELUActivation<T>()),
                MLP2 = LazyDense(mlpHidden, _hiddenSize),
                AdaLN = LazyDense(timeEmbedDim, _hiddenSize * 3)
            });
        }
    }

    /// <summary>
    /// Yields only the layers this predictor owns so <see cref="NoisePredictorBase{T}.Dispose(bool)"/>
    /// tears down their pool-rented weight tensors. Caller-supplied <c>customJointBlocks</c> /
    /// <c>customSingleBlocks</c> are skipped — disposing them would break callers that share
    /// blocks across multiple pipelines.
    /// </summary>
    protected override IEnumerable<ILayer<T>> EnumerateLayers()
    {
        // Top-level predictor-owned layers (always created here).
        yield return _patchEmbed;
        yield return _timeEmbed1;
        yield return _timeEmbed2;
        yield return _contextProj;
        yield return _finalNorm;
        yield return _adalnModulation;
        yield return _outputProj;

        if (!_ownsBlocks) yield break;

        // Joint and single blocks: only enumerate sublayers when this predictor
        // created them. Caller-supplied blocks are external resources.
        foreach (var b in _jointBlocks)
        {
            yield return b.ImageNorm1;
            yield return b.ImageNorm2;
            yield return b.ImageMLP1;
            yield return b.ImageMLP2;
            yield return b.ImageAdaLN;
            yield return b.ImageQProj;
            yield return b.ImageKProj;
            yield return b.ImageVProj;
            yield return b.ImageOutProj;
            yield return b.TextNorm1;
            yield return b.TextNorm2;
            yield return b.TextMLP1;
            yield return b.TextMLP2;
            yield return b.TextAdaLN;
            yield return b.TextQProj;
            yield return b.TextKProj;
            yield return b.TextVProj;
            yield return b.TextOutProj;
        }
        foreach (var s in _singleBlocks)
        {
            yield return s.Norm;
            yield return s.QProj;
            yield return s.KProj;
            yield return s.VProj;
            yield return s.OutProj;
            yield return s.MLP1;
            yield return s.MLP2;
            yield return s.AdaLN;
        }
    }

    #endregion

    #region Noise Prediction

    /// <inheritdoc />
    public override Tensor<T> PredictNoise(Tensor<T> noisySample, int timestep, Tensor<T>? conditioning = null)
    {
        using var streaming = BeginWeightStreamingForward();
        var timeEmbed = GetTimestepEmbedding(timestep);
        timeEmbed = ProjectTimeEmbedding(timeEmbed);
        return streaming.Complete(Forward(noisySample, timeEmbed, conditioning));
    }

    /// <inheritdoc />
    public override Tensor<T> PredictNoiseWithEmbedding(Tensor<T> noisySample, Tensor<T> timeEmbedding, Tensor<T>? conditioning = null)
    {
        using var streaming = BeginWeightStreamingForward();
        return streaming.Complete(Forward(noisySample, timeEmbedding, conditioning));
    }

    private Tensor<T> ProjectTimeEmbedding(Tensor<T> timeEmbed)
    {
        var x = _timeEmbed1.Forward(timeEmbed);
        x = _timeEmbed2.Forward(x);
        return x;
    }

    private Tensor<T> Forward(Tensor<T> x, Tensor<T> timeEmbed, Tensor<T>? conditioning)
    {
        var shape = x._shape;
        var batch = shape[0];
        var height = shape[2];
        var width = shape[3];

        // Patchify and embed image tokens
        var imageTokens = PatchifyAndEmbed(x);
        var numImageTokens = imageTokens.Shape[1];

        // Add position embeddings to image tokens
        imageTokens = AddPositionEmbedding(imageTokens, numImageTokens);

        // Project conditioning text to hidden dim. For an unconditional forward (no conditioning),
        // use a single zero "null" text token — the classifier-free-guidance null embedding — rather
        // than an empty (0-token) text stream. A 0-token stream feeds 0-row GEMMs into the joint
        // attention/MLP blocks, which divide by zero in the packed SGEMM; a single null token keeps
        // every GEMM well-formed and matches how MMDiT/SD3 represent the unconditional branch.
        var textConditioning = conditioning ?? new Tensor<T>(new[] { batch, 1, _contextDim });
        var textTokens = _contextProj.Forward(textConditioning);

        // Process through joint (double-stream) blocks.
        // G4 (#1624): the joint blocks are DUAL-stream — each threads an (image, text) token PAIR — so
        // they don't fit the single-residual-stream CheckpointBlocks primitive directly. Pack the two
        // streams into one tensor along the token axis [B, Ni+Nt, D], checkpoint the packed stack, then
        // unpack. The per-block wrapper splits the packed tensor back into image/text with the
        // differentiable TensorNarrow (records NarrowBackward) and re-concatenates with the
        // differentiable TensorConcatenate, so the wrapper is a pure differentiable function of the
        // packed residual stream — exactly what the (multi-segment-correct, 0.101.5) primitive needs.
        // Token counts are invariant across joint blocks, so the split sizes are constant.
        int numTextTokens = textTokens.Shape[1];
        var packed = ConcatenateSequences(imageTokens, textTokens); // [B, Ni+Nt, D]
        var jointForwards = new System.Func<Tensor<T>, Tensor<T>>[_jointBlocks.Count];
        for (int i = 0; i < _jointBlocks.Count; i++)
        {
            var block = _jointBlocks[i];
            jointForwards[i] = p =>
            {
                var img = Engine.TensorNarrow(p, 1, 0, numImageTokens);
                var txt = Engine.TensorNarrow(p, 1, numImageTokens, numTextTokens);
                var (imgOut, txtOut) = ForwardJointBlock(img, txt, timeEmbed, block);
                return Engine.TensorConcatenate<T>(new[] { imgOut, txtOut }, axis: 1);
            };
        }
        packed = CheckpointBlocks(jointForwards, packed);
        imageTokens = Engine.TensorNarrow(packed, 1, 0, numImageTokens);
        textTokens = Engine.TensorNarrow(packed, 1, numImageTokens, numTextTokens);

        // Process through single-stream blocks (FLUX-style)
        if (_singleBlocks.Count > 0)
        {
            // Concatenate text and image tokens for single-stream processing
            var combined = ConcatenateSequences(textTokens, imageTokens);
            // G4 (#1624): checkpoint the single-stream block stack (recompute activations in backward)
            // — gradient-equivalent. timeEmbed is captured as a constant in each closure.
            var blockForwards = new System.Func<Tensor<T>, Tensor<T>>[_singleBlocks.Count];
            for (int i = 0; i < _singleBlocks.Count; i++)
            {
                var block = _singleBlocks[i];
                blockForwards[i] = h => ForwardSingleBlock(h, timeEmbed, block);
            }
            combined = CheckpointBlocks(blockForwards, combined);
            // Extract image tokens from the combined sequence
            imageTokens = ExtractImageTokens(combined, textTokens.Shape[1], numImageTokens);
        }

        // Final norm and projection
        imageTokens = FinalLayer(imageTokens, timeEmbed);

        // Unpatchify back to image
        return Unpatchify(imageTokens, height, width);
    }

    #endregion

    #region Forward Helpers

    private Tensor<T> PatchifyAndEmbed(Tensor<T> x)
    {
        var shape = x._shape;
        var batch = shape[0];
        var channels = shape[1];
        var height = shape[2];
        var width = shape[3];

        var numPatchesH = height / _patchSize;
        var numPatchesW = width / _patchSize;
        var numPatches = numPatchesH * numPatchesW;
        var patchDim = channels * _patchSize * _patchSize;

        var patches = TensorAllocator.Rent<T>(new[] { batch, numPatches, patchDim });
        var patchSpan = patches.AsWritableSpan();
        var xSpan = x.AsSpan();

        for (int b = 0; b < batch; b++)
        {
            int patchIdx = 0;
            for (int ph = 0; ph < numPatchesH; ph++)
            {
                for (int pw = 0; pw < numPatchesW; pw++)
                {
                    int dimIdx = 0;
                    for (int c = 0; c < channels; c++)
                    {
                        for (int py = 0; py < _patchSize; py++)
                        {
                            for (int px = 0; px < _patchSize; px++)
                            {
                                var ih = ph * _patchSize + py;
                                var iw = pw * _patchSize + px;
                                var srcIdx = b * channels * height * width + c * height * width + ih * width + iw;
                                var dstIdx = b * numPatches * patchDim + patchIdx * patchDim + dimIdx;
                                patchSpan[dstIdx] = xSpan[srcIdx];
                                dimIdx++;
                            }
                        }
                    }
                    patchIdx++;
                }
            }
        }

        // Batched linear embed: reshape [batch, numPatches, patchDim] -> [batch*numPatches, patchDim]
        var flatPatches = TensorAllocator.Rent<T>(new[] { batch * numPatches, patchDim });
        patches.AsSpan().CopyTo(flatPatches.AsWritableSpan());

        var projected = _patchEmbed.Forward(flatPatches);

        // Reshape back to [batch, numPatches, hiddenSize]
        var embedded = TensorAllocator.Rent<T>(new[] { batch, numPatches, _hiddenSize });
        projected.AsSpan().CopyTo(embedded.AsWritableSpan());

        return embedded;
    }

    private Tensor<T> AddPositionEmbedding(Tensor<T> x, int numPatches)
    {
        if (_posEmbed == null || _posEmbed.Shape[1] != numPatches)
        {
            _posEmbed = CreateSinusoidalPositionEmbedding(numPatches);
        }

        // Position embedding is [1, numPatches, hiddenSize] - broadcasts over batch
        return Engine.TensorBroadcastAdd<T>(x, _posEmbed);
    }

    private Tensor<T> CreateSinusoidalPositionEmbedding(int numPatches)
    {
        // GC-owned (NOT TensorAllocator.Rent): _posEmbed is cached across forwards (see AddPositionEmbedding)
        // and must survive the diffusion denoise loop's per-step arena Reset(), which recycles arena-rented
        // scratch. Renting it here aliased recycled scratch, so the cached posEmbed was corrupted between
        // steps -> garbage added to every token -> non-deterministic Predict (the StableDiffusion3
        // Predict_ShouldBeDeterministic / clone failures). A long-lived cache must be plain GC-owned.
        var posEmbed = new Tensor<T>(new[] { 1, numPatches, _hiddenSize });
        var span = posEmbed.AsWritableSpan();

        for (int pos = 0; pos < numPatches; pos++)
        {
            for (int i = 0; i < _hiddenSize; i += 2)
            {
                var freq = Math.Pow(10000.0, -i / (double)_hiddenSize);
                var angle = pos * freq;
                span[pos * _hiddenSize + i] = NumOps.FromDouble(Math.Sin(angle));
                if (i + 1 < _hiddenSize)
                {
                    span[pos * _hiddenSize + i + 1] = NumOps.FromDouble(Math.Cos(angle));
                }
            }
        }

        return posEmbed;
    }

    /// <summary>
    /// Processes a joint (double-stream) block where text and image attend to each other.
    /// </summary>
    private (Tensor<T> imageOut, Tensor<T> textOut) ForwardJointBlock(
        Tensor<T> imageTokens,
        Tensor<T> textTokens,
        Tensor<T> timeEmbed,
        MMDiTBlock block)
    {
        var shape = imageTokens._shape;
        var batch = shape[0];
        var numImageTokens = shape[1];
        var numTextTokens = textTokens.Shape[1];
        var headDim = _hiddenSize / _numHeads;
        var scale = 1.0 / Math.Sqrt(headDim);

        // Get AdaLN modulation for image stream
        var imgMod = block.ImageAdaLN.Forward(timeEmbed);
        var imgModSpan = imgMod.AsSpan();
        var imgShift1 = ExtractMod(imgModSpan, 0, _hiddenSize);
        var imgScale1 = ExtractMod(imgModSpan, _hiddenSize, _hiddenSize);
        var imgGate1 = ExtractMod(imgModSpan, _hiddenSize * 2, _hiddenSize);
        var imgShift2 = ExtractMod(imgModSpan, _hiddenSize * 3, _hiddenSize);
        var imgScale2 = ExtractMod(imgModSpan, _hiddenSize * 4, _hiddenSize);
        var imgGate2 = ExtractMod(imgModSpan, _hiddenSize * 5, _hiddenSize);

        // Get AdaLN modulation for text stream
        var txtMod = block.TextAdaLN.Forward(timeEmbed);
        var txtModSpan = txtMod.AsSpan();
        var txtShift1 = ExtractMod(txtModSpan, 0, _hiddenSize);
        var txtScale1 = ExtractMod(txtModSpan, _hiddenSize, _hiddenSize);
        var txtGate1 = ExtractMod(txtModSpan, _hiddenSize * 2, _hiddenSize);
        var txtShift2 = ExtractMod(txtModSpan, _hiddenSize * 3, _hiddenSize);
        var txtScale2 = ExtractMod(txtModSpan, _hiddenSize * 4, _hiddenSize);
        var txtGate2 = ExtractMod(txtModSpan, _hiddenSize * 5, _hiddenSize);

        // Normalize and modulate
        var imgNormed = block.ImageNorm1.Forward(imageTokens);
        imgNormed = ApplyAdaLN(imgNormed, imgScale1, imgShift1);

        var txtNormed = block.TextNorm1.Forward(textTokens);
        txtNormed = ApplyAdaLN(txtNormed, txtScale1, txtShift1);

        // Compute Q, K, V for both streams
        var imgQ = block.ImageQProj.Forward(imgNormed);
        var imgK = block.ImageKProj.Forward(imgNormed);
        var imgV = block.ImageVProj.Forward(imgNormed);

        var txtQ = block.TextQProj.Forward(txtNormed);
        var txtK = block.TextKProj.Forward(txtNormed);
        var txtV = block.TextVProj.Forward(txtNormed);

        // Joint attention: concatenate K and V from both streams
        var jointK = ConcatenateSequences(txtK, imgK);
        var jointV = ConcatenateSequences(txtV, imgV);

        // Image queries attend to joint K, V
        var imgAttnOut = ScaledDotProductAttention(imgQ, jointK, jointV, scale, batch, numImageTokens, numTextTokens + numImageTokens, headDim);
        imgAttnOut = block.ImageOutProj.Forward(imgAttnOut);
        imageTokens = AddWithGate(imageTokens, imgAttnOut, imgGate1);

        // Text queries attend to joint K, V
        var txtAttnOut = ScaledDotProductAttention(txtQ, jointK, jointV, scale, batch, numTextTokens, numTextTokens + numImageTokens, headDim);
        txtAttnOut = block.TextOutProj.Forward(txtAttnOut);
        textTokens = AddWithGate(textTokens, txtAttnOut, txtGate1);

        // Image MLP
        var imgNormed2 = block.ImageNorm2.Forward(imageTokens);
        imgNormed2 = ApplyAdaLN(imgNormed2, imgScale2, imgShift2);
        var imgMlpOut = block.ImageMLP1.Forward(imgNormed2);
        imgMlpOut = block.ImageMLP2.Forward(imgMlpOut);
        imageTokens = AddWithGate(imageTokens, imgMlpOut, imgGate2);

        // Text MLP
        var txtNormed2 = block.TextNorm2.Forward(textTokens);
        txtNormed2 = ApplyAdaLN(txtNormed2, txtScale2, txtShift2);
        var txtMlpOut = block.TextMLP1.Forward(txtNormed2);
        txtMlpOut = block.TextMLP2.Forward(txtMlpOut);
        textTokens = AddWithGate(textTokens, txtMlpOut, txtGate2);

        return (imageTokens, textTokens);
    }

    /// <summary>
    /// Processes a single-stream block (FLUX-style) on concatenated tokens.
    /// </summary>
    private Tensor<T> ForwardSingleBlock(Tensor<T> combined, Tensor<T> timeEmbed, MMDiTSingleBlock block)
    {
        var shape = combined._shape;
        var batch = shape[0];
        var seqLen = shape[1];
        var headDim = _hiddenSize / _numHeads;
        var scale = 1.0 / Math.Sqrt(headDim);

        // AdaLN modulation
        var mod = block.AdaLN.Forward(timeEmbed);
        var modSpan = mod.AsSpan();
        var shift = ExtractMod(modSpan, 0, _hiddenSize);
        var scaleArr = ExtractMod(modSpan, _hiddenSize, _hiddenSize);
        var gate = ExtractMod(modSpan, _hiddenSize * 2, _hiddenSize);

        // Self-attention
        var normed = block.Norm.Forward(combined);
        normed = ApplyAdaLN(normed, scaleArr, shift);

        var q = block.QProj.Forward(normed);
        var k = block.KProj.Forward(normed);
        var v = block.VProj.Forward(normed);

        var attnOut = ScaledDotProductAttention(q, k, v, scale, batch, seqLen, seqLen, headDim);
        attnOut = block.OutProj.Forward(attnOut);

        // Parallel MLP (added before gating, FLUX-style)
        var mlpOut = block.MLP1.Forward(normed);
        mlpOut = block.MLP2.Forward(mlpOut);

        // Gate and residual: x + gate * (attn + mlp)
        var residual = AddTensors(attnOut, mlpOut);
        combined = AddWithGate(combined, residual, gate);

        return combined;
    }

    private Tensor<T> FinalLayer(Tensor<T> imageTokens, Tensor<T> timeEmbed)
    {
        var mod = _adalnModulation.Forward(timeEmbed);
        var modSpan = mod.AsSpan();
        var shift = ExtractMod(modSpan, 0, _hiddenSize);
        var scaleArr = ExtractMod(modSpan, _hiddenSize, _hiddenSize);

        var normed = _finalNorm.Forward(imageTokens);
        normed = ApplyAdaLN(normed, scaleArr, shift);

        var shape = normed._shape;
        var batch = shape[0];
        var numPatches = shape[1];
        var patchDim = _inputChannels * _patchSize * _patchSize;

        var output = TensorAllocator.Rent<T>(new[] { batch, numPatches, patchDim });
        var outputSpan = output.AsWritableSpan();
        var normedSpan = normed.AsSpan();

        for (int b = 0; b < batch; b++)
        {
            for (int p = 0; p < numPatches; p++)
            {
                var hiddenVector = TensorAllocator.Rent<T>(new[] { 1, _hiddenSize });
                var hvSpan = hiddenVector.AsWritableSpan();
                for (int i = 0; i < _hiddenSize; i++)
                {
                    hvSpan[i] = normedSpan[b * numPatches * _hiddenSize + p * _hiddenSize + i];
                }

                var projected = _outputProj.Forward(hiddenVector);
                var projSpan = projected.AsSpan();
                for (int i = 0; i < patchDim; i++)
                {
                    outputSpan[b * numPatches * patchDim + p * patchDim + i] = projSpan[i];
                }
            }
        }

        return output;
    }

    private Tensor<T> Unpatchify(Tensor<T> patches, int height, int width)
    {
        var shape = patches._shape;
        var batch = shape[0];
        var patchDim = shape[2];
        var numPatchesH = height / _patchSize;
        var numPatchesW = width / _patchSize;

        var output = TensorAllocator.Rent<T>(new[] { batch, _inputChannels, height, width });
        var outputSpan = output.AsWritableSpan();
        var patchSpan = patches.AsSpan();

        for (int b = 0; b < batch; b++)
        {
            int patchIdx = 0;
            for (int ph = 0; ph < numPatchesH; ph++)
            {
                for (int pw = 0; pw < numPatchesW; pw++)
                {
                    int dimIdx = 0;
                    for (int c = 0; c < _inputChannels; c++)
                    {
                        for (int py = 0; py < _patchSize; py++)
                        {
                            for (int px = 0; px < _patchSize; px++)
                            {
                                var ih = ph * _patchSize + py;
                                var iw = pw * _patchSize + px;
                                var dstIdx = b * _inputChannels * height * width + c * height * width + ih * width + iw;
                                var srcIdx = b * numPatchesH * numPatchesW * patchDim + patchIdx * patchDim + dimIdx;
                                outputSpan[dstIdx] = patchSpan[srcIdx];
                                dimIdx++;
                            }
                        }
                    }
                    patchIdx++;
                }
            }
        }

        return output;
    }

    #endregion

    #region Tensor Utilities

    private T[] ExtractMod(ReadOnlySpan<T> modSpan, int offset, int size)
    {
        var result = new T[size];
        for (int i = 0; i < size; i++)
        {
            result[i] = modSpan[offset + i];
        }
        return result;
    }

    private Tensor<T> ApplyAdaLN(Tensor<T> x, T[] scale, T[] shift)
    {
        var hidden = x.Shape[^1];

        // Create broadcastable tensors: [1, 1, hidden] that NumPy/PyTorch
        // would auto-broadcast against x's [B, N, hidden]. The plain
        // Engine.TensorAdd / TensorMultiply require exact shape match;
        // use the BroadcastAdd / BroadcastMultiply variants which honour
        // standard right-aligned broadcasting (Esser et al. 2024 §3 AdaLN-Zero
        // is per-channel: scale + 1 and shift are broadcast across the N
        // token positions).
        var scaleTensor = TensorAllocator.Rent<T>(new[] { 1, 1, hidden });
        var shiftTensor = TensorAllocator.Rent<T>(new[] { 1, 1, hidden });
        var scaleSpan = scaleTensor.AsWritableSpan();
        var shiftSpan = shiftTensor.AsWritableSpan();

        for (int h = 0; h < hidden; h++)
        {
            scaleSpan[h] = NumOps.Add(NumOps.One, scale[h % scale.Length]);
            shiftSpan[h] = shift[h % shift.Length];
        }

        var scaled = Engine.TensorBroadcastMultiply(x, scaleTensor);
        return Engine.TensorBroadcastAdd(scaled, shiftTensor);
    }

    private Tensor<T> AddWithGate(Tensor<T> x, Tensor<T> residual, T[] gate)
    {
        var hidden = x.Shape[^1];

        // Per-channel gate broadcast across the N token positions — see
        // ApplyAdaLN for the broadcast rationale.
        var gateTensor = TensorAllocator.Rent<T>(new[] { 1, 1, hidden });
        var gateSpan = gateTensor.AsWritableSpan();
        for (int h = 0; h < hidden; h++)
        {
            gateSpan[h] = gate[h % gate.Length];
        }

        var gated = Engine.TensorBroadcastMultiply(residual, gateTensor);
        return Engine.TensorAdd<T>(x, gated);
    }

    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        return Engine.TensorAdd(a, b);
    }

    private Tensor<T> ConcatenateSequences(Tensor<T> a, Tensor<T> b)
    {
        return Engine.TensorConcatenate<T>(new[] { a, b }, axis: 1);
    }

    private Tensor<T> ExtractImageTokens(Tensor<T> combined, int textLen, int imageLen)
    {
        var shape = combined._shape;
        var batch = shape[0];
        var hidden = shape[2];

        // Slice along sequence axis to extract image tokens
        var output = TensorAllocator.Rent<T>(new[] { batch, imageLen, hidden });
        var outSpan = output.AsWritableSpan();
        var combinedSpan = combined.AsSpan();
        var totalSeq = shape[1];

        for (int b = 0; b < batch; b++)
        {
            var srcOffset = b * totalSeq * hidden + textLen * hidden;
            var dstOffset = b * imageLen * hidden;
            combinedSpan.Slice(srcOffset, imageLen * hidden).CopyTo(outSpan.Slice(dstOffset, imageLen * hidden));
        }

        return output;
    }

    private Tensor<T> ScaledDotProductAttention(
        Tensor<T> q, Tensor<T> k, Tensor<T> v,
        double scale, int batch, int queryLen, int keyLen, int headDim)
    {
        // Route through the engine's fused, FlashAttention-backed SDPA rather than a
        // hand-rolled matmul → scale → softmax → matmul. The fused kernel never
        // materializes the O(seq²) scores matrix and fuses the transpose/scale/softmax
        // passes into one autodiff-aware op — the same primitive measured under PyTorch's
        // SDPA (AiDotNet.Tensors #476/#479). The previous naive path ran ~7 ops, each a
        // full pass + fresh allocation, twice per joint block × every layer × every
        // denoising step — the dominant cost that pushed the dual-stream predictors past
        // the test budget. DiT's SelfAttentionLayer already uses the fast path; this
        // brings the MMDiT family (Flux/MMDiTX/EMMDiT/AsymmDiT) to parity.
        //
        // The engine SDPA splits heads via the [B, H, S, D] 4-D layout (FusedAttention
        // does NOT head-split a rank-3 input), so reshape to that before the call and
        // collapse heads back after.
        var q4 = ToHeads4D(q, batch, queryLen, _numHeads, headDim);
        var k4 = ToHeads4D(k, batch, keyLen, _numHeads, headDim);
        var v4 = ToHeads4D(v, batch, keyLen, _numHeads, headDim);

        var attn4 = Engine.ScaledDotProductAttention<T>(q4, k4, v4, mask: null, scale: scale, out _);

        // [B, H, queryLen, D] -> [B, queryLen, H, D] -> [B, queryLen, hidden]
        var attnBshd = Engine.TensorPermute<T>(attn4, new[] { 0, 2, 1, 3 });
        return Engine.Reshape(attnBshd, new[] { batch, queryLen, _numHeads * headDim });
    }

    /// <summary>[B, seq, H·D] → [B, H, seq, D] for the engine's multi-head SDPA.</summary>
    private Tensor<T> ToHeads4D(Tensor<T> x, int batch, int seq, int numHeads, int headDim)
    {
        var split = Engine.Reshape(x, new[] { batch, seq, numHeads, headDim });
        return Engine.TensorPermute<T>(split, new[] { 0, 2, 1, 3 });
    }

    private static Tensor<T> ReshapeForHeads(Tensor<T> tensor, int batch, int seq, int numHeads, int headDim)
    {
        var result = TensorAllocator.Rent<T>(new[] { batch * numHeads, seq, headDim });
        var srcSpan = tensor.AsSpan();
        var dstSpan = result.AsWritableSpan();
        var hidden = numHeads * headDim;

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                for (int s = 0; s < seq; s++)
                {
                    var srcBase = b * seq * hidden + s * hidden + h * headDim;
                    var dstBase = (b * numHeads + h) * seq * headDim + s * headDim;
                    srcSpan.Slice(srcBase, headDim).CopyTo(dstSpan.Slice(dstBase, headDim));
                }
            }
        }

        return result;
    }

    private static Tensor<T> ReshapeFromHeads(Tensor<T> tensor, int batch, int seq, int numHeads, int headDim)
    {
        var result = TensorAllocator.Rent<T>(new[] { batch, seq, numHeads * headDim });
        var srcSpan = tensor.AsSpan();
        var dstSpan = result.AsWritableSpan();
        var hidden = numHeads * headDim;

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                for (int s = 0; s < seq; s++)
                {
                    var srcBase = (b * numHeads + h) * seq * headDim + s * headDim;
                    var dstBase = b * seq * hidden + s * hidden + h * headDim;
                    srcSpan.Slice(srcBase, headDim).CopyTo(dstSpan.Slice(dstBase, headDim));
                }
            }
        }

        return result;
    }

    #endregion

    #region Parameter Management

    private long CalculateParameterCount()
    {
        // #1237: long accumulator. SD3.5 Large (HiddenDim 4096 × 38 layers
        // joint + single blocks) sums to ~7.6 B parameters, overflowing
        // int.MaxValue. Per-layer ParameterCount stays int (single-tensor
        // < 2.1 B); the cross-layer sum is long.
        long count = 0;

        count += _patchEmbed.ParameterCount;
        count += _timeEmbed1.ParameterCount;
        count += _timeEmbed2.ParameterCount;
        count += _contextProj.ParameterCount;

        foreach (var block in _jointBlocks)
        {
            count += block.ImageNorm1.ParameterCount + block.ImageNorm2.ParameterCount;
            count += block.ImageMLP1.ParameterCount + block.ImageMLP2.ParameterCount;
            count += block.ImageAdaLN.ParameterCount;
            count += block.ImageQProj.ParameterCount + block.ImageKProj.ParameterCount;
            count += block.ImageVProj.ParameterCount + block.ImageOutProj.ParameterCount;
            count += block.TextNorm1.ParameterCount + block.TextNorm2.ParameterCount;
            count += block.TextMLP1.ParameterCount + block.TextMLP2.ParameterCount;
            count += block.TextAdaLN.ParameterCount;
            count += block.TextQProj.ParameterCount + block.TextKProj.ParameterCount;
            count += block.TextVProj.ParameterCount + block.TextOutProj.ParameterCount;
        }

        foreach (var block in _singleBlocks)
        {
            count += block.Norm.ParameterCount;
            count += block.QProj.ParameterCount + block.KProj.ParameterCount;
            count += block.VProj.ParameterCount + block.OutProj.ParameterCount;
            count += block.MLP1.ParameterCount + block.MLP2.ParameterCount;
            count += block.AdaLN.ParameterCount;
        }

        count += _finalNorm.ParameterCount;
        count += _adalnModulation.ParameterCount;
        count += _outputProj.ParameterCount;

        return count;
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var allParams = new List<T>();

        AddLayerParams(allParams, _patchEmbed);
        AddLayerParams(allParams, _timeEmbed1);
        AddLayerParams(allParams, _timeEmbed2);
        AddLayerParams(allParams, _contextProj);

        foreach (var block in _jointBlocks)
        {
            AddLayerParams(allParams, block.ImageNorm1);
            AddLayerParams(allParams, block.ImageQProj);
            AddLayerParams(allParams, block.ImageKProj);
            AddLayerParams(allParams, block.ImageVProj);
            AddLayerParams(allParams, block.ImageOutProj);
            AddLayerParams(allParams, block.ImageNorm2);
            AddLayerParams(allParams, block.ImageMLP1);
            AddLayerParams(allParams, block.ImageMLP2);
            AddLayerParams(allParams, block.ImageAdaLN);
            AddLayerParams(allParams, block.TextNorm1);
            AddLayerParams(allParams, block.TextQProj);
            AddLayerParams(allParams, block.TextKProj);
            AddLayerParams(allParams, block.TextVProj);
            AddLayerParams(allParams, block.TextOutProj);
            AddLayerParams(allParams, block.TextNorm2);
            AddLayerParams(allParams, block.TextMLP1);
            AddLayerParams(allParams, block.TextMLP2);
            AddLayerParams(allParams, block.TextAdaLN);
        }

        foreach (var block in _singleBlocks)
        {
            AddLayerParams(allParams, block.Norm);
            AddLayerParams(allParams, block.QProj);
            AddLayerParams(allParams, block.KProj);
            AddLayerParams(allParams, block.VProj);
            AddLayerParams(allParams, block.OutProj);
            AddLayerParams(allParams, block.MLP1);
            AddLayerParams(allParams, block.MLP2);
            AddLayerParams(allParams, block.AdaLN);
        }

        AddLayerParams(allParams, _finalNorm);
        AddLayerParams(allParams, _adalnModulation);
        AddLayerParams(allParams, _outputProj);

        return new Vector<T>(allParams.ToArray());
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int offset = 0;

        offset = SetLayerParams(_patchEmbed, parameters, offset);
        offset = SetLayerParams(_timeEmbed1, parameters, offset);
        offset = SetLayerParams(_timeEmbed2, parameters, offset);
        offset = SetLayerParams(_contextProj, parameters, offset);

        foreach (var block in _jointBlocks)
        {
            offset = SetLayerParams(block.ImageNorm1, parameters, offset);
            offset = SetLayerParams(block.ImageQProj, parameters, offset);
            offset = SetLayerParams(block.ImageKProj, parameters, offset);
            offset = SetLayerParams(block.ImageVProj, parameters, offset);
            offset = SetLayerParams(block.ImageOutProj, parameters, offset);
            offset = SetLayerParams(block.ImageNorm2, parameters, offset);
            offset = SetLayerParams(block.ImageMLP1, parameters, offset);
            offset = SetLayerParams(block.ImageMLP2, parameters, offset);
            offset = SetLayerParams(block.ImageAdaLN, parameters, offset);
            offset = SetLayerParams(block.TextNorm1, parameters, offset);
            offset = SetLayerParams(block.TextQProj, parameters, offset);
            offset = SetLayerParams(block.TextKProj, parameters, offset);
            offset = SetLayerParams(block.TextVProj, parameters, offset);
            offset = SetLayerParams(block.TextOutProj, parameters, offset);
            offset = SetLayerParams(block.TextNorm2, parameters, offset);
            offset = SetLayerParams(block.TextMLP1, parameters, offset);
            offset = SetLayerParams(block.TextMLP2, parameters, offset);
            offset = SetLayerParams(block.TextAdaLN, parameters, offset);
        }

        foreach (var block in _singleBlocks)
        {
            offset = SetLayerParams(block.Norm, parameters, offset);
            offset = SetLayerParams(block.QProj, parameters, offset);
            offset = SetLayerParams(block.KProj, parameters, offset);
            offset = SetLayerParams(block.VProj, parameters, offset);
            offset = SetLayerParams(block.OutProj, parameters, offset);
            offset = SetLayerParams(block.MLP1, parameters, offset);
            offset = SetLayerParams(block.MLP2, parameters, offset);
            offset = SetLayerParams(block.AdaLN, parameters, offset);
        }

        offset = SetLayerParams(_finalNorm, parameters, offset);
        offset = SetLayerParams(_adalnModulation, parameters, offset);
        SetLayerParams(_outputProj, parameters, offset);
    }

    /// <summary>
    /// The full layer list in the EXACT order GetParameters/SetParameters serialize it. Streaming and the
    /// flat path share this sequence so the chunk concatenation stays index-identical to GetParameters.
    /// </summary>
    private IEnumerable<ILayer<T>> MMDiTLayerSequence()
    {
        yield return _patchEmbed;
        yield return _timeEmbed1;
        yield return _timeEmbed2;
        yield return _contextProj;

        foreach (var block in _jointBlocks)
        {
            yield return block.ImageNorm1;
            yield return block.ImageQProj;
            yield return block.ImageKProj;
            yield return block.ImageVProj;
            yield return block.ImageOutProj;
            yield return block.ImageNorm2;
            yield return block.ImageMLP1;
            yield return block.ImageMLP2;
            yield return block.ImageAdaLN;
            yield return block.TextNorm1;
            yield return block.TextQProj;
            yield return block.TextKProj;
            yield return block.TextVProj;
            yield return block.TextOutProj;
            yield return block.TextNorm2;
            yield return block.TextMLP1;
            yield return block.TextMLP2;
            yield return block.TextAdaLN;
        }

        foreach (var block in _singleBlocks)
        {
            yield return block.Norm;
            yield return block.QProj;
            yield return block.KProj;
            yield return block.VProj;
            yield return block.OutProj;
            yield return block.MLP1;
            yield return block.MLP2;
            yield return block.AdaLN;
        }

        yield return _finalNorm;
        yield return _adalnModulation;
        yield return _outputProj;
    }

    /// <summary>
    /// True once this predictor's lazy weights have been materialized (its patch-embed is initialized,
    /// which the first forward triggers). A never-materialized foundation-scale model has nothing to copy,
    /// so internal callers (e.g. a wrapping model's <c>Clone</c>) can skip the multi-GB parameter copy and
    /// stay lazy. Internal: this is clone/streaming materialization plumbing, not public model behavior.
    /// </summary>
    internal bool WeightsMaterialized => _patchEmbed.IsInitialized;

    /// <inheritdoc />
    public override IEnumerable<Tensor<T>> GetParameterChunks()
    {
        // #1624 zero-copy: materialize each layer's lazy weights, then yield its resident trainable
        // tensors BY REFERENCE — one chunk per tensor, in canonical MMDiTLayerSequence × GetTrainable
        // order — instead of concatenating each layer's params into a transient multi-GB Vector<T>
        // (which GC-thrashes/OOMs at >2.1B FLUX/MMDiT scale). Consumers (Clone, LatentDiffusionModelBase)
        // pair Get/SetParameterChunks and count chunks dynamically, so per-tensor framing is consistent.
        foreach (var layer in MMDiTLayerSequence())
        {
            if (layer is not LayerBase<T> lb) continue;
            lb.MaterializeParameters();
            foreach (var t in lb.GetTrainableParameters())
                if (t.Length > 0) yield return t;
        }
    }

    /// <inheritdoc />
    public override void SetParameterChunks(IEnumerable<Tensor<T>> chunks)
    {
        using var e = chunks.GetEnumerator();
        foreach (var layer in MMDiTLayerSequence())
        {
            if (layer is not LayerBase<T> lb) continue;
            lb.MaterializeParameters();
            var dst = lb.GetTrainableParameters();
            // Pull one chunk per non-empty trainable tensor and copy the values IN PLACE
            // (CopyTrainableParametersFrom: no rebinding — which would alias clone↔source — and no flat
            // aggregate). Empty slots stay aligned to keep the per-tensor index identical to the getter.
            bool anyNonEmpty = false;
            foreach (var t in dst) if (t.Length > 0) { anyNonEmpty = true; break; }
            if (!anyNonEmpty) continue;
            var incoming = new Tensor<T>[dst.Count];
            for (int i = 0; i < dst.Count; i++)
            {
                if (dst[i].Length == 0) { incoming[i] = dst[i]; continue; }
                if (!e.MoveNext())
                    throw new System.ArgumentException(
                        "SetParameterChunks received fewer chunks than MMDiT has parameter tensors.",
                        nameof(chunks));
                incoming[i] = e.Current;
            }
            lb.CopyTrainableParametersFrom(incoming);
        }
        if (e.MoveNext())
            throw new System.ArgumentException(
                "SetParameterChunks received more chunks than MMDiT has parameter tensors.",
                nameof(chunks));
    }

    private void AddLayerParams(List<T> allParams, ILayer<T> layer)
    {
        var p = layer.GetParameters();
        for (int i = 0; i < p.Length; i++)
        {
            allParams.Add(p[i]);
        }
    }

    private int SetLayerParams(ILayer<T> layer, Vector<T> parameters, int offset)
    {
        int count = checked((int)layer.ParameterCount);
        var p = new T[count];
        for (int i = 0; i < count; i++)
        {
            p[i] = parameters[offset + i];
        }
        layer.SetParameters(new Vector<T>(p));
        return offset + count;
    }

    #endregion

    #region ICloneable Implementation

    /// <inheritdoc />
    public override INoisePredictor<T> Clone()
    {
        var clone = new MMDiTNoisePredictor<T>(
            inputChannels: _inputChannels,
            hiddenSize: _hiddenSize,
            numJointLayers: _numJointLayers,
            numSingleLayers: _numSingleLayers,
            numHeads: _numHeads,
            patchSize: _patchSize,
            contextDim: _contextDim,
            mlpRatio: _mlpRatio);

        ProbeMaterializeAndCopyInto(clone);
        return clone;
    }

    /// <summary>
    /// Materializes <paramref name="clone"/> through one throwaway probe forward (the same path the
    /// source's weights resolved on) and then copies this predictor's weights into it.
    /// </summary>
    /// <remarks>
    /// The LazyDense weights resolve+allocate through the FORWARD path (EnsureInitializedFromInput)
    /// — a different entry than the SetParameters/GetParameters path (EnsureInitialized). Copying
    /// parameters into a clone whose layers were never forwarded leaves its first real forward to
    /// re-resolve and RNG-initialize along the forward path, discarding the copied values and
    /// diverging from the source (the #1706 HiDream/MMDiTX Clone_ShouldProduceIdenticalOutput
    /// failure). Run one throwaway forward to materialize the clone through the same path the source
    /// used, THEN copy the source's weights so they persist. Shared by the base <see cref="Clone"/>
    /// and the <c>MMDiTXNoisePredictor</c> override so both materialize-then-copy rather than copy
    /// onto unmaterialized layers. Gated on the source having been forwarded (a never-forwarded
    /// foundation-scale model has nothing materialized to copy and must not pay a full forward here).
    /// </remarks>
    protected void ProbeMaterializeAndCopyInto(MMDiTNoisePredictor<T> clone)
    {
        if (!_patchEmbed.IsInitialized) return;

        int probeSpatial = _patchSize * 2;
        var probe = new Tensor<T>(new[] { 1, _inputChannels, probeSpatial, probeSpatial });
        // A null-conditioned probe only materializes the unconditional (image-stream) path.
        // When the source ran conditioned forwards its context projection (_contextProj) and
        // text-stream block layers are materialized, so probe the clone WITH a representative
        // text-conditioning tensor — otherwise those layers stay lazy on the clone and re-init
        // with fresh RNG on the first conditioned forward, diverging from the source.
        Tensor<T>? probeConditioning = _contextProj.IsInitialized
            ? new Tensor<T>(new[] { 1, 1, _contextDim })
            : null;
        clone.PredictNoise(probe, timestep: 0, conditioning: probeConditioning);
        // Layer-by-layer copy: each layer's GetParameters/SetParameters works on its own small
        // vector, so cloning never materializes one contiguous foundation-scale parameter vector
        // (the flat List<T> -> ToArray() path in GetParameters that OOMs at SD3/FLUX scale).
        clone.CopyParametersFrom(this);
        // The probe forward traced a compiled plan over the clone's random init; drop it so
        // the next real forward re-traces against the copied weights.
        clone.InvalidateCompiledPlans();
    }

    /// <summary>
    /// Copies trained weights from <paramref name="source"/> into this predictor layer by layer,
    /// without ever materializing a single contiguous parameter vector. Both predictors enumerate
    /// their layers via the same <see cref="EnumerateLayers"/> order, so each source layer is paired
    /// with its target and copied through that layer's own (small) <c>GetParameters</c>/
    /// <c>SetParameters</c>. This is the foundation-scale-safe path that <see cref="Clone"/> uses
    /// instead of <c>SetParameters(GetParameters())</c>, whose flat allocation OOMs at SD3/FLUX scale.
    /// </summary>
    private void CopyParametersFrom(MMDiTNoisePredictor<T> source)
    {
        Guard.NotNull(source);

        using var src = source.EnumerateLayers().GetEnumerator();
        using var dst = EnumerateLayers().GetEnumerator();

        while (src.MoveNext())
        {
            if (!dst.MoveNext())
            {
                throw new InvalidOperationException(
                    "Clone has fewer layers than the source MMDiT predictor; architectures differ.");
            }

            // Per-layer copy. A lazy source layer reports an empty vector and the corresponding
            // SetParameters is a no-op, which is correct: the clone's matching layer was probed to
            // the same materialization state, so both stay lazy together.
            dst.Current.SetParameters(src.Current.GetParameters());
        }

        if (dst.MoveNext())
        {
            throw new InvalidOperationException(
                "Clone has more layers than the source MMDiT predictor; architectures differ.");
        }
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    #endregion

    #region Block Structures

    /// <summary>
    /// Joint (double-stream) MMDiT block with separate image and text streams
    /// sharing attention but with independent MLPs and AdaLN.
    /// </summary>
    public class MMDiTBlock
    {
        // Image stream
        public required LayerNormalizationLayer<T> ImageNorm1 { get; set; }
        public required LayerNormalizationLayer<T> ImageNorm2 { get; set; }
        public required DenseLayer<T> ImageMLP1 { get; set; }
        public required DenseLayer<T> ImageMLP2 { get; set; }
        public required DenseLayer<T> ImageAdaLN { get; set; }
        public required DenseLayer<T> ImageQProj { get; set; }
        public required DenseLayer<T> ImageKProj { get; set; }
        public required DenseLayer<T> ImageVProj { get; set; }
        public required DenseLayer<T> ImageOutProj { get; set; }

        // Text stream
        public required LayerNormalizationLayer<T> TextNorm1 { get; set; }
        public required LayerNormalizationLayer<T> TextNorm2 { get; set; }
        public required DenseLayer<T> TextMLP1 { get; set; }
        public required DenseLayer<T> TextMLP2 { get; set; }
        public required DenseLayer<T> TextAdaLN { get; set; }
        public required DenseLayer<T> TextQProj { get; set; }
        public required DenseLayer<T> TextKProj { get; set; }
        public required DenseLayer<T> TextVProj { get; set; }
        public required DenseLayer<T> TextOutProj { get; set; }
    }

    /// <summary>
    /// Single-stream MMDiT block (FLUX-style) where text and image tokens
    /// are processed together through shared self-attention and parallel MLP.
    /// </summary>
    public class MMDiTSingleBlock
    {
        public required LayerNormalizationLayer<T> Norm { get; set; }
        public required DenseLayer<T> QProj { get; set; }
        public required DenseLayer<T> KProj { get; set; }
        public required DenseLayer<T> VProj { get; set; }
        public required DenseLayer<T> OutProj { get; set; }
        public required DenseLayer<T> MLP1 { get; set; }
        public required DenseLayer<T> MLP2 { get; set; }
        public required DenseLayer<T> AdaLN { get; set; }
    }

    #endregion

    #region Layer-Level Backpropagation

    protected override Vector<T> GetParameterGradients()
    {
        var allGrads = new List<T>();

        AddLayerGrads(allGrads, _patchEmbed);
        AddLayerGrads(allGrads, _timeEmbed1);
        AddLayerGrads(allGrads, _timeEmbed2);
        AddLayerGrads(allGrads, _contextProj);

        foreach (var block in _jointBlocks)
        {
            AddLayerGrads(allGrads, block.ImageNorm1);
            AddLayerGrads(allGrads, block.ImageQProj);
            AddLayerGrads(allGrads, block.ImageKProj);
            AddLayerGrads(allGrads, block.ImageVProj);
            AddLayerGrads(allGrads, block.ImageOutProj);
            AddLayerGrads(allGrads, block.ImageNorm2);
            AddLayerGrads(allGrads, block.ImageMLP1);
            AddLayerGrads(allGrads, block.ImageMLP2);
            AddLayerGrads(allGrads, block.ImageAdaLN);
            AddLayerGrads(allGrads, block.TextNorm1);
            AddLayerGrads(allGrads, block.TextQProj);
            AddLayerGrads(allGrads, block.TextKProj);
            AddLayerGrads(allGrads, block.TextVProj);
            AddLayerGrads(allGrads, block.TextOutProj);
            AddLayerGrads(allGrads, block.TextNorm2);
            AddLayerGrads(allGrads, block.TextMLP1);
            AddLayerGrads(allGrads, block.TextMLP2);
            AddLayerGrads(allGrads, block.TextAdaLN);
        }

        foreach (var block in _singleBlocks)
        {
            AddLayerGrads(allGrads, block.Norm);
            AddLayerGrads(allGrads, block.QProj);
            AddLayerGrads(allGrads, block.KProj);
            AddLayerGrads(allGrads, block.VProj);
            AddLayerGrads(allGrads, block.OutProj);
            AddLayerGrads(allGrads, block.MLP1);
            AddLayerGrads(allGrads, block.MLP2);
            AddLayerGrads(allGrads, block.AdaLN);
        }

        AddLayerGrads(allGrads, _finalNorm);
        AddLayerGrads(allGrads, _adalnModulation);
        AddLayerGrads(allGrads, _outputProj);

        return new Vector<T>(allGrads.ToArray());
    }

    private static void AddLayerGrads(List<T> list, ILayer<T>? layer)
    {
        if (layer == null) return;
        var g = layer.GetParameterGradients();
        for (int i = 0; i < g.Length; i++) list.Add(g[i]);
    }

    #endregion
}

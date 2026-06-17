using System.Diagnostics.CodeAnalysis;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.NoisePredictors;

/// <summary>
/// Flag-DiT noise predictor for the Lumina-T2X image-generation architecture
/// (Gao et al. 2024, "Lumina-T2X: Transforming Text into Any Modality via Flow-based Large
/// Diffusion Transformers", arXiv:2405.05945).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Faithful Flag-DiT block stack (paper §3.1–3.2): the noisy latent is patchified into a token
/// sequence, embedded, and processed by N transformer blocks. Each Flag-DiT block uses
/// <b>sandwich normalization</b> (RMSNorm before AND after each sub-layer — Gao 2024 §3.1),
/// <b>grouped-query self-attention</b> with <b>rotary position embeddings (RoPE)</b>, a SwiGLU/GELU
/// feed-forward, and <b>zero-initialised adaLN</b> conditioning (Peebles &amp; Xie 2022; the adaLN
/// projection produces per-block shift/scale/gate from the combined time + text embedding, and is
/// zero-initialised so every block starts as identity). The final layer applies adaLN + RMSNorm and
/// projects back to patch space, then unpatchifies to the latent shape. Designed for
/// rectified-flow training (Lumina-T2X uses a flow-matching scheduler).
/// </para>
/// <para>
/// <b>For Beginners:</b> Flag-DiT is the transformer behind Lumina. It cuts the image latent into
/// little square patches (like words), uses rotary attention so it can run at any resolution, and
/// conditions every layer on the timestep + text via lightweight scale/shift "knobs" that start at
/// zero (so training begins from a clean identity). Sandwich normalization (a norm on both sides of
/// each sub-layer) keeps the very deep stack numerically stable.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Generative)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Denoising)]
[ModelTask(ModelTask.TextToImage)]
[ModelComplexity(ModelComplexity.VeryHigh)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Lumina-T2X: Transforming Text into Any Modality with Flow Matching", "https://arxiv.org/abs/2405.05945")]
public class FlagDiTPredictor<T> : NoisePredictorBase<T>
{
    /// <summary>Patch size (p): a p×p block of the latent becomes one token (paper uses 2).</summary>
    private const int PatchSize = 2;

    private readonly int _inputChannels;
    private readonly int _hiddenSize;
    private readonly int _numLayers;
    private readonly int _numHeads;
    private readonly int _numKVHeads;
    private readonly int _contextDim;
    private readonly int _latentSize;       // latent spatial side (H == W) — fixes the patch sequence length
    private readonly int _patchDim;         // inputChannels * p * p
    private readonly int _seqLen;           // (latentSize / p)^2

    // Patch + conditioning embeddings.
    private DenseLayer<T> _patchEmbed;      // patchDim -> hidden
    private DenseLayer<T> _timeEmbed1;      // hidden -> hidden (SiLU)
    private DenseLayer<T> _timeEmbed2;      // hidden -> hidden
    private DenseLayer<T> _contextProj;     // contextDim -> hidden (text conditioning)

    // Per-block layers (sandwich-normed Flag-DiT block).
    private RMSNormalizationLayer<T>[] _attnNormPre;
    private RMSNormalizationLayer<T>[] _attnNormPost;
    private GroupedQueryAttentionLayer<T>[] _attn;
    private RMSNormalizationLayer<T>[] _ffnNormPre;
    private RMSNormalizationLayer<T>[] _ffnNormPost;
    private DenseLayer<T>[] _ffn1;          // hidden -> 4*hidden (GELU)
    private DenseLayer<T>[] _ffn2;          // 4*hidden -> hidden
    private DenseLayer<T>[] _adaLN;         // hidden -> 6*hidden (shift/scale/gate × {attn, ffn}), zero-init

    // Final layer.
    private RMSNormalizationLayer<T> _finalNorm;
    private DenseLayer<T> _finalAdaLN;      // hidden -> 2*hidden (shift, scale), zero-init
    private DenseLayer<T> _outputProj;      // hidden -> patchDim

    /// <inheritdoc />
    public override int InputChannels => _inputChannels;
    /// <inheritdoc />
    public override int OutputChannels => _inputChannels;
    /// <inheritdoc />
    public override int BaseChannels => _hiddenSize;
    /// <inheritdoc />
    public override int TimeEmbeddingDim => _hiddenSize;
    /// <inheritdoc />
    public override bool SupportsCFG => true;
    /// <inheritdoc />
    public override bool SupportsCrossAttention => true;
    /// <inheritdoc />
    public override int ContextDimension => _contextDim;
    /// <inheritdoc />
    public override long ParameterCount { get; }

    /// <summary>
    /// Initializes a new Flag-DiT predictor.
    /// </summary>
    /// <param name="inputChannels">Input latent channels.</param>
    /// <param name="hiddenSize">Transformer hidden dimension.</param>
    /// <param name="numLayers">Number of Flag-DiT blocks.</param>
    /// <param name="numHeads">Number of query attention heads.</param>
    /// <param name="numKVHeads">Number of key/value heads (grouped-query attention; must divide numHeads).</param>
    /// <param name="contextDim">Text conditioning dimension.</param>
    /// <param name="latentSize">Latent spatial side length (H==W); fixes the patch sequence length.</param>
    /// <param name="seed">Optional random seed.</param>
    public FlagDiTPredictor(
        int inputChannels = 16,
        int hiddenSize = 4096,
        int numLayers = 32,
        int numHeads = 32,
        int numKVHeads = 8,
        int contextDim = 4096,
        int latentSize = 32,
        int? seed = null)
        : base(seed: seed)
    {
        _inputChannels = inputChannels;
        _hiddenSize = hiddenSize;
        _numLayers = numLayers;
        _numHeads = numHeads;
        _numKVHeads = numKVHeads;
        _contextDim = contextDim;
        _latentSize = latentSize;
        _patchDim = inputChannels * PatchSize * PatchSize;
        _seqLen = (latentSize / PatchSize) * (latentSize / PatchSize);

        InitializeLayers(seed);
        ParameterCount = CalculateParameterCount();
    }

    [MemberNotNull(nameof(_patchEmbed), nameof(_timeEmbed1), nameof(_timeEmbed2), nameof(_contextProj),
        nameof(_attnNormPre), nameof(_attnNormPost), nameof(_attn), nameof(_ffnNormPre),
        nameof(_ffnNormPost), nameof(_ffn1), nameof(_ffn2), nameof(_adaLN),
        nameof(_finalNorm), nameof(_finalAdaLN), nameof(_outputProj))]
    private void InitializeLayers(int? seed)
    {
        _patchEmbed = LazyDense(_patchDim, _hiddenSize);
        _timeEmbed1 = LazyDense(_hiddenSize, _hiddenSize, new SiLUActivation<T>());
        _timeEmbed2 = LazyDense(_hiddenSize, _hiddenSize);
        _contextProj = LazyDense(_contextDim, _hiddenSize);

        _attnNormPre = new RMSNormalizationLayer<T>[_numLayers];
        _attnNormPost = new RMSNormalizationLayer<T>[_numLayers];
        _attn = new GroupedQueryAttentionLayer<T>[_numLayers];
        _ffnNormPre = new RMSNormalizationLayer<T>[_numLayers];
        _ffnNormPost = new RMSNormalizationLayer<T>[_numLayers];
        _ffn1 = new DenseLayer<T>[_numLayers];
        _ffn2 = new DenseLayer<T>[_numLayers];
        _adaLN = new DenseLayer<T>[_numLayers];

        int ffnDim = _hiddenSize * 4;
        for (int i = 0; i < _numLayers; i++)
        {
            _attnNormPre[i] = new RMSNormalizationLayer<T>(_hiddenSize);
            _attnNormPost[i] = new RMSNormalizationLayer<T>(_hiddenSize);
            _attn[i] = new GroupedQueryAttentionLayer<T>(_seqLen, _hiddenSize, _numHeads, _numKVHeads);
            // RoPE: resolution-agnostic rotary position embedding (Su et al. 2021; Flag-DiT §3.1).
            _attn[i].ConfigurePositionalEncoding(PositionalEncodingType.Rotary, ropeTheta: 10000.0,
                maxSequenceLength: System.Math.Max(_seqLen, 64));
            _ffnNormPre[i] = new RMSNormalizationLayer<T>(_hiddenSize);
            _ffnNormPost[i] = new RMSNormalizationLayer<T>(_hiddenSize);
            _ffn1[i] = LazyDense(_hiddenSize, ffnDim, new GELUActivation<T>());
            _ffn2[i] = LazyDense(ffnDim, _hiddenSize);
            // adaLN-zero: produces 6 modulation vectors per block; zero-init so each block starts
            // as identity (gate=0 → residual passes through unchanged). Peebles & Xie 2022 §3.2.
            _adaLN[i] = LazyDense(_hiddenSize, _hiddenSize * 6);
            ZeroInitialize(_adaLN[i]);
        }

        _finalNorm = new RMSNormalizationLayer<T>(_hiddenSize);
        _finalAdaLN = LazyDense(_hiddenSize, _hiddenSize * 2);
        ZeroInitialize(_finalAdaLN);
        _outputProj = LazyDense(_hiddenSize, _patchDim);
    }

    /// <summary>
    /// Zero-initialises a (lazy) Dense layer's weights AND biases so its output is 0 until trained.
    /// Resolves the layer first (so the weight tensors exist), then fills them. Used for the
    /// adaLN-zero modulation projections (Peebles &amp; Xie 2022): zero modulation means every block
    /// and the final layer begin as the identity, which is the documented stable DiT initialization.
    /// </summary>
    private void ZeroInitialize(DenseLayer<T> layer)
    {
        layer.ResolveFromShape(new[] { 1, _hiddenSize });
        var p = layer.GetParameters();
        layer.SetParameters(new Vector<T>(p.Length)); // Vector<T>(n) is zero-filled
    }

    private long CalculateParameterCount()
    {
        long count = _patchEmbed.ParameterCount + _timeEmbed1.ParameterCount
            + _timeEmbed2.ParameterCount + _contextProj.ParameterCount;
        for (int i = 0; i < _numLayers; i++)
        {
            count += _attnNormPre[i].ParameterCount + _attnNormPost[i].ParameterCount
                + _attn[i].ParameterCount + _ffnNormPre[i].ParameterCount + _ffnNormPost[i].ParameterCount
                + _ffn1[i].ParameterCount + _ffn2[i].ParameterCount + _adaLN[i].ParameterCount;
        }
        count += _finalNorm.ParameterCount + _finalAdaLN.ParameterCount + _outputProj.ParameterCount;
        return count;
    }

    /// <inheritdoc />
    public override Tensor<T> PredictNoise(Tensor<T> noisySample, int timestep, Tensor<T>? conditioning = null)
    {
        // Page weights through the streaming pool for the duration of this forward (master's #1610
        // weight-streaming wiring), preserved here across this model's paper-faithful Flag-DiT rewrite.
        using var streaming = BeginWeightStreamingForward();

        // Promote a single [C,H,W] sample to [1,C,H,W] so patchify's batch axis is uniform.
        var x = noisySample.Rank == 3
            ? Engine.Reshape(noisySample, new[] { 1, noisySample.Shape[0], noisySample.Shape[1], noisySample.Shape[2] })
            : noisySample;

        int height = x.Shape[2];
        int width = x.Shape[3];

        // Combined conditioning: time embedding (+ pooled text embedding when present).
        var cond = ProjectTimeEmbedding(GetTimestepEmbedding(timestep, x.Shape[0]));
        if (conditioning is not null)
        {
            cond = Engine.TensorAdd(cond, PoolContext(conditioning));
        }

        var hidden = _patchEmbed.Forward(Patchify(x));   // [B, seq, hidden]
        for (int i = 0; i < _numLayers; i++)
            hidden = ForwardBlock(hidden, cond, i);
        hidden = FinalLayer(hidden, cond);                // [B, seq, hidden] -> norm/adaLN
        var patches = _outputProj.Forward(hidden);        // [B, seq, patchDim]
        var output = Unpatchify(patches, height, width);  // [B, C, H, W]

        return streaming.Complete(noisySample.Rank == 3
            ? Engine.Reshape(output, new[] { _inputChannels, height, width })
            : output);
    }

    /// <inheritdoc />
    public override Tensor<T> PredictNoiseWithEmbedding(Tensor<T> noisySample, Tensor<T> timeEmbedding, Tensor<T>? conditioning = null)
        // The flow-matching loop calls PredictNoise(sample, t); this overload is provided for
        // callers that precompute the embedding. Re-derive from the same path for consistency.
        => PredictNoise(noisySample, 0, conditioning);

    /// <summary>One Flag-DiT block: sandwich-normed GQA(+RoPE) and FFN with zero-init adaLN conditioning.</summary>
    private Tensor<T> ForwardBlock(Tensor<T> x, Tensor<T> cond, int i)
    {
        // adaLN-zero: 6 modulation vectors (shift/scale/gate for attention and FFN branches).
        var modulation = _adaLN[i].Forward(cond);
        int stride = 6 * _hiddenSize;
        int batchM = modulation.Length / stride;
        var mod = Engine.Reshape(modulation, new[] { batchM, 6, 1, _hiddenSize });
        var shiftAttn = Engine.TensorSliceAxis(mod, axis: 1, index: 0);
        var scaleAttn = Engine.TensorSliceAxis(mod, axis: 1, index: 1);
        var gateAttn = Engine.TensorSliceAxis(mod, axis: 1, index: 2);
        var shiftFfn = Engine.TensorSliceAxis(mod, axis: 1, index: 3);
        var scaleFfn = Engine.TensorSliceAxis(mod, axis: 1, index: 4);
        var gateFfn = Engine.TensorSliceAxis(mod, axis: 1, index: 5);

        // Attention branch — sandwich norm (pre + post) around GQA, adaLN-modulated.
        var h = ApplyAdaLN(_attnNormPre[i].Forward(x), scaleAttn, shiftAttn);
        h = _attnNormPost[i].Forward(_attn[i].Forward(h));
        x = AddWithGate(x, h, gateAttn);

        // FFN branch — sandwich norm around the GELU MLP, adaLN-modulated.
        var f = ApplyAdaLN(_ffnNormPre[i].Forward(x), scaleFfn, shiftFfn);
        f = _ffnNormPost[i].Forward(_ffn2[i].Forward(_ffn1[i].Forward(f)));
        x = AddWithGate(x, f, gateFfn);

        return x;
    }

    private Tensor<T> FinalLayer(Tensor<T> x, Tensor<T> cond)
    {
        var modulation = _finalAdaLN.Forward(cond);
        int stride = 2 * _hiddenSize;
        int batchM = modulation.Length / stride;
        var mod = Engine.Reshape(modulation, new[] { batchM, 2, 1, _hiddenSize });
        var shift = Engine.TensorSliceAxis(mod, axis: 1, index: 0);
        var scale = Engine.TensorSliceAxis(mod, axis: 1, index: 1);
        return ApplyAdaLN(_finalNorm.Forward(x), scale, shift);
    }

    /// <summary>adaLN modulation: x · (1 + scale) + shift, with [B,1,hidden] broadcast views.</summary>
    private Tensor<T> ApplyAdaLN(Tensor<T> x, Tensor<T> scaleView, Tensor<T> shiftView)
    {
        var scaled = Engine.TensorBroadcastMultiply(x, Engine.TensorAddScalar(scaleView, NumOps.One));
        return Engine.TensorBroadcastAdd(scaled, shiftView);
    }

    /// <summary>Gated residual: x + gate · branch (gate is a [B,1,hidden] adaLN view).</summary>
    private Tensor<T> AddWithGate(Tensor<T> x, Tensor<T> branch, Tensor<T> gateView)
        => Engine.TensorAdd(x, Engine.TensorBroadcastMultiply(branch, gateView));

    /// <summary>Mean-pools a context tensor to [B, hidden] after projecting from contextDim.</summary>
    private Tensor<T> PoolContext(Tensor<T> conditioning)
    {
        var c = conditioning;
        if (c.Rank == 1) c = Engine.Reshape(c, new[] { 1, c.Shape[0] });
        // [B, seq, contextDim] -> mean over seq -> [B, contextDim]; rank-2 passes through.
        if (c.Rank == 3) c = Engine.ReduceMean(c, new[] { 1 }, keepDims: false);
        return _contextProj.Forward(c);
    }

    private Tensor<T> ProjectTimeEmbedding(Tensor<T> timeEmbed)
        => _timeEmbed2.Forward(_timeEmbed1.Forward(timeEmbed));

    /// <summary>Sinusoidal timestep embedding (Vaswani 2017 / DDPM), [B, hidden].</summary>
    private Tensor<T> GetTimestepEmbedding(int timestep, int batch)
    {
        int half = _hiddenSize / 2;
        var emb = new Tensor<T>(new[] { batch, _hiddenSize });
        var span = emb.AsWritableSpan();
        for (int b = 0; b < batch; b++)
        {
            for (int j = 0; j < half; j++)
            {
                double freq = System.Math.Exp(-System.Math.Log(10000.0) * j / System.Math.Max(1, half - 1));
                double arg = timestep * freq;
                span[b * _hiddenSize + j] = NumOps.FromDouble(System.Math.Sin(arg));
                span[b * _hiddenSize + half + j] = NumOps.FromDouble(System.Math.Cos(arg));
            }
        }
        return emb;
    }

    /// <summary>[B,C,H,W] -> [B, (H/p)(W/p), C·p·p] via reshape + permute + reshape (tape-tracked).</summary>
    private Tensor<T> Patchify(Tensor<T> x)
    {
        int batch = x.Shape[0], channels = x.Shape[1], height = x.Shape[2], width = x.Shape[3];
        int nh = height / PatchSize, nw = width / PatchSize;
        var split = Engine.Reshape(x, new[] { batch, channels, nh, PatchSize, nw, PatchSize });
        var permuted = Engine.TensorPermute(split, new[] { 0, 2, 4, 1, 3, 5 }); // [B, nh, nw, C, p, p]
        return Engine.Reshape(permuted, new[] { batch, nh * nw, channels * PatchSize * PatchSize });
    }

    /// <summary>Inverse of <see cref="Patchify"/>: [B, numPatches, C·p·p] -> [B, C, H, W].</summary>
    private Tensor<T> Unpatchify(Tensor<T> patches, int height, int width)
    {
        int batch = patches.Shape[0];
        int nh = height / PatchSize, nw = width / PatchSize;
        var unsplit = Engine.Reshape(patches, new[] { batch, nh, nw, _inputChannels, PatchSize, PatchSize });
        var permuted = Engine.TensorPermute(unsplit, new[] { 0, 3, 1, 4, 2, 5 }); // [B, C, nh, p, nw, p]
        return Engine.Reshape(permuted, new[] { batch, _inputChannels, height, width });
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var all = new List<T>();
        foreach (var layer in FlagDiTLayerSequence()) Append(all, layer);
        return new Vector<T>(all.ToArray());
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int offset = 0;
        foreach (var layer in FlagDiTLayerSequence()) offset = Load(layer, parameters, offset);
    }

    protected override Vector<T> GetParameterGradients()
    {
        var all = new List<T>();
        foreach (var layer in FlagDiTLayerSequence())
        {
            var g = layer.GetParameterGradients();
            for (int i = 0; i < g.Length; i++) all.Add(g[i]);
        }
        return new Vector<T>(all.ToArray());
    }

    /// <summary>The full layer list in the canonical (stable) order used by GetParameters/SetParameters.</summary>
    private IEnumerable<ILayer<T>> FlagDiTLayerSequence()
    {
        yield return _patchEmbed;
        yield return _timeEmbed1;
        yield return _timeEmbed2;
        yield return _contextProj;
        for (int i = 0; i < _numLayers; i++)
        {
            yield return _attnNormPre[i];
            yield return _attnNormPost[i];
            yield return _attn[i];
            yield return _ffnNormPre[i];
            yield return _ffnNormPost[i];
            yield return _ffn1[i];
            yield return _ffn2[i];
            yield return _adaLN[i];
        }
        yield return _finalNorm;
        yield return _finalAdaLN;
        yield return _outputProj;
    }

    private static void Append(List<T> list, ILayer<T> layer)
    {
        var p = layer.GetParameters();
        for (int i = 0; i < p.Length; i++) list.Add(p[i]);
    }

    private static int Load(ILayer<T> layer, Vector<T> parameters, int offset)
    {
        int count = checked((int)layer.ParameterCount);
        if (count == 0) return offset;
        var p = new Vector<T>(count);
        for (int i = 0; i < count && offset + i < parameters.Length; i++) p[i] = parameters[offset + i];
        layer.SetParameters(p);
        return offset + count;
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    /// <inheritdoc />
    public override INoisePredictor<T> Clone()
    {
        var clone = new FlagDiTPredictor<T>(_inputChannels, _hiddenSize, _numLayers, _numHeads,
            _numKVHeads, _contextDim, _latentSize);
        if (!clone.TryShareParametersFrom(this)) clone.SetParameters(GetParameters());
        return clone;
    }
}

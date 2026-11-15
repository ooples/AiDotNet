using AiDotNet.Attributes;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.NoisePredictors;

/// <summary>
/// A point in the U-Net forward pass, after the middle block or after one LDM output block, where a model can
/// transform the feature map before the next block reads it.
/// </summary>
/// <remarks>
/// An LDM output block (<c>openaimodel.py output_blocks[i]</c>) is one residual block with its attention, plus the
/// level's upsampling when it is the level's last block. SyncDreamer (Liu et al. 2023) inserts its depth-wise
/// attention at exactly these points: after the middle block and after output blocks
/// (<c>DepthWiseAttention.forward</c>).
/// </remarks>
internal readonly struct UNetFeatureSite
{
    public UNetFeatureSite(int decoderBlock, int level, bool isUpsample, int channels, int scale)
    {
        DecoderBlock = decoderBlock;
        Level = level;
        IsUpsample = isUpsample;
        Channels = channels;
        Scale = scale;
    }

    /// <summary>The LDM output block this site follows, or -1 for the middle block.</summary>
    public int DecoderBlock { get; }

    /// <summary>Whether this site follows the middle block.</summary>
    public bool IsMiddle => DecoderBlock < 0;

    /// <summary>The U-Net level the block belongs to (0 is the full latent resolution).</summary>
    public int Level { get; }

    /// <summary>Whether the output block ends with the upsampling that leaves <see cref="Level"/>.</summary>
    public bool IsUpsample { get; }

    /// <summary>The width of the feature map at this site.</summary>
    public int Channels { get; }

    /// <summary>The feature map is the latent side divided by 2^Scale here.</summary>
    public int Scale { get; }
}

public partial class UNetNoisePredictor<T>
{
    /// <summary>The residual blocks per level.</summary>
    internal int NumResBlocksPerLevel => _numResBlocks;

    /// <summary>The (input, output) channels of every default decoder residual block, in forward order.</summary>
    internal IReadOnlyList<(int In, int Out)> DecoderResBlockChannels => _decoderResBlockChannels;

    /// <summary>The time MLP's two linear layers (LDM <c>time_embed.0</c> and <c>time_embed.2</c>).</summary>
    internal (DenseLayer<T> First, DenseLayer<T> Second) TimeEmbeddingLayers
    {
        get
        {
            EnsureLayersInitialized();
            return (_timeEmbedMlp1, _timeEmbedMlp2);
        }
    }

    /// <summary>The time MLP applied to a sinusoidal embedding, [B, 4C].</summary>
    internal Tensor<T> ProjectTimestepEmbedding(Tensor<T> sinusoid)
    {
        EnsureLayersInitialized();
        return ProjectTimeEmbedding(sinusoid);
    }

    [Scratch]
    private readonly Dictionary<int, Tensor<T>> _ldmTimestepEmbeddingCache = new();

    [Scratch]
    private Tensor<T>? _ldmTimestepFrequencies;

    /// <summary>
    /// LDM's sinusoidal timestep embedding (<c>openaimodel.py timestep_embedding(t, model_channels)</c>), [1, C]
    /// with C the base channel count: <c>[cos(t f_i), sin(t f_i)]</c> for <c>f_i = exp(-ln(10000) i / (C / 2))</c>,
    /// zero-padded when C is odd. The time MLP lifts it to <see cref="TimeEmbeddingDim"/> = 4C.
    /// </summary>
    /// <param name="timestep">The timestep.</param>
    public override Tensor<T> GetTimestepEmbedding(int timestep)
    {
        ThrowIfDisposed();
        if (_ldmTimestepEmbeddingCache.TryGetValue(timestep, out var cached)) return cached;
        var embedding = new Tensor<double>(LdmTimestepEmbedding(timestep, _baseChannels), new[] { 1, _baseChannels }).Cast<T>();
        _ldmTimestepEmbeddingCache[timestep] = embedding;
        return embedding;
    }

    /// <summary>
    /// The LDM timestep embedding of a [B] tensor of timesteps, [B, C], from engine ops; row b equals
    /// <see cref="GetTimestepEmbedding(int)"/> of timestep b.
    /// </summary>
    /// <param name="timesteps">The timesteps.</param>
    public override Tensor<T> GetTimestepEmbedding(Tensor<T> timesteps)
    {
        ThrowIfDisposed();
        if (timesteps is null) throw new ArgumentNullException(nameof(timesteps));
        int batch = timesteps.Length;
        int half = _baseChannels / 2;
        var frequencies = _ldmTimestepFrequencies ??= new Tensor<double>(LdmFrequencies(half), new[] { 1, half }).Cast<T>();
        var angles = Engine.TensorMultiply(Engine.Reshape(timesteps, new[] { batch, 1 }), frequencies);   // [B, half]
        var parts = _baseChannels % 2 == 0
            ? new[] { Engine.TensorCos(angles), Engine.TensorSin(angles) }
            : new[] { Engine.TensorCos(angles), Engine.TensorSin(angles), new Tensor<T>(new[] { batch, 1 }) };
        return Engine.TensorConcatenate(parts, 1);
    }

    /// <summary>LDM's <c>timestep_embedding(t, dim)</c> in double precision.</summary>
    internal static double[] LdmTimestepEmbedding(int timestep, int dim)
    {
        int half = dim / 2;
        var frequencies = LdmFrequencies(half);
        var embedding = new double[dim];
        for (int i = 0; i < half; i++)
        {
            double angle = timestep * frequencies[i];
            embedding[i] = Math.Cos(angle);
            embedding[half + i] = Math.Sin(angle);
        }

        return embedding;
    }

    private static double[] LdmFrequencies(int half)
    {
        var frequencies = new double[half];
        for (int i = 0; i < half; i++) frequencies[i] = Math.Exp(-Math.Log(10000.0) * i / half);
        return frequencies;
    }

    /// <summary>
    /// Every hook site of the default-built network, in forward order: the middle block first, then each LDM output
    /// block (per level from the deepest: numResBlocks + 1 blocks, the last of which also upsamples to the next level).
    /// </summary>
    internal IReadOnlyList<UNetFeatureSite> FeatureSites()
    {
        int levels = _channelMultipliers.Length;
        var sites = new List<UNetFeatureSite>
        {
            new(-1, levels - 1, false, _baseChannels * _channelMultipliers[levels - 1], levels - 1),
        };
        int index = 0;
        for (int level = levels - 1; level >= 0; level--)
        {
            int channels = _baseChannels * _channelMultipliers[level];
            for (int block = 0; block <= _numResBlocks; block++)
            {
                bool upsamples = level > 0 && block == _numResBlocks;
                sites.Add(new UNetFeatureSite(index++, level, upsamples, channels, upsamples ? level - 1 : level));
            }
        }

        return sites;
    }

    /// <summary>
    /// Predicts noise as <see cref="PredictNoise"/> does, letting <paramref name="hook"/> replace the feature map at every
    /// <see cref="FeatureSites"/> site. The hook returns its input unchanged for a site it does not condition.
    /// </summary>
    /// <param name="noisySample">The noisy latent, [B, C, H, W].</param>
    /// <param name="timestep">The timestep.</param>
    /// <param name="conditioning">Cross-attention context, [B, L, context], or null.</param>
    /// <param name="hook">Receives each site and its feature map; returns the feature map to continue with.</param>
    /// <returns>The predicted noise.</returns>
    /// <exception cref="InvalidOperationException">The decoder was not built by the default layout, so the sites do not
    /// describe it.</exception>
    internal Tensor<T> PredictNoiseWithFeatureHook(Tensor<T> noisySample, int timestep, Tensor<T>? conditioning,
        Func<UNetFeatureSite, Tensor<T>, Tensor<T>> hook)
    {
        if (noisySample is null) throw new ArgumentNullException(nameof(noisySample));
        if (hook is null) throw new ArgumentNullException(nameof(hook));
        EnsureLayersInitialized();
        var sites = FeatureSites();
        int expected = sites.Count - 1 + (_channelMultipliers.Length - 1);
        if (!_ldmSkipLayout || _decoderBlocks.Count != expected || _middleBlocks.Count == 0)
            throw new InvalidOperationException(
                $"Feature hooks need the default LDM decoder layout: {expected} decoder blocks were expected but the " +
                $"network has {_decoderBlocks.Count}.");

        using var streaming = LayerBase<T>.IsInferringShapes ? null : BeginWeightStreamingForward();
        if (!LayerBase<T>.IsInferringShapes)
            _preserveMaterializedParameters = true;
        _lastInput = noisySample;

        // Inference runs in eval mode, as PredictNoiseCore does; a live tape keeps training mode.
        bool evalForInference = !LayerBase<T>.IsInferringShapes
            && !AiDotNet.Autodiff.GradientRecording.IsRecording<T>();
        if (evalForInference) SetAllLayersTrainingMode(false);
        try
        {
            var timeEmbed = ProjectTimeEmbedding(GetTimestepEmbedding(timestep));
            var output = ForwardUNetHooked(noisySample, timeEmbed, conditioning, sites, hook);
            _lastOutput = output;
            return streaming is null ? output : streaming.Complete(output);
        }
        finally
        {
            if (evalForInference) SetAllLayersTrainingMode(true);
        }
    }

    // ForwardUNet with the hook applied after the middle block and after every LDM output block.
    private Tensor<T> ForwardUNetHooked(Tensor<T> x, Tensor<T> timeEmbed, Tensor<T>? conditioning,
        IReadOnlyList<UNetFeatureSite> sites, Func<UNetFeatureSite, Tensor<T>, Tensor<T>> hook)
    {
        if (_inputConv is null || _outputConv is null)
            throw new InvalidOperationException("Convolutional layers not initialized.");

        x = _inputConv.Forward(x);
        var skips = new List<Tensor<T>>(_encoderBlocks.Count + 1) { x };
        foreach (var block in _encoderBlocks)
        {
            if (block.Downsample is not null)
            {
                x = block.Downsample.Forward(x);
                skips.Add(x);
                continue;
            }

            x = ApplyResBlock(block.ResBlock, x, timeEmbed);
            if (block.AttentionBlock is not null) x = block.AttentionBlock.Forward(x);
            if (block.CrossAttentionBlock is not null && conditioning is not null)
                x = ApplyCrossAttention(block.CrossAttentionBlock, x, conditioning);
            skips.Add(x);
        }

        foreach (var block in _middleBlocks)
        {
            x = ApplyResBlock(block.ResBlock, x, timeEmbed);
            if (block.AttentionBlock is not null) x = block.AttentionBlock.Forward(x);
            if (block.CrossAttentionBlock is not null && conditioning is not null)
                x = ApplyCrossAttention(block.CrossAttentionBlock, x, conditioning);
        }

        x = hook(sites[0], x);

        int skipIndex = skips.Count - 1;
        int site = 1;
        for (int i = 0; i < _decoderBlocks.Count; i++)
        {
            var block = _decoderBlocks[i];
            if (block.Upsample is not null)
            {
                x = block.Upsample.Forward(x);
            }
            else
            {
                if (skipIndex >= 0)
                {
                    x = ConcatenateChannels(x, skips[skipIndex]);
                    skipIndex--;
                }

                x = ApplyResBlock(block.ResBlock, x, timeEmbed);
                if (block.AttentionBlock is not null) x = block.AttentionBlock.Forward(x);
                if (block.CrossAttentionBlock is not null && conditioning is not null)
                    x = ApplyCrossAttention(block.CrossAttentionBlock, x, conditioning);
            }

            // An output block ends after its residual block unless the level's upsampling follows it.
            bool upsampleFollows = i + 1 < _decoderBlocks.Count && _decoderBlocks[i + 1].Upsample is not null;
            if (block.Upsample is not null || !upsampleFollows)
                x = hook(sites[site++], x);
        }

        return _outputConv.Forward(x);
    }
}

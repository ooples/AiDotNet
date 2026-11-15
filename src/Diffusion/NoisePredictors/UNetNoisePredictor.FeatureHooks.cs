using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.NoisePredictors;

/// <summary>
/// A point in the U-Net forward pass, after the middle block or after one decoder block, where a model can transform
/// the feature map before the next block reads it.
/// </summary>
/// <remarks>
/// SyncDreamer (Liu et al. 2023) inserts its depth-wise attention at exactly these points of the latent-diffusion
/// U-Net: after the middle block and after the decoder blocks (<c>DepthWiseAttention.forward</c>).
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

    /// <summary>The decoder block this site follows, or -1 for the middle block.</summary>
    public int DecoderBlock { get; }

    /// <summary>Whether this site follows the middle block.</summary>
    public bool IsMiddle => DecoderBlock < 0;

    /// <summary>The U-Net level the block belongs to (0 is the full latent resolution).</summary>
    public int Level { get; }

    /// <summary>Whether the block is the upsampling that leaves <see cref="Level"/>.</summary>
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

    /// <summary>
    /// Every hook site of the default-built network, in forward order: the middle block first, then each decoder block
    /// (per level from the deepest: the residual blocks, then the upsampling to the next level).
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
            for (int block = 0; block < _numResBlocks; block++)
                sites.Add(new UNetFeatureSite(index++, level, false, channels, level));
            if (level > 0)
                sites.Add(new UNetFeatureSite(index++, level, true, channels, level - 1));
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
        if (_decoderBlocks.Count != sites.Count - 1 || _middleBlocks.Count == 0)
            throw new InvalidOperationException(
                $"Feature hooks need the default decoder layout: {sites.Count - 1} decoder blocks were expected but the " +
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

    // ForwardUNet with the hook applied after the middle block and after every decoder block.
    private Tensor<T> ForwardUNetHooked(Tensor<T> x, Tensor<T> timeEmbed, Tensor<T>? conditioning,
        IReadOnlyList<UNetFeatureSite> sites, Func<UNetFeatureSite, Tensor<T>, Tensor<T>> hook)
    {
        if (_inputConv is null || _outputConv is null)
            throw new InvalidOperationException("Convolutional layers not initialized.");

        x = _inputConv.Forward(x);
        var skips = new List<Tensor<T>>(_encoderBlocks.Count);
        foreach (var block in _encoderBlocks)
        {
            if (block.Downsample is not null)
            {
                x = block.Downsample.Forward(x);
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

            x = hook(sites[i + 1], x);
        }

        return _outputConv.Forward(x);
    }
}

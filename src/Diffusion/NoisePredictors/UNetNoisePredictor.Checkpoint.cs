using AiDotNet.ModelLoading;

namespace AiDotNet.Diffusion.NoisePredictors;

/// <summary>
/// Loading an official LDM / Stable Diffusion U-Net checkpoint (<c>ldm/modules/diffusionmodules/openaimodel.py</c>
/// <c>UNetModel</c> state-dict names) into the default-built U-Net.
/// </summary>
public partial class UNetNoisePredictor<T>
{
    /// <summary>
    /// Loads an official LDM U-Net state dict: <c>time_embed.{0,2}</c>, <c>input_blocks.0.0</c>, every
    /// <c>input_blocks.{i}.0</c> residual block and <c>input_blocks.{i}.0.op</c> downsampling,
    /// <c>middle_block.{0,1,2}</c>, every <c>output_blocks.{j}.0</c> residual block and its
    /// <c>output_blocks.{j}.{1|2}.conv</c> upsampling, the <c>.1</c> spatial transformers, and <c>out.{0,2}</c>.
    /// </summary>
    /// <param name="state">The checkpoint tensors by name, for example read from a safetensors file.</param>
    /// <param name="prefix">The U-Net's prefix in the checkpoint; <c>model.diffusion_model.</c> for an LDM checkpoint.</param>
    /// <param name="allowPartial">
    /// When false (the default) a tensor this U-Net needs but the checkpoint lacks throws, and so does a checkpoint
    /// tensor under <paramref name="prefix"/> this U-Net has no module for. When true, modules with missing tensors keep
    /// their current weights and unknown tensors are ignored.
    /// </param>
    /// <returns>The names loaded, in order.</returns>
    /// <remarks>
    /// Requires the default-built layout. Attention loads into the LDM <c>SpatialTransformer</c> the multi-view option
    /// builds (<see cref="MultiViewUNetOptions.StableDiffusion1"/>, <see cref="MultiViewUNetOptions.StableDiffusion2"/>
    /// or MVDream's); a U-Net with another attention module cannot load an LDM transformer, so it throws unless
    /// <paramref name="allowPartial"/> is set.
    /// </remarks>
    public IReadOnlyList<string> LoadTorchWeights(IReadOnlyDictionary<string, Tensor<float>> state,
        string prefix = "model.diffusion_model.", bool allowPartial = false)
    {
        if (state is null) throw new ArgumentNullException(nameof(state));
        return LoadTorchWeights(TorchTensorSource.From(state), state.Keys, prefix, allowPartial);
    }

    /// <summary>
    /// Loads an official LDM U-Net checkpoint from <paramref name="source"/>; see the dictionary overload.
    /// <paramref name="checkpointNames"/>, when given, lists every checkpoint tensor so unknown names can be found.
    /// </summary>
    internal IReadOnlyList<string> LoadTorchWeights(TorchTensorSource source, IEnumerable<string>? checkpointNames,
        string prefix = "model.diffusion_model.", bool allowPartial = false)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        if (prefix is null) throw new ArgumentNullException(nameof(prefix));
        return LdmCheckpoint.LoadUnits(source, LdmCheckpointUnits(prefix, allowPartial), checkpointNames, prefix, allowPartial);
    }

    /// <summary>Every official tensor name this U-Net loads, in load order.</summary>
    internal IReadOnlyList<string> LdmTensorNames(string prefix = "model.diffusion_model.")
        => LdmCheckpointUnits(prefix, allowPartial: true).SelectMany(unit => unit.Names).ToList();

    private IEnumerable<LdmCheckpointUnit> LdmCheckpointUnits(string prefix, bool allowPartial)
    {
        EnsureLayersInitialized();
        if (!_ldmSkipLayout)
            throw new InvalidOperationException(
                "Only the default-built U-Net follows the LDM checkpoint layout; this one was built from custom blocks.");

        var units = new List<LdmCheckpointUnit>();
        void Conv(ConvolutionalLayer<T> conv, string name, int inputs, int outputs, int kernel)
            => units.Add(new LdmCheckpointUnit(LdmCheckpoint.Pair(name),
                source => LdmCheckpoint.Conv(source, conv, name, inputs, outputs, kernel)));
        void Dense(DenseLayer<T> layer, string name, int inputs, int outputs)
            => units.Add(new LdmCheckpointUnit(LdmCheckpoint.Pair(name),
                source => LdmCheckpoint.Linear(source, layer, name, inputs, outputs)));
        void Res(ILayer<T>? layer, string name)
        {
            var block = layer as DiffusionResBlock<T> ?? throw new InvalidOperationException(
                $"The U-Net module at {name} is not a DiffusionResBlock and cannot load an LDM ResBlock.");
            units.Add(new LdmCheckpointUnit(block.LdmTensorNames(name), source => block.LoadTorchWeights(source, name)));
        }
        void Attention(UNetBlock block, string name)
        {
            if (block.AttentionBlock is null && block.CrossAttentionBlock is null) return;
            var transformer = Unwrap(block.CrossAttentionBlock) as MultiViewTransformerLayer<T>
                ?? Unwrap(block.AttentionBlock) as MultiViewTransformerLayer<T>;
            if (transformer is null)
            {
                if (allowPartial) return;
                throw new InvalidOperationException(
                    $"The U-Net attention at {name} is not an LDM SpatialTransformer, so it cannot load the checkpoint's. " +
                    "Build the U-Net with MultiViewUNetOptions.StableDiffusion1() / StableDiffusion2(), or pass allowPartial: true.");
            }
            units.Add(new LdmCheckpointUnit(transformer.LdmTensorNames(name), source => transformer.LoadTorchWeights(source, name)));
        }

        var inputConv = _inputConv ?? throw new InvalidOperationException("The U-Net input convolution is not initialized.");
        var timeEmbed1 = _timeEmbedMlp1 ?? throw new InvalidOperationException("The U-Net time embedding is not initialized.");
        var timeEmbed2 = _timeEmbedMlp2 ?? throw new InvalidOperationException("The U-Net time embedding is not initialized.");
        var outputNorm = _outputNorm ?? throw new InvalidOperationException("The U-Net output norm is not initialized.");
        var outputConv = _outputConv ?? throw new InvalidOperationException("The U-Net output convolution is not initialized.");

        Dense(timeEmbed1, prefix + "time_embed.0", _baseChannels, _timeEmbeddingDim);
        Dense(timeEmbed2, prefix + "time_embed.2", _timeEmbeddingDim, _timeEmbeddingDim);
        Conv(inputConv, prefix + "input_blocks.0.0", _inputChannels, _baseChannels, 3);

        for (int i = 0; i < _encoderBlocks.Count; i++)
        {
            var block = _encoderBlocks[i];
            string name = $"{prefix}input_blocks.{i + 1}.";
            if (block.Downsample is not null)
            {
                var down = block.Downsample as ConvolutionalLayer<T> ?? throw new InvalidOperationException(
                    $"The U-Net downsampling at {name}0.op is not a convolution.");
                int channels = _encoderSkipChannels[i + 1];
                Conv(down, name + "0.op", channels, channels, 3);
                continue;
            }
            Res(block.ResBlock, name + "0.");
            Attention(block, name + "1.");
        }

        for (int m = 0, module = 0; m < _middleBlocks.Count; m++)
        {
            var block = _middleBlocks[m];
            Res(block.ResBlock, $"{prefix}middle_block.{module++}.");
            if (block.AttentionBlock is not null || block.CrossAttentionBlock is not null)
                Attention(block, $"{prefix}middle_block.{module++}.");
        }

        int output = -1;
        bool previousHasAttention = false;
        foreach (var block in _decoderBlocks)
        {
            if (block.Upsample is not null)
            {
                var up = block.Upsample as LdmUpsampleLayer<T> ?? throw new InvalidOperationException(
                    "The U-Net upsampling is not LDM's Upsample.");
                string upName = $"{prefix}output_blocks.{output}.{(previousHasAttention ? 2 : 1)}.";
                units.Add(new LdmCheckpointUnit(LdmCheckpoint.Pair(upName + "conv"), source => up.LoadTorchWeights(source, upName)));
                continue;
            }
            output++;
            string name = $"{prefix}output_blocks.{output}.";
            Res(block.ResBlock, name + "0.");
            Attention(block, name + "1.");
            previousHasAttention = block.AttentionBlock is not null || block.CrossAttentionBlock is not null;
        }

        units.Add(new LdmCheckpointUnit(LdmCheckpoint.Pair(prefix + "out.0"),
            source => LdmCheckpoint.GroupNorm(source, outputNorm, prefix + "out.0")));
        Conv(outputConv, prefix + "out.2", _baseChannels * _channelMultipliers[0], _outputChannels, 3);
        return units;
    }

    private static ILayer<T>? Unwrap(ILayer<T>? layer)
    {
        while (layer is IAttentionBlockDecorator<T> decorator) layer = decorator.Inner;
        return layer;
    }
}

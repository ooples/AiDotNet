namespace AiDotNet.Diffusion.NoisePredictors;

/// <summary>
/// The MVDream / ImageDream multi-view option of the shared U-Net (<see cref="MultiViewUNetOptions"/>). Kept in its own
/// file and members: with the option null none of this runs and the U-Net is built and computed exactly as before.
/// </summary>
public partial class UNetNoisePredictor<T>
{
    /// <summary>The multi-view configuration; null for the standard U-Net.</summary>
    private readonly MultiViewUNetOptions? _multiView;

    /// <summary>
    /// The MVDream / ImageDream multi-view configuration this U-Net was built with, or null for the standard U-Net.
    /// </summary>
    public MultiViewUNetOptions? MultiViewOptions => _multiView;

    // The transformer options every attention level is built with: the multi-view option when given, otherwise the
    // Stable Diffusion SpatialTransformer the context width implies (one view per object).
    private MultiViewUNetOptions? _transformerOptions;

    private MultiViewUNetOptions TransformerOptions
        => _transformerOptions ??= _multiView ?? MultiViewUNetOptions.ForContext(_contextDim, _numHeads);

    /// <summary>
    /// Every multi-view transformer of the default-built U-Net with its official state-dict prefix (after
    /// <c>model.diffusion_model.</c>): encoder block i is LDM's <c>input_blocks.{i+1}</c> (block 0 is the input
    /// convolution), the middle transformer is <c>middle_block.1</c>, and the j-th decoder residual block is
    /// <c>output_blocks.{j}</c>, whose transformer is always its second module (<c>.1</c>).
    /// </summary>
    internal IEnumerable<(string Prefix, MultiViewTransformerLayer<T> Transformer)> MultiViewTransformers()
    {
        EnsureLayersInitialized();
        if (!_ldmSkipLayout) yield break;

        for (int i = 0; i < _encoderBlocks.Count; i++)
        {
            if (Transformer(_encoderBlocks[i]) is { } encoder)
                yield return ($"input_blocks.{i + 1}.1.", encoder);
        }

        foreach (var block in _middleBlocks)
        {
            if (Transformer(block) is { } middle)
                yield return ("middle_block.1.", middle);
        }

        int output = 0;
        foreach (var block in _decoderBlocks)
        {
            if (block.Upsample is not null) continue;
            if (Transformer(block) is { } decoder)
                yield return ($"output_blocks.{output}.1.", decoder);
            output++;
        }
    }

    /// <summary>
    /// The cross-attention slot's forward. With a context it is <see cref="ApplyCrossAttention"/>. Without one only a
    /// multi-view transformer runs (its 3D self-attention and feed-forward, skipping attn2), since it also holds the
    /// U-Net's self-attention; the callers reach the context-free case only with the multi-view option set.
    /// </summary>
    private Tensor<T> ApplyCrossAttentionSlot(ILayer<T>? crossAttn, Tensor<T> x, Tensor<T>? conditioning)
    {
        if (crossAttn is null) return x;
        if (conditioning is not null) return ApplyCrossAttention(crossAttn, x, conditioning);
        if (crossAttn is IContextualAttentionBlock<T> contextual) return contextual.Forward(x, null);
        if (crossAttn is IAttentionBlockDecorator<T> decorator)
            return decorator.PostProcess(ApplyCrossAttentionSlot(decorator.Inner, x, null));
        return crossAttn is MultiViewTransformerLayer<T> transformer ? transformer.Forward(x, null) : x;
    }

    private static MultiViewTransformerLayer<T>? Transformer(UNetBlock block)
        => Unwrap(block.CrossAttentionBlock) as MultiViewTransformerLayer<T> ?? Unwrap(block.AttentionBlock) as MultiViewTransformerLayer<T>;

    /// <summary>
    /// Loads every multi-view transformer's weights under the official MVDream / ImageDream names
    /// (<paramref name="prefix"/> then, for example, <c>input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight</c>).
    /// </summary>
    /// <returns>The names read, in order.</returns>
    internal IReadOnlyList<string> LoadMultiViewTransformerWeights(AiDotNet.ModelLoading.TorchTensorSource source,
        string prefix = "model.diffusion_model.")
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        var names = new List<string>();
        foreach (var (blockPrefix, transformer) in MultiViewTransformers())
            names.AddRange(transformer.LoadTorchWeights(source, prefix + blockPrefix));
        return names;
    }
}

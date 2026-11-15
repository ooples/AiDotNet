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

    // With a context the SpatialTransformer3D sits in the cross-attention slot, where it receives the conditioning;
    // the self-attention slot stays empty, since the transformer holds the (3D) self-attention itself. Without a
    // context (contextDim 0) it sits in the self-attention slot and its attn2 is self-attention, as LDM's is when
    // context_dim is None.
    private ILayer<T>? CreateMultiViewAttentionBlock(int channels)
        => _multiView is null || _contextDim > 0 ? null : new MultiViewTransformerLayer<T>(channels, 0, _multiView);

    private ILayer<T>? CreateMultiViewCrossAttentionBlock(int channels)
        => _multiView is null ? null : new MultiViewTransformerLayer<T>(channels, _contextDim, _multiView);

    /// <summary>
    /// Every multi-view transformer of the default-built U-Net with its official state-dict prefix (after
    /// <c>model.diffusion_model.</c>): encoder block i is LDM's <c>input_blocks.{i+1}</c> (block 0 is the input
    /// convolution), the middle transformer is <c>middle_block.1</c>, and the j-th decoder residual block is
    /// <c>output_blocks.{j}</c>, whose transformer is always its second module (<c>.1</c>).
    /// </summary>
    internal IEnumerable<(string Prefix, MultiViewTransformerLayer<T> Transformer)> MultiViewTransformers()
    {
        EnsureLayersInitialized();
        if (_multiView is null || !_ldmSkipLayout) yield break;

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

    private static MultiViewTransformerLayer<T>? Transformer(UNetBlock block)
        => block.CrossAttentionBlock as MultiViewTransformerLayer<T> ?? block.AttentionBlock as MultiViewTransformerLayer<T>;

    /// <summary>
    /// Loads every multi-view transformer's weights under the official MVDream / ImageDream names
    /// (<paramref name="prefix"/> then, for example, <c>input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight</c>).
    /// </summary>
    /// <returns>The names read, in order.</returns>
    internal IReadOnlyList<string> LoadMultiViewTransformerWeights(AiDotNet.ModelLoading.TorchTensorSource source,
        string prefix = "model.diffusion_model.")
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        if (_multiView is null)
            throw new InvalidOperationException("This U-Net was built without the multi-view option and has no SpatialTransformer3D.");
        var names = new List<string>();
        foreach (var (blockPrefix, transformer) in MultiViewTransformers())
            names.AddRange(transformer.LoadTorchWeights(source, prefix + blockPrefix));
        return names;
    }
}

using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// The RealViformer network (archs/realviformer_arch.py): a unidirectional recurrent loop in which every frame's shallow
/// features query the flow-warped hidden state through the CAF, a three-level Restormer encoder-decoder with channel-
/// masked, channel-compressed transposed attention reconstructs it, a 1x1 convolution makes the next hidden state, and
/// x4 pixel-shuffle upsampling adds to the bilinearly upsampled frame.
/// </summary>
internal sealed class RealViformerBackbone<T>
{
    private readonly int _channels, _shallowConv, _reduce2, _compress, _up1, _up2, _hr, _last;
    private readonly int _down12, _down23, _up32, _up21;
    private readonly RealViformerTransformerBlock<T> _shallowBlock;
    private readonly RealViformerMerge<T> _merge;
    private readonly RealViformerTransformerBlock<T>[] _encoder1, _encoder2, _latent, _decoder2, _decoder1, _refinement;

    /// <param name="layers">The model's layer list.</param>
    /// <param name="channels">num_feat (48).</param>
    /// <param name="blocks">Transformer blocks per level and refinement ([2, 3, 4, 1]).</param>
    /// <param name="heads">Attention heads per level ([1, 2, 4]).</param>
    /// <param name="expansion">GDFN expansion factor (2.66).</param>
    /// <param name="mergeHeads">CAF heads (2).</param>
    /// <param name="squeeze">The attention's channel squeeze factor (4; 1 disables compression).</param>
    /// <param name="masked">Whether the attention carries the channel mask.</param>
    internal RealViformerBackbone(List<ILayer<T>> layers, int channels, int[] blocks, int[] heads, double expansion, int mergeHeads,
        int squeeze, bool masked)
    {
        _channels = channels;
        bool compress = squeeze > 1;
        RealViformerTransformerBlock<T>[] Level(int count, int dim, int h)
            => Enumerable.Range(0, count)
                .Select(_ => new RealViformerTransformerBlock<T>(layers, dim, h, expansion, false, masked, compress, squeeze)).ToArray();

        _shallowConv = layers.Count;
        layers.Add(RealViformerBlocks.Conv<T>(channels, 3, true));
        _shallowBlock = new RealViformerTransformerBlock<T>(layers, channels, heads[0], expansion, false, false, false, squeeze);
        SpyNet = new SpyNetFlowBlock<T>(layers);
        _merge = new RealViformerMerge<T>(layers, channels, mergeHeads, false);
        _encoder1 = Level(blocks[0], channels, heads[0]);
        _down12 = layers.Count;
        layers.Add(RealViformerBlocks.Conv<T>(channels / 2, 3, false));
        _encoder2 = Level(blocks[1], 2 * channels, heads[1]);
        _down23 = layers.Count;
        layers.Add(RealViformerBlocks.Conv<T>(channels, 3, false));
        _latent = Level(blocks[2], 4 * channels, heads[2]);
        _up32 = layers.Count;
        layers.Add(RealViformerBlocks.Conv<T>(8 * channels, 3, false));
        _reduce2 = layers.Count;
        layers.Add(RealViformerBlocks.Conv<T>(2 * channels, 1, false));
        _decoder2 = Level(blocks[1], 2 * channels, heads[1]);
        _up21 = layers.Count;
        layers.Add(RealViformerBlocks.Conv<T>(4 * channels, 3, false));
        _decoder1 = Level(blocks[0], 2 * channels, heads[0]);
        _refinement = Level(blocks[3], 2 * channels, heads[0]);
        _compress = layers.Count;
        layers.Add(RealViformerBlocks.Conv<T>(channels, 1, true));
        // LeakyReLU is elementwise and pixel shuffle a permutation, so the activation sits on the convolution.
        _up1 = layers.Count;
        layers.Add(BasicVsrBlocks.Conv<T>(channels * 4, 3, new LeakyReLUActivation<T>(0.1)));
        _up2 = layers.Count;
        layers.Add(BasicVsrBlocks.Conv<T>(64 * 4, 3, new LeakyReLUActivation<T>(0.1)));
        _hr = layers.Count;
        layers.Add(BasicVsrBlocks.Conv<T>(64, 3, new LeakyReLUActivation<T>(0.1)));
        _last = layers.Count;
        layers.Add(BasicVsrBlocks.Conv<T>(3, 3, null));
    }

    internal SpyNetFlowBlock<T> SpyNet { get; }

    /// <summary>The SPyNet parameters.</summary>
    internal IReadOnlyList<Tensor<T>> FlowParameters(IReadOnlyList<ILayer<T>> layers)
        => AiDotNet.Training.TapeTrainingStep<T>.CollectParameters(layers.Skip(SpyNet.First).Take(SpyNet.Count));

    private static Tensor<T> Run(IEngine engine, IReadOnlyList<ILayer<T>> layers, RealViformerTransformerBlock<T>[] blocks, Tensor<T> x)
    {
        foreach (var block in blocks) x = block.Forward(engine, layers, x);
        return x;
    }

    /// <summary>RealViformer.forward: [N, T, 3, H, W] in [0, 1], H and W multiples of 4, to [N, T, 3, 4H, 4W].</summary>
    internal Tensor<T> Run(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> lqs)
    {
        int n = lqs.Shape[0], t = lqs.Shape[1], h = lqs.Shape[3], w = lqs.Shape[4];
        if (h % 4 != 0 || w % 4 != 0)
            throw new ArgumentException($"RealViformer's encoder halves the frame twice; {h}x{w} is not a multiple of 4.", nameof(lqs));

        // get_flow: flows_forward[i] runs from frame i + 1 back to frame i.
        Tensor<T>? flows = null;
        if (t > 1)
        {
            var later = engine.Reshape(engine.TensorNarrow(lqs, 1, 1, t - 1), new[] { n * (t - 1), 3, h, w });
            var earlier = engine.Reshape(engine.TensorNarrow(lqs, 1, 0, t - 1), new[] { n * (t - 1), 3, h, w });
            flows = engine.Reshape(SpyNet.Flow(engine, layers, later, earlier), new[] { n, t - 1, 2, h, w });
        }

        var hidden = new Tensor<T>(new[] { n, _channels, h, w });
        var outputs = new Tensor<T>[t];
        for (int i = 0; i < t; i++)
        {
            var frame = engine.Reshape(engine.TensorNarrow(lqs, 1, i, 1), new[] { n, 3, h, w });
            var shallow = _shallowBlock.Forward(engine, layers, layers[_shallowConv].Forward(frame));
            if (i > 0 && flows is not null)
                hidden = BasicVsrBlocks.FlowWarp(engine, hidden, engine.Reshape(engine.TensorNarrow(flows, 1, i - 1, 1), new[] { n, 2, h, w }),
                    GridSamplePadding.Zeros);
            hidden = _merge.Forward(engine, layers, shallow, hidden);

            var enc1 = Run(engine, layers, _encoder1, hidden);
            var enc2 = Run(engine, layers, _encoder2, engine.TensorPixelUnshuffle(layers[_down12].Forward(enc1), 2));
            var latent = Run(engine, layers, _latent, engine.TensorPixelUnshuffle(layers[_down23].Forward(enc2), 2));
            var dec2 = engine.TensorConcatenate(new[] { engine.PixelShuffle(layers[_up32].Forward(latent), 2), enc2 }, 1);
            dec2 = Run(engine, layers, _decoder2, layers[_reduce2].Forward(dec2));
            var dec1 = engine.TensorConcatenate(new[] { engine.PixelShuffle(layers[_up21].Forward(dec2), 2), enc1 }, 1);
            var refined = Run(engine, layers, _refinement, Run(engine, layers, _decoder1, dec1));
            hidden = layers[_compress].Forward(refined);

            var hr = engine.PixelShuffle(layers[_up1].Forward(refined), 2);
            hr = engine.PixelShuffle(layers[_up2].Forward(hr), 2);
            hr = layers[_last].Forward(layers[_hr].Forward(hr));
            var baseFrame = engine.Interpolate(frame, new[] { 4 * h, 4 * w }, InterpolateMode.Bilinear, false);
            outputs[i] = engine.Reshape(engine.TensorAdd(hr, baseFrame), new[] { n, 1, 3, 4 * h, 4 * w });
        }
        return engine.TensorConcatenate(outputs, 1);
    }
}

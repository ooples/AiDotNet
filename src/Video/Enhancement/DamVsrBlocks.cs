using AiDotNet.ActivationFunctions;
using AiDotNet.Initialization;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// DAM-VSR's video ControlNet (Kong et al. 2025, Sec. 3.2): a copy of the SVD U-Net's conv_in, down stages and middle
/// block, temporal attention and 3D convolutions included, fed the noisy latent concatenated with the reference latent
/// plus the low-resolution video through ControlNet's conditioning embedding (3x3 convolutions with SiLU, three stride-2,
/// 16 to 256 channels, then a zero 3x3 convolution). Its skips and middle output reach the U-Net through zero-initialized
/// 1x1 convolutions, as in ControlNet (Zhang et al. 2023), so training starts from the unmodified SVD.
/// </summary>
internal sealed class DamVsrControlNet<T>
{
    private readonly SvdUNet<T> _encoder;
    private readonly int[] _embedding;
    private readonly int _embeddingOut;
    private readonly int[] _zeroSkips;
    private readonly int _zeroMiddle;

    /// <param name="downsamples">The conditioning embedding's stride-2 convolutions, log2 of the VAE's factor (3).</param>
    internal DamVsrControlNet(List<ILayer<T>> layers, int[] channels, int[] heads, int layersPerBlock, int transformerDepth, int addTimeDim,
        int downsamples)
    {
        _encoder = new SvdUNet<T>(layers, 0, channels, heads, layersPerBlock, transformerDepth, addTimeDim, encoderOnly: true);
        var widths = new[] { 16, 32, 96, 256 }.Take(downsamples + 1).ToArray();
        var embedding = new List<int>();
        embedding.Add(layers.Count);
        layers.Add(BasicVsrBlocks.Conv<T>(widths[0], 3, new SiLUActivation<T>()));
        for (int i = 0; i < widths.Length - 1; i++)
        {
            embedding.Add(layers.Count);
            layers.Add(BasicVsrBlocks.Conv<T>(widths[i], 3, new SiLUActivation<T>()));
            embedding.Add(layers.Count);
            layers.Add(new ConvolutionalLayer<T>(widths[i + 1], 3, 2, 1, new SiLUActivation<T>(), new TorchDefaultInitializationStrategy<T>()));
        }
        _embedding = embedding.ToArray();
        _embeddingOut = layers.Count;
        layers.Add(BasicVsrBlocks.Conv<T>(channels[0], 3, null, new ConstantInitializationStrategy<T>(0.0)));

        var skipChannels = new List<int> { channels[0] };
        for (int s = 0; s < channels.Length; s++)
        {
            for (int i = 0; i < layersPerBlock; i++) skipChannels.Add(channels[s]);
            if (s < channels.Length - 1) skipChannels.Add(channels[s]);
        }
        _zeroSkips = new int[skipChannels.Count];
        for (int i = 0; i < skipChannels.Count; i++)
        {
            _zeroSkips[i] = layers.Count;
            layers.Add(BasicVsrBlocks.Conv<T>(skipChannels[i], 1, null, new ConstantInitializationStrategy<T>(0.0)));
        }
        _zeroMiddle = layers.Count;
        layers.Add(BasicVsrBlocks.Conv<T>(channels[channels.Length - 1], 1, null, new ConstantInitializationStrategy<T>(0.0)));
    }

    /// <summary>The ControlNet's own embedding of the timestep and added time ids.</summary>
    internal Tensor<T> Embedding(IEngine engine, IReadOnlyList<ILayer<T>> layers, IReadOnlyList<double> timesteps,
        IReadOnlyList<double[]> addedTimeIds, int frames) => _encoder.Embedding(engine, layers, timesteps, addedTimeIds, frames);

    /// <param name="sample">[B·F, 8, h, w], the scaled noisy latent and the reference latent.</param>
    /// <param name="lowRes">[B·F, 3, 8h, 8w], the low-resolution frames brought to the output size, in [-1, 1].</param>
    /// <returns>The residuals for the U-Net's skips and middle output.</returns>
    internal (IReadOnlyList<Tensor<T>> Skips, Tensor<T> Middle) Forward(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> sample,
        Tensor<T> lowRes, Tensor<T> emb, Tensor<T> context, int frames)
    {
        var cond = lowRes;
        foreach (int index in _embedding) cond = layers[index].Forward(cond);
        cond = layers[_embeddingOut].Forward(cond);
        var (skips, bottom) = _encoder.Encode(engine, layers, sample, emb, context, frames, x => engine.TensorAdd(x, cond));
        var middle = _encoder.Middle(engine, layers, bottom, emb, context, frames);
        var residuals = new Tensor<T>[skips.Count];
        for (int i = 0; i < skips.Count; i++) residuals[i] = layers[_zeroSkips[i]].Forward(skips[i]);
        return (residuals, layers[_zeroMiddle].Forward(middle));
    }
}

/// <summary>
/// Stable Video Diffusion's image encoder, transformers' <c>CLIPVisionModelWithProjection</c> with OpenCLIP ViT-H/14's
/// configuration by default: 14x14 patches, a class token and learned positions, a pre-LayerNorm, pre-norm transformer
/// layers (biased attention projections, GELU MLP), the class token's post-LayerNorm and a bias-free projection.
/// </summary>
internal sealed class SvdImageEncoder<T>
{
    private readonly int _patch, _classToken, _positions, _preNorm, _postNorm, _projection, _width, _patchSize, _imageSize;
    private readonly (int Norm1, SvdAttention<T> Attention, int Norm2, int Fc1, int Fc2)[] _blocks;

    internal SvdImageEncoder(List<ILayer<T>> layers, int imageSize, int patchSize, int width, int depth, int heads, int mlpDim, int projection)
    {
        _imageSize = imageSize;
        _patchSize = patchSize;
        _width = width;
        int tokens = (imageSize / patchSize) * (imageSize / patchSize) + 1;
        _patch = layers.Count;
        layers.Add(new ConvolutionalLayer<T>(width, patchSize, patchSize, 0, null, new TorchDefaultInitializationStrategy<T>(), null, 1,
            AiDotNet.Enums.BiasMode.Never));
        _classToken = layers.Count;
        layers.Add(RealViformerBlocks.Vector<T>(width, 0.0));
        _positions = layers.Count;
        layers.Add(RealViformerBlocks.Vector<T>(tokens * width, 0.0));
        _preNorm = layers.Count;
        layers.Add(RvrtBlocks.Norm<T>(width));
        _blocks = new (int, SvdAttention<T>, int, int, int)[depth];
        for (int i = 0; i < depth; i++)
        {
            int norm1 = layers.Count;
            layers.Add(RvrtBlocks.Norm<T>(width));
            var attention = new SvdAttention<T>(layers, width, heads, width / heads, biasedProjections: true);
            int norm2 = layers.Count;
            layers.Add(RvrtBlocks.Norm<T>(width));
            int fc1 = layers.Count;
            layers.Add(RvrtBlocks.Linear<T>(mlpDim, new GELUActivation<T>()));
            int fc2 = layers.Count;
            layers.Add(RvrtBlocks.Linear<T>(width));
            _blocks[i] = (norm1, attention, norm2, fc1, fc2);
        }
        _postNorm = layers.Count;
        layers.Add(RvrtBlocks.Norm<T>(width));
        _projection = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(projection, bias: false));
    }

    /// <summary>The embedding [B, projection] of images [B, 3, H, W] in [-1, 1], resized and normalized as SVD's pipeline does.</summary>
    internal Tensor<T> Forward(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> images)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int b = images.Shape[0];
        // _encode_image: resize to 224 (bicubic with antialiasing in the release; bilinear here), map [-1, 1] to [0, 1],
        // then CLIP's per-channel mean and standard deviation.
        var x = engine.Interpolate(images, new[] { _imageSize, _imageSize }, InterpolateMode.Bilinear, false);
        x = engine.TensorMultiplyScalar(engine.TensorAddScalar(x, ops.One), ops.FromDouble(0.5));
        var mean = new[] { 0.48145466, 0.4578275, 0.40821073 };
        var std = new[] { 0.26862954, 0.26130258, 0.27577711 };
        var scale = new Tensor<T>(new[] { 1, 3, 1, 1 });
        var shift = new Tensor<T>(new[] { 1, 3, 1, 1 });
        for (int c = 0; c < 3; c++) { scale[c] = ops.FromDouble(1.0 / std[c]); shift[c] = ops.FromDouble(-mean[c] / std[c]); }
        var shape = x.Shape.ToArray();
        x = engine.TensorAdd(engine.TensorMultiply(x, engine.TensorBroadcastTo(scale, shape)), engine.TensorBroadcastTo(shift, shape));

        var patches = layers[_patch].Forward(x);
        int grid = patches.Shape[2] * patches.Shape[3];
        var tokens = engine.TensorPermute(engine.Reshape(patches, new[] { b, _width, grid }), new[] { 0, 2, 1 });
        var cls = engine.TensorBroadcastTo(engine.Reshape(RealViformerBlocks.Read(layers, _classToken), new[] { 1, 1, _width }), new[] { b, 1, _width });
        tokens = engine.TensorConcatenate(new[] { cls, tokens }, 1);
        var positions = engine.Reshape(RealViformerBlocks.Read(layers, _positions), new[] { 1, grid + 1, _width });
        tokens = layers[_preNorm].Forward(engine.TensorAdd(tokens, engine.TensorBroadcastTo(positions, tokens.Shape.ToArray())));
        foreach (var (norm1, attention, norm2, fc1, fc2) in _blocks)
        {
            tokens = engine.TensorAdd(tokens, attention.Forward(engine, layers, layers[norm1].Forward(tokens)));
            tokens = engine.TensorAdd(tokens, layers[fc2].Forward(layers[fc1].Forward(layers[norm2].Forward(tokens))));
        }
        var pooled = layers[_postNorm].Forward(engine.Reshape(engine.TensorNarrow(tokens, 1, 0, 1), new[] { b, _width }));
        return layers[_projection].Forward(pooled);
    }
}

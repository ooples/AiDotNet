using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// diffusers <c>TransformerSpatioTemporalModel</c>: GroupNorm and a linear projection in, then per layer a spatial
/// transformer block over each frame's positions and a temporal block over each position's frames (with a learned
/// embedding of the frame index added), blended by a learned α, a linear projection out and the input added back.
/// </summary>
internal sealed class SvdSpatioTemporalTransformer<T>
{
    private readonly int _norm, _projIn, _projOut, _channels;
    private readonly SvdBasicTransformerBlock<T>[] _spatial;
    private readonly SvdTemporalTransformerBlock<T>[] _temporal;
    private readonly SvdTimestepEmbedding<T> _timePosEmbed;
    private readonly SvdAlphaBlender<T> _mixer;

    internal SvdSpatioTemporalTransformer(List<ILayer<T>> layers, int channels, int heads, int depth)
    {
        _channels = channels;
        int headDim = channels / heads;
        _norm = layers.Count;
        layers.Add(SvdBlocks.GroupNorm<T>(channels, 1e-6));
        _projIn = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(channels));
        _spatial = new SvdBasicTransformerBlock<T>[depth];
        _temporal = new SvdTemporalTransformerBlock<T>[depth];
        for (int i = 0; i < depth; i++)
        {
            _spatial[i] = new SvdBasicTransformerBlock<T>(layers, channels, heads, headDim);
            _temporal[i] = new SvdTemporalTransformerBlock<T>(layers, channels, heads, headDim);
        }
        _timePosEmbed = new SvdTimestepEmbedding<T>(layers, 4 * channels, channels);
        _mixer = new SvdAlphaBlender<T>(layers, 0.5, false);
        _projOut = layers.Count;
        layers.Add(RvrtBlocks.Linear<T>(channels));
    }

    /// <summary>The transformer's width.</summary>
    internal int Channels => _channels;

    /// <summary>The temporal blocks, whose self-attention DAM-VSR's backward pass re-weights.</summary>
    internal IReadOnlyList<SvdTemporalTransformerBlock<T>> TemporalBlocks => _temporal;

    /// <param name="x">[B·F, C, H, W].</param>
    /// <param name="context">[B·F, M, Dc], the image embedding repeated per frame.</param>
    /// <param name="temporalAttention">Replacement temporal self-attentions, one per layer, or null.</param>
    internal Tensor<T> Forward(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> x, Tensor<T> context, int frames,
        IReadOnlyList<Func<Tensor<T>, Tensor<T>>>? temporalAttention = null)
    {
        int bf = x.Shape[0], c = x.Shape[1], h = x.Shape[2], w = x.Shape[3], b = bf / frames, l = h * w;
        // The first frame's context, one row per spatial position of the clip.
        var first = engine.Reshape(engine.TensorNarrow(engine.Reshape(context, new[] { b, frames, context.Shape[1], context.Shape[2] }), 1, 0, 1),
            new[] { b, context.Shape[1], context.Shape[2] });
        var timeContext = SvdBlocks.RepeatInterleave(engine, first, l);

        var tokens = engine.Reshape(engine.TensorPermute(layers[_norm].Forward(x), new[] { 0, 2, 3, 1 }), new[] { bf, l, c });
        tokens = layers[_projIn].Forward(tokens);
        var frameIndex = new double[bf];
        for (int i = 0; i < bf; i++) frameIndex[i] = i % frames;
        var emb = engine.Reshape(_timePosEmbed.Forward(layers, SvdBlocks.Sinusoid<T>(frameIndex, _channels)), new[] { bf, 1, _channels });
        var embAll = engine.TensorBroadcastTo(emb, new[] { bf, l, _channels });
        for (int i = 0; i < _spatial.Length; i++)
        {
            tokens = _spatial[i].Forward(engine, layers, tokens, context);
            var mixed = _temporal[i].Forward(engine, layers, engine.TensorAdd(tokens, embAll), frames, timeContext, temporalAttention?[i]);
            tokens = _mixer.Forward(engine, layers, tokens, mixed);
        }
        var output = engine.TensorPermute(engine.Reshape(layers[_projOut].Forward(tokens), new[] { bf, h, w, c }), new[] { 0, 3, 1, 2 });
        return engine.TensorAdd(output, x);
    }
}

/// <summary>One stage of the SVD U-Net: spatio-temporal ResNets, each optionally followed by a transformer.</summary>
internal sealed class SvdStage<T>
{
    internal SvdSpatioTemporalResBlock<T>[] Resnets { get; }
    internal SvdSpatioTemporalTransformer<T>?[] Attentions { get; }
    internal int Resample { get; }

    internal SvdStage(SvdSpatioTemporalResBlock<T>[] resnets, SvdSpatioTemporalTransformer<T>?[] attentions, int resample)
    {
        Resnets = resnets;
        Attentions = attentions;
        Resample = resample;
    }
}

/// <summary>
/// Stable Video Diffusion's denoiser, diffusers <c>UNetSpatioTemporalConditionModel</c>: conv_in, the time embedding plus
/// the embedded added time ids (fps, motion bucket, noise augmentation), three cross-attention down stages and one
/// plain, the middle block, one plain and three cross-attention up stages, GroupNorm, SiLU and conv_out. Optional
/// ControlNet residuals are added to the twelve skips and the middle output, as diffusers' ControlNet pipelines do.
/// </summary>
internal sealed class SvdUNet<T>
{
    private readonly int _convIn, _normOut = -1, _convOut = -1, _addTimeDim;
    private readonly int[] _channels;
    private readonly SvdTimestepEmbedding<T> _timeEmbedding, _addEmbedding;
    private readonly SvdStage<T>[] _down, _up = Array.Empty<SvdStage<T>>();
    private readonly SvdSpatioTemporalResBlock<T> _midResnet1, _midResnet2;
    private readonly SvdSpatioTemporalTransformer<T> _midAttention;

    /// <param name="layers">The layer list the U-Net appends to.</param>
    /// <param name="outChannels">conv_out's channels (4 latent channels).</param>
    /// <param name="channels">The stage widths (320, 640, 1280, 1280).</param>
    /// <param name="heads">The stage head counts (5, 10, 20, 20).</param>
    /// <param name="layersPerBlock">ResNets per down stage (2; up stages have one more).</param>
    /// <param name="transformerDepth">Transformer layers per attention (1).</param>
    /// <param name="addTimeDim">The width each added time id is embedded at (256).</param>
    /// <param name="encoderOnly">Builds only conv_in, the embeddings, the down stages and the middle block (a ControlNet's
    /// copy of the encoder).</param>
    internal SvdUNet(List<ILayer<T>> layers, int outChannels, int[] channels, int[] heads, int layersPerBlock, int transformerDepth,
        int addTimeDim, bool encoderOnly = false)
    {
        _channels = channels;
        _addTimeDim = addTimeDim;
        int timeDim = 4 * channels[0];
        int stages = channels.Length;
        _convIn = layers.Count;
        layers.Add(BasicVsrBlocks.Conv<T>(channels[0], 3, null));
        _timeEmbedding = new SvdTimestepEmbedding<T>(layers, timeDim, timeDim);
        _addEmbedding = new SvdTimestepEmbedding<T>(layers, timeDim, timeDim);

        _down = new SvdStage<T>[stages];
        int input = channels[0];
        for (int s = 0; s < stages; s++)
        {
            bool attention = s < stages - 1;
            var resnets = new SvdSpatioTemporalResBlock<T>[layersPerBlock];
            var attentions = new SvdSpatioTemporalTransformer<T>?[layersPerBlock];
            for (int i = 0; i < layersPerBlock; i++)
            {
                resnets[i] = new SvdSpatioTemporalResBlock<T>(layers, i == 0 ? input : channels[s], channels[s], 1e-5);
                if (attention) attentions[i] = new SvdSpatioTemporalTransformer<T>(layers, channels[s], heads[s], transformerDepth);
            }
            int downsample = -1;
            if (s < stages - 1)
            {
                downsample = layers.Count;
                layers.Add(new AiDotNet.NeuralNetworks.Layers.ConvolutionalLayer<T>(channels[s], 3, 2, 1, (IActivationFunction<T>?)null,
                    new AiDotNet.Initialization.TorchDefaultInitializationStrategy<T>()));
            }
            _down[s] = new SvdStage<T>(resnets, attentions, downsample);
            input = channels[s];
        }

        int last = channels[stages - 1];
        _midResnet1 = new SvdSpatioTemporalResBlock<T>(layers, last, last, 1e-5);
        _midAttention = new SvdSpatioTemporalTransformer<T>(layers, last, heads[stages - 1], transformerDepth);
        _midResnet2 = new SvdSpatioTemporalResBlock<T>(layers, last, last, 1e-5);
        if (encoderOnly) return;

        _up = new SvdStage<T>[stages];
        int previous = last;
        for (int u = 0; u < stages; u++)
        {
            int s = stages - 1 - u;
            int output = channels[s];
            int skipIn = channels[Math.Max(s - 1, 0)];
            bool attention = u > 0;
            var resnets = new SvdSpatioTemporalResBlock<T>[layersPerBlock + 1];
            var attentions = new SvdSpatioTemporalTransformer<T>?[layersPerBlock + 1];
            for (int i = 0; i <= layersPerBlock; i++)
            {
                int skip = i == layersPerBlock ? skipIn : output;
                int resIn = i == 0 ? previous : output;
                resnets[i] = new SvdSpatioTemporalResBlock<T>(layers, resIn + skip, output, 1e-5);
                if (attention) attentions[i] = new SvdSpatioTemporalTransformer<T>(layers, output, heads[s], transformerDepth);
            }
            int upsample = -1;
            if (u < stages - 1)
            {
                upsample = layers.Count;
                layers.Add(BasicVsrBlocks.Conv<T>(output, 3, null));
            }
            _up[u] = new SvdStage<T>(resnets, attentions, upsample);
            previous = output;
        }

        _normOut = layers.Count;
        layers.Add(SvdBlocks.GroupNorm<T>(channels[0], 1e-5));
        _convOut = layers.Count;
        layers.Add(BasicVsrBlocks.Conv<T>(outChannels, 3, null));
    }

    /// <summary>Every transformer in the order the forward pass runs them (down, middle, up).</summary>
    internal IEnumerable<SvdSpatioTemporalTransformer<T>> Transformers()
    {
        foreach (var stage in _down) foreach (var a in stage.Attentions) if (a is not null) yield return a;
        yield return _midAttention;
        foreach (var stage in _up) foreach (var a in stage.Attentions) if (a is not null) yield return a;
    }

    /// <summary>The time embedding plus the embedded added time ids, one row per frame: [B·F, 4·C0].</summary>
    internal Tensor<T> Embedding(IEngine engine, IReadOnlyList<ILayer<T>> layers, IReadOnlyList<double> timesteps,
        IReadOnlyList<double[]> addedTimeIds, int frames)
    {
        int b = timesteps.Count;
        var emb = _timeEmbedding.Forward(layers, SvdBlocks.Sinusoid<T>(timesteps, _channels[0]));
        var ids = addedTimeIds.SelectMany(row => row).ToArray();
        var addProj = engine.Reshape(SvdBlocks.Sinusoid<T>(ids, _addTimeDim), new[] { b, ids.Length / b * _addTimeDim });
        emb = engine.TensorAdd(emb, _addEmbedding.Forward(layers, addProj));
        return SvdBlocks.RepeatInterleave(engine, emb, frames);
    }

    /// <summary>conv_in and the down stages: the twelve skips (conv_in output first) and the bottom features.</summary>
    internal (List<Tensor<T>> Skips, Tensor<T> Bottom) Encode(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> sample,
        Tensor<T> emb, Tensor<T> context, int frames, Func<Tensor<T>, Tensor<T>>? afterConvIn = null)
    {
        var x = layers[_convIn].Forward(sample);
        if (afterConvIn is not null) x = afterConvIn(x);
        var skips = new List<Tensor<T>> { x };
        foreach (var stage in _down)
        {
            for (int i = 0; i < stage.Resnets.Length; i++)
            {
                x = stage.Resnets[i].Forward(engine, layers, x, emb, frames);
                if (stage.Attentions[i] is { } attention) x = attention.Forward(engine, layers, x, context, frames);
                skips.Add(x);
            }
            if (stage.Resample >= 0)
            {
                x = layers[stage.Resample].Forward(x);
                skips.Add(x);
            }
        }
        return (skips, x);
    }

    /// <summary>The middle block.</summary>
    internal Tensor<T> Middle(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> x, Tensor<T> emb, Tensor<T> context, int frames,
        Func<SvdSpatioTemporalTransformer<T>, IReadOnlyList<Func<Tensor<T>, Tensor<T>>>?>? temporalAttention = null)
    {
        x = _midResnet1.Forward(engine, layers, x, emb, frames);
        x = _midAttention.Forward(engine, layers, x, context, frames, temporalAttention?.Invoke(_midAttention));
        return _midResnet2.Forward(engine, layers, x, emb, frames);
    }

    /// <summary>
    /// The denoiser on [B·F, Cin, H, W] latents (H and W multiples of 8): the predicted [B·F, Cout, H, W] output.
    /// </summary>
    /// <param name="controlSkips">ControlNet residuals for the twelve skips, or null.</param>
    /// <param name="controlMiddle">The ControlNet residual for the middle output, or null.</param>
    /// <param name="temporalAttention">Replacement temporal self-attentions per transformer, or null.</param>
    internal Tensor<T> Forward(IEngine engine, IReadOnlyList<ILayer<T>> layers, Tensor<T> sample, Tensor<T> emb, Tensor<T> context,
        int frames, IReadOnlyList<Tensor<T>>? controlSkips = null, Tensor<T>? controlMiddle = null,
        Func<SvdSpatioTemporalTransformer<T>, IReadOnlyList<Func<Tensor<T>, Tensor<T>>>?>? temporalAttention = null)
    {
        var x = layers[_convIn].Forward(sample);
        var skips = new List<Tensor<T>> { x };
        foreach (var stage in _down)
        {
            for (int i = 0; i < stage.Resnets.Length; i++)
            {
                x = stage.Resnets[i].Forward(engine, layers, x, emb, frames);
                if (stage.Attentions[i] is { } attention)
                    x = attention.Forward(engine, layers, x, context, frames, temporalAttention?.Invoke(attention));
                skips.Add(x);
            }
            if (stage.Resample >= 0)
            {
                x = layers[stage.Resample].Forward(x);
                skips.Add(x);
            }
        }
        if (controlSkips is not null)
        {
            if (controlSkips.Count != skips.Count)
                throw new ArgumentException($"The U-Net has {skips.Count} skips; {controlSkips.Count} residuals were given.", nameof(controlSkips));
            for (int i = 0; i < skips.Count; i++) skips[i] = engine.TensorAdd(skips[i], controlSkips[i]);
        }

        x = Middle(engine, layers, x, emb, context, frames, temporalAttention);
        if (controlMiddle is not null) x = engine.TensorAdd(x, controlMiddle);

        int next = skips.Count - 1;
        foreach (var stage in _up)
        {
            for (int i = 0; i < stage.Resnets.Length; i++)
            {
                x = engine.TensorConcatenate(new[] { x, skips[next--] }, 1);
                x = stage.Resnets[i].Forward(engine, layers, x, emb, frames);
                if (stage.Attentions[i] is { } attention)
                    x = attention.Forward(engine, layers, x, context, frames, temporalAttention?.Invoke(attention));
            }
            if (stage.Resample >= 0)
            {
                // Upsample2D: nearest x2, then a 3x3 convolution.
                x = layers[stage.Resample].Forward(engine.Interpolate(x, new[] { 2 * x.Shape[2], 2 * x.Shape[3] }, InterpolateMode.Nearest, false));
            }
        }
        return layers[_convOut].Forward(engine.Swish(layers[_normOut].Forward(x)));
    }
}

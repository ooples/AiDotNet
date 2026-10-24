using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.ComputerVision.Segmentation.Semantic;

/// <summary>
/// SegNeXt (Guo et al. 2022) as mmsegmentation implements it: the MSCAN encoder and the LightHamHead decoder.
/// </summary>
/// <remarks>
/// <para>
/// <b>MSCAN.</b> A stem of two stride-2 3×3 convolutions (BatchNorm, GELU, BatchNorm), then stride-2 3×3 patch
/// embeddings with BatchNorm. Each block is <c>x + s₁·Attn(BN(x))</c> then <c>x + s₂·MLP(BN(x))</c> with layer scales
/// s (0.01 at init) and stochastic depth. Attn is proj₁ → GELU → MSCA → proj₂ plus a shortcut, where MSCA sums a 5×5
/// depth-wise convolution with three strip branches (1×7 then 7×1, 1×11 then 11×1, 1×21 then 21×1), mixes channels
/// with a 1×1 convolution and multiplies its input by the result. The MLP is fc₁ → 3×3 depth-wise → GELU → fc₂. Each
/// stage ends in a LayerNorm over channels.
/// </para>
/// <para>
/// <b>LightHamHead.</b> Stages 2–4 are resized (bilinear, align_corners false) to stage 2's 1/8 scale and concatenated.
/// A squeeze 1×1 convolution with GroupNorm and ReLU, the Hamburger (1×1 → ReLU → NMF → 1×1 with GroupNorm, then
/// ReLU of the residual sum), an align 1×1 convolution with GroupNorm and ReLU, Dropout2d and the 1×1 classifier give
/// the logits at 1/8. The NMF factors the non-negative map X ≈ B Cᵀ from bases drawn uniformly and L2-normalized on
/// every forward: C = softmax(XᵀB), then multiplicative updates of C and B (6 steps in training, 7 in evaluation) and
/// a final update of C. The gradient flows through every step, as in both official codes.
/// </para>
/// </remarks>
internal sealed class SegNeXtNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly List<LayerBase<T>> _owned = new();

    internal sealed record Block(
        BatchNormalizationLayer<T> Norm1, ConvolutionalLayer<T> Projection1,
        StripDepthwiseConvolution<T> Conv0, StripDepthwiseConvolution<T>[] Strips, ConvolutionalLayer<T> Conv3,
        ConvolutionalLayer<T> Projection2, ChannelScale<T> Scale1,
        BatchNormalizationLayer<T> Norm2, ConvolutionalLayer<T> Fc1, StripDepthwiseConvolution<T> DepthwiseConv,
        ConvolutionalLayer<T> Fc2, ChannelScale<T> Scale2, double DropPath);

    internal sealed record ConvNorm(ConvolutionalLayer<T> Conv, GroupNormalizationLayer<T> Norm);

    public SegNeXtNetwork(IEngine engine, List<LayerBase<T>> layers, SegNeXtConfiguration configuration)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        if (c.EmbedDims.Length != 4 || c.Depths.Length != 4 || c.MlpRatios.Length != 4)
            throw new ArgumentException("MSCAN has four stages.", nameof(configuration));
        if (c.HamChannels % c.NormGroups != 0 || c.HeadChannels % c.NormGroups != 0)
            throw new ArgumentException($"The head's channels must divide into {c.NormGroups} GroupNorm groups.", nameof(configuration));

        LayerBase<T> Own(LayerBase<T> layer)
        {
            _owned.Add(layer);
            return layer;
        }
        ConvolutionalLayer<T> Conv(int inputs, int outputs, int kernel, int stride, int padding, bool bias)
            => (ConvolutionalLayer<T>)Own(ConvolutionalLayer<T>.WithInputDepth(inputs, outputs, kernel, stride, padding,
                initializationStrategy: new Initialization.ZeroInitializationStrategy<T>(),
                biasMode: bias ? Enums.BiasMode.Always : Enums.BiasMode.Never));
        BatchNormalizationLayer<T> Bn(int channels) => (BatchNormalizationLayer<T>)Own(new BatchNormalizationLayer<T>(channels, 1e-5, 0.9));
        StripDepthwiseConvolution<T> Dw(int channels, int kh, int kw, int ph, int pw)
            => (StripDepthwiseConvolution<T>)Own(new StripDepthwiseConvolution<T>(channels, kh, kw, ph, pw));

        int d0 = c.EmbedDims[0];
        StemConv1 = Conv(c.InputChannels, d0 / 2, 3, 2, 1, bias: true);
        StemNorm1 = Bn(d0 / 2);
        StemConv2 = Conv(d0 / 2, d0, 3, 2, 1, bias: true);
        StemNorm2 = Bn(d0);
        int total = c.Depths.Sum(), index = 0;
        for (int stage = 0; stage < 4; stage++)
        {
            int d = c.EmbedDims[stage];
            if (stage > 0)
            {
                PatchConvs.Add(Conv(c.EmbedDims[stage - 1], d, 3, 2, 1, bias: true));
                PatchNorms.Add(Bn(d));
            }
            var blocks = new List<Block>();
            for (int j = 0; j < c.Depths[stage]; j++, index++)
            {
                int hidden = d * c.MlpRatios[stage];
                // torch.linspace(0, drop_path_rate, sum(depths))
                double dropPath = total == 1 ? 0.0 : c.DropPathRate * index / (total - 1);
                blocks.Add(new Block(
                    Bn(d), Conv(d, d, 1, 1, 0, true),
                    Dw(d, 5, 5, 2, 2),
                    new[] { Dw(d, 1, 7, 0, 3), Dw(d, 7, 1, 3, 0), Dw(d, 1, 11, 0, 5), Dw(d, 11, 1, 5, 0), Dw(d, 1, 21, 0, 10), Dw(d, 21, 1, 10, 0) },
                    Conv(d, d, 1, 1, 0, true), Conv(d, d, 1, 1, 0, true), (ChannelScale<T>)Own(new ChannelScale<T>(d)),
                    Bn(d), Conv(d, hidden, 1, 1, 0, true), Dw(hidden, 3, 3, 1, 1), Conv(hidden, d, 1, 1, 0, true),
                    (ChannelScale<T>)Own(new ChannelScale<T>(d)), dropPath));
            }
            Blocks.Add(blocks);
            StageNorms.Add((LayerNormalizationLayer<T>)Own(new LayerNormalizationLayer<T>(d, 1e-5)));
        }

        ConvNorm Head(int inputs, int outputs)
            => new(Conv(inputs, outputs, 1, 1, 0, bias: false),
                (GroupNormalizationLayer<T>)Own(new GroupNormalizationLayer<T>(c.NormGroups, outputs, 1e-5)));
        Squeeze = Head(c.EmbedDims[1] + c.EmbedDims[2] + c.EmbedDims[3], c.HamChannels);
        HamIn = Conv(c.HamChannels, c.HamChannels, 1, 1, 0, bias: true);
        HamOut = Head(c.HamChannels, c.HamChannels);
        Align = Head(c.HamChannels, c.HeadChannels);
        Classifier = new SegmentationClassifier<T>(engine, Own, c.HeadChannels, c.Classes, c.HeadDropout);
        layers.AddRange(_owned);
    }

    public SegNeXtConfiguration Configuration { get; }
    public ConvolutionalLayer<T> StemConv1 { get; }
    public BatchNormalizationLayer<T> StemNorm1 { get; }
    public ConvolutionalLayer<T> StemConv2 { get; }
    public BatchNormalizationLayer<T> StemNorm2 { get; }
    public List<ConvolutionalLayer<T>> PatchConvs { get; } = new();
    public List<BatchNormalizationLayer<T>> PatchNorms { get; } = new();
    public List<List<Block>> Blocks { get; } = new();
    public List<LayerNormalizationLayer<T>> StageNorms { get; } = new();
    public ConvNorm Squeeze { get; }
    public ConvolutionalLayer<T> HamIn { get; }
    public ConvNorm HamOut { get; }
    public ConvNorm Align { get; }
    public SegmentationClassifier<T> Classifier { get; }

    /// <summary>The layers this network added to its owner's list, in order.</summary>
    public IReadOnlyList<LayerBase<T>> OwnedLayers => _owned;

    /// <summary>The decode head's layers, which train at ten times the encoder's rate in the official recipe.</summary>
    public IEnumerable<LayerBase<T>> HeadLayers()
    {
        yield return Squeeze.Conv;
        yield return Squeeze.Norm;
        yield return HamIn;
        yield return HamOut.Conv;
        yield return HamOut.Norm;
        yield return Align.Conv;
        yield return Align.Norm;
        yield return Classifier.Conv;
    }

    /// <summary>The encoder layers whose official parameter names contain "norm" (no weight decay in the recipe):
    /// the blocks' norm1/norm2, the patch embeddings' norm and the stage norms. The stem's BatchNorms are
    /// <c>proj.1</c> and <c>proj.4</c>, so they are decayed.</summary>
    public IEnumerable<LayerBase<T>> NormNamedLayers()
    {
        foreach (var norm in PatchNorms) yield return norm;
        foreach (var norm in StageNorms) yield return norm;
        foreach (var block in Blocks.SelectMany(b => b))
        {
            yield return block.Norm1;
            yield return block.Norm2;
        }
    }

    private Tensor<T> Gelu(Tensor<T> x) => ActivationHelper.ExactGelu(x, _engine);

    private Tensor<T> DropPath(Tensor<T> x, double rate, bool training, Random random)
    {
        if (!training || rate <= 0) return x;
        var mask = new Tensor<T>(new[] { x.Shape[0], 1, 1, 1 });
        for (int b = 0; b < mask.Length; b++)
            mask[b] = random.NextDouble() < rate ? NumOps.Zero : NumOps.FromDouble(1.0 / (1.0 - rate));
        return _engine.TensorMultiply(x, _engine.TensorBroadcastTo(mask, x.Shape.ToArray()));
    }

    private Tensor<T> Dropout(Tensor<T> x, bool training, Random random)
    {
        double rate = Configuration.MlpDropout;
        if (!training || rate <= 0) return x;
        var mask = new Tensor<T>(x.Shape.ToArray());
        var keep = NumOps.FromDouble(1.0 / (1.0 - rate));
        for (int i = 0; i < mask.Length; i++) mask[i] = random.NextDouble() < rate ? NumOps.Zero : keep;
        return _engine.TensorMultiply(x, mask);
    }

    private Tensor<T> ForwardBlock(Block b, Tensor<T> x, bool training, Random random)
    {
        var shortcut = b.Norm1.Forward(x);
        var u = Gelu(b.Projection1.Forward(shortcut));
        var attention = b.Conv0.Forward(u);
        var sum = attention;
        for (int i = 0; i < 6; i += 2) sum = _engine.TensorAdd(sum, b.Strips[i + 1].Forward(b.Strips[i].Forward(attention)));
        var gated = _engine.TensorMultiply(b.Conv3.Forward(sum), u);
        var attn = _engine.TensorAdd(b.Projection2.Forward(gated), shortcut);
        x = _engine.TensorAdd(x, DropPath(b.Scale1.Forward(attn), b.DropPath, training, random));
        var h = Gelu(b.DepthwiseConv.Forward(b.Fc1.Forward(b.Norm2.Forward(x))));
        var mlp = Dropout(b.Fc2.Forward(Dropout(h, training, random)), training, random);
        return _engine.TensorAdd(x, DropPath(b.Scale2.Forward(mlp), b.DropPath, training, random));
    }

    /// <summary>MSCAN's four stage outputs [B, Cᵢ, H/2^(i+2), W/2^(i+2)].</summary>
    public List<Tensor<T>> Features(Tensor<T> images, bool training, Random random)
    {
        var outs = new List<Tensor<T>>();
        var x = StemNorm2.Forward(StemConv2.Forward(Gelu(StemNorm1.Forward(StemConv1.Forward(images)))));
        for (int stage = 0; stage < 4; stage++)
        {
            if (stage > 0) x = PatchNorms[stage - 1].Forward(PatchConvs[stage - 1].Forward(x));
            foreach (var block in Blocks[stage]) x = ForwardBlock(block, x, training, random);
            // LayerNorm over channels: [B, C, H, W] -> [B, H, W, C] and back.
            x = _engine.TensorPermute(StageNorms[stage].Forward(_engine.TensorPermute(x, new[] { 0, 2, 3, 1 })), new[] { 0, 3, 1, 2 });
            outs.Add(x);
        }
        return outs;
    }

    private static Tensor<T> Transpose(IEngine engine, Tensor<T> x) => engine.TensorPermute(x, new[] { 0, 2, 1 });

    /// <summary>Uniform bases on [0, 1), L2-normalized over the channels: [B, C, R] (NMF2D._build_bases).</summary>
    public Tensor<T> RandomBases(int batch, Random random)
    {
        int d = Configuration.HamChannels, r = Configuration.MatrixRank;
        var bases = new Tensor<T>(new[] { batch, d, r });
        var values = new double[batch * d * r];
        for (int i = 0; i < values.Length; i++) values[i] = random.NextDouble();
        for (int b = 0; b < batch; b++)
            for (int k = 0; k < r; k++)
            {
                double norm = 0;
                for (int c = 0; c < d; c++) norm += values[(b * d + c) * r + k] * values[(b * d + c) * r + k];
                norm = Math.Max(Math.Sqrt(norm), 1e-12);
                for (int c = 0; c < d; c++) bases[(b * d + c) * r + k] = NumOps.FromDouble(values[(b * d + c) * r + k] / norm);
            }
        return bases;
    }

    // NMF2D: X [B, C, N] ≈ B Cᵀ by multiplicative updates (inv_t = 1, ε = 1e-6).
    private Tensor<T> Nmf(Tensor<T> x, bool training, Tensor<T> bases)
    {
        var e = _engine;
        int batch = x.Shape[0], channels = x.Shape[1], height = x.Shape[2], width = x.Shape[3];
        var flat = e.Reshape(x, new[] { batch, channels, height * width });
        var flatT = Transpose(e, flat);
        var epsilon = NumOps.FromDouble(1e-6);
        var coef = e.Softmax(e.TensorBatchMatMul(flatT, bases), -1);
        Tensor<T> UpdateCoef(Tensor<T> c, Tensor<T> b)
        {
            var numerator = e.TensorBatchMatMul(flatT, b);
            var denominator = e.TensorBatchMatMul(c, e.TensorBatchMatMul(Transpose(e, b), b));
            return e.TensorDivide(e.TensorMultiply(c, numerator), e.TensorAddScalar(denominator, epsilon));
        }
        int steps = training ? Configuration.TrainSteps : Configuration.EvalSteps;
        for (int s = 0; s < steps; s++)
        {
            coef = UpdateCoef(coef, bases);
            var numerator = e.TensorBatchMatMul(flat, coef);
            var denominator = e.TensorBatchMatMul(bases, e.TensorBatchMatMul(Transpose(e, coef), coef));
            bases = e.TensorDivide(e.TensorMultiply(bases, numerator), e.TensorAddScalar(denominator, epsilon));
        }
        coef = UpdateCoef(coef, bases);
        return e.Reshape(e.TensorBatchMatMul(bases, Transpose(e, coef)), new[] { batch, channels, height, width });
    }

    private Tensor<T> ConvNormRelu(ConvNorm module, Tensor<T> x) => _engine.ReLU(module.Norm.Forward(module.Conv.Forward(x)));

    /// <summary>The class logits at stage 2's 1/8 scale.</summary>
    /// <param name="features">MSCAN's four stage outputs.</param>
    /// <param name="training">Training mode: 6 NMF steps and the dropouts.</param>
    /// <param name="random">The dropouts' and the bases' generator.</param>
    /// <param name="bases">The NMF's initial bases [B, C, R], or null to draw them (as the official code does every forward).</param>
    public Tensor<T> Logits(IReadOnlyList<Tensor<T>> features, bool training, Random random, Tensor<T>? bases = null)
    {
        var e = _engine;
        int h = features[1].Shape[2], w = features[1].Shape[3];
        var inputs = new[] { features[1], SpatialResampling.Resize(e, features[2], h, w), SpatialResampling.Resize(e, features[3], h, w) };
        var x = ConvNormRelu(Squeeze, e.TensorConcatenate(inputs, 1));
        var enjoy = e.ReLU(HamIn.Forward(x));
        enjoy = Nmf(enjoy, training, bases ?? RandomBases(x.Shape[0], random));
        enjoy = HamOut.Norm.Forward(HamOut.Conv.Forward(enjoy));
        var ham = e.ReLU(e.TensorAdd(x, enjoy));
        return Classifier.Forward(ConvNormRelu(Align, ham), training, random);
    }

    /// <summary>MSCAN then LightHamHead: the logits at 1/8 scale.</summary>
    public Tensor<T> Forward(Tensor<T> images, bool training, Random random, Tensor<T>? bases = null)
        => Logits(Features(images, training, random), training, random, bases);

    /// <summary>Loads mmsegmentation's state dict: <c>backbone.*</c> (MSCAN) and <c>decode_head.*</c> (LightHamHead).</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read)
    {
        void Conv(ConvolutionalLayer<T> conv, string name, bool bias)
        {
            int k = conv.KernelSize;
            var w = read($"{name}.weight", new[] { conv.OutputDepth, conv.InputDepth, k, k });
            var values = bias ? w.Concat(read($"{name}.bias", new[] { conv.OutputDepth })).ToArray() : w;
            conv.SetParameters(new Vector<T>(values.Select(NumOps.FromDouble).ToArray()));
        }
        void Bn(BatchNormalizationLayer<T> norm, string name, int c)
        {
            var s = new[] { c };
            TorchParameters.BatchNorm(norm, read($"{name}.weight", s), read($"{name}.bias", s), read($"{name}.running_mean", s), read($"{name}.running_var", s));
        }
        void Dw(StripDepthwiseConvolution<T> conv, string name, int c, int kh, int kw)
            => conv.Load(read($"{name}.weight", new[] { c, 1, kh, kw }), read($"{name}.bias", new[] { c }));
        void Gn(GroupNormalizationLayer<T> norm, string name, int c)
            => norm.LoadAffine(read($"{name}.weight", new[] { c }), read($"{name}.bias", new[] { c }));

        var cfg = Configuration;
        int d0 = cfg.EmbedDims[0];
        Conv(StemConv1, "backbone.patch_embed1.proj.0", true);
        Bn(StemNorm1, "backbone.patch_embed1.proj.1", d0 / 2);
        Conv(StemConv2, "backbone.patch_embed1.proj.3", true);
        Bn(StemNorm2, "backbone.patch_embed1.proj.4", d0);
        int[][] strips = { new[] { 1, 7 }, new[] { 7, 1 }, new[] { 1, 11 }, new[] { 11, 1 }, new[] { 1, 21 }, new[] { 21, 1 } };
        string[] stripNames = { "conv0_1", "conv0_2", "conv1_1", "conv1_2", "conv2_1", "conv2_2" };
        for (int stage = 0; stage < 4; stage++)
        {
            int d = cfg.EmbedDims[stage];
            if (stage > 0)
            {
                Conv(PatchConvs[stage - 1], $"backbone.patch_embed{stage + 1}.proj", true);
                Bn(PatchNorms[stage - 1], $"backbone.patch_embed{stage + 1}.norm", d);
            }
            for (int j = 0; j < Blocks[stage].Count; j++)
            {
                var b = Blocks[stage][j];
                string p = $"backbone.block{stage + 1}.{j}";
                int hidden = d * cfg.MlpRatios[stage];
                Bn(b.Norm1, $"{p}.norm1", d);
                Conv(b.Projection1, $"{p}.attn.proj_1", true);
                Dw(b.Conv0, $"{p}.attn.spatial_gating_unit.conv0", d, 5, 5);
                for (int i = 0; i < 6; i++) Dw(b.Strips[i], $"{p}.attn.spatial_gating_unit.{stripNames[i]}", d, strips[i][0], strips[i][1]);
                Conv(b.Conv3, $"{p}.attn.spatial_gating_unit.conv3", true);
                Conv(b.Projection2, $"{p}.attn.proj_2", true);
                b.Scale1.Load(read($"{p}.layer_scale_1", new[] { d }));
                b.Scale2.Load(read($"{p}.layer_scale_2", new[] { d }));
                Bn(b.Norm2, $"{p}.norm2", d);
                Conv(b.Fc1, $"{p}.mlp.fc1", true);
                Dw(b.DepthwiseConv, $"{p}.mlp.dwconv", hidden, 3, 3);
                Conv(b.Fc2, $"{p}.mlp.fc2", true);
            }
            TorchParameters.LayerNorm(_engine, StageNorms[stage], read($"backbone.norm{stage + 1}.weight", new[] { d }),
                read($"backbone.norm{stage + 1}.bias", new[] { d }));
        }
        Conv(Squeeze.Conv, "decode_head.squeeze.conv", false);
        Gn(Squeeze.Norm, "decode_head.squeeze.gn", cfg.HamChannels);
        Conv(HamIn, "decode_head.hamburger.ham_in.conv", true);
        Conv(HamOut.Conv, "decode_head.hamburger.ham_out.conv", false);
        Gn(HamOut.Norm, "decode_head.hamburger.ham_out.gn", cfg.HamChannels);
        Conv(Align.Conv, "decode_head.align.conv", false);
        Gn(Align.Norm, "decode_head.align.gn", cfg.HeadChannels);
        Classifier.Load(read, "decode_head.conv_seg");
    }

    /// <summary>
    /// The official initialization without a pretrained encoder: MSCAN.init_weights draws every convolution from
    /// N(0, 2/fan_out) with fan_out = k_h·k_w·out/groups and zero bias, LayerNorms 1 and 0; BatchNorms keep PyTorch's 1
    /// and 0; layer scales 0.01. The head's ConvModules use mmcv's kaiming_init (normal, fan_out, ReLU gain: the same
    /// N(0, 2/fan_out)), GroupNorms 1 and 0, and conv_seg N(0, 0.01²) with zero bias.
    /// </summary>
    public void InitializeLikeReference(Random random)
    {
        void Conv(ConvolutionalLayer<T> conv)
        {
            int k = conv.KernelSize, outputs = conv.OutputDepth, inputs = conv.InputDepth;
            var w = TorchParameters.Gaussian<T>(random, outputs * inputs * k * k, Math.Sqrt(2.0 / ((double)k * k * outputs)));
            int count = conv.GetParameters().Length;
            var full = new Vector<T>(count);
            for (int i = 0; i < w.Length; i++) full[i] = w[i];
            conv.SetParameters(full);
        }
        void Dw(StripDepthwiseConvolution<T> conv, int channels)
        {
            var w = TorchParameters.Gaussian<T>(random, channels * conv.KernelArea, Math.Sqrt(2.0 / conv.KernelArea));
            conv.Load(w.ToArray().Select(v => NumOps.ToDouble(v)).ToArray(), new double[channels]);
        }
        void Bn(BatchNormalizationLayer<T> norm, int c)
            => TorchParameters.BatchNorm(norm, Enumerable.Repeat(1.0, c).ToArray(), new double[c], new double[c], Enumerable.Repeat(1.0, c).ToArray());

        var cfg = Configuration;
        int d0 = cfg.EmbedDims[0];
        Conv(StemConv1);
        Bn(StemNorm1, d0 / 2);
        Conv(StemConv2);
        Bn(StemNorm2, d0);
        for (int stage = 0; stage < 4; stage++)
        {
            int d = cfg.EmbedDims[stage];
            if (stage > 0)
            {
                Conv(PatchConvs[stage - 1]);
                Bn(PatchNorms[stage - 1], d);
            }
            foreach (var b in Blocks[stage])
            {
                int hidden = d * cfg.MlpRatios[stage];
                Bn(b.Norm1, d);
                Conv(b.Projection1);
                Dw(b.Conv0, d);
                foreach (var strip in b.Strips) Dw(strip, d);
                Conv(b.Conv3);
                Conv(b.Projection2);
                b.Scale1.Load(Enumerable.Repeat(0.01, d).ToArray());
                b.Scale2.Load(Enumerable.Repeat(0.01, d).ToArray());
                Bn(b.Norm2, d);
                Conv(b.Fc1);
                Dw(b.DepthwiseConv, hidden);
                Conv(b.Fc2);
            }
            TorchParameters.LayerNorm(_engine, StageNorms[stage], Enumerable.Repeat(1.0, d).ToArray(), new double[d]);
        }
        foreach (var module in new[] { Squeeze, HamOut, Align })
        {
            Conv(module.Conv);
            module.Norm.LoadAffine(Enumerable.Repeat(1.0, module.Norm.NumChannels).ToArray(), new double[module.Norm.NumChannels]);
        }
        Conv(HamIn);
        Classifier.Initialize(random);
    }
}

using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.ModelLoading;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion.Models;

/// <summary>
/// The shared U-Net against LDM's <c>ldm/modules/diffusionmodules/openaimodel.py</c> <c>UNetModel</c> (Rombach et al.,
/// 2022): hand-derived Stable Diffusion parameter counts, every official state-dict name loaded once with its shape,
/// and the <c>Upsample</c> and <c>ResBlock</c> modules against hand references.
/// </summary>
public class LdmUNetCheckpointTests
{
    // ------------------------------------------------------------------ hand-derived counts (openaimodel.py)

    private static long GroupNorm(long c) => 2 * c;
    private static long Conv(long i, long o, long k) => (i * o * k * k) + o;
    private static long Linear(long i, long o, bool bias = true) => (i * o) + (bias ? o : 0);

    // ResBlock: in_layers (GroupNorm, SiLU, conv3x3), emb_layers (SiLU, Linear), out_layers (GroupNorm, SiLU, dropout,
    // conv3x3), skip_connection (conv1x1 when the channel count changes).
    private static long ResBlock(long i, long o, long emb)
        => GroupNorm(i) + Conv(i, o, 3) + Linear(emb, o) + GroupNorm(o) + Conv(o, o, 3) + (i != o ? Conv(i, o, 1) : 0);

    // SpatialTransformer (ldm/modules/attention.py), depth 1: norm, proj_in, BasicTransformerBlock (norm1, attn1 with
    // bias-free q/k/v, norm2, attn2 with k/v from the context, norm3, GEGLU ff), proj_out. A conv1x1 projection and
    // a linear one hold the same parameters; the head count changes none.
    private static long SpatialTransformer(long c, long context)
        => GroupNorm(c) + Linear(c, c)
           + GroupNorm(c) + (3 * c * c) + Linear(c, c)
           + GroupNorm(c) + (c * c) + (2 * context * c) + Linear(c, c)
           + GroupNorm(c) + Linear(c, 8 * c) + Linear(4 * c, c)
           + Linear(c, c);

    private static long LdmUNet(int modelChannels, int[] mult, int numRes, int[] attention, int context, int inChannels,
        int outChannels)
    {
        long emb = 4L * modelChannels;
        long total = Linear(modelChannels, emb) + Linear(emb, emb) + Conv(inChannels, modelChannels, 3);
        var chans = new List<long> { modelChannels };
        long ch = modelChannels;
        for (int level = 0; level < mult.Length; level++)
        {
            for (int r = 0; r < numRes; r++)
            {
                total += ResBlock(ch, mult[level] * (long)modelChannels, emb);
                ch = mult[level] * (long)modelChannels;
                if (attention.Contains(level)) total += SpatialTransformer(ch, context);
                chans.Add(ch);
            }
            if (level != mult.Length - 1)
            {
                total += Conv(ch, ch, 3);
                chans.Add(ch);
            }
        }

        total += ResBlock(ch, ch, emb) + SpatialTransformer(ch, context) + ResBlock(ch, ch, emb);
        for (int level = mult.Length - 1; level >= 0; level--)
        {
            for (int r = 0; r <= numRes; r++)
            {
                long skip = chans[chans.Count - 1];
                chans.RemoveAt(chans.Count - 1);
                total += ResBlock(ch + skip, mult[level] * (long)modelChannels, emb);
                ch = mult[level] * (long)modelChannels;
                if (attention.Contains(level)) total += SpatialTransformer(ch, context);
                if (level > 0 && r == numRes) total += Conv(ch, ch, 3);
            }
        }

        return total + GroupNorm(ch) + Conv(ch, outChannels, 3);
    }

    private static UNetNoisePredictor<float> StableDiffusion(int context, MultiViewUNetOptions? options)
        => new UNetNoisePredictor<float>(inputChannels: 4, outputChannels: 4, baseChannels: 320,
            channelMultipliers: new[] { 1, 2, 4, 4 }, numResBlocks: 2, attentionResolutions: new[] { 0, 1, 2 },
            contextDim: context, numHeads: 8, inputHeight: 64, multiView: options);

    [Fact]
    public void StableDiffusion15UNet_HasTheHandDerivedParameterCount()
    {
        // 859,520,964 is diffusers' UNet2DConditionModel.num_parameters() for runwayml/stable-diffusion-v1-5 (the SD
        // 1.x config: 320 channels, mult 1-2-4-4, 2 res blocks, attention at the first three levels, context 768).
        long expected = LdmUNet(320, new[] { 1, 2, 4, 4 }, 2, new[] { 0, 1, 2 }, 768, 4, 4);
        Assert.Equal(859_520_964L, expected);
        Assert.Equal(expected, StableDiffusion(768, MultiViewUNetOptions.StableDiffusion1()).ParameterCount);
        // The default-built U-Net (no option) is the same LDM UNetModel.
        Assert.Equal(expected, StableDiffusion(768, null).ParameterCount);
    }

    [Fact]
    public void StableDiffusion2BaseUNet_HasTheHandDerivedParameterCount()
    {
        // 865,910,724 is diffusers' num_parameters() for stabilityai/stable-diffusion-2-base (context 1024, linear
        // transformer projections, 64-wide heads).
        long expected = LdmUNet(320, new[] { 1, 2, 4, 4 }, 2, new[] { 0, 1, 2 }, 1024, 4, 4);
        Assert.Equal(865_910_724L, expected);
        Assert.Equal(expected, StableDiffusion(1024, MultiViewUNetOptions.StableDiffusion2()).ParameterCount);
        Assert.Equal(expected, StableDiffusion(1024, null).ParameterCount);
    }

    // ------------------------------------------------------------------ official names and shapes

    private const string Prefix = "model.diffusion_model.";
    private const int Emb = 128;

    // A tiny LDM U-Net: 32 channels, mult 1-2, one res block per level, attention at level 1, context 8, linear
    // projections. Its state dict, written out by hand from openaimodel.py.
    private static UNetNoisePredictor<double> TinyUNet() => new UNetNoisePredictor<double>(inputChannels: 4,
        outputChannels: 4, baseChannels: 32, channelMultipliers: new[] { 1, 2 }, numResBlocks: 1,
        attentionResolutions: new[] { 1 }, contextDim: 8, numHeads: 1, inputHeight: 8, seed: 3,
        multiView: MultiViewUNetOptions.StableDiffusion2());

    private static Dictionary<string, int[]> TinyStateDictShapes() => TinyStateDictShapes(8, convolutionProjections: false);

    // proj_in / proj_out are [C, C, 1, 1] convolutions in SD 1.x (use_linear_in_transformer False) and [C, C] linears
    // in SD 2.x.
    private static Dictionary<string, int[]> TinyStateDictShapes(int contextWidth, bool convolutionProjections)
    {
        var shapes = new Dictionary<string, int[]>(StringComparer.Ordinal);
        void Pair(string name, int[] weight, int bias) { shapes[name + ".weight"] = weight; shapes[name + ".bias"] = new[] { bias }; }
        void Norm(string name, int c) => Pair(name, new[] { c }, c);
        void ConvShape(string name, int i, int o, int k) => Pair(name, new[] { o, i, k, k }, o);
        void Res(string name, int i, int o)
        {
            Norm(name + "in_layers.0", i);
            ConvShape(name + "in_layers.2", i, o, 3);
            Pair(name + "emb_layers.1", new[] { o, Emb }, o);
            Norm(name + "out_layers.0", o);
            ConvShape(name + "out_layers.3", o, o, 3);
            if (i != o) ConvShape(name + "skip_connection", i, o, 1);
        }
        void Transformer(string name, int c, int context)
        {
            Norm(name + "norm", c);
            Pair(name + "proj_in", convolutionProjections ? new[] { c, c, 1, 1 } : new[] { c, c }, c);
            string b = name + "transformer_blocks.0.";
            Norm(b + "norm1", c);
            shapes[b + "attn1.to_q.weight"] = new[] { c, c };
            shapes[b + "attn1.to_k.weight"] = new[] { c, c };
            shapes[b + "attn1.to_v.weight"] = new[] { c, c };
            Pair(b + "attn1.to_out.0", new[] { c, c }, c);
            Norm(b + "norm2", c);
            shapes[b + "attn2.to_q.weight"] = new[] { c, c };
            shapes[b + "attn2.to_k.weight"] = new[] { c, context };
            shapes[b + "attn2.to_v.weight"] = new[] { c, context };
            Pair(b + "attn2.to_out.0", new[] { c, c }, c);
            Norm(b + "norm3", c);
            Pair(b + "ff.net.0.proj", new[] { 8 * c, c }, 8 * c);
            Pair(b + "ff.net.2", new[] { c, 4 * c }, c);
            Pair(name + "proj_out", convolutionProjections ? new[] { c, c, 1, 1 } : new[] { c, c }, c);
        }

        string p = Prefix;
        Pair(p + "time_embed.0", new[] { Emb, 32 }, Emb);
        Pair(p + "time_embed.2", new[] { Emb, Emb }, Emb);
        ConvShape(p + "input_blocks.0.0", 4, 32, 3);
        Res(p + "input_blocks.1.0.", 32, 32);
        ConvShape(p + "input_blocks.2.0.op", 32, 32, 3);
        Res(p + "input_blocks.3.0.", 32, 64);
        Transformer(p + "input_blocks.3.1.", 64, contextWidth);
        Res(p + "middle_block.0.", 64, 64);
        Transformer(p + "middle_block.1.", 64, contextWidth);
        Res(p + "middle_block.2.", 64, 64);
        // input_block_chans = [32, 32, 32, 64], popped from the end.
        Res(p + "output_blocks.0.0.", 128, 64);
        Transformer(p + "output_blocks.0.1.", 64, contextWidth);
        Res(p + "output_blocks.1.0.", 96, 64);
        Transformer(p + "output_blocks.1.1.", 64, contextWidth);
        ConvShape(p + "output_blocks.1.2.conv", 64, 64, 3);
        Res(p + "output_blocks.2.0.", 96, 32);
        Res(p + "output_blocks.3.0.", 64, 32);
        Norm(p + "out.0", 32);
        ConvShape(p + "out.2", 32, 4, 3);
        return shapes;
    }

    private static int Count(int[] shape) => shape.Aggregate(1, (a, b) => a * b);

    private static Dictionary<string, Tensor<float>> TinyStateDict(int seed)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        var state = new Dictionary<string, Tensor<float>>(StringComparer.Ordinal);
        foreach (var (name, shape) in TinyStateDictShapes())
        {
            var values = new float[Count(shape)];
            for (int i = 0; i < values.Length; i++) values[i] = (float)((2.0 * random.NextDouble()) - 1.0) * 0.1f;
            state[name] = new Tensor<float>(values, shape);
        }
        return state;
    }

    [Fact]
    public void LoadTorchWeights_ReadsEveryOfficialNameOnceWithItsShape()
    {
        var expected = TinyStateDictShapes();
        var requested = new Dictionary<string, int[]>(StringComparer.Ordinal);
        var source = TorchTensorSource.From((string name, int[] shape) =>
        {
            Assert.False(requested.ContainsKey(name), $"'{name}' was read twice.");
            requested[name] = shape;
            return new double[Count(shape)];
        });

        var unet = TinyUNet();
        var names = unet.LoadTorchWeights(source, null);

        Assert.Equal(expected.Count, names.Count);
        Assert.Equal(names.Count, names.Distinct().Count());
        Assert.Equal(expected.Keys.OrderBy(n => n, StringComparer.Ordinal), names.OrderBy(n => n, StringComparer.Ordinal));
        foreach (var (name, shape) in expected)
            Assert.True(shape.SequenceEqual(requested[name]),
                $"'{name}' was read as [{string.Join(", ", requested[name])}], expected [{string.Join(", ", shape)}].");
    }

    [Fact]
    public void LoadTorchWeights_FromAStateDict_LoadsAllAndThrowsOnMissingOrUnknownNames()
    {
        var state = TinyStateDict(5);
        var names = TinyUNet().LoadTorchWeights(state);
        Assert.Equal(state.Count, names.Count);

        var missing = new Dictionary<string, Tensor<float>>(state, StringComparer.Ordinal);
        missing.Remove(Prefix + "output_blocks.1.2.conv.weight");
        var ex = Assert.Throws<InvalidOperationException>(() => TinyUNet().LoadTorchWeights(missing));
        Assert.Contains("output_blocks.1.2.conv.weight", ex.Message);
        var partial = TinyUNet().LoadTorchWeights(missing, allowPartial: true);
        Assert.Equal(state.Count - 2, partial.Count);
        Assert.DoesNotContain(Prefix + "output_blocks.1.2.conv.bias", partial);

        var unknown = new Dictionary<string, Tensor<float>>(state, StringComparer.Ordinal)
        {
            [Prefix + "output_blocks.9.0.in_layers.0.weight"] = new Tensor<float>(new float[4], new[] { 4 })
        };
        ex = Assert.Throws<InvalidOperationException>(() => TinyUNet().LoadTorchWeights(unknown));
        Assert.Contains("output_blocks.9.0.in_layers.0.weight", ex.Message);
        Assert.Equal(state.Count, TinyUNet().LoadTorchWeights(unknown, allowPartial: true).Count);

        // Tensors outside the U-Net's prefix (the VAE's, the text encoder's) are not the U-Net's to reject.
        var withVae = new Dictionary<string, Tensor<float>>(state, StringComparer.Ordinal)
        {
            ["first_stage_model.encoder.conv_in.weight"] = new Tensor<float>(new float[4], new[] { 4 })
        };
        Assert.Equal(state.Count, TinyUNet().LoadTorchWeights(withVae).Count);
    }

    [Fact]
    public void LoadTorchWeights_ProducesTheSamePredictionFromTheSameCheckpoint()
    {
        var state = TinyStateDict(9);
        var a = TinyUNet();
        var b = new UNetNoisePredictor<double>(inputChannels: 4, outputChannels: 4, baseChannels: 32,
            channelMultipliers: new[] { 1, 2 }, numResBlocks: 1, attentionResolutions: new[] { 1 }, contextDim: 8,
            numHeads: 1, inputHeight: 8, seed: 77, multiView: MultiViewUNetOptions.StableDiffusion2());
        a.LoadTorchWeights(state);
        b.LoadTorchWeights(state);

        var x = new Tensor<double>(Values(4 * 8 * 8, 1), new[] { 1, 4, 8, 8 });
        var context = new Tensor<double>(Values(3 * 8, 2), new[] { 1, 3, 8 });
        var first = a.PredictNoise(x, 500, context).ToArray();
        Assert.Equal(first, b.PredictNoise(x, 500, context).ToArray());
        // out.2 is loaded (non-zero), so the prediction is no longer the zero-initialised head's zeros.
        Assert.Contains(first, v => Math.Abs(v) > 1e-6);
    }

    // ------------------------------------------------------------------ module references

    private static double[] Values(int count, int seed)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        var values = new double[count];
        for (int i = 0; i < count; i++) values[i] = (2.0 * random.NextDouble()) - 1.0;
        return values;
    }

    private sealed class Weights
    {
        private readonly Dictionary<string, double[]> _values = new(StringComparer.Ordinal);
        private int _seed;

        public double[] Read(string name, int[] shape)
        {
            if (!_values.TryGetValue(name, out var values))
            {
                values = Values(Count(shape), ++_seed);
                if (name.Contains("in_layers.0.weight") || name.Contains("out_layers.0.weight"))
                    values = values.Select(v => 1.0 + (0.3 * v)).ToArray();
                _values[name] = values;
            }
            return values;
        }

        public double[] this[string name] => _values[name];

        public TorchTensorSource Source() => TorchTensorSource.From((string name, int[] shape) => Read(name, shape));
    }

    // Zero-padded 3x3 (or 1x1) convolution, torch layout weight [out, in, k, k].
    private static double[] ReferenceConv(double[] x, int inC, int h, int w, double[] weight, double[] bias, int outC, int k)
    {
        int pad = k / 2;
        var y = new double[outC * h * w];
        for (int o = 0; o < outC; o++)
            for (int r = 0; r < h; r++)
                for (int c = 0; c < w; c++)
                {
                    double sum = bias[o];
                    for (int i = 0; i < inC; i++)
                        for (int kr = 0; kr < k; kr++)
                            for (int kc = 0; kc < k; kc++)
                            {
                                int rr = r + kr - pad, cc = c + kc - pad;
                                if (rr < 0 || rr >= h || cc < 0 || cc >= w) continue;
                                sum += weight[(((o * inC) + i) * k * k) + (kr * k) + kc] * x[(((i * h) + rr) * w) + cc];
                            }
                    y[(((o * h) + r) * w) + c] = sum;
                }
        return y;
    }

    private static double[] ReferenceGroupNormSiLU(double[] x, int channels, int groups, int spatial, double[] gamma,
        double[] beta, double eps = 1e-5)
    {
        var y = new double[x.Length];
        int per = channels / groups;
        for (int g = 0; g < groups; g++)
        {
            int start = g * per * spatial, count = per * spatial;
            double mean = 0.0;
            for (int i = 0; i < count; i++) mean += x[start + i];
            mean /= count;
            double variance = 0.0;
            for (int i = 0; i < count; i++) variance += (x[start + i] - mean) * (x[start + i] - mean);
            variance /= count;
            double inv = 1.0 / Math.Sqrt(variance + eps);
            for (int i = 0; i < count; i++)
            {
                int channel = (start + i) / spatial;
                double v = ((x[start + i] - mean) * inv * gamma[channel]) + beta[channel];
                y[start + i] = v / (1.0 + Math.Exp(-v));
            }
        }
        return y;
    }

    private static double MaxAbsDifference(double[] a, double[] b)
    {
        Assert.Equal(a.Length, b.Length);
        double max = 0.0;
        for (int i = 0; i < a.Length; i++) max = Math.Max(max, Math.Abs(a[i] - b[i]));
        return max;
    }

    [Fact]
    public void Upsample_IsNearestTimesTwoThenConv3x3()
    {
        const int channels = 3, h = 3, w = 2;
        var layer = new LdmUpsampleLayer<double>(channels);
        var weights = new Weights();
        var names = layer.LoadTorchWeights(weights.Source(), "up.");
        Assert.Equal(new[] { "up.conv.weight", "up.conv.bias" }, names);

        var x = Values(channels * h * w, 40);
        var output = layer.Forward(new Tensor<double>(x, new[] { 1, channels, h, w }));
        Assert.Equal(new[] { 1, channels, 2 * h, 2 * w }, output.Shape.ToArray());

        // F.interpolate(x, scale_factor=2, mode="nearest"): up[c, r, s] = x[c, r / 2, s / 2].
        var up = new double[channels * 4 * h * w];
        for (int c = 0; c < channels; c++)
            for (int r = 0; r < 2 * h; r++)
                for (int s = 0; s < 2 * w; s++)
                    up[(((c * 2 * h) + r) * 2 * w) + s] = x[(((c * h) + (r / 2)) * w) + (s / 2)];
        var expected = ReferenceConv(up, channels, 2 * h, 2 * w, weights["up.conv.weight"], weights["up.conv.bias"], channels, 3);
        Assert.True(MaxAbsDifference(expected, output.ToArray()) < 1e-10);
    }

    [Fact]
    public void ResBlock_MatchesTheOpenaimodelReference()
    {
        const int inC = 4, outC = 6, size = 3, emb = 5, groups = 2;
        var block = new DiffusionResBlock<double>(inC, outC, size, timeEmbedDim: emb, numGroups: groups);
        var weights = new Weights();
        var names = block.LoadTorchWeights(weights.Source(), "r.");
        Assert.Equal(new[]
        {
            "r.in_layers.0.weight", "r.in_layers.0.bias", "r.in_layers.2.weight", "r.in_layers.2.bias",
            "r.emb_layers.1.weight", "r.emb_layers.1.bias", "r.out_layers.0.weight", "r.out_layers.0.bias",
            "r.out_layers.3.weight", "r.out_layers.3.bias", "r.skip_connection.weight", "r.skip_connection.bias"
        }, names);
        block.SetTrainingMode(false);

        int spatial = size * size;
        var x = Values(inC * spatial, 50);
        var t = Values(emb, 51);
        var output = block.Forward(new Tensor<double>(x, new[] { 1, inC, size, size }), new Tensor<double>(t, new[] { 1, emb }));

        var h = ReferenceGroupNormSiLU(x, inC, groups, spatial, weights["r.in_layers.0.weight"], weights["r.in_layers.0.bias"]);
        h = ReferenceConv(h, inC, size, size, weights["r.in_layers.2.weight"], weights["r.in_layers.2.bias"], outC, 3);
        // emb_layers: SiLU then Linear, added per channel.
        var siluT = t.Select(v => v / (1.0 + Math.Exp(-v))).ToArray();
        var embW = weights["r.emb_layers.1.weight"];
        var embB = weights["r.emb_layers.1.bias"];
        for (int o = 0; o < outC; o++)
        {
            double e = embB[o];
            for (int i = 0; i < emb; i++) e += embW[(o * emb) + i] * siluT[i];
            for (int s = 0; s < spatial; s++) h[(o * spatial) + s] += e;
        }
        h = ReferenceGroupNormSiLU(h, outC, groups, spatial, weights["r.out_layers.0.weight"], weights["r.out_layers.0.bias"]);
        h = ReferenceConv(h, outC, size, size, weights["r.out_layers.3.weight"], weights["r.out_layers.3.bias"], outC, 3);
        var skip = ReferenceConv(x, inC, size, size, weights["r.skip_connection.weight"], weights["r.skip_connection.bias"], outC, 1);
        var expected = h.Zip(skip, (a, b) => a + b).ToArray();

        Assert.True(MaxAbsDifference(expected, output.ToArray()) < 1e-9);
    }

    // ------------------------------------------------------------------ the default-built attention

    [Fact]
    public void DefaultSd15ConfigUNet_LoadsEveryOfficialNameOnceWithoutAllowPartial()
    {
        // No multi-view option: a 768-wide context makes the default attention SD 1.x's SpatialTransformer
        // (v1-inference.yaml: num_heads, conv1x1 proj_in / proj_out).
        var unet = new UNetNoisePredictor<double>(inputChannels: 4, outputChannels: 4, baseChannels: 32,
            channelMultipliers: new[] { 1, 2 }, numResBlocks: 1, attentionResolutions: new[] { 1 }, contextDim: 768,
            numHeads: 8, inputHeight: 8, seed: 3);
        Assert.Null(unet.MultiViewOptions);

        var expected = TinyStateDictShapes(768, convolutionProjections: true);
        var requested = new Dictionary<string, int[]>(StringComparer.Ordinal);
        var source = TorchTensorSource.From((string name, int[] shape) =>
        {
            Assert.False(requested.ContainsKey(name), $"'{name}' was read twice.");
            requested[name] = shape;
            return new double[Count(shape)];
        });
        var names = unet.LoadTorchWeights(source, expected.Keys, allowPartial: false);

        Assert.Equal(expected.Count, names.Count);
        Assert.Equal(expected.Keys.OrderBy(n => n, StringComparer.Ordinal), names.OrderBy(n => n, StringComparer.Ordinal));
        foreach (var (name, shape) in expected)
            Assert.True(shape.SequenceEqual(requested[name]),
                $"'{name}' was read as [{string.Join(", ", requested[name])}], expected [{string.Join(", ", shape)}].");
    }

    private static UNetNoisePredictor<double> TinyContextUNet(int seed) => new UNetNoisePredictor<double>(inputChannels: 4,
        outputChannels: 4, baseChannels: 32, channelMultipliers: new[] { 1, 2 }, numResBlocks: 1,
        attentionResolutions: new[] { 1 }, contextDim: 8, numHeads: 2, inputHeight: 8, seed: seed);

    private static double[] Pattern(int count, double scale, double frequency)
    {
        var values = new double[count];
        for (int i = 0; i < count; i++) values[i] = scale * Math.Sin(frequency * (i + 1));
        return values;
    }

    [Fact]
    public void TLoRADecorator_WrapsTheSpatialTransformerAndReachesTheOutput()
    {
        var unet = TinyContextUNet(4);
        var adapters = new List<AiDotNet.Diffusion.StyleTransfer.TLoRAAttentionAdapter<double>>();
        unet.DecorateAttentionBlocks((block, channels, isCrossAttention) =>
        {
            Assert.True(isCrossAttention);
            Assert.IsType<MultiViewTransformerLayer<double>>(block);
            var adapter = new AiDotNet.Diffusion.StyleTransfer.TLoRAAttentionAdapter<double>(block, channels, rank: 2,
                totalTimesteps: 1000, random: RandomHelper.CreateSeededRandom(8));
            adapters.Add(adapter);
            return adapter;
        });
        // input_blocks.3.1, middle_block.1, output_blocks.0.1 and output_blocks.1.1.
        Assert.Equal(4, adapters.Count);

        // Non-zero weights everywhere (proj_out and the output conv start at zero), so the attention reaches the output.
        unet.SetParameters(new Vector<double>(Pattern(unet.GetParameters().Length, 0.05, 0.37)));
        var x = new Tensor<double>(Values(4 * 8 * 8, 11), new[] { 1, 4, 8, 8 });
        var context = new Tensor<double>(Values(3 * 8, 12), new[] { 1, 3, 8 });
        var before = unet.PredictNoise(x, 200, context).ToArray();

        var adapter0 = adapters[2];
        var state = adapter0.GetAdapterState().ToArray().Select((v, i) => v + (0.1 * Math.Cos(i + 1.0))).ToArray();
        adapter0.SetAdapterState(new Vector<double>(state));
        var after = unet.PredictNoise(x, 200, context).ToArray();
        Assert.Contains(before.Zip(after, (a, b) => Math.Abs(a - b)), d => d > 1e-9);
    }

    [Fact]
    public void Wonder3DDecorator_RunsMultiViewSelfAttentionAndCrossDomainAttentionInsideTheTransformer()
    {
        const int channels = 4, views = 2;
        var transformer = new MultiViewTransformerLayer<double>(channels, 0, MultiViewUNetOptions.StableDiffusion2());
        var block = new AiDotNet.Diffusion.ThreeD.Wonder3DAttentionBlock<double>(transformer, channels, views, 1, seed: 5);
        Assert.Same(transformer, block.Inner);
        int perView = channels * 2 * 2;
        var x = Pattern(2 * views * perView, 0.8, 0.17); // [N0, N1, C0, C1]
        block.Forward(new Tensor<double>(x, new[] { 2 * views, channels, 2, 2 }));
        transformer.SetParameters(new Vector<double>(Pattern(transformer.GetParameters().Length, 0.4, 0.43)));

        double[] Run(double[] input) => block.Forward(new Tensor<double>(input, new[] { 2 * views, channels, 2, 2 })).ToArray();
        double MaxDifference(double[] a, double[] b, int view)
        {
            double m = 0;
            for (int i = view * perView; i < (view + 1) * perView; i++) m = Math.Max(m, Math.Abs(a[i] - b[i]));
            return m;
        }

        var baseline = Run(x);
        var perturbed = (double[])x.Clone();
        perturbed[(1 * perView) + 2] += 0.9; // normal view 1
        var changed = Run(perturbed);
        // attn1 joins the views of one domain (MVAttnProcessor); the cross-domain output projection starts at zero.
        Assert.True(MaxDifference(baseline, changed, 0) > 1e-6);    // normal view 0 attends to normal view 1
        Assert.Equal(0.0, MaxDifference(baseline, changed, 2), 1e-12); // the color views do not
        Assert.Equal(0.0, MaxDifference(baseline, changed, 3), 1e-12);

        // With the cross-domain attention trained, color view 1 reads normal view 1 inside the transformer.
        int count = block.CrossDomainAttention.GetParameters().Length;
        block.CrossDomainAttention.SetParameters(new Vector<double>(Pattern(count, 0.4, 0.23)));
        baseline = Run(x);
        changed = Run(perturbed);
        Assert.True(MaxDifference(baseline, changed, 3) > 1e-6);
    }
}

using AiDotNet.Diffusion.Conditioning;
using AiDotNet.Diffusion.ThreeD;
using AiDotNet.Diffusion.VAE;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion.Models;

/// <summary>
/// MVDream's and ImageDream's paper conditioners, built by default and lazy: the OpenCLIP ViT-H/14 text encoder as
/// LDM's <c>FrozenOpenCLIPEmbedder</c> runs it (layer "penultimate"), and the OpenCLIP ViT-H/14 image encoder. Neither
/// allocates a weight until it is used; the CLIP image encoder deep-copies.
/// </summary>
public class MVDreamDefaultConditionerTests
{
    // ------------------------------------------------------------------ helpers

    private sealed class Weights
    {
        private readonly Dictionary<string, double[]> _values = new(StringComparer.Ordinal);
        private int _seed = 100;

        public double[] Read(string name, int[] shape)
        {
            if (!_values.TryGetValue(name, out var values))
            {
                int count = shape.Aggregate(1, (a, b) => a * b);
                var random = RandomHelper.CreateSeededRandom(++_seed);
                bool norm = name.Contains("ln_") && name.EndsWith(".weight", StringComparison.Ordinal);
                values = new double[count];
                for (int i = 0; i < count; i++)
                {
                    double u = (2.0 * random.NextDouble()) - 1.0;
                    values[i] = norm ? 1.0 + (0.3 * u) : 0.3 * u;
                }
                _values[name] = values;
            }
            return values;
        }

        public double[] this[string name] => _values[name];
    }

    private static double Erf(double z)
    {
        double a = Math.Abs(z);
        double value;
        if (a < 2.5)
        {
            double term = a, sum = a;
            for (int n = 1; n < 80; n++)
            {
                term *= -a * a / n;
                sum += term / ((2 * n) + 1);
            }
            value = 2.0 / Math.Sqrt(Math.PI) * sum;
        }
        else
        {
            double k = a;
            for (int n = 80; n >= 1; n--) k = a + (n / 2.0 / k);
            value = 1.0 - (Math.Exp(-a * a) / (Math.Sqrt(Math.PI) * k));
        }
        return z < 0 ? -value : value;
    }

    private static double[] Linear(double[] weight, double[] bias, double[] x, int outputs, int offset = 0)
    {
        int inputs = x.Length;
        var y = new double[outputs];
        for (int o = 0; o < outputs; o++)
        {
            double sum = bias[offset + o];
            for (int i = 0; i < inputs; i++) sum += weight[((offset + o) * inputs) + i] * x[i];
            y[o] = sum;
        }
        return y;
    }

    private static double[] LayerNorm(double[] x, double[] gamma, double[] beta)
    {
        double mean = x.Average();
        double variance = x.Select(v => (v - mean) * (v - mean)).Average();
        double inv = 1.0 / Math.Sqrt(variance + 1e-5);
        return x.Select((v, i) => ((v - mean) * inv * gamma[i]) + beta[i]).ToArray();
    }

    // ------------------------------------------------------------------ OpenCLIP text encoder

    private static OpenClipTextConditioner<double> TinyText() => new OpenClipTextConditioner<double>(new OpenClipTextEncoderOptions
    {
        ContextLength = 6,
        Width = 8,
        Layers = 2,
        Heads = 2,
        SkippedFinalLayers = 1
    });

    [Fact]
    public void OpenClipViTH14Text_HasTheHandDerivedCountAndAllocatesNothingOnConstruction()
    {
#if NET
        long before = GC.GetAllocatedBytesForCurrentThread();
#endif
        var encoder = new OpenClipTextConditioner<float>();
#if NET
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        // ViT-H's text tower is 353M weights (1.4 GB in float); construction must not allocate them.
        Assert.True(allocated < 16L * 1024 * 1024, $"Construction allocated {allocated} bytes.");
#endif
        Assert.False(encoder.IsMaterialized);

        // open_clip ViT-H-14 text tower (width 1024, 24 resblocks, 16 heads, context 77, vocab 49408): token and position
        // embeddings, per block ln_1, in_proj [3W, W] + bias, out_proj, ln_2, c_fc [4W, W], c_proj [W, 4W], then
        // ln_final. open_clip's 354,032,640 adds text_projection [1024, 1024], which LDM's embedder does not use.
        const long w = 1024;
        long block = (2 * w) + ((3 * w * w) + (3 * w)) + ((w * w) + w) + (2 * w) + ((w * 4 * w) + (4 * w)) + ((4 * w * w) + w);
        long expected = (49408 * w) + (77 * w) + (24 * block) + (2 * w);
        Assert.Equal(352_984_064L, expected);
        Assert.Equal(354_032_640L, expected + (w * w));
        Assert.Equal(expected, encoder.ParameterCount);
        Assert.False(encoder.IsMaterialized);
        Assert.Equal(1024, encoder.EmbeddingDimension);
        Assert.Equal(77, encoder.MaxSequenceLength);
    }

    [Fact]
    public void OpenClipText_TokenizesLikeOpenClipAndNeedsATokenizerOnlyForText()
    {
        var encoder = TinyText();
        Assert.Equal(new double[] { 49406, 49407, 0, 0, 0, 0 }, encoder.Tokenize(string.Empty).ToArray());
        Assert.Equal(new double[] { 49406, 49407, 0, 0, 0, 0 }, encoder.Tokenize("   ").ToArray());
        Assert.Throws<InvalidOperationException>(() => encoder.Tokenize("a photo of a cat"));
        Assert.False(encoder.IsMaterialized);
    }

    [Fact]
    public void OpenClipText_FirstUseMaterializesAndMatchesTheEncodeWithTransformerReference()
    {
        var encoder = TinyText();
        Assert.False(encoder.IsMaterialized);
        var weights = new Weights();
        var names = encoder.LoadTorchWeights(weights.Read, "e.");
        Assert.Equal(2 + (2 * 12) + 2, names.Count);
        Assert.Contains("e.token_embedding.weight", names);
        Assert.Contains("e.positional_embedding", names);
        Assert.Contains("e.transformer.resblocks.1.attn.in_proj_weight", names);
        Assert.Contains("e.transformer.resblocks.0.mlp.c_proj.bias", names);
        Assert.Contains("e.ln_final.weight", names);
        Assert.True(encoder.IsMaterialized);

        var tokens = encoder.Tokenize(string.Empty);
        var output = encoder.EncodeText(tokens).ToArray();

        // encode_with_transformer: token + position embeddings, the resblocks up to the penultimate (block 0 of 2),
        // ln_final. Each resblock: x + out_proj(causal MHA(ln_1 x)), then x + c_proj(GELU(c_fc(ln_2 x))).
        const int n = 6, width = 8, heads = 2, hd = 4;
        var ids = tokens.ToArray().Select(v => (int)v).ToArray();
        var table = weights["e.token_embedding.weight"];
        var positions = weights["e.positional_embedding"];
        var x = new double[n][];
        for (int t = 0; t < n; t++)
            x[t] = Enumerable.Range(0, width).Select(c => table[(ids[t] * width) + c] + positions[(t * width) + c]).ToArray();

        string b = "e.transformer.resblocks.0.";
        var normed = x.Select(v => LayerNorm(v, weights[b + "ln_1.weight"], weights[b + "ln_1.bias"])).ToArray();
        var inW = weights[b + "attn.in_proj_weight"];
        var inB = weights[b + "attn.in_proj_bias"];
        var q = normed.Select(v => Linear(inW, inB, v, width, 0)).ToArray();
        var k = normed.Select(v => Linear(inW, inB, v, width, width)).ToArray();
        var vv = normed.Select(v => Linear(inW, inB, v, width, 2 * width)).ToArray();
        for (int t = 0; t < n; t++)
        {
            var context = new double[width];
            for (int h = 0; h < heads; h++)
            {
                var scores = new double[t + 1];
                for (int s = 0; s <= t; s++)
                {
                    double dot = 0.0;
                    for (int e = h * hd; e < (h + 1) * hd; e++) dot += q[t][e] * k[s][e];
                    scores[s] = dot / Math.Sqrt(hd);
                }
                double max = scores.Max();
                var p = scores.Select(sc => Math.Exp(sc - max)).ToArray();
                double total = p.Sum();
                for (int s = 0; s <= t; s++)
                    for (int e = h * hd; e < (h + 1) * hd; e++) context[e] += p[s] / total * vv[s][e];
            }
            var attended = Linear(weights[b + "attn.out_proj.weight"], weights[b + "attn.out_proj.bias"], context, width);
            x[t] = x[t].Zip(attended, (a, c) => a + c).ToArray();
        }
        for (int t = 0; t < n; t++)
        {
            var m = LayerNorm(x[t], weights[b + "ln_2.weight"], weights[b + "ln_2.bias"]);
            var f = Linear(weights[b + "mlp.c_fc.weight"], weights[b + "mlp.c_fc.bias"], m, 4 * width)
                .Select(v => 0.5 * v * (1.0 + Erf(v / Math.Sqrt(2.0)))).ToArray();
            var projected = Linear(weights[b + "mlp.c_proj.weight"], weights[b + "mlp.c_proj.bias"], f, width);
            x[t] = x[t].Zip(projected, (a, c) => a + c).ToArray();
        }
        var expected = x.SelectMany(v => LayerNorm(v, weights["e.ln_final.weight"], weights["e.ln_final.bias"])).ToArray();

        Assert.Equal(expected.Length, output.Length);
        for (int i = 0; i < expected.Length; i++) Assert.True(Math.Abs(expected[i] - output[i]) < 1e-8, $"index {i}");

        // The unconditional embedding is the empty prompt's (MVDream's get_learned_conditioning([""])).
        Assert.Equal(output, encoder.GetUnconditionalEmbedding(1).ToArray());
    }

    [Fact]
    public void OpenClipText_FirstEncodeMaterializesTheWeights()
    {
        var encoder = TinyText();
        Assert.False(encoder.IsMaterialized);
        var embedding = encoder.GetUnconditionalEmbedding(2);
        Assert.Equal(new[] { 2, 6, 8 }, embedding.Shape.ToArray());
        Assert.True(encoder.IsMaterialized);
    }

    // ------------------------------------------------------------------ CLIP image encoder

    private static ClipVisionEncoder<double> TinyClip(int seed) => new ClipVisionEncoder<double>(new ClipVisionEncoderOptions
    {
        ImageSize = 28,
        PatchSize = 14,
        Width = 8,
        Layers = 2,
        Heads = 2,
        MlpWidth = 16,
        ProjectionDim = 4
    }, seed);

    private static Tensor<double> Image(int seed)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        var values = new double[3 * 28 * 28];
        for (int i = 0; i < values.Length; i++) values[i] = (2.0 * random.NextDouble()) - 1.0;
        return new Tensor<double>(values, new[] { 1, 3, 28, 28 });
    }

    [Fact]
    public void ClipViTH14Image_HasTheHandDerivedCountAndAllocatesNothingOnConstruction()
    {
#if NET
        long before = GC.GetAllocatedBytesForCurrentThread();
#endif
        var encoder = new ClipVisionEncoder<float>();
#if NET
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.True(allocated < 16L * 1024 * 1024, $"Construction allocated {allocated} bytes.");
#endif
        Assert.False(encoder.IsMaterialized);

        // open_clip ViT-H-14 visual (width 1280, 32 layers, MLP 5120, 224 px / 14 px patches, embed 1024): conv1
        // (bias-free), class embedding, 257 positions, ln_pre; per layer ln_1, q/k/v/out, ln_2, c_fc, c_proj; ln_post,
        // proj. 632,076,800 is open_clip's count for the ViT-H-14 visual tower.
        const long w = 1280, mlp = 5120;
        long layer = (2 * w) + (4 * ((w * w) + w)) + (2 * w) + ((w * mlp) + mlp) + ((mlp * w) + w);
        long expected = (3 * 14 * 14 * w) + w + (257 * w) + (2 * w) + (32 * layer) + (2 * w) + (w * 1024);
        Assert.Equal(632_076_800L, expected);
        Assert.Equal(expected, encoder.ParameterCount);
        Assert.False(encoder.IsMaterialized);
    }

    [Fact]
    public void ClipImage_FirstUseMaterializesAndTheCountMatchesTheBuiltLayers()
    {
        var encoder = TinyClip(3);
        long declared = encoder.ParameterCount;
        Assert.False(encoder.IsMaterialized);
        var embeds = encoder.EncodeImage(Image(1));
        Assert.Equal(new[] { 1, 4 }, embeds.Shape.ToArray());
        Assert.True(encoder.IsMaterialized);
        Assert.Equal(declared, encoder.GetParameters().Length);
        Assert.Equal(declared, encoder.ParameterCount);
    }

    [Fact]
    public void ClipImage_SameSeed_GivesTheSameWeightsWhenEverTheyMaterialize()
    {
        var first = TinyClip(9);
        var second = TinyClip(9);
        _ = second.ParameterCount;
        Assert.Equal(first.EncodeImage(Image(2)).ToArray(), second.EncodeImage(Image(2)).ToArray());
    }

    [Fact]
    public void ClipImage_Clone_IsADeepCopy()
    {
        var source = TinyClip(5);
        var image = Image(3);
        var before = source.EncodeImage(image).ToArray();

        var copy = source.Clone();
        Assert.NotSame(source, copy);
        Assert.Equal(source.GetParameters().ToArray(), copy.GetParameters().ToArray());
        Assert.Equal(before, copy.EncodeImage(image).ToArray());

        // Writing the copy's weights must not reach the source.
        var changed = copy.GetParameters().ToArray().Select(v => v + 0.25).ToArray();
        copy.SetParameters(new Vector<double>(changed));
        Assert.Equal(changed, copy.GetParameters().ToArray());
        Assert.Equal(before, source.EncodeImage(image).ToArray());
        Assert.NotEqual(before, copy.EncodeImage(image).ToArray());

        // An encoder not yet materialized copies without materializing, and both draw the same weights.
        var lazy = TinyClip(6);
        var lazyCopy = lazy.DeepCopy();
        Assert.False(lazy.IsMaterialized);
        Assert.False(lazyCopy.IsMaterialized);
        Assert.Equal(lazy.GetParameters().ToArray(), lazyCopy.GetParameters().ToArray());
    }

    [Fact]
    public void MVDreamClone_DoesNotShareTheImageEncoder()
    {
        var encoder = TinyClip(7);
        var model = new MVDreamModel<double>(
            multiViewUNet: new MultiViewUNet<double>(
                inputChannels: 4, outputChannels: 4, baseChannels: 8, numViews: 2, contextDim: 16, seed: 1),
            imageVAE: new StandardVAE<double>(
                inputChannels: 3, latentChannels: 4, baseChannels: 8,
                channelMultipliers: new[] { 1, 2 }, numResBlocksPerLevel: 1, seed: 1),
            seed: 1, imageEncoder: encoder);

        var clone = Assert.IsType<MVDreamModel<double>>(model.Clone());
        Assert.NotNull(clone.ImageEncoder);
        Assert.NotSame(model.ImageEncoder, clone.ImageEncoder);
        var cloneEncoder = clone.ImageEncoder ?? throw new InvalidOperationException("The clone lost its encoder.");
        Assert.Equal(encoder.GetParameters().ToArray(), cloneEncoder.GetParameters().ToArray());
    }

    // ------------------------------------------------------------------ the models' defaults

    [Fact]
    public void DefaultMVDream_HoldsTheLazyOpenClipTextEncoder()
    {
        var model = new MVDreamModel<float>();
        var text = Assert.IsType<OpenClipTextConditioner<float>>(model.Conditioner);
        Assert.False(text.IsMaterialized);
        Assert.Equal(MVDreamModel<float>.MVDREAM_CONTEXT_DIM, text.EmbeddingDimension);
        Assert.Null(model.ImageEncoder);
    }

    [Fact]
    public void DefaultLGM_ImageDreamModel_HoldsTheLazyOpenClipTextAndImageEncoders()
    {
        var model = new LGMModel<float>();
        var imageDream = model.ImageMultiViewModel;
        Assert.True(imageDream.MultiViewUNet.HasImagePrompt);
        var text = Assert.IsType<OpenClipTextConditioner<float>>(imageDream.Conditioner);
        Assert.False(text.IsMaterialized);
        var image = imageDream.ImageEncoder ?? throw new InvalidOperationException("ImageDream has no image encoder.");
        Assert.False(image.IsMaterialized);
        Assert.Equal(1280, image.HiddenDimension);
        Assert.Equal(1024, image.EmbeddingDimension);
        Assert.IsType<OpenClipTextConditioner<float>>(model.MultiViewModel.Conditioner);
    }
}

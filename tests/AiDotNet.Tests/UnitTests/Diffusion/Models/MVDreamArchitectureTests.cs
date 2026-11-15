using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.ThreeD;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Interfaces;
using AiDotNet.ModelLoading;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion.Models;

/// <summary>
/// MVDream's multi-view U-Net (Shi et al., 2023, arXiv 2308.16512; official bytedance/MVDream) and ImageDream's image
/// prompt (Wang and Shi, 2023, arXiv 2312.02201; official bytedance/ImageDream), checked against hand references
/// written from the official code: SpatialTransformer3D's 3D self-attention, camera_embed added to the time embedding,
/// the perceiver Resampler, the decoupled image cross-attention, and get_camera.
/// </summary>
public class MVDreamArchitectureTests
{
    // ------------------------------------------------------------------ named weights

    /// <summary>Deterministic weights, drawn per official name on first read and remembered for the reference.</summary>
    private sealed class NamedWeights
    {
        private readonly Dictionary<string, double[]> _values = new();
        private readonly Random _random;
        private readonly double _scale;

        public NamedWeights(int seed, double scale = 0.4)
        {
            _random = RandomHelper.CreateSeededRandom(seed);
            _scale = scale;
        }

        public double[] Read(string name, int[] shape)
        {
            if (_values.TryGetValue(name, out var existing)) return existing;
            int count = 1;
            foreach (var d in shape) count *= d;
            bool normScale = name.EndsWith(".weight", StringComparison.Ordinal) && name.Contains("norm");
            var values = new double[count];
            for (int i = 0; i < count; i++)
            {
                double u = (2.0 * _random.NextDouble()) - 1.0;
                values[i] = normScale ? 1.0 + (0.2 * u) : _scale * u;
            }
            _values[name] = values;
            return values;
        }

        public double[] this[string name] => _values[name];

        public TorchTensorSource Source() => TorchTensorSource.From((string name, int[] shape) => Read(name, shape));
    }

    // ------------------------------------------------------------------ reference helpers (torch layouts)

    // y = W x + b with W [outputs, inputs] row-major.
    private static double[] Linear(double[] weight, double[]? bias, double[] x, int outputs)
    {
        int inputs = x.Length;
        var y = new double[outputs];
        for (int o = 0; o < outputs; o++)
        {
            double sum = bias is null ? 0.0 : bias[o];
            for (int i = 0; i < inputs; i++) sum += weight[(o * inputs) + i] * x[i];
            y[o] = sum;
        }
        return y;
    }

    private static double[] LayerNorm(double[] x, double[] gamma, double[] beta, double eps = 1e-5)
    {
        double mean = x.Average();
        double variance = x.Select(v => (v - mean) * (v - mean)).Average();
        double inv = 1.0 / Math.Sqrt(variance + eps);
        return x.Select((v, i) => ((v - mean) * inv * gamma[i]) + beta[i]).ToArray();
    }

    // softmax(q k^T / sqrt(d)) v per head, before the output projection.
    private static double[][] Attend(double[][] queries, double[][] context, double[] toQ, double[] toK, double[] toV,
        int dim, int heads)
    {
        int d = dim / heads;
        var keys = context.Select(c => Linear(toK, null, c, dim)).ToArray();
        var values = context.Select(c => Linear(toV, null, c, dim)).ToArray();
        var result = new double[queries.Length][];
        for (int n = 0; n < queries.Length; n++)
        {
            var q = Linear(toQ, null, queries[n], dim);
            var output = new double[dim];
            for (int h = 0; h < heads; h++)
            {
                var scores = new double[keys.Length];
                for (int j = 0; j < keys.Length; j++)
                {
                    double s = 0.0;
                    for (int e = h * d; e < (h + 1) * d; e++) s += q[e] * keys[j][e];
                    scores[j] = s / Math.Sqrt(d);
                }
                double max = scores.Max();
                var weights = scores.Select(s => Math.Exp(s - max)).ToArray();
                double total = weights.Sum();
                for (int j = 0; j < keys.Length; j++)
                    for (int e = h * d; e < (h + 1) * d; e++)
                        output[e] += weights[j] / total * values[j][e];
            }
            result[n] = output;
        }
        return result;
    }

    // erf by its Maclaurin series (|z| < 2.5) or its continued fraction (|z| >= 2.5).
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

    private static double Gelu(double x) => 0.5 * x * (1.0 + Erf(x / Math.Sqrt(2.0)));

    /// <summary>
    /// The official SpatialTransformer3D (use_linear, depth 1, context_dim None) over x [views, C, H, W], written from
    /// mvdream/ldm/modules/attention.py: GroupNorm(eps 1e-6), proj_in, "(b f) l c -> b (f l) c" then
    /// attn1(norm1) + x, attn2(norm2) + x per view, GEGLU ff(norm3) + x, proj_out, + x_in.
    /// </summary>
    private static double[] TransformerReference(double[] input, int views, int channels, int height, int width,
        int heads, int frames, NamedWeights w, string p)
    {
        int hw = height * width;
        int groups = DiffusionResBlock<double>.ComputeNumGroups(channels, 32);
        int perGroup = channels / groups;
        var gamma = w[p + "norm.weight"];
        var beta = w[p + "norm.bias"];
        var normed = new double[input.Length];
        for (int v = 0; v < views; v++)
        {
            for (int g = 0; g < groups; g++)
            {
                var indices = new List<int>();
                for (int c = g * perGroup; c < (g + 1) * perGroup; c++)
                    for (int pos = 0; pos < hw; pos++) indices.Add((((v * channels) + c) * hw) + pos);
                double mean = indices.Average(i => input[i]);
                double variance = indices.Average(i => (input[i] - mean) * (input[i] - mean));
                foreach (int i in indices)
                {
                    int c = (i / hw) % channels;
                    normed[i] = ((input[i] - mean) / Math.Sqrt(variance + 1e-6) * gamma[c]) + beta[c];
                }
            }
        }

        var tokens = new double[views * hw][];
        for (int v = 0; v < views; v++)
            for (int pos = 0; pos < hw; pos++)
            {
                var t = new double[channels];
                for (int c = 0; c < channels; c++) t[c] = normed[(((v * channels) + c) * hw) + pos];
                tokens[(v * hw) + pos] = Linear(w[p + "proj_in.weight"], w[p + "proj_in.bias"], t, channels);
            }

        string b = p + "transformer_blocks.0.";
        // attn1: every token of an object (frames consecutive views) attends to all of that object's tokens.
        int span = frames * hw;
        var afterSelf = new double[tokens.Length][];
        for (int start = 0; start < tokens.Length; start += span)
        {
            var group = tokens.Skip(start).Take(span).Select(t => LayerNorm(t, w[b + "norm1.weight"], w[b + "norm1.bias"])).ToArray();
            var attended = Attend(group, group, w[b + "attn1.to_q.weight"], w[b + "attn1.to_k.weight"],
                w[b + "attn1.to_v.weight"], channels, heads);
            for (int n = 0; n < span; n++)
            {
                var o = Linear(w[b + "attn1.to_out.0.weight"], w[b + "attn1.to_out.0.bias"], attended[n], channels);
                afterSelf[start + n] = tokens[start + n].Zip(o, (x, y) => x + y).ToArray();
            }
        }
        tokens = afterSelf;

        // attn2: per view (self-attention, no context).
        var afterCross = new double[tokens.Length][];
        for (int v = 0; v < views; v++)
        {
            var view = tokens.Skip(v * hw).Take(hw).Select(t => LayerNorm(t, w[b + "norm2.weight"], w[b + "norm2.bias"])).ToArray();
            var attended = Attend(view, view, w[b + "attn2.to_q.weight"], w[b + "attn2.to_k.weight"],
                w[b + "attn2.to_v.weight"], channels, heads);
            for (int n = 0; n < hw; n++)
            {
                var o = Linear(w[b + "attn2.to_out.0.weight"], w[b + "attn2.to_out.0.bias"], attended[n], channels);
                afterCross[(v * hw) + n] = tokens[(v * hw) + n].Zip(o, (x, y) => x + y).ToArray();
            }
        }
        tokens = afterCross;

        int inner = 4 * channels;
        var output = new double[input.Length];
        for (int i = 0; i < tokens.Length; i++)
        {
            var n3 = LayerNorm(tokens[i], w[b + "norm3.weight"], w[b + "norm3.bias"]);
            var projected = Linear(w[b + "ff.net.0.proj.weight"], w[b + "ff.net.0.proj.bias"], n3, 2 * inner);
            var gated = new double[inner];
            for (int k = 0; k < inner; k++) gated[k] = projected[k] * Gelu(projected[inner + k]);
            var ff = Linear(w[b + "ff.net.2.weight"], w[b + "ff.net.2.bias"], gated, channels);
            var h = tokens[i].Zip(ff, (x, y) => x + y).ToArray();
            var o = Linear(w[p + "proj_out.weight"], w[p + "proj_out.bias"], h, channels);
            int v = i / hw, pos = i % hw;
            for (int c = 0; c < channels; c++)
            {
                int index = (((v * channels) + c) * hw) + pos;
                output[index] = input[index] + o[c];
            }
        }
        return output;
    }

    private static double[] RandomValues(int count, int seed)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        var values = new double[count];
        for (int i = 0; i < count; i++) values[i] = (2.0 * random.NextDouble()) - 1.0;
        return values;
    }

    private static double MaxAbsDifference(double[] a, double[] b)
    {
        Assert.Equal(a.Length, b.Length);
        double worst = 0.0;
        for (int i = 0; i < a.Length; i++) worst = Math.Max(worst, Math.Abs(a[i] - b[i]));
        return worst;
    }

    private static MultiViewTransformerLayer<double> LoadedTransformer(int channels, MultiViewUNetOptions options,
        NamedWeights weights)
    {
        var layer = new MultiViewTransformerLayer<double>(channels, 0, options);
        layer.LoadTorchWeights(weights.Source(), "t.");
        return layer;
    }

    // ------------------------------------------------------------------ 3D self-attention

    [Fact]
    public void ThreeDSelfAttention_EqualsAttentionOverTheConcatenatedViewTokens()
    {
        const int views = 4, channels = 4, height = 1, width = 2;
        var options = new MultiViewUNetOptions { HeadChannels = 2 };
        var weights = new NamedWeights(11);
        var layer = LoadedTransformer(channels, options, weights);
        Assert.Equal(2, layer.HeadCount);

        var input = RandomValues(views * channels * height * width, 3);
        var output = layer.Forward(new Tensor<double>(input, new[] { views, channels, height, width }), null).ToArray();

        var reference = TransformerReference(input, views, channels, height, width, heads: 2, frames: views, weights, "t.");
        Assert.True(MaxAbsDifference(reference, output) < 1e-9,
            $"SpatialTransformer3D differs from the hand reference by {MaxAbsDifference(reference, output)}.");

        // The reference with one view per object is a different function: the 3D attention is what differs.
        var perView = TransformerReference(input, views, channels, height, width, heads: 2, frames: 1, weights, "t.");
        Assert.True(MaxAbsDifference(reference, perView) > 1e-6);
    }

    [Fact]
    public void ThreeDSelfAttention_LetsView3ReachATokenOfView0()
    {
        const int views = 4, channels = 4, height = 1, width = 2;
        int perView = channels * height * width;
        var weights = new NamedWeights(12);
        var joint = LoadedTransformer(channels, new MultiViewUNetOptions { HeadChannels = 2 }, weights);
        var separate = LoadedTransformer(channels, new MultiViewUNetOptions { HeadChannels = 2, FramesPerObject = 1 }, weights);

        var input = RandomValues(views * perView, 5);
        var changed = input.ToArray();
        for (int i = 3 * perView; i < 4 * perView; i++) changed[i] += 0.5;
        var shape = new[] { views, channels, height, width };

        double[] View0(MultiViewTransformerLayer<double> layer, double[] x)
            => layer.Forward(new Tensor<double>(x, shape), null).ToArray().Take(perView).ToArray();

        Assert.True(MaxAbsDifference(View0(joint, input), View0(joint, changed)) > 1e-6,
            "view 0's tokens must attend to view 3's");
        Assert.True(MaxAbsDifference(View0(separate, input), View0(separate, changed)) < 1e-12,
            "with one view per object, view 3 must not reach view 0");
    }

    // ------------------------------------------------------------------ camera embedding

    [Fact]
    public void CameraEmbedding_HasTheHandDerivedSizeAndIsAddedToTheTimeEmbedding()
    {
        const int baseChannels = 8;
        var unet = new MultiViewUNet<double>(4, 4, baseChannels, numViews: 2, contextDim: 0, seed: 3);

        // time_embed_dim = model_channels * 4; camera_embed = Linear(16, E), SiLU, Linear(E, E), both with bias.
        int e = unet.BaseUNet.TimeEmbeddingDim;
        Assert.Equal(4 * baseChannels, e);
        Assert.Equal((16L * e) + e + ((long)e * e) + e, unet.CameraEmbedding.ParameterCount);

        var cameras = CameraEmbedding<double>.GetCameras(2, elevation: 15.0, azimuthStart: 90.0);
        var cameraTensor = CameraEmbedding<double>.CameraTensor(cameras);
        var p = unet.CameraEmbedding.GetParameters().ToArray();
        var hand = new double[2 * e];
        for (int n = 0; n < 2; n++)
        {
            var hidden = new double[e];
            for (int i = 0; i < e; i++)
            {
                double s = p[(e * 16) + i];
                for (int k = 0; k < 16; k++) s += p[(i * 16) + k] * cameras[n][k];
                hidden[i] = s / (1.0 + Math.Exp(-s));
            }
            int w2 = (e * 16) + e;
            for (int i = 0; i < e; i++)
            {
                double s = p[w2 + (e * e) + i];
                for (int k = 0; k < e; k++) s += p[w2 + (i * e) + k] * hidden[k];
                hand[(n * e) + i] = s;
            }
        }
        Assert.True(MaxAbsDifference(hand, unet.CameraEmbedding.EmbedCameras(cameraTensor).ToArray()) < 1e-10);

        var x = new Tensor<double>(RandomValues(2 * 4 * 16 * 16, 7), new[] { 2, 4, 16, 16 });
        var withCamera = unet.PredictNoiseMultiView(x, 500, new Tensor<double>?[] { null, null }, cameraTensor).ToArray();
        var handAdded = unet.BaseUNet.PredictNoiseWithAddedTimeEmbedding(x, 500, null,
            new Tensor<double>(hand, new[] { 2, e })).ToArray();
        Assert.True(MaxAbsDifference(withCamera, handAdded) < 1e-10,
            "the camera must enter as camera_embed(camera) added to the time embedding");

        var zeroAdded = unet.BaseUNet.PredictNoiseWithAddedTimeEmbedding(x, 500, null, new Tensor<double>(new[] { 2, e })).ToArray();
        var plain = unet.BaseUNet.PredictNoise(x, 500, null).ToArray();
        Assert.True(MaxAbsDifference(zeroAdded, plain) < 1e-12, "a zero added embedding is no camera");
        Assert.True(MaxAbsDifference(withCamera, plain) > 1e-9, "the camera must change the prediction");
    }

    [Fact]
    public void GetCameras_MatchesTheOfficialBlenderCameraToWorldMatrices()
    {
        // create_camera_to_world_matrix then flip_yz (mvdream/camera_utils.py), derived by hand: azimuth 0 puts the
        // camera at +z looking down -z; azimuth 90 at +x. ImageDream's extra view is the all-zero camera.
        var cameras = CameraEmbedding<double>.GetCameras(2, elevation: 0.0, azimuthStart: 0.0, azimuthSpan: 180.0, extraView: true);
        Assert.Equal(3, cameras.Length);
        var expected0 = new double[] { 1, 0, 0, 0, 0, 0, -1, -1, 0, 1, 0, 0, 0, 0, 0, 1 };
        var expected90 = new double[] { 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1 };
        Assert.True(MaxAbsDifference(expected0, cameras[0]) < 1e-12);
        Assert.True(MaxAbsDifference(expected90, cameras[1]) < 1e-12);
        Assert.All(cameras[2], value => Assert.Equal(0.0, value));
    }

    // ------------------------------------------------------------------ ImageDream resampler

    [Fact]
    public void Resampler_HasTheHandDerivedShapesParameterCountAndOfficialNames()
    {
        // dim 8, depth 2, dim_head 4, heads 2 (inner 8), 3 queries, embedding 6, output 5, ff_mult 2.
        var resampler = new ImageDreamResampler<double>(dim: 8, depth: 2, headDim: 4, heads: 2, queries: 3,
            embeddingDim: 6, outputDim: 5, feedForwardMultiplier: 2);
        long latents = 3 * 8;
        long projIn = (6 * 8) + 8;
        long attention = (2 * 8) + (2 * 8) + (8 * 8) + (8 * 16) + (8 * 8);  // norm1, norm2, to_q, to_kv, to_out
        long feedForward = (2 * 8) + (8 * 16) + (16 * 8);                    // LayerNorm, Linear, Linear (no bias)
        long projOut = (8 * 5) + 5;
        long normOut = 2 * 5;
        Assert.Equal(latents + projIn + (2 * (attention + feedForward)) + projOut + normOut, resampler.ParameterCount);

        var output = resampler.Forward(new Tensor<double>(RandomValues(2 * 7 * 6, 9), new[] { 2, 7, 6 }));
        Assert.Equal(new[] { 2, 3, 5 }, output.Shape.ToArray());

        var names = resampler.NamedTensors("image_embed").Select(n => n.Name).ToList();
        foreach (var name in new[]
        {
            "image_embed.latents", "image_embed.proj_in.weight", "image_embed.proj_in.bias",
            "image_embed.layers.0.0.norm1.weight", "image_embed.layers.0.0.norm2.bias", "image_embed.layers.0.0.to_q.weight",
            "image_embed.layers.1.0.to_kv.weight", "image_embed.layers.1.0.to_out.weight", "image_embed.layers.1.1.0.weight",
            "image_embed.layers.1.1.1.weight", "image_embed.layers.1.1.3.weight", "image_embed.proj_out.weight",
            "image_embed.proj_out.bias", "image_embed.norm_out.weight", "image_embed.norm_out.bias"
        })
            Assert.Contains(name, names);
        Assert.DoesNotContain("image_embed.layers.0.0.to_q.bias", names);
        Assert.DoesNotContain("image_embed.layers.0.1.1.bias", names);
    }

    // ------------------------------------------------------------------ decoupled image cross-attention

    [Fact]
    public void DecoupledAttention_IsTextAttentionPlusScaledImageAttention()
    {
        const int dim = 4, contextDim = 6, heads = 2, textTokens = 3, imageTokens = 2;
        const double scale = 0.7;
        var weights = new NamedWeights(21);
        var layer = new ImagePromptCrossAttentionLayer<double>(dim, contextDim, heads, imageTokens, scale);
        layer.LoadTorchWeights(weights.Source(), "a.");

        var query = RandomValues(3 * dim, 1);
        var context = RandomValues((textTokens + imageTokens) * contextDim, 2);
        var output = layer.Forward(new Tensor<double>(query, new[] { 1, 3, dim }),
            new Tensor<double>(context, new[] { 1, textTokens + imageTokens, contextDim })).ToArray();

        double[][] Rows(double[] flat, int width, int skip, int take)
            => Enumerable.Range(skip, take).Select(r => flat.Skip(r * width).Take(width).ToArray()).ToArray();
        var queries = Rows(query, dim, 0, 3);
        var text = Attend(queries, Rows(context, contextDim, 0, textTokens),
            weights["a.to_q.weight"], weights["a.to_k.weight"], weights["a.to_v.weight"], dim, heads);
        var image = Attend(queries, Rows(context, contextDim, textTokens, imageTokens),
            weights["a.to_q.weight"], weights["a.to_k_ip.weight"], weights["a.to_v_ip.weight"], dim, heads);
        var expected = new List<double>();
        for (int n = 0; n < 3; n++)
        {
            var combined = text[n].Zip(image[n], (t, i) => t + (scale * i)).ToArray();
            expected.AddRange(Linear(weights["a.to_out.0.weight"], weights["a.to_out.0.bias"], combined, dim));
        }
        Assert.True(MaxAbsDifference(expected.ToArray(), output) < 1e-10);
    }

    // ------------------------------------------------------------------ option off

    [Fact]
    public void MultiViewOptionOff_LeavesTheSharedUNetBitIdentical()
    {
        UNetNoisePredictor<double> Build(bool passNull) => passNull
            ? new UNetNoisePredictor<double>(inputChannels: 4, outputChannels: 4, baseChannels: 8,
                channelMultipliers: new[] { 1, 2 }, numResBlocks: 1, attentionResolutions: new[] { 0, 1 }, contextDim: 6,
                numHeads: 2, inputHeight: 8, seed: 5, multiView: null)
            : new UNetNoisePredictor<double>(inputChannels: 4, outputChannels: 4, baseChannels: 8,
                channelMultipliers: new[] { 1, 2 }, numResBlocks: 1, attentionResolutions: new[] { 0, 1 }, contextDim: 6,
                numHeads: 2, inputHeight: 8, seed: 5);

        var standard = Build(passNull: false);
        var optionOff = Build(passNull: true);
        Assert.Null(optionOff.MultiViewOptions);
        Assert.Equal(standard.ParameterCount, optionOff.ParameterCount);
        optionOff.SetParameterChunks(standard.GetParameterChunks());

        var x = new Tensor<double>(RandomValues(2 * 4 * 8 * 8, 4), new[] { 2, 4, 8, 8 });
        var context = new Tensor<double>(RandomValues(2 * 3 * 6, 6), new[] { 2, 3, 6 });
        Assert.Equal(standard.PredictNoise(x, 300, context).ToArray(), optionOff.PredictNoise(x, 300, context).ToArray());
        Assert.Equal(standard.PredictNoise(x, 300, null).ToArray(), optionOff.PredictNoise(x, 300, null).ToArray());

        var multiView = new UNetNoisePredictor<double>(inputChannels: 4, outputChannels: 4, baseChannels: 8,
            channelMultipliers: new[] { 1, 2 }, numResBlocks: 1, attentionResolutions: new[] { 0, 1 }, contextDim: 6,
            numHeads: 2, inputHeight: 8, seed: 5, multiView: MultiViewUNetOptions.MVDream());
        Assert.NotNull(multiView.MultiViewOptions);
        Assert.NotEqual(standard.ParameterCount, multiView.ParameterCount);
    }

    // ------------------------------------------------------------------ ImageDream image path

    private sealed class FixedConditioner : IConditioningModule<double>
    {
        private readonly int _width;

        public FixedConditioner(int width, ConditioningType type)
        {
            _width = width;
            ConditioningType = type;
        }

        /// <summary>The features an image encodes to (image conditioner role).</summary>
        public Tensor<double>? Features { get; set; }

        public int EmbeddingDimension => _width;
        public ConditioningType ConditioningType { get; }
        public bool ProducesPooledOutput => false;
        public int MaxSequenceLength => 2;

        public Tensor<double> Encode(Tensor<double> input)
            => Features ?? throw new InvalidOperationException("No features set.");

        public Tensor<double> EncodeText(Tensor<double> tokenIds, Tensor<double>? attentionMask = null)
            => new Tensor<double>(RandomValues(2 * _width, 31), new[] { 1, 2, _width });

        public Tensor<double> GetPooledEmbedding(Tensor<double> sequenceEmbeddings)
            => new Tensor<double>(new[] { 1, _width });

        public Tensor<double> GetUnconditionalEmbedding(int batchSize)
            => new Tensor<double>(RandomValues(batchSize * _width, 32), new[] { batchSize, 1, _width });

        public Tensor<double> Tokenize(string text) => new Tensor<double>(new double[1], new[] { 1, 1 });

        public Tensor<double> TokenizeBatch(string[] texts) => new Tensor<double>(new double[texts.Length], new[] { texts.Length, 1 });
    }

    private const int ContextDim = 16;
    private const int FeatureDim = 8;

    private static MVDreamModel<double> TinyImageDream(int seed, ClipVisionEncoder<double>? encoder,
        IConditioningModule<double>? imageConditioner)
    {
        var options = MultiViewUNetOptions.ImageDream();
        options.ImageFeatureDim = FeatureDim;
        options.ResamplerDepth = 1;
        options.ResamplerHeads = 2;
        options.ResamplerHeadDim = 4;
        var unet = new MultiViewUNet<double>(4, 4, baseChannels: 8, numViews: 4, contextDim: ContextDim, seed: seed,
            options: options);
        // Non-zero transformer, camera and resampler weights under the official names, so the decoupled attention's
        // output reaches the prediction (proj_out starts at zero).
        var weights = new NamedWeights(seed, scale: 0.2);
        var names = unet.LoadTorchWeights((name, shape) => weights.Read(name, shape));
        Assert.Contains("model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k_ip.weight", names);
        Assert.Contains("model.diffusion_model.middle_block.1.proj_out.weight", names);
        Assert.Contains("model.diffusion_model.output_blocks.11.1.norm.weight", names);
        Assert.Contains("model.diffusion_model.camera_embed.2.bias", names);
        Assert.Contains("model.diffusion_model.image_embed.latents", names);
        // The whole LDM U-Net loads (openaimodel.py names): residual blocks, the op downsamplers, the conv
        // upsamplers (.1 after a level without attention, .2 after one with it), time_embed and out.
        Assert.Contains("model.diffusion_model.input_blocks.3.0.op.weight", names);
        Assert.Contains("model.diffusion_model.input_blocks.4.0.skip_connection.weight", names);
        Assert.Contains("model.diffusion_model.output_blocks.2.1.conv.weight", names);
        Assert.Contains("model.diffusion_model.output_blocks.5.2.conv.bias", names);
        Assert.Contains("model.diffusion_model.time_embed.2.weight", names);
        Assert.Contains("model.diffusion_model.out.0.weight", names);
        Assert.Equal(names.Count, names.Distinct().Count());

        var vae = new StandardVAE<double>(inputChannels: 3, latentChannels: 4, baseChannels: 16,
            channelMultipliers: new[] { 1, 2 }, numResBlocksPerLevel: 1, seed: seed);
        return new MVDreamModel<double>(multiViewUNet: unet, imageVAE: vae,
            textConditioner: new FixedConditioner(ContextDim, ConditioningType.Text),
            imageConditioner: imageConditioner, seed: seed, imageEncoder: encoder);
    }

    private static ClipVisionEncoder<double> TinyClip() => new ClipVisionEncoder<double>(new ClipVisionEncoderOptions
    {
        ImageSize = 28,
        PatchSize = 14,
        Width = FeatureDim,
        Layers = 2,
        Heads = 2,
        MlpWidth = 16,
        ProjectionDim = 4
    }, seed: 2);

    private static double[] Flatten(Tensor<double>[] views) => views.SelectMany(view => view.ToArray()).ToArray();

    [Fact]
    public void ImageDreamPath_IsDeterministicUnderASeedAndDependsOnTheImage()
    {
        var model = TinyImageDream(41, TinyClip(), imageConditioner: null);
        Assert.True(model.MultiViewUNet.HasImagePrompt);
        var image = MVDreamImageConditioningTests.RandomImage(1);

        var first = Flatten(model.GenerateNovelViewsFromImage(image, 4, 2, 5.0, seed: 3));
        var second = Flatten(model.GenerateNovelViewsFromImage(image, 4, 2, 5.0, seed: 3));
        Assert.Equal(first, second);
        Assert.All(first, value => Assert.False(double.IsNaN(value) || double.IsInfinity(value)));

        var other = Flatten(model.GenerateNovelViewsFromImage(MVDreamImageConditioningTests.RandomImage(2), 4, 2, 5.0, seed: 3));
        Assert.True(MaxAbsDifference(first, other) > 1e-9, "different input images must give different views");
    }

    [Fact]
    public void ImageDreamPath_ReachesTheViewsThroughTheResampledImageTokens()
    {
        // Same image (so the same pixel-controller frame), different CLIP features: only the resampler and the
        // decoupled image cross-attention see the difference.
        var conditioner = new FixedConditioner(FeatureDim, ConditioningType.Image);
        var model = TinyImageDream(42, encoder: null, imageConditioner: conditioner);
        var image = MVDreamImageConditioningTests.RandomImage(5);

        conditioner.Features = new Tensor<double>(RandomValues(5 * FeatureDim, 51), new[] { 1, 5, FeatureDim });
        var first = Flatten(model.GenerateNovelViewsFromImage(image, 4, 2, 1.0, seed: 8));
        conditioner.Features = new Tensor<double>(RandomValues(5 * FeatureDim, 52), new[] { 1, 5, FeatureDim });
        var second = Flatten(model.GenerateNovelViewsFromImage(image, 4, 2, 1.0, seed: 8));

        Assert.True(MaxAbsDifference(first, second) > 1e-9, "the image features must reach the views");
    }

    [Fact]
    public void ImageDreamUNet_RejectsTextOnlyGeneration()
    {
        var model = TinyImageDream(43, encoder: null, imageConditioner: null);
        Assert.Throws<InvalidOperationException>(() => model.GenerateMultiView("a robot", numInferenceSteps: 1));
    }

    // ------------------------------------------------------------------ CLIP vision presets

    [Fact]
    public void ClipVisionPresets_HaveTheOfficialShapes()
    {
        var h = ClipVisionEncoderOptions.ForPreset(ClipVisionPreset.ViTH14OpenClip);
        Assert.Equal((224, 14, 1280, 32, 16, 5120, 1024), (h.ImageSize, h.PatchSize, h.Width, h.Layers, h.Heads, h.MlpWidth, h.ProjectionDim));
        Assert.Equal(ClipVisionActivation.Gelu, h.Activation);
        Assert.Equal(ClipWeightLayout.OpenClip, h.WeightLayout);

        var l = ClipVisionEncoderOptions.ForPreset(ClipVisionPreset.ViTL14OpenAI);
        Assert.Equal((224, 14, 1024, 24, 16, 4096, 768), (l.ImageSize, l.PatchSize, l.Width, l.Layers, l.Heads, l.MlpWidth, l.ProjectionDim));
        Assert.Equal(ClipVisionActivation.QuickGelu, l.Activation);
        Assert.Equal(ClipWeightLayout.HuggingFace, l.WeightLayout);

        var clip = TinyClip();
        Assert.Equal(5, clip.TokenCount);
        var image = MVDreamImageConditioningTests.RandomImage(7);
        Assert.Equal(new[] { 1, 5, FeatureDim }, clip.EncodePenultimateHiddenStates(image).Shape.ToArray());
        Assert.Equal(new[] { 1, 4 }, clip.EncodeImage(image).Shape.ToArray());
    }
}

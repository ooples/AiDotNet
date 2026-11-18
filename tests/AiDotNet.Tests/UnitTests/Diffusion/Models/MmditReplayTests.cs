using System.Reflection;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion.Models;

[Collection("FoundationScaleSerial")]
public sealed class MmditReplayTests : IDisposable
{
    private readonly IEngine _prior = AiDotNetEngine.Current;
    public MmditReplayTests() => AiDotNetEngine.Current = new CpuEngine();
    public void Dispose() => AiDotNetEngine.Current = _prior;

    [Fact]
    public void Patch_layout_and_inverse_preserve_batch_channel_and_non_square_pixel_order()
    {
        using var fixture = new Fixture(channels: 2, height: 4, width: 6, hidden: 8, joint: 0, single: 0);
        var dense = (DenseLayer<double>)fixture.Layers["patch"];
        var parameters = new double[checked((int)dense.ParameterCount)];
        for (int i = 0; i < 8; i++) parameters[i * 8 + i] = 1;
        dense.SetParameters(new Vector<double>(parameters));
        var pixels = Enumerable.Range(0, 96).Select(i => (double)(i + 1)).ToArray();
        var image = new Tensor<double>(pixels, [2, 2, 4, 6]);
        var patches = InvokeTensor(fixture.Model, "PatchifyAndEmbed", image);
        var expected = PatchRows(pixels, 2, 2, 4, 6, 2).SelectMany(row => row).ToArray();
        Check(expected, patches.ToArray(), 1e-12);
        Check(pixels, InvokeTensor(fixture.Model, "Unpatchify", patches, 4, 6).ToArray(), 1e-12);

        using var tape = new GradientTape<double>();
        var output = InvokeTensor(fixture.Model, "Unpatchify",
            InvokeTensor(fixture.Model, "PatchifyAndEmbed", image), 4, 6);
        var weights = new Tensor<double>(pixels.Select(v => 0.01 * v).ToArray(), [2, 2, 4, 6]);
        var loss = AiDotNetEngine.Current.ReduceSum(AiDotNetEngine.Current.TensorMultiply(output, weights), null);
        Check(weights.ToArray(), tape.ComputeGradients(loss, new[] { image })[image].ToArray(), 1e-11);
    }

    [Theory]
    [InlineData(false, 0)]
    [InlineData(true, 0)]
    [InlineData(true, 1)]
    public void Complete_forward_and_eager_gradients_match_independent_scalar_reference_and_finite_differences(
        bool conditioned, int singleLayers)
    {
        using var fixture = new Fixture(single: singleLayers);
        var state = fixture.Snapshot();
        var image = fixture.Image(0);
        var times = fixture.Times(0);
        var context = conditioned ? fixture.Context(0) : null;
        var weights = fixture.Weights();
        var expected = Reference(fixture, state, image.ToArray(), times.ToArray(), context?.ToArray());
        using var tape = new GradientTape<double>();
        var output = fixture.Forward(image, times, context);
        Check(expected, output.ToArray(), 2e-9);
        var loss = AiDotNetEngine.Current.ReduceSum(AiDotNetEngine.Current.TensorMultiply(output, weights), null);
        var watched = new[] { image, times }.Concat(context is null ? [] : new[] { context })
            .Concat(fixture.Parameters).ToArray();
        var gradients = tape.ComputeGradients(loss, watched);
        double[] xs = image.ToArray(), ts = times.ToArray();
        double[]? cs = context?.ToArray();
        var ws = weights.ToArray();
        double Objective() => Dot(Reference(fixture, state, xs, ts, cs), ws);
        CheckFiniteDifferences(xs, gradients[image].ToArray(), Objective);
        CheckFiniteDifferences(ts, gradients[times].ToArray(), Objective);
        if (context is not null) CheckFiniteDifferences(cs!, gradients[context].ToArray(), Objective);
        foreach (var parameter in fixture.Parameters)
        {
            var actual = gradients.TryGetValue(parameter, out var gradient)
                ? gradient.ToArray() : new double[parameter.Length];
            CheckFiniteDifferences(state[parameter], actual, Objective, sampleOnly: true);
        }
        foreach (string layer in new[] { "patch", "time1", "joint.ImageAdaLN", "joint.ImageQProj", "output" })
        {
            var parameter = DenseWeights(fixture.Layers[layer]);
            Assert.True(gradients.TryGetValue(parameter, out var gradient), $"Missing gradient for {layer}.");
            Assert.Contains(gradient!.ToArray(), value => Math.Abs(value) > 1e-10);
        }
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void Compiled_objective_rebinds_three_batches_and_preserves_frozen_parameters(bool conditioned, bool checkpoint)
    {
        using var fixture = new Fixture();
        fixture.Model.ActivationCheckpointingEnabled = checkpoint;
        var frozen = fixture.LayerParameters("joint.TextQProj").ToArray();
        var frozenValues = frozen.Select(t => t.ToArray()).ToArray();
        var parameters = fixture.Parameters.Except(frozen).ToArray();
        var image = fixture.Image(0);
        var times = fixture.Times(0);
        Tensor<double>? context = conditioned ? fixture.Context(0) : null;
        var weights = fixture.Weights();
        Tensor<double>? observed = null;
        Tensor<double> Objective()
        {
            observed = fixture.Forward(image, times, context);
            GradientRecording.RetainOutput(observed);
            return AiDotNetEngine.Current.ReduceSum(AiDotNetEngine.Current.TensorMultiply(observed, weights), null);
        }
        using var recording = ClosureObjectiveGuard<double>.BeginRecording();
        CompiledTrainingPlan<double> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var loss = Objective();
            Assert.Null(scope.HostReadOp);
            Assert.False(Assert.IsType<LazyNode<double>>(observed!.LazySource).IsRealized);
            plan = scope.CompileTraining(parameters, loss);
        }
        var tracedOutput = observed!;
        using (plan)
        {
            var guard = ClosureObjectiveGuard<double>.TakeRecorded(out var reason);
            Assert.True(guard is not null, reason);
            guard!.AttachTo(plan);
            plan.ConfigureOptimizer(OptimizerType.AdamW, learningRate: 0.001, weightDecay: 0);
            for (int step = 0; step < 3; step++)
            {
                image = fixture.Image(step);
                times = fixture.Times(step);
                context = conditioned ? fixture.Context(step) : null;
                var state = fixture.Snapshot();
                var xs = image.ToArray();
                var ts = times.ToArray();
                var cs = context?.ToArray();
                var ws = weights.ToArray();
                var expected = Reference(fixture, state, xs, ts, cs);
                Assert.True(guard.TryRebind(Objective, parameters, out _) == ClosureGuardResult.Matched, guard.LastReason);
                Near(Dot(expected, ws), plan.Step().ToArray()[0], 2e-9);
                Check(expected, tracedOutput.ToArray(), 2e-9);
                double ScalarLoss() => Dot(Reference(fixture, state, xs, ts, cs), ws);
                for (int p = 0; p < parameters.Length; p++)
                    CheckFiniteDifferences(state[parameters[p]], plan.Gradients[p].ToArray(), ScalarLoss, sampleOnly: true);
                for (int p = 0; p < frozen.Length; p++) Assert.Equal(frozenValues[p], frozen[p].ToArray());
                Assert.Equal(parameters.Length, plan.ParameterAcceptedUpdates.Count);
                Assert.Contains(plan.ParameterAcceptedUpdates, age => age == step + 1L);
            }
        }
    }

    [Fact]
    public void Checkpointing_preserves_image_time_conditioning_and_parameter_gradients()
    {
        using var fixture = new Fixture();
        Dictionary<Tensor<double>, double[]> Gradients(bool checkpoint)
        {
            fixture.Model.ActivationCheckpointingEnabled = checkpoint;
            var image = fixture.Image(1);
            var times = fixture.Times(1);
            var context = fixture.Context(1);
            using var tape = new GradientTape<double>();
            var output = fixture.Forward(image, times, context);
            var loss = AiDotNetEngine.Current.ReduceSum(
                AiDotNetEngine.Current.TensorMultiply(output, fixture.Weights()), null);
            var watched = new[] { image, times, context }.Concat(fixture.Parameters).ToArray();
            var gradients = tape.ComputeGradients(loss, watched);
            var result = new Dictionary<Tensor<double>, double[]>();
            // Stable model tensor keys compare parameter VJPs, while three separate keys preserve input VJPs.
            foreach (var parameter in fixture.Parameters)
                result[parameter] = gradients.TryGetValue(parameter, out var gradient)
                    ? gradient.ToArray() : new double[parameter.Length];
            result[fixture.InputGradientKey] = gradients[image].ToArray();
            result[fixture.TimeGradientKey] = gradients[times].ToArray();
            result[fixture.ContextGradientKey] = gradients[context].ToArray();
            return result;
        }
        var ordinary = Gradients(false);
        var checkpointed = Gradients(true);
        foreach (var pair in ordinary) Check(pair.Value, checkpointed[pair.Key], 2e-9);
    }

    [Fact]
    public void Zero_gates_preserve_residuals_and_still_receive_each_batchs_gradient()
    {
        using var fixture = new Fixture();
        var input = new Tensor<double>(Enumerable.Range(0, 24).Select(i => 0.1 * i).ToArray(), [2, 3, 4]);
        var residual = new Tensor<double>(Enumerable.Range(0, 24).Select(i => Math.Cos(i)).ToArray(), [2, 3, 4]);
        var gate = new Tensor<double>([2, 1, 4]);
        var weights = new Tensor<double>(Enumerable.Range(0, 24).Select(i => Math.Sin(0.3 * i)).ToArray(), [2, 3, 4]);
        using var tape = new GradientTape<double>();
        var output = InvokeTensor(fixture.Model, "AddWithGate", input, residual, gate);
        Assert.Equal(input.ToArray(), output.ToArray());
        var loss = AiDotNetEngine.Current.ReduceSum(AiDotNetEngine.Current.TensorMultiply(output, weights), null);
        var gradients = tape.ComputeGradients(loss, new[] { input, residual, gate });
        Check(weights.ToArray(), gradients[input].ToArray(), 1e-12);
        Assert.All(gradients[residual].ToArray(), value => Assert.Equal(0.0, value));
        var expected = new double[8];
        var rs = residual.ToArray();
        var ws = weights.ToArray();
        for (int b = 0; b < 2; b++)
        for (int n = 0; n < 3; n++)
        for (int h = 0; h < 4; h++) expected[b * 4 + h] += rs[(b * 3 + n) * 4 + h] * ws[(b * 3 + n) * 4 + h];
        Check(expected, gradients[gate].ToArray(), 1e-12);
    }

    [Fact]
    public void Invalid_image_metadata_records_nothing_and_scope_recovers()
    {
        using var fixture = new Fixture();
        using var scope = GraphMode.EnableTraining(fixture.Parameters);
        var time = new Tensor<double>([2, fixture.Model.TimeEmbeddingDim]);
        foreach (int[] shape in new[] { new[] { 2, 1, 4 }, new[] { 2, 2, 2, 4 }, new[] { 2, 1, 3, 4 } })
        {
            Assert.Throws<ArgumentException>(() => fixture.Model.PredictNoiseWithEmbedding(new Tensor<double>(shape), time));
            Assert.Equal(0, scope.NodeCount);
            Assert.Same(scope, GraphMode.Current);
        }
        Assert.NotNull(fixture.Forward(fixture.Image(0), fixture.Times(0), fixture.Context(0)).LazySource);
        Assert.Null(scope.HostReadOp);
    }

    private sealed class Fixture : IDisposable
    {
        internal readonly MMDiTNoisePredictor<double> Model;
        internal readonly Dictionary<string, object> Layers = new();
        internal readonly int Channels, Height, Width, Hidden, Patch = 2;
        internal readonly Tensor<double>[] Parameters;
        internal readonly Tensor<double> InputGradientKey = new([1]);
        internal readonly Tensor<double> TimeGradientKey = new([1]);
        internal readonly Tensor<double> ContextGradientKey = new([1]);

        internal Fixture(int channels = 1, int height = 2, int width = 4, int hidden = 4, int joint = 1, int single = 1)
        {
            Channels = channels; Height = height; Width = width; Hidden = hidden;
            Model = new MMDiTNoisePredictor<double>(inputChannels: channels, hiddenSize: hidden,
                numJointLayers: joint, numSingleLayers: single, numHeads: 2, patchSize: Patch, contextDim: 3,
                mlpRatio: 1, seed: 42);
            Forward(Image(0), Times(0), Context(0)).ToArray(); // Resolve lazy weights before recording or selecting leaves.
            foreach (var pair in new[] { ("patch", "_patchEmbed"), ("time1", "_timeEmbed1"), ("time2", "_timeEmbed2"),
                ("context", "_contextProj"), ("finalNorm", "_finalNorm"), ("finalMod", "_adalnModulation"), ("output", "_outputProj") })
                Layers.Add(pair.Item1, Field<object>(Model, pair.Item2));
            var joints = Field<List<MMDiTNoisePredictor<double>.MMDiTBlock>>(Model, "_jointBlocks");
            var singles = Field<List<MMDiTNoisePredictor<double>.MMDiTSingleBlock>>(Model, "_singleBlocks");
            if (joints.Count > 0) AddBlock("joint", joints[0]);
            if (singles.Count > 0) AddBlock("single", singles[0]);
            int layerIndex = 0;
            foreach (var layer in Layers.Values)
            {
                var typed = (ILayer<double>)layer;
                var values = new double[checked((int)typed.ParameterCount)];
                for (int i = 0; i < values.Length; i++) values[i] = 0.17 * Math.Sin(0.61 * i + 0.37 * ++layerIndex);
                typed.SetParameters(new Vector<double>(values));
                if (layer is LayerNormalizationLayer<double> norm)
                {
                    norm.GetGammaTensor().AsWritableSpan().Fill(0.9);
                    norm.GetGammaTensor().IncrementVersion();
                }
            }
            Parameters = Layers.Keys.SelectMany(LayerParameters).ToArray();
            Assert.Equal(Parameters.Length, Parameters.Distinct().Count());
        }
        private void AddBlock(string prefix, object block)
        {
            foreach (var property in block.GetType().GetProperties()) Layers.Add(prefix + "." + property.Name, property.GetValue(block)!);
        }
        internal IEnumerable<Tensor<double>> LayerParameters(string name)
        {
            var layer = Layers[name];
            if (layer is LayerNormalizationLayer<double> norm)
                return new[] { norm.GetGammaTensor(), norm.GetBetaTensor() };
            return new[] { DenseWeights(layer), Field<Tensor<double>>(layer, "_biases") };
        }
        internal Dictionary<Tensor<double>, double[]> Snapshot() => Parameters.ToDictionary(t => t, t => t.ToArray());
        internal Tensor<double> Image(int step) => new(
            Enumerable.Range(0, 2 * Channels * Height * Width).Select(i => Math.Sin(0.41 * i + 0.31 * step)).ToArray(),
            [2, Channels, Height, Width]);
        internal Tensor<double> Times(int step) => new(new[] { 0.7 + 0.2 * step, 2.1 - 0.15 * step }, [2]);
        internal Tensor<double> Context(int step) => new(
            Enumerable.Range(0, 12).Select(i => 0.5 * Math.Cos(0.37 * i - 0.23 * step)).ToArray(), [2, 2, 3]);
        internal Tensor<double> Weights() => new(
            Enumerable.Range(0, 2 * Channels * Height * Width).Select(i => Math.Cos(0.29 * i)).ToArray(),
            [2, Channels, Height, Width]);
        internal Tensor<double> Forward(Tensor<double> image, Tensor<double> times, Tensor<double>? context) =>
            Model.PredictNoiseWithEmbedding(image, Model.GetTimestepEmbedding(times), context);
        public void Dispose() => Model.Dispose();
    }

    // Independent scalar reference. Rows are mathematical token vectors, never engine reshapes or attention calls.
    private static double[] Reference(Fixture f, Dictionary<Tensor<double>, double[]> state,
        double[] image, double[] times, double[]? context)
    {
        const int batch = 2, heads = 2;
        int hidden = f.Hidden, imageCount = f.Height / f.Patch * (f.Width / f.Patch);
        int textCount = context is null ? 1 : 2;
        double[][] Linear(double[][] rows, string name, int activation = 0)
        {
            var weight = DenseWeights(f.Layers[name]);
            int input = weight._shape[0], output = weight._shape[1];
            var w = state[weight];
            var bias = state[Field<Tensor<double>>(f.Layers[name], "_biases")];
            return rows.Select(row => Enumerable.Range(0, output).Select(o =>
            {
                double value = bias[o];
                for (int i = 0; i < input; i++) value += row[i] * w[i * output + o];
                return activation == 1 ? value / (1 + Math.Exp(-value)) : activation == 2
                    ? 0.5 * value * (1 + Math.Tanh(Math.Sqrt(2 / Math.PI) * (value + 0.044715 * value * value * value))) : value;
            }).ToArray()).ToArray();
        }
        double[][] Normalize(double[][] rows, string name)
        {
            var layer = (LayerNormalizationLayer<double>)f.Layers[name];
            var gamma = state[layer.GetGammaTensor()];
            var beta = state[layer.GetBetaTensor()];
            double epsilon = layer.GetEpsilon();
            return rows.Select(row =>
            {
                double mean = row.Average(), variance = row.Sum(x => (x - mean) * (x - mean)) / row.Length;
                return row.Select((x, h) => (x - mean) / Math.Sqrt(variance + epsilon) * gamma[h] + beta[h]).ToArray();
            }).ToArray();
        }
        double[][] Modulate(double[][] rows, double[][] modulation, int tokens, int shiftOffset, int scaleOffset) =>
            rows.Select((row, i) => row.Select((x, h) => x * (1 + modulation[i / tokens][scaleOffset + h])
                + modulation[i / tokens][shiftOffset + h]).ToArray()).ToArray();
        double[][] Residual(double[][] rows, double[][] delta, double[][] modulation, int tokens, int offset) =>
            rows.Select((row, i) => row.Select((x, h) => x + delta[i][h] * modulation[i / tokens][offset + h]).ToArray()).ToArray();
        double[][] Concat(double[][] first, int firstCount, double[][] second, int secondCount) =>
            Enumerable.Range(0, batch).SelectMany(b => first.Skip(b * firstCount).Take(firstCount)
                .Concat(second.Skip(b * secondCount).Take(secondCount))).ToArray();
        double[][] Attention(double[][] q, double[][] k, double[][] v, int queries, int keys)
        {
            int dim = hidden / heads;
            var result = q.Select(_ => new double[hidden]).ToArray();
            for (int b = 0; b < batch; b++)
            for (int head = 0; head < heads; head++)
            for (int row = 0; row < queries; row++)
            {
                var scores = new double[keys];
                for (int col = 0; col < keys; col++)
                    for (int d = 0; d < dim; d++) scores[col] += q[b * queries + row][head * dim + d]
                        * k[b * keys + col][head * dim + d] / Math.Sqrt(dim);
                double max = scores.Max(), sum = scores.Sum(s => Math.Exp(s - max));
                for (int col = 0; col < keys; col++)
                    for (int d = 0; d < dim; d++) result[b * queries + row][head * dim + d] +=
                        Math.Exp(scores[col] - max) / sum * v[b * keys + col][head * dim + d];
            }
            return result;
        }
        int half = f.Model.TimeEmbeddingDim / 2;
        var rawTime = times.Select(t => Enumerable.Range(0, half).Select(i =>
            Math.Sin(t * Math.Exp(-i * Math.Log(10000) / (half - 1)))).Concat(Enumerable.Range(0, half).Select(i =>
            Math.Cos(t * Math.Exp(-i * Math.Log(10000) / (half - 1))))).ToArray()).ToArray();
        var time = Linear(Linear(rawTime, "time1", 1), "time2", 1);
        var imageRows = Linear(PatchRows(image, batch, f.Channels, f.Height, f.Width, f.Patch), "patch");
        for (int b = 0; b < batch; b++)
        for (int n = 0; n < imageCount; n++)
        for (int h = 0; h < hidden; h += 2)
        {
            double angle = n * Math.Pow(10000, -(double)h / hidden);
            imageRows[b * imageCount + n][h] += Math.Sin(angle);
            imageRows[b * imageCount + n][h + 1] += Math.Cos(angle);
        }
        var textRows = Linear(Enumerable.Range(0, batch * textCount).Select(i => context is null
            ? new double[3] : context.Skip(i * 3).Take(3).ToArray()).ToArray(), "context");
        if (f.Layers.ContainsKey("joint.ImageAdaLN"))
        {
            var im = Linear(time, "joint.ImageAdaLN");
            var tm = Linear(time, "joint.TextAdaLN");
            var ni = Modulate(Normalize(imageRows, "joint.ImageNorm1"), im, imageCount, 0, hidden);
            var nt = Modulate(Normalize(textRows, "joint.TextNorm1"), tm, textCount, 0, hidden);
            var keys = Concat(Linear(nt, "joint.TextKProj"), textCount, Linear(ni, "joint.ImageKProj"), imageCount);
            var values = Concat(Linear(nt, "joint.TextVProj"), textCount, Linear(ni, "joint.ImageVProj"), imageCount);
            var ai = Attention(Linear(ni, "joint.ImageQProj"), keys, values, imageCount, imageCount + textCount);
            var at = Attention(Linear(nt, "joint.TextQProj"), keys, values, textCount, imageCount + textCount);
            imageRows = Residual(imageRows, Linear(ai, "joint.ImageOutProj"), im, imageCount, 2 * hidden);
            textRows = Residual(textRows, Linear(at, "joint.TextOutProj"), tm, textCount, 2 * hidden);
            ni = Modulate(Normalize(imageRows, "joint.ImageNorm2"), im, imageCount, 3 * hidden, 4 * hidden);
            nt = Modulate(Normalize(textRows, "joint.TextNorm2"), tm, textCount, 3 * hidden, 4 * hidden);
            imageRows = Residual(imageRows, Linear(Linear(ni, "joint.ImageMLP1", 2), "joint.ImageMLP2"), im, imageCount, 5 * hidden);
            textRows = Residual(textRows, Linear(Linear(nt, "joint.TextMLP1", 2), "joint.TextMLP2"), tm, textCount, 5 * hidden);
        }
        if (f.Layers.ContainsKey("single.AdaLN"))
        {
            int count = textCount + imageCount;
            var combined = Concat(textRows, textCount, imageRows, imageCount);
            var modulation = Linear(time, "single.AdaLN");
            var normed = Modulate(Normalize(combined, "single.Norm"), modulation, count, 0, hidden);
            var attention = Linear(Attention(Linear(normed, "single.QProj"), Linear(normed, "single.KProj"),
                Linear(normed, "single.VProj"), count, count), "single.OutProj");
            var mlp = Linear(Linear(normed, "single.MLP1", 2), "single.MLP2");
            var delta = attention.Select((row, i) => row.Select((x, h) => x + mlp[i][h]).ToArray()).ToArray();
            combined = Residual(combined, delta, modulation, count, 2 * hidden);
            imageRows = Enumerable.Range(0, batch).SelectMany(b => combined.Skip(b * count + textCount).Take(imageCount)).ToArray();
        }
        var finalMod = Linear(time, "finalMod");
        var outputRows = Linear(Modulate(Normalize(imageRows, "finalNorm"), finalMod, imageCount, 0, hidden), "output");
        var output = new double[image.Length];
        int patchColumns = f.Width / f.Patch;
        for (int b = 0; b < batch; b++)
        for (int n = 0; n < imageCount; n++)
        for (int c = 0; c < f.Channels; c++)
        for (int py = 0; py < f.Patch; py++)
        for (int px = 0; px < f.Patch; px++)
            output[((b * f.Channels + c) * f.Height + n / patchColumns * f.Patch + py) * f.Width + n % patchColumns * f.Patch + px] =
                outputRows[b * imageCount + n][(c * f.Patch + py) * f.Patch + px];
        return output;
    }

    private static double[][] PatchRows(double[] image, int batch, int channels, int height, int width, int patch)
    {
        var rows = new List<double[]>();
        for (int b = 0; b < batch; b++)
        for (int y = 0; y < height; y += patch)
        for (int x = 0; x < width; x += patch)
        {
            var row = new List<double>();
            for (int c = 0; c < channels; c++)
            for (int py = 0; py < patch; py++)
            for (int px = 0; px < patch; px++) row.Add(image[((b * channels + c) * height + y + py) * width + x + px]);
            rows.Add(row.ToArray());
        }
        return rows.ToArray();
    }
    private static void CheckFiniteDifferences(double[] values, double[] actual, Func<double> loss, bool sampleOnly = false)
    {
        IEnumerable<int> indices = sampleOnly ? new[] { 0, values.Length / 2, values.Length - 1 }.Distinct() : Enumerable.Range(0, values.Length);
        foreach (int i in indices)
        {
            const double epsilon = 1e-6;
            double saved = values[i];
            values[i] = saved + epsilon;
            double plus = loss();
            values[i] = saved - epsilon;
            double minus = loss();
            values[i] = saved;
            Near((plus - minus) / (2 * epsilon), actual[i], 3e-6);
        }
    }
    private static Tensor<double> DenseWeights(object layer) => Field<Tensor<double>>(layer, "_weights");
    private static T Field<T>(object instance, string name) =>
        (T)instance.GetType().GetField(name, BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(instance)!;
    private static Tensor<double> InvokeTensor(object instance, string name, params object[] arguments) =>
        (Tensor<double>)instance.GetType().GetMethod(name, BindingFlags.NonPublic | BindingFlags.Instance)!.Invoke(instance, arguments)!;
    private static double Dot(double[] first, double[] second) => first.Zip(second, (a, b) => a * b).Sum();
    private static void Check(double[] expected, double[] actual, double tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++) Near(expected[i], actual[i], tolerance);
    }
    private static void Near(double expected, double actual, double tolerance) =>
        Assert.True(!double.IsNaN(actual) && !double.IsInfinity(actual) && Math.Abs(expected - actual) <= tolerance,
            $"Expected {expected:R}, got {actual:R}; tolerance {tolerance:R}.");
}

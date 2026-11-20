using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.TextToSpeech.Vocoders;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

[Collection("FusedTrainingSerial")]
public sealed class HiFiGanCpuResidualAccumulatorTests
{
    [Theory]
    [InlineData(1)]
    [InlineData(8)]
    [InlineData(16)]
    public void MultiStageAccumulatorIsBitwiseEquivalentAndDoesNotOverwritePublishedOutputs(int batch)
    {
        var previousEngine = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = new CpuEngine();
            var generator = Create<float>(new[] { 2, 4, 2 }, new[] { 4, 8, 4 });
            var input = new Tensor<float>(Enumerable.Range(0, batch * 4 * 9)
                .Select(i => .2f * (float)Math.Sin(i * .13)).ToArray(), new[] { batch, 4, 9 });
            var inputCopy = input.ToArray();
            var changedInput = new Tensor<float>(inputCopy.Select(value => value * .7f + .03f).ToArray(), new[] { batch, 4, 9 });
            var parameters = generator.Layers.SelectMany(layer => layer.GetParameters().ToArray()).ToArray();
            float[] expected;
            float[] changedExpected;
            using (new NoGradScope<float>())
            {
                expected = generator.Forward(input).ToArray();
                changedExpected = generator.Forward(changedInput).ToArray();
            }
            Assert.Equal(0, generator.PrivateCpuInferenceCalls);

            using var callerArena = TensorArena.Create();
            var sentinel = TensorAllocator.Rent<float>(new[] { batch * 32 });
            sentinel.Fill(123.0f);
            using var noGrad = new NoGradScope<float>();
            var published = generator.Forward(input);
            AssertBitwise(expected, published.ToArray());
            for (int repeat = 0; repeat < 4; repeat++)
            {
                bool changed = repeat % 2 == 0;
                AssertBitwise(changed ? changedExpected : expected, generator.Forward(changed ? changedInput : input).ToArray());
                AssertBitwise(expected, published.ToArray());
                Assert.All(sentinel.ToArray(), value => Assert.Equal(123.0f, value));
            }
            Assert.Equal(inputCopy, input.ToArray());
            Assert.Equal(parameters, generator.Layers.SelectMany(layer => layer.GetParameters().ToArray()).ToArray());
            Assert.Equal(5, generator.PrivateCpuInferenceCalls);
        }
        finally { AiDotNetEngine.Current = previousEngine; }
    }

    [Fact]
    public void PrivateCpuWorkspacePreservesPublishedOutputsAcrossBatchGrowthAndReuse()
    {
        var previousEngine = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = new CpuEngine();
            var generator = Create<float>(new[] { 2, 4, 2 }, new[] { 4, 8, 4 });
            int[] batches = { 1, 8, 16, 1 };
            var inputs = batches.Select(batch => new Tensor<float>(Enumerable.Range(0, batch * 4 * 9)
                .Select(i => .13f * (float)Math.Sin(i * .19 + batch)).ToArray(), new[] { batch, 4, 9 })).ToArray();
            float[][] expected;
            using (new NoGradScope<float>()) expected = inputs.Select(input => generator.Forward(input).ToArray()).ToArray();
            Assert.Equal(0, generator.PrivateCpuInferenceCalls);
            var published = new List<Tensor<float>>();
            using var arena = TensorArena.Create();
            using var noGrad = new NoGradScope<float>();
            for (int call = 0; call < inputs.Length; call++)
            {
                published.Add(generator.Forward(inputs[call]));
                for (int prior = 0; prior <= call; prior++) AssertBitwise(expected[prior], published[prior].ToArray());
            }
            Assert.Equal(4, generator.PrivateCpuInferenceCalls);
        }
        finally { AiDotNetEngine.Current = previousEngine; }
    }

    [Fact]
    public void PrefixShapesPreserveOddUpsamplingAndConditionInputsAcrossCalls()
    {
        var previousEngine = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = new CpuEngine();
            // Stage element counts first grow, shrink, then grow again; odd kernel/rate differences add one frame.
            var generator = Create<double>(new[] { 2, 1, 3 }, new[] { 5, 3, 6 }, condition: true);
            var input = new Tensor<double>(Enumerable.Range(0, 2 * 4 * 11)
                .Select(i => .1 * Math.Sin(i * .17)).ToArray(), new[] { 2, 4, 11 });
            var condition = new Tensor<double>(new[] { .2, .3, .4, .5, .6, .7 }, new[] { 2, 3, 1 });
            var inputCopy = input.ToArray();
            var conditionCopy = condition.ToArray();
            double[] expected;
            using (new NoGradScope<double>()) expected = generator.Forward(input, condition).ToArray();

            using var callerArena = TensorArena.Create();
            using var nestedArena = TensorArena.Create();
            var sentinel = TensorAllocator.Rent<double>(new[] { 2, 8, 23 });
            sentinel.Fill(-321.0);
            using var noGrad = new NoGradScope<double>();
            for (int repeat = 0; repeat < 4; repeat++)
            {
                AssertBitwise(expected, generator.Forward(input, condition).ToArray());
                Assert.All(sentinel.ToArray(), value => Assert.Equal(-321.0, value));
            }
            Assert.Equal(inputCopy, input.ToArray());
            Assert.Equal(conditionCopy, condition.ToArray());
        }
        finally { AiDotNetEngine.Current = previousEngine; }
    }

    [Theory]
    [InlineData(false, 1)]
    [InlineData(true, 1)]
    [InlineData(true, 3)]
    public void IdentityAverageAndCustomCpuEngineKeepTheirExistingLifetimeBehavior(bool customEngine, int branches)
    {
        var previousEngine = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = customEngine ? new IdentityScalarCpuEngine() : new CpuEngine();
            var generator = Create<double>(new[] { 2, 2 }, new[] { 4, 4 }, branches: branches);
            var input = new Tensor<double>(Enumerable.Range(0, 4 * 9)
                .Select(i => .1 * Math.Sin(i * .11)).ToArray(), new[] { 1, 4, 9 });
            var changedInput = new Tensor<double>(input.ToArray().Select(value => value + .15).ToArray(), new[] { 1, 4, 9 });
            double[] expected;
            double[] changedExpected;
            using (new NoGradScope<double>())
            {
                expected = generator.Forward(input).ToArray();
                changedExpected = generator.Forward(changedInput).ToArray();
            }
            using var callerArena = TensorArena.Create();
            using var noGrad = new NoGradScope<double>();
            var published = generator.Forward(input);
            for (int repeat = 0; repeat < 4; repeat++)
            {
                bool changed = repeat % 2 == 0;
                AssertBitwise(changed ? changedExpected : expected, generator.Forward(changed ? changedInput : input).ToArray());
                AssertBitwise(expected, published.ToArray());
            }
        }
        finally { AiDotNetEngine.Current = previousEngine; }
    }

    [Fact]
    public void RecycledOwnedAverageDoesNotChangeLaterTapedTrainingOrPublishedOutput()
    {
        var previousEngine = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = new CpuEngine();
            var generator = Create<double>(new[] { 2, 1, 3 }, new[] { 5, 3, 6 }, condition: true);
            var input = new Tensor<double>(Enumerable.Range(0, 4 * 7)
                .Select(i => .15 * Math.Sin(i * .13)).ToArray(), new[] { 1, 4, 7 });
            var condition = new Tensor<double>(new[] { .2, -.1, .3 }, new[] { 1, 3, 1 });
            var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(
                generator.Layers.Cast<ILayer<double>>().ToList());
            double[][] TrainingGradients()
            {
                foreach (var layer in generator.Layers) layer.SetTrainingMode(true);
                using var tape = new GradientTape<double>();
                var output = generator.Forward(input, condition);
                var engine = AiDotNetEngine.Current;
                var loss = engine.ReduceSum(engine.TensorMultiply(output, output), null, false);
                var gradients = tape.ComputeGradients(loss, sources: parameters);
                return parameters.Select(parameter => gradients[parameter].ToArray()).ToArray();
            }
            var expectedGradients = TrainingGradients();
            foreach (var layer in generator.Layers) layer.SetTrainingMode(false);
            double[] expectedOutput;
            using (new NoGradScope<double>()) expectedOutput = generator.Forward(input, condition).ToArray();
            using var callerArena = TensorArena.Create();
            Tensor<double> published;
            using (new NoGradScope<double>())
            {
                published = generator.Forward(input, condition);
                AssertBitwise(expectedOutput, published.ToArray());
                var changed = new Tensor<double>(input.ToArray().Select(value => value * -.7 + .1).ToArray(), input.Shape.ToArray());
                _ = generator.Forward(changed, condition).ToArray();
                AssertBitwise(expectedOutput, published.ToArray());
            }
            // Keep the caller arena and published eval output alive while training resumes.
            // The taped route must preserve every parameter gradient instead of overwriting
            // a stage value needed by an earlier convolution's backward action.
            var actualGradients = TrainingGradients();
            Assert.Equal(expectedGradients.Length, actualGradients.Length);
            for (int i = 0; i < expectedGradients.Length; i++)
                AssertBitwise(expectedGradients[i], actualGradients[i]);
            AssertBitwise(expectedOutput, published.ToArray());
        }
        finally { AiDotNetEngine.Current = previousEngine; }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(8)]
    public void UpsamplingWorkspaceStaysOutsideCallerLifetime(int batch)
    {
        var previousEngine = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = new CpuEngine();
            // Halving channels while doubling time keeps all four stage payloads
            // equal. Retaining each upsampler's temporaries in the caller would
            // accumulate slots despite their disjoint stage lifetimes.
            var generator = Create<float>(new[] { 2, 2, 2, 2 }, new[] { 4, 4, 4, 4 });
            var input = new Tensor<float>(Enumerable.Range(0, batch * 4 * 64)
                .Select(index => .15f * (float)Math.Sin(index * .11)).ToArray(), new[] { batch, 4, 64 });
            var original = input.ToArray();
            float[] expected;
            using (new NoGradScope<float>()) expected = generator.Forward(input).ToArray();

            using var arena = TensorArena.Create();
            using var noGrad = new NoGradScope<float>();
            var output = generator.Forward(input);
            AssertBitwise(expected, output.ToArray());
            Assert.Equal(original, input.ToArray());
            // The caller owns pre/post-convolution values and one shared stage
            // boundary, rather than three new full-size upsampling values per
            // stage. Allow nine largest-stage payloads for those public lifetimes.
            long largestStageBytes = (long)batch * 16 * 64 * sizeof(float);
            Assert.True(arena.PeakBackingBytes <= 9 * largestStageBytes,
                $"Caller retained {arena.PeakBackingBytes} bytes for a {largestStageBytes}-byte stage.");
        }
        finally { AiDotNetEngine.Current = previousEngine; }
    }

    private static HiFiGanGenerator<T> Create<T>(int[] rates, int[] kernels, int branches = 3, bool condition = false)
    {
        var generator = new HiFiGanGenerator<T>(AiDotNetEngine.Current, 4, 16, rates, kernels,
            new[] { 3, 5, 7 }.Take(branches).ToArray(),
            Enumerable.Range(0, branches).Select(_ => new[] { 1, 3 }).ToArray(),
            resblockType1: true, conditionChannels: condition ? 3 : 0,
            conditionInEachUpsamplingLayer: condition, initialization: new Random(1234));
        foreach (var layer in generator.Layers) layer.SetTrainingMode(false);
        return generator;
    }

    private sealed class IdentityScalarCpuEngine : CpuEngine
    {
        public override Tensor<T> TensorMultiplyScalar<T>(Tensor<T> tensor, T scalar)
        {
            var ops = MathHelper.GetNumericOperations<T>();
            return ops.Equals(scalar, ops.One) ? tensor : base.TensorMultiplyScalar(tensor, scalar);
        }
    }

    private static void AssertBitwise(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        Assert.All(expected, value => Assert.False(float.IsNaN(value) || float.IsInfinity(value)));
        var expectedBytes = new byte[expected.Length * sizeof(float)];
        var actualBytes = new byte[actual.Length * sizeof(float)];
        Buffer.BlockCopy(expected, 0, expectedBytes, 0, expectedBytes.Length);
        Buffer.BlockCopy(actual, 0, actualBytes, 0, actualBytes.Length);
        Assert.Equal(expectedBytes, actualBytes);
    }

    private static void AssertBitwise(double[] expected, double[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.False(double.IsNaN(expected[i]) || double.IsInfinity(expected[i]));
            Assert.Equal(BitConverter.DoubleToInt64Bits(expected[i]), BitConverter.DoubleToInt64Bits(actual[i]));
        }
    }
}

using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.TextToSpeech.Vocoders;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

[Collection("FusedTrainingSerial")]
public sealed class HiFiGanDetachedGenerationTests
{
    [Theory]
    [InlineData(1)]
    [InlineData(8)]
    public void DetachedTrainingGenerationPreservesValuesAndPublicLifetimes(int batch)
    {
        var previous = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = new CpuEngine();
            var generator = Create<float>();
            var input = new Tensor<float>(Enumerable.Range(0, batch * 4 * 64)
                .Select(i => .15f * (float)Math.Sin(i * .11)).ToArray(), new[] { batch, 4, 64 });
            var original = input.ToArray();
            var parameters = generator.Layers.SelectMany(layer => layer.GetParameters().ToArray()).ToArray();
            float[] expected;
            using (new NoGradScope<float>()) expected = generator.Forward(input).ToArray();

            Assert.Null(TensorArena.Current);
            Tensor<float> withoutCaller;
            using (new NoGradScope<float>())
            {
                withoutCaller = generator.ForwardDetachedForDiscriminator(input);
                AssertBits(expected, withoutCaller.ToArray());
                _ = generator.ForwardDetachedForDiscriminator(input);
                AssertBits(expected, withoutCaller.ToArray());
            }
            Assert.Null(TensorArena.Current);

            using var arena = TensorArena.Create();
            var sentinel = TensorAllocator.Rent<float>(new[] { batch * 32 });
            sentinel.Fill(321f);
            using var noGrad = new NoGradScope<float>();
            var published = generator.ForwardDetachedForDiscriminator(input);
            AssertBits(expected, published.ToArray());
            long largestStageBytes = (long)batch * 16 * 64 * sizeof(float);
            Assert.True(arena.PeakBackingBytes <= 10 * largestStageBytes,
                $"Detached caller retained {arena.PeakBackingBytes} bytes for a {largestStageBytes}-byte stage.");
            for (int i = 0; i < 3; i++)
            {
                AssertBits(expected, generator.ForwardDetachedForDiscriminator(input).ToArray());
                AssertBits(expected, published.ToArray());
                AssertBits(expected, withoutCaller.ToArray());
                Assert.All(sentinel.ToArray(), value => Assert.Equal(321f, value));
            }
            Assert.Equal(original, input.ToArray());
            Assert.All(generator.Layers, layer => Assert.True(layer.InTrainingMode));
            Assert.Equal(parameters, generator.Layers.SelectMany(layer => layer.GetParameters().ToArray()).ToArray());
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    [Fact]
    public void DetachedEntryOutsideNoGradPreservesEveryTapedParameterGradient()
    {
        var previous = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = new CpuEngine();
            var generator = Create<double>();
            var input = new Tensor<double>(Enumerable.Range(0, 4 * 7)
                .Select(i => .15 * Math.Sin(i * .13)).ToArray(), new[] { 1, 4, 7 });
            var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(
                generator.Layers.Cast<ILayer<double>>().ToList());
            double[][] Gradients(bool detachedEntry)
            {
                using var tape = new GradientTape<double>();
                var output = detachedEntry ? generator.ForwardDetachedForDiscriminator(input) : generator.Forward(input);
                var engine = AiDotNetEngine.Current;
                var loss = engine.ReduceSum(engine.TensorMultiply(output, output), null, false);
                var gradients = tape.ComputeGradients(loss, sources: parameters);
                return parameters.Select(parameter => gradients[parameter].ToArray()).ToArray();
            }
            var expected = Gradients(false);
            using var arena = TensorArena.Create();
            double[] publishedValues;
            Tensor<double> published;
            using (new NoGradScope<double>())
            {
                published = generator.ForwardDetachedForDiscriminator(input);
                publishedValues = published.ToArray();
                var other = new Tensor<double>(input.ToArray().Select(value => -.7 * value + .1).ToArray(), input.Shape.ToArray());
                _ = generator.ForwardDetachedForDiscriminator(other);
                Assert.Equal(publishedValues, published.ToArray());
            }
            var actual = Gradients(true);
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i]);
            Assert.Equal(publishedValues, published.ToArray());
            Assert.All(generator.Layers, layer => Assert.True(layer.InTrainingMode));
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    private static HiFiGanGenerator<T> Create<T>()
    {
        var generator = new HiFiGanGenerator<T>(AiDotNetEngine.Current, 4, 16,
            new[] { 2, 2, 2, 2 }, new[] { 4, 4, 4, 4 }, new[] { 3, 5, 7 },
            new[] { new[] { 1, 3 }, new[] { 1, 3 }, new[] { 1, 3 } },
            resblockType1: true, initialization: new Random(1234), defaultInitialization: new Random(1245));
        foreach (var layer in generator.Layers) layer.SetTrainingMode(true);
        return generator;
    }

    private static void AssertBits(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        var expectedBytes = new byte[expected.Length * sizeof(float)];
        var actualBytes = new byte[actual.Length * sizeof(float)];
        Buffer.BlockCopy(expected, 0, expectedBytes, 0, expectedBytes.Length);
        Buffer.BlockCopy(actual, 0, actualBytes, 0, actualBytes.Length);
        Assert.Equal(expectedBytes, actualBytes);
    }
}

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Training;

[Collection("EngineCurrentGlobalState")]
public sealed class CompiledStepRandomDataTests : IDisposable
{
    private readonly IEngine _prior = AiDotNetEngine.Current;
    public void Dispose() => AiDotNetEngine.Current = _prior;

    private sealed class CountingRandom(int seed) : Random(seed)
    {
        internal int Calls { get; private set; }
        public override double NextDouble() { Calls++; return base.NextDouble(); }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Foreign_double_draws_keep_precision_and_primary_recorder_order(bool half)
    {
        if (half) CheckMixedDraws<Half>();
        else CheckMixedDraws<float>();
    }

    private static void CheckMixedDraws<T>()
    {
        AiDotNetEngine.Current = new CpuEngine();
        var ops = MathHelper.GetNumericOperations<T>();
        var random = new CountingRandom(19);
        var reference = new Random(19);
        var plan = new object();
        using var recording = CompiledStepRandom<T>.BeginRecording();
        var primary = CompiledStepRandom<T>.Uniform(new[] { 1 }, random);
        var data = CompiledStepRandom<T>.DrawData<double>(new[] { 2 }, t =>
        {
            t[0] = random.NextDouble() + Math.Pow(2, -40);
            t[1] = random.NextDouble();
        });
        Assert.Equal(ops.FromDouble(reference.NextDouble()), primary[0]);
        Assert.Equal(reference.NextDouble() + Math.Pow(2, -40), data[0]);
        Assert.Equal(reference.NextDouble(), data[1]);
        Assert.Equal(3, random.Calls);
        CompiledStepRandom<T>.Attach(plan, recording);
        Assert.Equal(2, CompiledStepRandom<T>.DrawCount(plan));
        for (int step = 0; step < 2; step++)
        {
            CompiledStepRandom<T>.RedrawFor(plan);
            Assert.Equal(ops.FromDouble(reference.NextDouble()), primary[0]);
            Assert.Equal(reference.NextDouble() + Math.Pow(2, -40), data[0]);
            Assert.Equal(reference.NextDouble(), data[1]);
            Assert.Equal(3 * (step + 2), random.Calls);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Suppression_and_skipped_probe_draw_neither_primary_nor_foreign_data(bool suppress)
    {
        var random = new CountingRandom(7);
        using var recording = CompiledStepRandom<float>.BeginRecording();
        using (suppress ? (IDisposable)CompiledStepRandom<float>.SuppressDraws() : CompiledStepRandom<float>.SkipDraws())
        {
            Assert.Equal(new[] { 0f, 0f }, CompiledStepRandom<float>.Uniform(new[] { 2 }, random).ToArray());
            Assert.Equal(new[] { 0.0, 0.0 }, CompiledStepRandom<float>.DrawData<double>(new[] { 2 }, t =>
            {
                t[0] = random.NextDouble(); t[1] = random.NextDouble();
            }).ToArray());
        }
        Assert.Equal(0, random.Calls);
        Assert.Empty(recording.Draws);
    }

    [Fact]
    public void Captured_mixed_draws_retrace_without_consuming_the_generator_twice()
    {
        AiDotNetEngine.Current = new CpuEngine();
        var random = new CountingRandom(29);
        float expectedPrimary;
        double expectedData;
        var capture = CompiledStepRandom<float>.CaptureDraws();
        using (capture)
        {
            expectedPrimary = CompiledStepRandom<float>.Uniform(new[] { 1 }, random)[0];
            expectedData = CompiledStepRandom<float>.DrawData<double>(new[] { 1 }, t => t[0] = random.NextDouble())[0];
        }
        Assert.Equal(2, random.Calls);
        using var recording = CompiledStepRandom<float>.BeginRecording();
        using (CompiledStepRandom<float>.ReplayDraws(capture))
        {
            Assert.Equal(expectedPrimary, CompiledStepRandom<float>.Uniform(new[] { 1 }, random)[0]);
            Assert.Equal(expectedData, CompiledStepRandom<float>.DrawData<double>(new[] { 1 }, t => t[0] = random.NextDouble())[0]);
        }
        Assert.Equal(2, random.Calls);
        var plan = new object();
        CompiledStepRandom<float>.Attach(plan, recording);
        Assert.Equal(2, CompiledStepRandom<float>.DrawCount(plan));
        CompiledStepRandom<float>.RedrawFor(plan);
        Assert.Equal(4, random.Calls);
    }

    [Fact]
    public void Replay_does_not_copy_a_same_shape_payload_of_the_wrong_element_type()
    {
        var random = new CountingRandom(31);
        var reference = new Random(31);
        var capture = CompiledStepRandom<float>.CaptureDraws();
        float first;
        using (capture) first = CompiledStepRandom<float>.Uniform(new[] { 1 }, random)[0];
        Assert.Equal((float)reference.NextDouble(), first);
        using (CompiledStepRandom<float>.ReplayDraws(capture))
        {
            var foreign = CompiledStepRandom<float>.DrawData<double>(new[] { 1 }, t => t[0] = random.NextDouble());
            Assert.Equal(reference.NextDouble(), foreign[0]);
            Assert.Equal(first, CompiledStepRandom<float>.Uniform(new[] { 1 }, random)[0]);
        }
        Assert.Equal(2, random.Calls);
    }
}

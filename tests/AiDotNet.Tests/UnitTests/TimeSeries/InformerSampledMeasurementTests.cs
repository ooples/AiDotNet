using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.TimeSeries;
using Xunit;

namespace AiDotNet.Tests.UnitTests.TimeSeries;

public sealed class InformerSampledMeasurementTests
{
    [Fact]
    public void SampledMeasurementPreservesRepeatedKeysWithoutQueryByKeyAllocations()
    {
        var engine = new CpuEngine();
        var net = Create(engine);
        const int bh = 2, queries = 64, keys = 8192, width = 4, samples = 3;
        var q = new Tensor<double>(new[] { bh, queries, width });
        var k = new Tensor<double>(new[] { bh, keys, width });
        var indices = new Tensor<int>(new[] { bh, queries, samples });
        for (int i = 0; i < q.Length; i++) q[i] = Math.Sin(.3 * i);
        for (int i = 0; i < k.Length; i++) k[i] = Math.Cos(.7 * i);
        for (int b = 0; b < bh; b++)
            for (int i = 0; i < queries; i++)
                for (int s = 0; s < samples; s++) indices[b, i, s] = b * keys + (s == 2 ? keys - 1 : 0);
#if NET8_0_OR_GREATER
        long allocated = GC.GetAllocatedBytesForCurrentThread();
#endif
        var actual = net.SampledQueryMeasurement(q, k, indices);
#if NET8_0_OR_GREATER
        allocated = GC.GetAllocatedBytesForCurrentThread() - allocated;
        Assert.True(allocated < 4_000_000, $"Sampled measurement allocated {allocated} bytes for 384 sampled pairs.");
#endif
        for (int b = 0; b < bh; b++)
            for (int i = 0; i < queries; i++)
            {
                double sum = 0, maximum = double.NegativeInfinity;
                for (int s = 0; s < samples; s++)
                {
                    double dot = 0;
                    int j = indices[b, i, s] - b * keys;
                    for (int d = 0; d < width; d++) dot += q[b, i, d] * k[b, j, d];
                    sum += dot;
                    maximum = Math.Max(maximum, dot);
                }
                Assert.InRange(Math.Abs(actual[b, i] - (maximum - sum / keys)), 0, 1e-12);
            }
    }

    [Theory]
    [InlineData(12, 4)]
    [InlineData(37, 3)]
    public void SampledMeasurementUsesTheSameDrawsOnEveryHead(int keys, int samples)
    {
        var net = Create(new CpuEngine());
        const int heads = 2, queries = 5, width = 8;
        var q = new Tensor<double>(new[] { heads, queries, width });
        var k = new Tensor<double>(new[] { heads, keys, width });
        for (int i = 0; i < q.Length; i++) q[i] = Math.Sin(0.21 * i);
        for (int i = 0; i < k.Length; i++) k[i] = Math.Cos(0.37 * i);
        var draws = new int[queries, samples];
        for (int i = 0; i < queries; i++)
            for (int s = 0; s < samples; s++) draws[i, s] = s < 2 ? keys - 1 : i;
        net.KeySampler = (_, _, _) => draws;

        var actual = net.SampledQueryMeasurement(q, k, samples, new Random(3));
        for (int b = 0; b < heads; b++)
            for (int i = 0; i < queries; i++)
            {
                double sum = 0, maximum = double.NegativeInfinity;
                for (int s = 0; s < samples; s++)
                {
                    double dot = 0;
                    for (int d = 0; d < width; d++) dot += q[b, i, d] * k[b, draws[i, s], d];
                    sum += dot;
                    maximum = Math.Max(maximum, dot);
                }
                Assert.InRange(Math.Abs(actual[b, i] - (maximum - sum / keys)), 0, 1e-12);
            }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CompiledMeasurementReplaysCurrentKeysAndSampleIndices(bool scoreTable)
    {
        var engine = new CpuEngine();
        var net = Create(engine);
        var q = new Tensor<double>(new[] { 2.0, 1.0 }, new[] { 1, 2, 1 });
        var k = new Tensor<double>(new[] { 1.0, 3.0, -1.0 }, new[] { 1, 3, 1 });
        int secondRow = scoreTable ? 3 : 0;
        var indices = new Tensor<int>(new[] { 0, 0, secondRow + 1, secondRow + 2 }, new[] { 1, 2, 2 });
        var parameter = new Tensor<double>(new[] { 1.0, 1.0 }, new[] { 1, 2 });
        CompiledTrainingPlan<double> plan;
        Tensor<double> measurement;
        using (var scope = GraphMode.EnableTraining(new[] { parameter }))
        {
            using (GradientTape<double>.NoGrad()) measurement = net.SampledQueryMeasurement(q, k, indices, scoreTable);
            var loss = engine.ReduceSum(engine.TensorMultiply(parameter, measurement), null, false);
            Assert.Null(scope.HostReadOp);
            plan = scope.CompileTraining(new[] { parameter }, loss);
        }
        using (plan)
        {
            plan.Step();
            Assert.Equal(new[] { 2.0 / 3, 7.0 / 3 }, measurement.ToArray(), new DoubleComparer());
            indices[0, 0, 1] = 1;
            k[0, 1, 0] = 6;
            plan.Step();
            Assert.Equal(new[] { 22.0 / 3, 13.0 / 3 }, measurement.ToArray(), new DoubleComparer());
        }
    }

    private static InformerNetwork<double> Create(IEngine engine) => new(engine, new List<LayerBase<double>>(),
        new InformerConfiguration(1, 1, 1, 4, 1, 1, 1, 8, 1, 0, false, false,
            InformerAttention.ProbSparse, false, 4, 1, 1));

    private sealed class DoubleComparer : IEqualityComparer<double>
    {
        public bool Equals(double x, double y) => Math.Abs(x - y) < 1e-12;
        public int GetHashCode(double value) => 0;
    }
}

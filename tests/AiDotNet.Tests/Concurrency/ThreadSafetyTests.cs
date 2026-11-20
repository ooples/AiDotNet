using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tests.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.Concurrency;

[Collection("EngineCurrentGlobalState")]
[Trait("Category", "GPU")]
[Trait("GpuContract", "Required")]
public class ThreadSafetyTests
{
    private const int Workers = 16;
    private readonly ITestOutputHelper _output;
    public ThreadSafetyTests(ITestOutputHelper output) => _output = output;

    private static async Task Concurrent(Action<int> operation, int repetitions = 8)
    {
        int completed = 0;
        using var start = new ManualResetEventSlim(false);
        var tasks = Enumerable.Range(0, Workers).Select(worker => Task.Run(() =>
        {
            start.Wait();
            for (int iteration = 0; iteration < repetitions; iteration++)
            {
                operation(worker * repetitions + iteration);
                Interlocked.Increment(ref completed);
            }
        })).ToArray();
        start.Set();
        await Task.WhenAll(tasks);
        Assert.Equal(Workers * repetitions, completed);
    }

    [Fact(Timeout = 120000)]
    public async Task VectorOperations_ConcurrentThreads_MatchIndependentReference()
    {
        using var gpu = new RequiredGpuTestContract(_output);
        await Concurrent(seed => RequiredGpuTestContract.CheckAddition(gpu.Dispatch, seed));
        gpu.AssertExecuted();
    }

    [Fact(Timeout = 120000)]
    public async Task MatrixMultiply_ConcurrentThreads_MatchIndependentReference()
    {
        using var gpu = new RequiredGpuTestContract(_output);
        await Concurrent(seed => RequiredGpuTestContract.CheckMatrix(gpu.Dispatch, seed));
        gpu.AssertExecuted();
    }

    [Fact(Timeout = 120000)]
    public async Task MixedOperations_ConcurrentThreads_KeepResultsIndependent()
    {
        using var gpu = new RequiredGpuTestContract(_output);
        await Concurrent(seed =>
        {
            if ((seed & 1) == 0) RequiredGpuTestContract.CheckMatrix(gpu.Dispatch, seed);
            else RequiredGpuTestContract.CheckAddition(gpu.Dispatch, seed);
        });
        gpu.AssertExecuted();
    }

    [Fact(Timeout = 120000)]
    public async Task Conv2D_ConcurrentThreads_MatchIndependentReference()
    {
        using var gpu = new RequiredGpuTestContract(_output);
        await Concurrent(seed =>
        {
            var input = new Tensor<float>(new[] { 1, 1, 8, 8 });
            var kernel = new Tensor<float>(new[] { 1, 1, 3, 3 });
            var inputValues = new float[input.Length];
            var kernelValues = new float[kernel.Length];
            for (int i = 0; i < input.Length; i++) input[i] = inputValues[i] = ((i * i % 97) + seed * (i % 7 + 1)) * 0.125f;
            for (int i = 0; i < kernel.Length; i++) kernel[i] = kernelValues[i] = (i + 1) * 0.0625f;
            var result = gpu.Dispatch.Conv2D(input, kernel, stride: 1, padding: 0, dilation: 1);
            Assert.True(result.HasPendingGpuData, "Conv2D returned a CPU result in the required GPU lane.");
            Assert.Equal(new[] { 1, 1, 6, 6 }, result.Shape.ToArray());
            for (int i = 0; i < input.Length; i++) Assert.Equal(inputValues[i], input[i]);
            for (int i = 0; i < kernel.Length; i++) Assert.Equal(kernelValues[i], kernel[i]);
            for (int row = 0; row < 6; row++)
            for (int col = 0; col < 6; col++)
            {
                float expected = 0;
                for (int y = 0; y < 3; y++)
                for (int x = 0; x < 3; x++) expected += inputValues[(row + y) * 8 + col + x] * kernelValues[y * 3 + x];
                Assert.Equal(expected, result[row * 6 + col], precision: 4);
            }
        }, repetitions: 4);
        gpu.AssertExecuted();
    }

    [Fact(Timeout = 120000)]
    public async Task Pooling_ConcurrentThreads_MatchIndependentReference()
    {
        using var gpu = new RequiredGpuTestContract(_output);
        await Concurrent(seed =>
        {
            var input = new Tensor<float>(new[] { 1, 1, 8, 8 });
            for (int i = 0; i < input.Length; i++) input[i] = i - seed;
            var max = gpu.Dispatch.MaxPool2D(input, poolSize: 2, stride: 2, padding: 0);
            Assert.True(max.HasPendingGpuData, "MaxPool2D returned a CPU result in the required GPU lane.");
            var avg = gpu.Dispatch.AvgPool2D(input, poolSize: 2, stride: 2, padding: 0);
            Assert.True(avg.HasPendingGpuData, "AvgPool2D returned a CPU result in the required GPU lane.");
            Assert.Equal(new[] { 1, 1, 4, 4 }, max.Shape.ToArray());
            Assert.Equal(new[] { 1, 1, 4, 4 }, avg.Shape.ToArray());
            for (int i = 0; i < input.Length; i++) Assert.Equal((float)(i - seed), input[i]);
            for (int row = 0; row < 4; row++)
            for (int col = 0; col < 4; col++)
            {
                int origin = row * 16 + col * 2;
                Assert.Equal((float)(origin + 9 - seed), max[row * 4 + col]);
                Assert.Equal(origin + 4.5f - seed, avg[row * 4 + col]);
            }
        }, repetitions: 4);
        gpu.AssertExecuted();
    }

    [Fact(Timeout = 120000)]
    public async Task CapabilityReads_ConcurrentWithOperations_RemainConsistent()
    {
        using var gpu = new RequiredGpuTestContract(_output);
        string name = gpu.Engine.Name;
        await Concurrent(seed =>
        {
            Assert.True(gpu.Engine.SupportsGpu);
            Assert.Equal(name, gpu.Engine.Name);
            RequiredGpuTestContract.CheckAddition(gpu.Dispatch, seed);
        });
        gpu.AssertExecuted();
    }

    [Fact(Timeout = 120000)]
    public async Task RepeatedMixedLoad_CompletesEveryOperationWithoutDeadlock()
    {
        using var gpu = new RequiredGpuTestContract(_output);
        await Concurrent(seed =>
        {
            RequiredGpuTestContract.CheckAddition(gpu.Dispatch, seed);
            RequiredGpuTestContract.CheckMatrix(gpu.Dispatch, seed);
        }, repetitions: 16);
        gpu.AssertExecuted();
    }
}

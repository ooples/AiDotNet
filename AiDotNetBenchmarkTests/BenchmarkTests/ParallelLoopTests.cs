using AiDotNet.Tensors.Helpers;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net471, baseline: true)]
[SimpleJob(RuntimeMoniker.Net80)]
public class ParallelLoopTests
{
    [Params(100, 1000, 10000)]
    public int N { get; set; }

    private double[] Array { get; set; }
    private double Factor { get; set; }

    public ParallelLoopTests()
    {
        Array = System.Array.Empty<double>();
    }

    [GlobalSetup]
    public void Setup()
    {
        Array = new double[100];
        var random = RandomHelper.CreateSecureRandom();

        for (var i = 0; i < Array.Length; i++)
        {
            Array[i] = random.NextDouble();
        }
    }

    [Benchmark]
    public void Serial()
    {
        for (var i = 0; i < Array.Length; i++)
        {
            Array[i] = Array[i] * Factor;
        }
    }

    [Benchmark]
    public void ParallelFor()
    {
        Parallel.For(
            0, Array.Length, i => { Array[i] = Array[i] * Factor; });
    }

    [Benchmark]
    public void ParallelForDegreeOfParallelism()
    {
        Parallel.For(
            0, Array.Length, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
            i => { Array[i] = Array[i] * Factor; });
    }

    [Benchmark]
    public void CustomParallel()
    {
        var degreeOfParallelism = Environment.ProcessorCount;
        var tasks = new Task[degreeOfParallelism];

        for (var taskNumber = 0; taskNumber < degreeOfParallelism; taskNumber++)
        {
            var taskNumberCopy = taskNumber;

            tasks[taskNumberCopy] = Task.Factory.StartNew(
                () =>
                {
                    var min = Array.Length * (int)Math.Floor((double)taskNumberCopy / degreeOfParallelism);
                    var max = Array.Length * (int)Math.Ceiling((double)(taskNumberCopy + 1) / degreeOfParallelism);
                    for (var i = min; i < max; i++)
                    {
                        Array[i] *= Factor;
                    }
                });
        }

        Task.WaitAll(tasks);
    }

    [Benchmark]
    public void CustomParallelExtractedMax()
    {
        var degreeOfParallelism = Environment.ProcessorCount;
        var tasks = new Task[degreeOfParallelism];

        for (var taskNumber = 0; taskNumber < degreeOfParallelism; taskNumber++)
        {
            var taskNumberCopy = taskNumber;

            tasks[taskNumberCopy] = Task.Factory.StartNew(
                () =>
                {
                    var min = Array.Length * (int)Math.Floor((double)taskNumberCopy / degreeOfParallelism);
                    var max = Array.Length * (int)Math.Ceiling((double)(taskNumberCopy + 1) / degreeOfParallelism);
                    for (var i = min; i < max; i++)
                    {
                        Array[i] *= Factor;
                    }
                });
        }

        Task.WaitAll(tasks);
    }

    [Benchmark]
    public void CustomParallelExtractedMaxHalfParallelism()
    {
        var degreeOfParallelism = Environment.ProcessorCount / 2;
        var tasks = new Task[degreeOfParallelism];

        for (var taskNumber = 0; taskNumber < degreeOfParallelism; taskNumber++)
        {
            var taskNumberCopy = taskNumber;

            tasks[taskNumberCopy] = Task.Factory.StartNew(
                () =>
                {
                    var min = Array.Length * (int)Math.Floor((double)taskNumberCopy / degreeOfParallelism);
                    var max = Array.Length * (int)Math.Ceiling((double)(taskNumberCopy + 1) / degreeOfParallelism);
                    for (var i = min; i < max; i++)
                    {
                        Array[i] *= Factor;
                    }
                });
        }

        Task.WaitAll(tasks);
    }

    [Benchmark]
    public void CustomParallelFalseSharing()
    {
        var degreeOfParallelism = Environment.ProcessorCount;
        var tasks = new Task[degreeOfParallelism];
        var i = -1;

        for (var taskNumber = 0; taskNumber < degreeOfParallelism; taskNumber++)
        {
            tasks[taskNumber] = Task.Factory.StartNew(
                () =>
                {
                    var j = Interlocked.Increment(ref i);
                    while (j < Array.Length)
                    {
                        Array[j] = Array[j] * Factor;
                        j = Interlocked.Increment(ref i);
                    }
                });
        }

        Task.WaitAll(tasks);
    }
}

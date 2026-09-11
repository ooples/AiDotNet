using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text.Json;
using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.ObjectDetection;
using AiDotNet.Metrics;
using AiDotNet.Tensors.Engines;

// A CPU metric microbenchmark, not a detector/GPU pipeline benchmark. Identical source, inputs,
// warmups and iteration counts are used with each separately retained production assembly.
const int seed = 2154;
const int imageCount = 12;
const int classCount = 4;
const int boxesPerClass = 24;
#if !AP_WORKLOAD_COUNTER
const int measuredRuns = 9;
#endif
var random = new Random(seed);
var predictions = new List<IReadOnlyList<Detection<double>>>();
var truth = new List<IReadOnlyList<Detection<double>>>();
for (int image = 0; image < imageCount; image++)
{
    var actual = new List<Detection<double>>();
    var predicted = new List<Detection<double>>();
    for (int classId = 0; classId < classCount; classId++)
    {
        for (int box = 0; box < boxesPerClass; box++)
        {
            double x = box % 6 * 12;
            double y = box / 6 * 12;
            actual.Add(new Detection<double>(new BoundingBox<double>(x, y, x + 10, y + 10), classId, 1));
            for (int duplicate = 0; duplicate < 3; duplicate++)
            {
                double width = 4 + random.NextDouble() * 6;
                double height = 4 + random.NextDouble() * 6;
                predicted.Add(new Detection<double>(new BoundingBox<double>(x, y, x + width, y + height), classId, random.NextDouble()));
            }
        }
    }
    truth.Add(actual);
    predictions.Add(predicted);
}

string assembly = typeof(ObjectDetectionMetrics<double>).Assembly.Location;
using var assemblyStream = File.OpenRead(assembly);
string assemblyHash = Convert.ToHexString(SHA256.HashData(assemblyStream));
var metrics = new ObjectDetectionMetrics<double>();
var cases = new[]
{
    (MetricWorkload.SingleMeanAveragePrecision, 1),
    (MetricWorkload.FullPrecisionRecallCurve, 1),
    (MetricWorkload.ThresholdRange, 1),
    (MetricWorkload.ThresholdRange, 10),
    (MetricWorkload.ThresholdRange, 32),
    (MetricWorkload.ThresholdRange, 33),
    (MetricWorkload.ThresholdRange, 65)
};
foreach (var (workload, thresholds) in cases)
{
    double step = thresholds == 10 ? 0.05 : 1.0 / 128;
    double maximum = 0.5 + (thresholds - 1) * step;
    double Score()
    {
        switch (workload)
        {
            case MetricWorkload.SingleMeanAveragePrecision:
                return metrics.MeanAveragePrecision(predictions, truth);
            case MetricWorkload.FullPrecisionRecallCurve:
                var curve = metrics.PrecisionRecallCurve(predictions, truth, 0);
                double checksum = 0;
                for (int point = 0; point < curve.Precision.Length; point++)
                    checksum += curve.Precision[point] + curve.Recall[point];
                return checksum;
            case MetricWorkload.ThresholdRange:
                return metrics.MeanAveragePrecisionRange(predictions, truth, 0.5, maximum, step);
            default:
                throw new ArgumentOutOfRangeException(nameof(workload));
        }
    }
#if AP_WORKLOAD_COUNTER
    WorkloadProbe.IoUCalls = 0;
    double result = Score();
    Console.WriteLine(JsonSerializer.Serialize(new
    {
        instrumentedAssemblyHash = assemblyHash,
        workload = workload.ToString(),
        thresholds,
        minimumIoU = 0.5,
        maximumIoU = maximum,
        iouStep = step,
        score = result,
        iouCalls = WorkloadProbe.IoUCalls
    }));
#else
    double expected = Score();
    var warmupTimer = Stopwatch.StartNew();
    do
    {
        if (Score() != expected) throw new InvalidOperationException("Warmup changed the deterministic metric score.");
    } while (warmupTimer.Elapsed < TimeSpan.FromSeconds(1));

    var times = new double[measuredRuns];
    var allocations = new long[measuredRuns];
    for (int iteration = 0; iteration < measuredRuns; iteration++)
    {
        long allocated = GC.GetAllocatedBytesForCurrentThread();
        long start = Stopwatch.GetTimestamp();
        double actual = Score();
        times[iteration] = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
        allocations[iteration] = GC.GetAllocatedBytesForCurrentThread() - allocated;
        if (actual != expected) throw new InvalidOperationException("Measured run changed the deterministic metric score.");
    }
    Array.Sort(times);
    Array.Sort(allocations);
    Console.WriteLine(JsonSerializer.Serialize(new
    {
        assemblyHash,
        seed,
        imageCount,
        classCount,
        boxesPerClass,
        predictions = imageCount * classCount * boxesPerClass * 3,
        groundTruth = imageCount * classCount * boxesPerClass,
        workload = workload.ToString(),
        thresholds,
        minimumIoU = 0.5,
        maximumIoU = maximum,
        iouStep = step,
        measuredRuns,
        score = expected,
        medianMilliseconds = times[measuredRuns / 2],
        minimumMilliseconds = times[0],
        medianAllocatedBytes = allocations[measuredRuns / 2]
    }));
#endif
}

internal enum MetricWorkload { SingleMeanAveragePrecision, FullPrecisionRecallCurve, ThresholdRange }

internal static class BenchmarkEnvironment
{
    // Select CPU before Main's workload is touched. The engine's static initialization may
    // probe GPUs first; that startup is outside the warmup and measurement intervals.
    [ModuleInitializer]
    internal static void Initialize() => AiDotNetEngine.ResetToCpu();
}

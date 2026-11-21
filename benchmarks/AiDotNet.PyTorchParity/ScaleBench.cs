using System.Collections.Concurrent;
using System.Diagnostics;
using System.IO.MemoryMappedFiles;
using System.Text.Json;
using AiDotNet.Tensors.LinearAlgebra;

// Large-data mode (--data DIR): trains a model for whole epochs over a dataset written by scale/prepare_data.py, streaming
// batches from the memory-mapped files, and scores the held-out split after every epoch. pytorch/scale_data.py is the
// PyTorch twin: the same files, the same shuffled order (ScaleShuffle), the same normalisation, the same warm-up and the
// same loss sampling. Each side feeds batches its own fastest way: AiDotNet prefetches one batch ahead on a background
// thread; PyTorch gathers inline into pinned memory, which measured faster for it than a thread or DataLoader workers.

/// <summary>A dataset directory: meta.json plus the four raw uint8 arrays.</summary>
internal sealed class ScaleDataset : IDisposable
{
    private readonly MemoryMappedFile _trainFile;
    private readonly MemoryMappedViewAccessor _train;
    private readonly MemoryMappedFile _testFile;
    private readonly MemoryMappedViewAccessor _test;

    public ScaleDataset(string directory)
    {
        using var meta = JsonDocument.Parse(File.ReadAllText(Path.Combine(directory, "meta.json")));
        var root = meta.RootElement;
        Name = root.GetProperty("name").GetString() ?? throw new InvalidDataException("meta.json has no name");
        SampleShape = root.GetProperty("sample_shape").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        Classes = root.GetProperty("classes").GetInt32();
        TrainSamples = root.GetProperty("train_samples").GetInt32();
        TestSamples = root.GetProperty("test_samples").GetInt32();
        Mean = root.GetProperty("mean").EnumerateArray().Select(e => (float)e.GetDouble()).ToArray();
        Std = root.GetProperty("std").EnumerateArray().Select(e => (float)e.GetDouble()).ToArray();
        TrainSha256 = root.GetProperty("files").GetProperty("train-x").GetProperty("sha256").GetString() ?? "";
        SampleSize = SampleShape.Aggregate(1, (a, b) => a * b);
        if (SampleSize % Mean.Length != 0 || Mean.Length != Std.Length)
        {
            throw new InvalidDataException($"{Name}: {Mean.Length} channel statistics do not divide a {SampleSize}-byte sample");
        }

        (_trainFile, _train) = Map(Path.Combine(directory, "train-x.u8"), (long)TrainSamples * SampleSize);
        (_testFile, _test) = Map(Path.Combine(directory, "test-x.u8"), (long)TestSamples * SampleSize);
        TrainLabels = ReadLabels(Path.Combine(directory, "train-y.u8"), TrainSamples);
        TestLabels = ReadLabels(Path.Combine(directory, "test-y.u8"), TestSamples);

        // (b / 255 - mean[c]) / std[c] in float32, the operation order the PyTorch twin applies on its device.
        Lookup = new float[Mean.Length][];
        for (var c = 0; c < Mean.Length; c++)
        {
            Lookup[c] = new float[256];
            for (var b = 0; b < 256; b++) Lookup[c][b] = ((float)b / 255f - Mean[c]) / Std[c];
        }
    }

    public string Name { get; }
    public int[] SampleShape { get; }
    public int Classes { get; }
    public int TrainSamples { get; }
    public int TestSamples { get; }
    public float[] Mean { get; }
    public float[] Std { get; }
    public string TrainSha256 { get; }
    public int SampleSize { get; }
    public byte[] TrainLabels { get; }
    public byte[] TestLabels { get; }
    private float[][] Lookup { get; }

    private static (MemoryMappedFile, MemoryMappedViewAccessor) Map(string path, long expectedBytes)
    {
        var length = new FileInfo(path).Length;
        if (length != expectedBytes) throw new InvalidDataException($"{path} has {length} bytes, expected {expectedBytes}");
        var file = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        return (file, file.CreateViewAccessor(0, length, MemoryMappedFileAccess.Read));
    }

    private static byte[] ReadLabels(string path, int count)
    {
        var labels = File.ReadAllBytes(path);
        if (labels.Length != count) throw new InvalidDataException($"{path} has {labels.Length} labels, expected {count}");
        return labels;
    }

    /// <summary>Normalised inputs, one-hot labels and class indices for the given samples of one split.</summary>
    public (float[] Inputs, float[] OneHot, byte[] Classes) Gather(bool train, ReadOnlySpan<int> indices, byte[] scratch)
    {
        var view = train ? _train : _test;
        var labels = train ? TrainLabels : TestLabels;
        var inputs = new float[indices.Length * SampleSize];
        var oneHot = new float[indices.Length * Classes];
        var classes = new byte[indices.Length];
        var perChannel = SampleSize / Mean.Length;
        for (var i = 0; i < indices.Length; i++)
        {
            view.ReadArray((long)indices[i] * SampleSize, scratch, 0, SampleSize);
            var offset = i * SampleSize;
            for (var c = 0; c < Mean.Length; c++)
            {
                var lut = Lookup[c];
                var start = c * perChannel;
                for (var k = 0; k < perChannel; k++) inputs[offset + start + k] = lut[scratch[start + k]];
            }
            classes[i] = labels[indices[i]];
            oneHot[i * Classes + classes[i]] = 1f;
        }
        return (inputs, oneHot, classes);
    }

    public void Dispose()
    {
        _train.Dispose();
        _trainFile.Dispose();
        _test.Dispose();
        _testFile.Dispose();
    }
}

/// <summary>
/// The epoch's sample order: a Fisher-Yates shuffle driven by SplitMix64, seeded by (seed, epoch). scale_data.py
/// implements the same generator, so both frameworks visit the samples in the same order.
/// </summary>
internal static class ScaleShuffle
{
    public static int[] Permutation(int count, int seed, int epoch)
    {
        var order = new int[count];
        for (var i = 0; i < count; i++) order[i] = i;
        ulong state = unchecked((ulong)seed * 1_000_003UL + (ulong)epoch);
        for (var i = count - 1; i > 0; i--)
        {
            state = unchecked(state + 0x9E3779B97F4A7C15UL);
            ulong z = state;
            z = unchecked((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL);
            z = unchecked((z ^ (z >> 27)) * 0x94D049BB133111EBUL);
            z ^= z >> 31;
            var j = (int)(z % (ulong)(i + 1));
            (order[i], order[j]) = (order[j], order[i]);
        }
        return order;
    }
}

internal sealed record ScaleEpochReport(int Epoch, double TrainSeconds, double SamplesPerSecond, double DataWaitSeconds,
    double? TrainLoss, double EvalSeconds, double TestAccuracy, double CumulativeTrainSeconds);

internal sealed record ScaleReport(string Dataset, string TrainSha256, int TrainSamples, int TestSamples, int BatchSize,
    int TestBatchSize, int StepsPerEpoch, int WarmupSteps, int LossEvery, double? FirstLoss, List<ScaleEpochReport> Epochs,
    double TotalTrainSeconds, ResourceReport Resources);

internal static class ScaleBench
{
    /// <summary>Training loss is read every LossEvery steps: each read costs both frameworks a device sync.</summary>
    public const int LossEvery = 100;
    public const int TestBatchSize = 250;
    /// <summary>Untimed updates before epoch 0, on its first batches, so JIT and compilation stay out of the epoch times.</summary>
    public const int WarmupSteps = 20;

    public static ScaleReport Train(IBenchmarkModel model, int[] modelShape, ScaleDataset data, int epochs, int batchSize, int seed)
    {
        var modelSize = modelShape.Aggregate(1, (a, b) => a * b);
        if (modelSize != data.SampleSize)
        {
            throw new InvalidDataException($"{data.Name} samples hold {data.SampleSize} values; the model reads {modelSize}");
        }
        if (data.TestSamples % TestBatchSize != 0)
        {
            throw new InvalidDataException($"{data.Name} has {data.TestSamples} test samples, not a multiple of {TestBatchSize}");
        }

        // Whole batches only, so every step has the same shape (the PyTorch twin's compiled graphs need it too).
        var steps = data.TrainSamples / batchSize;
        var inputShape = new[] { batchSize }.Concat(modelShape).ToArray();
        var testShape = new[] { TestBatchSize }.Concat(modelShape).ToArray();
        var reports = new List<ScaleEpochReport>();
        double? firstLoss = null;
        double cumulative = 0;
        var total = Stopwatch.StartNew();
        using var monitor = ResourceMonitor.Start();
        {
            var warmupOrder = ScaleShuffle.Permutation(data.TrainSamples, seed, 0);
            var scratch = new byte[data.SampleSize];
            for (var step = 0; step < Math.Min(WarmupSteps, steps); step++)
            {
                var (inputs, oneHot, _) = data.Gather(true, warmupOrder.AsSpan(step * batchSize, batchSize), scratch);
                model.CurrentBatch = (new Tensor<float>(inputs, inputShape), new Tensor<float>(oneHot, [batchSize, data.Classes]));
                model.Backward();
            }
            var testIndices = Enumerable.Range(0, TestBatchSize).ToArray();
            model.Predict(new Tensor<float>(data.Gather(false, testIndices, scratch).Inputs, testShape));
            BenchStats.SynchronizeDevice();
        }
        for (var epoch = 0; epoch < epochs; epoch++)
        {
            var order = ScaleShuffle.Permutation(data.TrainSamples, seed, epoch);
            using var batches = new BlockingCollection<(float[] Inputs, float[] OneHot)>(boundedCapacity: 2);
            // One batch ahead: the producer gathers and normalises the next batch while the model trains on this one.
            var producer = Task.Run(() =>
            {
                var scratch = new byte[data.SampleSize];
                try
                {
                    for (var step = 0; step < steps; step++)
                    {
                        var (inputs, oneHot, _) = data.Gather(true, order.AsSpan(step * batchSize, batchSize), scratch);
                        batches.Add((inputs, oneHot));
                    }
                }
                finally
                {
                    batches.CompleteAdding();
                }
            });

            var losses = new List<double>();
            double waited = 0;
            var epochTimer = Stopwatch.StartNew();
            for (var step = 0; step < steps; step++)
            {
                var waitTimer = Stopwatch.StartNew();
                var (inputs, oneHot) = batches.Take();
                waited += waitTimer.Elapsed.TotalSeconds;
                model.CurrentBatch = (new Tensor<float>(inputs, inputShape), new Tensor<float>(oneHot, [batchSize, data.Classes]));
                model.Backward();
                if (step % LossEvery == 0 && model.LastLoss is double loss)
                {
                    firstLoss ??= loss;
                    losses.Add(loss);
                }
            }
            BenchStats.SynchronizeDevice();
            epochTimer.Stop();
            producer.GetAwaiter().GetResult();
            var trainSeconds = epochTimer.Elapsed.TotalSeconds;
            cumulative += trainSeconds;

            var evalTimer = Stopwatch.StartNew();
            var accuracy = Evaluate(model, data, testShape);
            evalTimer.Stop();
            var report = new ScaleEpochReport(epoch, Round6(trainSeconds), Round6(steps * (double)batchSize / trainSeconds),
                Round6(waited), losses.Count == 0 ? null : Round6(losses.Average()), Round6(evalTimer.Elapsed.TotalSeconds),
                Round6(accuracy), Round6(cumulative));
            reports.Add(report);
            Console.WriteLine($"[scale] {data.Name} epoch {epoch}: {report.TrainSeconds:F1}s, {report.SamplesPerSecond:F0} samples/s, " +
                              $"loss {report.TrainLoss:F4}, test accuracy {report.TestAccuracy:P2}");
        }
        total.Stop();
        return new ScaleReport(data.Name, data.TrainSha256, data.TrainSamples, data.TestSamples, batchSize, TestBatchSize,
            steps, Math.Min(WarmupSteps, steps), LossEvery, firstLoss, reports, Round6(cumulative), monitor.Summary());
    }

    private static double Evaluate(IBenchmarkModel model, ScaleDataset data, int[] testShape)
    {
        var scratch = new byte[data.SampleSize];
        var indices = new int[TestBatchSize];
        long correct = 0;
        for (var start = 0; start < data.TestSamples; start += TestBatchSize)
        {
            for (var i = 0; i < TestBatchSize; i++) indices[i] = start + i;
            var (inputs, _, classes) = data.Gather(false, indices, scratch);
            var logits = model.Predict(new Tensor<float>(inputs, testShape)).ToArray();
            for (var row = 0; row < TestBatchSize; row++)
            {
                var best = 0;
                for (var c = 1; c < data.Classes; c++)
                {
                    if (logits[row * data.Classes + c] > logits[row * data.Classes + best]) best = c;
                }
                if (best == classes[row]) correct++;
            }
        }
        return correct / (double)data.TestSamples;
    }

    private static double Round6(double value) => Math.Round(value, 6);
}

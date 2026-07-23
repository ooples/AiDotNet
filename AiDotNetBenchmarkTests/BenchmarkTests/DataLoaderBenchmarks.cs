using AiDotNet.Data.Sampling;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Inputs;
using AiDotNet.Optimizers;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Benchmarks for the DataLoader and Batching infrastructure in AiDotNet.
/// Tests batch iteration performance, shuffle overhead, sampler performance,
/// and memory allocation patterns for different batch sizes.
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net471, baseline: true)]
[SimpleJob(RuntimeMoniker.Net80)]
public class DataLoaderBenchmarks
{
    // Test data sizes
    private const int SmallDatasetSize = 100;
    private const int MediumDatasetSize = 1000;
    private const int LargeDatasetSize = 10000;
    private const int NumFeatures = 32;

    // Pre-allocated test data to avoid allocation during benchmarks
    private OptimizationInputData<double, Matrix<double>, Vector<double>> _smallData = new();
    private OptimizationInputData<double, Matrix<double>, Vector<double>> _mediumData = new();
    private OptimizationInputData<double, Matrix<double>, Vector<double>> _largeData = new();

    // Pre-allocated samplers
    private RandomSampler _randomSamplerSmall = new(SmallDatasetSize, seed: 42);
    private RandomSampler _randomSamplerMedium = new(MediumDatasetSize, seed: 42);
    private SequentialSampler _sequentialSamplerMedium = new(MediumDatasetSize);

    // Curriculum learning data
    private double[] _difficultiesMedium = Array.Empty<double>();

    [GlobalSetup]
    public void Setup()
    {
        _smallData = CreateTestData(SmallDatasetSize, NumFeatures);
        _mediumData = CreateTestData(MediumDatasetSize, NumFeatures);
        _largeData = CreateTestData(LargeDatasetSize, NumFeatures);

        _randomSamplerSmall = new RandomSampler(SmallDatasetSize, seed: 42);
        _randomSamplerMedium = new RandomSampler(MediumDatasetSize, seed: 42);
        _sequentialSamplerMedium = new SequentialSampler(MediumDatasetSize);

        // Create difficulty scores for curriculum learning
        _difficultiesMedium = new double[MediumDatasetSize];
        for (int i = 0; i < MediumDatasetSize; i++)
        {
            _difficultiesMedium[i] = i / (double)MediumDatasetSize;
        }
    }

    private static OptimizationInputData<double, Matrix<double>, Vector<double>> CreateTestData(
        int numSamples, int numFeatures)
    {
        var xTrain = new Matrix<double>(numSamples, numFeatures);
        var yTrain = new Vector<double>(numSamples);

        for (int i = 0; i < numSamples; i++)
        {
            for (int j = 0; j < numFeatures; j++)
            {
                xTrain[i, j] = i * numFeatures + j + 1.0;
            }
            yTrain[i] = i * 0.1;
        }

        return new OptimizationInputData<double, Matrix<double>, Vector<double>>
        {
            XTrain = xTrain,
            YTrain = yTrain,
            XValidation = xTrain,
            YValidation = yTrain,
            XTest = xTrain,
            YTest = yTrain
        };
    }

    #region Batcher Creation Benchmarks

    [Benchmark(Baseline = true)]
    public OptimizationDataBatcher<double, Matrix<double>, Vector<double>> CreateBatcher_Small_NoShuffle()
    {
        return new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
            _smallData, batchSize: 32, shuffle: false);
    }

    [Benchmark]
    public OptimizationDataBatcher<double, Matrix<double>, Vector<double>> CreateBatcher_Small_WithShuffle()
    {
        return new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
            _smallData, batchSize: 32, shuffle: true, seed: 42);
    }

    [Benchmark]
    public OptimizationDataBatcher<double, Matrix<double>, Vector<double>> CreateBatcher_Medium_NoShuffle()
    {
        return new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
            _mediumData, batchSize: 32, shuffle: false);
    }

    [Benchmark]
    public OptimizationDataBatcher<double, Matrix<double>, Vector<double>> CreateBatcher_Medium_WithShuffle()
    {
        return new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
            _mediumData, batchSize: 32, shuffle: true, seed: 42);
    }

    #endregion

    #region Batch Iteration Benchmarks

    [Benchmark]
    public int IterateBatches_Small_BatchSize32()
    {
        var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
            _smallData, batchSize: 32, shuffle: false);

        int count = 0;
        foreach (var batch in batcher.GetBatches())
        {
            count += batch.Indices.Length;
        }
        return count;
    }

    [Benchmark]
    public int IterateBatches_Medium_BatchSize32()
    {
        var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
            _mediumData, batchSize: 32, shuffle: false);

        int count = 0;
        foreach (var batch in batcher.GetBatches())
        {
            count += batch.Indices.Length;
        }
        return count;
    }

    [Benchmark]
    public int IterateBatches_Large_BatchSize32()
    {
        var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
            _largeData, batchSize: 32, shuffle: false);

        int count = 0;
        foreach (var batch in batcher.GetBatches())
        {
            count += batch.Indices.Length;
        }
        return count;
    }

    #endregion

    #region Batch Size Comparison Benchmarks

    [Benchmark]
    public int IterateBatches_Medium_BatchSize1()
    {
        var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
            _mediumData, batchSize: 1, shuffle: false);

        int count = 0;
        foreach (var batch in batcher.GetBatches())
        {
            count += batch.Indices.Length;
        }
        return count;
    }

    [Benchmark]
    public int IterateBatches_Medium_BatchSize16()
    {
        var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
            _mediumData, batchSize: 16, shuffle: false);

        int count = 0;
        foreach (var batch in batcher.GetBatches())
        {
            count += batch.Indices.Length;
        }
        return count;
    }

    [Benchmark]
    public int IterateBatches_Medium_BatchSize64()
    {
        var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
            _mediumData, batchSize: 64, shuffle: false);

        int count = 0;
        foreach (var batch in batcher.GetBatches())
        {
            count += batch.Indices.Length;
        }
        return count;
    }

    [Benchmark]
    public int IterateBatches_Medium_BatchSize128()
    {
        var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
            _mediumData, batchSize: 128, shuffle: false);

        int count = 0;
        foreach (var batch in batcher.GetBatches())
        {
            count += batch.Indices.Length;
        }
        return count;
    }

    #endregion

    #region Shuffle vs No-Shuffle Benchmarks

    [Benchmark]
    public int IterateBatches_Medium_NoShuffle()
    {
        var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
            _mediumData, batchSize: 32, shuffle: false);

        int count = 0;
        foreach (var batch in batcher.GetBatches())
        {
            count += batch.Indices.Length;
        }
        return count;
    }

    [Benchmark]
    public int IterateBatches_Medium_WithShuffle()
    {
        var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
            _mediumData, batchSize: 32, shuffle: true, seed: 42);

        int count = 0;
        foreach (var batch in batcher.GetBatches())
        {
            count += batch.Indices.Length;
        }
        return count;
    }

    #endregion

    #region Sampler Benchmarks

    [Benchmark]
    public int[] RandomSampler_GetIndices_Small()
    {
        return _randomSamplerSmall.GetIndices().ToArray();
    }

    [Benchmark]
    public int[] RandomSampler_GetIndices_Medium()
    {
        return _randomSamplerMedium.GetIndices().ToArray();
    }

    [Benchmark]
    public int[] SequentialSampler_GetIndices_Medium()
    {
        return _sequentialSamplerMedium.GetIndices().ToArray();
    }

    [Benchmark]
    public int IterateBatches_WithRandomSampler()
    {
        var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
            _mediumData, batchSize: 32, shuffle: false, sampler: _randomSamplerMedium);

        int count = 0;
        foreach (var batch in batcher.GetBatches())
        {
            count += batch.Indices.Length;
        }
        return count;
    }

    [Benchmark]
    public int IterateBatches_WithSequentialSampler()
    {
        var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
            _mediumData, batchSize: 32, shuffle: false, sampler: _sequentialSamplerMedium);

        int count = 0;
        foreach (var batch in batcher.GetBatches())
        {
            count += batch.Indices.Length;
        }
        return count;
    }

    #endregion

    #region Curriculum Sampler Benchmarks

    [Benchmark]
    public int CurriculumSampler_EarlyEpoch()
    {
        var sampler = new CurriculumSampler<double>(_difficultiesMedium, totalEpochs: 100,
            strategy: CurriculumStrategy.Linear, seed: 42);
        sampler.OnEpochStart(0);

        int count = 0;
        foreach (int idx in sampler.GetIndices())
        {
            count++;
        }
        return count;
    }

    [Benchmark]
    public int CurriculumSampler_MidEpoch()
    {
        var sampler = new CurriculumSampler<double>(_difficultiesMedium, totalEpochs: 100,
            strategy: CurriculumStrategy.Linear, seed: 42);
        sampler.OnEpochStart(50);

        int count = 0;
        foreach (int idx in sampler.GetIndices())
        {
            count++;
        }
        return count;
    }

    [Benchmark]
    public int CurriculumSampler_LateEpoch()
    {
        var sampler = new CurriculumSampler<double>(_difficultiesMedium, totalEpochs: 100,
            strategy: CurriculumStrategy.Linear, seed: 42);
        sampler.OnEpochStart(99);

        int count = 0;
        foreach (int idx in sampler.GetIndices())
        {
            count++;
        }
        return count;
    }

    #endregion

    #region GetBatchIndices (Lightweight) Benchmarks

    [Benchmark]
    public int GetBatchIndices_Medium_BatchSize32()
    {
        var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
            _mediumData, batchSize: 32, shuffle: false);

        int count = 0;
        foreach (var indices in batcher.GetBatchIndices())
        {
            count += indices.Length;
        }
        return count;
    }

    [Benchmark]
    public int GetBatches_VsGetBatchIndices_Comparison()
    {
        // This benchmark compares GetBatches (extracts data) vs GetBatchIndices (just indices)
        var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
            _mediumData, batchSize: 32, shuffle: false);

        int count = 0;
        // Use GetBatchIndices which is lighter weight
        foreach (var indices in batcher.GetBatchIndices())
        {
            count += indices.Length;
        }
        return count;
    }

    #endregion

    #region DropLast Benchmarks

    [Benchmark]
    public int IterateBatches_WithDropLast()
    {
        var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
            _mediumData, batchSize: 32, shuffle: false, dropLast: true);

        int count = 0;
        foreach (var batch in batcher.GetBatches())
        {
            count += batch.Indices.Length;
        }
        return count;
    }

    [Benchmark]
    public int IterateBatches_WithoutDropLast()
    {
        var batcher = new OptimizationDataBatcher<double, Matrix<double>, Vector<double>>(
            _mediumData, batchSize: 32, shuffle: false, dropLast: false);

        int count = 0;
        foreach (var batch in batcher.GetBatches())
        {
            count += batch.Indices.Length;
        }
        return count;
    }

    #endregion

    #region WeightedSampler Benchmarks

    [Benchmark]
    public double[] WeightedSampler_CreateBalancedWeights()
    {
        // Simulate imbalanced dataset: 100 class 0, 900 class 1
        var labels = Enumerable.Repeat(0, 100)
            .Concat(Enumerable.Repeat(1, 900))
            .ToList();

        return WeightedSampler<double>.CreateBalancedWeights(labels, numClasses: 2);
    }

    [Benchmark]
    public int[] WeightedSampler_GetIndices()
    {
        var weights = Enumerable.Range(0, MediumDatasetSize)
            .Select(i => 1.0 + i * 0.001)
            .ToArray();

        var sampler = new WeightedSampler<double>(weights, numSamples: 100, replacement: true, seed: 42);
        return sampler.GetIndices().ToArray();
    }

    #endregion
}

using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ModelCompression;
using AiDotNet.Pruning;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Benchmarks for Model Compression operations.
/// Tests compression/decompression latency and throughput for various techniques.
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net471, baseline: true)]
[SimpleJob(RuntimeMoniker.Net80)]
public class ModelCompressionBenchmarks
{
    [Params(1000, 10000, 100000)]
    public int WeightCount { get; set; }

    private Vector<double> _weights = null!;
    private Matrix<double> _weightMatrix = null!;

    // Compression instances
    private DeepCompression<double> _deepCompression = null!;
    private WeightClusteringCompression<double> _weightClustering = null!;
    private HuffmanEncodingCompression<double> _huffmanCompression = null!;
    private SparsePruningCompression<double> _sparsePruning = null!;

    // Pruning instances
    private MagnitudePruningStrategy<double> _magnitudePruning = null!;
    private GradientPruningStrategy<double> _gradientPruning = null!;
    private StructuredPruningStrategy<double> _structuredPruning = null!;
    private LotteryTicketPruningStrategy<double> _lotteryTicketPruning = null!;

    // Pre-computed results for decompression benchmarks
    private Vector<double> _deepCompressed = null!;
    private ICompressionMetadata<double> _deepMetadata = null!;
    private Vector<double> _clusteringCompressed = null!;
    private ICompressionMetadata<double> _clusteringMetadata = null!;
    private Vector<double> _huffmanCompressed = null!;
    private ICompressionMetadata<double> _huffmanMetadata = null!;

    // Pre-computed importance scores for pruning
    private Matrix<double> _importanceScores = null!;

    [GlobalSetup]
    public void Setup()
    {
        // Using seeded Random for reproducible benchmark data - not used for security purposes
        var random = RandomHelper.CreateSeededRandom(42); // NOSONAR S2245 - benchmarks don't need cryptographic randomness

        // Initialize weight vector with realistic distribution
        _weights = new Vector<double>(WeightCount);
        for (int i = 0; i < WeightCount; i++)
        {
            // Simulate typical neural network weight distribution (most weights near zero)
            double value = random.NextDouble() * 2 - 1; // [-1, 1]
            _weights[i] = value * Math.Exp(-Math.Abs(value)); // Peaked at zero
        }

        // Initialize weight matrix (for structured pruning)
        // Use ceiling division to ensure all weights fit in the matrix
        int rows = (int)Math.Sqrt(WeightCount);
        int cols = (WeightCount + rows - 1) / rows; // Ceiling division: ensures rows * cols >= WeightCount
        _weightMatrix = new Matrix<double>(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                int idx = i * cols + j;
                // Fill with weight data if available, otherwise zero (padding for alignment)
                _weightMatrix[i, j] = idx < WeightCount ? _weights[idx] : 0.0;
            }
        }

        // Initialize compression algorithms
        _deepCompression = new DeepCompression<double>(
            pruningSparsity: 0.5,
            numClusters: 32,
            huffmanPrecision: 4);

        _weightClustering = new WeightClusteringCompression<double>(
            numClusters: 256,
            randomSeed: 42);

        _huffmanCompression = new HuffmanEncodingCompression<double>(precision: 4);

        _sparsePruning = new SparsePruningCompression<double>(sparsityTarget: 0.5);

        // Initialize pruning strategies
        _magnitudePruning = new MagnitudePruningStrategy<double>();
        _gradientPruning = new GradientPruningStrategy<double>();
        _structuredPruning = new StructuredPruningStrategy<double>(
            StructuredPruningStrategy<double>.StructurePruningType.Neuron);
        _lotteryTicketPruning = new LotteryTicketPruningStrategy<double>(iterativeRounds: 3);

        // Pre-compute compressed data for decompression benchmarks
        (_deepCompressed, _deepMetadata) = _deepCompression.Compress(_weights);
        (_clusteringCompressed, _clusteringMetadata) = _weightClustering.Compress(_weights);
        (_huffmanCompressed, _huffmanMetadata) = _huffmanCompression.Compress(_weights);

        // Pre-compute importance scores for pruning benchmarks
        _importanceScores = _magnitudePruning.ComputeImportanceScores(_weightMatrix);
    }

    #region Deep Compression Benchmarks

    [Benchmark(Baseline = true)]
    [BenchmarkCategory("Compression")]
    public (Vector<double>, ICompressionMetadata<double>) DeepCompression_Compress()
    {
        return _deepCompression.Compress(_weights);
    }

    [Benchmark]
    [BenchmarkCategory("Decompression")]
    public Vector<double> DeepCompression_Decompress()
    {
        return _deepCompression.Decompress(_deepCompressed, _deepMetadata);
    }

    [Benchmark]
    [BenchmarkCategory("RoundTrip")]
    public Vector<double> DeepCompression_RoundTrip()
    {
        var (compressed, metadata) = _deepCompression.Compress(_weights);
        return _deepCompression.Decompress(compressed, metadata);
    }

    #endregion

    #region Weight Clustering Benchmarks

    [Benchmark]
    [BenchmarkCategory("Compression")]
    public (Vector<double>, ICompressionMetadata<double>) WeightClustering_Compress()
    {
        return _weightClustering.Compress(_weights);
    }

    [Benchmark]
    [BenchmarkCategory("Decompression")]
    public Vector<double> WeightClustering_Decompress()
    {
        return _weightClustering.Decompress(_clusteringCompressed, _clusteringMetadata);
    }

    #endregion

    #region Huffman Encoding Benchmarks

    [Benchmark]
    [BenchmarkCategory("Compression")]
    public (Vector<double>, ICompressionMetadata<double>) HuffmanEncoding_Compress()
    {
        return _huffmanCompression.Compress(_weights);
    }

    [Benchmark]
    [BenchmarkCategory("Decompression")]
    public Vector<double> HuffmanEncoding_Decompress()
    {
        return _huffmanCompression.Decompress(_huffmanCompressed, _huffmanMetadata);
    }

    #endregion

    #region Sparse Pruning Compression Benchmarks

    [Benchmark]
    [BenchmarkCategory("Compression")]
    public (Vector<double>, ICompressionMetadata<double>) SparsePruning_Compress()
    {
        return _sparsePruning.Compress(_weights);
    }

    #endregion

    #region Pruning Strategy Benchmarks

    [Benchmark]
    [BenchmarkCategory("Pruning")]
    public Matrix<double> MagnitudePruning_ComputeScores()
    {
        return _magnitudePruning.ComputeImportanceScores(_weightMatrix);
    }

    [Benchmark]
    [BenchmarkCategory("Pruning")]
    public IPruningMask<double> MagnitudePruning_CreateMask()
    {
        return _magnitudePruning.CreateMask(_importanceScores, targetSparsity: 0.5);
    }

    [Benchmark]
    [BenchmarkCategory("Pruning")]
    public IPruningMask<double> MagnitudePruning_FullPipeline()
    {
        var scores = _magnitudePruning.ComputeImportanceScores(_weightMatrix);
        return _magnitudePruning.CreateMask(scores, targetSparsity: 0.5);
    }

    [Benchmark]
    [BenchmarkCategory("Pruning")]
    public IPruningMask<double> StructuredPruning_FullPipeline()
    {
        var scores = _structuredPruning.ComputeImportanceScores(_weightMatrix);
        return _structuredPruning.CreateMask(scores, targetSparsity: 0.5);
    }

    [Benchmark]
    [BenchmarkCategory("Pruning")]
    public IPruningMask<double> LotteryTicketPruning_FullPipeline()
    {
        var scores = _lotteryTicketPruning.ComputeImportanceScores(_weightMatrix);
        return _lotteryTicketPruning.CreateMask(scores, targetSparsity: 0.5);
    }

    #endregion

    #region Mask Application Benchmarks

    [Benchmark]
    [BenchmarkCategory("MaskApplication")]
    public Matrix<double> PruningMask_Apply()
    {
        var mask = _magnitudePruning.CreateMask(_importanceScores, targetSparsity: 0.5);
        return mask.Apply(_weightMatrix);
    }

    #endregion

    #region Throughput Benchmarks (Operations Per Second)

    [Benchmark]
    [BenchmarkCategory("Throughput")]
    public int CompressionThroughput_WeightsPerSecond()
    {
        // Compress and return actual compressed length to prevent JIT optimization
        var (compressed, _) = _weightClustering.Compress(_weights);
        return compressed.Length;
    }

    #endregion
}

/// <summary>
/// Benchmarks specifically for structured sparsity patterns (N:M).
/// </summary>
/// <remarks>
/// <para>
/// <b>N:M Sparsity Convention:</b> In this codebase, N:M means "prune N elements per M"
/// (i.e., keep M-N elements per group of M). For example:
/// <list type="bullet">
/// <item>2:4 = prune 2 of every 4 elements (50% sparsity, NVIDIA Ampere compatible)</item>
/// <item>4:8 = prune 4 of every 8 elements (50% sparsity)</item>
/// </list>
/// This is the "N zeros per M" convention, not "N non-zeros per M".
/// </para>
/// </remarks>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net471, baseline: true)]
[SimpleJob(RuntimeMoniker.Net80)]
public class StructuredSparsityBenchmarks
{
    [Params(1024, 4096, 16384)]
    public int TensorSize { get; set; }

    private Tensor<double> _tensor = null!;
    private MagnitudePruningStrategy<double> _strategy = null!;
    private Tensor<double> _scores = null!;

    [GlobalSetup]
    public void Setup()
    {
        // Using seeded Random for reproducible benchmark data - not used for security purposes
        var random = RandomHelper.CreateSeededRandom(42); // NOSONAR S2245 - benchmarks don't need cryptographic randomness
        _tensor = new Tensor<double>(new[] { 1, TensorSize });
        for (int i = 0; i < TensorSize; i++)
            _tensor[0, i] = random.NextDouble();

        _strategy = new MagnitudePruningStrategy<double>();
        _scores = _strategy.ComputeImportanceScores(_tensor);
    }

    [Benchmark(Baseline = true)]
    public IPruningMask<double> StandardPruning_50Percent()
    {
        return _strategy.CreateMask(_scores, targetSparsity: 0.5);
    }

    /// <summary>
    /// Benchmark for 2:4 structured sparsity (prunes 2 of every 4 elements, 50% sparsity).
    /// This pattern is hardware-accelerated on NVIDIA Ampere GPUs.
    /// </summary>
    [Benchmark]
    public IPruningMask<double> StructuredSparsity_Prune2of4()
    {
        return _strategy.Create2to4Mask(_scores);
    }

    /// <summary>
    /// Benchmark for 4:8 structured sparsity (prunes 4 of every 8 elements, 50% sparsity).
    /// </summary>
    [Benchmark]
    public IPruningMask<double> StructuredSparsity_Prune4of8()
    {
        return _strategy.CreateNtoMMask(_scores, n: 4, m: 8);
    }
}

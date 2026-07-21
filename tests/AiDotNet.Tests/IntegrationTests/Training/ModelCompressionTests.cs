using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Interfaces;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Training;

/// <summary>
/// Asserts that ConfigureModelCompressionStrategy runs a real post-build compression audit: it compresses the
/// trained weights, rebuilds the model, re-evaluates it, and surfaces the size-versus-accuracy frontier, the
/// knee, and reconstruction error on AiModelResult — and that leaving it unconfigured leaves the report null.
/// </summary>
public class ModelCompressionTests
{
    /// <summary>
    /// A deterministic lossy strategy: quantizes each weight to a coarse grid (so reconstruction error and
    /// accuracy loss are real) and reports the compressed vector as half the byte size of the original.
    /// </summary>
    private sealed class QuantizeToGridStrategy : IModelCompressionStrategy<double>
    {
        public const double Grid = 1.0;

        public (Vector<double> compressedWeights, ICompressionMetadata<double> metadata) Compress(Vector<double> weights)
        {
            var q = new Vector<double>(weights.Length);
            for (int i = 0; i < weights.Length; i++) q[i] = Math.Round(weights[i] / Grid) * Grid;
            return (q, new GridMetadata());
        }

        public Vector<double> Decompress(Vector<double> compressedWeights, ICompressionMetadata<double> metadata)
            => compressedWeights;

        public double CalculateCompressionRatio(long originalSize, long compressedSize)
            => compressedSize <= 0 ? 0 : originalSize / (double)compressedSize;

        // Pretend the quantized weights pack into 4 bytes each (vs 8 for the double originals) -> ~2x.
        public long GetCompressedSize(Vector<double> compressedWeights, ICompressionMetadata<double> metadata)
            => compressedWeights.Length * 4L;

        public (Matrix<double> compressedWeights, ICompressionMetadata<double> metadata) CompressMatrix(Matrix<double> weights)
            => (weights, new GridMetadata());
        public Matrix<double> DecompressMatrix(Matrix<double> compressedWeights, ICompressionMetadata<double> metadata)
            => compressedWeights;
        public long GetCompressedSize(Matrix<double> compressedWeights, ICompressionMetadata<double> metadata)
            => compressedWeights.Rows * compressedWeights.Columns * 4L;

        public (Tensor<double> compressedWeights, ICompressionMetadata<double> metadata) CompressTensor(Tensor<double> weights)
            => (weights, new GridMetadata());
        public Tensor<double> DecompressTensor(Tensor<double> compressedWeights, ICompressionMetadata<double> metadata)
            => compressedWeights;
        public long GetCompressedSize(Tensor<double> compressedWeights, ICompressionMetadata<double> metadata)
            => compressedWeights.Length * 4L;

        private sealed class GridMetadata : ICompressionMetadata<double>
        {
            public AiDotNet.Enums.CompressionType Type => AiDotNet.Enums.CompressionType.None;
            public int OriginalLength => 0;
            public long GetMetadataSize() => 0;
        }
    }

    private static (Matrix<double> X, Vector<double> Y) BuildData(int rows = 80, int cols = 4)
    {
        var x = new Matrix<double>(rows, cols);
        var y = new Vector<double>(rows);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++) x[i, j] = Math.Sin((i + j) * 0.13) + (i * 0.01);
            y[i] = 1.5 * x[i, 0] - 0.8 * x[i, 2] + 0.3 * x[i, 3];
        }

        return (x, y);
    }

    [Fact(Timeout = 120000)]
    public async Task ConfigureModelCompressionStrategy_SurfacesSizeAccuracyFrontier()
    {
        var (x, y) = BuildData();

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new MultipleRegression<double>())
            .ConfigureModelCompressionStrategy(new QuantizeToGridStrategy())
            .ConfigureDataLoader(new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(x, y))
            .BuildAsync();

        var report = result.ModelCompression;
        Assert.NotNull(report);
        Assert.Equal(nameof(QuantizeToGridStrategy), report?.StrategyName);
        Assert.True((report?.ParameterCount ?? 0) > 0);

        // A real sweep: several magnitude-ranked operating points, fractions strictly increasing.
        Assert.True(report?.SweepAvailable == true);
        Assert.True((report?.Frontier.Count ?? 0) > 1);
        var fractions = report?.Frontier.Select(p => p.Fraction).ToArray() ?? Array.Empty<double>();
        for (int i = 1; i < fractions.Length; i++) Assert.True(fractions[i] > fractions[i - 1]);

        // Compression made the model smaller, and accuracy retention is a real [0,1] measure.
        Assert.True((report?.CompressionRatio ?? 0) > 1.0);
        Assert.True((report?.CompressedSizeBytes ?? long.MaxValue) < (report?.OriginalSizeBytes ?? 0));
        Assert.InRange(report?.AccuracyRetained ?? -1, 0.0, 1.0);
        Assert.True((report?.ReconstructionError ?? -1) >= 0.0);

        // Reconstruction error grows as more (smaller-magnitude-first) weights are compressed.
        var errors = report?.Frontier.Select(p => p.ReconstructionError).ToArray() ?? Array.Empty<double>();
        for (int i = 1; i < errors.Length; i++) Assert.True(errors[i] >= errors[i - 1] - 1e-9);

        // Heavy (coarse-grid) compression genuinely degrades the rebuilt model's fit — the retention is measured
        // by re-evaluation, not assumed, so it drops below 1.0 when the weights are perturbed enough.
        Assert.True((report?.AccuracyRetained ?? 1.0) < 1.0,
            $"Expected coarse-grid compression to reduce accuracy: " +
            $"baseline={report?.BaselineLoss}, compressed={report?.CompressedLoss}, " +
            $"retained={report?.AccuracyRetained}, reconstruction={report?.ReconstructionError}.");

        // The knee is a real frontier point that clears the retention tolerance.
        Assert.Contains(report?.Frontier ?? Enumerable.Empty<AiDotNet.ModelCompression.CompressionFrontierPoint<double>>(),
            p => Math.Abs(p.Fraction - (report?.KneeFraction ?? -1)) < 1e-9);
        Assert.True((report?.KneeAccuracyRetained ?? 0) >= (report?.RetentionTolerance ?? 1.0) - 1e-9);
    }

    [Fact(Timeout = 120000)]
    public async Task NoCompressionStrategy_LeavesReportNull()
    {
        var (x, y) = BuildData();

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new MultipleRegression<double>())
            .ConfigureDataLoader(new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(x, y))
            .BuildAsync();

        Assert.Null(result.ModelCompression);
    }
}

using AiDotNet.Data.Structures;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.MetaLearning.Data;

/// <summary>
/// A synthetic meta-dataset for regression where each task is a sinusoidal function
/// with random amplitude and phase. This is the standard benchmark from the MAML paper
/// (Finn et al., ICML 2017).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <typeparam name="TInput">The input data type (Matrix&lt;T&gt; or Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (Vector&lt;T&gt; or Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This dataset generates sine wave tasks like y = A * sin(x + phase).
/// Each "class" is defined by a unique (amplitude, phase) pair. The meta-learner must
/// learn to quickly fit a sine wave from just a few (x, y) points.
/// </para>
/// </remarks>
public class SineWaveMetaDataset<T, TInput, TOutput> : MetaDatasetBase<T, TInput, TOutput>
{
    private readonly double _amplitudeMin;
    private readonly double _amplitudeMax;
    private readonly double _phaseMin;
    private readonly double _phaseMax;
    private readonly double _xMin;
    private readonly double _xMax;
    private readonly int _numClasses;
    private readonly int _examplesPerClass;

    /// <inheritdoc/>
    public override string Name => "SineWave";

    /// <inheritdoc/>
    public override int TotalClasses => _numClasses;

    /// <inheritdoc/>
    public override int TotalExamples => _numClasses * _examplesPerClass;

    /// <inheritdoc/>
    public override IReadOnlyDictionary<int, int> ClassExampleCounts
    {
        get
        {
            var d = new Dictionary<int, int>();
            for (int i = 0; i < _numClasses; i++) d[i] = _examplesPerClass;
            return d;
        }
    }

    /// <summary>
    /// Creates a sine wave meta-dataset.
    /// </summary>
    /// <param name="numClasses">Number of distinct sine wave tasks (amplitude/phase pairs). Default: 100.</param>
    /// <param name="examplesPerClass">Number of (x, y) samples available per task. Default: 50.</param>
    /// <param name="amplitudeMin">Minimum amplitude. Default: 0.1.</param>
    /// <param name="amplitudeMax">Maximum amplitude. Default: 5.0.</param>
    /// <param name="phaseMin">Minimum phase shift. Default: 0.</param>
    /// <param name="phaseMax">Maximum phase shift. Default: pi.</param>
    /// <param name="xMin">Minimum x value. Default: -5.0.</param>
    /// <param name="xMax">Maximum x value. Default: 5.0.</param>
    /// <param name="seed">Optional random seed.</param>
    public SineWaveMetaDataset(
        int numClasses = 100,
        int examplesPerClass = 50,
        double amplitudeMin = 0.1,
        double amplitudeMax = 5.0,
        double phaseMin = 0.0,
        double phaseMax = Math.PI,
        double xMin = -5.0,
        double xMax = 5.0,
        int? seed = null)
        : base(seed)
    {
        _numClasses = Math.Max(2, numClasses);
        _examplesPerClass = Math.Max(2, examplesPerClass);
        _amplitudeMin = amplitudeMin;
        _amplitudeMax = amplitudeMax;
        _phaseMin = phaseMin;
        _phaseMax = phaseMax;
        _xMin = xMin;
        _xMax = xMax;
    }

    /// <inheritdoc/>
    protected override IMetaLearningTask<T, TInput, TOutput> SampleTaskCore(
        int numWays, int numShots, int numQueryPerClass)
    {
        // For regression, numWays == 1 (single task), but we support multi-way for consistency.
        // Each "way" is a different sine wave; support + query are sampled x-values.
        int totalPerWay = numShots + numQueryPerClass;

        int supportCount = numWays * numShots;
        int queryCount = numWays * numQueryPerClass;

        var supportX = new Matrix<T>(supportCount, 1);
        var supportY = new Vector<T>(supportCount);
        var queryX = new Matrix<T>(queryCount, 1);
        var queryY = new Vector<T>(queryCount);

        int sIdx = 0, qIdx = 0;
        for (int w = 0; w < numWays; w++)
        {
            double amplitude = _amplitudeMin + Rng.NextDouble() * (_amplitudeMax - _amplitudeMin);
            double phase = _phaseMin + Rng.NextDouble() * (_phaseMax - _phaseMin);

            // Generate all x values for this wave and shuffle
            var xValues = new double[totalPerWay];
            for (int i = 0; i < totalPerWay; i++)
            {
                xValues[i] = _xMin + Rng.NextDouble() * (_xMax - _xMin);
            }

            // Support
            for (int i = 0; i < numShots; i++)
            {
                double x = xValues[i];
                double y = amplitude * Math.Sin(x + phase);
                supportX[sIdx, 0] = NumOps.FromDouble(x);
                supportY[sIdx] = NumOps.FromDouble(y);
                sIdx++;
            }

            // Query
            for (int i = numShots; i < totalPerWay; i++)
            {
                double x = xValues[i];
                double y = amplitude * Math.Sin(x + phase);
                queryX[qIdx, 0] = NumOps.FromDouble(x);
                queryY[qIdx] = NumOps.FromDouble(y);
                qIdx++;
            }
        }

        return BuildTask(supportX, supportY, queryX, queryY, numWays, numShots, numQueryPerClass);
    }

    private static MetaLearningTask<T, TInput, TOutput> BuildTask(
        Matrix<T> sX, Vector<T> sY, Matrix<T> qX, Vector<T> qY,
        int numWays, int numShots, int numQueryPerClass)
    {
        return new MetaLearningTask<T, TInput, TOutput>
        {
            SupportSetX = ConvertInput(sX),
            SupportSetY = ConvertOutput(sY),
            QuerySetX = ConvertInput(qX),
            QuerySetY = ConvertOutput(qY),
            NumWays = numWays,
            NumShots = numShots,
            NumQueryPerClass = numQueryPerClass
        };
    }

    private static TInput ConvertInput(Matrix<T> m)
    {
        if (typeof(TInput) == typeof(Matrix<T>)) return (TInput)(object)m;
        if (typeof(TInput) == typeof(Tensor<T>)) return (TInput)(object)Tensor<T>.FromRowMatrix(m);
        throw new NotSupportedException($"Unsupported TInput type: {typeof(TInput).Name}");
    }

    private static TOutput ConvertOutput(Vector<T> v)
    {
        if (typeof(TOutput) == typeof(Vector<T>)) return (TOutput)(object)v;
        if (typeof(TOutput) == typeof(Tensor<T>)) return (TOutput)(object)Tensor<T>.FromVector(v);
        throw new NotSupportedException($"Unsupported TOutput type: {typeof(TOutput).Name}");
    }
}

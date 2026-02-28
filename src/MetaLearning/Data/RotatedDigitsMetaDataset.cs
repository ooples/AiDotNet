using AiDotNet.Data.Structures;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.MetaLearning.Data;

/// <summary>
/// A synthetic meta-dataset for image-like classification where each class is a rotated "digit"
/// pattern (a simple 2D feature vector derived from an angle). Different tasks use different
/// subsets of rotation angles, simulating few-shot image classification benchmarks.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This creates a simplified version of the Omniglot-style benchmark.
/// Each "digit" is represented by a rotation angle that produces a 2D feature vector via
/// (cos(angle), sin(angle)) plus noise. The meta-learner must learn to classify which rotation
/// group a new point belongs to from just a few examples.</para>
/// </remarks>
public class RotatedDigitsMetaDataset<T, TInput, TOutput> : MetaDatasetBase<T, TInput, TOutput>
{
    private readonly int _featureDim;
    private readonly int _numClasses;
    private readonly int _examplesPerClass;
    private readonly double _noiseStdDev;
    private readonly double[] _classAngles;

    /// <inheritdoc/>
    public override string Name => "RotatedDigits";

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
    /// Creates a rotated digits meta-dataset.
    /// </summary>
    /// <param name="numClasses">Total number of rotation classes. Default: 50.</param>
    /// <param name="examplesPerClass">Number of examples per class. Default: 30.</param>
    /// <param name="featureDim">Dimensionality of each example. Minimum 2 (cos/sin pair).
    /// Extra dimensions are harmonic extensions. Default: 8.</param>
    /// <param name="noiseStdDev">Standard deviation of Gaussian noise added to features. Default: 0.1.</param>
    /// <param name="seed">Optional random seed.</param>
    public RotatedDigitsMetaDataset(
        int numClasses = 50,
        int examplesPerClass = 30,
        int featureDim = 8,
        double noiseStdDev = 0.1,
        int? seed = null)
        : base(seed)
    {
        _numClasses = Math.Max(2, numClasses);
        _examplesPerClass = Math.Max(2, examplesPerClass);
        _featureDim = Math.Max(2, featureDim);
        _noiseStdDev = noiseStdDev;

        // Pre-compute class angles spread uniformly over [0, 2*pi)
        _classAngles = new double[_numClasses];
        for (int i = 0; i < _numClasses; i++)
            _classAngles[i] = 2.0 * Math.PI * i / _numClasses;
    }

    /// <inheritdoc/>
    protected override IMetaLearningTask<T, TInput, TOutput> SampleTaskCore(
        int numWays, int numShots, int numQueryPerClass)
    {
        // Select numWays random classes
        int[] selectedClasses = SampleWithoutReplacement(_numClasses, numWays);

        int supportCount = numWays * numShots;
        int queryCount = numWays * numQueryPerClass;

        var supportX = new Matrix<T>(supportCount, _featureDim);
        var supportY = new Vector<T>(supportCount);
        var queryX = new Matrix<T>(queryCount, _featureDim);
        var queryY = new Vector<T>(queryCount);

        int sIdx = 0, qIdx = 0;
        for (int w = 0; w < numWays; w++)
        {
            double baseAngle = _classAngles[selectedClasses[w]];

            // Support examples
            for (int i = 0; i < numShots; i++)
            {
                GenerateExample(supportX, sIdx, baseAngle);
                supportY[sIdx] = NumOps.FromDouble(w);
                sIdx++;
            }

            // Query examples
            for (int i = 0; i < numQueryPerClass; i++)
            {
                GenerateExample(queryX, qIdx, baseAngle);
                queryY[qIdx] = NumOps.FromDouble(w);
                qIdx++;
            }
        }

        return BuildTask(supportX, supportY, queryX, queryY, numWays, numShots, numQueryPerClass);
    }

    /// <summary>
    /// Generates a single feature vector for the given base angle with harmonic extensions and noise.
    /// Features: [cos(angle), sin(angle), cos(2*angle), sin(2*angle), ...]
    /// </summary>
    private void GenerateExample(Matrix<T> matrix, int row, double baseAngle)
    {
        // Add per-example angle jitter (simulating within-class variation)
        double angle = baseAngle + NormalSample() * 0.2;

        for (int d = 0; d < _featureDim; d++)
        {
            int harmonic = d / 2 + 1;
            double value = d % 2 == 0
                ? Math.Cos(harmonic * angle)
                : Math.Sin(harmonic * angle);
            value += NormalSample() * _noiseStdDev;
            matrix[row, d] = NumOps.FromDouble(value);
        }
    }

    private double NormalSample()
    {
        double u1 = 1.0 - Rng.NextDouble();
        double u2 = Rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
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

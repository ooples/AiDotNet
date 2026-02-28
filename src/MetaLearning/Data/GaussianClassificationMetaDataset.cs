using AiDotNet.Data.Structures;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.MetaLearning.Data;

/// <summary>
/// A synthetic meta-dataset for classification where each class is a Gaussian blob in
/// feature space. Each task consists of N classes with randomly sampled means and a
/// shared covariance, making it a standard benchmark for few-shot classification.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <typeparam name="TInput">The input data type (Matrix&lt;T&gt; or Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (Vector&lt;T&gt; or Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This dataset creates classification tasks where each class is a
/// cluster of points centered around a random location. The meta-learner must learn to
/// quickly identify which cluster a new point belongs to, even with very few examples.
/// </para>
/// </remarks>
public class GaussianClassificationMetaDataset<T, TInput, TOutput> : MetaDatasetBase<T, TInput, TOutput>
{
    private readonly int _featureDim;
    private readonly double _classSeparation;
    private readonly double _clusterStdDev;
    private readonly int _numClasses;
    private readonly int _examplesPerClass;

    /// <inheritdoc/>
    public override string Name => "GaussianClassification";

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
    /// Creates a Gaussian classification meta-dataset.
    /// </summary>
    /// <param name="numClasses">Total number of distinct classes. Default: 100.</param>
    /// <param name="examplesPerClass">Samples available per class. Default: 50.</param>
    /// <param name="featureDim">Dimensionality of each feature vector. Default: 20.</param>
    /// <param name="classSeparation">Scale of the class mean positions. Default: 3.0.</param>
    /// <param name="clusterStdDev">Standard deviation within each cluster. Default: 0.5.</param>
    /// <param name="seed">Optional random seed.</param>
    public GaussianClassificationMetaDataset(
        int numClasses = 100,
        int examplesPerClass = 50,
        int featureDim = 20,
        double classSeparation = 3.0,
        double clusterStdDev = 0.5,
        int? seed = null)
        : base(seed)
    {
        _numClasses = Math.Max(2, numClasses);
        _examplesPerClass = Math.Max(2, examplesPerClass);
        _featureDim = Math.Max(1, featureDim);
        _classSeparation = classSeparation;
        _clusterStdDev = clusterStdDev;
    }

    /// <inheritdoc/>
    protected override IMetaLearningTask<T, TInput, TOutput> SampleTaskCore(
        int numWays, int numShots, int numQueryPerClass)
    {
        int totalPerWay = numShots + numQueryPerClass;
        int supportCount = numWays * numShots;
        int queryCount = numWays * numQueryPerClass;

        var supportX = new Matrix<T>(supportCount, _featureDim);
        var supportY = new Vector<T>(supportCount);
        var queryX = new Matrix<T>(queryCount, _featureDim);
        var queryY = new Vector<T>(queryCount);

        int sIdx = 0, qIdx = 0;
        for (int w = 0; w < numWays; w++)
        {
            // Random class mean
            var mean = new double[_featureDim];
            for (int d = 0; d < _featureDim; d++)
            {
                mean[d] = SampleGaussian() * _classSeparation;
            }

            // Generate support examples
            for (int i = 0; i < numShots; i++)
            {
                for (int d = 0; d < _featureDim; d++)
                {
                    double val = mean[d] + SampleGaussian() * _clusterStdDev;
                    supportX[sIdx, d] = NumOps.FromDouble(val);
                }
                supportY[sIdx] = NumOps.FromDouble(w);
                sIdx++;
            }

            // Generate query examples
            for (int i = 0; i < numQueryPerClass; i++)
            {
                for (int d = 0; d < _featureDim; d++)
                {
                    double val = mean[d] + SampleGaussian() * _clusterStdDev;
                    queryX[qIdx, d] = NumOps.FromDouble(val);
                }
                queryY[qIdx] = NumOps.FromDouble(w);
                qIdx++;
            }
        }

        return new MetaLearningTask<T, TInput, TOutput>
        {
            SupportSetX = ConvertInput(supportX),
            SupportSetY = ConvertOutput(supportY),
            QuerySetX = ConvertInput(queryX),
            QuerySetY = ConvertOutput(queryY),
            NumWays = numWays,
            NumShots = numShots,
            NumQueryPerClass = numQueryPerClass
        };
    }

    /// <summary>
    /// Samples from a standard normal distribution using Box-Muller transform.
    /// </summary>
    private double SampleGaussian()
    {
        double u1 = 1.0 - Rng.NextDouble();
        double u2 = Rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
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

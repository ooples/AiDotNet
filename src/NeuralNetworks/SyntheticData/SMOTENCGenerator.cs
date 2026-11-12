using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// SMOTE-NC generator that creates synthetic minority samples by interpolating between
/// existing minority samples and their k-nearest neighbors, supporting both continuous
/// and categorical features.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SMOTE-NC (Nominal and Continuous) operates as follows:
/// 1. Extract all minority class samples from the training data
/// 2. For each minority sample, find its k nearest neighbors among other minority samples
/// 3. Generate synthetic samples by interpolating between samples and randomly chosen neighbors
/// </para>
/// <para>
/// <b>For Beginners:</b> SMOTE-NC generates new minority samples by "mixing" existing ones:
///
/// Given a minority sample and a nearby neighbor:
/// - For numbers (age, income): pick a random point between the two values
///   Example: if sample has age=30 and neighbor has age=40, synthetic might have age=35
/// - For categories (gender, region): use the most common value among the k neighbors
///   Example: if 3 of 5 neighbors have region="East", synthetic gets region="East"
///
/// This is simpler and faster than GANs, and works well for structured tabular data.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.SyntheticDataGenerator)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("SMOTE: Synthetic Minority Over-sampling Technique",
    "https://arxiv.org/abs/1106.1813",
    Year = 2002,
    Authors = "Nitesh V. Chawla, Kevin W. Bowyer, Lawrence O. Hall, W. Philip Kegelmeyer")]
public class SMOTENCGenerator<T> : SyntheticTabularGeneratorBase<T>
{
    private readonly SMOTENCOptions<T> _options;

    // Stored minority class samples from the original data
    [AiDotNet.Attributes.FittedParameter]
    private Matrix<T>? _minoritySamples;

    // Column metadata (numerical vs categorical)
    private IReadOnlyList<ColumnMetadata>? _columns;

    // Median standard deviation of continuous features (used for SMOTE-NC distance metric)
    private T _medianStdContinuous = NumOps.One;

    // The minority samples as a tensor, and 1/0 masks of the continuous and categorical columns ([1, features]),
    // for the distance computation.
    private Tensor<T>? _minorityTensor;
    private Tensor<T>? _continuousMask;
    private Tensor<T>? _categoricalMask;

    /// <summary>The median standard deviation of the continuous features fitted for the SMOTE-NC metric.</summary>
    internal T MedianStdContinuous => _medianStdContinuous;

    private int _numFeatures;

    /// <summary>
    /// Initializes a new instance of the <see cref="SMOTENCGenerator{T}"/> class.
    /// </summary>
    /// <param name="options">Configuration options for the SMOTE-NC model.</param>
    public SMOTENCGenerator(SMOTENCOptions<T> options) : base(options.Seed)
    {
        _options = options;
    }

    /// <inheritdoc />
    protected override void FitInternal(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs)
    {
        _numFeatures = data.Columns;
        _columns = columns;

        // Extract minority samples
        int labelCol = _options.LabelColumnIndex;
        if (labelCol < 0 || labelCol >= _numFeatures) labelCol = _numFeatures - 1;

        var engine = AiDotNetEngine.Current;
        var rows = Tensor<T>.FromMatrix(data);
        // Rows whose label rounds (half to even) to the minority class value.
        var isMinority = engine.TensorEquals(
            engine.TensorRound(engine.TensorSlice(rows, [0, labelCol], [data.Rows, 1])),
            NumOps.FromDouble(_options.MinorityClassValue)).Cast<int>().ToArray();
        var minorityRows = new List<int>();
        for (int i = 0; i < isMinority.Length; i++)
        {
            if (isMinority[i] != 0)
                minorityRows.Add(i);
        }

        // If no minority samples found, use all data
        _minorityTensor = minorityRows.Count == 0
            ? rows
            : engine.TensorIndexSelect(rows, new Tensor<int>(minorityRows.ToArray(), [minorityRows.Count]), 0);
        _minoritySamples = _minorityTensor.ToMatrix();

        var continuous = new int[_numFeatures];
        var categorical = new int[_numFeatures];
        for (int j = 0; j < _numFeatures; j++)
        {
            if (_columns[j].IsCategorical) categorical[j] = 1;
            else continuous[j] = 1;
        }
        _continuousMask = new Tensor<int>(continuous, [1, _numFeatures]).Cast<T>();
        _categoricalMask = new Tensor<int>(categorical, [1, _numFeatures]).Cast<T>();

        // Compute median standard deviation of continuous features for SMOTE-NC distance metric.
        // Per Chawla et al., the categorical mismatch cost should be (medianStd)^2 per mismatch.
        ComputeMedianStdOfContinuousFeatures();
    }

    /// <inheritdoc />
    protected override Matrix<T> GenerateInternal(int numSamples, Vector<T>? conditionColumn, Vector<T>? conditionValue)
    {
        if (_minoritySamples is null || _columns is null)
            throw new InvalidOperationException("Generator is not fitted.");

        int nMinority = _minoritySamples.Rows;
        int k = Math.Min(_options.K, nMinority - 1);
        if (k <= 0) k = 1;

        var result = new Matrix<T>(numSamples, _numFeatures);

        for (int i = 0; i < numSamples; i++)
        {
            // Pick a random minority sample
            int sampleIdx = Random.Next(nMinority);

            // Find k nearest neighbors
            var neighbors = FindKNearestNeighbors(sampleIdx, k);

            // Pick a random neighbor
            int neighborIdx = neighbors[Random.Next(neighbors.Length)];

            // Generate synthetic sample by interpolation
            for (int j = 0; j < _numFeatures; j++)
            {
                if (_columns[j].IsCategorical)
                {
                    // Categorical: majority vote among k nearest neighbors
                    result[i, j] = MajorityVoteAmongNeighbors(j, neighbors);
                }
                else
                {
                    // Continuous: linear interpolation
                    double val1 = NumOps.ToDouble(_minoritySamples[sampleIdx, j]);
                    double val2 = NumOps.ToDouble(_minoritySamples[neighborIdx, j]);
                    double lambda = Random.NextDouble();
                    result[i, j] = NumOps.FromDouble(val1 + lambda * (val2 - val1));
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Finds the k nearest neighbors for a given minority sample using mixed-type distance.
    /// </summary>
    private int[] FindKNearestNeighbors(int sampleIndex, int k)
    {
        if (_minoritySamples is null || _columns is null) return Array.Empty<int>();

        int n = _minoritySamples.Rows;
        var distances = DistancesFrom(sampleIndex);

        // Order the other samples by distance, ties by index (a NaN distance sorts first, as double.CompareTo
        // orders it). A sample's rank is the number of samples strictly closer.
        var engine = AiDotNetEngine.Current;
        var keys = engine.TensorNanToNum(distances, -1.0, double.PositiveInfinity, double.NegativeInfinity);
        var sorted = engine.TensorSort(keys, 0, false).Values;
        var ranks = engine.TensorSearchSorted(sorted, keys, right: false).ToArray();
        var order = new long[n - 1];
        var others = new int[n - 1];
        int count = 0;
        for (int i = 0; i < n; i++)
        {
            if (i == sampleIndex) continue;
            order[count] = (long)ranks[i] * n + i;
            others[count++] = i;
        }
        Array.Sort(order, others);

        int numNeighbors = Math.Min(k, count);
        var result = new int[numNeighbors];
        Array.Copy(others, result, numNeighbors);
        return result;
    }

    /// <summary>
    /// Computes the median standard deviation of continuous features from the minority samples.
    /// This is used in the SMOTE-NC distance metric per Chawla et al.
    /// </summary>
    private void ComputeMedianStdOfContinuousFeatures()
    {
        if (_minorityTensor is null || _columns is null) return;

        var continuousColumns = new List<int>();
        for (int j = 0; j < _numFeatures; j++)
        {
            if (!_columns[j].IsCategorical) continuousColumns.Add(j);
        }
        if (continuousColumns.Count == 0) return;

        // Sample standard deviation (n - 1; 1 for a single row) of each continuous feature, floored at 1e-10.
        var engine = AiDotNetEngine.Current;
        int n = _minorityTensor.Shape[0];
        int c = continuousColumns.Count;
        var x = engine.TensorIndexSelect(_minorityTensor, new Tensor<int>(continuousColumns.ToArray(), [c]), 1);
        Tensor<T> stds;
        if (n > 1)
        {
            var mean = engine.ReduceMean(x, [0], keepDims: true);
            var centered = engine.TensorSubtract(x, engine.TensorTile(mean, [n, 1]));
            stds = engine.TensorSqrt(engine.TensorDivideScalar(
                engine.ReduceSum(engine.TensorMultiply(centered, centered), [0], keepDims: false), NumOps.FromDouble(n - 1)));
        }
        else
        {
            stds = new Tensor<T>([c]);
            stds.Fill(NumOps.One);
        }
        var floor = new Tensor<T>([c]);
        floor.Fill(NumOps.FromDouble(1e-10));
        stds = engine.TensorWhere(engine.TensorGreaterThan(floor, stds), floor, stds);

        // The median (Chawla et al. 2002, section 6.1): the middle value, or the mean of the two middle values
        // when the count is even.
        var sortedStds = engine.TensorSort(stds, 0, false).Values;
        var middle = engine.TensorSlice(sortedStds, [(c - 1) / 2], [2 - c % 2]);
        _medianStdContinuous = engine.TensorMean(middle);
    }

    /// <summary>
    /// SMOTE-NC distances (Chawla et al. 2002, section 6.1) from sample <paramref name="sampleIndex"/> to every
    /// minority sample: squared differences over the continuous features plus (median std)^2 for each categorical
    /// feature whose values differ by more than 0.5, square-rooted.
    /// </summary>
    private Tensor<T> DistancesFrom(int sampleIndex)
    {
        if (_minorityTensor is null || _continuousMask is null || _categoricalMask is null)
            throw new InvalidOperationException("Generator is not fitted.");

        var engine = AiDotNetEngine.Current;
        int n = _minorityTensor.Shape[0];
        var sample = engine.TensorSlice(_minorityTensor, [sampleIndex, 0], [1, _numFeatures]);
        var diff = engine.TensorSubtract(_minorityTensor, engine.TensorTile(sample, [n, 1]));
        var continuousTerm = engine.TensorMultiply(engine.TensorMultiply(diff, diff),
            engine.TensorTile(_continuousMask, [n, 1]));
        var mismatch = engine.TensorMultiply(engine.TensorGreaterThan(engine.TensorAbs(diff), NumOps.FromDouble(0.5)),
            engine.TensorTile(_categoricalMask, [n, 1]));
        var penalty = NumOps.Multiply(_medianStdContinuous, _medianStdContinuous);
        var sumSq = engine.ReduceSum(engine.TensorAdd(continuousTerm, engine.TensorMultiplyScalar(mismatch, penalty)),
            [1], keepDims: false);
        return engine.TensorSqrt(sumSq);
    }

    /// <summary>
    /// Returns the majority vote value for a categorical feature among the given neighbors.
    /// </summary>
    private T MajorityVoteAmongNeighbors(int featureIndex, int[] neighbors)
    {
        if (_minorityTensor is null || neighbors.Length == 0) return NumOps.Zero;

        // The most common rounded value; on a tie, the one met first in neighbor order.
        var engine = AiDotNetEngine.Current;
        int k = neighbors.Length;
        var values = engine.TensorRound(engine.TensorGather(
            engine.TensorSlice(_minorityTensor, [0, featureIndex], [_minorityTensor.Shape[0], 1]).Reshape([_minorityTensor.Shape[0]]),
            new Tensor<int>(neighbors, [k]), 0));
        var same = engine.TensorEquals(
            engine.TensorTile(values.Reshape([k, 1]), [1, k]), engine.TensorTile(values.Reshape([1, k]), [k, 1]));
        var counts = engine.ReduceSum(same, [1], keepDims: false);
        var best = engine.TensorArgMax(counts.Reshape([1, k]), 1);
        return engine.TensorGather(values, best, 0).ToArray()[0];
    }
}

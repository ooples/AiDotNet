using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Classification.MultiLabel;

/// <summary>
/// Implements ML-kNN (Multi-Label k-Nearest Neighbors) for multi-label classification.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> ML-kNN extends k-NN to multi-label problems using Bayesian inference.
/// For each label, it estimates the probability that a sample has the label given how many of its
/// k nearest neighbors have that label.</para>
///
/// <para><b>How it works:</b>
/// <list type="number">
/// <item>Find k nearest neighbors of the query sample</item>
/// <item>Count how many neighbors have each label</item>
/// <item>Use Bayesian inference with prior probabilities from training data</item>
/// <item>Predict label if P(label=1|neighbors) > P(label=0|neighbors)</item>
/// </list>
/// </para>
///
/// <para><b>Key formula:</b>
/// P(H_l | E_l) = P(E_l | H_l) * P(H_l) / P(E_l)
/// where H_l = label l is present, E_l = count of neighbors with label l</para>
///
/// <para><b>Reference:</b> Zhang &amp; Zhou, "ML-KNN: A lazy learning approach to multi-label learning" (2007)</para>
/// </remarks>
/// <example>
/// <code>
/// // Create ML-kNN for multi-label classification with Bayesian inference
/// var options = new MLkNNOptions&lt;double&gt;();
///
/// // Prepare features and multi-label targets
/// var features = new Matrix&lt;double&gt;(4, 2);
/// features[0, 0] = 1.0; features[0, 1] = 2.0;
/// features[1, 0] = 3.0; features[1, 1] = 4.0;
/// features[2, 0] = 5.0; features[2, 1] = 6.0;
/// features[3, 0] = 7.0; features[3, 1] = 8.0;
/// var labels = new Matrix&lt;double&gt;(4, 3);
/// labels[0, 0] = 1; labels[0, 1] = 0; labels[0, 2] = 1;
/// labels[1, 0] = 1; labels[1, 1] = 1; labels[1, 2] = 0;
/// labels[2, 0] = 0; labels[2, 1] = 1; labels[2, 2] = 1;
/// labels[3, 0] = 0; labels[3, 1] = 0; labels[3, 2] = 1;
///
/// // Train by storing instances and computing Bayesian priors
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Matrix&lt;double&gt;&gt;()
///     .ConfigureModel(new MLkNNClassifier&lt;double&gt;(options))
///     .Build(features, labels);
///
/// // Predict labels using k-NN neighbor counts and Bayesian inference
/// var newSample = new Matrix&lt;double&gt;(1, 2);
/// newSample[0, 0] = 2.0; newSample[0, 1] = 3.0;
/// var prediction = result.Predict(newSample);
/// </code>
/// </example>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.InstanceBased)]
[ModelCategory(ModelCategory.Bayesian)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ResearchPaper("ML-KNN: A Lazy Learning Approach to Multi-Label Learning", "https://doi.org/10.1016/j.patcog.2006.12.019", Year = 2007, Authors = "Min-Ling Zhang, Zhi-Hua Zhou")]
public partial class MLkNNClassifier<T> : MultiLabelClassifierBase<T>
{
    private readonly MLkNNOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private readonly Random _random;
    [AiDotNet.Attributes.FittedParameter]
    private Matrix<T> _trainFeatures = new Matrix<T>(0, 0);
    [AiDotNet.Attributes.FittedParameter]
    private Matrix<T> _trainLabels = new Matrix<T>(0, 0);
    private Vector<T>? _priorProbs; // P(H_l = 1)
    private Tensor<T> _condProbsPos = new Tensor<T>(new[] { 0, 0 }); // P(E_l = j | H_l = 1) for j = 0..k, [labels, k + 1]
    private Tensor<T> _condProbsNeg = new Tensor<T>(new[] { 0, 0 }); // P(E_l = j | H_l = 0) for j = 0..k, [labels, k + 1]

    /// <summary>
    /// Maximum query x training x feature elements per exact distance batch.
    /// </summary>
    private const int DistanceBatchElements = 1 << 22;

    /// <summary>
    /// Creates a new ML-kNN classifier.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    public MLkNNClassifier(MLkNNOptions<T>? options = null)
        : base()
    {
        _options = options ?? new MLkNNOptions<T>();
        _random = _options.RandomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.RandomSeed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <inheritdoc />
    protected override void TrainMultiLabelCore(Matrix<T> features, Matrix<T> labels)
    {
        _trainFeatures = features;
        _trainLabels = labels;

        var engine = Engine;
        int n = features.Rows;
        int k = _options.KNeighbors;
        int width = k + 1;
        double s = _options.Smoothing;

        // Neighbor label counts per sample (k nearest neighbors, excluding self) and the sample's own labels
        var neighborCounts = CountNeighborLabels(NearestNeighbors(features, k, excludeSelf: true));
        var hasLabel = ZeroOneToInt(PositiveLabels(labels));

        // C[l, j] = count of samples with (or without) label l having j neighbors with label l, counted
        // in int (sums of T ones stop at 2048 for Half): codes of samples without the label are shifted
        // past the first table so that one bin count fills both tables
        int table = NumLabels * width;
        var countPos = new Tensor<int>(new[] { table });
        var countNeg = new Tensor<int>(new[] { table });
        var totalPos = new Tensor<int>(new[] { NumLabels });
        if (n > 0 && NumLabels > 0)
        {
            var codes = CountCodes(neighborCounts, width).Reshape(n * NumLabels);
            var negativeOffset = engine.TensorMultiplyScalar(engine.ScalarMinusTensor(1, hasLabel.Reshape(n * NumLabels)), table);
            var counts = engine.TensorBinCount(engine.TensorAdd(codes, negativeOffset), 2 * table);
            countPos = counts.Slice(0, 0, table);
            countNeg = counts.Slice(0, table, 2 * table);
            totalPos = engine.ReduceSum(hasLabel, new[] { 0 }, false);
        }

        var totalNeg = engine.ScalarMinusTensor(n, totalPos);

        // Prior and conditional probabilities with smoothing, divided in double so counts above 2048 stay exact
        _priorProbs = engine.TensorDivideScalar(engine.TensorAddScalar(totalPos.Cast<double>(), s), n + 2 * s).Cast<T>().ToVector();
        _condProbsPos = SmoothedConditional(countPos, totalPos, s, width);
        _condProbsNeg = SmoothedConditional(countNeg, totalNeg, s, width);
    }

    /// <inheritdoc />
    public override Matrix<T> PredictMultiLabelProbabilities(Matrix<T> features)
    {
        if (_trainFeatures is null || _priorProbs is null || _condProbsPos is null || _condProbsNeg is null)
        {
            throw new InvalidOperationException("Model must be trained before prediction.");
        }

        int n = features.Rows;
        int k = _options.KNeighbors;
        if (n == 0 || NumLabels == 0)
        {
            return new Matrix<T>(n, NumLabels);
        }

        // Count neighbors with each label among the k nearest training samples
        var engine = Engine;
        var neighborCounts = CountNeighborLabels(NearestNeighbors(features, k, excludeSelf: false));
        var codes = CountCodes(neighborCounts, k + 1).Reshape(n * NumLabels);

        // Bayesian inference: P(H_l=1|E_l=j) = P(E_l=j|H_l=1)*P(H_l=1) / P(E_l=j)
        var prior = _priorProbs.AsTensor().Reshape(1, NumLabels);
        var condPos = engine.TensorGather(_condProbsPos.Reshape(_condProbsPos.Length), codes, 0).Reshape(n, NumLabels);
        var condNeg = engine.TensorGather(_condProbsNeg.Reshape(_condProbsNeg.Length), codes, 0).Reshape(n, NumLabels);
        var pPos = engine.TensorMultiply(condPos, prior);
        var pNeg = engine.TensorMultiply(condNeg, engine.ScalarMinusTensor(NumOps.One, prior));
        var denominator = engine.TensorAddScalar(engine.TensorAdd(pPos, pNeg), Tolerance(1e-10));
        return engine.TensorDivide(pPos, denominator).ToMatrix();
    }

    /// <summary>
    /// (count + s) / (total + s * width) for a flattened [labels * width] count table and per-label totals.
    /// </summary>
    private Tensor<T> SmoothedConditional(Tensor<int> counts, Tensor<int> totals, double s, int width)
    {
        var engine = Engine;
        // Contiguous first: Tensor.Cast reads a sliced view from the start of its storage, so the "without the
        // label" table (the second half of the shared bin count) came out as a copy of the "with the label" one
        var numerator = engine.TensorAddScalar(counts.Contiguous().Cast<double>().Reshape(NumLabels, width), s);
        var denominator = engine.TensorAddScalar(totals.Contiguous().Cast<double>(), s * width).Reshape(NumLabels, 1);
        return engine.TensorDivide(numerator, engine.TensorBroadcastTo(denominator, new[] { NumLabels, width })).Cast<T>();
    }

    /// <summary>
    /// 1 where a label value is above 0.5, otherwise 0, as a [rows, NumLabels] tensor.
    /// </summary>
    private Tensor<T> PositiveLabels(Matrix<T> labels)
    {
        if (labels.Rows == 0 || NumLabels == 0)
        {
            return new Tensor<T>(new[] { labels.Rows, NumLabels });
        }

        var values = Tensor<T>.FromMatrix(labels);
        if (labels.Columns != NumLabels)
        {
            values = values.Slice(1, 0, NumLabels);
        }

        return Engine.TensorGreaterThan(values, NumOps.FromDouble(0.5));
    }

    /// <summary>
    /// Number of neighbors having each label: [queries, NumLabels] integer counts.
    /// </summary>
    private Tensor<int> CountNeighborLabels(int[][] neighbors)
    {
        int queries = neighbors.Length;
        int perQuery = queries == 0 ? 0 : neighbors[0].Length;
        if (queries == 0 || perQuery == 0 || NumLabels == 0)
        {
            return new Tensor<int>(new[] { queries, NumLabels });
        }

        var flat = new int[queries * perQuery];
        for (int q = 0; q < queries; q++)
        {
            Array.Copy(neighbors[q], 0, flat, q * perQuery, perQuery);
        }

        var trainHasLabel = ZeroOneToInt(PositiveLabels(_trainLabels));
        var gathered = Engine.TensorGather(trainHasLabel, new Tensor<int>(flat, new[] { flat.Length }), 0);
        return Engine.ReduceSum(gathered.Reshape(queries, perQuery, NumLabels), new[] { 1 }, false);
    }

    /// <summary>
    /// Flattened table codes l * width + count for each [query, label] neighbor count.
    /// </summary>
    private Tensor<int> CountCodes(Tensor<int> neighborCounts, int width)
    {
        var labelOffsets = Engine.TensorMultiplyScalar(Engine.TensorArange<int>(0, NumLabels), width).Reshape(1, NumLabels);
        return Engine.TensorAdd(neighborCounts, labelOffsets);
    }

    /// <summary>
    /// The k nearest training samples of every query row by Euclidean distance (ties go to the lower
    /// index). With <paramref name="excludeSelf"/>, query i is training sample i and is skipped.
    /// </summary>
    private int[][] NearestNeighbors(Matrix<T> query, int k, bool excludeSelf)
    {
        var engine = Engine;
        int queries = query.Rows;
        int m = _trainFeatures.Rows;
        int d = _trainFeatures.Columns;
        int available = excludeSelf ? Math.Max(0, m - 1) : m;
        int take = Math.Max(0, Math.Min(k, available));
        var result = new int[queries][];
        if (queries == 0 || take == 0)
        {
            for (int q = 0; q < queries; q++)
            {
                result[q] = Array.Empty<int>();
            }

            return result;
        }

        int scan = Math.Min(m, take + (excludeSelf ? 1 : 0));
        var train = Tensor<T>.FromMatrix(_trainFeatures).Reshape(1, m, d);
        var queryTensor = Tensor<T>.FromMatrix(query);
        int batch = Math.Max(1, DistanceBatchElements / Math.Max(1, m * Math.Max(1, d)));

        for (int start = 0; start < queries; start += batch)
        {
            int rows = Math.Min(batch, queries - start);
            var block = queryTensor.Slice(0, start, start + rows).Reshape(rows, 1, d);

            // Exact distances sqrt(sum_c (q_c - t_c)^2), scaled so the squares cannot overflow in Half
            var diff = engine.TensorSubtract(block, train);
            var distances = ClassifierBase<T>.ScaledNorms(diff, 2, 2.0);

            for (int r = 0; r < rows; r++)
            {
                // Stable per-row order (the engine sort is not stable): ties go to the lower index
                int self = start + r;
                var order = AiDotNet.ActiveLearning.ActiveLearningEngineOps.StableOrder(distances.Slice(0, r, r + 1), false);
                var picked = new int[take];
                int count = 0;
                for (int j = 0; j < scan && count < take; j++)
                {
                    int candidate = order[j];
                    if (!excludeSelf || candidate != self)
                    {
                        picked[count++] = candidate;
                    }
                }

                result[self] = picked;
            }
        }

        return result;
    }

    /// <inheritdoc />
    protected override void RegisterComponents()
    {
        base.RegisterComponents();
        RegisterParameterComponent(
            "bayesian-probabilities",
            new AiDotNet.Models.Parameters.VariableLengthParameterSource<T>(
                GetProbabilityParameterCount,
                GetProbabilityParameters,
                RestoreProbabilityParameters),
            AiDotNet.Models.Parameters.ParameterSlotRole.LearnedState);
    }

    private long GetProbabilityParameterCount()
    {
        if (_priorProbs is null) return 0;
        return checked((long)_priorProbs.Length + _condProbsPos.Length + _condProbsNeg.Length);
    }

    private Vector<T> GetProbabilityParameters()
    {
        if (_priorProbs is null) return new Vector<T>(0);

        int labels = _priorProbs.Length;
        int conditionalWidth = _options.KNeighbors + 1;
        if (_condProbsPos.Shape[0] != labels ||
            _condProbsNeg.Shape[0] != labels ||
            _condProbsPos.Shape[1] != conditionalWidth ||
            _condProbsNeg.Shape[1] != conditionalWidth)
        {
            throw new InvalidOperationException(
                "ML-kNN probability arrays do not share the fitted label and neighbor dimensions.");
        }

        // [priors | positive conditionals (row-major) | negative conditionals (row-major)]
        return Engine.TensorConcatenate(new[]
        {
            _priorProbs.AsTensor(),
            _condProbsPos.Reshape(_condProbsPos.Length),
            _condProbsNeg.Reshape(_condProbsNeg.Length)
        }, 0).ToVector();
    }

    private void RestoreProbabilityParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));

        int conditionalWidth = _options.KNeighbors + 1;
        int valuesPerLabel = checked(1 + 2 * conditionalWidth);
        if (parameters.Length % valuesPerLabel != 0)
        {
            throw new ArgumentException(
                $"ML-kNN parameters must contain {valuesPerLabel} values per label; got " +
                $"{parameters.Length} values.", nameof(parameters));
        }

        int labels = parameters.Length / valuesPerLabel;
        int tableSize = labels * conditionalWidth;
        NumLabels = labels;
        _priorProbs = parameters.GetSubVector(0, labels);
        _condProbsPos = parameters.GetSubVector(labels, tableSize).AsTensor().Reshape(labels, conditionalWidth);
        _condProbsNeg = parameters.GetSubVector(labels + tableSize, tableSize).AsTensor().Reshape(labels, conditionalWidth);
    }

    /// <inheritdoc />
}

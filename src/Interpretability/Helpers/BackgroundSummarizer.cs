using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.Interpretability.Helpers;

/// <summary>
/// Provides methods for summarizing background data for SHAP and other interpretability methods.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SHAP and similar methods need "background data" to compare against.
/// But using all your training data as background can be VERY slow.
///
/// This class provides ways to summarize large datasets into smaller representative sets:
/// - <b>KMeans:</b> Group similar samples together, use cluster centers as representatives
/// - <b>Stratified Sampling:</b> Sample proportionally from different groups/categories
/// - <b>Random Sampling:</b> Simple random subset selection
///
/// <b>Example:</b> If you have 10,000 training samples, you might summarize to 100 samples.
/// SHAP computation goes from O(10000) to O(100) = 100x faster!
///
/// <b>Trade-off:</b>
/// - Fewer background samples = faster but less accurate
/// - More background samples = slower but more accurate
/// - 50-200 samples is usually a good balance
/// </para>
/// </remarks>
public static class BackgroundSummarizer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Summarizes data using KMeans clustering.
    /// </summary>
    /// <param name="data">The full dataset to summarize (rows = samples).</param>
    /// <param name="k">Number of clusters/summary samples to create.</param>
    /// <param name="maxIterations">Maximum KMeans iterations.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <returns>Summary containing cluster centers and their weights.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is like the Python `shap.kmeans(data, k)` function.
    ///
    /// <b>How it works:</b>
    /// 1. Run KMeans clustering to find k clusters
    /// 2. Each cluster center becomes a summary sample
    /// 3. The weight of each center is proportional to cluster size
    ///
    /// <b>Why it works:</b> Cluster centers represent "typical" samples from each
    /// region of your data. More common patterns get higher weights.
    ///
    /// <b>Choosing k:</b>
    /// - k=50 is a good default for most datasets
    /// - k=100 for more accuracy
    /// - k=200 for high-dimensional or complex data
    /// - Never more than your original sample size!
    /// </para>
    /// </remarks>
    public static BackgroundSummary<T> KMeans(
        Matrix<T> data,
        int k = 50,
        int maxIterations = 100,
        int? randomState = null)
    {
        if (data.Rows <= k)
        {
            // Data is smaller than k, return all with equal weights
            return new BackgroundSummary<T>(
                data.Clone(),
                CreateUniformWeights(data.Rows));
        }

        var rand = randomState.HasValue
            ? RandomHelper.CreateSeededRandom(randomState.Value)
            : RandomHelper.CreateSecureRandom();

        int n = data.Rows;
        int d = data.Columns;

        // Initialize cluster centers using KMeans++ initialization
        var engine = AiDotNetEngine.Current;
        var points = Tensor<T>.FromMatrix(data);
        var centers = Tensor<T>.FromMatrix(InitializeCentersKMeansPlusPlus(data, k, rand));
        var assignments = new int[n];
        var clusterSizes = new Tensor<T>(new[] { k });

        for (int iter = 0; iter < maxIterations; iter++)
        {
            // Assign each point to nearest center
            var nearest = AssignToNearestCenters(points, centers);
            bool changed = !nearest.SequenceEqual(assignments);
            assignments = nearest;
            var clusterIndex = new Tensor<int>(assignments, new[] { n });
            clusterSizes = engine.TensorIndexAdd(new Tensor<T>(new[] { k }), 0, clusterIndex,
                engine.TensorFullLike(new Tensor<T>(new[] { n }), NumOps.One));

            if (!changed) break;

            // Update cluster centers: member mean, or zero for an empty cluster
            var sums = engine.TensorIndexAdd(new Tensor<T>(new[] { k, d }), 0, clusterIndex, points);
            var divisors = engine.TensorMax(clusterSizes, NumOps.One).Reshape(new[] { k, 1 });
            centers = engine.TensorDivide(sums, engine.TensorBroadcastTo(divisors, new[] { k, d }));
        }

        // Create weights proportional to cluster sizes
        var weights = engine.TensorDivideScalar(clusterSizes, NumOps.FromDouble(n));

        return new BackgroundSummary<T>(centers.ToMatrix(), weights.ToVector());
    }

    /// <summary>
    /// Summarizes data using stratified sampling.
    /// </summary>
    /// <param name="data">The full dataset to summarize.</param>
    /// <param name="stratifyColumn">Column index to stratify by (categorical feature).</param>
    /// <param name="nPerStratum">Number of samples per stratum (category).</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <returns>Stratified sample with equal weights.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Stratified sampling ensures each category is represented.
    ///
    /// <b>Example:</b> If you have a "gender" column with 60% male, 40% female:
    /// - Random sampling might give you 70% male by chance
    /// - Stratified sampling ensures you get ~60% male, ~40% female
    ///
    /// <b>When to use:</b>
    /// - When you have important categorical features
    /// - When minority classes are important
    /// - When you want to preserve the data distribution exactly
    /// </para>
    /// </remarks>
    public static BackgroundSummary<T> StratifiedSample(
        Matrix<T> data,
        int stratifyColumn,
        int nPerStratum = 10,
        int? randomState = null)
    {
        var rand = randomState.HasValue
            ? RandomHelper.CreateSeededRandom(randomState.Value)
            : RandomHelper.CreateSecureRandom();

        // Group samples by stratum value
        var strata = new Dictionary<string, List<int>>();
        var stratumKeys = Enumerable.Select(data.GetColumn(stratifyColumn), value => FormatStratumKey(value)).ToArray();
        for (int i = 0; i < data.Rows; i++)
        {
            string key = stratumKeys[i];
            if (!strata.ContainsKey(key))
            {
                strata[key] = new List<int>();
            }
            strata[key].Add(i);
        }

        // Sample from each stratum
        var selectedIndices = new List<int>();
        foreach (var (_, indices) in strata)
        {
            var shuffled = indices.OrderBy(_ => rand.Next()).ToList();
            selectedIndices.AddRange(shuffled.Take(Math.Min(nPerStratum, shuffled.Count)));
        }

        // Create output matrix
        var result = SelectRows(data, selectedIndices);

        return new BackgroundSummary<T>(result, CreateUniformWeights(selectedIndices.Count));
    }

    /// <summary>
    /// Summarizes data using random sampling.
    /// </summary>
    /// <param name="data">The full dataset to summarize.</param>
    /// <param name="nSamples">Number of samples to select.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <returns>Random sample with equal weights.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Simple random sampling - just pick random rows.
    ///
    /// This is the simplest approach:
    /// - Fast to compute
    /// - Works for any data type
    /// - May not represent rare patterns well
    ///
    /// Use when you don't have specific categorical features to stratify by.
    /// </para>
    /// </remarks>
    public static BackgroundSummary<T> RandomSample(
        Matrix<T> data,
        int nSamples = 100,
        int? randomState = null)
    {
        if (data.Rows <= nSamples)
        {
            return new BackgroundSummary<T>(data.Clone(), CreateUniformWeights(data.Rows));
        }

        var rand = randomState.HasValue
            ? RandomHelper.CreateSeededRandom(randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var indices = Enumerable.Range(0, data.Rows)
            .OrderBy(_ => rand.Next())
            .Take(nSamples)
            .ToList();

        var result = SelectRows(data, indices);

        return new BackgroundSummary<T>(result, CreateUniformWeights(nSamples));
    }

    /// <summary>
    /// Summarizes data using a combination of methods for mixed data.
    /// </summary>
    /// <param name="data">The full dataset to summarize.</param>
    /// <param name="categoricalColumns">Indices of categorical columns.</param>
    /// <param name="nSamples">Target number of summary samples.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <returns>Combined summary preserving categorical and continuous distributions.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Real-world data often has BOTH categorical and continuous features.
    ///
    /// This method:
    /// 1. Groups data by categorical features (stratification)
    /// 2. Within each group, uses KMeans for continuous features
    /// 3. Combines results with appropriate weights
    ///
    /// Best of both worlds - preserves categorical distribution AND
    /// continuous feature patterns.
    /// </para>
    /// </remarks>
    public static BackgroundSummary<T> MixedSummary(
        Matrix<T> data,
        int[] categoricalColumns,
        int nSamples = 100,
        int? randomState = null)
    {
        if (categoricalColumns.Length == 0)
        {
            return KMeans(data, nSamples, randomState: randomState);
        }

        var rand = randomState.HasValue
            ? RandomHelper.CreateSeededRandom(randomState.Value)
            : RandomHelper.CreateSecureRandom();

        // Group by categorical features
        var columnKeys = categoricalColumns
            .Select(c => Enumerable.Select(data.GetColumn(c), value => FormatStratumKey(value)).ToArray())
            .ToArray();
        var groups = new Dictionary<string, List<int>>();
        for (int i = 0; i < data.Rows; i++)
        {
            var key = string.Join("_", columnKeys.Select(keys => keys[i]));
            if (!groups.ContainsKey(key))
            {
                groups[key] = new List<int>();
            }
            groups[key].Add(i);
        }

        // Group weight = group size / rows; a sample kept as-is carries groupWeight / groupSize.
        var engine = AiDotNetEngine.Current;
        var groupList = groups.Values.ToList();
        var groupSizes = new Tensor<int>(groupList.Select(g => g.Count).ToArray(), new[] { groupList.Count }).Cast<T>();
        var groupWeights = engine.TensorDivideScalar(groupSizes, NumOps.FromDouble(data.Rows));
        var sampleWeights = engine.TensorDivide(groupWeights, groupSizes);

        var allSamples = new List<Tensor<T>>();
        var allWeights = new List<Tensor<T>>();

        foreach (var (indices, groupWeightT, sampleWeight) in groupList
            .Zip(groupWeights.ToVector(), (indices, weight) => (indices, weight))
            .Zip(sampleWeights.ToVector(), (group, sample) => (group.indices, group.weight, sample)))
        {
            double groupWeight = (double)indices.Count / data.Rows;
            int groupSamples = Math.Max(1, (int)(nSamples * groupWeight));

            if (indices.Count <= groupSamples)
            {
                // Use all samples from this group
                allSamples.Add(Tensor<T>.FromMatrix(SelectRows(data, indices)));
                allWeights.Add(engine.Fill(indices.Count, sampleWeight).AsTensor());
            }
            else
            {
                // Apply KMeans within this group
                var groupData = SelectRows(data, indices);

                var groupSummary = KMeans(groupData, groupSamples, randomState: rand.Next());
                allSamples.Add(Tensor<T>.FromMatrix(groupSummary.Data));
                allWeights.Add(engine.TensorMultiplyScalar(groupSummary.Weights.AsTensor(), groupWeightT));
            }
        }

        // Normalize weights
        var weights = engine.TensorConcatenate(allWeights.ToArray(), 0);
        weights = engine.TensorDivideScalar(weights, engine.TensorSum(weights));

        var resultData = engine.TensorConcatenate(allSamples.ToArray(), 0).ToMatrix();

        return new BackgroundSummary<T>(resultData, weights.ToVector());
    }

    /// <summary>
    /// Automatically chooses the best summarization method.
    /// </summary>
    /// <param name="data">The full dataset to summarize.</param>
    /// <param name="nSamples">Target number of summary samples.</param>
    /// <param name="categoricalColumns">Optional categorical column indices.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <returns>Automatically summarized background data.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Not sure which method to use? This function chooses for you!
    ///
    /// Decision logic:
    /// - Small data (less than nSamples): Return as-is
    /// - Has categorical features: Use MixedSummary
    /// - Only continuous features: Use KMeans
    /// </para>
    /// </remarks>
    public static BackgroundSummary<T> Auto(
        Matrix<T> data,
        int nSamples = 100,
        int[]? categoricalColumns = null,
        int? randomState = null)
    {
        if (data.Rows <= nSamples)
        {
            return new BackgroundSummary<T>(data.Clone(), CreateUniformWeights(data.Rows));
        }

        if (categoricalColumns != null && categoricalColumns.Length > 0)
        {
            return MixedSummary(data, categoricalColumns, nSamples, randomState);
        }

        return KMeans(data, nSamples, randomState: randomState);
    }

    /// <summary>
    /// Initializes cluster centers using KMeans++ algorithm.
    /// </summary>
    private static Matrix<T> InitializeCentersKMeansPlusPlus(Matrix<T> data, int k, Random rand)
    {
        int n = data.Rows;
        int d = data.Columns;
        var centers = new Matrix<T>(k, d);
        var selectedIndices = new HashSet<int>();

        // Choose first center randomly
        int first = rand.Next(n);
        selectedIndices.Add(first);
        for (int j = 0; j < d; j++)
        {
            centers[0, j] = data[first, j];
        }

        // Choose remaining centers with probability proportional to distance squared
        for (int c = 1; c < k; c++)
        {
            var distances = new double[n];
            double totalDist = 0;

            for (int i = 0; i < n; i++)
            {
                if (selectedIndices.Contains(i))
                {
                    distances[i] = 0;
                    continue;
                }

                double minDist = double.MaxValue;
                for (int cc = 0; cc < c; cc++)
                {
                    double dist = ComputeDistance(data.GetRow(i), centers.GetRow(cc));
                    if (dist < minDist) minDist = dist;
                }
                distances[i] = minDist * minDist;
                totalDist += distances[i];
            }

            // Sample proportionally
            double threshold = rand.NextDouble() * totalDist;
            double cumulative = 0;
            int chosen = 0;

            for (int i = 0; i < n; i++)
            {
                cumulative += distances[i];
                if (cumulative >= threshold)
                {
                    chosen = i;
                    break;
                }
            }

            selectedIndices.Add(chosen);
            for (int j = 0; j < d; j++)
            {
                centers[c, j] = data[chosen, j];
            }
        }

        return centers;
    }

    /// <summary>
    /// Computes Euclidean distance between two vectors.
    /// </summary>
    private static double ComputeDistance(Vector<T> a, Vector<T> b)
    {
        return NumOps.ToDouble(VectorHelper.EuclideanDistance(a, b));
    }

    /// <summary>
    /// Group key for a stratum value: the value with two decimals.
    /// </summary>
    private static string FormatStratumKey(T value)
    {
        return string.Format(System.Globalization.CultureInfo.CurrentCulture, "{0:F2}", value);
    }

    /// <summary>
    /// Creates uniform weights that sum to 1.
    /// </summary>
    private static Vector<T> CreateUniformWeights(int n)
    {
        return AiDotNetEngine.Current.Fill(n, NumOps.FromDouble(1.0 / n));
    }

    /// <summary>
    /// The rows of <paramref name="data"/> at <paramref name="indices"/>, in that order.
    /// </summary>
    private static Matrix<T> SelectRows(Matrix<T> data, IReadOnlyList<int> indices)
    {
        if (indices.Count == 0)
            return new Matrix<T>(0, data.Columns);

        var positions = new Tensor<int>(indices.ToArray(), new[] { indices.Count });
        return AiDotNetEngine.Current.TensorIndexSelect(Tensor<T>.FromMatrix(data), positions, 0).ToMatrix();
    }

    /// <summary>
    /// The index of the nearest center (Euclidean distance, first on ties) for every row of the data.
    /// </summary>
    private static int[] AssignToNearestCenters(Tensor<T> data, Tensor<T> centers)
    {
        var engine = AiDotNetEngine.Current;
        int n = data.Shape[0];
        int d = data.Shape[1];
        int k = centers.Shape[0];
        var assignments = new int[n];

        // Rows are processed in chunks so the [rows, k, d] difference tensor stays bounded.
        int chunk = Math.Max(1, (1 << 22) / Math.Max(1, k * d));
        var centerGrid = centers.Reshape(new[] { 1, k, d });
        for (int start = 0; start < n; start += chunk)
        {
            int rows = Math.Min(chunk, n - start);
            var block = engine.TensorSlice(data, new[] { start, 0 }, new[] { rows, d }).Reshape(new[] { rows, 1, d });
            var shape = new[] { rows, k, d };
            var differences = engine.TensorSubtract(engine.TensorBroadcastTo(block, shape), engine.TensorBroadcastTo(centerGrid, shape));
            var distances = engine.TensorSqrt(engine.ReduceSum(engine.TensorSquare(differences), new[] { 2 }, false));
            AiDotNet.AnomalyDetection.AnomalyTensorOps.FirstArgMinRows(distances).ToArray().CopyTo(assignments, start);
        }

        return assignments;
    }
}

/// <summary>
/// Represents summarized background data for interpretability methods.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This class holds the summarized background data
/// along with weights for each sample.
///
/// The weights indicate how "representative" each sample is:
/// - Higher weight = represents more of the original data
/// - Weights sum to 1.0
///
/// <b>Example:</b> If cluster A had 1000 samples and cluster B had 100 samples,
/// the center of cluster A would have ~10x the weight of cluster B.
/// </para>
/// </remarks>
public class BackgroundSummary<T>
{
    /// <summary>
    /// Gets the summarized background data (rows = samples).
    /// </summary>
    public Matrix<T> Data { get; }

    /// <summary>
    /// Gets the weight for each background sample.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use these weights when computing SHAP values.
    /// Higher-weight samples should contribute more to the expected value.
    /// </para>
    /// </remarks>
    public Vector<T> Weights { get; }

    /// <summary>
    /// Gets the number of background samples.
    /// </summary>
    public int NumSamples => Data.Rows;

    /// <summary>
    /// Gets the number of features.
    /// </summary>
    public int NumFeatures => Data.Columns;

    /// <summary>
    /// Initializes a new background summary.
    /// </summary>
    public BackgroundSummary(Matrix<T> data, Vector<T> weights)
    {
        Guard.NotNull(data);
        Data = data;
        Guard.NotNull(weights);
        Weights = weights;

        if (data.Rows != weights.Length)
        {
            throw new ArgumentException("Data rows must match weights length.");
        }
    }

    /// <summary>
    /// Gets the weighted mean of each feature.
    /// </summary>
    /// <returns>Weighted mean feature values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the "expected" feature value,
    /// which is often used as the baseline in SHAP.
    /// </para>
    /// </remarks>
    public Vector<T> GetWeightedMean()
    {
        var engine = AiDotNetEngine.Current;
        return engine.MatrixVectorMultiply(engine.MatrixTranspose(Data), Weights);
    }

    /// <summary>
    /// Gets the expected prediction using the background data.
    /// </summary>
    /// <param name="predictFunction">Function to get predictions.</param>
    /// <returns>Weighted average prediction.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the "base value" in SHAP - the prediction
    /// you'd expect when you don't know any feature values.
    /// </para>
    /// </remarks>
    public Vector<T> GetExpectedPrediction(Func<Vector<T>, Vector<T>> predictFunction)
    {
        if (NumSamples == 0)
            return new Vector<T>(0);

        var predictions = new Tensor<T>[NumSamples];
        for (int i = 0; i < NumSamples; i++)
        {
            predictions[i] = predictFunction(Data.GetRow(i)).AsTensor();
        }

        // Weighted sum of the predictions: weights^T [samples, outputs]
        var engine = AiDotNetEngine.Current;
        var stacked = engine.TensorStack(predictions, 0);
        return engine.TensorMatMul(Weights.AsTensor().Reshape(new[] { 1, NumSamples }), stacked).ToVector();
    }

    /// <summary>
    /// Returns a human-readable summary.
    /// </summary>
    public override string ToString()
    {
        if (NumSamples == 0)
        {
            return $"BackgroundSummary: {NumSamples} samples, {NumFeatures} features, " +
                   $"weights range: [{double.MaxValue:F4}, {double.MinValue:F4}]";
        }

        var engine = AiDotNetEngine.Current;
        T minWeight = engine.TensorMinValue(Weights.AsTensor());
        T maxWeight = engine.TensorMaxValue(Weights.AsTensor());

        return $"BackgroundSummary: {NumSamples} samples, {NumFeatures} features, " +
               $"weights range: [{minWeight:F4}, {maxWeight:F4}]";
    }
}

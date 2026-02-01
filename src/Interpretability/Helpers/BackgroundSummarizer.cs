using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

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
        var centers = InitializeCentersKMeansPlusPlus(data, k, rand);
        var assignments = new int[n];
        var clusterCounts = new int[k];

        for (int iter = 0; iter < maxIterations; iter++)
        {
            // Assign each point to nearest center
            bool changed = false;
            Array.Clear(clusterCounts, 0, k);

            for (int i = 0; i < n; i++)
            {
                int nearest = FindNearestCenter(data.GetRow(i), centers);
                if (assignments[i] != nearest)
                {
                    changed = true;
                    assignments[i] = nearest;
                }
                clusterCounts[nearest]++;
            }

            if (!changed) break;

            // Update cluster centers
            centers = new Matrix<T>(k, d);
            for (int i = 0; i < n; i++)
            {
                int cluster = assignments[i];
                for (int j = 0; j < d; j++)
                {
                    centers[cluster, j] = NumOps.Add(centers[cluster, j], data[i, j]);
                }
            }

            for (int c = 0; c < k; c++)
            {
                if (clusterCounts[c] > 0)
                {
                    for (int j = 0; j < d; j++)
                    {
                        centers[c, j] = NumOps.Divide(centers[c, j],
                            NumOps.FromDouble(clusterCounts[c]));
                    }
                }
            }
        }

        // Create weights proportional to cluster sizes
        var weights = new T[k];
        for (int c = 0; c < k; c++)
        {
            weights[c] = NumOps.FromDouble((double)clusterCounts[c] / n);
        }

        return new BackgroundSummary<T>(centers, new Vector<T>(weights));
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
        for (int i = 0; i < data.Rows; i++)
        {
            string key = NumOps.ToDouble(data[i, stratifyColumn]).ToString("F2");
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
        var result = new Matrix<T>(selectedIndices.Count, data.Columns);
        for (int i = 0; i < selectedIndices.Count; i++)
        {
            int srcIdx = selectedIndices[i];
            for (int j = 0; j < data.Columns; j++)
            {
                result[i, j] = data[srcIdx, j];
            }
        }

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

        var result = new Matrix<T>(nSamples, data.Columns);
        for (int i = 0; i < nSamples; i++)
        {
            for (int j = 0; j < data.Columns; j++)
            {
                result[i, j] = data[indices[i], j];
            }
        }

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
        var groups = new Dictionary<string, List<int>>();
        for (int i = 0; i < data.Rows; i++)
        {
            var key = string.Join("_", categoricalColumns.Select(c =>
                NumOps.ToDouble(data[i, c]).ToString("F2")));
            if (!groups.ContainsKey(key))
            {
                groups[key] = new List<int>();
            }
            groups[key].Add(i);
        }

        // Allocate samples proportionally to group sizes
        int totalGroups = groups.Count;
        int samplesPerGroup = Math.Max(1, nSamples / totalGroups);

        var allSamples = new List<Vector<T>>();
        var allWeights = new List<double>();

        foreach (var (_, indices) in groups)
        {
            double groupWeight = (double)indices.Count / data.Rows;
            int groupSamples = Math.Max(1, (int)(nSamples * groupWeight));

            if (indices.Count <= groupSamples)
            {
                // Use all samples from this group
                foreach (var idx in indices)
                {
                    allSamples.Add(data.GetRow(idx));
                    allWeights.Add(groupWeight / indices.Count);
                }
            }
            else
            {
                // Apply KMeans within this group
                var groupData = new Matrix<T>(indices.Count, data.Columns);
                for (int i = 0; i < indices.Count; i++)
                {
                    for (int j = 0; j < data.Columns; j++)
                    {
                        groupData[i, j] = data[indices[i], j];
                    }
                }

                var groupSummary = KMeans(groupData, groupSamples, randomState: rand.Next());
                for (int i = 0; i < groupSummary.Data.Rows; i++)
                {
                    allSamples.Add(groupSummary.Data.GetRow(i));
                    allWeights.Add(groupWeight * NumOps.ToDouble(groupSummary.Weights[i]));
                }
            }
        }

        // Normalize weights
        double weightSum = allWeights.Sum();
        for (int i = 0; i < allWeights.Count; i++)
        {
            allWeights[i] /= weightSum;
        }

        var resultData = new Matrix<T>(allSamples.Count, data.Columns);
        for (int i = 0; i < allSamples.Count; i++)
        {
            for (int j = 0; j < data.Columns; j++)
            {
                resultData[i, j] = allSamples[i][j];
            }
        }

        return new BackgroundSummary<T>(
            resultData,
            new Vector<T>(allWeights.Select(w => NumOps.FromDouble(w)).ToArray()));
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
    /// Finds the nearest center for a data point.
    /// </summary>
    private static int FindNearestCenter(Vector<T> point, Matrix<T> centers)
    {
        int nearest = 0;
        double minDist = double.MaxValue;

        for (int c = 0; c < centers.Rows; c++)
        {
            double dist = ComputeDistance(point, centers.GetRow(c));
            if (dist < minDist)
            {
                minDist = dist;
                nearest = c;
            }
        }

        return nearest;
    }

    /// <summary>
    /// Computes Euclidean distance between two vectors.
    /// </summary>
    private static double ComputeDistance(Vector<T> a, Vector<T> b)
    {
        double sum = 0;
        int len = Math.Min(a.Length, b.Length);

        for (int i = 0; i < len; i++)
        {
            double diff = NumOps.ToDouble(a[i]) - NumOps.ToDouble(b[i]);
            sum += diff * diff;
        }

        return Math.Sqrt(sum);
    }

    /// <summary>
    /// Creates uniform weights that sum to 1.
    /// </summary>
    private static Vector<T> CreateUniformWeights(int n)
    {
        var weights = new T[n];
        T weight = NumOps.FromDouble(1.0 / n);
        for (int i = 0; i < n; i++)
        {
            weights[i] = weight;
        }
        return new Vector<T>(weights);
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
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

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
        Data = data ?? throw new ArgumentNullException(nameof(data));
        Weights = weights ?? throw new ArgumentNullException(nameof(weights));

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
        var mean = new T[NumFeatures];

        for (int i = 0; i < NumSamples; i++)
        {
            double w = NumOps.ToDouble(Weights[i]);
            for (int j = 0; j < NumFeatures; j++)
            {
                mean[j] = NumOps.Add(mean[j],
                    NumOps.Multiply(Data[i, j], NumOps.FromDouble(w)));
            }
        }

        return new Vector<T>(mean);
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
        Vector<T>? result = null;

        for (int i = 0; i < NumSamples; i++)
        {
            var prediction = predictFunction(Data.GetRow(i));
            double w = NumOps.ToDouble(Weights[i]);

            if (result == null)
            {
                result = new Vector<T>(prediction.Length);
            }

            for (int j = 0; j < prediction.Length; j++)
            {
                result[j] = NumOps.Add(result[j],
                    NumOps.Multiply(prediction[j], NumOps.FromDouble(w)));
            }
        }

        return result ?? new Vector<T>(0);
    }

    /// <summary>
    /// Returns a human-readable summary.
    /// </summary>
    public override string ToString()
    {
        double minWeight = double.MaxValue, maxWeight = double.MinValue;
        for (int i = 0; i < NumSamples; i++)
        {
            double w = NumOps.ToDouble(Weights[i]);
            if (w < minWeight) minWeight = w;
            if (w > maxWeight) maxWeight = w;
        }

        return $"BackgroundSummary: {NumSamples} samples, {NumFeatures} features, " +
               $"weights range: [{minWeight:F4}, {maxWeight:F4}]";
    }
}

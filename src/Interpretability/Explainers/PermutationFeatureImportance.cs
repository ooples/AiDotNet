using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Interpretability.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interpretability.Explainers;

/// <summary>
/// Model-agnostic Permutation Feature Importance calculator.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Permutation Feature Importance measures how important each feature
/// is by randomly shuffling that feature's values and measuring how much worse the model performs.
///
/// The intuition is simple:
/// - If a feature is important, shuffling it destroys valuable information, and the model performs worse
/// - If a feature isn't important, shuffling it doesn't matter much
///
/// This method works with ANY model because it only looks at the model's predictions,
/// not its internal structure. It's like testing which ingredient is most important in a recipe
/// by randomly swapping each ingredient and seeing how much the dish changes.
/// </para>
/// </remarks>
public class PermutationFeatureImportance<T> : IGlobalExplainer<T, FeatureImportanceResult<T>>, IGPUAcceleratedExplainer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly Func<Matrix<T>, Vector<T>> _predictFunction;
    private readonly Func<Vector<T>, Vector<T>, T> _scoreFunction;
    private readonly int _nRepeats;
    private readonly int? _randomState;
    private readonly string[]? _featureNames;
    private GPUExplainerHelper<T>? _gpuHelper;

    /// <inheritdoc/>
    public string MethodName => "PermutationFeatureImportance";

    /// <inheritdoc/>
    public bool SupportsLocalExplanations => false;

    /// <inheritdoc/>
    public bool SupportsGlobalExplanations => true;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When GPU acceleration is enabled, permutations and predictions
    /// are computed in parallel, significantly speeding up importance computation.
    /// </para>
    /// </remarks>
    public bool IsGPUAccelerated => _gpuHelper?.IsGPUEnabled ?? false;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Call this method to enable GPU acceleration for permutation processing.
    /// </para>
    /// </remarks>
    public void SetGPUHelper(GPUExplainerHelper<T>? helper)
    {
        _gpuHelper = helper;
    }

    /// <summary>
    /// Initializes a new Permutation Feature Importance calculator.
    /// </summary>
    /// <param name="predictFunction">A function that takes input data and returns predictions.</param>
    /// <param name="scoreFunction">A function that computes a score given (actual, predicted). Higher = better.</param>
    /// <param name="nRepeats">Number of times to repeat the permutation (default: 5).</param>
    /// <param name="featureNames">Optional names for features.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// - <b>predictFunction</b>: Your model's prediction method
    /// - <b>scoreFunction</b>: How to measure model quality (e.g., accuracy, R², negative MSE)
    /// - <b>nRepeats</b>: More repeats = more stable importance estimates. Start with 5.
    /// </para>
    /// </remarks>
    public PermutationFeatureImportance(
        Func<Matrix<T>, Vector<T>> predictFunction,
        Func<Vector<T>, Vector<T>, T> scoreFunction,
        int nRepeats = 5,
        string[]? featureNames = null,
        int? randomState = null)
    {
        _predictFunction = predictFunction ?? throw new ArgumentNullException(nameof(predictFunction));
        _scoreFunction = scoreFunction ?? throw new ArgumentNullException(nameof(scoreFunction));

        if (nRepeats < 1)
            throw new ArgumentException("Number of repeats must be at least 1.", nameof(nRepeats));

        _nRepeats = nRepeats;
        _featureNames = featureNames;
        _randomState = randomState;
    }

    /// <summary>
    /// Creates a Permutation Feature Importance calculator from a model.
    /// </summary>
    public static PermutationFeatureImportance<T> FromModel<TInput, TOutput>(
        IFullModel<T, TInput, TOutput> model,
        Func<Vector<T>, Vector<T>, T> scoreFunction,
        int nRepeats = 5,
        string[]? featureNames = null,
        int? randomState = null)
    {
        Func<Matrix<T>, Vector<T>> predictFunc = data =>
        {
            var input = ConvertToModelInput<TInput>(data);
            var output = model.Predict(input);
            return ConvertFromModelOutput<TOutput>(output);
        };

        return new PermutationFeatureImportance<T>(predictFunc, scoreFunction, nRepeats, featureNames, randomState);
    }

    /// <summary>
    /// Creates a calculator using R² as the scoring function (for regression).
    /// </summary>
    public static PermutationFeatureImportance<T> ForRegression<TInput, TOutput>(
        IFullModel<T, TInput, TOutput> model,
        int nRepeats = 5,
        string[]? featureNames = null,
        int? randomState = null)
    {
        return FromModel(model, ComputeR2, nRepeats, featureNames, randomState);
    }

    /// <summary>
    /// Creates a calculator using accuracy as the scoring function (for classification).
    /// </summary>
    public static PermutationFeatureImportance<T> ForClassification<TInput, TOutput>(
        IFullModel<T, TInput, TOutput> model,
        int nRepeats = 5,
        string[]? featureNames = null,
        int? randomState = null)
    {
        return FromModel(model, ComputeAccuracy, nRepeats, featureNames, randomState);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// PFI requires target values to compute feature importance. Use <see cref="Calculate(Matrix{T}, Vector{T})"/> instead.
    /// </remarks>
    public FeatureImportanceResult<T> ExplainGlobal(Matrix<T> data)
    {
        throw new NotSupportedException(
            "PermutationFeatureImportance requires target values. " +
            "Use Calculate(data, target) instead of ExplainGlobal.");
    }

    /// <summary>
    /// Calculates permutation feature importance.
    /// </summary>
    /// <param name="X">The feature matrix.</param>
    /// <param name="y">The target values.</param>
    /// <returns>Feature importance results.</returns>
    public FeatureImportanceResult<T> Calculate(Matrix<T> X, Vector<T> y)
    {
        if (X.Rows != y.Length)
            throw new ArgumentException("Number of rows in X must match length of y.");

        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        int numFeatures = X.Columns;

        // Compute baseline score
        var basePredictions = _predictFunction(X);
        var baselineScore = _scoreFunction(y, basePredictions);

        // Compute importance for each feature
        var importances = new T[numFeatures];
        var importanceStds = new T[numFeatures];

        for (int j = 0; j < numFeatures; j++)
        {
            var scores = new double[_nRepeats];

            for (int r = 0; r < _nRepeats; r++)
            {
                // Create permuted copy
                var XPermuted = X.Clone();
                PermuteColumn(XPermuted, j, rand);

                // Compute score with permuted feature
                var permutedPredictions = _predictFunction(XPermuted);
                var permutedScore = _scoreFunction(y, permutedPredictions);

                // Importance = drop in score
                scores[r] = NumOps.ToDouble(baselineScore) - NumOps.ToDouble(permutedScore);
            }

            // Compute mean and std of importance
            importances[j] = NumOps.FromDouble(scores.Average());
            double variance = scores.Select(s => (s - scores.Average()) * (s - scores.Average())).Average();
            importanceStds[j] = NumOps.FromDouble(Math.Sqrt(variance));
        }

        return new FeatureImportanceResult<T>(
            importances: new Vector<T>(importances),
            importanceStds: new Vector<T>(importanceStds),
            baselineScore: baselineScore,
            featureNames: _featureNames);
    }

    /// <summary>
    /// Permutes (shuffles) a column in the matrix.
    /// </summary>
    private void PermuteColumn(Matrix<T> matrix, int columnIndex, Random rand)
    {
        int n = matrix.Rows;

        // Fisher-Yates shuffle
        for (int i = n - 1; i > 0; i--)
        {
            int j = rand.Next(i + 1);
            (matrix[i, columnIndex], matrix[j, columnIndex]) = (matrix[j, columnIndex], matrix[i, columnIndex]);
        }
    }

    private static T ComputeR2(Vector<T> actual, Vector<T> predicted)
    {
        if (actual.Length != predicted.Length)
            throw new ArgumentException("Vectors must have same length.");

        int n = actual.Length;
        if (n == 0) return NumOps.Zero;

        double meanActual = 0;
        for (int i = 0; i < n; i++)
            meanActual += NumOps.ToDouble(actual[i]);
        meanActual /= n;

        double ssRes = 0, ssTot = 0;
        for (int i = 0; i < n; i++)
        {
            double a = NumOps.ToDouble(actual[i]);
            double p = NumOps.ToDouble(predicted[i]);
            ssRes += (a - p) * (a - p);
            ssTot += (a - meanActual) * (a - meanActual);
        }

        if (ssTot < 1e-10) return NumOps.Zero;
        return NumOps.FromDouble(1 - ssRes / ssTot);
    }

    private static T ComputeAccuracy(Vector<T> actual, Vector<T> predicted)
    {
        if (actual.Length != predicted.Length)
            throw new ArgumentException("Vectors must have same length.");

        int n = actual.Length;
        if (n == 0) return NumOps.Zero;

        int correct = 0;
        for (int i = 0; i < n; i++)
        {
            // Round predictions for classification
            int actualClass = (int)Math.Round(NumOps.ToDouble(actual[i]));
            int predictedClass = (int)Math.Round(NumOps.ToDouble(predicted[i]));
            if (actualClass == predictedClass) correct++;
        }

        return NumOps.FromDouble((double)correct / n);
    }

    private static TInput ConvertToModelInput<TInput>(Matrix<T> data)
    {
        object result;
        if (typeof(TInput) == typeof(Matrix<T>))
            result = data;
        else if (typeof(TInput) == typeof(Tensor<T>))
            result = Tensor<T>.FromRowMatrix(data);
        else if (typeof(TInput) == typeof(Vector<T>) && data.Rows == 1)
            result = data.GetRow(0);
        else
            throw new NotSupportedException($"Cannot convert Matrix<T> to {typeof(TInput).Name}");

        return (TInput)result;
    }

    private static Vector<T> ConvertFromModelOutput<TOutput>(TOutput output)
    {
        if (output is Vector<T> vector)
            return vector;
        if (output is Matrix<T> matrix)
            return matrix.GetColumn(0);
        if (output is Tensor<T> tensor)
            return tensor.ToVector();
        if (output is T scalar)
            return new Vector<T>([scalar]);

        throw new NotSupportedException($"Cannot convert {typeof(TOutput).Name} to Vector<T>");
    }
}

/// <summary>
/// Represents the result of permutation feature importance calculation.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class FeatureImportanceResult<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets the importance score for each feature (mean drop in performance when permuted).
    /// </summary>
    public Vector<T> Importances { get; }

    /// <summary>
    /// Gets the standard deviation of importance scores across repeats.
    /// </summary>
    public Vector<T> ImportanceStds { get; }

    /// <summary>
    /// Gets the baseline model score (before any permutation).
    /// </summary>
    public T BaselineScore { get; }

    /// <summary>
    /// Gets the feature names, if available.
    /// </summary>
    public string[]? FeatureNames { get; }

    /// <summary>
    /// Gets the number of features.
    /// </summary>
    public int NumFeatures => Importances.Length;

    /// <summary>
    /// Initializes a new feature importance result.
    /// </summary>
    public FeatureImportanceResult(
        Vector<T> importances,
        Vector<T> importanceStds,
        T baselineScore,
        string[]? featureNames = null)
    {
        Importances = importances ?? throw new ArgumentNullException(nameof(importances));
        ImportanceStds = importanceStds ?? throw new ArgumentNullException(nameof(importanceStds));
        BaselineScore = baselineScore;
        FeatureNames = featureNames;
    }

    /// <summary>
    /// Gets features sorted by importance (most important first).
    /// </summary>
    public IEnumerable<(int Index, string? Name, T Importance, T Std)> GetSortedFeatures()
    {
        return Enumerable.Range(0, NumFeatures)
            .Select(i => (
                Index: i,
                Name: FeatureNames?[i],
                Importance: Importances[i],
                Std: ImportanceStds[i]))
            .OrderByDescending(x => NumOps.ToDouble(x.Importance));
    }

    /// <summary>
    /// Gets the top N most important features.
    /// </summary>
    public IEnumerable<(int Index, string? Name, T Importance, T Std)> GetTopFeatures(int n)
    {
        return GetSortedFeatures().Take(n);
    }

    /// <summary>
    /// Converts to a dictionary mapping feature names/indices to importance scores.
    /// </summary>
    public Dictionary<string, T> ToDictionary()
    {
        var result = new Dictionary<string, T>();
        for (int i = 0; i < NumFeatures; i++)
        {
            string key = FeatureNames?[i] ?? $"Feature{i}";
            result[key] = Importances[i];
        }
        return result;
    }

    /// <summary>
    /// Returns a human-readable summary.
    /// </summary>
    public override string ToString()
    {
        var top = GetTopFeatures(10).ToList();
        var lines = new List<string>
        {
            $"Permutation Feature Importance:",
            $"  Baseline Score: {BaselineScore}",
            $"  Feature Rankings:"
        };

        int rank = 1;
        foreach (var (index, name, importance, std) in top)
        {
            var featureLabel = name ?? $"Feature {index}";
            lines.Add($"    {rank}. {featureLabel}: {importance} (±{std})");
            rank++;
        }

        return string.Join(Environment.NewLine, lines);
    }
}

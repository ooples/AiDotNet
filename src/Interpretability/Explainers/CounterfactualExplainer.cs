using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interpretability.Explainers;

/// <summary>
/// Model-agnostic Counterfactual explainer that finds minimal changes needed for a different prediction.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Counterfactual explanations answer the question:
/// "What would need to change for the model to give a different prediction?"
///
/// Example: If a loan application was denied, a counterfactual might say:
/// "If your income was $5,000 higher, the loan would have been approved."
///
/// Key features:
/// - Shows the MINIMUM changes needed (fewest features changed)
/// - Changes should be realistic and actionable
/// - Helps users understand what they can do to get a different outcome
///
/// This is especially useful for:
/// - Credit decisions (what to improve for approval)
/// - Medical diagnoses (what tests would change the diagnosis)
/// - Any scenario where users want to know "what if"
/// </para>
/// </remarks>
public class CounterfactualExplainer<T> : ILocalExplainer<T, CounterfactualExplanation<T>>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly Func<Matrix<T>, Vector<T>> _predictFunction;
    private readonly int _numFeatures;
    private readonly int _maxIterations;
    private readonly double _stepSize;
    private readonly int _maxChanges;
    private readonly int? _randomState;
    private readonly string[]? _featureNames;
    private readonly T[]? _featureMins;
    private readonly T[]? _featureMaxs;
    private readonly bool[]? _featuresMutable;
    private readonly double _targetThreshold;

    /// <inheritdoc/>
    public string MethodName => "Counterfactual";

    /// <inheritdoc/>
    public bool SupportsLocalExplanations => true;

    /// <inheritdoc/>
    public bool SupportsGlobalExplanations => false;

    /// <summary>
    /// Initializes a new Counterfactual explainer.
    /// </summary>
    /// <param name="predictFunction">A function that takes input data and returns predictions.</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="maxIterations">Maximum optimization iterations (default: 1000).</param>
    /// <param name="stepSize">Step size for optimization (default: 0.1).</param>
    /// <param name="maxChanges">Maximum number of features to change (default: 5).</param>
    /// <param name="targetThreshold">Threshold for considering prediction changed (default: 0.5).</param>
    /// <param name="featureNames">Optional names for features.</param>
    /// <param name="featureMins">Optional minimum values for each feature.</param>
    /// <param name="featureMaxs">Optional maximum values for each feature.</param>
    /// <param name="featuresMutable">Optional flags indicating which features can be changed.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// - <b>maxChanges</b>: Limits how many features can change (simpler explanations)
    /// - <b>featuresMutable</b>: Some features can't change (e.g., can't change someone's age in the past)
    /// - <b>targetThreshold</b>: For classification, how confident the new prediction should be
    /// </para>
    /// </remarks>
    public CounterfactualExplainer(
        Func<Matrix<T>, Vector<T>> predictFunction,
        int numFeatures,
        int maxIterations = 1000,
        double stepSize = 0.1,
        int maxChanges = 5,
        double targetThreshold = 0.5,
        string[]? featureNames = null,
        T[]? featureMins = null,
        T[]? featureMaxs = null,
        bool[]? featuresMutable = null,
        int? randomState = null)
    {
        _predictFunction = predictFunction ?? throw new ArgumentNullException(nameof(predictFunction));

        if (numFeatures < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(numFeatures));
        if (maxIterations < 1)
            throw new ArgumentOutOfRangeException(nameof(maxIterations), "Max iterations must be at least 1.");
        if (stepSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(stepSize), "Step size must be positive.");
        if (featureNames != null && featureNames.Length != numFeatures)
            throw new ArgumentException($"featureNames length ({featureNames.Length}) must match numFeatures ({numFeatures}).", nameof(featureNames));
        if (featureMins != null && featureMins.Length != numFeatures)
            throw new ArgumentException($"featureMins length ({featureMins.Length}) must match numFeatures ({numFeatures}).", nameof(featureMins));
        if (featureMaxs != null && featureMaxs.Length != numFeatures)
            throw new ArgumentException($"featureMaxs length ({featureMaxs.Length}) must match numFeatures ({numFeatures}).", nameof(featureMaxs));
        if (featuresMutable != null && featuresMutable.Length != numFeatures)
            throw new ArgumentException($"featuresMutable length ({featuresMutable.Length}) must match numFeatures ({numFeatures}).", nameof(featuresMutable));

        _numFeatures = numFeatures;
        _maxIterations = maxIterations;
        _stepSize = stepSize;
        _maxChanges = maxChanges;
        _targetThreshold = targetThreshold;
        _featureNames = featureNames;
        _featureMins = featureMins;
        _featureMaxs = featureMaxs;
        _featuresMutable = featuresMutable;
        _randomState = randomState;
    }

    /// <summary>
    /// Finds a counterfactual explanation for the given instance.
    /// </summary>
    /// <param name="instance">The input instance to explain.</param>
    /// <returns>A counterfactual explanation showing minimal changes for a different prediction.</returns>
    public CounterfactualExplanation<T> Explain(Vector<T> instance)
    {
        // Use default target: flip the binary prediction
        var instanceMatrix = CreateSingleRowMatrix(instance);
        var originalPred = _predictFunction(instanceMatrix)[0];
        double originalValue = NumOps.ToDouble(originalPred);

        // Target is the opposite class
        double targetValue = originalValue >= _targetThreshold ? 0.0 : 1.0;
        var target = new Vector<T>(new[] { NumOps.FromDouble(targetValue) });

        return Explain(instance, target);
    }

    /// <summary>
    /// Finds a counterfactual explanation to achieve a specific target prediction.
    /// </summary>
    /// <param name="instance">The input instance to explain.</param>
    /// <param name="targetPrediction">The desired prediction value.</param>
    /// <returns>A counterfactual explanation showing minimal changes to achieve the target.</returns>
    public CounterfactualExplanation<T> Explain(Vector<T> instance, Vector<T> targetPrediction)
    {
        if (instance.Length != _numFeatures)
            throw new ArgumentException($"Instance has {instance.Length} features but expected {_numFeatures}.");

        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        // Get original prediction
        var instanceMatrix = CreateSingleRowMatrix(instance);
        var originalPred = _predictFunction(instanceMatrix);
        double targetValue = NumOps.ToDouble(targetPrediction[0]);

        // Initialize counterfactual as copy of instance
        var counterfactual = new T[_numFeatures];
        for (int i = 0; i < _numFeatures; i++)
            counterfactual[i] = instance[i];

        CounterfactualExplanation<T>? bestResult = null;
        double bestDistance = double.MaxValue;

        // Try multiple random restarts with different feature subsets
        int numRestarts = Math.Min(10, (int)Math.Pow(2, Math.Min(_numFeatures, 5)));

        for (int restart = 0; restart < numRestarts; restart++)
        {
            // Reset counterfactual
            for (int i = 0; i < _numFeatures; i++)
                counterfactual[i] = instance[i];

            // Select which features to modify (up to maxChanges)
            var mutableFeatures = SelectMutableFeatures(rand);

            // Optimization loop
            for (int iter = 0; iter < _maxIterations; iter++)
            {
                var cfMatrix = CreateSingleRowMatrix(new Vector<T>(counterfactual));
                var currentPred = _predictFunction(cfMatrix)[0];
                double currentValue = NumOps.ToDouble(currentPred);

                // Check if we reached the target
                bool reachedTarget = (targetValue >= _targetThreshold && currentValue >= _targetThreshold) ||
                                     (targetValue < _targetThreshold && currentValue < _targetThreshold);

                if (reachedTarget)
                {
                    double distance = ComputeDistance(instance, new Vector<T>(counterfactual));
                    if (distance < bestDistance)
                    {
                        bestDistance = distance;
                        bestResult = CreateExplanation(instance, new Vector<T>(counterfactual), originalPred, new Vector<T>(new[] { currentPred }));
                    }
                    break;
                }

                // Gradient-free optimization: try small perturbations
                foreach (int f in mutableFeatures)
                {
                    double originalVal = NumOps.ToDouble(counterfactual[f]);

                    // Try positive perturbation
                    double delta = _stepSize * (1 + rand.NextDouble());
                    double newVal = originalVal + delta * (targetValue >= _targetThreshold ? 1 : -1);
                    newVal = ClipToRange(newVal, f);

                    counterfactual[f] = NumOps.FromDouble(newVal);
                    var testMatrix = CreateSingleRowMatrix(new Vector<T>(counterfactual));
                    var testPred = _predictFunction(testMatrix)[0];
                    double testValue = NumOps.ToDouble(testPred);

                    // Check if this moves us closer to target
                    double improvement = (targetValue >= _targetThreshold)
                        ? testValue - currentValue
                        : currentValue - testValue;

                    if (improvement <= 0)
                    {
                        // Revert if no improvement
                        counterfactual[f] = NumOps.FromDouble(originalVal);
                    }
                }
            }
        }

        // If no counterfactual found, return best attempt
        if (bestResult == null)
        {
            bestResult = CreateExplanation(instance, new Vector<T>(counterfactual),
                originalPred, _predictFunction(CreateSingleRowMatrix(new Vector<T>(counterfactual))));
        }

        return bestResult;
    }

    /// <inheritdoc/>
    public CounterfactualExplanation<T>[] ExplainBatch(Matrix<T> instances)
    {
        var explanations = new CounterfactualExplanation<T>[instances.Rows];
        for (int i = 0; i < instances.Rows; i++)
        {
            explanations[i] = Explain(instances.GetRow(i));
        }
        return explanations;
    }

    /// <summary>
    /// Selects which features can be modified.
    /// </summary>
    private List<int> SelectMutableFeatures(Random rand)
    {
        var candidates = new List<int>();

        for (int f = 0; f < _numFeatures; f++)
        {
            if (_featuresMutable == null || _featuresMutable[f])
                candidates.Add(f);
        }

        // Shuffle and take up to maxChanges
        for (int i = candidates.Count - 1; i > 0; i--)
        {
            int j = rand.Next(i + 1);
            (candidates[i], candidates[j]) = (candidates[j], candidates[i]);
        }

        return candidates.Take(_maxChanges).ToList();
    }

    /// <summary>
    /// Clips a value to the valid range for a feature.
    /// </summary>
    private double ClipToRange(double value, int featureIndex)
    {
        double min = _featureMins != null ? NumOps.ToDouble(_featureMins[featureIndex]) : double.MinValue;
        double max = _featureMaxs != null ? NumOps.ToDouble(_featureMaxs[featureIndex]) : double.MaxValue;
        return Math.Max(min, Math.Min(max, value));
    }

    /// <summary>
    /// Computes the distance between original and counterfactual.
    /// </summary>
    private double ComputeDistance(Vector<T> original, Vector<T> counterfactual)
    {
        double sumSq = 0;
        for (int i = 0; i < original.Length; i++)
        {
            double diff = NumOps.ToDouble(original[i]) - NumOps.ToDouble(counterfactual[i]);
            sumSq += diff * diff;
        }
        return Math.Sqrt(sumSq);
    }

    /// <summary>
    /// Creates a CounterfactualExplanation from the results.
    /// </summary>
    private CounterfactualExplanation<T> CreateExplanation(
        Vector<T> original,
        Vector<T> counterfactual,
        Vector<T> originalPred,
        Vector<T> counterfactualPred)
    {
        var featureChanges = new Dictionary<int, T>();

        for (int i = 0; i < _numFeatures; i++)
        {
            double origVal = NumOps.ToDouble(original[i]);
            double cfVal = NumOps.ToDouble(counterfactual[i]);

            if (Math.Abs(origVal - cfVal) > 1e-6)
            {
                featureChanges[i] = NumOps.FromDouble(cfVal - origVal);
            }
        }

        return new CounterfactualExplanation<T>
        {
            OriginalInput = Tensor<T>.FromVector(original, new[] { original.Length }),
            CounterfactualInput = Tensor<T>.FromVector(counterfactual, new[] { counterfactual.Length }),
            OriginalPrediction = Tensor<T>.FromVector(originalPred, new[] { originalPred.Length }),
            CounterfactualPrediction = Tensor<T>.FromVector(counterfactualPred, new[] { counterfactualPred.Length }),
            FeatureChanges = featureChanges,
            Distance = NumOps.FromDouble(ComputeDistance(original, counterfactual)),
            MaxChanges = _maxChanges
        };
    }

    private Matrix<T> CreateSingleRowMatrix(Vector<T> row)
    {
        var matrix = new Matrix<T>(1, row.Length);
        for (int j = 0; j < row.Length; j++)
            matrix[0, j] = row[j];
        return matrix;
    }
}

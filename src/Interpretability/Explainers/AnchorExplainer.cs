using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interpretability.Explainers;

/// <summary>
/// Model-agnostic Anchor explainer that provides rule-based explanations.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Anchors are "if-then" rules that explain predictions.
/// Unlike SHAP or LIME which give feature weights, Anchors give you rules like:
/// "IF Age > 30 AND Income > 50000 THEN the model predicts 'Approved'"
///
/// Key concepts:
/// - <b>Precision</b>: How often the rule correctly predicts the same outcome (e.g., 95% of the time)
/// - <b>Coverage</b>: What fraction of all instances the rule applies to
///
/// Anchors are great when you need to explain to non-technical stakeholders
/// because rules are intuitive and easy to understand.
///
/// Example output: "The loan was approved because Age >= 25 AND CreditScore >= 700"
/// This rule holds for 95% of similar applicants (precision) and covers 30% of all applicants (coverage).
/// </para>
/// </remarks>
public class AnchorExplainer<T> : ILocalExplainer<T, AnchorExplanation<T>>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly Func<Matrix<T>, Vector<T>> _predictFunction;
    private readonly int _numFeatures;
    private readonly T _precisionThreshold;
    private readonly int _maxAnchorSize;
    private readonly int _beamWidth;
    private readonly int _nSamples;
    private readonly int? _randomState;
    private readonly string[]? _featureNames;
    private readonly T[]? _featureMins;
    private readonly T[]? _featureMaxs;

    /// <inheritdoc/>
    public string MethodName => "Anchors";

    /// <inheritdoc/>
    public bool SupportsLocalExplanations => true;

    /// <inheritdoc/>
    public bool SupportsGlobalExplanations => false;

    /// <summary>
    /// Initializes a new Anchor explainer.
    /// </summary>
    /// <param name="predictFunction">A function that takes input data and returns predictions.</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="precisionThreshold">Minimum precision required for an anchor (default: 0.95).</param>
    /// <param name="maxAnchorSize">Maximum number of features in an anchor (default: 6).</param>
    /// <param name="beamWidth">Beam search width (default: 4).</param>
    /// <param name="nSamples">Number of samples for precision estimation (default: 1000).</param>
    /// <param name="featureNames">Optional names for features.</param>
    /// <param name="featureMins">Optional minimum values for each feature (for sampling).</param>
    /// <param name="featureMaxs">Optional maximum values for each feature (for sampling).</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// - <b>precisionThreshold</b>: How accurate the rule must be (0.95 = 95% accurate)
    /// - <b>maxAnchorSize</b>: Maximum conditions in the rule (e.g., 3 means at most 3 "AND" conditions)
    /// - <b>beamWidth</b>: How many candidate rules to explore at each step (higher = better but slower)
    /// </para>
    /// </remarks>
    public AnchorExplainer(
        Func<Matrix<T>, Vector<T>> predictFunction,
        int numFeatures,
        double precisionThreshold = 0.95,
        int maxAnchorSize = 6,
        int beamWidth = 4,
        int nSamples = 1000,
        string[]? featureNames = null,
        T[]? featureMins = null,
        T[]? featureMaxs = null,
        int? randomState = null)
    {
        _predictFunction = predictFunction ?? throw new ArgumentNullException(nameof(predictFunction));

        if (numFeatures < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(numFeatures));
        if (precisionThreshold <= 0 || precisionThreshold > 1)
            throw new ArgumentOutOfRangeException(nameof(precisionThreshold), "Precision threshold must be in (0, 1].");
        if (maxAnchorSize < 1)
            throw new ArgumentOutOfRangeException(nameof(maxAnchorSize), "Max anchor size must be at least 1.");
        if (beamWidth < 1)
            throw new ArgumentOutOfRangeException(nameof(beamWidth), "Beam width must be at least 1.");
        if (nSamples < 1)
            throw new ArgumentOutOfRangeException(nameof(nSamples), "Number of samples must be at least 1.");
        if (featureNames != null && featureNames.Length != numFeatures)
            throw new ArgumentException($"featureNames length ({featureNames.Length}) must match numFeatures ({numFeatures}).", nameof(featureNames));
        if (featureMins != null && featureMins.Length != numFeatures)
            throw new ArgumentException($"featureMins length ({featureMins.Length}) must match numFeatures ({numFeatures}).", nameof(featureMins));
        if (featureMaxs != null && featureMaxs.Length != numFeatures)
            throw new ArgumentException($"featureMaxs length ({featureMaxs.Length}) must match numFeatures ({numFeatures}).", nameof(featureMaxs));

        _numFeatures = numFeatures;
        _precisionThreshold = NumOps.FromDouble(precisionThreshold);
        _maxAnchorSize = maxAnchorSize;
        _beamWidth = beamWidth;
        _nSamples = nSamples;
        _featureNames = featureNames;
        _featureMins = featureMins;
        _featureMaxs = featureMaxs;
        _randomState = randomState;
    }

    /// <inheritdoc/>
    public AnchorExplanation<T> Explain(Vector<T> instance)
    {
        if (instance.Length != _numFeatures)
            throw new ArgumentException($"Instance has {instance.Length} features but expected {_numFeatures}.");

        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        // Get the prediction for this instance
        var instanceMatrix = CreateSingleRowMatrix(instance);
        var originalPred = _predictFunction(instanceMatrix)[0];
        var originalClass = (int)Math.Round(NumOps.ToDouble(originalPred));

        // Initialize beam search with empty anchor
        var beam = new List<(HashSet<int> features, Dictionary<int, (T min, T max)> rules, T precision, T coverage)>
        {
            (new HashSet<int>(), new Dictionary<int, (T min, T max)>(), NumOps.One, NumOps.One)
        };

        AnchorExplanation<T>? bestAnchor = null;

        for (int size = 1; size <= _maxAnchorSize; size++)
        {
            var candidates = new List<(HashSet<int> features, Dictionary<int, (T min, T max)> rules, T precision, T coverage)>();

            foreach (var (anchorFeatures, anchorRules, _, _) in beam)
            {
                // Try adding each unused feature
                for (int f = 0; f < _numFeatures; f++)
                {
                    if (anchorFeatures.Contains(f))
                        continue;

                    var newFeatures = new HashSet<int>(anchorFeatures) { f };
                    var newRules = new Dictionary<int, (T min, T max)>(anchorRules);

                    // Create rule for this feature based on instance value
                    var featureValue = instance[f];
                    var (min, max) = CreateFeatureRule(f, featureValue, rand);
                    newRules[f] = (min, max);

                    // Estimate precision and coverage
                    var (precision, coverage) = EstimatePrecisionAndCoverage(
                        instance, newRules, originalClass, rand);

                    candidates.Add((newFeatures, newRules, precision, coverage));
                }
            }

            // Sort by precision (descending), then coverage (descending)
            candidates.Sort((a, b) =>
            {
                int precCompare = NumOps.ToDouble(b.precision).CompareTo(NumOps.ToDouble(a.precision));
                if (precCompare != 0) return precCompare;
                return NumOps.ToDouble(b.coverage).CompareTo(NumOps.ToDouble(a.coverage));
            });

            // Keep top beam_width candidates
            beam = candidates.Take(_beamWidth).ToList();

            // Check if any candidate meets precision threshold
            foreach (var (features, rules, precision, coverage) in beam)
            {
                if (NumOps.ToDouble(precision) >= NumOps.ToDouble(_precisionThreshold))
                {
                    if (bestAnchor == null ||
                        NumOps.ToDouble(coverage) > NumOps.ToDouble(bestAnchor.Coverage) ||
                        (NumOps.ToDouble(coverage) == NumOps.ToDouble(bestAnchor.Coverage) &&
                         features.Count < bestAnchor.AnchorFeatures.Count))
                    {
                        bestAnchor = CreateExplanation(features, rules, precision, coverage);
                    }
                }
            }

            if (bestAnchor != null && beam.All(b => NumOps.ToDouble(b.precision) >= NumOps.ToDouble(_precisionThreshold)))
                break;
        }

        // If no anchor meets threshold, return the best we found
        if (bestAnchor == null && beam.Count > 0)
        {
            var best = beam.OrderByDescending(b => NumOps.ToDouble(b.precision)).First();
            bestAnchor = CreateExplanation(best.features, best.rules, best.precision, best.coverage);
        }

        return bestAnchor ?? new AnchorExplanation<T>
        {
            Precision = NumOps.Zero,
            Coverage = NumOps.Zero,
            Threshold = _precisionThreshold,
            Description = "No anchor found"
        };
    }

    /// <inheritdoc/>
    public AnchorExplanation<T>[] ExplainBatch(Matrix<T> instances)
    {
        var explanations = new AnchorExplanation<T>[instances.Rows];
        for (int i = 0; i < instances.Rows; i++)
        {
            explanations[i] = Explain(instances.GetRow(i));
        }
        return explanations;
    }

    /// <summary>
    /// Creates a feature rule (min, max) based on the instance value.
    /// </summary>
    private (T min, T max) CreateFeatureRule(int featureIndex, T value, Random rand)
    {
        double val = NumOps.ToDouble(value);

        // Use provided bounds or estimate from value
        double featureMin = _featureMins != null ? NumOps.ToDouble(_featureMins[featureIndex]) : val - Math.Abs(val) * 0.5 - 1;
        double featureMax = _featureMaxs != null ? NumOps.ToDouble(_featureMaxs[featureIndex]) : val + Math.Abs(val) * 0.5 + 1;

        // Create a rule that includes the instance value with some margin
        double range = featureMax - featureMin;
        double margin = range * 0.1; // 10% margin

        double ruleMin = val - margin;
        double ruleMax = val + margin;

        return (NumOps.FromDouble(ruleMin), NumOps.FromDouble(ruleMax));
    }

    /// <summary>
    /// Estimates precision and coverage of an anchor rule.
    /// </summary>
    private (T precision, T coverage) EstimatePrecisionAndCoverage(
        Vector<T> instance,
        Dictionary<int, (T min, T max)> rules,
        int originalClass,
        Random rand)
    {
        int covered = 0;
        int correct = 0;

        for (int i = 0; i < _nSamples; i++)
        {
            // Generate a random sample
            var sample = GenerateRandomSample(instance, rules, rand);

            // Check if sample is covered by anchor
            bool isCovered = true;
            foreach (var (featureIdx, (min, max)) in rules)
            {
                double sampleVal = NumOps.ToDouble(sample[featureIdx]);
                if (sampleVal < NumOps.ToDouble(min) || sampleVal > NumOps.ToDouble(max))
                {
                    isCovered = false;
                    break;
                }
            }

            if (isCovered)
            {
                covered++;

                // Check if prediction matches original
                var sampleMatrix = CreateSingleRowMatrix(sample);
                var pred = _predictFunction(sampleMatrix)[0];
                int predClass = (int)Math.Round(NumOps.ToDouble(pred));

                if (predClass == originalClass)
                    correct++;
            }
        }

        double precision = covered > 0 ? (double)correct / covered : 0;
        double coverage = (double)covered / _nSamples;

        return (NumOps.FromDouble(precision), NumOps.FromDouble(coverage));
    }

    /// <summary>
    /// Generates a random sample, keeping anchor features fixed.
    /// </summary>
    private Vector<T> GenerateRandomSample(Vector<T> instance, Dictionary<int, (T min, T max)> rules, Random rand)
    {
        var sample = new T[_numFeatures];

        for (int f = 0; f < _numFeatures; f++)
        {
            if (rules.ContainsKey(f))
            {
                // Keep anchor feature within rule bounds
                var (min, max) = rules[f];
                double minVal = NumOps.ToDouble(min);
                double maxVal = NumOps.ToDouble(max);
                sample[f] = NumOps.FromDouble(minVal + rand.NextDouble() * (maxVal - minVal));
            }
            else
            {
                // Randomize non-anchor features
                double instVal = NumOps.ToDouble(instance[f]);
                double featureMin = _featureMins != null ? NumOps.ToDouble(_featureMins[f]) : instVal - Math.Abs(instVal) * 2 - 1;
                double featureMax = _featureMaxs != null ? NumOps.ToDouble(_featureMaxs[f]) : instVal + Math.Abs(instVal) * 2 + 1;
                sample[f] = NumOps.FromDouble(featureMin + rand.NextDouble() * (featureMax - featureMin));
            }
        }

        return new Vector<T>(sample);
    }

    /// <summary>
    /// Creates an AnchorExplanation from the found anchor.
    /// </summary>
    private AnchorExplanation<T> CreateExplanation(
        HashSet<int> features,
        Dictionary<int, (T min, T max)> rules,
        T precision,
        T coverage)
    {
        var explanation = new AnchorExplanation<T>
        {
            AnchorFeatures = features.OrderBy(f => f).ToList(),
            AnchorRules = rules,
            Precision = precision,
            Coverage = coverage,
            Threshold = _precisionThreshold
        };

        // Build human-readable description
        var conditions = new List<string>();
        foreach (var f in explanation.AnchorFeatures)
        {
            var (min, max) = rules[f];
            string featureName = _featureNames?[f] ?? $"Feature {f}";
            conditions.Add($"{NumOps.ToDouble(min):F2} <= {featureName} <= {NumOps.ToDouble(max):F2}");
        }

        explanation.Description = conditions.Count > 0
            ? "IF " + string.Join(" AND ", conditions)
            : "No conditions";

        return explanation;
    }

    private Matrix<T> CreateSingleRowMatrix(Vector<T> row)
    {
        var matrix = new Matrix<T>(1, row.Length);
        for (int j = 0; j < row.Length; j++)
            matrix[0, j] = row[j];
        return matrix;
    }
}

using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.Interpretability.Explainers;

/// <summary>
/// Contrastive explainer that answers "Why X and not Y?" questions.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Contrastive explanations answer a specific type of question:
/// "Why did the model predict X instead of Y?"
///
/// This is how humans naturally ask for explanations:
/// - "Why was my loan denied (instead of approved)?"
/// - "Why is this classified as cat (and not dog)?"
/// - "Why is this patient high risk (rather than low risk)?"
///
/// Key concepts:
/// 1. <b>Fact</b>: What actually happened (the model's prediction)
/// 2. <b>Foil</b>: What you're comparing against (the alternative outcome)
/// 3. <b>Pertinent Positives</b>: Features that support the fact
/// 4. <b>Pertinent Negatives</b>: Features that, if changed, would lead to the foil
///
/// Why contrastive explanations are useful:
/// - Match how humans think about explanations
/// - Focus on what matters for the specific comparison
/// - Actionable: "Change these features to get the alternative outcome"
///
/// Example: "Loan denied instead of approved"
/// - Pertinent Positive: "Credit score is 580 (below 620 threshold)"
/// - Pertinent Negative: "If income were $10K higher, loan would be approved"
/// </para>
/// </remarks>
public class ContrastiveExplainer<T> : ILocalExplainer<T, ContrastiveExplanation<T>>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly Func<Matrix<T>, Vector<T>> _predictFunction;
    private readonly int _numFeatures;
    private readonly string[]? _featureNames;
    private readonly string[]? _classNames;
    private readonly T[]? _featureMins;
    private readonly T[]? _featureMaxs;
    private readonly double _perturbationStep;

    /// <inheritdoc/>
    public string MethodName => "Contrastive";

    /// <inheritdoc/>
    public bool SupportsLocalExplanations => true;

    /// <inheritdoc/>
    public bool SupportsGlobalExplanations => false;

    /// <summary>
    /// Initializes a new Contrastive explainer.
    /// </summary>
    /// <param name="predictFunction">Function that takes batch input and returns predictions.</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="featureNames">Optional names for features.</param>
    /// <param name="classNames">Optional names for classes.</param>
    /// <param name="featureMins">Optional minimum values for features.</param>
    /// <param name="featureMaxs">Optional maximum values for features.</param>
    /// <param name="perturbationStep">Step size for perturbations (default: 0.1).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// - <b>classNames</b>: Names like ["Denied", "Approved"] help make explanations readable.
    /// - <b>perturbationStep</b>: Controls how much to change features when searching for
    ///   pertinent negatives. Smaller = more precise but slower.
    /// </para>
    /// </remarks>
    public ContrastiveExplainer(
        Func<Matrix<T>, Vector<T>> predictFunction,
        int numFeatures,
        string[]? featureNames = null,
        string[]? classNames = null,
        T[]? featureMins = null,
        T[]? featureMaxs = null,
        double perturbationStep = 0.1)
    {
        Guard.NotNull(predictFunction);
        _predictFunction = predictFunction;

        if (numFeatures < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(numFeatures));
        if (featureNames != null && featureNames.Length != numFeatures)
            throw new ArgumentException($"featureNames length ({featureNames.Length}) must match numFeatures ({numFeatures}).", nameof(featureNames));
        if (featureMins != null && featureMins.Length != numFeatures)
            throw new ArgumentException($"featureMins length ({featureMins.Length}) must match numFeatures ({numFeatures}).", nameof(featureMins));
        if (featureMaxs != null && featureMaxs.Length != numFeatures)
            throw new ArgumentException($"featureMaxs length ({featureMaxs.Length}) must match numFeatures ({numFeatures}).", nameof(featureMaxs));
        if (perturbationStep <= 0)
            throw new ArgumentOutOfRangeException(nameof(perturbationStep), "perturbationStep must be positive.");

        _numFeatures = numFeatures;
        _featureNames = featureNames;
        _classNames = classNames;
        _featureMins = featureMins;
        _featureMaxs = featureMaxs;
        _perturbationStep = perturbationStep;
    }

    /// <summary>
    /// Explains why the model predicted one class instead of another.
    /// </summary>
    /// <param name="instance">The input instance to explain.</param>
    /// <returns>Contrastive explanation comparing fact to most likely alternative.</returns>
    public ContrastiveExplanation<T> Explain(Vector<T> instance)
    {
        if (instance.Length != _numFeatures)
            throw new ArgumentException($"Instance has {instance.Length} features but expected {_numFeatures}.", nameof(instance));

        // Get prediction
        var instanceMatrix = CreateSingleRowMatrix(instance);
        var predictions = _predictFunction(instanceMatrix);

        // When the predict function returns a scalar (batch mode with 1 row),
        // expand to binary class probabilities for contrastive analysis
        if (predictions.Length < 2)
        {
            predictions = ExpandToClassProbabilities(predictions, 2);
        }

        // Find fact (predicted class) and foil (next most likely class)
        int factClass = 0;
        int foilClass = 1;
        double maxVal = double.MinValue;
        double secondMaxVal = double.MinValue;

        for (int i = 0; i < predictions.Length; i++)
        {
            double val = NumOps.ToDouble(predictions[i]);
            if (val > maxVal)
            {
                secondMaxVal = maxVal;
                foilClass = factClass;
                maxVal = val;
                factClass = i;
            }
            else if (val > secondMaxVal)
            {
                secondMaxVal = val;
                foilClass = i;
            }
        }

        return Explain(instance, factClass, foilClass);
    }

    /// <summary>
    /// Explains why the model predicted factClass instead of foilClass.
    /// </summary>
    /// <param name="instance">The input instance to explain.</param>
    /// <param name="factClass">The actual predicted class (what happened).</param>
    /// <param name="foilClass">The alternative class (what didn't happen).</param>
    /// <returns>Contrastive explanation.</returns>
    public ContrastiveExplanation<T> Explain(Vector<T> instance, int factClass, int foilClass)
    {
        if (instance.Length != _numFeatures)
            throw new ArgumentException($"Instance has {instance.Length} features but expected {_numFeatures}.", nameof(instance));

        var instanceMatrix = CreateSingleRowMatrix(instance);
        var predictions = _predictFunction(instanceMatrix);

        // When the predict function returns per-row results (batch mode) rather than
        // per-class probabilities, expand into a class-probability vector.
        // A single scalar indicates batch predict on a 1-row matrix.
        int numClassesNeeded = Math.Max(factClass, foilClass) + 1;
        if (predictions.Length < numClassesNeeded)
        {
            predictions = ExpandToClassProbabilities(predictions, numClassesNeeded);
        }

        if (factClass < 0 || factClass >= predictions.Length)
            throw new ArgumentOutOfRangeException(nameof(factClass), $"factClass ({factClass}) must be between 0 and {predictions.Length - 1}.");
        if (foilClass < 0 || foilClass >= predictions.Length)
            throw new ArgumentOutOfRangeException(nameof(foilClass), $"foilClass ({foilClass}) must be between 0 and {predictions.Length - 1}.");
        if (factClass == foilClass)
            throw new ArgumentException("factClass and foilClass must be different.", nameof(foilClass));

        double factScore = NumOps.ToDouble(predictions[factClass]);
        double foilScore = NumOps.ToDouble(predictions[foilClass]);

        // Find pertinent positives (features that support the fact)
        var pertinentPositives = FindPertinentPositives(instance, factClass, foilClass);

        // Find pertinent negatives (features that, if changed, would flip to foil)
        var pertinentNegatives = FindPertinentNegatives(instance, factClass, foilClass);

        // Compute feature contributions using gradients
        var featureContributions = ComputeFeatureContributions(instance, factClass, foilClass);

        return new ContrastiveExplanation<T>
        {
            Input = instance,
            Prediction = predictions,
            FactClass = factClass,
            FoilClass = foilClass,
            FactClassName = _classNames != null && factClass < _classNames.Length
                ? _classNames[factClass]
                : $"Class {factClass}",
            FoilClassName = _classNames != null && foilClass < _classNames.Length
                ? _classNames[foilClass]
                : $"Class {foilClass}",
            FactScore = NumOps.FromDouble(factScore),
            FoilScore = NumOps.FromDouble(foilScore),
            PertinentPositives = pertinentPositives,
            PertinentNegatives = pertinentNegatives,
            FeatureContributions = featureContributions,
            FeatureNames = _featureNames ?? Enumerable.Range(0, _numFeatures).Select(i => $"Feature {i}").ToArray()
        };
    }

    /// <inheritdoc/>
    public ContrastiveExplanation<T>[] ExplainBatch(Matrix<T> instances)
    {
        var explanations = new ContrastiveExplanation<T>[instances.Rows];
        for (int i = 0; i < instances.Rows; i++)
        {
            explanations[i] = Explain(instances.GetRow(i));
        }
        return explanations;
    }

    /// <summary>
    /// Finds features that support the fact (pertinent positives).
    /// These are features that, if changed toward the foil's typical values, would weaken the fact.
    /// </summary>
    private List<PertinentFeature<T>> FindPertinentPositives(Vector<T> instance, int factClass, int foilClass)
    {
        var pertinentPositives = new List<PertinentFeature<T>>();

        for (int f = 0; f < _numFeatures; f++)
        {
            double originalVal = NumOps.ToDouble(instance[f]);

            // Try zeroing out this feature
            var modifiedInstance = instance.Clone();
            modifiedInstance[f] = NumOps.Zero;

            var modifiedMatrix = CreateSingleRowMatrix(modifiedInstance);
            var modifiedPred = _predictFunction(modifiedMatrix);
            int numClassesNeeded = Math.Max(factClass, foilClass) + 1;
            if (modifiedPred.Length < numClassesNeeded)
            {
                modifiedPred = ExpandToClassProbabilities(modifiedPred, numClassesNeeded);
            }

            double originalDiff = GetScoreDifference(instance, factClass, foilClass);
            double modifiedFactScore = NumOps.ToDouble(modifiedPred[factClass]);
            double modifiedFoilScore = NumOps.ToDouble(modifiedPred[foilClass]);
            double modifiedDiff = modifiedFactScore - modifiedFoilScore;

            // If removing this feature reduces the advantage of fact over foil, it's a pertinent positive
            if (modifiedDiff < originalDiff - 0.01)
            {
                pertinentPositives.Add(new PertinentFeature<T>
                {
                    FeatureIndex = f,
                    FeatureName = _featureNames != null && f < _featureNames.Length ? _featureNames[f] : $"Feature {f}",
                    CurrentValue = instance[f],
                    ContrastValue = NumOps.Zero,
                    ImpactOnFact = NumOps.FromDouble(originalDiff - modifiedDiff),
                    Explanation = $"This feature value supports the '{GetClassName(factClass)}' prediction"
                });
            }
        }

        return pertinentPositives
            .OrderByDescending(p => NumOps.ToDouble(p.ImpactOnFact))
            .Take(5)
            .ToList();
    }

    /// <summary>
    /// Finds features that could flip the prediction (pertinent negatives).
    /// These are minimal changes that would result in the foil class.
    /// </summary>
    private List<PertinentFeature<T>> FindPertinentNegatives(Vector<T> instance, int factClass, int foilClass)
    {
        var pertinentNegatives = new List<PertinentFeature<T>>();

        for (int f = 0; f < _numFeatures; f++)
        {
            double originalVal = NumOps.ToDouble(instance[f]);
            double minVal = _featureMins != null ? NumOps.ToDouble(_featureMins[f]) : originalVal - Math.Abs(originalVal) - 1;
            double maxVal = _featureMaxs != null ? NumOps.ToDouble(_featureMaxs[f]) : originalVal + Math.Abs(originalVal) + 1;

            // Skip if min > max (invalid range)
            if (minVal > maxVal)
                continue;

            // Search for value that would flip to foil
            double? flipValue = null;
            double minFlipDistance = double.MaxValue;

            // Try values in both directions
            for (double delta = _perturbationStep; delta <= maxVal - minVal; delta += _perturbationStep)
            {
                // Try positive direction
                double testVal = Math.Min(maxVal, originalVal + delta);
                var modified = instance.Clone();
                modified[f] = NumOps.FromDouble(testVal);
                var modifiedMatrix = CreateSingleRowMatrix(modified);
                var pred = _predictFunction(modifiedMatrix);

                int predictedClass = GetPredictedClass(pred);
                if (predictedClass == foilClass && Math.Abs(testVal - originalVal) < minFlipDistance)
                {
                    flipValue = testVal;
                    minFlipDistance = Math.Abs(testVal - originalVal);
                }

                // Try negative direction
                testVal = Math.Max(minVal, originalVal - delta);
                modified[f] = NumOps.FromDouble(testVal);
                modifiedMatrix = CreateSingleRowMatrix(modified);
                pred = _predictFunction(modifiedMatrix);

                predictedClass = GetPredictedClass(pred);
                if (predictedClass == foilClass && Math.Abs(testVal - originalVal) < minFlipDistance)
                {
                    flipValue = testVal;
                    minFlipDistance = Math.Abs(testVal - originalVal);
                }

                if (flipValue.HasValue && minFlipDistance < _perturbationStep * 2)
                    break;
            }

            if (flipValue.HasValue)
            {
                double change = flipValue.Value - originalVal;
                string direction = change > 0 ? "increased" : "decreased";

                pertinentNegatives.Add(new PertinentFeature<T>
                {
                    FeatureIndex = f,
                    FeatureName = _featureNames != null && f < _featureNames.Length ? _featureNames[f] : $"Feature {f}",
                    CurrentValue = instance[f],
                    ContrastValue = NumOps.FromDouble(flipValue.Value),
                    ImpactOnFact = NumOps.FromDouble(minFlipDistance),
                    Explanation = $"If {direction} by {Math.Abs(change):F2}, would predict '{GetClassName(foilClass)}'"
                });
            }
        }

        return pertinentNegatives
            .OrderBy(p => NumOps.ToDouble(p.ImpactOnFact))
            .Take(5)
            .ToList();
    }

    /// <summary>
    /// Computes feature contributions to the fact-foil decision.
    /// </summary>
    private Dictionary<int, T> ComputeFeatureContributions(Vector<T> instance, int factClass, int foilClass)
    {
        var contributions = new Dictionary<int, T>();
        double epsilon = 1e-4;

        for (int f = 0; f < _numFeatures; f++)
        {
            // Compute gradient of (fact_score - foil_score) w.r.t. this feature
            var instancePlus = instance.Clone();
            var instanceMinus = instance.Clone();

            instancePlus[f] = NumOps.FromDouble(NumOps.ToDouble(instance[f]) + epsilon);
            instanceMinus[f] = NumOps.FromDouble(NumOps.ToDouble(instance[f]) - epsilon);

            double diffPlus = GetScoreDifference(instancePlus, factClass, foilClass);
            double diffMinus = GetScoreDifference(instanceMinus, factClass, foilClass);

            double gradient = (diffPlus - diffMinus) / (2 * epsilon);
            double contribution = gradient * NumOps.ToDouble(instance[f]);

            contributions[f] = NumOps.FromDouble(contribution);
        }

        return contributions;
    }

    /// <summary>
    /// Gets the score difference between fact and foil classes.
    /// </summary>
    private double GetScoreDifference(Vector<T> instance, int factClass, int foilClass)
    {
        var matrix = CreateSingleRowMatrix(instance);
        var pred = _predictFunction(matrix);
        int numClassesNeeded = Math.Max(factClass, foilClass) + 1;
        if (pred.Length < numClassesNeeded)
        {
            pred = ExpandToClassProbabilities(pred, numClassesNeeded);
        }
        double factScore = NumOps.ToDouble(pred[factClass]);
        double foilScore = NumOps.ToDouble(pred[foilClass]);
        return factScore - foilScore;
    }

    /// <summary>
    /// Gets the predicted class from scores.
    /// </summary>
    private int GetPredictedClass(Vector<T> predictions)
    {
        int maxIdx = 0;
        double maxVal = double.MinValue;
        for (int i = 0; i < predictions.Length; i++)
        {
            double val = NumOps.ToDouble(predictions[i]);
            if (val > maxVal)
            {
                maxVal = val;
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    private string GetClassName(int classIndex)
    {
        return _classNames != null && classIndex < _classNames.Length
            ? _classNames[classIndex]
            : $"Class {classIndex}";
    }

    private Matrix<T> CreateSingleRowMatrix(Vector<T> row)
    {
        var matrix = new Matrix<T>(1, row.Length);
        for (int j = 0; j < row.Length; j++)
            matrix[0, j] = row[j];
        return matrix;
    }

    /// <summary>
    /// Expands a per-row prediction vector into a per-class probability vector.
    /// When the predict function is a batch classifier returning one argmax per row,
    /// and we call it with a single row, the result has length 1 (the predicted class index).
    /// This method creates a pseudo-probability vector suitable for contrastive analysis.
    /// </summary>
    private static Vector<T> ExpandToClassProbabilities(Vector<T> batchPredictions, int numClasses)
    {
        var classProbs = new Vector<T>(numClasses);
        // Interpret the scalar output as the predicted class index
        int predictedClass = (int)Math.Round(NumOps.ToDouble(batchPredictions[0]));
        if (predictedClass < 0) predictedClass = 0;
        if (predictedClass >= numClasses) predictedClass = numClasses - 1;

        // Create a pseudo-probability distribution: high probability at predicted class
        for (int c = 0; c < numClasses; c++)
        {
            classProbs[c] = c == predictedClass
                ? NumOps.FromDouble(0.9)
                : NumOps.FromDouble(0.1 / (numClasses - 1));
        }

        return classProbs;
    }
}

/// <summary>
/// Represents a pertinent feature in a contrastive explanation.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class PertinentFeature<T>
{
    /// <summary>
    /// Gets or sets the feature index.
    /// </summary>
    public int FeatureIndex { get; set; }

    /// <summary>
    /// Gets or sets the feature name.
    /// </summary>
    public string FeatureName { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the current value of the feature.
    /// </summary>
    public T CurrentValue { get; set; } = default!;

    /// <summary>
    /// Gets or sets the contrasting value (what would lead to foil).
    /// </summary>
    public T ContrastValue { get; set; } = default!;

    /// <summary>
    /// Gets or sets the impact on the fact prediction.
    /// </summary>
    public T ImpactOnFact { get; set; } = default!;

    /// <summary>
    /// Gets or sets a human-readable explanation.
    /// </summary>
    public string Explanation { get; set; } = string.Empty;
}

/// <summary>
/// Represents the result of a Contrastive explanation.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ContrastiveExplanation<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets or sets the input instance.
    /// </summary>
    public Vector<T> Input { get; set; } = new Vector<T>(0);

    /// <summary>
    /// Gets or sets the model prediction.
    /// </summary>
    public Vector<T> Prediction { get; set; } = new Vector<T>(0);

    /// <summary>
    /// Gets or sets the fact class (what was predicted).
    /// </summary>
    public int FactClass { get; set; }

    /// <summary>
    /// Gets or sets the foil class (the alternative).
    /// </summary>
    public int FoilClass { get; set; }

    /// <summary>
    /// Gets or sets the name of the fact class.
    /// </summary>
    public string FactClassName { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the name of the foil class.
    /// </summary>
    public string FoilClassName { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the score for the fact class.
    /// </summary>
    public T FactScore { get; set; } = default!;

    /// <summary>
    /// Gets or sets the score for the foil class.
    /// </summary>
    public T FoilScore { get; set; } = default!;

    /// <summary>
    /// Gets or sets the pertinent positives (features supporting the fact).
    /// </summary>
    public List<PertinentFeature<T>> PertinentPositives { get; set; } = new();

    /// <summary>
    /// Gets or sets the pertinent negatives (features that could flip to foil).
    /// </summary>
    public List<PertinentFeature<T>> PertinentNegatives { get; set; } = new();

    /// <summary>
    /// Gets or sets feature contributions to the fact-vs-foil decision.
    /// </summary>
    public Dictionary<int, T> FeatureContributions { get; set; } = new();

    /// <summary>
    /// Gets or sets the feature names.
    /// </summary>
    public string[] FeatureNames { get; set; } = Array.Empty<string>();

    /// <summary>
    /// Returns a human-readable summary.
    /// </summary>
    public override string ToString()
    {
        var lines = new List<string>
        {
            $"Contrastive Explanation: Why '{FactClassName}' and not '{FoilClassName}'?",
            $"  Fact score: {NumOps.ToDouble(FactScore):F4}",
            $"  Foil score: {NumOps.ToDouble(FoilScore):F4}",
            ""
        };

        if (PertinentPositives.Count > 0)
        {
            lines.Add("Pertinent Positives (why it IS this class):");
            foreach (var pp in PertinentPositives)
            {
                lines.Add($"  - {pp.FeatureName} = {NumOps.ToDouble(pp.CurrentValue):F2}");
                lines.Add($"    {pp.Explanation}");
            }
            lines.Add("");
        }

        if (PertinentNegatives.Count > 0)
        {
            lines.Add("Pertinent Negatives (how to get the other class):");
            foreach (var pn in PertinentNegatives)
            {
                lines.Add($"  - {pn.FeatureName}: {NumOps.ToDouble(pn.CurrentValue):F2} -> {NumOps.ToDouble(pn.ContrastValue):F2}");
                lines.Add($"    {pn.Explanation}");
            }
            lines.Add("");
        }

        // Top contributing features
        var topContrib = FeatureContributions
            .OrderByDescending(kvp => Math.Abs(NumOps.ToDouble(kvp.Value)))
            .Take(5);

        lines.Add("Top Contributing Features:");
        foreach (var kvp in topContrib)
        {
            string name = kvp.Key < FeatureNames.Length ? FeatureNames[kvp.Key] : $"Feature {kvp.Key}";
            double contrib = NumOps.ToDouble(kvp.Value);
            string direction = contrib > 0 ? "supports" : "opposes";
            lines.Add($"  {name}: {direction} '{FactClassName}' ({contrib:+0.0000;-0.0000})");
        }

        return string.Join(Environment.NewLine, lines);
    }
}

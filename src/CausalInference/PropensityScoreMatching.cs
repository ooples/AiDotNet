using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

using AiDotNet.Models.Parameters;

namespace AiDotNet.CausalInference;

/// <summary>
/// Implements Propensity Score Matching for causal effect estimation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Propensity Score Matching (PSM) estimates treatment effects by matching treated individuals
/// to control individuals with similar propensity scores (probability of treatment).
/// </para>
/// <para>
/// <b>For Beginners:</b> PSM works by finding "twins" - pairs of people who were equally
/// likely to be treated but one actually was and one wasn't. By comparing these matched
/// pairs, we can estimate the causal effect of treatment.
///
/// How it works:
/// 1. Estimate propensity scores (probability of treatment for each person)
/// 2. For each treated person, find a control person with the most similar propensity score
/// 3. Compare outcomes between matched pairs
/// 4. Average the differences to get the treatment effect
///
/// The key insight: If two people have the same propensity score, the "assignment"
/// of treatment is essentially random between them. This mimics a randomized experiment.
///
/// Example - Job Training Study:
/// - Person A: propensity=0.7, treated=yes, salary=$50,000
/// - Person B: propensity=0.7, treated=no, salary=$45,000
/// - Estimated effect: $50,000 - $45,000 = $5,000
///
/// Advantages:
/// - Intuitive and easy to explain
/// - Creates a matched sample that looks like a mini experiment
/// - Can visually verify match quality
///
/// Limitations:
/// - Discards unmatched observations
/// - Sensitive to match quality
/// - Only adjusts for measured confounders
///
/// References:
/// - Rosenbaum &amp; Rubin (1983). "The Central Role of the Propensity Score"
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var covariates = new Matrix&lt;double&gt;(new double[,]
/// {
///     { 45, 1 }, { 52, 0 }, { 38, 1 }, { 61, 0 }, { 47, 1 }, { 55, 0 }
/// });
/// var treatment = new Vector&lt;int&gt;(new int[] { 0, 1, 0, 1, 0, 1 });   // who was treated
/// var outcome = new Vector&lt;double&gt;(new double[] { 3.1, 7.2, 2.8, 8.0, 3.4, 7.6 });
///
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new PropensityScoreMatching&lt;double&gt;(caliper: 0.2, matchRatio: 1))
///     .Build(covariates, treatment, outcome);
///
/// var (att, se) = result.EstimateATT(covariates, treatment, outcome);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Healthcare)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.Statistical)]
[ModelCategory(ModelCategory.InstanceBased)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
    [ResearchPaper("The Central Role of the Propensity Score in Observational Studies for Causal Effects", "https://doi.org/10.1093/biomet/70.1.41")]
public partial class PropensityScoreMatching<T> : CausalModelBase<T>
{

    /// <inheritdoc />
    /// <remarks>The propensity-score coefficients, and the same off-by-one fix as CausalForest.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(new VectorFieldParameterSource<T>(
            () => _propensityCoefficients,
            value =>
            {
                _propensityCoefficients = value;
                NumFeatures = value.Length - 1;
            }));
    }
    /// <summary>
    /// Stores the logistic regression coefficients for propensity score estimation.
    /// </summary>
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _propensityCoefficients;

    /// <summary>
    /// The caliper width for matching (maximum allowed propensity score difference).
    /// </summary>
    private double _caliper;

    /// <summary>
    /// Whether to use replacement in matching.
    /// </summary>
    private bool _withReplacement;

    /// <summary>
    /// Number of matches per treated individual.
    /// </summary>
    private int _matchRatio;

    /// <summary>
    /// Random number generator for tie-breaking.
    /// </summary>
    private Random _random;

    /// <summary>
    /// Cached treatment vector from fitting.
    /// </summary>
    [Scratch]
    private Vector<int>? _cachedTreatment;

    /// <summary>
    /// Cached outcome vector from fitting.
    /// </summary>
    [Scratch]
    private Vector<T>? _cachedOutcome;

    /// <summary>
    /// Cached feature matrix from fitting.
    /// </summary>
    [Scratch]
    private Matrix<T>? _cachedFeatures;

    /// <summary>
    /// Gets the model type.
    /// </summary>

    /// <summary>
    /// Initializes a new instance of the PropensityScoreMatching class.
    /// </summary>
    /// <param name="caliper">Maximum allowed propensity score difference for matching. Default is 0.2.</param>
    /// <param name="withReplacement">Whether a control can be matched to multiple treated. Default is true.</param>
    /// <param name="matchRatio">Number of controls to match per treated. Default is 1.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Parameters control how strict matching is:
    ///
    /// - caliper: How similar propensity scores must be. Smaller = stricter matching.
    ///   If caliper=0.05, only pairs with propensity scores within 0.05 of each other match.
    ///
    /// - withReplacement: If true, one control can match multiple treated individuals.
    ///   Useful when treated outnumber controls, but reduces effective sample size.
    ///
    /// - matchRatio: Number of controls per treated. More matches = less variance but
    ///   potentially worse match quality.
    ///
    /// Usage:
    /// <code>
    /// var features = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
    /// var outcome = new Vector&lt;double&gt;(new double[] { 0.0, 1.0, 0.0, 1.0 });
    /// var treatment = new Vector&lt;int&gt;(new int[] { 0, 1, 0, 1 });   // who was treated
    /// var psm = new PropensityScoreMatching&lt;double&gt;(caliper: 0.1, matchRatio: 2);
    /// var (ate, se) = psm.EstimateATE(features, treatment, outcome);
    /// </code>
    /// </para>
    /// </remarks>
    public PropensityScoreMatching(
        double caliper = 0.2,
        bool withReplacement = true,
        int matchRatio = 1,
        int? seed = null)
        : base(seed)
    {
        if (caliper <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(caliper),
                "Caliper must be positive. A negative caliper would prevent any matches.");
        }

        if (matchRatio < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(matchRatio),
                "Match ratio must be at least 1.");
        }

        _caliper = caliper;
        _withReplacement = withReplacement;
        _matchRatio = matchRatio;
        _random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Fits the propensity score model to the data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This step learns how to predict treatment assignment from covariates.
    /// The fitted model is then used to calculate propensity scores for matching.
    /// </para>
    /// </remarks>
    public void Fit(Matrix<T> x, Vector<int> treatment)
    {
        ValidateFitData(x, treatment);
        NumFeatures = x.Columns;

        // Fit logistic regression for propensity scores
        _propensityCoefficients = FitLogisticRegression(x, treatment);

        // Cache data for later predictions
        _cachedFeatures = x;
        _cachedTreatment = treatment;
        _cachedOutcome = null; // Clear stale outcomes from prior fits

        IsFitted = true;
    }

    /// <summary>
    /// Fits the causal model using the ICausalModel interface signature.
    /// </summary>
    /// <param name="features">The feature matrix (covariates).</param>
    /// <param name="treatment">Treatment indicators as generic type (0 or 1).</param>
    /// <param name="outcome">The outcome variable.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method converts the generic treatment vector to integer format
    /// and fits the propensity score model. The outcome is cached for predictions.
    /// </para>
    /// </remarks>
    public override void Fit(Matrix<T> features, Vector<T> treatment, Vector<T> outcome)
    {
        // Convert treatment vector to int with validation
        var treatmentInt = RoundBinaryTreatment(treatment);

        // Validate outcome alignment before caching
        if (features.Rows != outcome.Length)
        {
            throw new ArgumentException(
                $"Number of samples in X ({features.Rows}) must match number of outcomes ({outcome.Length}).",
                nameof(outcome));
        }

        // Call the original fit method first (this clears _cachedOutcome for fresh fits)
        Fit(features, treatmentInt);

        // Cache outcome AFTER the base Fit call, since Fit(x, treatment) nulls _cachedOutcome
        _cachedOutcome = outcome;
    }

    /// <summary>
    /// Estimates treatment effects for individuals using matched pairs.
    /// </summary>
    /// <param name="features">The feature matrix for which to estimate effects.</param>
    /// <returns>A vector of estimated treatment effects.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Propensity Score Matching doesn't directly model heterogeneous effects.
    /// For each individual, we return the average treatment effect estimated from the training data.
    /// For personalized predictions, consider CausalForest or DoublyRobustEstimator.
    /// </para>
    /// </remarks>
    public override Vector<T> EstimateTreatmentEffect(Matrix<T> features)
    {
        EnsureFitted();

        if (_cachedFeatures is null || _cachedTreatment is null || _cachedOutcome is null)
        {
            throw new InvalidOperationException(
                "Model must be fitted with outcome data to estimate treatment effects. " +
                "Use Fit(features, treatment, outcome) instead of Fit(features, treatment).");
        }

        // PSM doesn't naturally produce heterogeneous effects
        // We return the estimated ATE for all individuals
        var (ate, _) = EstimateATE(_cachedFeatures, _cachedTreatment, _cachedOutcome);
        return Constant(features.Rows, ate);
    }

    /// <summary>
    /// Predicts outcomes under treatment for the given features.
    /// </summary>
    /// <param name="features">The feature matrix.</param>
    /// <returns>Predicted outcomes if treated.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This estimates what the outcome would be if each individual received treatment.
    /// For PSM, we use the average outcome of treated individuals with similar propensity scores.
    /// </para>
    /// </remarks>
    public override Vector<T> PredictTreated(Matrix<T> features)
    {
        EnsureFitted();

        if (_cachedFeatures is null || _cachedTreatment is null || _cachedOutcome is null)
        {
            throw new InvalidOperationException("Model must be fitted with outcome data for treated predictions.");
        }

        // Inverse-distance weighted outcomes of the treated individuals within the caliper
        return CaliperWeightedOutcome(features, _cachedFeatures, _cachedTreatment, _cachedOutcome, 1, "treated");
    }

    /// <summary>
    /// Predicts outcomes under control for the given features.
    /// </summary>
    /// <param name="features">The feature matrix.</param>
    /// <returns>Predicted outcomes if not treated.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This estimates what the outcome would be if each individual did not receive treatment.
    /// For PSM, we use the average outcome of control individuals with similar propensity scores.
    /// </para>
    /// </remarks>
    public override Vector<T> PredictControl(Matrix<T> features)
    {
        EnsureFitted();

        if (_cachedFeatures is null || _cachedTreatment is null || _cachedOutcome is null)
        {
            throw new InvalidOperationException("Model must be fitted with outcome data for control predictions.");
        }

        // Inverse-distance weighted outcomes of the control individuals within the caliper
        return CaliperWeightedOutcome(features, _cachedFeatures, _cachedTreatment, _cachedOutcome, 0, "control");
    }

    /// <summary>
    /// For each query row, the outcomes of the cached rows in <paramref name="group"/> whose propensity score is
    /// within the caliper, weighted by 1 / (distance + ε) with ε the larger of 1e-8 and T's resolution.
    /// </summary>
    private Vector<T> CaliperWeightedOutcome(
        Matrix<T> features, Matrix<T> cachedFeatures, Vector<int> cachedTreatment, Vector<T> cachedOutcome,
        int group, string groupName)
    {
        var queryScores = EstimatePropensityScores(features).AsTensor();
        var referenceScores = EstimatePropensityScores(cachedFeatures).AsTensor();
        int m = queryScores.Length;
        int n = referenceScores.Length;
        if (m == 0)
            return new Vector<T>(0);

        var inGroup = Engine.Reshape(Engine.TensorEquals(cachedTreatment.AsTensor(), group).Cast<T>(), new[] { 1, n });
        T outcomeScale = PeakMagnitude(cachedOutcome.AsTensor());
        var scaledOutcome = Engine.Reshape(Engine.TensorDivideScalar(cachedOutcome.AsTensor(), outcomeScale), new[] { n, 1 });
        var reference = Engine.Reshape(referenceScores, new[] { 1, n });
        T caliper = NumOps.FromDouble(_caliper);
        T epsilon = NumOps.FromDouble(Math.Max(1e-8, Math.Pow(2.0, -NumOps.PrecisionBits)));

        // Queries are processed in blocks so the [block, n] weights stay a bounded size.
        int blockRows = Math.Max(1, (1 << 22) / Math.Max(1, n));
        var weightBlocks = new List<Tensor<T>>();
        var outcomeBlocks = new List<Tensor<T>>();
        for (int start = 0; start < m; start += blockRows)
        {
            int rows = Math.Min(blockRows, m - start);
            var query = Engine.Reshape(Engine.TensorNarrow(queryScores, 0, start, rows), new[] { rows, 1 });
            var distance = Engine.TensorAbs(Engine.TensorSubtract(
                Engine.TensorBroadcastTo(query, new[] { rows, n }), Engine.TensorBroadcastTo(reference, new[] { rows, n })));
            var withinCaliper = Engine.TensorSubtract(Engine.TensorOnesLike(distance), Engine.TensorGreaterThan(distance, caliper));
            var eligible = Engine.TensorMultiply(withinCaliper, Engine.TensorBroadcastTo(inGroup, new[] { rows, n }));
            var weight = Engine.TensorMultiply(eligible, Engine.TensorReciprocal(Engine.TensorAddScalar(distance, epsilon)));

            weightBlocks.Add(Engine.ReduceSum(weight, new[] { 1 }, false));
            outcomeBlocks.Add(Engine.Reshape(Engine.TensorMatMul(weight, scaledOutcome), new[] { rows }));
        }

        var sumWeight = Engine.TensorConcatenate(weightBlocks.ToArray(), 0);
        var empty = Engine.TensorNonzero(Engine.TensorEquals(sumWeight, NumOps.Zero));
        if (empty.Length > 0)
        {
            throw new InvalidOperationException(
                $"No {groupName} matches found within caliper {_caliper} for query at index {empty[0, 0]}.");
        }

        var sumOutcome = Engine.TensorConcatenate(outcomeBlocks.ToArray(), 0);
        return Engine.TensorMultiplyScalar(Engine.TensorDivide(sumOutcome, sumWeight), outcomeScale).ToVector();
    }

    private void ValidateFitData(Matrix<T> x, Vector<int> treatment)
    {
        if (x.Rows != treatment.Length)
        {
            throw new ArgumentException(
                $"Number of samples in X ({x.Rows}) must match number of treatments ({treatment.Length}).");
        }

        CountTreatedRequiringBinary(treatment);
    }

    /// <summary>
    /// Estimates propensity scores using the fitted model.
    /// </summary>
    protected override Vector<T> EstimatePropensityScoresCore(Matrix<T> x)
    {
        if (_propensityCoefficients is null)
        {
            throw new InvalidOperationException("Model must be fitted first.");
        }

        return PredictPropensityWithCoefficients(x, _propensityCoefficients);
    }

    /// <summary>
    /// Estimates the Average Treatment Effect using matched pairs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> ATE is estimated by:
    /// 1. Computing propensity scores
    /// 2. Matching treated to controls (the ATT) and controls to treated (the ATC) with similar scores
    /// 3. Averaging the two, weighted by how many individuals of each group found a match
    /// </para>
    /// <para>
    /// This is DoWhy's propensity-score matching estimator. Matching only from the treated side estimates the
    /// effect on the treated, which is what <see cref="EstimateATT"/> returns.
    /// </para>
    /// </remarks>
    public override (T estimate, T standardError) EstimateATE(
        Matrix<T> x, Vector<int> treatment, Vector<T> outcome)
    {
        if (!IsFitted)
        {
            Fit(x, treatment);
        }

        ValidateCausalData(x, treatment, outcome);

        T ate = EstimateATEInternal(x, treatment, outcome);
        T se = CalculateBootstrapStandardError(
            (xB, tB, oB) => EstimateATEInternal(xB, tB, oB),
            x, treatment, outcome);

        return (ate, se);
    }

    private T EstimateATEInternal(Matrix<T> x, Vector<int> treatment, Vector<T> outcome)
    {
        var propensityScores = EstimatePropensityScores(x).AsTensor();

        // Treated matched to controls give the ATT, controls matched to treated the ATC; the ATE is their average
        // weighted by the matched units of each group (DoWhy's propensity-score matching estimator).
        var att = MatchedEffect(propensityScores, treatment, outcome, 1);
        var atc = MatchedEffect(propensityScores, treatment, outcome, 0);
        int matchedUnits = att.MatchedUnits + atc.MatchedUnits;
        if (matchedUnits == 0)
        {
            throw new InvalidOperationException(
                $"No matches found within caliper {_caliper}. ATE is undefined. " +
                "Consider relaxing the caliper or adjusting the match ratio.");
        }

        T weighted = NumOps.Add(
            NumOps.Multiply(att.Effect, NumOps.FromDouble(att.MatchedUnits)),
            NumOps.Multiply(atc.Effect, NumOps.FromDouble(atc.MatchedUnits)));
        return NumOps.Divide(weighted, NumOps.FromDouble(matchedUnits));
    }

    /// <summary>The ATT from matching each treated individual to controls.</summary>
    private T EstimateATTInternal(Matrix<T> x, Vector<int> treatment, Vector<T> outcome)
    {
        var att = MatchedEffect(EstimatePropensityScores(x).AsTensor(), treatment, outcome, 1);
        if (att.MatchedUnits == 0)
        {
            throw new InvalidOperationException(
                $"No matches found within caliper {_caliper}. ATT is undefined. " +
                "Consider relaxing the caliper or adjusting the match ratio.");
        }

        return att.Effect;
    }

    /// <summary>
    /// Matches each individual of <paramref name="group"/> to the other group and averages, over the matched
    /// individuals, treated outcome minus control outcome (the matches' outcomes averaged first).
    /// </summary>
    private (T Effect, int MatchedUnits, Tensor<T> UnitEffect, Tensor<T> Matched) MatchedEffect(
        Tensor<T> propensityScores, Vector<int> treatment, Vector<T> outcome, int group)
    {
        int n = treatment.Length;
        var (sources, targets) = PerformMatching(propensityScores, treatment, group);

        var y = outcome.AsTensor();
        T scale = PeakMagnitude(y);
        var scaled = Engine.TensorDivideScalar(y, scale);
        var zeros = new Tensor<T>(new[] { n });
        var matchSum = Engine.TensorScatterAdd(zeros, sources, Engine.TensorIndexSelect(scaled, targets, 0), 0);
        var matchCount = Engine.TensorScatterAdd(new Tensor<T>(new[] { n }), sources, Engine.TensorOnesLike(Engine.TensorIndexSelect(scaled, targets, 0)), 0);
        var matched = Engine.TensorGreaterThan(matchCount, NumOps.Zero);
        int matchedUnits = Engine.TensorCountNonzero(matched);

        // Own outcome minus the mean matched outcome, signed treated − control
        var safeCount = Engine.TensorWhere(matched, matchCount, Engine.TensorOnesLike(matchCount));
        var difference = Engine.TensorSubtract(scaled, Engine.TensorDivide(matchSum, safeCount));
        if (group == 0)
            difference = Engine.TensorNegate(difference);
        var unitEffect = Engine.TensorMultiplyScalar(Engine.TensorMultiply(matched, difference), scale);

        T effect = matchedUnits == 0
            ? NumOps.Zero
            : NumOps.Divide(Engine.TensorSum(unitEffect), NumOps.FromDouble(matchedUnits));
        return (effect, matchedUnits, unitEffect, matched);
    }

    /// <summary>
    /// Estimates the Average Treatment Effect on the Treated.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Each treated person is matched to controls with similar
    /// propensity scores; the ATT is the average, over the matched treated, of their
    /// outcome minus the mean outcome of their matches.
    /// </para>
    /// </remarks>
    public override (T estimate, T standardError) EstimateATT(
        Matrix<T> x, Vector<int> treatment, Vector<T> outcome)
    {
        if (!IsFitted)
        {
            Fit(x, treatment);
        }

        ValidateCausalData(x, treatment, outcome);

        // Matching from treated to controls estimates the effect on the treated
        T att = EstimateATTInternal(x, treatment, outcome);
        T se = CalculateBootstrapStandardError(
            (xB, tB, oB) => EstimateATTInternal(xB, tB, oB),
            x, treatment, outcome);

        return (att, se);
    }

    /// <summary>
    /// Estimates individual treatment effects using matched pairs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For each matched person, we estimate their individual
    /// effect as the difference between the treated and control outcomes of them and
    /// their match(es); unmatched people get the average effect.
    /// </para>
    /// </remarks>
    public override Vector<T> EstimateCATEPerIndividual(
        Matrix<T> x, Vector<int> treatment, Vector<T> outcome)
    {
        if (!IsFitted)
        {
            Fit(x, treatment);
        }

        ValidateCausalData(x, treatment, outcome);

        var propensityScores = EstimatePropensityScores(x).AsTensor();
        var treatedSide = MatchedEffect(propensityScores, treatment, outcome, 1);
        var controlSide = MatchedEffect(propensityScores, treatment, outcome, 0);

        // Average effect for the unmatched
        T avgEffect = EstimateATE(x, treatment, outcome).estimate;

        // Individual effects for the matched: each treated against its matched controls, each control against its
        // matched treated (a unit is a source in only one direction, so the two never overlap)
        var matched = Engine.TensorAdd(treatedSide.Matched, controlSide.Matched);
        var individual = Engine.TensorAdd(treatedSide.UnitEffect, controlSide.UnitEffect);
        return Engine.TensorWhere(matched, individual, Tensor<T>.CreateDefault(new[] { x.Rows }, avgEffect)).ToVector();
    }

    /// <summary>
    /// Predicts treatment effects for new individuals.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> PSM doesn't directly model heterogeneous effects, so this
    /// returns the overall ATE for all individuals. For personalized predictions,
    /// consider using CausalForest or DoublyRobustEstimator.
    /// </para>
    /// </remarks>
    public override Vector<T> PredictTreatmentEffect(Matrix<T> x)
    {
        EnsureFitted();

        if (_cachedFeatures is null || _cachedTreatment is null || _cachedOutcome is null)
        {
            throw new InvalidOperationException(
                "Model must be fitted with outcome data to predict treatment effects. " +
                "Use Fit(features, treatment, outcome) instead of Fit(features, treatment).");
        }

        // PSM doesn't model heterogeneous effects well
        // Return a constant effect (the ATE from training data)
        var (ate, _) = EstimateATE(_cachedFeatures, _cachedTreatment, _cachedOutcome);
        return Constant(x.Rows, ate);
    }

    /// <summary>
    /// Performs the actual matching of treated to control individuals.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is where the "matching" happens:
    /// - For each person in the group, find people of the other group with similar propensity scores
    /// - Only keep matches within the caliper
    /// - If without replacement, each control can only be matched once
    /// </para>
    /// </remarks>
    private (Tensor<int> Sources, Tensor<int> Targets) PerformMatching(
        Tensor<T> propensityScores, Vector<int> treatment, int group)
    {
        // Get indices of the matched group and of the other group
        var assignment = treatment.AsTensor();
        var sourceCoordinates = Engine.TensorNonzero(Engine.TensorEquals(assignment, group));
        var targetCoordinates = Engine.TensorNonzero(Engine.TensorNotEquals(assignment, group));
        var sourceIndices = Engine.Reshape(sourceCoordinates, new[] { sourceCoordinates.Length });
        var targetIndices = Engine.Reshape(targetCoordinates, new[] { targetCoordinates.Length });
        int targetCount = targetIndices.Length;

        // Shuffle the sources for random matching order (one draw per source, as before)
        var shuffled = sourceIndices.ToArray().OrderBy(_ => _random.Next()).ToArray();
        var sourceScores = Engine.TensorIndexSelect(propensityScores, new Tensor<int>(shuffled, new[] { shuffled.Length }), 0);
        var targetScores = Engine.TensorIndexSelect(propensityScores, targetIndices, 0);

        T caliper = NumOps.FromDouble(_caliper);
        var available = Engine.TensorOnesLike(targetScores);
        var unavailable = Tensor<T>.CreateDefault(new[] { targetCount }, NumOps.MaxValue);
        var sources = new List<Tensor<int>>();
        var targets = new List<Tensor<int>>();

        for (int k = 0; k < shuffled.Length && targetCount > 0; k++)
        {
            // Distances to the still-available candidates within the caliper
            var score = Engine.TensorBroadcastTo(Engine.TensorNarrow(sourceScores, 0, k, 1), new[] { targetCount });
            var distance = Engine.TensorAbs(Engine.TensorSubtract(score, targetScores));
            var eligible = Engine.TensorMultiply(available,
                Engine.TensorSubtract(Engine.TensorOnesLike(distance), Engine.TensorGreaterThan(distance, caliper)));
            int candidates = Engine.TensorCountNonzero(eligible);
            if (candidates == 0)
                continue;

            // Sort by distance and take the closest matches
            int take = Math.Min(_matchRatio, candidates);
            var order = Engine.TensorArgsort(Engine.TensorWhere(eligible, distance, unavailable), 0, false);
            var chosen = Engine.TensorNarrow(order, 0, 0, take);
            targets.Add(Engine.TensorIndexSelect(targetIndices, chosen, 0));
            sources.Add(Tensor<int>.CreateDefault(new[] { take }, shuffled[k]));

            if (!_withReplacement)
            {
                available = Engine.TensorScatter(available, chosen, new Tensor<T>(new[] { take }), 0);
            }
        }

        if (sources.Count == 0)
            return (new Tensor<int>(new[] { 0 }), new Tensor<int>(new[] { 0 }));

        return (Engine.TensorConcatenate(sources.ToArray(), 0), Engine.TensorConcatenate(targets.ToArray(), 0));
    }

    /// <summary>
    /// Standard prediction - returns propensity scores.
    /// </summary>
    public override Vector<T> Predict(Matrix<T> input)
    {
        // Accept both the augmented [N, p+1] form (column 0 = treatment indicator,
        // columns 1.. = covariates) that Train()/CausalModelBase consumes, and the
        // plain [N, p] covariate form. Strip the treatment column when present so
        // propensity-coefficient indexing stays in range — mirrors TLearner/SLearner.
        if (input.Columns == NumFeatures + 1)
        {
            return EstimatePropensityScores(DropLeadingColumn(input));
        }

        return EstimatePropensityScores(input);
    }

    #region IFullModel Implementation

    /// <summary>
    /// Creates a new instance with specified parameters.
    /// </summary>
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newModel = new PropensityScoreMatching<T>(_caliper, _withReplacement, _matchRatio);
        newModel.SetParameters(parameters);
        return newModel;
    }

    /// <summary>
    /// Gets additional model data for serialization.
    /// </summary>
    protected override Dictionary<string, object> GetAdditionalModelData()
    {
        var data = new Dictionary<string, object>
        {
            { "Caliper", _caliper },
            { "WithReplacement", _withReplacement },
            { "MatchRatio", _matchRatio },
            { "RandomSeed", RandomSeed.HasValue ? RandomSeed.Value : (object)string.Empty }
        };

        if (_propensityCoefficients is not null)
        {
            data["PropensityCoefficients"] = _propensityCoefficients.ToArray();
        }

        return data;
    }

    /// <summary>
    /// Loads additional model data from deserialization.
    /// </summary>
    protected override void LoadAdditionalModelData(Newtonsoft.Json.Linq.JObject modelDataObj)
    {
        // Restore matching hyperparameters
        var caliperToken = modelDataObj["Caliper"];
        if (caliperToken is not null)
            _caliper = caliperToken.ToObject<double>();

        var withReplacementToken = modelDataObj["WithReplacement"];
        if (withReplacementToken is not null)
            _withReplacement = withReplacementToken.ToObject<bool>();

        var matchRatioToken = modelDataObj["MatchRatio"];
        if (matchRatioToken is not null)
            _matchRatio = matchRatioToken.ToObject<int>();

        // The generator's internal state cannot be restored, but the seed can — and the seed is what
        // makes a standard error reproducible, so a round-tripped model keeps producing the same one.
        var seedToken = modelDataObj["RandomSeed"];
        int? restoredSeed = seedToken is not null
            && seedToken.Type == Newtonsoft.Json.Linq.JTokenType.Integer
                ? seedToken.ToObject<int>()
                : null;
        RestoreRandomSeed(restoredSeed);
        _random = restoredSeed.HasValue
            ? RandomHelper.CreateSeededRandom(restoredSeed.Value)
            : RandomHelper.CreateSecureRandom();

        var coeffsToken = modelDataObj["PropensityCoefficients"];
        if (coeffsToken is not null)
        {
            var coeffs = coeffsToken.ToObject<T[]>();
            if (coeffs is not null && coeffs.Length > 0)
            {
                _propensityCoefficients = new Vector<T>(coeffs);
                NumFeatures = coeffs.Length - 1; // -1 for intercept
            }
        }
    }

    #endregion

    /// <summary>
    /// Gets the number of matched pairs.
    /// </summary>
    /// <param name="x">Covariate matrix.</param>
    /// <param name="treatment">Treatment indicators.</param>
    /// <returns>Number of successful matches.</returns>
    public int GetNumberOfMatches(Matrix<T> x, Vector<int> treatment)
    {
        if (!IsFitted)
        {
            Fit(x, treatment);
        }

        var propensityScores = EstimatePropensityScores(x).AsTensor();
        return PerformMatching(propensityScores, treatment, 1).Sources.Length;
    }

    /// <summary>
    /// Gets match quality statistics (balance check).
    /// </summary>
    /// <param name="x">Covariate matrix.</param>
    /// <param name="treatment">Treatment indicators.</param>
    /// <returns>Standardized mean differences before and after matching.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Standardized mean difference (SMD) measures how different
    /// the treated and control groups are on each covariate. After matching, SMD should
    /// be close to 0 (typically &lt; 0.1) indicating good balance.
    /// </para>
    /// </remarks>
    public (Vector<T> beforeSMD, Vector<T> afterSMD) GetMatchQuality(Matrix<T> x, Vector<int> treatment)
    {
        if (!IsFitted)
        {
            Fit(x, treatment);
        }

        var propensityScores = EstimatePropensityScores(x).AsTensor();
        var (sources, targets) = PerformMatching(propensityScores, treatment, 1);
        int n = x.Rows;

        // SMD is scale-free, so the covariates are scaled into [-1, 1] per column first (nothing squared overflows)
        var features = Tensor<T>.FromMatrix(x);
        var peak = Engine.ReduceMax(Engine.TensorAbs(features), new[] { 0 }, true);
        var columnScale = Engine.TensorWhere(Engine.TensorEquals(peak, NumOps.Zero), Engine.TensorOnesLike(peak), peak);
        var scaled = Engine.TensorDivide(features, Engine.TensorBroadcastTo(columnScale, new[] { n, x.Columns }));

        // Calculate SMD before matching: every treated against every other individual
        var treated = Engine.TensorEquals(treatment.AsTensor(), 1).Cast<T>();
        var control = Engine.TensorSubtract(Engine.TensorOnesLike(treated), treated);
        var beforeSMD = StandardizedMeanDifference(scaled, treated, control);

        // Calculate SMD after matching: the treated with a match against the distinct matched controls
        var ones = Tensor<T>.CreateDefault(new[] { sources.Length }, NumOps.One);
        var matchedTreated = Engine.TensorGreaterThan(Engine.TensorScatterAdd(new Tensor<T>(new[] { n }), sources, ones, 0), NumOps.Zero);
        var matchedControls = Engine.TensorGreaterThan(Engine.TensorScatterAdd(new Tensor<T>(new[] { n }), targets, ones, 0), NumOps.Zero);
        var afterSMD = StandardizedMeanDifference(scaled, matchedTreated, matchedControls);

        return (beforeSMD, afterSMD);
    }

    /// <summary>
    /// Per-column (mean₁ − mean₀) / √((s₁² + s₀²) / 2) between the rows in two 0/1 masks, with n − 1 sample
    /// variances (0 for a group of one); all zeros when either group is empty, and 0 for a column with no spread.
    /// </summary>
    private Vector<T> StandardizedMeanDifference(Tensor<T> features, Tensor<T> first, Tensor<T> second)
    {
        int n = features.Shape[0];
        int p = features.Shape[1];
        int firstCount = Engine.TensorCountNonzero(first);
        int secondCount = Engine.TensorCountNonzero(second);

        // Guard against divide-by-zero when a group has zero samples
        if (firstCount == 0 || secondCount == 0)
            return new Vector<T>(p);

        var (meanFirst, varianceFirst) = GroupMoments(features, first, firstCount, n, p);
        var (meanSecond, varianceSecond) = GroupMoments(features, second, secondCount, n, p);

        var pooledStd = Engine.TensorSqrt(Engine.TensorMultiplyScalar(Engine.TensorAdd(varianceFirst, varianceSecond), NumOps.FromDouble(0.5)));
        var spread = Engine.TensorGreaterThan(pooledStd, NumOps.Zero);
        var safeStd = Engine.TensorWhere(spread, pooledStd, Engine.TensorOnesLike(pooledStd));
        var difference = Engine.TensorDivide(Engine.TensorSubtract(meanFirst, meanSecond), safeStd);
        return Engine.TensorMultiply(spread, difference).ToVector();
    }

    /// <summary>Column means and n − 1 sample variances (0 for one row) of the masked rows.</summary>
    private (Tensor<T> Mean, Tensor<T> Variance) GroupMoments(Tensor<T> features, Tensor<T> mask, int count, int n, int p)
    {
        var rowMask = Engine.TensorBroadcastTo(Engine.Reshape(mask, new[] { n, 1 }), new[] { n, p });
        var mean = Engine.TensorDivideScalar(
            Engine.ReduceSum(Engine.TensorMultiply(features, rowMask), new[] { 0 }, false), NumOps.FromDouble(count));
        if (count < 2)
            return (mean, new Tensor<T>(new[] { p }));

        var deviation = Engine.TensorMultiply(
            Engine.TensorSubtract(features, Engine.TensorBroadcastTo(Engine.Reshape(mean, new[] { 1, p }), new[] { n, p })), rowMask);
        var variance = Engine.TensorDivideScalar(
            Engine.ReduceSum(Engine.TensorSquare(deviation), new[] { 0 }, false), NumOps.FromDouble(count - 1));
        return (mean, variance);
    }
}

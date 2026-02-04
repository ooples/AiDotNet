using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

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
public class PropensityScoreMatching<T> : CausalModelBase<T>
{
    /// <summary>
    /// Stores the logistic regression coefficients for propensity score estimation.
    /// </summary>
    private Vector<T>? _propensityCoefficients;

    /// <summary>
    /// The caliper width for matching (maximum allowed propensity score difference).
    /// </summary>
    private readonly double _caliper;

    /// <summary>
    /// Whether to use replacement in matching.
    /// </summary>
    private readonly bool _withReplacement;

    /// <summary>
    /// Number of matches per treated individual.
    /// </summary>
    private readonly int _matchRatio;

    /// <summary>
    /// Random number generator for tie-breaking.
    /// </summary>
    private readonly Random _random;

    /// <summary>
    /// Cached treatment vector from fitting.
    /// </summary>
    private Vector<int>? _cachedTreatment;

    /// <summary>
    /// Cached outcome vector from fitting.
    /// </summary>
    private Vector<T>? _cachedOutcome;

    /// <summary>
    /// Cached feature matrix from fitting.
    /// </summary>
    private Matrix<T>? _cachedFeatures;

    /// <summary>
    /// Gets the model type.
    /// </summary>
    public override ModelType GetModelType() => ModelType.PropensityScoreMatching;

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
        var treatmentInt = new Vector<int>(treatment.Length);
        for (int i = 0; i < treatment.Length; i++)
        {
            double raw = NumOps.ToDouble(treatment[i]);
            double rounded = Math.Round(raw);
            if (Math.Abs(raw - rounded) > 1e-6 || (rounded != 0 && rounded != 1))
            {
                throw new ArgumentException("Treatment indicators must be 0 or 1.", nameof(treatment));
            }
            treatmentInt[i] = (int)rounded;
        }

        // Validate outcome alignment before caching
        if (features.Rows != outcome.Length)
        {
            throw new ArgumentException(
                $"Number of samples in X ({features.Rows}) must match number of outcomes ({outcome.Length}).",
                nameof(outcome));
        }

        // Cache outcome for predictions
        _cachedOutcome = outcome;

        // Call the original fit method
        Fit(features, treatmentInt);
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

        var effects = new Vector<T>(features.Rows);
        for (int i = 0; i < features.Rows; i++)
        {
            effects[i] = ate;
        }

        return effects;
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

        var propensityScores = EstimatePropensityScores(features);
        var cachedPropensityScores = EstimatePropensityScores(_cachedFeatures);
        var predictions = new Vector<T>(features.Rows);

        for (int i = 0; i < features.Rows; i++)
        {
            double queryScore = NumOps.ToDouble(propensityScores[i]);

            // Find treated individuals with similar propensity scores
            double sumOutcome = 0;
            double sumWeight = 0;

            for (int j = 0; j < _cachedTreatment.Length; j++)
            {
                if (_cachedTreatment[j] == 1)
                {
                    double refScore = NumOps.ToDouble(cachedPropensityScores[j]);
                    double distance = Math.Abs(queryScore - refScore);

                    if (distance <= _caliper)
                    {
                        double weight = 1.0 / (distance + 1e-8);
                        sumOutcome += weight * NumOps.ToDouble(_cachedOutcome[j]);
                        sumWeight += weight;
                    }
                }
            }

            predictions[i] = sumWeight > 0
                ? NumOps.FromDouble(sumOutcome / sumWeight)
                : NumOps.Zero;
        }

        return predictions;
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

        var propensityScores = EstimatePropensityScores(features);
        var cachedPropensityScores = EstimatePropensityScores(_cachedFeatures);
        var predictions = new Vector<T>(features.Rows);

        for (int i = 0; i < features.Rows; i++)
        {
            double queryScore = NumOps.ToDouble(propensityScores[i]);

            // Find control individuals with similar propensity scores
            double sumOutcome = 0;
            double sumWeight = 0;

            for (int j = 0; j < _cachedTreatment.Length; j++)
            {
                if (_cachedTreatment[j] == 0)
                {
                    double refScore = NumOps.ToDouble(cachedPropensityScores[j]);
                    double distance = Math.Abs(queryScore - refScore);

                    if (distance <= _caliper)
                    {
                        double weight = 1.0 / (distance + 1e-8);
                        sumOutcome += weight * NumOps.ToDouble(_cachedOutcome[j]);
                        sumWeight += weight;
                    }
                }
            }

            predictions[i] = sumWeight > 0
                ? NumOps.FromDouble(sumOutcome / sumWeight)
                : NumOps.Zero;
        }

        return predictions;
    }

    private void ValidateFitData(Matrix<T> x, Vector<int> treatment)
    {
        if (x.Rows != treatment.Length)
        {
            throw new ArgumentException(
                $"Number of samples in X ({x.Rows}) must match number of treatments ({treatment.Length}).");
        }

        for (int i = 0; i < treatment.Length; i++)
        {
            if (treatment[i] != 0 && treatment[i] != 1)
            {
                throw new ArgumentException(
                    $"Treatment indicators must be 0 or 1. Found {treatment[i]} at index {i}.");
            }
        }
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
    /// 2. Matching treated to controls with similar scores
    /// 3. Computing the average outcome difference in matched pairs
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

        var propensityScores = EstimatePropensityScores(x);
        var matches = PerformMatching(propensityScores, treatment);

        // Calculate ATE from matched pairs
        double sumDiff = 0;
        int numMatches = 0;

        foreach (var (treatedIdx, controlIdxs) in matches)
        {
            double treatedOutcome = NumOps.ToDouble(outcome[treatedIdx]);
            foreach (int controlIdx in controlIdxs)
            {
                double controlOutcome = NumOps.ToDouble(outcome[controlIdx]);
                sumDiff += treatedOutcome - controlOutcome;
                numMatches++;
            }
        }

        T ate = NumOps.FromDouble(numMatches > 0 ? sumDiff / numMatches : 0);
        T se = CalculateBootstrapStandardError(
            (xB, tB, oB) => EstimateATEInternal(xB, tB, oB),
            x, treatment, outcome);

        return (ate, se);
    }

    private T EstimateATEInternal(Matrix<T> x, Vector<int> treatment, Vector<T> outcome)
    {
        var propensityScores = EstimatePropensityScores(x);
        var matches = PerformMatching(propensityScores, treatment);

        double sumDiff = 0;
        int numMatches = 0;

        foreach (var (treatedIdx, controlIdxs) in matches)
        {
            double treatedOutcome = NumOps.ToDouble(outcome[treatedIdx]);
            foreach (int controlIdx in controlIdxs)
            {
                double controlOutcome = NumOps.ToDouble(outcome[controlIdx]);
                sumDiff += treatedOutcome - controlOutcome;
                numMatches++;
            }
        }

        return NumOps.FromDouble(numMatches > 0 ? sumDiff / numMatches : 0);
    }

    /// <summary>
    /// Estimates the Average Treatment Effect on the Treated.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For PSM, ATT is the same as ATE since we're matching
    /// treated to controls (every match is centered on a treated individual).
    /// </para>
    /// </remarks>
    public override (T estimate, T standardError) EstimateATT(
        Matrix<T> x, Vector<int> treatment, Vector<T> outcome)
    {
        // For PSM, ATT = ATE since matching is done from treated to controls
        return EstimateATE(x, treatment, outcome);
    }

    /// <summary>
    /// Estimates individual treatment effects using matched pairs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For each treated person, we estimate their individual
    /// effect as the difference between their outcome and their matched control(s).
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

        var propensityScores = EstimatePropensityScores(x);
        var matches = PerformMatching(propensityScores, treatment);
        var cate = new Vector<T>(x.Rows);

        // Initialize with average effect for unmatched
        T avgEffect = EstimateATE(x, treatment, outcome).estimate;
        for (int i = 0; i < x.Rows; i++)
        {
            cate[i] = avgEffect;
        }

        // Set individual effects for matched treated
        foreach (var (treatedIdx, controlIdxs) in matches)
        {
            double treatedOutcome = NumOps.ToDouble(outcome[treatedIdx]);
            double avgControlOutcome = 0;
            foreach (int controlIdx in controlIdxs)
            {
                avgControlOutcome += NumOps.ToDouble(outcome[controlIdx]);
            }
            avgControlOutcome /= controlIdxs.Count;

            cate[treatedIdx] = NumOps.FromDouble(treatedOutcome - avgControlOutcome);
        }

        return cate;
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

        var effects = new Vector<T>(x.Rows);
        for (int i = 0; i < x.Rows; i++)
        {
            effects[i] = ate;
        }

        return effects;
    }

    /// <summary>
    /// Performs the actual matching of treated to control individuals.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is where the "matching" happens:
    /// - For each treated person, find controls with similar propensity scores
    /// - Only keep matches within the caliper
    /// - If without replacement, each control can only be matched once
    /// </para>
    /// </remarks>
    private Dictionary<int, List<int>> PerformMatching(Vector<T> propensityScores, Vector<int> treatment)
    {
        var matches = new Dictionary<int, List<int>>();
        var usedControls = new HashSet<int>();

        // Get indices of treated and control
        var treatedIndices = new List<int>();
        var controlIndices = new List<int>();

        for (int i = 0; i < treatment.Length; i++)
        {
            if (treatment[i] == 1)
                treatedIndices.Add(i);
            else
                controlIndices.Add(i);
        }

        // Shuffle treated for random matching order
        var shuffledTreated = treatedIndices.OrderBy(x => _random.Next()).ToList();

        foreach (int treatedIdx in shuffledTreated)
        {
            double treatedScore = NumOps.ToDouble(propensityScores[treatedIdx]);

            // Find best matches among controls
            var candidateMatches = new List<(int idx, double distance)>();

            foreach (int controlIdx in controlIndices)
            {
                if (!_withReplacement && usedControls.Contains(controlIdx))
                    continue;

                double controlScore = NumOps.ToDouble(propensityScores[controlIdx]);
                double distance = Math.Abs(treatedScore - controlScore);

                if (distance <= _caliper)
                {
                    candidateMatches.Add((controlIdx, distance));
                }
            }

            // Sort by distance and take top matches
            candidateMatches.Sort((a, b) => a.distance.CompareTo(b.distance));
            var selectedMatches = candidateMatches.Take(_matchRatio).ToList();

            if (selectedMatches.Count > 0)
            {
                matches[treatedIdx] = selectedMatches.Select(m => m.idx).ToList();

                if (!_withReplacement)
                {
                    foreach (var match in selectedMatches)
                    {
                        usedControls.Add(match.idx);
                    }
                }
            }
        }

        return matches;
    }

    /// <summary>
    /// Standard prediction - returns propensity scores.
    /// </summary>
    public override Vector<T> Predict(Matrix<T> input)
    {
        return EstimatePropensityScores(input);
    }

    #region IFullModel Implementation

    /// <summary>
    /// Gets the model parameters (propensity score coefficients).
    /// </summary>
    public override Vector<T> GetParameters()
    {
        if (_propensityCoefficients is null)
        {
            return new Vector<T>(0);
        }

        return _propensityCoefficients;
    }

    /// <summary>
    /// Sets the model parameters.
    /// </summary>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length > 0)
        {
            _propensityCoefficients = parameters;
            NumFeatures = parameters.Length - 1; // -1 for intercept
        }
    }

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
    /// Creates a new instance of this type.
    /// </summary>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new PropensityScoreMatching<T>(_caliper, _withReplacement, _matchRatio);
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
            { "MatchRatio", _matchRatio }
        };

        if (_propensityCoefficients is not null)
        {
            var coeffs = new double[_propensityCoefficients.Length];
            for (int i = 0; i < _propensityCoefficients.Length; i++)
            {
                coeffs[i] = NumOps.ToDouble(_propensityCoefficients[i]);
            }
            data["PropensityCoefficients"] = coeffs;
        }

        return data;
    }

    /// <summary>
    /// Loads additional model data from deserialization.
    /// </summary>
    protected override void LoadAdditionalModelData(Newtonsoft.Json.Linq.JObject modelDataObj)
    {
        var coeffsToken = modelDataObj["PropensityCoefficients"];
        if (coeffsToken is not null)
        {
            var coeffs = coeffsToken.ToObject<double[]>();
            if (coeffs is not null && coeffs.Length > 0)
            {
                _propensityCoefficients = new Vector<T>(coeffs.Length);
                for (int i = 0; i < coeffs.Length; i++)
                {
                    _propensityCoefficients[i] = NumOps.FromDouble(coeffs[i]);
                }
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

        var propensityScores = EstimatePropensityScores(x);
        var matches = PerformMatching(propensityScores, treatment);

        return matches.Values.Sum(m => m.Count);
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

        var propensityScores = EstimatePropensityScores(x);
        var matches = PerformMatching(propensityScores, treatment);

        var beforeSMD = new Vector<T>(x.Columns);
        var afterSMD = new Vector<T>(x.Columns);

        // Calculate SMD before matching
        for (int j = 0; j < x.Columns; j++)
        {
            double meanTreated = 0, meanControl = 0;
            double varTreated = 0, varControl = 0;
            int nTreated = 0, nControl = 0;

            // First pass: means
            for (int i = 0; i < x.Rows; i++)
            {
                double val = NumOps.ToDouble(x[i, j]);
                if (treatment[i] == 1)
                {
                    meanTreated += val;
                    nTreated++;
                }
                else
                {
                    meanControl += val;
                    nControl++;
                }
            }

            // Guard against divide-by-zero when a group has zero samples
            if (nTreated == 0 || nControl == 0)
            {
                beforeSMD[j] = NumOps.Zero;
                continue;
            }

            meanTreated /= nTreated;
            meanControl /= nControl;

            // Second pass: variances
            for (int i = 0; i < x.Rows; i++)
            {
                double val = NumOps.ToDouble(x[i, j]);
                if (treatment[i] == 1)
                {
                    varTreated += (val - meanTreated) * (val - meanTreated);
                }
                else
                {
                    varControl += (val - meanControl) * (val - meanControl);
                }
            }
            // Guard against divide-by-zero when group has < 2 samples
            varTreated = nTreated > 1 ? varTreated / (nTreated - 1) : 0;
            varControl = nControl > 1 ? varControl / (nControl - 1) : 0;

            double pooledStd = Math.Sqrt((varTreated + varControl) / 2);
            beforeSMD[j] = NumOps.FromDouble(pooledStd > 0 ? (meanTreated - meanControl) / pooledStd : 0);
        }

        // Calculate SMD after matching
        for (int j = 0; j < x.Columns; j++)
        {
            double meanTreated = 0, meanControl = 0;
            double varTreated = 0, varControl = 0;
            int nTreated = 0, nControl = 0;

            var matchedControlIndices = new HashSet<int>();
            foreach (var (_, controlIdxs) in matches)
            {
                foreach (int idx in controlIdxs)
                {
                    matchedControlIndices.Add(idx);
                }
            }

            // First pass: means
            foreach (var (treatedIdx, _) in matches)
            {
                meanTreated += NumOps.ToDouble(x[treatedIdx, j]);
                nTreated++;
            }
            foreach (int controlIdx in matchedControlIndices)
            {
                meanControl += NumOps.ToDouble(x[controlIdx, j]);
                nControl++;
            }

            if (nTreated > 0) meanTreated /= nTreated;
            if (nControl > 0) meanControl /= nControl;

            // Second pass: variances
            foreach (var (treatedIdx, _) in matches)
            {
                double val = NumOps.ToDouble(x[treatedIdx, j]);
                varTreated += (val - meanTreated) * (val - meanTreated);
            }
            foreach (int controlIdx in matchedControlIndices)
            {
                double val = NumOps.ToDouble(x[controlIdx, j]);
                varControl += (val - meanControl) * (val - meanControl);
            }

            if (nTreated > 1) varTreated /= (nTreated - 1);
            if (nControl > 1) varControl /= (nControl - 1);

            double pooledStd = Math.Sqrt((varTreated + varControl) / 2);
            afterSMD[j] = NumOps.FromDouble(pooledStd > 0 ? (meanTreated - meanControl) / pooledStd : 0);
        }

        return (beforeSMD, afterSMD);
    }
}

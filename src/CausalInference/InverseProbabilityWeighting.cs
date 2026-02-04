using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.CausalInference;

/// <summary>
/// Implements Inverse Probability Weighting (IPW) for causal effect estimation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// IPW estimates treatment effects by weighting observations inversely by their probability
/// of receiving their actual treatment status, creating a pseudo-population where treatment
/// is independent of confounders.
/// </para>
/// <para>
/// <b>For Beginners:</b> IPW works by giving more weight to "surprising" observations:
/// - A treated person who was unlikely to be treated gets high weight
/// - A control person who was likely to be treated gets high weight
///
/// Why? In observational data, treatment isn't random - some people are more likely
/// to be treated. IPW corrects for this by "up-weighting" under-represented cases.
///
/// Example - Job Training Study:
/// - Highly motivated person who didn't get training: weight = 1/(1-0.9) = 10
/// - Less motivated person who did get training: weight = 1/0.2 = 5
/// - This gives more influence to the "surprising" cases
///
/// Mathematical formula:
/// ATE = E[T×Y/e(X)] - E[(1-T)×Y/(1-e(X))]
///
/// Where:
/// - T = treatment (0 or 1)
/// - Y = outcome
/// - e(X) = propensity score = P(T=1|X)
///
/// Advantages:
/// - Uses all data (unlike matching which discards unmatched)
/// - Computationally simple
/// - Easy to combine with regression (augmented IPW)
///
/// Limitations:
/// - Can be unstable if propensity scores are extreme (near 0 or 1)
/// - Sensitive to propensity score model misspecification
///
/// References:
/// - Horvitz &amp; Thompson (1952). "A Generalization of Sampling Without Replacement"
/// - Robins, Rotnitzky &amp; Zhao (1994). "Estimation of Regression Coefficients"
/// </para>
/// </remarks>
public class InverseProbabilityWeighting<T> : CausalModelBase<T>
{
    /// <summary>
    /// Stores the logistic regression coefficients for propensity score estimation.
    /// </summary>
    private Vector<T>? _propensityCoefficients;

    /// <summary>
    /// Minimum propensity score to avoid extreme weights.
    /// </summary>
    private double _trimMin;

    /// <summary>
    /// Maximum propensity score to avoid extreme weights.
    /// </summary>
    private double _trimMax;

    /// <summary>
    /// Whether to use stabilized weights.
    /// </summary>
    private bool _stabilizedWeights;

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
    public override ModelType GetModelType() => ModelType.InverseProbabilityWeighting;

    /// <summary>
    /// Initializes a new instance of the InverseProbabilityWeighting class.
    /// </summary>
    /// <param name="trimMin">Minimum propensity score (clips extreme values). Default is 0.01.</param>
    /// <param name="trimMax">Maximum propensity score (clips extreme values). Default is 0.99.</param>
    /// <param name="stabilizedWeights">Whether to use stabilized weights for variance reduction. Default is true.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Parameters control weight behavior:
    ///
    /// - trimMin/trimMax: Propensity scores are clipped to [trimMin, trimMax] to prevent
    ///   extreme weights. If e(X) = 0.001, weight would be 1000 - way too much influence!
    ///   Trimming to 0.01 limits the weight to 100.
    ///
    /// - stabilizedWeights: Multiplies weights by the marginal treatment probability.
    ///   This reduces variance while still providing unbiased estimates.
    ///   Standard weight: 1/e(X)
    ///   Stabilized weight: P(T)/e(X)
    ///
    /// Usage:
    /// <code>
    /// var ipw = new InverseProbabilityWeighting&lt;double&gt;(trimMin: 0.05, trimMax: 0.95);
    /// var (ate, se) = ipw.EstimateATE(features, treatment, outcome);
    /// </code>
    /// </para>
    /// </remarks>
    public InverseProbabilityWeighting(
        double trimMin = 0.01,
        double trimMax = 0.99,
        bool stabilizedWeights = true)
    {
        // Validate trimming bounds to prevent infinite weights
        if (trimMin <= 0 || trimMin >= 1)
            throw new ArgumentOutOfRangeException(nameof(trimMin),
                "trimMin must be in (0, 1) to avoid infinite weights.");
        if (trimMax <= 0 || trimMax >= 1)
            throw new ArgumentOutOfRangeException(nameof(trimMax),
                "trimMax must be in (0, 1) to avoid infinite weights.");
        if (trimMin >= trimMax)
            throw new ArgumentException("trimMin must be less than trimMax.");

        _trimMin = trimMin;
        _trimMax = trimMax;
        _stabilizedWeights = stabilizedWeights;
    }

    /// <summary>
    /// Fits the propensity score model to the data.
    /// </summary>
    public void Fit(Matrix<T> x, Vector<int> treatment)
    {
        ValidateFitData(x, treatment);
        NumFeatures = x.Columns;

        // Cache data for predictions
        _cachedFeatures = x;
        _cachedTreatment = treatment;

        // Fit logistic regression for propensity scores
        _propensityCoefficients = FitLogisticRegression(x, treatment);

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
    /// and fits the propensity score model. IPW uses propensity scores to weight observations,
    /// giving more weight to "surprising" treatment assignments.
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
    /// Estimates treatment effects for individuals using IPW.
    /// </summary>
    /// <param name="features">The feature matrix for which to estimate effects.</param>
    /// <returns>A vector of estimated treatment effects.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> IPW doesn't directly model individual treatment effects - it's designed
    /// for average effects. This returns the estimated ATE for all individuals. For personalized
    /// treatment effect estimates, consider using CausalForest instead.
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

        // IPW doesn't model heterogeneous effects
        // Return the estimated ATE for all individuals
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
    /// For IPW, we use weighted averages of outcomes from treated individuals in the training data
    /// with similar propensity scores.
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

                    // Kernel weighting
                    double weight = Math.Exp(-distance * distance / 0.1);
                    sumOutcome += weight * NumOps.ToDouble(_cachedOutcome[j]);
                    sumWeight += weight;
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
    /// <b>For Beginners:</b> This estimates what the outcome would be if each individual did NOT receive treatment.
    /// For IPW, we use weighted averages of outcomes from control individuals in the training data
    /// with similar propensity scores.
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

                    // Kernel weighting
                    double weight = Math.Exp(-distance * distance / 0.1);
                    sumOutcome += weight * NumOps.ToDouble(_cachedOutcome[j]);
                    sumWeight += weight;
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

        var rawScores = PredictPropensityWithCoefficients(x, _propensityCoefficients);

        // Apply trimming
        for (int i = 0; i < rawScores.Length; i++)
        {
            double score = NumOps.ToDouble(rawScores[i]);
            score = Math.Max(_trimMin, Math.Min(_trimMax, score));
            rawScores[i] = NumOps.FromDouble(score);
        }

        return rawScores;
    }

    /// <summary>
    /// Computes IPW weights for each observation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The IPW weight formula is:
    /// - For treated: w = 1/e(X) (inverse of probability of treatment)
    /// - For control: w = 1/(1-e(X)) (inverse of probability of no treatment)
    ///
    /// With stabilized weights, we multiply by the marginal probability:
    /// - For treated: w = P(T=1)/e(X)
    /// - For control: w = P(T=0)/(1-e(X))
    /// </para>
    /// </remarks>
    public Vector<T> ComputeWeights(Matrix<T> x, Vector<int> treatment)
    {
        if (treatment.Length == 0)
        {
            throw new ArgumentException("Treatment vector cannot be empty.", nameof(treatment));
        }

        if (x.Rows != treatment.Length)
        {
            throw new ArgumentException(
                $"Number of samples in X ({x.Rows}) must match number of treatments ({treatment.Length}).");
        }

        var propensityScores = EstimatePropensityScores(x);
        var weights = new Vector<T>(x.Rows);

        // Calculate marginal treatment probability for stabilization
        double marginalP = 0;
        for (int i = 0; i < treatment.Length; i++)
        {
            marginalP += treatment[i];
        }
        marginalP /= treatment.Length;

        for (int i = 0; i < x.Rows; i++)
        {
            double e = NumOps.ToDouble(propensityScores[i]);

            double weight;
            if (treatment[i] == 1)
            {
                weight = 1.0 / e;
                if (_stabilizedWeights)
                {
                    weight *= marginalP;
                }
            }
            else
            {
                weight = 1.0 / (1.0 - e);
                if (_stabilizedWeights)
                {
                    weight *= (1.0 - marginalP);
                }
            }

            weights[i] = NumOps.FromDouble(weight);
        }

        return weights;
    }

    /// <summary>
    /// Estimates the Average Treatment Effect using IPW.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> IPW-ATE is computed as:
    /// ATE = (sum of weighted treated outcomes / sum of treated weights)
    ///     - (sum of weighted control outcomes / sum of control weights)
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

        T ate = ComputeIPWEstimate(x, treatment, outcome, EstimateType.ATE);
        T se = CalculateBootstrapStandardError(
            (xB, tB, oB) => ComputeIPWEstimate(xB, tB, oB, EstimateType.ATE),
            x, treatment, outcome);

        return (ate, se);
    }

    /// <summary>
    /// Estimates the Average Treatment Effect on the Treated using IPW.
    /// </summary>
    public override (T estimate, T standardError) EstimateATT(
        Matrix<T> x, Vector<int> treatment, Vector<T> outcome)
    {
        if (!IsFitted)
        {
            Fit(x, treatment);
        }

        ValidateCausalData(x, treatment, outcome);

        T att = ComputeIPWEstimate(x, treatment, outcome, EstimateType.ATT);
        T se = CalculateBootstrapStandardError(
            (xB, tB, oB) => ComputeIPWEstimate(xB, tB, oB, EstimateType.ATT),
            x, treatment, outcome);

        return (att, se);
    }

    private enum EstimateType { ATE, ATT }

    private T ComputeIPWEstimate(Matrix<T> x, Vector<int> treatment, Vector<T> outcome, EstimateType type)
    {
        var propensityScores = EstimatePropensityScores(x);
        int n = x.Rows;

        // Compute weighted outcomes
        double sumWeightedTreated = 0;
        double sumWeightedControl = 0;
        double sumWeightsTreated = 0;
        double sumWeightsControl = 0;

        // Compute marginal probability for stabilized weights
        double marginalP = treatment.Select(t => (double)t).Average();

        for (int i = 0; i < n; i++)
        {
            double e = NumOps.ToDouble(propensityScores[i]);
            double y = NumOps.ToDouble(outcome[i]);

            if (treatment[i] == 1)
            {
                double weight;
                if (type == EstimateType.ATT)
                {
                    // For ATT, treated units are not weighted
                    weight = 1.0;
                }
                else // ATE
                {
                    weight = 1.0 / e;
                    if (_stabilizedWeights)
                    {
                        weight *= marginalP;
                    }
                }
                sumWeightedTreated += weight * y;
                sumWeightsTreated += weight;
            }
            else
            {
                double weight;
                if (type == EstimateType.ATE)
                {
                    weight = 1.0 / (1.0 - e);
                    if (_stabilizedWeights)
                    {
                        weight *= (1.0 - marginalP);
                    }
                }
                else // ATT
                {
                    // For ATT, reweight controls by odds ratio
                    weight = e / (1.0 - e);
                }
                sumWeightedControl += weight * y;
                sumWeightsControl += weight;
            }
        }

        double meanTreated = sumWeightsTreated > 0 ? sumWeightedTreated / sumWeightsTreated : 0;
        double meanControl = sumWeightsControl > 0 ? sumWeightedControl / sumWeightsControl : 0;

        return NumOps.FromDouble(meanTreated - meanControl);
    }

    /// <summary>
    /// Estimates individual treatment effects.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> IPW doesn't directly estimate individual effects (it's designed
    /// for average effects). This returns the weighted contribution of each observation
    /// to the overall ATE, normalized to represent pseudo-individual effects.
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
        var cate = new Vector<T>(x.Rows);

        // IPW doesn't naturally give individual effects
        // Compute pseudo-effects based on IPW contributions
        T ate = EstimateATE(x, treatment, outcome).estimate;

        for (int i = 0; i < x.Rows; i++)
        {
            // Use ATE as best guess for individuals
            // (for true individual effects, use CausalForest)
            cate[i] = ate;
        }

        return cate;
    }

    /// <summary>
    /// Predicts treatment effects for new individuals.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> IPW doesn't model heterogeneous effects - it estimates
    /// a constant average treatment effect (ATE). This returns the ATE for all individuals.
    /// For personalized treatment effects, consider CausalForest.
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

        // IPW doesn't model heterogeneous effects - return constant ATE
        var (ate, _) = EstimateATE(_cachedFeatures, _cachedTreatment, _cachedOutcome);

        var effects = new Vector<T>(x.Rows);
        for (int i = 0; i < x.Rows; i++)
        {
            effects[i] = ate;
        }

        return effects;
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
            NumFeatures = parameters.Length - 1;
        }
    }

    /// <summary>
    /// Creates a new instance with specified parameters.
    /// </summary>
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newModel = new InverseProbabilityWeighting<T>(_trimMin, _trimMax, _stabilizedWeights);
        newModel.SetParameters(parameters);
        return newModel;
    }

    /// <summary>
    /// Creates a new instance of this type.
    /// </summary>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new InverseProbabilityWeighting<T>(_trimMin, _trimMax, _stabilizedWeights);
    }

    /// <summary>
    /// Gets additional model data for serialization.
    /// </summary>
    protected override Dictionary<string, object> GetAdditionalModelData()
    {
        var data = new Dictionary<string, object>
        {
            { "TrimMin", _trimMin },
            { "TrimMax", _trimMax },
            { "StabilizedWeights", _stabilizedWeights }
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
        // Restore trim/stabilization settings
        var trimMinToken = modelDataObj["TrimMin"];
        if (trimMinToken is not null)
            _trimMin = trimMinToken.ToObject<double>();

        var trimMaxToken = modelDataObj["TrimMax"];
        if (trimMaxToken is not null)
            _trimMax = trimMaxToken.ToObject<double>();

        var stabilizedToken = modelDataObj["StabilizedWeights"];
        if (stabilizedToken is not null)
            _stabilizedWeights = stabilizedToken.ToObject<bool>();

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
                // Restore NumFeatures (coefficients include bias, so features = length - 1)
                NumFeatures = coeffs.Length - 1;
            }
        }
    }

    #endregion

    /// <summary>
    /// Gets the effective sample size after weighting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When using weights, the "effective" sample size is smaller
    /// than the actual sample size because some observations dominate. ESS tells you
    /// how many unweighted observations your weighted sample is worth.
    ///
    /// Formula: ESS = (sum of weights)² / (sum of squared weights)
    ///
    /// Low ESS indicates that a few observations have extreme weights and dominate the estimate.
    /// </para>
    /// </remarks>
    public (T treatedESS, T controlESS) GetEffectiveSampleSize(Matrix<T> x, Vector<int> treatment)
    {
        var weights = ComputeWeights(x, treatment);

        double sumWeightsTreated = 0, sumSqWeightsTreated = 0;
        double sumWeightsControl = 0, sumSqWeightsControl = 0;

        for (int i = 0; i < x.Rows; i++)
        {
            double w = NumOps.ToDouble(weights[i]);

            if (treatment[i] == 1)
            {
                sumWeightsTreated += w;
                sumSqWeightsTreated += w * w;
            }
            else
            {
                sumWeightsControl += w;
                sumSqWeightsControl += w * w;
            }
        }

        double essTreated = sumSqWeightsTreated > 0
            ? (sumWeightsTreated * sumWeightsTreated) / sumSqWeightsTreated
            : 0;
        double essControl = sumSqWeightsControl > 0
            ? (sumWeightsControl * sumWeightsControl) / sumSqWeightsControl
            : 0;

        return (NumOps.FromDouble(essTreated), NumOps.FromDouble(essControl));
    }
}

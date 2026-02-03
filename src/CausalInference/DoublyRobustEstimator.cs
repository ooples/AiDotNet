using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.CausalInference;

/// <summary>
/// Implements the Doubly Robust (DR) estimator for causal effect estimation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// The Doubly Robust estimator combines outcome regression with propensity score weighting,
/// providing consistent estimates if EITHER the outcome model OR the propensity model is correct.
/// </para>
/// <para>
/// <b>For Beginners:</b> The DR estimator is like having two insurance policies:
///
/// 1. Outcome Model: Predicts outcomes based on features and treatment
/// 2. Propensity Model: Predicts who gets treated
///
/// If either model is correct, you get unbiased treatment effect estimates.
/// This "double protection" makes DR very popular in practice.
///
/// The formula combines:
/// - Predicted outcomes (if we trust the outcome model)
/// - IPW-adjusted residuals (correction if outcome model is wrong)
///
/// DR estimator:
/// τ̂ = (1/n) Σ [μ₁(Xᵢ) - μ₀(Xᵢ)]  +  (1/n) Σ [Tᵢ(Yᵢ - μ₁(Xᵢ))/e(Xᵢ) - (1-Tᵢ)(Yᵢ - μ₀(Xᵢ))/(1-e(Xᵢ))]
///
/// Where:
/// - μ₁(X), μ₀(X) = predicted outcomes under treatment/control
/// - e(X) = propensity score
/// - T = treatment indicator
/// - Y = observed outcome
///
/// The first term uses the outcome model predictions.
/// The second term "corrects" using IPW when the outcome model is wrong.
///
/// Advantages:
/// - Doubly robust: consistent if either model is correct
/// - More efficient than IPW alone when both models are good
/// - Semiparametric efficiency (achieves best possible variance)
///
/// References:
/// - Robins, Rotnitzky &amp; Zhao (1994). "Estimation of Regression Coefficients"
/// - Bang &amp; Robins (2005). "Doubly Robust Estimation"
/// </para>
/// </remarks>
public class DoublyRobustEstimator<T> : CausalModelBase<T>
{
    /// <summary>
    /// Stores the logistic regression coefficients for propensity score estimation.
    /// </summary>
    private Vector<T>? _propensityCoefficients;

    /// <summary>
    /// Stores the outcome regression coefficients for treated group.
    /// </summary>
    private Vector<T>? _outcomeCoefficients1;

    /// <summary>
    /// Stores the outcome regression coefficients for control group.
    /// </summary>
    private Vector<T>? _outcomeCoefficients0;

    /// <summary>
    /// Minimum propensity score to avoid extreme weights.
    /// </summary>
    private readonly double _trimMin;

    /// <summary>
    /// Maximum propensity score to avoid extreme weights.
    /// </summary>
    private readonly double _trimMax;

    /// <summary>
    /// Whether to use cross-fitting for debiased estimation.
    /// </summary>
    private readonly bool _useCrossFitting;

    /// <summary>
    /// Number of folds for cross-fitting.
    /// </summary>
    private readonly int _numFolds;

    /// <summary>
    /// Gets the model type.
    /// </summary>
    public override ModelType GetModelType() => ModelType.DoublyRobustEstimator;

    /// <summary>
    /// Initializes a new instance of the DoublyRobustEstimator class.
    /// </summary>
    /// <param name="trimMin">Minimum propensity score. Default is 0.01.</param>
    /// <param name="trimMax">Maximum propensity score. Default is 0.99.</param>
    /// <param name="useCrossFitting">Whether to use cross-fitting for debiased estimation. Default is false.</param>
    /// <param name="numFolds">Number of folds for cross-fitting. Default is 5.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Parameters:
    ///
    /// - trimMin/trimMax: Clip propensity scores to avoid extreme weights
    ///
    /// - useCrossFitting: When true, uses "cross-fitting" which:
    ///   1. Splits data into K folds
    ///   2. For each fold, fits models on other folds and predicts on this fold
    ///   3. Combines predictions to avoid overfitting bias
    ///   This is recommended for large datasets and is required for valid inference
    ///   with flexible machine learning models.
    ///
    /// - numFolds: Number of cross-fitting folds (typically 5-10)
    ///
    /// Usage:
    /// <code>
    /// var dr = new DoublyRobustEstimator&lt;double&gt;(useCrossFitting: true);
    /// var (ate, se) = dr.EstimateATE(features, treatment, outcome);
    /// </code>
    /// </para>
    /// </remarks>
    public DoublyRobustEstimator(
        double trimMin = 0.01,
        double trimMax = 0.99,
        bool useCrossFitting = false,
        int numFolds = 5)
    {
        _trimMin = trimMin;
        _trimMax = trimMax;
        _useCrossFitting = useCrossFitting;
        _numFolds = numFolds;
    }

    /// <summary>
    /// Fits both propensity score and outcome models to the data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This fits two types of models:
    /// 1. Propensity score model: P(treatment | features)
    /// 2. Outcome models: E[outcome | features] separately for treated and control
    /// </para>
    /// </remarks>
    public void Fit(Matrix<T> x, Vector<int> treatment, Vector<T> outcome)
    {
        ValidateCausalData(x, treatment, outcome);
        NumFeatures = x.Columns;

        if (_useCrossFitting)
        {
            // With cross-fitting, we fit models on the full data for GetParameters
            // but use cross-fit predictions for estimation
            _propensityCoefficients = FitLogisticRegression(x, treatment);
            (_outcomeCoefficients1, _outcomeCoefficients0) = FitOutcomeModels(x, treatment, outcome);
        }
        else
        {
            // Fit propensity score model
            _propensityCoefficients = FitLogisticRegression(x, treatment);

            // Fit outcome regression models separately for treated and control
            (_outcomeCoefficients1, _outcomeCoefficients0) = FitOutcomeModels(x, treatment, outcome);
        }

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
    /// and fits both the propensity score model and outcome regression models. The doubly robust
    /// estimator combines these to provide consistent estimates even if one model is misspecified.
    /// </para>
    /// </remarks>
    public override void Fit(Matrix<T> features, Vector<T> treatment, Vector<T> outcome)
    {
        // Convert treatment vector to int
        var treatmentInt = new Vector<int>(treatment.Length);
        for (int i = 0; i < treatment.Length; i++)
        {
            treatmentInt[i] = (int)Math.Round(NumOps.ToDouble(treatment[i]));
        }

        // Call the original fit method
        Fit(features, treatmentInt, outcome);
    }

    /// <summary>
    /// Estimates treatment effects for individuals using the doubly robust estimator.
    /// </summary>
    /// <param name="features">The feature matrix for which to estimate effects.</param>
    /// <returns>A vector of estimated treatment effects.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The doubly robust estimator provides treatment effect estimates
    /// by combining outcome model predictions (what we expect to happen) with propensity score
    /// weighting (who is likely to be treated). This gives you robust estimates even if one
    /// of the models is somewhat wrong.
    /// </para>
    /// </remarks>
    public override Vector<T> EstimateTreatmentEffect(Matrix<T> features)
    {
        return PredictTreatmentEffect(features);
    }

    /// <summary>
    /// Predicts outcomes under treatment for the given features.
    /// </summary>
    /// <param name="features">The feature matrix.</param>
    /// <returns>Predicted outcomes if treated.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This predicts what each individual's outcome would be if they
    /// received treatment. The doubly robust estimator has separate outcome models for
    /// treated and control groups, so this uses the treated outcome model μ₁(X).
    /// </para>
    /// </remarks>
    public override Vector<T> PredictTreated(Matrix<T> features)
    {
        EnsureFitted();

        if (_outcomeCoefficients1 is null)
        {
            throw new InvalidOperationException("Outcome models not fitted.");
        }

        return PredictOutcome(features, _outcomeCoefficients1);
    }

    /// <summary>
    /// Predicts outcomes under control for the given features.
    /// </summary>
    /// <param name="features">The feature matrix.</param>
    /// <returns>Predicted outcomes if not treated.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This predicts what each individual's outcome would be if they
    /// did NOT receive treatment. The doubly robust estimator has separate outcome models for
    /// treated and control groups, so this uses the control outcome model μ₀(X).
    /// </para>
    /// </remarks>
    public override Vector<T> PredictControl(Matrix<T> features)
    {
        EnsureFitted();

        if (_outcomeCoefficients0 is null)
        {
            throw new InvalidOperationException("Outcome models not fitted.");
        }

        return PredictOutcome(features, _outcomeCoefficients0);
    }

    /// <summary>
    /// Fits separate linear regression models for treated and control groups.
    /// </summary>
    private (Vector<T> coef1, Vector<T> coef0) FitOutcomeModels(
        Matrix<T> x, Vector<int> treatment, Vector<T> outcome)
    {
        int p = x.Columns;

        // Separate data by treatment group
        var treatedX = new List<double[]>();
        var treatedY = new List<double>();
        var controlX = new List<double[]>();
        var controlY = new List<double>();

        for (int i = 0; i < x.Rows; i++)
        {
            var features = new double[p + 1];
            features[0] = 1.0; // intercept
            for (int j = 0; j < p; j++)
            {
                features[j + 1] = NumOps.ToDouble(x[i, j]);
            }

            if (treatment[i] == 1)
            {
                treatedX.Add(features);
                treatedY.Add(NumOps.ToDouble(outcome[i]));
            }
            else
            {
                controlX.Add(features);
                controlY.Add(NumOps.ToDouble(outcome[i]));
            }
        }

        var coef1 = FitLinearRegressionOLS(treatedX, treatedY, p + 1);
        var coef0 = FitLinearRegressionOLS(controlX, controlY, p + 1);

        return (coef1, coef0);
    }

    /// <summary>
    /// Simple OLS linear regression fitting.
    /// </summary>
    private Vector<T> FitLinearRegressionOLS(List<double[]> xData, List<double> yData, int numCoefs)
    {
        if (xData.Count == 0)
        {
            return new Vector<T>(numCoefs);
        }

        int n = xData.Count;
        int p = numCoefs;

        // X'X matrix
        var xtx = new double[p, p];
        var xty = new double[p];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                for (int k = 0; k < p; k++)
                {
                    xtx[j, k] += xData[i][j] * xData[i][k];
                }
                xty[j] += xData[i][j] * yData[i];
            }
        }

        // Add small ridge for numerical stability
        for (int j = 0; j < p; j++)
        {
            xtx[j, j] += 1e-6;
        }

        // Solve using simple Gaussian elimination
        var coefs = SolveLinearSystem(xtx, xty, p);

        var result = new Vector<T>(p);
        for (int j = 0; j < p; j++)
        {
            result[j] = NumOps.FromDouble(coefs[j]);
        }

        return result;
    }

    /// <summary>
    /// Solves a linear system using Gaussian elimination with partial pivoting.
    /// </summary>
    private double[] SolveLinearSystem(double[,] a, double[] b, int n)
    {
        var augmented = new double[n, n + 1];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                augmented[i, j] = a[i, j];
            }
            augmented[i, n] = b[i];
        }

        // Forward elimination with partial pivoting
        for (int k = 0; k < n; k++)
        {
            // Find pivot
            int maxRow = k;
            for (int i = k + 1; i < n; i++)
            {
                if (Math.Abs(augmented[i, k]) > Math.Abs(augmented[maxRow, k]))
                {
                    maxRow = i;
                }
            }

            // Swap rows
            for (int j = k; j <= n; j++)
            {
                (augmented[k, j], augmented[maxRow, j]) = (augmented[maxRow, j], augmented[k, j]);
            }

            // Eliminate
            for (int i = k + 1; i < n; i++)
            {
                double factor = augmented[k, k] != 0 ? augmented[i, k] / augmented[k, k] : 0;
                for (int j = k; j <= n; j++)
                {
                    augmented[i, j] -= factor * augmented[k, j];
                }
            }
        }

        // Back substitution
        var x = new double[n];
        for (int i = n - 1; i >= 0; i--)
        {
            x[i] = augmented[i, n];
            for (int j = i + 1; j < n; j++)
            {
                x[i] -= augmented[i, j] * x[j];
            }
            x[i] = augmented[i, i] != 0 ? x[i] / augmented[i, i] : 0;
        }

        return x;
    }

    /// <summary>
    /// Predicts outcomes using the outcome regression model.
    /// </summary>
    private Vector<T> PredictOutcome(Matrix<T> x, Vector<T> coefficients)
    {
        var predictions = new Vector<T>(x.Rows);
        int p = x.Columns;

        for (int i = 0; i < x.Rows; i++)
        {
            double pred = NumOps.ToDouble(coefficients[0]); // intercept
            for (int j = 0; j < p; j++)
            {
                pred += NumOps.ToDouble(coefficients[j + 1]) * NumOps.ToDouble(x[i, j]);
            }
            predictions[i] = NumOps.FromDouble(pred);
        }

        return predictions;
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
    /// Estimates the Average Treatment Effect using the Doubly Robust estimator.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The DR-ATE combines outcome model predictions with IPW corrections:
    ///
    /// τ̂_DR = (1/n) Σ [μ̂₁(X) - μ̂₀(X)] + (1/n) Σ [T(Y - μ̂₁(X))/e(X) - (1-T)(Y - μ̂₀(X))/(1-e(X))]
    ///
    /// The first term is the predicted treatment effect from the outcome model.
    /// The second term corrects for errors in the outcome model using IPW.
    /// </para>
    /// </remarks>
    public override (T estimate, T standardError) EstimateATE(
        Matrix<T> x, Vector<int> treatment, Vector<T> outcome)
    {
        if (!IsFitted)
        {
            Fit(x, treatment, outcome);
        }

        ValidateCausalData(x, treatment, outcome);

        T ate;
        if (_useCrossFitting)
        {
            ate = ComputeDRATE_CrossFit(x, treatment, outcome);
        }
        else
        {
            ate = ComputeDRATE(x, treatment, outcome);
        }

        T se = CalculateBootstrapStandardError(
            (xB, tB, oB) => ComputeDRATE(xB, tB, oB),
            x, treatment, outcome);

        return (ate, se);
    }

    private T ComputeDRATE(Matrix<T> x, Vector<int> treatment, Vector<T> outcome)
    {
        int n = x.Rows;
        var propensityScores = EstimatePropensityScores(x);
        var mu1 = PredictOutcome(x, _outcomeCoefficients1!);
        var mu0 = PredictOutcome(x, _outcomeCoefficients0!);

        double sumDR = 0;

        for (int i = 0; i < n; i++)
        {
            double e = NumOps.ToDouble(propensityScores[i]);
            double y = NumOps.ToDouble(outcome[i]);
            double m1 = NumOps.ToDouble(mu1[i]);
            double m0 = NumOps.ToDouble(mu0[i]);
            int t = treatment[i];

            // DR estimator: outcome model + IPW correction
            double outcomeModel = m1 - m0;
            double ipwCorrection = t * (y - m1) / e - (1 - t) * (y - m0) / (1 - e);

            sumDR += outcomeModel + ipwCorrection;
        }

        return NumOps.FromDouble(sumDR / n);
    }

    private T ComputeDRATE_CrossFit(Matrix<T> x, Vector<int> treatment, Vector<T> outcome)
    {
        int n = x.Rows;
        var random = RandomHelper.CreateSecureRandom();

        // Create fold assignments
        var indices = Enumerable.Range(0, n).OrderBy(i => random.Next()).ToArray();
        var foldAssignment = new int[n];
        for (int i = 0; i < n; i++)
        {
            foldAssignment[indices[i]] = i % _numFolds;
        }

        // Store cross-fit predictions
        var mu1CrossFit = new double[n];
        var mu0CrossFit = new double[n];
        var propCrossFit = new double[n];

        for (int fold = 0; fold < _numFolds; fold++)
        {
            // Create train/test split
            var trainIndices = new List<int>();
            var testIndices = new List<int>();

            for (int i = 0; i < n; i++)
            {
                if (foldAssignment[i] == fold)
                    testIndices.Add(i);
                else
                    trainIndices.Add(i);
            }

            // Create training data
            var xTrain = new Matrix<T>(trainIndices.Count, x.Columns);
            var tTrain = new Vector<int>(trainIndices.Count);
            var yTrain = new Vector<T>(trainIndices.Count);

            for (int i = 0; i < trainIndices.Count; i++)
            {
                int idx = trainIndices[i];
                for (int j = 0; j < x.Columns; j++)
                {
                    xTrain[i, j] = x[idx, j];
                }
                tTrain[i] = treatment[idx];
                yTrain[i] = outcome[idx];
            }

            // Fit models on training data
            var propCoefs = FitLogisticRegression(xTrain, tTrain);
            var (outCoefs1, outCoefs0) = FitOutcomeModels(xTrain, tTrain, yTrain);

            // Predict on test data
            foreach (int idx in testIndices)
            {
                // Create single-row matrix for prediction
                var xRow = new Matrix<T>(1, x.Columns);
                for (int j = 0; j < x.Columns; j++)
                {
                    xRow[0, j] = x[idx, j];
                }

                var prop = PredictPropensityWithCoefficients(xRow, propCoefs);
                propCrossFit[idx] = Math.Max(_trimMin, Math.Min(_trimMax, NumOps.ToDouble(prop[0])));

                var m1 = PredictOutcome(xRow, outCoefs1);
                var m0 = PredictOutcome(xRow, outCoefs0);
                mu1CrossFit[idx] = NumOps.ToDouble(m1[0]);
                mu0CrossFit[idx] = NumOps.ToDouble(m0[0]);
            }
        }

        // Compute DR estimate using cross-fit predictions
        double sumDR = 0;
        for (int i = 0; i < n; i++)
        {
            double e = propCrossFit[i];
            double y = NumOps.ToDouble(outcome[i]);
            double m1 = mu1CrossFit[i];
            double m0 = mu0CrossFit[i];
            int t = treatment[i];

            double outcomeModel = m1 - m0;
            double ipwCorrection = t * (y - m1) / e - (1 - t) * (y - m0) / (1 - e);

            sumDR += outcomeModel + ipwCorrection;
        }

        return NumOps.FromDouble(sumDR / n);
    }

    /// <summary>
    /// Estimates the Average Treatment Effect on the Treated.
    /// </summary>
    public override (T estimate, T standardError) EstimateATT(
        Matrix<T> x, Vector<int> treatment, Vector<T> outcome)
    {
        if (!IsFitted)
        {
            Fit(x, treatment, outcome);
        }

        ValidateCausalData(x, treatment, outcome);

        T att = ComputeDRATT(x, treatment, outcome);
        T se = CalculateBootstrapStandardError(
            (xB, tB, oB) => ComputeDRATT(xB, tB, oB),
            x, treatment, outcome);

        return (att, se);
    }

    private T ComputeDRATT(Matrix<T> x, Vector<int> treatment, Vector<T> outcome)
    {
        int n = x.Rows;
        var propensityScores = EstimatePropensityScores(x);
        var mu0 = PredictOutcome(x, _outcomeCoefficients0!);

        double sumTreated = 0;
        double sumIPW = 0;
        int nTreated = 0;

        for (int i = 0; i < n; i++)
        {
            double e = NumOps.ToDouble(propensityScores[i]);
            double y = NumOps.ToDouble(outcome[i]);
            double m0 = NumOps.ToDouble(mu0[i]);
            int t = treatment[i];

            if (t == 1)
            {
                sumTreated += y - m0;
                nTreated++;
            }
            else
            {
                // IPW correction for control group
                double weight = e / (1 - e);
                sumIPW += weight * (y - m0);
            }
        }

        double att = nTreated > 0 ? (sumTreated - sumIPW) / nTreated : 0;
        return NumOps.FromDouble(att);
    }

    /// <summary>
    /// Estimates individual treatment effects.
    /// </summary>
    public override Vector<T> EstimateCATEPerIndividual(
        Matrix<T> x, Vector<int> treatment, Vector<T> outcome)
    {
        if (!IsFitted)
        {
            Fit(x, treatment, outcome);
        }

        ValidateCausalData(x, treatment, outcome);

        var mu1 = PredictOutcome(x, _outcomeCoefficients1!);
        var mu0 = PredictOutcome(x, _outcomeCoefficients0!);
        var cate = new Vector<T>(x.Rows);

        for (int i = 0; i < x.Rows; i++)
        {
            // CATE from outcome model
            cate[i] = NumOps.Subtract(mu1[i], mu0[i]);
        }

        return cate;
    }

    /// <summary>
    /// Predicts treatment effects for new individuals.
    /// </summary>
    public override Vector<T> PredictTreatmentEffect(Matrix<T> x)
    {
        EnsureFitted();

        if (_outcomeCoefficients1 is null || _outcomeCoefficients0 is null)
        {
            throw new InvalidOperationException("Outcome models not fitted.");
        }

        var mu1 = PredictOutcome(x, _outcomeCoefficients1);
        var mu0 = PredictOutcome(x, _outcomeCoefficients0);
        var effects = new Vector<T>(x.Rows);

        for (int i = 0; i < x.Rows; i++)
        {
            effects[i] = NumOps.Subtract(mu1[i], mu0[i]);
        }

        return effects;
    }

    /// <summary>
    /// Standard prediction - returns predicted treatment effects.
    /// </summary>
    public override Vector<T> Predict(Matrix<T> input)
    {
        return PredictTreatmentEffect(input);
    }

    #region IFullModel Implementation

    /// <summary>
    /// Gets the model parameters (propensity + outcome coefficients).
    /// </summary>
    public override Vector<T> GetParameters()
    {
        if (_propensityCoefficients is null ||
            _outcomeCoefficients1 is null ||
            _outcomeCoefficients0 is null)
        {
            return new Vector<T>(0);
        }

        int totalLength = _propensityCoefficients.Length +
                         _outcomeCoefficients1.Length +
                         _outcomeCoefficients0.Length;

        var parameters = new Vector<T>(totalLength);
        int idx = 0;

        for (int i = 0; i < _propensityCoefficients.Length; i++)
            parameters[idx++] = _propensityCoefficients[i];
        for (int i = 0; i < _outcomeCoefficients1.Length; i++)
            parameters[idx++] = _outcomeCoefficients1[i];
        for (int i = 0; i < _outcomeCoefficients0.Length; i++)
            parameters[idx++] = _outcomeCoefficients0[i];

        return parameters;
    }

    /// <summary>
    /// Sets the model parameters.
    /// </summary>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length == 0) return;

        // Assume equal split for simplicity
        int propLength = (parameters.Length + 2) / 3;
        int outLength = propLength;

        _propensityCoefficients = new Vector<T>(propLength);
        _outcomeCoefficients1 = new Vector<T>(outLength);
        _outcomeCoefficients0 = new Vector<T>(outLength);

        int idx = 0;
        for (int i = 0; i < propLength && idx < parameters.Length; i++)
            _propensityCoefficients[i] = parameters[idx++];
        for (int i = 0; i < outLength && idx < parameters.Length; i++)
            _outcomeCoefficients1[i] = parameters[idx++];
        for (int i = 0; i < outLength && idx < parameters.Length; i++)
            _outcomeCoefficients0[i] = parameters[idx++];

        NumFeatures = propLength - 1;
    }

    /// <summary>
    /// Creates a new instance with specified parameters.
    /// </summary>
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newModel = new DoublyRobustEstimator<T>(_trimMin, _trimMax, _useCrossFitting, _numFolds);
        newModel.SetParameters(parameters);
        return newModel;
    }

    /// <summary>
    /// Creates a new instance of this type.
    /// </summary>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new DoublyRobustEstimator<T>(_trimMin, _trimMax, _useCrossFitting, _numFolds);
    }

    #endregion
}

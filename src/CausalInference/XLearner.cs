using System.Text;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using Newtonsoft.Json;

namespace AiDotNet.CausalInference;

/// <summary>
/// Implements the X-Learner (Cross-learner) for treatment effect estimation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> X-Learner is a sophisticated meta-learner that adapts to the data
/// by using cross-fitting. It's especially effective when treatment and control groups have
/// different sizes, as it leverages information from both groups more efficiently.</para>
///
/// <para><b>How it works (5 stages):</b>
/// <list type="number">
/// <item>Train μ₀(X) and μ₁(X) using T-Learner approach</item>
/// <item>Impute treatment effects: D₁ᵢ = Y₁ᵢ - μ₀(X₁ᵢ) for treated, D₀ᵢ = μ₁(X₀ᵢ) - Y₀ᵢ for control</item>
/// <item>Train τ₁(X) on D₁ (treated imputed effects) and τ₀(X) on D₀ (control imputed effects)</item>
/// <item>Estimate propensity score e(X) = P(T=1|X)</item>
/// <item>Combine: τ(X) = e(X)·τ₀(X) + (1-e(X))·τ₁(X)</item>
/// </list>
/// </para>
///
/// <para><b>Key Insight:</b> The weighted combination uses propensity scores to give more weight
/// to the model trained on the larger group, making X-Learner robust to imbalanced data.</para>
///
/// <para><b>Pros and Cons:</b>
/// <list type="bullet">
/// <item><b>Pro:</b> Excellent for imbalanced treatment groups</item>
/// <item><b>Pro:</b> Can outperform T-Learner when one group is much smaller</item>
/// <item><b>Pro:</b> Adapts to the data structure through propensity weighting</item>
/// <item><b>Con:</b> More complex, requires fitting 5 models</item>
/// <item><b>Con:</b> Propensity estimation can be sensitive</item>
/// </list>
/// </para>
///
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>When treatment/control groups are imbalanced</item>
/// <item>When you want state-of-the-art CATE estimation</item>
/// <item>When you have sufficient data for multiple model fitting</item>
/// </list>
/// </para>
///
/// <para><b>Reference:</b> Künzel et al., "Metalearners for Estimating Heterogeneous Treatment Effects" (2019)</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class XLearner<T> : CausalModelBase<T>
{
    /// <summary>
    /// Weights for the control outcome model μ₀.
    /// </summary>
    private Vector<T>? _weightsControl;

    /// <summary>
    /// Weights for the treatment outcome model μ₁.
    /// </summary>
    private Vector<T>? _weightsTreated;

    /// <summary>
    /// Weights for the treatment effect model τ₀ (trained on control imputed effects).
    /// </summary>
    private Vector<T>? _weightsTau0;

    /// <summary>
    /// Weights for the treatment effect model τ₁ (trained on treated imputed effects).
    /// </summary>
    private Vector<T>? _weightsTau1;

    /// <summary>
    /// Weights for the propensity score model.
    /// </summary>
    private Vector<T>? _weightsPropensity;

    /// <summary>
    /// Bias terms for each model.
    /// </summary>
    private T _biasControl, _biasTreated, _biasTau0, _biasTau1, _biasPropensity;

    /// <summary>
    /// Gets the maximum iterations for training.
    /// </summary>
    public int MaxIterations { get; }

    /// <summary>
    /// Gets the learning rate for training.
    /// </summary>
    public double LearningRate { get; }

    /// <summary>
    /// Gets the L2 regularization strength.
    /// </summary>
    public double Lambda { get; }

    /// <summary>
    /// Creates a new X-Learner.
    /// </summary>
    /// <param name="maxIterations">Maximum training iterations (default: 100).</param>
    /// <param name="learningRate">Learning rate (default: 0.1).</param>
    /// <param name="lambda">L2 regularization (default: 0.01).</param>
    public XLearner(int maxIterations = 100, double learningRate = 0.1, double lambda = 0.01) : base()
    {
        MaxIterations = maxIterations;
        LearningRate = learningRate;
        Lambda = lambda;
        _biasControl = NumOps.Zero;
        _biasTreated = NumOps.Zero;
        _biasTau0 = NumOps.Zero;
        _biasTau1 = NumOps.Zero;
        _biasPropensity = NumOps.Zero;
    }

    /// <summary>
    /// Fits the X-Learner model using the 5-stage algorithm.
    /// </summary>
    public override void Fit(Matrix<T> features, Vector<T> treatment, Vector<T> outcome)
    {
        int n = features.Rows;
        int p = features.Columns;
        NumFeatures = p;

        // Convert treatment to int
        var treatmentInt = new Vector<int>(n);
        for (int i = 0; i < n; i++)
            treatmentInt[i] = NumOps.ToDouble(treatment[i]) > 0.5 ? 1 : 0;

        ValidateCausalData(features, treatmentInt, outcome);

        // Split indices by treatment status
        var treatedIndices = new List<int>();
        var controlIndices = new List<int>();

        for (int i = 0; i < n; i++)
        {
            if (treatmentInt[i] == 1)
                treatedIndices.Add(i);
            else
                controlIndices.Add(i);
        }

        // Stage 1: Train outcome models (T-Learner style)
        (_weightsTreated, _biasTreated) = TrainLinearModel(features, outcome, treatedIndices.ToArray());
        (_weightsControl, _biasControl) = TrainLinearModel(features, outcome, controlIndices.ToArray());

        // Stage 2: Compute imputed treatment effects
        // For treated: D₁ = Y₁ - μ₀(X₁) (actual outcome minus predicted control outcome)
        var imputedTreated = new Vector<T>(treatedIndices.Count);
        for (int idx = 0; idx < treatedIndices.Count; idx++)
        {
            int i = treatedIndices[idx];
            double actualOutcome = NumOps.ToDouble(outcome[i]);
            double predictedControl = PredictSingle(features, i, _weightsControl!, _biasControl);
            imputedTreated[idx] = NumOps.FromDouble(actualOutcome - predictedControl);
        }

        // For control: D₀ = μ₁(X₀) - Y₀ (predicted treated outcome minus actual outcome)
        var imputedControl = new Vector<T>(controlIndices.Count);
        for (int idx = 0; idx < controlIndices.Count; idx++)
        {
            int i = controlIndices[idx];
            double predictedTreated = PredictSingle(features, i, _weightsTreated!, _biasTreated);
            double actualOutcome = NumOps.ToDouble(outcome[i]);
            imputedControl[idx] = NumOps.FromDouble(predictedTreated - actualOutcome);
        }

        // Stage 3: Train treatment effect models
        (_weightsTau1, _biasTau1) = TrainLinearModelWithOutcome(features, imputedTreated, treatedIndices.ToArray());
        (_weightsTau0, _biasTau0) = TrainLinearModelWithOutcome(features, imputedControl, controlIndices.ToArray());

        // Stage 4: Train propensity score model (logistic regression)
        (_weightsPropensity, _biasPropensity) = TrainPropensityModel(features, treatmentInt);

        IsFitted = true;
    }

    /// <summary>
    /// Trains a linear regression model on specified indices.
    /// </summary>
    private (Vector<T> weights, T bias) TrainLinearModel(Matrix<T> features, Vector<T> outcome, int[] indices)
    {
        int n = indices.Length;
        int p = features.Columns;

        var weights = new Vector<T>(p);
        var bias = NumOps.Zero;

        for (int iter = 0; iter < MaxIterations; iter++)
        {
            var gradWeights = new double[p];
            double gradBias = 0;

            for (int idx = 0; idx < n; idx++)
            {
                int i = indices[idx];

                double pred = NumOps.ToDouble(bias);
                for (int j = 0; j < p; j++)
                    pred += NumOps.ToDouble(weights[j]) * NumOps.ToDouble(features[i, j]);

                double error = pred - NumOps.ToDouble(outcome[i]);

                gradBias += error;
                for (int j = 0; j < p; j++)
                    gradWeights[j] += error * NumOps.ToDouble(features[i, j]);
            }

            bias = NumOps.FromDouble(NumOps.ToDouble(bias) - LearningRate * gradBias / n);
            for (int j = 0; j < p; j++)
            {
                double grad = gradWeights[j] / n + Lambda * NumOps.ToDouble(weights[j]);
                weights[j] = NumOps.FromDouble(NumOps.ToDouble(weights[j]) - LearningRate * grad);
            }
        }

        return (weights, bias);
    }

    /// <summary>
    /// Trains a linear model with a separate outcome vector (for imputed effects).
    /// </summary>
    private (Vector<T> weights, T bias) TrainLinearModelWithOutcome(Matrix<T> features, Vector<T> targetOutcome, int[] indices)
    {
        int n = indices.Length;
        int p = features.Columns;

        var weights = new Vector<T>(p);
        var bias = NumOps.Zero;

        for (int iter = 0; iter < MaxIterations; iter++)
        {
            var gradWeights = new double[p];
            double gradBias = 0;

            for (int idx = 0; idx < n; idx++)
            {
                int i = indices[idx];

                double pred = NumOps.ToDouble(bias);
                for (int j = 0; j < p; j++)
                    pred += NumOps.ToDouble(weights[j]) * NumOps.ToDouble(features[i, j]);

                double error = pred - NumOps.ToDouble(targetOutcome[idx]);

                gradBias += error;
                for (int j = 0; j < p; j++)
                    gradWeights[j] += error * NumOps.ToDouble(features[i, j]);
            }

            bias = NumOps.FromDouble(NumOps.ToDouble(bias) - LearningRate * gradBias / n);
            for (int j = 0; j < p; j++)
            {
                double grad = gradWeights[j] / n + Lambda * NumOps.ToDouble(weights[j]);
                weights[j] = NumOps.FromDouble(NumOps.ToDouble(weights[j]) - LearningRate * grad);
            }
        }

        return (weights, bias);
    }

    /// <summary>
    /// Trains a logistic regression model for propensity scores.
    /// </summary>
    private (Vector<T> weights, T bias) TrainPropensityModel(Matrix<T> features, Vector<int> treatment)
    {
        int n = features.Rows;
        int p = features.Columns;

        var weights = new Vector<T>(p);
        var bias = NumOps.Zero;

        for (int iter = 0; iter < MaxIterations; iter++)
        {
            var gradWeights = new double[p];
            double gradBias = 0;

            for (int i = 0; i < n; i++)
            {
                double z = NumOps.ToDouble(bias);
                for (int j = 0; j < p; j++)
                    z += NumOps.ToDouble(weights[j]) * NumOps.ToDouble(features[i, j]);

                double prob = 1.0 / (1.0 + Math.Exp(-z));
                double error = prob - treatment[i];

                gradBias += error;
                for (int j = 0; j < p; j++)
                    gradWeights[j] += error * NumOps.ToDouble(features[i, j]);
            }

            bias = NumOps.FromDouble(NumOps.ToDouble(bias) - LearningRate * gradBias / n);
            for (int j = 0; j < p; j++)
            {
                double grad = gradWeights[j] / n + Lambda * NumOps.ToDouble(weights[j]);
                weights[j] = NumOps.FromDouble(NumOps.ToDouble(weights[j]) - LearningRate * grad);
            }
        }

        return (weights, bias);
    }

    /// <summary>
    /// Predicts a single outcome using given weights.
    /// </summary>
    private double PredictSingle(Matrix<T> features, int rowIndex, Vector<T> weights, T bias)
    {
        double pred = NumOps.ToDouble(bias);
        for (int j = 0; j < features.Columns; j++)
            pred += NumOps.ToDouble(weights[j]) * NumOps.ToDouble(features[rowIndex, j]);
        return pred;
    }

    /// <summary>
    /// Estimates the Conditional Average Treatment Effect (CATE) using propensity-weighted combination.
    /// </summary>
    public override Vector<T> EstimateTreatmentEffect(Matrix<T> features)
    {
        EnsureFitted();

        int n = features.Rows;
        var effects = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            // Compute propensity score e(X)
            double z = NumOps.ToDouble(_biasPropensity);
            for (int j = 0; j < features.Columns; j++)
                z += NumOps.ToDouble(_weightsPropensity![j]) * NumOps.ToDouble(features[i, j]);
            double propensity = 1.0 / (1.0 + Math.Exp(-z));

            // Compute τ₀(X) - effect estimated from control group
            double tau0 = NumOps.ToDouble(_biasTau0);
            for (int j = 0; j < features.Columns; j++)
                tau0 += NumOps.ToDouble(_weightsTau0![j]) * NumOps.ToDouble(features[i, j]);

            // Compute τ₁(X) - effect estimated from treated group
            double tau1 = NumOps.ToDouble(_biasTau1);
            for (int j = 0; j < features.Columns; j++)
                tau1 += NumOps.ToDouble(_weightsTau1![j]) * NumOps.ToDouble(features[i, j]);

            // Combine: τ(X) = e(X)·τ₀(X) + (1-e(X))·τ₁(X)
            double effect = propensity * tau0 + (1 - propensity) * tau1;
            effects[i] = NumOps.FromDouble(effect);
        }

        return effects;
    }

    /// <summary>
    /// Predicts outcome under treatment using μ₁(X).
    /// </summary>
    public override Vector<T> PredictTreated(Matrix<T> features)
    {
        EnsureFitted();

        var result = new Vector<T>(features.Rows);
        for (int i = 0; i < features.Rows; i++)
        {
            double pred = NumOps.ToDouble(_biasTreated);
            for (int j = 0; j < features.Columns; j++)
                pred += NumOps.ToDouble(_weightsTreated![j]) * NumOps.ToDouble(features[i, j]);
            result[i] = NumOps.FromDouble(pred);
        }
        return result;
    }

    /// <summary>
    /// Predicts outcome under control using μ₀(X).
    /// </summary>
    public override Vector<T> PredictControl(Matrix<T> features)
    {
        EnsureFitted();

        var result = new Vector<T>(features.Rows);
        for (int i = 0; i < features.Rows; i++)
        {
            double pred = NumOps.ToDouble(_biasControl);
            for (int j = 0; j < features.Columns; j++)
                pred += NumOps.ToDouble(_weightsControl![j]) * NumOps.ToDouble(features[i, j]);
            result[i] = NumOps.FromDouble(pred);
        }
        return result;
    }

    /// <summary>
    /// Standard prediction - returns treatment effect.
    /// </summary>
    public override Vector<T> Predict(Matrix<T> input)
    {
        return EstimateTreatmentEffect(input);
    }

    /// <inheritdoc />
    public override (T estimate, T standardError) EstimateATE(Matrix<T> x, Vector<int> treatment, Vector<T> outcome)
    {
        var effects = EstimateTreatmentEffect(x);
        double mean = effects.ToArray().Average(e => NumOps.ToDouble(e));
        double variance = effects.ToArray().Sum(e => Math.Pow(NumOps.ToDouble(e) - mean, 2)) / Math.Max(1, effects.Length - 1);
        double se = Math.Sqrt(variance / effects.Length);
        return (NumOps.FromDouble(mean), NumOps.FromDouble(se));
    }

    /// <inheritdoc />
    public override (T estimate, T standardError) EstimateATT(Matrix<T> x, Vector<int> treatment, Vector<T> outcome)
    {
        var treatedIndices = Enumerable.Range(0, treatment.Length).Where(i => treatment[i] == 1).ToArray();
        if (treatedIndices.Length == 0)
            return (NumOps.Zero, NumOps.Zero);

        var treatedFeatures = new Matrix<T>(treatedIndices.Length, x.Columns);
        for (int i = 0; i < treatedIndices.Length; i++)
            for (int j = 0; j < x.Columns; j++)
                treatedFeatures[i, j] = x[treatedIndices[i], j];

        var effects = EstimateTreatmentEffect(treatedFeatures);
        double mean = effects.ToArray().Average(e => NumOps.ToDouble(e));
        double variance = effects.ToArray().Sum(e => Math.Pow(NumOps.ToDouble(e) - mean, 2)) / Math.Max(1, effects.Length - 1);
        double se = Math.Sqrt(variance / effects.Length);
        return (NumOps.FromDouble(mean), NumOps.FromDouble(se));
    }

    /// <inheritdoc />
    public override Vector<T> EstimateCATEPerIndividual(Matrix<T> x, Vector<int> treatment, Vector<T> outcome)
    {
        return EstimateTreatmentEffect(x);
    }

    /// <inheritdoc />
    public override Vector<T> PredictTreatmentEffect(Matrix<T> x)
    {
        return EstimateTreatmentEffect(x);
    }

    /// <inheritdoc />
    protected override Vector<T> EstimatePropensityScoresCore(Matrix<T> x)
    {
        var result = new Vector<T>(x.Rows);
        for (int i = 0; i < x.Rows; i++)
        {
            double z = NumOps.ToDouble(_biasPropensity);
            for (int j = 0; j < x.Columns; j++)
                z += NumOps.ToDouble(_weightsPropensity![j]) * NumOps.ToDouble(x[i, j]);
            double prob = 1.0 / (1.0 + Math.Exp(-z));
            result[i] = NumOps.FromDouble(prob);
        }
        return result;
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        int p = _weightsControl?.Length ?? 0;
        if (p == 0)
            return new Vector<T>(5); // Just biases

        // Format: [5 biases] + [5 weight vectors]
        var parameters = new Vector<T>(5 + 5 * p);
        parameters[0] = _biasControl;
        parameters[1] = _biasTreated;
        parameters[2] = _biasTau0;
        parameters[3] = _biasTau1;
        parameters[4] = _biasPropensity;

        int offset = 5;
        for (int i = 0; i < p; i++)
        {
            parameters[offset + i] = _weightsControl![i];
            parameters[offset + p + i] = _weightsTreated![i];
            parameters[offset + 2 * p + i] = _weightsTau0![i];
            parameters[offset + 3 * p + i] = _weightsTau1![i];
            parameters[offset + 4 * p + i] = _weightsPropensity![i];
        }

        return parameters;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length < 5) return;

        _biasControl = parameters[0];
        _biasTreated = parameters[1];
        _biasTau0 = parameters[2];
        _biasTau1 = parameters[3];
        _biasPropensity = parameters[4];

        int remaining = parameters.Length - 5;
        if (remaining > 0 && remaining % 5 == 0)
        {
            int p = remaining / 5;
            _weightsControl = new Vector<T>(p);
            _weightsTreated = new Vector<T>(p);
            _weightsTau0 = new Vector<T>(p);
            _weightsTau1 = new Vector<T>(p);
            _weightsPropensity = new Vector<T>(p);

            int offset = 5;
            for (int i = 0; i < p; i++)
            {
                _weightsControl[i] = parameters[offset + i];
                _weightsTreated[i] = parameters[offset + p + i];
                _weightsTau0[i] = parameters[offset + 2 * p + i];
                _weightsTau1[i] = parameters[offset + 3 * p + i];
                _weightsPropensity[i] = parameters[offset + 4 * p + i];
            }
        }
    }

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var copy = new XLearner<T>(MaxIterations, LearningRate, Lambda);
        copy.SetParameters(parameters);
        return copy;
    }

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new XLearner<T>(MaxIterations, LearningRate, Lambda);
    }

    /// <inheritdoc />
    public override ModelType GetModelType() => ModelType.XLearner;
}

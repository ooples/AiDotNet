using System.Text;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using Newtonsoft.Json;

namespace AiDotNet.CausalInference;

/// <summary>
/// Implements the T-Learner (Two-model learner) for treatment effect estimation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> T-Learner trains two separate models: one for the treated group
/// and one for the control group. Treatment effects are estimated by subtracting the control
/// model prediction from the treatment model prediction.</para>
///
/// <para><b>How it works:</b>
/// <list type="number">
/// <item>Train μ₁(X) on treated samples: {(Xᵢ, Yᵢ) : Tᵢ = 1}</item>
/// <item>Train μ₀(X) on control samples: {(Xᵢ, Yᵢ) : Tᵢ = 0}</item>
/// <item>Estimate CATE: τ(X) = μ₁(X) - μ₀(X)</item>
/// </list>
/// </para>
///
/// <para><b>Pros and Cons:</b>
/// <list type="bullet">
/// <item><b>Pro:</b> Can capture complex heterogeneous treatment effects</item>
/// <item><b>Pro:</b> Each model is trained only on relevant data</item>
/// <item><b>Con:</b> Requires sufficient data in both groups</item>
/// <item><b>Con:</b> May have high variance when groups have different covariate distributions</item>
/// </list>
/// </para>
///
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>When you have enough data in both treatment groups</item>
/// <item>When treatment effects are expected to be heterogeneous</item>
/// <item>When covariate distributions are similar across groups</item>
/// </list>
/// </para>
///
/// <para><b>Reference:</b> Künzel et al., "Metalearners for Estimating Heterogeneous Treatment Effects" (2019)</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class TLearner<T> : CausalModelBase<T>
{
    /// <summary>
    /// Weights for the treatment model.
    /// </summary>
    private Vector<T>? _weightsTreated;

    /// <summary>
    /// Weights for the control model.
    /// </summary>
    private Vector<T>? _weightsControl;

    /// <summary>
    /// Bias for the treatment model.
    /// </summary>
    private T _biasTreated;

    /// <summary>
    /// Bias for the control model.
    /// </summary>
    private T _biasControl;

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
    /// Creates a new T-Learner.
    /// </summary>
    /// <param name="maxIterations">Maximum training iterations (default: 100).</param>
    /// <param name="learningRate">Learning rate (default: 0.1).</param>
    /// <param name="lambda">L2 regularization (default: 0.01).</param>
    public TLearner(int maxIterations = 100, double learningRate = 0.1, double lambda = 0.01) : base()
    {
        MaxIterations = maxIterations;
        LearningRate = learningRate;
        Lambda = lambda;
        _biasTreated = NumOps.Zero;
        _biasControl = NumOps.Zero;
    }

    /// <summary>
    /// Fits the T-Learner model.
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

        // Split data by treatment status
        var treatedIndices = new List<int>();
        var controlIndices = new List<int>();

        for (int i = 0; i < n; i++)
        {
            if (treatmentInt[i] == 1)
                treatedIndices.Add(i);
            else
                controlIndices.Add(i);
        }

        // Train treatment model
        (_weightsTreated, _biasTreated) = TrainLinearModel(features, outcome, treatedIndices.ToArray());

        // Train control model
        (_weightsControl, _biasControl) = TrainLinearModel(features, outcome, controlIndices.ToArray());

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

                // Predict
                double pred = NumOps.ToDouble(bias);
                for (int j = 0; j < p; j++)
                    pred += NumOps.ToDouble(weights[j]) * NumOps.ToDouble(features[i, j]);

                // Error
                double error = pred - NumOps.ToDouble(outcome[i]);

                // Gradients
                gradBias += error;
                for (int j = 0; j < p; j++)
                    gradWeights[j] += error * NumOps.ToDouble(features[i, j]);
            }

            // Update with regularization
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
    /// Estimates the Conditional Average Treatment Effect (CATE).
    /// </summary>
    public override Vector<T> EstimateTreatmentEffect(Matrix<T> features)
    {
        EnsureFitted();

        int n = features.Rows;
        var effects = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            // Predict with treatment model
            double predTreated = NumOps.ToDouble(_biasTreated);
            for (int j = 0; j < features.Columns; j++)
                predTreated += NumOps.ToDouble(_weightsTreated![j]) * NumOps.ToDouble(features[i, j]);

            // Predict with control model
            double predControl = NumOps.ToDouble(_biasControl);
            for (int j = 0; j < features.Columns; j++)
                predControl += NumOps.ToDouble(_weightsControl![j]) * NumOps.ToDouble(features[i, j]);

            effects[i] = NumOps.FromDouble(predTreated - predControl);
        }

        return effects;
    }

    /// <summary>
    /// Predicts outcome under treatment.
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
    /// Predicts outcome under control.
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
            result[i] = NumOps.FromDouble(0.5);
        return result;
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        if (_weightsTreated is null || _weightsControl is null)
            return new Vector<T>(2) { [0] = _biasTreated, [1] = _biasControl };

        int p = _weightsTreated.Length;
        var parameters = new Vector<T>(2 + 2 * p);
        parameters[0] = _biasTreated;
        parameters[1] = _biasControl;
        for (int i = 0; i < p; i++)
        {
            parameters[2 + i] = _weightsTreated[i];
            parameters[2 + p + i] = _weightsControl[i];
        }
        return parameters;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length < 2) return;
        _biasTreated = parameters[0];
        _biasControl = parameters[1];

        int remaining = parameters.Length - 2;
        if (remaining > 0 && remaining % 2 == 0)
        {
            int p = remaining / 2;
            _weightsTreated = new Vector<T>(p);
            _weightsControl = new Vector<T>(p);
            for (int i = 0; i < p; i++)
            {
                _weightsTreated[i] = parameters[2 + i];
                _weightsControl[i] = parameters[2 + p + i];
            }
        }
    }

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var copy = new TLearner<T>(MaxIterations, LearningRate, Lambda);
        copy.SetParameters(parameters);
        return copy;
    }

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new TLearner<T>(MaxIterations, LearningRate, Lambda);
    }

    /// <inheritdoc />
    public override ModelType GetModelType() => ModelType.TLearner;
}

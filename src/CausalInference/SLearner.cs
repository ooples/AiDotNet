using System.Text;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using Newtonsoft.Json;

namespace AiDotNet.CausalInference;

/// <summary>
/// Implements the S-Learner (Single-model learner) for treatment effect estimation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> S-Learner is the simplest meta-learner. It trains a single model
/// that predicts outcomes using both covariates AND the treatment indicator as features.
/// Treatment effects are estimated by comparing predictions with treatment=1 vs treatment=0.</para>
///
/// <para><b>How it works:</b>
/// <list type="number">
/// <item>Train a single model: Y = f(X, T) where T is the treatment indicator</item>
/// <item>For each subject, predict Y₁ = f(X, T=1) and Y₀ = f(X, T=0)</item>
/// <item>Treatment effect τ(X) = Y₁ - Y₀</item>
/// </list>
/// </para>
///
/// <para><b>Pros and Cons:</b>
/// <list type="bullet">
/// <item><b>Pro:</b> Simple, uses all data efficiently</item>
/// <item><b>Pro:</b> Works with any supervised learning method</item>
/// <item><b>Con:</b> May underestimate heterogeneous treatment effects if treatment has small signal</item>
/// <item><b>Con:</b> Regularization may shrink treatment effect toward zero</item>
/// </list>
/// </para>
///
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>When you have limited data and can't afford separate models</item>
/// <item>When treatment effects are expected to be relatively homogeneous</item>
/// <item>As a baseline to compare against more complex learners</item>
/// </list>
/// </para>
///
/// <para><b>Reference:</b> Künzel et al., "Metalearners for Estimating Heterogeneous Treatment Effects" (2019)</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SLearner<T> : CausalModelBase<T>
{
    /// <summary>
    /// The model weights (including treatment as a feature).
    /// </summary>
    private Vector<T>? _weights;

    /// <summary>
    /// The bias term.
    /// </summary>
    private T _bias;

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
    /// Creates a new S-Learner.
    /// </summary>
    /// <param name="maxIterations">Maximum training iterations (default: 100).</param>
    /// <param name="learningRate">Learning rate (default: 0.1).</param>
    /// <param name="lambda">L2 regularization (default: 0.01).</param>
    public SLearner(int maxIterations = 100, double learningRate = 0.1, double lambda = 0.01) : base()
    {
        MaxIterations = maxIterations;
        LearningRate = learningRate;
        Lambda = lambda;
        _bias = NumOps.Zero;
    }

    /// <summary>
    /// Fits the S-Learner model.
    /// </summary>
    public override void Fit(Matrix<T> features, Vector<T> treatment, Vector<T> outcome)
    {
        int n = features.Rows;
        int p = features.Columns;
        NumFeatures = p;

        // Convert treatment to int for validation
        var treatmentInt = new Vector<int>(n);
        for (int i = 0; i < n; i++)
            treatmentInt[i] = NumOps.ToDouble(treatment[i]) > 0.5 ? 1 : 0;

        ValidateCausalData(features, treatmentInt, outcome);

        // Create augmented feature matrix with treatment as last column
        var augmentedFeatures = new Matrix<T>(n, p + 1);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
                augmentedFeatures[i, j] = features[i, j];
            augmentedFeatures[i, p] = treatment[i];
        }

        // Initialize weights (p features + 1 treatment)
        _weights = new Vector<T>(p + 1);
        _bias = NumOps.Zero;

        // Train with gradient descent (ridge regression)
        for (int iter = 0; iter < MaxIterations; iter++)
        {
            var gradWeights = new double[p + 1];
            double gradBias = 0;

            for (int i = 0; i < n; i++)
            {
                // Predict
                double pred = NumOps.ToDouble(_bias);
                for (int j = 0; j <= p; j++)
                    pred += NumOps.ToDouble(_weights[j]) * NumOps.ToDouble(augmentedFeatures[i, j]);

                // Error
                double error = pred - NumOps.ToDouble(outcome[i]);

                // Gradients
                gradBias += error;
                for (int j = 0; j <= p; j++)
                    gradWeights[j] += error * NumOps.ToDouble(augmentedFeatures[i, j]);
            }

            // Update with regularization
            _bias = NumOps.FromDouble(NumOps.ToDouble(_bias) - LearningRate * gradBias / n);
            for (int j = 0; j <= p; j++)
            {
                double grad = gradWeights[j] / n + Lambda * NumOps.ToDouble(_weights[j]);
                _weights[j] = NumOps.FromDouble(NumOps.ToDouble(_weights[j]) - LearningRate * grad);
            }
        }

        IsFitted = true;
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
            // Predict with treatment = 1
            double predTreated = NumOps.ToDouble(_bias);
            for (int j = 0; j < features.Columns; j++)
                predTreated += NumOps.ToDouble(_weights![j]) * NumOps.ToDouble(features[i, j]);
            predTreated += NumOps.ToDouble(_weights![features.Columns]) * 1.0; // T = 1

            // Predict with treatment = 0
            double predControl = NumOps.ToDouble(_bias);
            for (int j = 0; j < features.Columns; j++)
                predControl += NumOps.ToDouble(_weights[j]) * NumOps.ToDouble(features[i, j]);
            predControl += NumOps.ToDouble(_weights[features.Columns]) * 0.0; // T = 0

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
            double pred = NumOps.ToDouble(_bias);
            for (int j = 0; j < features.Columns; j++)
                pred += NumOps.ToDouble(_weights![j]) * NumOps.ToDouble(features[i, j]);
            pred += NumOps.ToDouble(_weights![features.Columns]) * 1.0;
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
            double pred = NumOps.ToDouble(_bias);
            for (int j = 0; j < features.Columns; j++)
                pred += NumOps.ToDouble(_weights![j]) * NumOps.ToDouble(features[i, j]);
            // Treatment = 0, so treatment weight not added
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
        // S-Learner doesn't model propensity scores directly
        // Return uniform 0.5 probabilities
        var result = new Vector<T>(x.Rows);
        for (int i = 0; i < x.Rows; i++)
            result[i] = NumOps.FromDouble(0.5);
        return result;
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        if (_weights is null)
            return new Vector<T>(1) { [0] = _bias };

        var parameters = new Vector<T>(_weights.Length + 1);
        parameters[0] = _bias;
        for (int i = 0; i < _weights.Length; i++)
            parameters[i + 1] = _weights[i];
        return parameters;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length == 0) return;
        _bias = parameters[0];
        if (parameters.Length > 1)
        {
            _weights = new Vector<T>(parameters.Length - 1);
            for (int i = 0; i < _weights.Length; i++)
                _weights[i] = parameters[i + 1];
        }
    }

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var copy = new SLearner<T>(MaxIterations, LearningRate, Lambda);
        copy.SetParameters(parameters);
        return copy;
    }

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new SLearner<T>(MaxIterations, LearningRate, Lambda);
    }

    /// <inheritdoc />
    public override ModelType GetModelType() => ModelType.SLearner;
}

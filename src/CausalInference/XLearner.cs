using System.Text;
using AiDotNet.Attributes;
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
/// <example>
/// <code>
/// var newFeatures = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// var outcome = new Vector&lt;double&gt;(new double[] { 0.0, 1.0, 0.0, 1.0 });
/// var treatment = new Vector&lt;double&gt;(new double[] { 0.0, 1.0, 0.0, 1.0 });
/// var xLearner = new XLearner&lt;double&gt;(maxIterations: 100, learningRate: 0.1);
/// xLearner.Fit(features, treatment, outcome);
/// Vector&lt;double&gt; cate = xLearner.EstimateCate(newFeatures);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Healthcare)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Metalearners for estimating heterogeneous treatment effects using machine learning", "https://doi.org/10.1073/pnas.1804597116", Year = 2019, Authors = "Sören R. Künzel, Jasjeet S. Sekhon, Peter J. Bickel, Bin Yu")]
public partial class XLearner<T> : CausalModelBase<T>
{
    /// <inheritdoc />
    /// <remarks>
    /// Derived from the getter, which is what ModelBase already does. The inherited override
    /// computes NumFeatures x NumClasses, and this model has no such dense weight matrix -- it is
    /// a forest / ensemble / AFT fit -- so the formula answered 0 while the getter returned real
    /// values. SetParameters pairs the two by length, so the disagreement is not cosmetic.
    /// </remarks>
    private const int ControlModel = 0;
    private const int TreatedModel = 1;
    private const int Tau0Model = 2;
    private const int Tau1Model = 3;
    private const int PropensityModel = 4;
    private const int ModelCount = 5;

    /// <summary>
    /// One row per X-Learner regression: control outcome, treated outcome, the two imputed-effect
    /// models, and propensity. The fixed leading axis makes ownership explicit while the deferred
    /// feature axis is inferred from fitted data or a restore payload.
    /// </summary>
    [TrainableParameter(Availability = AiDotNet.Models.Parameters.ParameterAvailability.Fit)]
    private Tensor<T> _weights = new([ModelCount, 0]);

    /// <summary>
    /// Bias terms for each model.
    /// </summary>
    [TrainableParameter(Role = PersistentTensorRole.Biases)]
    private readonly Tensor<T> _biases = new([ModelCount]);

    private T BiasControl { get => _biases[ControlModel]; set => _biases[ControlModel] = value; }
    private T BiasTreated { get => _biases[TreatedModel]; set => _biases[TreatedModel] = value; }
    private T BiasTau0 { get => _biases[Tau0Model]; set => _biases[Tau0Model] = value; }
    private T BiasTau1 { get => _biases[Tau1Model]; set => _biases[Tau1Model] = value; }
    private T BiasPropensity { get => _biases[PropensityModel]; set => _biases[PropensityModel] = value; }

    private T Weight(int model, int feature) => _weights[model, feature];

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
        BiasControl = NumOps.Zero;
        BiasTreated = NumOps.Zero;
        BiasTau0 = NumOps.Zero;
        BiasTau1 = NumOps.Zero;
        BiasPropensity = NumOps.Zero;
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
            treatmentInt[i] = NumOps.GreaterThan(treatment[i], NumOps.FromDouble(0.5)) ? 1 : 0;

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
        var (weightsTreated, biasTreated) = TrainLinearModel(features, outcome, treatedIndices.ToArray());
        var (weightsControl, biasControl) = TrainLinearModel(features, outcome, controlIndices.ToArray());

        // Stage 2: Compute imputed treatment effects
        // For treated: D₁ = Y₁ - μ₀(X₁) (actual outcome minus predicted control outcome)
        var imputedTreated = new Vector<T>(treatedIndices.Count);
        for (int idx = 0; idx < treatedIndices.Count; idx++)
        {
            int i = treatedIndices[idx];
            double actualOutcome = NumOps.ToDouble(outcome[i]);
            double predictedControl = PredictSingle(features, i, weightsControl, biasControl);
            imputedTreated[idx] = NumOps.FromDouble(actualOutcome - predictedControl);
        }

        // For control: D₀ = μ₁(X₀) - Y₀ (predicted treated outcome minus actual outcome)
        var imputedControl = new Vector<T>(controlIndices.Count);
        for (int idx = 0; idx < controlIndices.Count; idx++)
        {
            int i = controlIndices[idx];
            double predictedTreated = PredictSingle(features, i, weightsTreated, biasTreated);
            double actualOutcome = NumOps.ToDouble(outcome[i]);
            imputedControl[idx] = NumOps.FromDouble(predictedTreated - actualOutcome);
        }

        // Stage 3: Train treatment effect models
        var (weightsTau1, biasTau1) = TrainLinearModelWithOutcome(features, imputedTreated, treatedIndices.ToArray());
        var (weightsTau0, biasTau0) = TrainLinearModelWithOutcome(features, imputedControl, controlIndices.ToArray());

        // Stage 4: Train propensity score model (logistic regression)
        var (weightsPropensity, biasPropensity) = TrainPropensityModel(features, treatmentInt);

        _weights = new Tensor<T>([ModelCount, p]);
        CopyWeightRow(ControlModel, weightsControl);
        CopyWeightRow(TreatedModel, weightsTreated);
        CopyWeightRow(Tau0Model, weightsTau0);
        CopyWeightRow(Tau1Model, weightsTau1);
        CopyWeightRow(PropensityModel, weightsPropensity);
        BiasControl = biasControl;
        BiasTreated = biasTreated;
        BiasTau0 = biasTau0;
        BiasTau1 = biasTau1;
        BiasPropensity = biasPropensity;

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

    private void CopyWeightRow(int model, Vector<T> values)
    {
        if (values.Length != NumFeatures)
            throw new InvalidOperationException(
                $"Model row has {values.Length} weights; expected {NumFeatures}.");
        for (int feature = 0; feature < values.Length; feature++)
            _weights[model, feature] = values[feature];
    }

    /// <summary>
    /// Estimates the Conditional Average Treatment Effect (CATE) using propensity-weighted combination.
    /// </summary>
    public override Vector<T> EstimateTreatmentEffect(Matrix<T> features)
    {
        EnsureFitted();
        features = NormalizeFeatureInput(features, nameof(features));

        int n = features.Rows;
        var effects = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            // Compute propensity score e(X)
            double z = NumOps.ToDouble(BiasPropensity);
            for (int j = 0; j < features.Columns; j++)
                z += NumOps.ToDouble(Weight(PropensityModel, j)) * NumOps.ToDouble(features[i, j]);
            double propensity = 1.0 / (1.0 + Math.Exp(-z));

            // Compute τ₀(X) - effect estimated from control group
            double tau0 = NumOps.ToDouble(BiasTau0);
            for (int j = 0; j < features.Columns; j++)
                tau0 += NumOps.ToDouble(Weight(Tau0Model, j)) * NumOps.ToDouble(features[i, j]);

            // Compute τ₁(X) - effect estimated from treated group
            double tau1 = NumOps.ToDouble(BiasTau1);
            for (int j = 0; j < features.Columns; j++)
                tau1 += NumOps.ToDouble(Weight(Tau1Model, j)) * NumOps.ToDouble(features[i, j]);

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
        features = NormalizeFeatureInput(features, nameof(features));

        var result = new Vector<T>(features.Rows);
        for (int i = 0; i < features.Rows; i++)
        {
            double pred = NumOps.ToDouble(BiasTreated);
            for (int j = 0; j < features.Columns; j++)
                pred += NumOps.ToDouble(Weight(TreatedModel, j)) * NumOps.ToDouble(features[i, j]);
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
        features = NormalizeFeatureInput(features, nameof(features));

        var result = new Vector<T>(features.Rows);
        for (int i = 0; i < features.Rows; i++)
        {
            double pred = NumOps.ToDouble(BiasControl);
            for (int j = 0; j < features.Columns; j++)
                pred += NumOps.ToDouble(Weight(ControlModel, j)) * NumOps.ToDouble(features[i, j]);
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

    private Matrix<T> NormalizeFeatureInput(Matrix<T> input, string paramName)
    {
        if (input.Columns == NumFeatures)
            return input;

        if (input.Columns == NumFeatures + 1)
        {
            var features = new Matrix<T>(input.Rows, NumFeatures);
            for (int i = 0; i < input.Rows; i++)
                for (int j = 0; j < NumFeatures; j++)
                    features[i, j] = input[i, j + 1];
            return features;
        }

        throw new ArgumentException(
            $"Input must have {NumFeatures} covariate columns, or {NumFeatures + 1} " +
            $"columns where the first is the treatment indicator. Got {input.Columns}.",
            paramName);
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
        x = NormalizeFeatureInput(x, nameof(x));
        var result = new Vector<T>(x.Rows);
        for (int i = 0; i < x.Rows; i++)
        {
            double z = NumOps.ToDouble(BiasPropensity);
            for (int j = 0; j < x.Columns; j++)
                z += NumOps.ToDouble(Weight(PropensityModel, j)) * NumOps.ToDouble(x[i, j]);
            double prob = 1.0 / (1.0 + Math.Exp(-z));
            result[i] = NumOps.FromDouble(prob);
        }
        return result;
    }

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var copy = new XLearner<T>(MaxIterations, LearningRate, Lambda);
        copy.SetParameters(parameters);
        return copy;
    }

    /// <inheritdoc />
    protected override Dictionary<string, object> GetAdditionalModelData()
    {
        var data = base.GetAdditionalModelData();
        data["BiasControl"] = NumOps.ToDouble(BiasControl);
        data["BiasTreated"] = NumOps.ToDouble(BiasTreated);
        data["BiasTau0"] = NumOps.ToDouble(BiasTau0);
        data["BiasTau1"] = NumOps.ToDouble(BiasTau1);
        data["BiasPropensity"] = NumOps.ToDouble(BiasPropensity);
        data["WeightsControl"] = ToDoubleArray(ControlModel);
        data["WeightsTreated"] = ToDoubleArray(TreatedModel);
        data["WeightsTau0"] = ToDoubleArray(Tau0Model);
        data["WeightsTau1"] = ToDoubleArray(Tau1Model);
        data["WeightsPropensity"] = ToDoubleArray(PropensityModel);
        return data;
    }

    /// <inheritdoc />
    protected override void LoadAdditionalModelData(Newtonsoft.Json.Linq.JObject modelDataObj)
    {
        base.LoadAdditionalModelData(modelDataObj);
        if (modelDataObj["BiasControl"] is not null)
            BiasControl = NumOps.FromDouble(modelDataObj["BiasControl"]!.ToObject<double>());
        if (modelDataObj["BiasTreated"] is not null)
            BiasTreated = NumOps.FromDouble(modelDataObj["BiasTreated"]!.ToObject<double>());
        if (modelDataObj["BiasTau0"] is not null)
            BiasTau0 = NumOps.FromDouble(modelDataObj["BiasTau0"]!.ToObject<double>());
        if (modelDataObj["BiasTau1"] is not null)
            BiasTau1 = NumOps.FromDouble(modelDataObj["BiasTau1"]!.ToObject<double>());
        if (modelDataObj["BiasPropensity"] is not null)
            BiasPropensity = NumOps.FromDouble(modelDataObj["BiasPropensity"]!.ToObject<double>());

        var rows = new[]
        {
            FromJsonArray(modelDataObj["WeightsControl"]),
            FromJsonArray(modelDataObj["WeightsTreated"]),
            FromJsonArray(modelDataObj["WeightsTau0"]),
            FromJsonArray(modelDataObj["WeightsTau1"]),
            FromJsonArray(modelDataObj["WeightsPropensity"]),
        };
        int width = rows[0].Length;
        if (rows.Any(row => row.Length != width))
            throw new JsonSerializationException("X-Learner weight rows must all have the same width.");
        _weights = new Tensor<T>([ModelCount, width]);
        for (int model = 0; model < ModelCount; model++)
            for (int feature = 0; feature < width; feature++)
                _weights[model, feature] = rows[model][feature];
    }

    private double[] ToDoubleArray(int model)
    {
        int width = _weights.Shape.Length == 2 ? _weights.Shape[1] : 0;
        var values = new double[width];
        for (int i = 0; i < width; i++)
            values[i] = NumOps.ToDouble(_weights[model, i]);
        return values;
    }

    private T[] FromJsonArray(Newtonsoft.Json.Linq.JToken? token)
    {
        if (token is not Newtonsoft.Json.Linq.JArray array)
            return Array.Empty<T>();

        var vector = new T[array.Count];
        for (int i = 0; i < array.Count; i++)
            vector[i] = NumOps.FromDouble(array[i].ToObject<double>());
        return vector;
    }

    /// <inheritdoc />
}

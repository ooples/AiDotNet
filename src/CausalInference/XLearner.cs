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
/// var covariates = new Matrix&lt;double&gt;(new double[,]
/// {
///     { 45, 1 }, { 52, 0 }, { 38, 1 }, { 61, 0 }, { 47, 1 }, { 55, 0 }
/// });
/// var treatment = new Vector&lt;int&gt;(new int[] { 0, 1, 0, 1, 0, 1 });   // who was treated
/// var outcome = new Vector&lt;double&gt;(new double[] { 3.1, 7.2, 2.8, 8.0, 3.4, 7.6 });
///
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new XLearner&lt;double&gt;(lambda: 0.01))
///     .Build(covariates, treatment, outcome);
///
/// // One effect per person, over the covariates alone.
/// Vector&lt;double&gt; cate = result.EstimateTreatmentEffect(covariates);
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

    /// <summary>The fitted slopes of one of the five models.</summary>
    private Vector<T> WeightRow(int model)
        => Engine.Reshape(Engine.TensorNarrow(_weights, 0, model, 1), new[] { _weights.Shape[1] }).ToVector();

    /// <summary>
    /// Gets the L2 regularization strength.
    /// </summary>
    public double Lambda { get; }

    /// <summary>
    /// The removed gradient-descent signature, kept only so that a positional call such as <c>new XLearner&lt;T&gt;(100)</c>
    /// fails to compile instead of silently binding the integer to <c>lambda</c>.
    /// </summary>
    /// <param name="maxIterations">No longer used: the outcome models are solved exactly.</param>
    /// <remarks>
    /// Only the one-integer form needs a guard: the old two- and three-argument calls match no constructor and
    /// already fail to compile.
    /// </remarks>
    [Obsolete("The iteration count and learning rate are gone: the outcome models are ridge regressions solved " +
        "exactly. Use XLearner(lambda: ...).", error: true)]
    public XLearner(int maxIterations) : this()
    {
    }

    /// <summary>
    /// Creates a new X-Learner.
    /// </summary>
    /// <param name="lambda">L2 regularization of the ridge outcome and effect models (default: 0.01).</param>
    /// <remarks>
    /// The outcome and effect models are ridge regressions solved exactly and the propensity model is the base
    /// logistic regression (scikit-learn's default L2 fit, as DoWhy uses). They used to be approximated by a fixed
    /// number of gradient steps, which diverged on unscaled covariates, and is why the iteration count and learning
    /// rate are gone.
    /// </remarks>
    public XLearner(double lambda = 0.01) : base()
    {
        Lambda = lambda;
        BiasControl = NumOps.Zero;
        BiasTreated = NumOps.Zero;
        BiasTau0 = NumOps.Zero;
        BiasTau1 = NumOps.Zero;
        BiasPropensity = NumOps.Zero;
    }

    /// <summary>
    /// Fits the X-Learner model.
    /// </summary>
    public override void Fit(Matrix<T> features, Vector<T> treatment, Vector<T> outcome)
    {
        int p = features.Columns;
        NumFeatures = p;

        var treatmentInt = BinarizeTreatment(treatment);
        ValidateCausalData(features, treatmentInt, outcome);

        // Split by treatment status
        var treatedIndices = GroupIndices(treatmentInt, 1);
        var controlIndices = GroupIndices(treatmentInt, 0);
        var treatedFeatures = SelectRows(features, treatedIndices);
        var controlFeatures = SelectRows(features, controlIndices);
        var treatedOutcome = SelectElements(outcome, treatedIndices);
        var controlOutcome = SelectElements(outcome, controlIndices);

        // Stage 1: Train outcome models (T-Learner style)
        var (weightsTreated, biasTreated) = FitRidgeRegression(treatedFeatures, treatedOutcome, Lambda);
        var (weightsControl, biasControl) = FitRidgeRegression(controlFeatures, controlOutcome, Lambda);

        // Stage 2: Compute imputed treatment effects
        // For treated: D₁ = Y₁ - μ₀(X₁) (actual outcome minus predicted control outcome)
        var imputedTreated = Engine.Subtract(treatedOutcome, PredictLinear(treatedFeatures, weightsControl, biasControl));

        // For control: D₀ = μ₁(X₀) - Y₀ (predicted treated outcome minus actual outcome)
        var imputedControl = Engine.Subtract(PredictLinear(controlFeatures, weightsTreated, biasTreated), controlOutcome);

        // Stage 3: Train treatment effect models
        var (weightsTau1, biasTau1) = FitRidgeRegression(treatedFeatures, imputedTreated, Lambda);
        var (weightsTau0, biasTau0) = FitRidgeRegression(controlFeatures, imputedControl, Lambda);

        // Stage 4: Train propensity score model (logistic regression; intercept first)
        var propensity = FitLogisticRegression(features, treatmentInt);

        var rows = new Vector<T>[ModelCount];
        rows[ControlModel] = weightsControl;
        rows[TreatedModel] = weightsTreated;
        rows[Tau0Model] = weightsTau0;
        rows[Tau1Model] = weightsTau1;
        rows[PropensityModel] = propensity.GetSubVector(1, p);
        _weights = StackRows(rows);
        BiasControl = biasControl;
        BiasTreated = biasTreated;
        BiasTau0 = biasTau0;
        BiasTau1 = biasTau1;
        BiasPropensity = propensity[0];

        IsFitted = true;
    }

    /// <summary>Stacks the five models' slopes into the [5, p] weight tensor.</summary>
    private Tensor<T> StackRows(Vector<T>[] rows)
    {
        var tensors = new Tensor<T>[ModelCount];
        for (int model = 0; model < ModelCount; model++)
        {
            if (rows[model].Length != NumFeatures)
                throw new InvalidOperationException(
                    $"Model row has {rows[model].Length} weights; expected {NumFeatures}.");
            tensors[model] = rows[model].AsTensor();
        }

        return Engine.TensorStack(tensors, 0);
    }

    /// <summary>
    /// Estimates the Conditional Average Treatment Effect (CATE).
    /// </summary>
    public override Vector<T> EstimateTreatmentEffect(Matrix<T> features)
    {
        EnsureFitted();
        features = NormalizeFeatureInput(features, nameof(features));

        // Propensity score e(X), and τ₀(X), τ₁(X) estimated from the control and treated groups
        var propensity = Engine.Sigmoid(PredictLinear(features, WeightRow(PropensityModel), BiasPropensity)).AsTensor();
        var tau0 = PredictLinear(features, WeightRow(Tau0Model), BiasTau0).AsTensor();
        var tau1 = PredictLinear(features, WeightRow(Tau1Model), BiasTau1).AsTensor();

        // Combine: τ(X) = e(X)·τ₀(X) + (1-e(X))·τ₁(X) = τ₁(X) + e(X)·(τ₀(X) − τ₁(X))
        return Engine.TensorAdd(tau1, Engine.TensorMultiply(propensity, Engine.TensorSubtract(tau0, tau1))).ToVector();
    }

    /// <summary>
    /// Predicts outcome under treatment.
    /// </summary>
    public override Vector<T> PredictTreated(Matrix<T> features)
    {
        EnsureFitted();
        features = NormalizeFeatureInput(features, nameof(features));
        return PredictLinear(features, WeightRow(TreatedModel), BiasTreated);
    }

    /// <summary>
    /// Predicts outcome under control.
    /// </summary>
    public override Vector<T> PredictControl(Matrix<T> features)
    {
        EnsureFitted();
        features = NormalizeFeatureInput(features, nameof(features));
        return PredictLinear(features, WeightRow(ControlModel), BiasControl);
    }

    /// <summary>
    /// Standard prediction - returns treatment effects.
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
            return DropLeadingColumn(input);

        throw new ArgumentException(
            $"Input must have {NumFeatures} covariate columns, or {NumFeatures + 1} " +
            $"columns where the first is the treatment indicator. Got {input.Columns}.",
            paramName);
    }

    /// <inheritdoc />
    public override (T estimate, T standardError) EstimateATE(Matrix<T> x, Vector<int> treatment, Vector<T> outcome)
    {
        return MeanWithStandardError(EstimateTreatmentEffect(x));
    }

    /// <inheritdoc />
    public override (T estimate, T standardError) EstimateATT(Matrix<T> x, Vector<int> treatment, Vector<T> outcome)
    {
        var treatedIndices = GroupIndices(treatment, 1);
        if (treatedIndices.Length == 0)
            return (NumOps.Zero, NumOps.Zero);

        return MeanWithStandardError(EstimateTreatmentEffect(SelectRows(x, treatedIndices)));
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
        return Engine.Sigmoid(PredictLinear(x, WeightRow(PropensityModel), BiasPropensity));
    }

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var copy = new XLearner<T>(Lambda);
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
        data["WeightsControl"] = WeightArray(ControlModel);
        data["WeightsTreated"] = WeightArray(TreatedModel);
        data["WeightsTau0"] = WeightArray(Tau0Model);
        data["WeightsTau1"] = WeightArray(Tau1Model);
        data["WeightsPropensity"] = WeightArray(PropensityModel);
        return data;
    }

    /// <inheritdoc />
    protected override void LoadAdditionalModelData(Newtonsoft.Json.Linq.JObject modelDataObj)
    {
        base.LoadAdditionalModelData(modelDataObj);
        if (modelDataObj["BiasControl"] is not null)
            BiasControl = NumOps.FromDouble((modelDataObj["BiasControl"] ?? throw new InvalidOperationException("modelDataObj[\"BiasControl\"] is null where it is required.")).ToObject<double>());
        if (modelDataObj["BiasTreated"] is not null)
            BiasTreated = NumOps.FromDouble((modelDataObj["BiasTreated"] ?? throw new InvalidOperationException("modelDataObj[\"BiasTreated\"] is null where it is required.")).ToObject<double>());
        if (modelDataObj["BiasTau0"] is not null)
            BiasTau0 = NumOps.FromDouble((modelDataObj["BiasTau0"] ?? throw new InvalidOperationException("modelDataObj[\"BiasTau0\"] is null where it is required.")).ToObject<double>());
        if (modelDataObj["BiasTau1"] is not null)
            BiasTau1 = NumOps.FromDouble((modelDataObj["BiasTau1"] ?? throw new InvalidOperationException("modelDataObj[\"BiasTau1\"] is null where it is required.")).ToObject<double>());
        if (modelDataObj["BiasPropensity"] is not null)
            BiasPropensity = NumOps.FromDouble((modelDataObj["BiasPropensity"] ?? throw new InvalidOperationException("modelDataObj[\"BiasPropensity\"] is null where it is required.")).ToObject<double>());
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
        _weights = Engine.TensorStack(rows.Select(row => new Tensor<T>(row, new[] { width })).ToArray(), 0);
    }

    private T[] WeightArray(int model)
        => _weights.Shape.Length == 2 ? WeightRow(model).ToArray() : Array.Empty<T>();

    private static T[] FromJsonArray(Newtonsoft.Json.Linq.JToken? token)
    {
        if (token is not Newtonsoft.Json.Linq.JArray array)
            return Array.Empty<T>();

        return array.ToObject<T[]>() ?? Array.Empty<T>();
    }
}

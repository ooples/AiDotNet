using System.Text;
using AiDotNet.Attributes;
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
/// <para><b>Recommended:</b> Use <c>AiModelBuilder</c> for the simplest entry point.</para>
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
///     .ConfigureModel(new SLearner&lt;double&gt;(lambda: 0.01))
///     .Build(covariates, treatment, outcome);
///
/// // One effect per person, over the covariates alone.
/// Vector&lt;double&gt; cate = result.EstimateTreatmentEffect(covariates);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.CausalInference)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Metalearners for Estimating Heterogeneous Treatment Effects using Machine Learning", "https://doi.org/10.1073/pnas.1804597116", Year = 2019, Authors = "Sören R. Künzel, Jasjeet S. Sekhon, Peter J. Bickel, Bin Yu")]
public partial class SLearner<T> : CausalModelBase<T>
{
    /// <summary>
    /// The model weights (including treatment as a feature).
    /// </summary>
    [TrainableParameter(Availability = AiDotNet.Models.Parameters.ParameterAvailability.Fit)]
    private Tensor<T> _weights = new([0]);

    /// <summary>
    /// The bias term.
    /// </summary>
    [TrainableParameter(Role = PersistentTensorRole.Biases)]
    private T _bias;

    /// <summary>
    /// Gets the L2 regularization strength.
    /// </summary>
    public double Lambda { get; }

    /// <summary>
    /// Creates a new S-Learner.
    /// </summary>
    /// <param name="lambda">L2 regularization of the ridge outcome model (default: 0.01).</param>
    /// <remarks>
    /// The outcome model is ridge regression on <c>[X, T]</c>, solved exactly. It used to be approximated by a fixed
    /// number of gradient steps, which diverged on unscaled covariates (ages, incomes) and is why the iteration count
    /// and learning rate are gone.
    /// </remarks>
    public SLearner(double lambda = 0.01) : base()
    {
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

        ValidateCausalData(features, BinarizeTreatment(treatment), outcome);

        // Augmented feature matrix with treatment as the last column
        var augmentedFeatures = Engine.TensorConcatenate(new[]
        {
            Tensor<T>.FromMatrix(features),
            Engine.Reshape(treatment.AsTensor(), new[] { n, 1 }),
        }, 1).ToMatrix();

        var (weights, bias) = FitRidgeRegression(augmentedFeatures, outcome, Lambda);
        _weights = Tensor<T>.FromVector(weights);
        _bias = bias;

        IsFitted = true;
    }

    /// <summary>
    /// Estimates the Conditional Average Treatment Effect (CATE).
    /// </summary>
    /// <remarks>
    /// The outcome model is linear in the treatment indicator, so f(x, 1) − f(x, 0) is the treatment weight for every
    /// subject.
    /// </remarks>
    public override Vector<T> EstimateTreatmentEffect(Matrix<T> features)
    {
        EnsureFitted();
        features = ExtractCovariates(features);
        return Tensor<T>.CreateDefault(new[] { features.Rows }, TreatmentWeight).ToVector();
    }

    /// <summary>
    /// Predicts outcome under treatment.
    /// </summary>
    public override Vector<T> PredictTreated(Matrix<T> features)
    {
        EnsureFitted();
        return PredictLinear(ExtractCovariates(features), CovariateWeights, NumOps.Add(_bias, TreatmentWeight));
    }

    /// <summary>
    /// Predicts outcome under control.
    /// </summary>
    public override Vector<T> PredictControl(Matrix<T> features)
    {
        EnsureFitted();
        // Treatment = 0, so the treatment weight is not added
        return PredictLinear(ExtractCovariates(features), CovariateWeights, _bias);
    }

    /// <summary>
    /// Standard prediction — returns the predicted outcome for each input row.
    /// </summary>
    /// <remarks>
    /// <see cref="CausalModelBase{T}.Train(Matrix{T}, Vector{T})"/> splits the
    /// input matrix into <c>features</c> (columns 1..end) and <c>treatment</c>
    /// (column 0) before fitting on the joint <c>[features ∥ treatment] → outcome</c>
    /// regression. The standard <see cref="IFullModel{T, TInput, TOutput}.Predict"/>
    /// contract returns predicted outcomes (so <c>R²</c> / residual / coefficient-sign
    /// invariants from <c>RegressionModelTestBase</c> are well-defined): read each row's
    /// treatment from column 0, evaluate <c>bias + Σ_j w_j · feature_j +
    /// w_treatment · treatment</c>, and return the row's predicted outcome.
    /// Treatment-effect (CATE) consumers should call
    /// <see cref="EstimateTreatmentEffect"/> directly.
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        EnsureFitted();

        // Accept both [N, p+1] (treatment in col 0, covariates 1..) — the
        // augmented form Train uses — and [N, p] (pure covariates with no
        // treatment column). The augmented form is what the regression test
        // bases pass; the pure-features form is what <see cref="EstimateTreatmentEffect"/>
        // documents and tests like InterceptRecovery_ConstantTarget_ShouldPredictConstant
        // exercise via the regression base too.
        bool hasTreatmentCol = input.Columns == NumFeatures + 1;
        if (!hasTreatmentCol && input.Columns != NumFeatures)
        {
            throw new ArgumentException(
                $"Predict input has {input.Columns} columns but model was trained on " +
                $"{NumFeatures} covariate(s). Pass either [N, {NumFeatures}] (pure covariates, " +
                $"treatment defaulted to 0) or [N, {NumFeatures + 1}] (treatment in col 0, " +
                $"covariates 1..{NumFeatures}).",
                nameof(input));
        }

        if (!hasTreatmentCol)
        {
            // Pure covariates with no treatment column — assume control (T=0)
            // so the user gets the no-treatment outcome prediction.
            return PredictLinear(input, CovariateWeights, _bias);
        }

        // Layout: [treatment, covariate_0, covariate_1, ...]. The covariate weights
        // _weights[0..p-1] map to columns 1..p of input, and _weights[p] is the treatment weight.
        var layout = Tensor<T>.FromMatrix(input);
        var treatment = Engine.Reshape(Engine.TensorNarrow(layout, 1, 0, 1), new[] { input.Rows });
        var outcome = PredictLinear(DropLeadingColumn(input), CovariateWeights, _bias).AsTensor();
        return Engine.TensorAdd(outcome, Engine.TensorMultiplyScalar(treatment, TreatmentWeight)).ToVector();
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
        // S-Learner doesn't model propensity scores directly
        // Return uniform 0.5 probabilities
        return Tensor<T>.CreateDefault(new[] { x.Rows }, NumOps.FromDouble(0.5)).ToVector();
    }

    /// <summary>The fitted covariate slopes, <c>_weights[0..p-1]</c>.</summary>
    private Vector<T> CovariateWeights => Engine.TensorNarrow(_weights, 0, 0, NumFeatures).ToVector();

    /// <summary>The fitted slope of the treatment indicator, <c>_weights[p]</c>.</summary>
    private T TreatmentWeight => _weights[NumFeatures];

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var copy = new SLearner<T>(Lambda);
        copy.SetParameters(parameters);
        return copy;
    }

    /// <inheritdoc />
    /// <remarks>
    /// SLearner's trainable state is the bias scalar and the weight vector
    /// (covariate weights + treatment weight). Without this hook the
    /// CausalModelBase serializer only persists NumFeatures + IsFitted, so
    /// Clone() / SaveModel() / LoadModel() round-trips would create a clone
    /// with reset weights and Predict would either return the bias-only
    /// constant or hit an out-of-range when treating NumFeatures as a weight
    /// index against a length-0 vector.
    /// </remarks>
    protected override Dictionary<string, object> GetAdditionalModelData()
    {
        var data = base.GetAdditionalModelData();
        data["Bias"] = NumOps.ToDouble(_bias);
        data["Weights"] = _weights.ToArray();
        return data;
    }

    /// <inheritdoc />
    protected override void LoadAdditionalModelData(Newtonsoft.Json.Linq.JObject modelDataObj)
    {
        base.LoadAdditionalModelData(modelDataObj);
        if (modelDataObj["Bias"] is not null)
            _bias = NumOps.FromDouble((modelDataObj["Bias"] ?? throw new InvalidOperationException("modelDataObj[\"Bias\"] is null where it is required.")).ToObject<double>());
        if (modelDataObj["Weights"] is Newtonsoft.Json.Linq.JArray weightArr)
        {
            var weights = weightArr.ToObject<T[]>() ?? Array.Empty<T>();
            _weights = new Tensor<T>(weights, new[] { weights.Length });
        }
    }
}

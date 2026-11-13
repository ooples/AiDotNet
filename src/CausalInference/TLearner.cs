using System.Text;
using AiDotNet.Attributes;
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
///     .ConfigureModel(new TLearner&lt;double&gt;(lambda: 0.01))
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
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
    [ResearchPaper("Metalearners for Estimating Heterogeneous Treatment Effects Using Machine Learning", "https://doi.org/10.1073/pnas.1804597116")]
public partial class TLearner<T> : CausalModelBase<T>
{
    /// <summary>
    /// Weights for the treatment model.
    /// </summary>
    private Vector<T> _weightsTreated = new Vector<T>(0);

    /// <summary>
    /// Weights for the control model.
    /// </summary>
    private Vector<T> _weightsControl = new Vector<T>(0);

    /// <summary>
    /// Bias for the treatment model.
    /// </summary>
    private T _biasTreated;

    /// <summary>
    /// Bias for the control model.
    /// </summary>
    private T _biasControl;

    /// <summary>
    /// Gets the L2 regularization strength.
    /// </summary>
    public double Lambda { get; }

    /// <summary>
    /// Creates a new T-Learner.
    /// </summary>
    /// <param name="lambda">L2 regularization of the two ridge outcome models (default: 0.01).</param>
    /// <remarks>
    /// Each outcome model is ridge regression solved exactly. It used to be approximated by a fixed number of
    /// gradient steps, which diverged on unscaled covariates, and is why the iteration count and learning rate are
    /// gone.
    /// </remarks>
    public TLearner(double lambda = 0.01) : base()
    {
        Lambda = lambda;
        _biasTreated = NumOps.Zero;
        _biasControl = NumOps.Zero;
    }

    /// <summary>
    /// Fits the T-Learner model.
    /// </summary>
    public override void Fit(Matrix<T> features, Vector<T> treatment, Vector<T> outcome)
    {
        NumFeatures = features.Columns;

        var treatmentInt = BinarizeTreatment(treatment);
        ValidateCausalData(features, treatmentInt, outcome);

        // Split data by treatment status
        var treatedIndices = GroupIndices(treatmentInt, 1);
        var controlIndices = GroupIndices(treatmentInt, 0);

        // Train treatment model
        (_weightsTreated, _biasTreated) = FitRidgeRegression(
            SelectRows(features, treatedIndices), SelectElements(outcome, treatedIndices), Lambda);

        // Train control model
        (_weightsControl, _biasControl) = FitRidgeRegression(
            SelectRows(features, controlIndices), SelectElements(outcome, controlIndices), Lambda);

        IsFitted = true;
    }

    /// <summary>
    /// Estimates the Conditional Average Treatment Effect (CATE).
    /// </summary>
    public override Vector<T> EstimateTreatmentEffect(Matrix<T> features)
    {
        EnsureFitted();
        return Engine.Subtract(PredictTreated(features), PredictControl(features));
    }

    /// <summary>
    /// Predicts outcome under treatment.
    /// </summary>
    public override Vector<T> PredictTreated(Matrix<T> features)
    {
        EnsureFitted();
        return PredictLinear(features, _weightsTreated, _biasTreated);
    }

    /// <summary>
    /// Predicts outcome under control.
    /// </summary>
    public override Vector<T> PredictControl(Matrix<T> features)
    {
        EnsureFitted();
        return PredictLinear(features, _weightsControl, _biasControl);
    }

    /// <summary>
    /// Standard prediction - returns treatment effects.
    /// </summary>
    /// <remarks>
    /// Accepts covariates alone, or the [treatment | covariates] layout <see cref="CausalModelBase{T}.Train"/>
    /// consumes, in which case the treatment column is dropped.
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (input.Columns == NumFeatures + 1)
        {
            return EstimateTreatmentEffect(DropLeadingColumn(input));
        }
        if (input.Columns != NumFeatures)
        {
            throw new ArgumentException(
                $"Input must have {NumFeatures} covariate columns, or {NumFeatures + 1} " +
                $"columns where the first is the treatment indicator. Got {input.Columns}.",
                nameof(input));
        }
        return EstimateTreatmentEffect(input);
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
        // T-Learner doesn't model propensity scores directly
        return Tensor<T>.CreateDefault(new[] { x.Rows }, NumOps.FromDouble(0.5)).ToVector();
    }

    /// <inheritdoc />
    protected override void RegisterComponents()
    {
        base.RegisterComponents();
        RegisterParameterComponent(
            "linear-models",
            new AiDotNet.Models.Parameters.VariableLengthParameterSource<T>(
                () => 2L + _weightsTreated.Length + _weightsControl.Length,
                GetLinearModelParameters,
                RestoreLinearModelParameters));
    }

    private Vector<T> GetLinearModelParameters()
    {
        if (_weightsTreated is null || _weightsControl is null)
            return new Vector<T>(2) { [0] = _biasTreated, [1] = _biasControl };

        // [biasTreated, biasControl, weightsTreated..., weightsControl...]
        return Engine.TensorConcatenate(new[]
        {
            Tensor<T>.CreateDefault(new[] { 1 }, _biasTreated),
            Tensor<T>.CreateDefault(new[] { 1 }, _biasControl),
            _weightsTreated.AsTensor(),
            _weightsControl.AsTensor(),
        }, 0).ToVector();
    }

    private void RestoreLinearModelParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        if (parameters.Length < 2 || (parameters.Length - 2) % 2 != 0)
        {
            throw new ArgumentException(
                "T-Learner parameters must contain two biases followed by equally sized " +
                "treated and control weight vectors.", nameof(parameters));
        }

        _biasTreated = parameters[0];
        _biasControl = parameters[1];

        int p = (parameters.Length - 2) / 2;
        _weightsTreated = parameters.GetSubVector(2, p);
        _weightsControl = parameters.GetSubVector(2 + p, p);
    }

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var copy = new TLearner<T>(Lambda);
        copy.SetParameters(parameters);
        return copy;
    }

    /// <inheritdoc />
    protected override System.Collections.Generic.Dictionary<string, object> GetAdditionalModelData()
    {
        var data = base.GetAdditionalModelData();
        data["BiasTreated"] = NumOps.ToDouble(_biasTreated);
        data["BiasControl"] = NumOps.ToDouble(_biasControl);
        data["WeightsTreated"] = _weightsTreated.ToArray();
        data["WeightsControl"] = _weightsControl.ToArray();
        return data;
    }

    /// <inheritdoc />
    protected override void LoadAdditionalModelData(Newtonsoft.Json.Linq.JObject modelDataObj)
    {
        base.LoadAdditionalModelData(modelDataObj);

        var biasTreated = modelDataObj["BiasTreated"];
        if (biasTreated is not null)
            _biasTreated = NumOps.FromDouble(biasTreated.ToObject<double>());

        var biasControl = modelDataObj["BiasControl"];
        if (biasControl is not null)
            _biasControl = NumOps.FromDouble(biasControl.ToObject<double>());

        if (modelDataObj["WeightsTreated"] is Newtonsoft.Json.Linq.JArray weightsTreatedArr)
        {
            _weightsTreated = new Vector<T>(weightsTreatedArr.ToObject<T[]>() ?? Array.Empty<T>());
        }
        if (modelDataObj["WeightsControl"] is Newtonsoft.Json.Linq.JArray weightsControlArr)
        {
            _weightsControl = new Vector<T>(weightsControlArr.ToObject<T[]>() ?? Array.Empty<T>());
        }

        // base.LoadAdditionalModelData restored IsFitted. Persisted state that claims to be fitted
        // but omits or truncates either weight array leaves that vector at length zero, and
        // PredictTreated / PredictControl / EstimateTreatmentEffect then index it once per feature
        // and throw from deep inside prediction. Reject it here, where the message can say what is
        // actually wrong with the file.
        if (IsFitted)
        {
            if (_weightsTreated.Length != NumFeatures || _weightsControl.Length != NumFeatures)
            {
                throw new InvalidOperationException(
                    $"Serialized {nameof(TLearner<T>)} is marked fitted but its weight vectors do not "
                    + $"match its {NumFeatures} features: WeightsTreated has {_weightsTreated.Length} "
                    + $"and WeightsControl has {_weightsControl.Length}. The model data is incomplete "
                    + $"or was written by an incompatible version.");
            }
        }
    }
}

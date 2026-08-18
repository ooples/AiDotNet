using AiDotNet.Attributes;
using AiDotNet.Data.Structures;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// sparse-MAML: meta-learn WHICH weights the inner loop is allowed to change, rather than adapting
/// all of them.
/// </summary>
/// <remarks>
/// <para>
/// von Oswald, Zhao, Kobayashi, Schug, Caccia, Zucchet and Sacramento, "Learning where to learn:
/// Gradient sparsity in meta and continual learning" (NeurIPS 2021, arXiv:2110.14402). The premise:
/// "this form of meta-learning can be improved by letting the learning algorithm decide which
/// weights to change, i.e., by learning where to learn." The reported consequence is that "patterned
/// sparsity emerges from this process, with the pattern of sparsity varying on a problem-by-problem
/// basis", giving "better generalization and less interference in a range of few-shot and continual
/// learning problems".
/// </para>
/// <code>
///   gate_d = sigmoid(phi_d)                       one learned logit per parameter
///   inner:  theta_d &lt;- theta_d - eta * gate_d * grad_d
///   outer:  theta  &lt;- meta-update as usual
///           phi    &lt;- meta-update against the QUERY loss
/// </code>
/// <para>
/// <b>Three properties carry the method, and all three are implemented rather than described.</b>
/// </para>
/// <list type="number">
/// <item><description><b>The gate is LEARNED, not computed from the gradients.</b> This class
/// previously derived a mask from per-parameter gradient-magnitude z-scores against an EMA — a fixed
/// rule, with nothing meta-learned and no way for the pattern to "vary on a problem-by-problem
/// basis". Under a rule, sparsity cannot EMERGE; it is imposed.</description></item>
/// <item><description><b>The gate is judged by the QUERY loss.</b> Which weights to adapt is chosen
/// by how well the adapted model generalizes, not by how large its support gradients were. Large
/// gradients mark where the support set pulls hardest, which is exactly what overfits it.</description></item>
/// <item><description><b>Sparsity is discovered, not imposed.</b> Every gate starts open, so any
/// sparsity in the result was selected for.</description></item>
/// </list>
/// <para>
/// <b>Estimator.</b> The gate's gradient flows through the inner-loop update, so computing it
/// exactly needs second-order autodiff through an arbitrary model, which the generic
/// <c>IFullModel</c> contract cannot provide. The logits are therefore meta-updated by SPSA, the
/// estimator this library already uses for auxiliary parameters. The OBJECTIVE is the paper's; only
/// the way its gradient is estimated differs.
/// </para>
/// <para>
/// <b>For Beginners:</b> When a model learns from only a handful of examples, changing every one of
/// its millions of weights is a good way to memorize those examples and learn nothing general. This
/// learns a switch for each weight saying whether it may change at all. The switches are tuned by
/// what actually generalizes to held-out examples, and the useful pattern — most switches off — is
/// something the training discovers rather than something imposed up front.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input type.</typeparam>
/// <typeparam name="TOutput">The output type.</typeparam>
/// <example>
/// <code>
/// var options = new SparseMAMLOptions&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;
/// {
///     MetaModel = model,
///     InitialGateLogit = 1.0,   // start with every gate open; sparsity is learned, not imposed
///     GateLearningRate = 1e-2,
/// };
/// var sparseMaml = new SparseMAMLAlgorithm&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;(options);
/// double metaLoss = Convert.ToDouble(sparseMaml.MetaTrain(taskBatch));
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Learning where to learn: Gradient sparsity in meta and continual learning",
    "https://arxiv.org/abs/2110.14402",
    Year = 2021,
    Authors = "Johannes von Oswald, Dominic Zhao, Seijin Kobayashi, Simon Schug, " +
              "Massimo Caccia, Nicolas Zucchet, Joao Sacramento")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class SparseMAMLAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private IParameterizable<T, TInput, TOutput>? _cachedParamModel;
    private IParameterizable<T, TInput, TOutput> ParamModel => _cachedParamModel ??= InterfaceGuard.Parameterizable(MetaModel);

    private readonly SparseMAMLOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;

    /// <summary>phi — one learned logit per parameter; the gate is its sigmoid.</summary>
    [AiDotNet.Attributes.Buffer]
    private Vector<T> _gateLogits;

    /// <summary>
    /// Meta-learned per-parameter learning-rate multipliers, for the paper's "more expressive model
    /// where learning rates are meta-learned". Null unless that variant is enabled.
    /// </summary>
    private Vector<T>? _perParameterRates;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.SparseMAML;

    /// <summary>Gets the gate logits phi.</summary>
    public Vector<T> GateLogits => _gateLogits;

    /// <summary>
    /// Gets the gate for parameter <paramref name="index"/>: <c>sigmoid(phi)</c>, in (0, 1).
    /// </summary>
    public double Gate(int index) => Sigmoid(NumOps.ToDouble(_gateLogits[index]));

    /// <summary>
    /// Gets the fraction of parameters currently gated OFF, i.e. the discovered sparsity.
    /// </summary>
    public double Sparsity
    {
        get
        {
            int closed = 0;
            for (int d = 0; d < _paramDim; d++) if (Gate(d) < _algoOptions.SparsityThreshold) closed++;
            return _paramDim > 0 ? (double)closed / _paramDim : 0.0;
        }
    }

    /// <summary>
    /// Initializes sparse-MAML over a model.
    /// </summary>
    /// <param name="options">Options; by default every gate starts open so sparsity must be discovered.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="options"/> is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the meta-model has no parameters.</exception>
    public SparseMAMLAlgorithm(SparseMAMLOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = InterfaceGuard.Parameterizable(options.MetaModel).GetParameters().Length;
        if (_paramDim == 0)
            throw new ArgumentException("MetaModel has zero parameters.", nameof(options));

        _gateLogits = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++)
        {
            _gateLogits[d] = NumOps.FromDouble(options.InitialGateLogit);
        }

        if (options.MetaLearnPerParameterRates)
        {
            _perParameterRates = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) _perParameterRates[d] = NumOps.One;
        }
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        // ACLAlgorithm.MetaTrain sets the contract for this base class: throw on null, and return
        // zero for an empty batch. Without the first, the foreach below reports a
        // NullReferenceException that names neither the parameter nor the caller's mistake; without
        // the second, ApplyOuterUpdate steps on no gradients and ComputeMean has no defined value.
        if (taskBatch is null) throw new ArgumentNullException(nameof(taskBatch));
        if (taskBatch.Tasks is null || taskBatch.Tasks.Length == 0) return NumOps.Zero;

        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = ParamModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            var adapted = AdaptWithGate(task, initParams);
            ParamModel.SetParameters(adapted);
            losses.Add(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        ApplyOuterUpdate(initParams, metaGradients, _algoOptions.OuterLearningRate);

        // The gate is meta-learned against the QUERY loss — how well the ADAPTED model generalizes,
        // not how large the support gradients were.
        UpdateGatesSpsa(taskBatch, ParamModel.GetParameters());

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null) throw new ArgumentNullException(nameof(task));
        var initParams = ParamModel.GetParameters();
        var adapted = AdaptWithGate(task, initParams);
        ParamModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adapted);
    }

    /// <summary>
    /// The inner loop, gated: <c>theta_d &lt;- theta_d - eta * gate_d * grad_d</c>.
    /// </summary>
    /// <remarks>
    /// The gate multiplies the UPDATE, not the parameter. Scaling the parameter would change the
    /// function the model computes; scaling the update changes only where learning is allowed to
    /// happen, which is what "learning where to learn" means.
    /// </remarks>
    private Vector<T> AdaptWithGate(IMetaLearningTask<T, TInput, TOutput> task, Vector<T> initParams)
    {
        var adapted = initParams.Clone();

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            ParamModel.SetParameters(adapted);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));

            for (int d = 0; d < _paramDim; d++)
            {
                double gate = Gate(d);
                double rate = _algoOptions.InnerLearningRate;
                if (_perParameterRates is not null) rate *= NumOps.ToDouble(_perParameterRates[d]);

                adapted[d] = NumOps.Subtract(adapted[d],
                    NumOps.FromDouble(rate * gate * NumOps.ToDouble(grad[d])));
            }
        }
        return adapted;
    }

    /// <summary>
    /// SPSA meta-update of the gate logits against the query loss of the adapted model.
    /// </summary>
    /// <remarks>
    /// Each probe must RE-RUN the gated inner loop, because the gate reaches the query loss only
    /// through that adaptation. Probing the logits while measuring a model that was adapted under
    /// the unperturbed gate would measure a quantity independent of them, and the estimate would be
    /// identically zero.
    /// </remarks>
    private void UpdateGatesSpsa(TaskBatch<T, TInput, TOutput> taskBatch, Vector<T> metaParams)
    {
        if (taskBatch.Tasks.Length == 0) return;

        const double perturbation = 1e-2;
        var delta = new double[_paramDim];
        for (int d = 0; d < _paramDim; d++) delta[d] = RandomGenerator.NextDouble() < 0.5 ? -1.0 : 1.0;

        var baseLogits = _gateLogits.Clone();

        double Probe(double sign)
        {
            var shifted = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++)
            {
                shifted[d] = NumOps.FromDouble(
                    NumOps.ToDouble(baseLogits[d]) + sign * perturbation * delta[d]);
            }
            _gateLogits = shifted;

            double total = 0.0;
            foreach (var task in taskBatch.Tasks)
            {
                var adapted = AdaptWithGate(task, metaParams);
                ParamModel.SetParameters(adapted);
                total += NumOps.ToDouble(
                    ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
            }
            return total / taskBatch.Tasks.Length;
        }

        // RESTORED ON EVERY EXIT, NOT JUST THE HAPPY ONE. Probe assigns _gateLogits and runs a full
        // gated inner loop plus a forward pass per task; anything in there that throws used to leave
        // this instance holding PERTURBED gates and an adapted ParamModel, and the corruption is
        // silent -- it simply persists into the next meta-step and biases it. The restore belongs in
        // a finally so it survives that path.
        double lossPlus;
        double lossMinus;
        try
        {
            lossPlus = Probe(+1.0);
            lossMinus = Probe(-1.0);
        }
        finally
        {
            _gateLogits = baseLogits;
            ParamModel.SetParameters(metaParams);
        }

        double scaled = (lossPlus - lossMinus) / (2.0 * perturbation);
        if (double.IsNaN(scaled) || double.IsInfinity(scaled)) return;

        var updated = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++)
        {
            updated[d] = NumOps.FromDouble(
                NumOps.ToDouble(baseLogits[d]) - _algoOptions.GateLearningRate * scaled / delta[d]);
        }
        _gateLogits = updated;
    }

    private static double Sigmoid(double x) =>
        x >= 0 ? 1.0 / (1.0 + Math.Exp(-x)) : Math.Exp(x) / (1.0 + Math.Exp(x));
}

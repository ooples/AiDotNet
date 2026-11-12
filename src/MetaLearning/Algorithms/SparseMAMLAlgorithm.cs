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
///   gate = 1[m &gt;= 0]                                   one learned logit per parameter (Eq. 4)
///   inner:  phi_{k+1} = phi_k - alpha * gate * grad L_in(phi_k),  phi_0 = theta
///   outer:  theta &lt;- theta - gamma_theta * E[grad L_out(phi_K)]
///           m     &lt;- m + alpha * gamma_m * E[grad L_out(phi_K) * sum_k grad L_in(phi_k)]   (Eq. 5)
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
/// <b>Estimator.</b> As in the paper, the step function is differentiated with the straight-through
/// estimator and second-order terms are dropped (first-order, like FOMAML), which gives Eq. 5. The
/// <see cref="SparseMAMLGateFunction.Sigmoid"/> relaxation multiplies that by sigmoid'(m). Gate logits
/// start at the constant <c>InitialGateLogit</c>: a generic parameter vector carries no fan-in, so the
/// paper's Kaiming initialization of m is not available.
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
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(inputFeatures: 4, outputSize: 2);
/// var model = new NeuralNetwork&lt;double&gt;(architecture);
/// var options = new SparseMAMLOptions&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(model)
/// {
///     InitialGateLogit = 1.0,   // start with every gate open; sparsity is learned, not imposed
///     GateLearningRate = 1e-2,
/// };
/// var sparseMaml = new SparseMAMLAlgorithm&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(options);
/// var taskBatch = new TaskBatch&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(
///     new IMetaLearningTask&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;[]
///     {
///         new MetaLearningTask&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;()
///     });
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

    private readonly SparseMAMLOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;

    /// <summary>phi — one learned logit per parameter; the gate is its sigmoid.</summary>
    [AiDotNet.Attributes.Buffer]
    private Vector<T> _gateLogits;

    /// <summary>
    /// Meta-learned per-parameter learning-rate multipliers, for the paper's "more expressive model
    /// where learning rates are meta-learned". Null unless that variant is enabled.
    /// </summary>
    /// <remarks>
    /// Empty when <c>MetaLearnPerParameterRates</c> is off: a resolved slot with no parameters. It used to be
    /// null, which the parameter registry reads as a shape not yet known, so the learner's whole parameter
    /// surface - GetParameters, Serialize, every copy - threw ParameterLayoutNotReadyException.
    /// </remarks>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _perParameterRates = new Vector<T>(0);

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.SparseMAML;

    /// <summary>Gets the gate logits m.</summary>
    public Vector<T> GateLogits => _gateLogits;

    /// <summary>
    /// Gets the gate for parameter <paramref name="index"/>: 1[m &gt;= 0] for the paper's step gate, or
    /// sigmoid(m) for the relaxation.
    /// </summary>
    public T Gate(int index) => Gates(_gateLogits)[index];

    /// <summary>
    /// Gets the fraction of parameters currently gated OFF, i.e. the discovered sparsity.
    /// </summary>
    /// <remarks>
    /// The closed gates are counted as an int and the ratio is formed in double: summing the 0/1 mask in T stalls
    /// a Half count at 2048, and a model past 65504 parameters made both count and size inf in Half (inf / inf = NaN).
    /// </remarks>
    public T Sparsity
        => NumOps.FromDouble((double)Engine.TensorCountNonzero(Engine.TensorLessThan(
                Tensor<T>.FromVector(Gates(_gateLogits)), NumOps.FromDouble(_algoOptions.SparsityThreshold)))
            / _paramDim);
    /// <summary>
    /// Initializes sparse-MAML over a model.
    /// </summary>
    /// <param name="options">Options; by default every gate starts open so sparsity must be discovered.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="options"/> is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the meta-model has no parameters.</exception>
    public SparseMAMLAlgorithm(SparseMAMLOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer,
               optimizerUse: MetaLearnerOptimizerUse.Outer)
    {
        _algoOptions = options;
        _paramDim = InterfaceGuard.Parameterizable(options.MetaModel).GetParameters().Length;
        if (_paramDim == 0)
            throw new ArgumentException("MetaModel has zero parameters.", nameof(options));

        _gateLogits = Engine.Fill(_paramDim, NumOps.FromDouble(options.InitialGateLogit));

        if (options.MetaLearnPerParameterRates)
        {
            _perParameterRates = Engine.Fill(_paramDim, NumOps.One);
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
        var logitGradient = new Vector<T>(_paramDim);
        var rateGradient = new Vector<T>(_paramDim);

        foreach (var task in taskBatch.Tasks)
        {
            var (adapted, innerGradientSum) = AdaptWithGate(task, initParams);
            ParamModel.SetParameters(adapted);
            losses.Add(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
            var outerGradient = ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput));
            metaGradients.Add(outerGradient);

            // -dL_out/d(gate * rate) / alpha = grad L_out(phi_K) * sum_k grad L_in(phi_k), first order (Eq. 5).
            var product = Engine.Multiply(outerGradient, innerGradientSum);
            logitGradient = Engine.Add(logitGradient, Engine.Multiply(product, Rates()));
            if (_perParameterRates.Length > 0)
                rateGradient = Engine.Add(rateGradient, Engine.Multiply(product, Gates(_gateLogits)));
        }

        // theta <- theta - gamma_theta E[grad L_out(phi_K)].
        ApplyOuterUpdate(initParams, metaGradients, _algoOptions.OuterLearningRate);

        // m <- m + alpha gamma_m E[...] (Eq. 5): the gate is judged by the QUERY loss of the adapted model.
        T step = NumOps.FromDouble(
            _algoOptions.InnerLearningRate * _algoOptions.GateLearningRate / taskBatch.Tasks.Length);
        _gateLogits = Engine.Add(_gateLogits, Engine.Multiply(Engine.Multiply(logitGradient, GateSlope()), step));
        if (_perParameterRates.Length > 0)
            _perParameterRates = Engine.Add(_perParameterRates, Engine.Multiply(rateGradient, step));

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null) throw new ArgumentNullException(nameof(task));
        var initParams = ParamModel.GetParameters();
        var (adapted, _) = AdaptWithGate(task, initParams);
        ParamModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adapted);
    }

    /// <summary>
    /// The gated inner loop (Eq. 4): <c>phi &lt;- phi - alpha * rate * gate * grad L_in(phi)</c>. Also returns
    /// the sum of the inner gradients, which Eq. 5 needs.
    /// </summary>
    /// <remarks>
    /// The gate multiplies the UPDATE, not the parameter. Scaling the parameter would change the
    /// function the model computes; scaling the update changes only where learning is allowed to
    /// happen, which is what "learning where to learn" means.
    /// </remarks>
    private (Vector<T> Adapted, Vector<T> InnerGradientSum) AdaptWithGate(
        IMetaLearningTask<T, TInput, TOutput> task, Vector<T> initParams)
    {
        var adapted = initParams.Clone();
        var gradientSum = new Vector<T>(_paramDim);
        var scale = Engine.Multiply(
            Engine.Multiply(Gates(_gateLogits), Rates()), NumOps.FromDouble(_algoOptions.InnerLearningRate));

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            ParamModel.SetParameters(adapted);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            gradientSum = Engine.Add(gradientSum, grad);
            adapted = Engine.Subtract(adapted, Engine.Multiply(scale, grad));
        }
        return (adapted, gradientSum);
    }

    /// <summary>The gates for <paramref name="logits"/>: 1[m &gt;= 0], or sigmoid(m) for the relaxation.</summary>
    private Vector<T> Gates(Vector<T> logits)
        => _algoOptions.GateFunction == SparseMAMLGateFunction.Sigmoid
            ? Engine.Sigmoid(logits)
            : Engine.Subtract(
                Engine.Fill(_paramDim, NumOps.One),
                Engine.TensorLessThan(Tensor<T>.FromVector(logits), NumOps.Zero).ToVector());

    /// <summary>
    /// d gate / d m: 1 under the straight-through estimator for the step gate, sigmoid(m)(1 - sigmoid(m))
    /// for the relaxation.
    /// </summary>
    private Vector<T> GateSlope()
    {
        if (_algoOptions.GateFunction != SparseMAMLGateFunction.Sigmoid)
            return Engine.Fill(_paramDim, NumOps.One);
        var g = Engine.Sigmoid(_gateLogits);
        return Engine.Multiply(g, Engine.Subtract(Engine.Fill(_paramDim, NumOps.One), g));
    }

    /// <summary>The meta-learned per-parameter rates, or ones when that variant is off.</summary>
    private Vector<T> Rates()
        => _perParameterRates.Length > 0 ? _perParameterRates : Engine.Fill(_paramDim, NumOps.One);
}

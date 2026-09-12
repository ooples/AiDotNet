using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;
using AiDotNet.Data.Structures;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of Meta-SGD (Meta Stochastic Gradient Descent) algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// Meta-SGD learns per-parameter learning rates for meta-learning. Instead of
/// learning just initialization parameters like MAML, it learns the learning
/// rate, momentum, and direction for each parameter individually, which can be
/// seen as learning a custom optimizer for each parameter.
/// </para>
/// <para>
/// <b>For Beginners:</b> Meta-SGD learns how to update each parameter individually:
/// </para>
/// <para>
/// In regular training, you use one learning rate for all weights. But different
/// parts of a neural network benefit from different learning rates. Meta-SGD
/// figures this out automatically by learning:
/// - <b>α_i:</b> The optimal learning rate for parameter i
/// - <b>β_i:</b> The optimal momentum for parameter i (optional)
/// - <b>d_i:</b> The optimal update direction/sign for parameter i (optional)
/// </para>
/// <para>
/// <b>Algorithm - Meta-SGD:</b>
/// <code>
/// # Learn per-parameter optimizers
/// for each parameter θ_i:
///     learning_rate_i = learnable_parameter
///     momentum_i = learnable_parameter (optional)
///     direction_i = learnable_parameter (optional)
///
/// # Meta-training episode
/// for each task in task_batch:
///     # Inner loop: adapt to task
///     adapted_params = initial_params.copy()
///     for step = 1 to K_inner:
///         gradients = compute_gradients(adapted_params, support_set)
///         for i in range(num_params):
///             # Per-parameter update rule
///             adapted_params[i] = update_rule_i(
///                 adapted_params[i],
///                 gradients[i],
///                 learning_rate_i,
///                 momentum_i,
///                 direction_i
///             )
///
///     # Evaluate on query set
///     query_loss = evaluate(adapted_params, query_set)
///
///     # Meta-update: optimize per-parameter coefficients
///     meta_gradients = compute_meta_gradients(query_loss)
///     update_per_parameter_optimizers(meta_gradients)
/// </code>
/// </para>
/// <para>
/// <b>Key Insights:</b>
/// 1. <b>Per-Parameter Optimization:</b> Each parameter gets its own learned
///    optimizer configuration, allowing heterogeneous learning rates across layers.
/// 2. <b>First-Order Method:</b> No Hessian computation needed, much faster than
///    second-order MAML while maintaining strong performance.
/// 3. <b>Interpretable:</b> Learned per-parameter learning rates reveal which
///    parameters are most important for quick adaptation.
/// 4. <b>Flexible Update Rules:</b> Can combine with various base optimizers
///    (SGD, Adam, RMSprop) for different adaptation characteristics.
/// </para>
/// <para>
/// <b>Reference:</b> Li, Z., Zhou, F., Chen, F., &amp; Li, H. (2017).
/// Meta-SGD: Learning to Learn Quickly for Few-Shot Learning.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Meta-SGD: Learning to Learn Quickly for Few-Shot Learning",
    "https://arxiv.org/abs/1707.09835",
    Year = 2017,
    Authors = "Zhenguo Li, Fengwei Zhou, Fei Chen, Hang Li")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class MetaSGDAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly MetaSGDOptions<T, TInput, TOutput> _metaSGDOptions;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _metaSGDOptions;
    private readonly PerParameterOptimizer<T, TInput, TOutput> _optimizer;

    /// <summary>
    /// Initializes a new instance of the MetaSGDAlgorithm class.
    /// </summary>
    /// <param name="options">Meta-SGD configuration options containing the model and all hyperparameters.</param>
    /// <exception cref="ArgumentNullException">Thrown when options is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when required components are not set in options.</exception>
    /// <exception cref="ArgumentException">Thrown when configuration validation fails.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates a Meta-SGD model that learns per-parameter optimizers:
    /// </para>
    /// <para>
    /// <b>What Meta-SGD needs:</b>
    /// - <b>MetaModel:</b> Neural network to be meta-trained (required)
    /// - <b>UpdateRuleType:</b> Type of update rule to learn (SGD, Adam, etc.)
    /// - <b>LearnLearningRate:</b> Whether to learn per-parameter learning rates (default: true)
    /// - <b>LearnMomentum:</b> Whether to learn per-parameter momentum (default: false)
    /// - <b>LearnDirection:</b> Whether to learn update direction sign (default: true)
    /// </para>
    /// <para>
    /// <b>What makes it different from MAML:</b>
    /// - MAML: Same learning rate for all parameters
    /// - Meta-SGD: Different learning rate per parameter
    /// - MAML learns the initialization; Meta-SGD learns the initialization AND the per-parameter
    ///   learning rates together (Li et al. 2017, eq. 3)
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// // Create Meta-SGD with minimal configuration
    /// var options = new MetaSGDOptions&lt;double, Tensor, Tensor&gt;(myNeuralNetwork);
    /// var metaSGD = new MetaSGDAlgorithm&lt;double, Tensor, Tensor&gt;(options);
    ///
    /// // Create Meta-SGD with full per-parameter optimization
    /// var options = new MetaSGDOptions&lt;double, Tensor, Tensor&gt;(myNeuralNetwork)
    /// {
    ///     UpdateRuleType = MetaSGDUpdateRuleType.Adam,
    ///     LearnLearningRate = true,
    ///     LearnMomentum = true,
    ///     LearnDirection = true,
    ///     LearnAdamBetas = true
    /// };
    /// var metaSGD = new MetaSGDAlgorithm&lt;double, Tensor, Tensor&gt;(options);
    /// </code>
    /// </example>
    public MetaSGDAlgorithm(MetaSGDOptions<T, TInput, TOutput> options)
        : base(
            options?.MetaModel ?? throw new ArgumentNullException(nameof(options), "MetaModel must be set in options."),
            options.LossFunction ?? options.MetaModel.DefaultLossFunction,
            options,
            options.DataLoader,
            options.MetaOptimizer,
            options.InnerOptimizer)
    {
        _metaSGDOptions = options;

        // Validate configuration
        if (!_metaSGDOptions.IsValid())
        {
            throw new ArgumentException("Meta-SGD configuration is invalid. Check all parameters.", nameof(options));
        }

        // Initialize per-parameter optimizer with learned coefficients
        var numParams = ParamModel.GetParameters().Length;
        _optimizer = new PerParameterOptimizer<T, TInput, TOutput>(numParams, _metaSGDOptions, Engine);

        // Initialize optimizer with warm-start values if enabled
        if (_metaSGDOptions.UseWarmStart)
        {
            InitializeOptimizer();
        }
    }

    /// <summary>
    /// Gets the algorithm type identifier for this meta-learner.
    /// </summary>
    /// <value>Returns <see cref="MetaLearningAlgorithmType.MetaSGD"/>.</value>
    /// <remarks>
    /// <para>
    /// This property identifies the algorithm as Meta-SGD, a first-order meta-learning
    /// algorithm that learns per-parameter learning rates, momentum terms, and update
    /// directions for fast task adaptation.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This tells the framework which meta-learning algorithm
    /// is being used. Meta-SGD is characterized by its per-parameter optimization
    /// approach, which is simpler and faster than MAML while achieving competitive
    /// performance on few-shot learning tasks.
    /// </para>
    /// </remarks>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.MetaSGD;

    /// <summary>
    /// Performs one meta-training step using Meta-SGD's per-parameter optimization approach.
    /// </summary>
    /// <param name="taskBatch">A batch of tasks to meta-train on, each containing support and query sets.</param>
    /// <returns>The average loss across all tasks in the batch (evaluated on query sets).</returns>
    /// <exception cref="ArgumentException">Thrown when the task batch is null or empty.</exception>
    /// <remarks>
    /// <para>
    /// Meta-SGD meta-training optimizes per-parameter learning coefficients:
    /// </para>
    /// <para>
    /// <b>For each task:</b>
    /// 1. Clone the meta-model with current meta-parameters
    /// 2. Perform K gradient descent steps using learned per-parameter optimizers
    /// 3. Evaluate adapted model on query set
    /// </para>
    /// <para>
    /// <b>Meta-Update:</b>
    /// 1. Compute gradients of query loss w.r.t. per-parameter coefficients
    /// 2. Update learning rates: α_i = α_i - η × ∂L_query/∂α_i
    /// 3. Update momentum (if enabled): β_i = β_i - η × ∂L_query/∂β_i
    /// 4. Update direction (if enabled): d_i = d_i - η × ∂L_query/∂d_i
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Meta-SGD learns how fast each weight should change.
    /// After seeing many tasks, it discovers that some weights need big updates
    /// (high learning rate) while others need small updates (low learning rate).
    /// This makes adaptation to new tasks much faster and more effective.
    /// </para>
    /// <para>
    /// <b>Key Difference from MAML:</b> While MAML computes how initialization
    /// affects final loss (requires second-order gradients), Meta-SGD directly
    /// learns the optimal update magnitude for each parameter (first-order only).
    /// </para>
    /// </remarks>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null || taskBatch.BatchSize == 0)
        {
            throw new ArgumentException("Task batch cannot be null or empty.", nameof(taskBatch));
        }

        // Li et al. 2017, Algorithm 1: every task adapts from the same (theta, alpha); the gradients of the
        // query losses are summed over the batch; then one step updates theta and alpha together.
        var coefficients = ReadCoefficients(_optimizer);
        T totalLoss = NumOps.Zero;
        double[]? thetaGradient = null;
        double[]? coefficientGradient = null;

        foreach (var task in taskBatch.Tasks)
        {
            var (loss, taskTheta, taskCoefficients) = Episode(task, coefficients);
            totalLoss = NumOps.Add(totalLoss, loss);
            thetaGradient = Accumulate(thetaGradient, taskTheta);
            coefficientGradient = Accumulate(coefficientGradient, taskCoefficients);
        }

        // It used to update only the per-parameter coefficients, so theta - the initialization Meta-SGD
        // learns - stayed at its starting weights however long the model meta-trained.
        var theta = ParamModel.GetParameters();
        var updated = new Vector<T>(theta.Length);
        for (int i = 0; i < theta.Length; i++)
        {
            updated[i] = NumOps.FromDouble(NumOps.ToDouble(theta[i])
                - _metaSGDOptions.OuterLearningRate * (thetaGradient?[i] ?? 0.0));
        }

        ParamModel.SetParameters(updated);
        _optimizer.UpdateMetaParameters(ToVector(coefficientGradient ?? Array.Empty<double>()));

        return NumOps.Divide(totalLoss, NumOps.FromDouble(taskBatch.BatchSize));
    }

    /// <summary>
    /// Adapts the meta-learned model to a new task using the learned per-parameter optimizers.
    /// </summary>
    /// <param name="task">The new task containing support set examples for adaptation.</param>
    /// <returns>A new model instance that has been adapted to the given task using learned optimizers.</returns>
    /// <exception cref="ArgumentNullException">Thrown when task is null.</exception>
    /// <remarks>
    /// <para>
    /// Meta-SGD adaptation uses the learned per-parameter learning rates, momentum,
    /// and directions to perform highly optimized gradient descent on the support set.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> When adapting to a new task, Meta-SGD uses the learned
    /// per-parameter optimizers to update the model. Each weight gets updated at its
    /// own optimal rate, making adaptation much faster than using a single learning
    /// rate for all weights.
    /// </para>
    /// <para>
    /// <b>Adaptation Process:</b>
    /// <code>
    /// for each adaptation step:
    ///     gradients = compute_gradients(model, support_set)
    ///     for each parameter i:
    ///         update_i = α_i × d_i × gradients[i] + β_i × velocity[i]
    ///         params[i] = params[i] - update_i
    /// </code>
    /// Where α_i, d_i, β_i are the learned per-parameter coefficients.
    /// </para>
    /// <para>
    /// <b>Advantages over MAML adaptation:</b>
    /// - Uses optimized per-parameter learning rates (not one rate for all)
    /// - Can include learned momentum for faster convergence
    /// - Direction coefficients can flip/scale gradients as needed
    /// </para>
    /// </remarks>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null)
        {
            throw new ArgumentNullException(nameof(task));
        }

        // Clone the meta model for task-specific adaptation
        var adaptedModel = CloneModel();

        // Create a clone of the per-parameter optimizer for this adaptation
        var taskOptimizer = _optimizer.Clone();

        // Perform adaptation using learned per-parameter optimizer
        AdaptWithLearnedOptimizer(task, adaptedModel, taskOptimizer);

        // Return the adapted model wrapped for the interface
        return new MetaSGDAdaptedModel<T, TInput, TOutput>(adaptedModel, taskOptimizer, _metaSGDOptions);
    }

    /// <summary>The per-parameter coefficients one episode adapts with, read once so every task shares them.</summary>
    private readonly struct Coefficients
    {
        public Coefficients(double[] rate, double[] momentum, double[] direction, double[] beta1, double[] beta2, double[] epsilon)
        {
            Rate = rate;
            Momentum = momentum;
            Direction = direction;
            Beta1 = beta1;
            Beta2 = beta2;
            Epsilon = epsilon;
        }

        public double[] Rate { get; }
        public double[] Momentum { get; }
        public double[] Direction { get; }
        public double[] Beta1 { get; }
        public double[] Beta2 { get; }
        public double[] Epsilon { get; }
    }

    /// <summary>What one inner step recorded for the reverse pass: where it started and what it saw.</summary>
    private sealed class InnerStep
    {
        public InnerStep(Vector<T> parameters, double[] rawGradient, double clipScale,
            double[] velocity, double[] firstMoment, double[] secondMoment, double[] accumulatedDelta)
        {
            Parameters = parameters;
            RawGradient = rawGradient;
            ClipScale = clipScale;
            Velocity = velocity;
            FirstMoment = firstMoment;
            SecondMoment = secondMoment;
            AccumulatedDelta = accumulatedDelta;
        }

        public Vector<T> Parameters { get; }
        public double[] RawGradient { get; }
        public double ClipScale { get; }
        public double[] Velocity { get; }
        public double[] FirstMoment { get; }
        public double[] SecondMoment { get; }
        public double[] AccumulatedDelta { get; }
    }

    private static Coefficients ReadCoefficients(PerParameterOptimizer<T, TInput, TOutput> optimizer)
        => new Coefficients(
            ToArray(optimizer.LearningRates), ToArray(optimizer.Momentums), ToArray(optimizer.Directions),
            ToArray(optimizer.AdamBeta1), ToArray(optimizer.AdamBeta2), ToArray(optimizer.AdamEpsilon));

    private static double[] ToArray(Vector<T> vector)
    {
        var values = new double[vector.Length];
        for (int i = 0; i < values.Length; i++) values[i] = NumOps.ToDouble(vector[i]);
        return values;
    }

    private static Vector<T> ToVector(double[] values)
    {
        var vector = new Vector<T>(values.Length);
        for (int i = 0; i < values.Length; i++) vector[i] = NumOps.FromDouble(values[i]);
        return vector;
    }

    private static double[] Accumulate(double[]? sum, double[] values)
    {
        if (sum is null) return (double[])values.Clone();
        for (int i = 0; i < sum.Length; i++) sum[i] += values[i];
        return sum;
    }

    /// <summary>
    /// One task's query loss and the exact gradients of that loss with respect to theta and to every learned
    /// coefficient, in the layout <see cref="PerParameterOptimizer{T, TInput, TOutput}.UpdateMetaParameters"/> reads.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Replaces a one-sided finite difference for alpha, re-running the whole inner loop once per model
    /// parameter, and "gradients" for momentum, direction and the Adam betas that were scaled copies of alpha's
    /// (0.1 x, sign x 0.01, 0.001 x). Theta was never updated at all.
    /// </para>
    /// <para>
    /// The inner loop starts from zero optimizer state for every task. It used to step the meta-level
    /// optimizer itself, whose velocities and moments carried over from task to task.
    /// </para>
    /// </remarks>
    private (T Loss, double[] ThetaGradient, double[] CoefficientGradient) Episode(
        IMetaLearningTask<T, TInput, TOutput> task, Coefficients coefficients)
    {
        var model = CloneModel();
        var (_, trace) = RunInnerLoop(
            model, ParamModel.GetParameters(), task.SupportInput, task.SupportOutput,
            _metaSGDOptions.InnerSteps, coefficients, record: true);

        T loss = ComputeLossFromOutput(model.Predict(task.QueryInput), task.QueryOutput);
        var queryGradient = ToArray(ComputeGradients(model, task.QueryInput, task.QueryOutput));

        var (theta, rate, momentum, direction, beta1, beta2, epsilon) = Backpropagate(
            model, trace ?? new List<InnerStep>(), queryGradient, task.SupportInput, task.SupportOutput, coefficients);

        return (loss, theta, PackCoefficientGradients(rate, momentum, direction, beta1, beta2, epsilon));
    }

    private double[] PackCoefficientGradients(
        double[] rate, double[] momentum, double[] direction, double[] beta1, double[] beta2, double[] epsilon)
    {
        var packed = new List<double>();
        if (_metaSGDOptions.LearnLearningRate) packed.AddRange(rate);
        if (_metaSGDOptions.LearnMomentum) packed.AddRange(momentum);
        if (_metaSGDOptions.LearnDirection) packed.AddRange(direction);
        if (_metaSGDOptions.UpdateRuleType == MetaSGDUpdateRuleType.Adam && _metaSGDOptions.LearnAdamBetas)
        {
            for (int i = 0; i < beta1.Length; i++)
            {
                packed.Add(beta1[i]);
                packed.Add(beta2[i]);
                packed.Add(epsilon[i]);
            }
        }

        return packed.ToArray();
    }

    /// <summary>
    /// Runs the per-parameter inner loop from <paramref name="start"/> with fresh optimizer state, leaving the
    /// model at the adapted parameters; with <paramref name="record"/> it keeps what the reverse pass needs.
    /// </summary>
    private (Vector<T> Adapted, List<InnerStep>? Trace) RunInnerLoop(
        IFullModel<T, TInput, TOutput> model, Vector<T> start, TInput input, TOutput target,
        int steps, Coefficients c, bool record)
    {
        var parameters = InterfaceGuard.Parameterizable(model);
        int n = start.Length;
        var theta = new Vector<T>(n);
        for (int i = 0; i < n; i++) theta[i] = start[i];

        var velocity = new double[n];
        var firstMoment = new double[n];
        var secondMoment = new double[n];
        var accumulatedDelta = new double[n];
        var trace = record ? new List<InnerStep>(steps) : null;

        for (int k = 0; k < steps; k++)
        {
            parameters.SetParameters(theta);
            var raw = ToArray(ComputeGradients(model, input, target));
            double scale = ClipScale(raw);

            trace?.Add(new InnerStep(theta, raw, scale,
                (double[])velocity.Clone(), (double[])firstMoment.Clone(),
                (double[])secondMoment.Clone(), (double[])accumulatedDelta.Clone()));

            var next = new Vector<T>(n);
            for (int i = 0; i < n; i++)
            {
                double u = StepElement(
                    _metaSGDOptions.UpdateRuleType, k + 1, _metaSGDOptions.LearnDirection, raw[i] * scale,
                    ref velocity[i], ref firstMoment[i], ref secondMoment[i], ref accumulatedDelta[i],
                    c.Rate[i], c.Momentum[i], c.Direction[i], c.Beta1[i], c.Beta2[i], c.Epsilon[i]);
                next[i] = NumOps.FromDouble(NumOps.ToDouble(theta[i]) - u);
            }

            theta = next;
        }

        parameters.SetParameters(theta);
        return (theta, trace);
    }

    /// <summary>Global-norm clipping of the inner gradient (the factor multiplying it; 1 when not clipped).</summary>
    private double ClipScale(double[] gradient)
    {
        if (!_metaSGDOptions.GradientClipThreshold.HasValue) return 1.0;
        double norm = Math.Sqrt(gradient.Sum(g => g * g));
        double threshold = _metaSGDOptions.GradientClipThreshold.Value;
        return norm > threshold && norm > 0 ? threshold / norm : 1.0;
    }

    /// <summary>One element of one inner step: returns the update u (theta' = theta - u) and advances the state.</summary>
    /// <remarks>
    /// Adam carries its bias correction (Kingma and Ba 2015); the shared optimizer used to skip it "assuming
    /// many iterations", which is exactly wrong for a few-step inner loop.
    /// </remarks>
    private static double StepElement(
        MetaSGDUpdateRuleType rule, int t, bool useDirection, double g,
        ref double velocity, ref double firstMoment, ref double secondMoment, ref double accumulatedDelta,
        double rate, double momentum, double direction, double beta1, double beta2, double epsilon)
    {
        double scale = useDirection ? rate * direction : rate;
        switch (rule)
        {
            case MetaSGDUpdateRuleType.SGD:
                return scale * g;
            case MetaSGDUpdateRuleType.SGDWithMomentum:
                velocity = momentum * velocity + scale * g;
                return velocity;
            case MetaSGDUpdateRuleType.Adam:
            {
                firstMoment = beta1 * firstMoment + (1 - beta1) * g;
                secondMoment = beta2 * secondMoment + (1 - beta2) * g * g;
                double mHat = firstMoment / (1 - Math.Pow(beta1, t));
                double sHat = secondMoment / (1 - Math.Pow(beta2, t));
                return rate * mHat / (Math.Sqrt(Math.Max(0, sHat)) + epsilon);
            }
            case MetaSGDUpdateRuleType.RMSprop:
                secondMoment = RmsDecay * secondMoment + (1 - RmsDecay) * g * g;
                return rate * g / (Math.Sqrt(Math.Max(0, secondMoment)) + RmsEpsilon);
            case MetaSGDUpdateRuleType.AdaGrad:
                secondMoment += g * g;
                return rate * g / (Math.Sqrt(Math.Max(0, secondMoment)) + RmsEpsilon);
            case MetaSGDUpdateRuleType.AdaDelta:
            {
                // Zeiler 2012, Algorithm 1: RMS[x] = sqrt(E[x^2] + eps).
                secondMoment = AdaDeltaRho * secondMoment + (1 - AdaDeltaRho) * g * g;
                double rmsGradient = Math.Sqrt(secondMoment + AdaDeltaEpsilon);
                double rmsDelta = Math.Sqrt(accumulatedDelta + AdaDeltaEpsilon);
                double u = rmsDelta / rmsGradient * g;
                accumulatedDelta = AdaDeltaRho * accumulatedDelta + (1 - AdaDeltaRho) * u * u;
                return u;
            }
            default:
                return rate * g;
        }
    }

    private const double RmsDecay = 0.9;
    private const double RmsEpsilon = 1e-6;
    private const double AdaDeltaRho = 0.95;
    private const double AdaDeltaEpsilon = 1e-6;

    /// <summary>
    /// Reverse pass through a recorded inner loop: the exact gradient of the query loss with respect to the
    /// starting parameters theta and to every per-parameter coefficient.
    /// </summary>
    /// <remarks>
    /// For each step, theta_{k+1} = theta_k - u(g_k, state_k; coefficients) with g_k the support gradient at
    /// theta_k. Walking back, lambda_theta_k = lambda_theta_{k+1} + H_k lambda_g, where lambda_g is the update's
    /// sensitivity to g_k and H_k the support loss's Hessian at theta_k (a Hessian-vector product). First-order
    /// mode drops the H_k term.
    /// </remarks>
    private (double[] Theta, double[] Rate, double[] Momentum, double[] Direction, double[] Beta1, double[] Beta2, double[] Epsilon)
        Backpropagate(IFullModel<T, TInput, TOutput> model, List<InnerStep> trace, double[] queryGradient,
            TInput supportInput, TOutput supportTarget, Coefficients c)
    {
        int n = queryGradient.Length;
        var lambdaTheta = (double[])queryGradient.Clone();
        var lambdaVelocity = new double[n];
        var lambdaFirst = new double[n];
        var lambdaSecond = new double[n];
        var lambdaDelta = new double[n];
        var gRate = new double[n];
        var gMomentum = new double[n];
        var gDirection = new double[n];
        var gBeta1 = new double[n];
        var gBeta2 = new double[n];
        var gEpsilon = new double[n];

        for (int k = trace.Count - 1; k >= 0; k--)
        {
            var step = trace[k];
            var lambdaG = new double[n];
            for (int i = 0; i < n; i++)
            {
                StepElementBackward(
                    _metaSGDOptions.UpdateRuleType, k + 1, _metaSGDOptions.LearnDirection,
                    step.RawGradient[i] * step.ClipScale,
                    step.Velocity[i], step.FirstMoment[i], step.SecondMoment[i], step.AccumulatedDelta[i],
                    c.Rate[i], c.Momentum[i], c.Direction[i], c.Beta1[i], c.Beta2[i], c.Epsilon[i],
                    -lambdaTheta[i],
                    ref lambdaVelocity[i], ref lambdaFirst[i], ref lambdaSecond[i], ref lambdaDelta[i],
                    out lambdaG[i],
                    ref gRate[i], ref gMomentum[i], ref gDirection[i], ref gBeta1[i], ref gBeta2[i], ref gEpsilon[i]);
            }

            // Through the clip: g_used = s(g) g with s = threshold / |g| when clipped.
            if (step.ClipScale < 1.0)
            {
                double normSquared = step.RawGradient.Sum(g => g * g);
                double projection = 0;
                for (int i = 0; i < n; i++) projection += step.RawGradient[i] * lambdaG[i];
                for (int i = 0; i < n; i++)
                    lambdaG[i] = step.ClipScale * (lambdaG[i] - step.RawGradient[i] * projection / normSquared);
            }

            if (!_metaSGDOptions.UseFirstOrder)
            {
                var hv = HessianVectorProduct(model, step.Parameters, lambdaG, supportInput, supportTarget);
                for (int i = 0; i < n; i++) lambdaTheta[i] += hv[i];
            }
        }

        return (lambdaTheta, gRate, gMomentum, gDirection, gBeta1, gBeta2, gEpsilon);
    }

    /// <summary>
    /// The derivative of one element's update, run backwards. <paramref name="lambdaU"/> is the loss's
    /// sensitivity to the update u; the state adjoints come in for the state after the step and leave for the
    /// state before it.
    /// </summary>
    private static void StepElementBackward(
        MetaSGDUpdateRuleType rule, int t, bool useDirection, double g,
        double velocity, double firstMoment, double secondMoment, double accumulatedDelta,
        double rate, double momentum, double direction, double beta1, double beta2, double epsilon,
        double lambdaU,
        ref double lambdaVelocity, ref double lambdaFirst, ref double lambdaSecond, ref double lambdaDelta,
        out double lambdaG,
        ref double gRate, ref double gMomentum, ref double gDirection, ref double gBeta1, ref double gBeta2, ref double gEpsilon)
    {
        double dirFactor = useDirection ? direction : 1.0;
        double scale = rate * dirFactor;
        switch (rule)
        {
            case MetaSGDUpdateRuleType.SGD:
                lambdaG = lambdaU * scale;
                gRate += lambdaU * dirFactor * g;
                if (useDirection) gDirection += lambdaU * rate * g;
                return;

            case MetaSGDUpdateRuleType.SGDWithMomentum:
            {
                // v' = mu v + scale g, u = v'
                double lv = lambdaVelocity + lambdaU;
                gMomentum += lv * velocity;
                gRate += lv * dirFactor * g;
                if (useDirection) gDirection += lv * rate * g;
                lambdaG = lv * scale;
                lambdaVelocity = lv * momentum;
                return;
            }

            case MetaSGDUpdateRuleType.Adam:
            {
                double mNew = beta1 * firstMoment + (1 - beta1) * g;
                double sNew = beta2 * secondMoment + (1 - beta2) * g * g;
                double c1 = 1 - Math.Pow(beta1, t);
                double c2 = 1 - Math.Pow(beta2, t);
                double mHat = mNew / c1;
                double sHat = sNew / c2;
                double r = Math.Sqrt(Math.Max(0, sHat));
                double den = r + epsilon;

                gRate += lambdaU * mHat / den;
                gEpsilon += lambdaU * (-rate * mHat / (den * den));
                double lmHat = lambdaU * rate / den;
                double lsHat = r > 0 ? lambdaU * (-rate * mHat / (den * den)) * 0.5 / r : 0.0;

                double lmNew = lambdaFirst + lmHat / c1;
                gBeta1 += lmHat * mNew * t * Math.Pow(beta1, t - 1) / (c1 * c1);
                double lsNew = lambdaSecond + lsHat / c2;
                gBeta2 += lsHat * sNew * t * Math.Pow(beta2, t - 1) / (c2 * c2);

                gBeta1 += lmNew * (firstMoment - g);
                gBeta2 += lsNew * (secondMoment - g * g);
                lambdaFirst = lmNew * beta1;
                lambdaSecond = lsNew * beta2;
                lambdaG = lmNew * (1 - beta1) + lsNew * (1 - beta2) * 2 * g;
                return;
            }

            case MetaSGDUpdateRuleType.RMSprop:
            case MetaSGDUpdateRuleType.AdaGrad:
            {
                bool rms = rule == MetaSGDUpdateRuleType.RMSprop;
                double decay = rms ? RmsDecay : 1.0;
                double gain = rms ? 1 - RmsDecay : 1.0;
                double sNew = decay * secondMoment + gain * g * g;
                double r = Math.Sqrt(Math.Max(0, sNew));
                double den = r + RmsEpsilon;

                gRate += lambdaU * g / den;
                lambdaG = lambdaU * rate / den;
                double lDen = lambdaU * (-rate * g / (den * den));
                double lsNew = lambdaSecond + (r > 0 ? lDen * 0.5 / r : 0.0);
                lambdaSecond = lsNew * decay;
                lambdaG += lsNew * gain * 2 * g;
                return;
            }

            case MetaSGDUpdateRuleType.AdaDelta:
            {
                double sNew = AdaDeltaRho * secondMoment + (1 - AdaDeltaRho) * g * g;
                double rmsGradient = Math.Sqrt(sNew + AdaDeltaEpsilon);
                double rmsDelta = Math.Sqrt(accumulatedDelta + AdaDeltaEpsilon);
                double u = rmsDelta / rmsGradient * g;

                // The accumulated update after this step depends on u, so its adjoint flows into u first.
                double lu = lambdaU + lambdaDelta * (1 - AdaDeltaRho) * 2 * u;
                lambdaG = lu * rmsDelta / rmsGradient;
                double lRmsDelta = lu * g / rmsGradient;
                double lRmsGradient = lu * (-rmsDelta * g / (rmsGradient * rmsGradient));
                double lsNew = lambdaSecond + lRmsGradient * 0.5 / rmsGradient;
                lambdaSecond = lsNew * AdaDeltaRho;
                lambdaG += lsNew * (1 - AdaDeltaRho) * 2 * g;
                lambdaDelta = lambdaDelta * AdaDeltaRho + lRmsDelta * 0.5 / rmsDelta;
                return;
            }

            default:
                lambdaG = lambdaU * rate;
                gRate += lambdaU * g;
                return;
        }
    }

    /// <summary>
    /// H v for the support loss at <paramref name="at"/>: the central difference of two gradients along v.
    /// </summary>
    /// <remarks>
    /// Exact for a loss quadratic in the parameters (a linear model under squared error) and second-order
    /// accurate otherwise. It works for any model that can report its gradient.
    /// </remarks>
    private double[] HessianVectorProduct(
        IFullModel<T, TInput, TOutput> model, Vector<T> at, double[] v, TInput input, TOutput target)
    {
        int n = v.Length;
        double vNorm = Math.Sqrt(v.Sum(x => x * x));
        if (vNorm == 0) return new double[n];

        double atNorm = 0;
        for (int i = 0; i < n; i++) atNorm += NumOps.ToDouble(at[i]) * NumOps.ToDouble(at[i]);
        double h = 1e-5 * (1.0 + Math.Sqrt(atNorm)) / vNorm;

        var parameters = InterfaceGuard.Parameterizable(model);
        var plus = new Vector<T>(n);
        var minus = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            double value = NumOps.ToDouble(at[i]);
            plus[i] = NumOps.FromDouble(value + h * v[i]);
            minus[i] = NumOps.FromDouble(value - h * v[i]);
        }

        parameters.SetParameters(plus);
        var gradientPlus = ToArray(ComputeGradients(model, input, target));
        parameters.SetParameters(minus);
        var gradientMinus = ToArray(ComputeGradients(model, input, target));

        var hv = new double[n];
        for (int i = 0; i < n; i++) hv[i] = (gradientPlus[i] - gradientMinus[i]) / (2 * h);
        return hv;
    }

    /// <summary>The per-parameter optimizer holding the learned coefficients (for tests and inspection).</summary>
    internal PerParameterOptimizer<T, TInput, TOutput> LearnedOptimizer => _optimizer;

    /// <summary>One task's query loss and its exact gradients, for gradient checks.</summary>
    internal (T Loss, double[] ThetaGradient, double[] CoefficientGradient) EpisodeGradientForTesting(
        IMetaLearningTask<T, TInput, TOutput> task)
        => Episode(task, ReadCoefficients(_optimizer));

    /// <summary>One task's query loss after the inner loop from the current theta and coefficients.</summary>
    internal T EpisodeLossForTesting(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var model = CloneModel();
        RunInnerLoop(model, ParamModel.GetParameters(), task.SupportInput, task.SupportOutput,
            _metaSGDOptions.InnerSteps, ReadCoefficients(_optimizer), record: false);
        return ComputeLossFromOutput(model.Predict(task.QueryInput), task.QueryOutput);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// The learned coefficients are Meta-SGD's learned state alongside theta, which the parameter vector
    /// carries. They were marked [Scratch] inside the optimizer and declared nowhere, so a saved or cloned
    /// Meta-SGD came back with its initial learning rates.
    /// </remarks>
    protected override void RegisterState(AiDotNet.Models.ModelStateRegistry<T> state)
    {
        base.RegisterState(state);
        state.Declare("MetaSGD.LearningRates", () => _optimizer.LearningRates, v => { if (v is not null) _optimizer.LearningRates = v; });
        state.Declare("MetaSGD.Momentums", () => _optimizer.Momentums, v => { if (v is not null) _optimizer.Momentums = v; });
        state.Declare("MetaSGD.Directions", () => _optimizer.Directions, v => { if (v is not null) _optimizer.Directions = v; });
        state.Declare("MetaSGD.AdamBeta1", () => _optimizer.AdamBeta1, v => { if (v is not null) _optimizer.AdamBeta1 = v; });
        state.Declare("MetaSGD.AdamBeta2", () => _optimizer.AdamBeta2, v => { if (v is not null) _optimizer.AdamBeta2 = v; });
        state.Declare("MetaSGD.AdamEpsilon", () => _optimizer.AdamEpsilon, v => { if (v is not null) _optimizer.AdamEpsilon = v; });
    }

    /// <summary>
    /// Performs adaptation using the learned per-parameter optimizer.
    /// </summary>
    /// <param name="task">The task to adapt to.</param>
    /// <param name="model">The model to adapt.</param>
    /// <param name="optimizer">The per-parameter optimizer to use.</param>
    /// <remarks>
    /// <para>
    /// Uses the learned per-parameter learning rates, momentum, and directions
    /// to perform optimized gradient descent on the task's support set.
    /// </para>
    /// </remarks>
    private void AdaptWithLearnedOptimizer(
        IMetaLearningTask<T, TInput, TOutput> task,
        IFullModel<T, TInput, TOutput> model,
        PerParameterOptimizer<T, TInput, TOutput> optimizer)
    {
        // The same inner loop meta-training differentiates, from fresh optimizer state: adapting to a task must
        // not depend on what an earlier task left in the optimizer's velocities or moments.
        RunInnerLoop(model, InterfaceGuard.Parameterizable(model).GetParameters(), task.SupportInput, task.SupportOutput,
            _metaSGDOptions.AdaptationSteps, ReadCoefficients(optimizer), record: false);
    }

    /// <summary>
    /// Initializes the per-parameter optimizer with warm-start values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Sets reasonable initial values for all learned per-parameter coefficients
    /// based on the configuration. This helps with training stability and faster
    /// convergence.
    /// </para>
    /// </remarks>
    private void InitializeOptimizer()
    {
        int numParams = ParamModel.GetParameters().Length;

        for (int i = 0; i < numParams; i++)
        {
            // Initialize learning rate based on initialization strategy
            T initialLR = GetInitialLearningRate(i, numParams);
            _optimizer.SetLearningRate(i, initialLR);

            // Initialize momentum if enabled
            if (_metaSGDOptions.LearnMomentum)
            {
                _optimizer.SetMomentum(i, NumOps.FromDouble(0.9));
            }

            // Initialize direction if enabled
            if (_metaSGDOptions.LearnDirection)
            {
                _optimizer.SetDirection(i, NumOps.One);
            }

            // Initialize Adam parameters if using Adam
            if (_metaSGDOptions.UpdateRuleType == MetaSGDUpdateRuleType.Adam)
            {
                _optimizer.SetAdamBeta1(i, NumOps.FromDouble(_metaSGDOptions.AdamBeta1Init));
                _optimizer.SetAdamBeta2(i, NumOps.FromDouble(_metaSGDOptions.AdamBeta2Init));
                _optimizer.SetAdamEpsilon(i, NumOps.FromDouble(_metaSGDOptions.AdamEpsilonInit));
            }
        }
    }

    /// <summary>
    /// Gets the initial learning rate for a parameter based on the initialization strategy.
    /// </summary>
    /// <param name="paramIndex">Index of the parameter.</param>
    /// <param name="totalParams">Total number of parameters.</param>
    /// <returns>The initial learning rate for this parameter.</returns>
    private T GetInitialLearningRate(int paramIndex, int totalParams)
    {
        double baseLR = _metaSGDOptions.InnerLearningRate;

        switch (_metaSGDOptions.LearningRateInitialization)
        {
            case MetaSGDLearningRateInitialization.Uniform:
                return NumOps.FromDouble(baseLR);

            case MetaSGDLearningRateInitialization.Random:
                double range = _metaSGDOptions.LearningRateInitRange;
                double randomLR = baseLR + (RandomGenerator.NextDouble() - 0.5) * range;
                return NumOps.FromDouble(Math.Max(_metaSGDOptions.MinLearningRate ?? double.NegativeInfinity,
                    Math.Min(_metaSGDOptions.MaxLearningRate ?? double.PositiveInfinity, randomLR)));

            case MetaSGDLearningRateInitialization.LayerBased:
                if (_metaSGDOptions.UseLayerWiseDecay)
                {
                    // Approximate layer from parameter index
                    double layerFraction = (double)paramIndex / totalParams;
                    double decay = Math.Pow(_metaSGDOptions.LayerDecayFactor, layerFraction * 10);
                    return NumOps.FromDouble(baseLR * decay);
                }
                return NumOps.FromDouble(baseLR);

            case MetaSGDLearningRateInitialization.Xavier:
                // Xavier-inspired: scale by 1/sqrt(n)
                double xavierScale = 1.0 / Math.Sqrt(totalParams);
                return NumOps.FromDouble(baseLR * xavierScale);

            case MetaSGDLearningRateInitialization.MagnitudeBased:
                // Would need parameter values - use uniform for simplicity
                return NumOps.FromDouble(baseLR);

            default:
                return NumOps.FromDouble(baseLR);
        }
    }
}

/// <summary>
/// Per-parameter optimizer for Meta-SGD that learns individual optimization coefficients.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// This optimizer maintains learned coefficients for each parameter:
/// - Learning rates: α_i for each parameter
/// - Momentum: β_i for each parameter (optional)
/// - Direction: d_i for each parameter (optional)
/// - Adam parameters: beta1, beta2, epsilon (if using Adam)
/// </para>
/// <para>
/// <b>For Beginners:</b> This is a special optimizer where each weight in the
/// network gets its own set of optimization settings that are learned during
/// meta-training.
/// </para>
/// </remarks>
public class PerParameterOptimizer<T, TInput, TOutput>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly int _numParameters;
    private readonly MetaSGDOptions<T, TInput, TOutput> _options;
    private readonly IEngine _engine;

    // Per-parameter learned coefficients (Vector<T> for Engine vectorization). These are what Meta-SGD
    // learns; they were marked [Scratch], which is state nothing persists.
    private Vector<T> _learningRates;
    private Vector<T> _momentums;
    private Vector<T> _directions;

    // Adam-specific parameters
    private Vector<T> _adamBeta1;
    private Vector<T> _adamBeta2;
    private Vector<T> _adamEpsilon;

    // Optimizer state
    private Vector<T> _firstMoments;
    private Vector<T> _secondMoments;
    private Vector<T> _velocities;
    private int[] _adamSteps;

    /// <summary>Gets or sets the learned per-parameter learning rates (alpha).</summary>
    internal Vector<T> LearningRates { get => _learningRates; set => _learningRates = Checked(value); }

    /// <summary>Gets or sets the learned per-parameter momentum coefficients.</summary>
    internal Vector<T> Momentums { get => _momentums; set => _momentums = Checked(value); }

    /// <summary>Gets or sets the learned per-parameter direction coefficients.</summary>
    internal Vector<T> Directions { get => _directions; set => _directions = Checked(value); }

    /// <summary>Gets or sets the learned per-parameter Adam beta1.</summary>
    internal Vector<T> AdamBeta1 { get => _adamBeta1; set => _adamBeta1 = Checked(value); }

    /// <summary>Gets or sets the learned per-parameter Adam beta2.</summary>
    internal Vector<T> AdamBeta2 { get => _adamBeta2; set => _adamBeta2 = Checked(value); }

    /// <summary>Gets or sets the learned per-parameter Adam epsilon.</summary>
    internal Vector<T> AdamEpsilon { get => _adamEpsilon; set => _adamEpsilon = Checked(value); }

    private Vector<T> Checked(Vector<T> value)
    {
        if (value is null) throw new ArgumentNullException(nameof(value));
        if (value.Length != _numParameters)
            throw new ArgumentException($"Expected {_numParameters} coefficients, got {value.Length}.", nameof(value));
        var copy = new Vector<T>(_numParameters);
        for (int i = 0; i < _numParameters; i++) copy[i] = value[i];
        return copy;
    }

    /// <summary>
    /// Gets the number of model parameters this optimizer manages.
    /// </summary>
    public int NumParameters => _numParameters;

    /// <summary>
    /// Initializes a new instance of the PerParameterOptimizer.
    /// </summary>
    /// <param name="numParameters">Number of model parameters.</param>
    /// <param name="options">Meta-SGD options.</param>
    /// <param name="engine">Engine instance for vectorized operations.</param>
    public PerParameterOptimizer(int numParameters, MetaSGDOptions<T, TInput, TOutput> options, IEngine engine)
    {
        _numParameters = numParameters;
        _options = options;
        _engine = engine;

        // Initialize per-parameter vectors with default values
        _learningRates = Vector<T>.CreateDefault(numParameters, NumOps.FromDouble(options.InnerLearningRate));
        _momentums = new Vector<T>(numParameters);
        _directions = Vector<T>.CreateDefault(numParameters, NumOps.One);
        _adamBeta1 = Vector<T>.CreateDefault(numParameters, NumOps.FromDouble(options.AdamBeta1Init));
        _adamBeta2 = Vector<T>.CreateDefault(numParameters, NumOps.FromDouble(options.AdamBeta2Init));
        _adamEpsilon = Vector<T>.CreateDefault(numParameters, NumOps.FromDouble(options.AdamEpsilonInit));
        _firstMoments = new Vector<T>(numParameters);
        _secondMoments = new Vector<T>(numParameters);
        _velocities = new Vector<T>(numParameters);
        _adamSteps = new int[numParameters];
    }

    /// <summary>
    /// Updates a single parameter using its learned optimization coefficients.
    /// </summary>
    /// <param name="parameterIndex">Index of the parameter to update.</param>
    /// <param name="parameter">Current parameter value.</param>
    /// <param name="gradient">Gradient for this parameter.</param>
    /// <returns>Updated parameter value.</returns>
    /// <remarks>
    /// <para>
    /// Applies the learned per-parameter update rule:
    /// - SGD: θ = θ - α_i × d_i × g
    /// - SGDWithMomentum: v = β_i × v + α_i × d_i × g; θ = θ - v
    /// - Adam: Full Adam with learned β1, β2, ε
    /// </para>
    /// </remarks>
    public T UpdateParameter(int parameterIndex, T parameter, T gradient)
    {
        var lr = _learningRates[parameterIndex];
        T update;

        switch (_options.UpdateRuleType)
        {
            case MetaSGDUpdateRuleType.SGD:
                if (_options.LearnDirection)
                {
                    update = NumOps.Multiply(lr, NumOps.Multiply(_directions[parameterIndex], gradient));
                }
                else
                {
                    update = NumOps.Multiply(lr, gradient);
                }
                break;

            case MetaSGDUpdateRuleType.SGDWithMomentum:
                T gradUpdate;
                if (_options.LearnDirection)
                {
                    gradUpdate = NumOps.Multiply(lr, NumOps.Multiply(_directions[parameterIndex], gradient));
                }
                else
                {
                    gradUpdate = NumOps.Multiply(lr, gradient);
                }
                update = NumOps.Add(
                    NumOps.Multiply(_momentums[parameterIndex], _velocities[parameterIndex]),
                    gradUpdate);
                _velocities[parameterIndex] = update;
                break;

            case MetaSGDUpdateRuleType.Adam:
                // Adam update with learned parameters
                T oneMinusBeta1 = NumOps.Subtract(NumOps.One, _adamBeta1[parameterIndex]);
                T oneMinusBeta2 = NumOps.Subtract(NumOps.One, _adamBeta2[parameterIndex]);

                // Update biased first moment
                _firstMoments[parameterIndex] = NumOps.Add(
                    NumOps.Multiply(_adamBeta1[parameterIndex], _firstMoments[parameterIndex]),
                    NumOps.Multiply(oneMinusBeta1, gradient));

                // Update biased second moment
                T gradSquared = NumOps.Multiply(gradient, gradient);
                _secondMoments[parameterIndex] = NumOps.Add(
                    NumOps.Multiply(_adamBeta2[parameterIndex], _secondMoments[parameterIndex]),
                    NumOps.Multiply(oneMinusBeta2, gradSquared));

                // Bias correction (Kingma and Ba 2015): a few-step inner loop is exactly where it matters.
                int adamStep = ++_adamSteps[parameterIndex];
                T biasCorrectedFirst = NumOps.Divide(_firstMoments[parameterIndex],
                    NumOps.FromDouble(1 - Math.Pow(NumOps.ToDouble(_adamBeta1[parameterIndex]), adamStep)));
                T biasCorrectedSecond = NumOps.Divide(_secondMoments[parameterIndex],
                    NumOps.FromDouble(1 - Math.Pow(NumOps.ToDouble(_adamBeta2[parameterIndex]), adamStep)));

                // Compute update
                double secondMomentVal = NumOps.ToDouble(biasCorrectedSecond);
                T sqrtSecond = NumOps.FromDouble(Math.Sqrt(Math.Max(0, secondMomentVal)));
                T denominator = NumOps.Add(sqrtSecond, _adamEpsilon[parameterIndex]);
                update = NumOps.Divide(NumOps.Multiply(lr, biasCorrectedFirst), denominator);
                break;

            case MetaSGDUpdateRuleType.RMSprop:
                // RMSprop update
                T decayRate = NumOps.FromDouble(0.9);
                T oneMinusDecay = NumOps.FromDouble(0.1);
                _secondMoments[parameterIndex] = NumOps.Add(
                    NumOps.Multiply(decayRate, _secondMoments[parameterIndex]),
                    NumOps.Multiply(oneMinusDecay, NumOps.Multiply(gradient, gradient)));

                double secondVal = NumOps.ToDouble(_secondMoments[parameterIndex]);
                T sqrtSecondRms = NumOps.FromDouble(Math.Sqrt(Math.Max(0, secondVal)));
                T denomRms = NumOps.Add(sqrtSecondRms, NumOps.FromDouble(1e-6));
                update = NumOps.Divide(NumOps.Multiply(lr, gradient), denomRms);
                break;

            case MetaSGDUpdateRuleType.AdaGrad:
                // AdaGrad update
                _secondMoments[parameterIndex] = NumOps.Add(
                    _secondMoments[parameterIndex],
                    NumOps.Multiply(gradient, gradient));

                double accumVal = NumOps.ToDouble(_secondMoments[parameterIndex]);
                T sqrtAccum = NumOps.FromDouble(Math.Sqrt(Math.Max(0, accumVal)));
                T denomAda = NumOps.Add(sqrtAccum, NumOps.FromDouble(1e-6));
                update = NumOps.Divide(NumOps.Multiply(lr, gradient), denomAda);
                break;

            case MetaSGDUpdateRuleType.AdaDelta:
                // AdaDelta update (simplified)
                T rho = NumOps.FromDouble(0.95);
                T oneMinusRho = NumOps.FromDouble(0.05);

                _secondMoments[parameterIndex] = NumOps.Add(
                    NumOps.Multiply(rho, _secondMoments[parameterIndex]),
                    NumOps.Multiply(oneMinusRho, NumOps.Multiply(gradient, gradient)));

                // Zeiler 2012, Algorithm 1: RMS[x] = sqrt(E[x^2] + eps), eps = 1e-6.
                double accGradVal = NumOps.ToDouble(_secondMoments[parameterIndex]);
                T rmsGrad = NumOps.FromDouble(Math.Sqrt(accGradVal + 1e-6));

                // Use velocity as accumulated parameter updates
                double accDeltaVal = NumOps.ToDouble(_velocities[parameterIndex]);
                T rmsDelta = NumOps.FromDouble(Math.Sqrt(accDeltaVal + 1e-6));

                update = NumOps.Multiply(NumOps.Divide(rmsDelta, rmsGrad), gradient);

                // Update accumulated parameter updates
                _velocities[parameterIndex] = NumOps.Add(
                    NumOps.Multiply(rho, _velocities[parameterIndex]),
                    NumOps.Multiply(oneMinusRho, NumOps.Multiply(update, update)));
                break;

            default:
                update = NumOps.Multiply(lr, gradient);
                break;
        }

        return NumOps.Subtract(parameter, update);
    }

    /// <summary>
    /// Updates the meta-parameters (learned coefficients) of the optimizer.
    /// </summary>
    /// <param name="metaGradients">Gradients for the meta-parameters.</param>
    /// <remarks>
    /// <para>
    /// Updates learning rates, momentum, direction, and Adam parameters based on
    /// the computed meta-gradients. Also applies regularization and clipping.
    /// </para>
    /// </remarks>
    public void UpdateMetaParameters(Vector<T> metaGradients)
    {
        int index = 0;
        double metaLR = _options.OuterLearningRate;

        // Update learning rates
        if (_options.LearnLearningRate)
        {
            for (int i = 0; i < _numParameters; i++)
            {
                T update = NumOps.Multiply(metaGradients[index++], NumOps.FromDouble(metaLR));
                _learningRates[i] = NumOps.Subtract(_learningRates[i], update);
            }
        }

        // Update momentums
        if (_options.LearnMomentum)
        {
            for (int i = 0; i < _numParameters; i++)
            {
                T update = NumOps.Multiply(metaGradients[index++], NumOps.FromDouble(metaLR));
                _momentums[i] = NumOps.Subtract(_momentums[i], update);
            }
        }

        // Update directions
        if (_options.LearnDirection)
        {
            for (int i = 0; i < _numParameters; i++)
            {
                T update = NumOps.Multiply(metaGradients[index++], NumOps.FromDouble(metaLR));
                _directions[i] = NumOps.Subtract(_directions[i], update);
            }
        }

        // Update Adam parameters
        if (_options.UpdateRuleType == MetaSGDUpdateRuleType.Adam && _options.LearnAdamBetas)
        {
            for (int i = 0; i < _numParameters; i++)
            {
                T updateBeta1 = NumOps.Multiply(metaGradients[index++], NumOps.FromDouble(metaLR));
                T updateBeta2 = NumOps.Multiply(metaGradients[index++], NumOps.FromDouble(metaLR));
                T updateEps = NumOps.Multiply(metaGradients[index++], NumOps.FromDouble(metaLR));

                _adamBeta1[i] = NumOps.Subtract(_adamBeta1[i], updateBeta1);
                _adamBeta2[i] = NumOps.Subtract(_adamBeta2[i], updateBeta2);
                _adamEpsilon[i] = NumOps.Subtract(_adamEpsilon[i], updateEps);
            }
        }

        // Apply regularization and clipping
        ApplyRegularization();
    }

    /// <summary>
    /// Gets the total number of meta-parameters being learned.
    /// </summary>
    /// <returns>Count of learned meta-parameters.</returns>
    public int GetMetaParameterCount()
    {
        int count = 0;

        if (_options.LearnLearningRate)
            count += _numParameters;

        if (_options.LearnMomentum)
            count += _numParameters;

        if (_options.LearnDirection)
            count += _numParameters;

        if (_options.UpdateRuleType == MetaSGDUpdateRuleType.Adam && _options.LearnAdamBetas)
            count += 3 * _numParameters; // beta1, beta2, epsilon

        return count;
    }

    /// <summary>
    /// Creates a deep copy of this per-parameter optimizer.
    /// </summary>
    /// <returns>A new PerParameterOptimizer with copied state.</returns>
    public PerParameterOptimizer<T, TInput, TOutput> Clone()
    {
        var cloned = new PerParameterOptimizer<T, TInput, TOutput>(_numParameters, _options, _engine);

        // Vector<T> converts to T[] by copying (ToArray), so Array.Copy with a vector destination wrote into a
        // temporary and the clone kept its constructor defaults: Adapt, which steps with a clone, ran on the
        // initial learning rates instead of the learned ones.
        cloned._learningRates = cloned.Checked(_learningRates);
        cloned._momentums = cloned.Checked(_momentums);
        cloned._directions = cloned.Checked(_directions);
        cloned._adamBeta1 = cloned.Checked(_adamBeta1);
        cloned._adamBeta2 = cloned.Checked(_adamBeta2);
        cloned._adamEpsilon = cloned.Checked(_adamEpsilon);
        cloned._firstMoments = cloned.Checked(_firstMoments);
        cloned._secondMoments = cloned.Checked(_secondMoments);
        cloned._velocities = cloned.Checked(_velocities);
        Array.Copy(_adamSteps, cloned._adamSteps, _numParameters);

        return cloned;
    }

    #region Getter/Setter Methods

    /// <summary>Gets the learning rate for a specific parameter.</summary>
    public T GetLearningRate(int parameterIndex) => _learningRates[parameterIndex];

    /// <summary>Sets the learning rate for a specific parameter.</summary>
    public void SetLearningRate(int parameterIndex, T learningRate) => _learningRates[parameterIndex] = learningRate;

    /// <summary>Sets the momentum for a specific parameter.</summary>
    public void SetMomentum(int parameterIndex, T momentum) => _momentums[parameterIndex] = momentum;

    /// <summary>Sets the direction for a specific parameter.</summary>
    public void SetDirection(int parameterIndex, T direction) => _directions[parameterIndex] = direction;

    /// <summary>Sets Adam beta1 for a specific parameter.</summary>
    public void SetAdamBeta1(int parameterIndex, T beta1) => _adamBeta1[parameterIndex] = beta1;

    /// <summary>Sets Adam beta2 for a specific parameter.</summary>
    public void SetAdamBeta2(int parameterIndex, T beta2) => _adamBeta2[parameterIndex] = beta2;

    /// <summary>Sets Adam epsilon for a specific parameter.</summary>
    public void SetAdamEpsilon(int parameterIndex, T epsilon) => _adamEpsilon[parameterIndex] = epsilon;

    #endregion

    /// <summary>
    /// Applies regularization and clipping to learned coefficients.
    /// </summary>
    private void ApplyRegularization()
    {
        // Apply L2 regularization to learning rates — vectorized Engine.Multiply
        if (_options.LearningRateL2Reg > 0.0)
        {
            T regFactor = NumOps.FromDouble(1.0 - _options.LearningRateL2Reg);
            _learningRates = (Vector<T>)_engine.Multiply(_learningRates, regFactor);
        }

        // Clip learning rates only to bounds the caller set. Alpha is free by default: its sign is the update
        // direction (Li et al. 2017), which a positive floor made impossible.
        if (_options.MinLearningRate.HasValue || _options.MaxLearningRate.HasValue)
        {
            _learningRates = _engine.Clamp(_learningRates,
                NumOps.FromDouble(_options.MinLearningRate ?? double.NegativeInfinity),
                NumOps.FromDouble(_options.MaxLearningRate ?? double.PositiveInfinity));
        }

        // Clip momentum to [0, 1] — vectorized Engine.Clamp
        if (_options.LearnMomentum)
        {
            _momentums = _engine.Clamp(_momentums, NumOps.Zero, NumOps.One);
        }

        // Clip Adam betas to valid range — vectorized Engine.Clamp
        if (_options.UpdateRuleType == MetaSGDUpdateRuleType.Adam && _options.LearnAdamBetas)
        {
            T minBeta = NumOps.FromDouble(0.0001);
            T maxBeta = NumOps.FromDouble(0.9999);
            T minEps = NumOps.FromDouble(1e-10);
            T maxEps = NumOps.FromDouble(1e-4);

            _adamBeta1 = _engine.Clamp(_adamBeta1, minBeta, maxBeta);
            _adamBeta2 = _engine.Clamp(_adamBeta2, minBeta, maxBeta);
            _adamEpsilon = _engine.Clamp(_adamEpsilon, minEps, maxEps);
        }
    }
}

/// <summary>
/// Wrapper model for Meta-SGD adapted models that includes the per-parameter optimizer.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <typeparam name="TInput">The input type.</typeparam>
/// <typeparam name="TOutput">The output type.</typeparam>
/// <remarks>
/// <para>
/// This model wraps an adapted model along with its per-parameter optimizer,
/// allowing for further adaptation or inspection of learned coefficients.
/// </para>
/// </remarks>
public class MetaSGDAdaptedModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
{
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly PerParameterOptimizer<T, TInput, TOutput> _optimizer;
    private readonly MetaSGDOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// Initializes a new instance of the MetaSGDAdaptedModel.
    /// </summary>
    /// <param name="model">The adapted model.</param>
    /// <param name="optimizer">The per-parameter optimizer used for adaptation.</param>
    /// <param name="options">The Meta-SGD options.</param>
    public MetaSGDAdaptedModel(
        IFullModel<T, TInput, TOutput> model,
        PerParameterOptimizer<T, TInput, TOutput> optimizer,
        MetaSGDOptions<T, TInput, TOutput> options)
    {
        Guard.NotNull(model);
        _model = model;
        Guard.NotNull(optimizer);
        _optimizer = optimizer;
        Guard.NotNull(options);
        _options = options;
    }

    /// <summary>
    /// Gets the model metadata.
    /// </summary>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    /// <summary>
    /// Gets the per-parameter optimizer (for inspection or further adaptation).
    /// </summary>
    public PerParameterOptimizer<T, TInput, TOutput> Optimizer => _optimizer;

    /// <summary>
    /// Makes predictions using the adapted model.
    /// </summary>
    /// <param name="input">The input data.</param>
    /// <returns>The model predictions.</returns>
    public TOutput Predict(TInput input)
    {
        return _model.Predict(input);
    }

    /// <summary>
    /// Trains the model on the given data.
    /// </summary>
    /// <param name="inputs">The input data.</param>
    /// <param name="targets">The target outputs.</param>
    /// <remarks>
    /// <para>
    /// For Meta-SGD adapted models, training is typically done through the
    /// meta-learning adaptation process rather than direct training.
    /// This method delegates to the underlying model's training.
    /// </para>
    /// </remarks>
    public void Train(TInput inputs, TOutput targets)
    {
        _model.Train(inputs, targets);
    }

    /// <summary>
    /// Gets the model metadata.
    /// </summary>
    /// <returns>The metadata for this model.</returns>
    public ModelMetadata<T> GetModelMetadata()
    {
        return Metadata;
    }

    /// <summary>
    /// Gets the current model parameters.
    /// </summary>
    /// <returns>The parameter vector.</returns>
    public Vector<T> GetParameters()
    {
        return InterfaceGuard.Parameterizable(_model).GetParameters();
    }

    /// <summary>
    /// Sets the model parameters.
    /// </summary>
    /// <param name="parameters">The new parameters.</param>
    public void SetParameters(Vector<T> parameters)
    {
        InterfaceGuard.Parameterizable(_model).SetParameters(parameters);
    }
}

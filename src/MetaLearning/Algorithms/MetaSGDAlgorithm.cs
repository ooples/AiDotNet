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
    /// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(inputFeatures: 4, outputSize: 2);
    /// var myNeuralNetwork = new NeuralNetwork&lt;double&gt;(architecture);
    /// // Create Meta-SGD with minimal configuration
    /// var options = new MetaSGDOptions&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(myNeuralNetwork);
    /// var metaSGD = new MetaSGDAlgorithm&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(options);
    ///
    /// // Create Meta-SGD with full per-parameter optimization
    /// var options2 = new MetaSGDOptions&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(myNeuralNetwork)
    /// {
    ///     UpdateRuleType = MetaSGDUpdateRuleType.Adam,
    ///     LearnLearningRate = true,
    ///     LearnMomentum = true,
    ///     LearnDirection = true,
    ///     LearnAdamBetas = true
    /// };
    /// var metaSGD2 = new MetaSGDAlgorithm&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(options2);
    /// </code>
    /// </example>
    public MetaSGDAlgorithm(MetaSGDOptions<T, TInput, TOutput> options)
        : base(
            options?.MetaModel ?? throw new ArgumentNullException(nameof(options), "MetaModel must be set in options."),
            options.LossFunction ?? options.MetaModel.DefaultLossFunction,
            options,
            options.DataLoader,
            options.MetaOptimizer,
            options.InnerOptimizer,
            optimizerUse: MetaLearnerOptimizerUse.None)
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
        var losses = new List<T>(taskBatch.BatchSize);
        Vector<T>? thetaGradient = null;
        Vector<T>? coefficientGradient = null;

        foreach (var task in taskBatch.Tasks)
        {
            var (loss, taskTheta, taskCoefficients) = Episode(task, coefficients);
            losses.Add(loss);
            thetaGradient = thetaGradient is null ? taskTheta : Engine.Add(thetaGradient, taskTheta);
            coefficientGradient = coefficientGradient is null
                ? taskCoefficients
                : Engine.Add(coefficientGradient, taskCoefficients);
        }

        // It used to update only the per-parameter coefficients, so theta - the initialization Meta-SGD
        // learns - stayed at its starting weights however long the model meta-trained.
        double metaLearningRate = ScheduledMetaLearningRate(_currentIteration);
        var theta = ParamModel.GetParameters();
        if (thetaGradient is not null)
        {
            theta = Engine.Subtract(theta, Engine.Multiply(thetaGradient, NumOps.FromDouble(metaLearningRate)));
        }

        ParamModel.SetParameters(theta);
        _optimizer.UpdateMetaParameters(coefficientGradient ?? new Vector<T>(0), metaLearningRate);

        // The schedule reads _currentIteration, so a direct MetaTrain call (no data loader) advances it too; the
        // base MetaTrainStep wrappers then leave it as it is instead of counting the update twice.
        _currentIteration++;
        return ComputeMean(losses);
    }

    /// <summary>
    /// The outer (meta) learning rate for meta-iteration <paramref name="iteration"/> (zero-based).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Li et al. (2017) meta-train with a fixed meta step size, which is what the default
    /// (<see cref="MetaSGDOptions{T, TInput, TOutput}.UseLearningRateSchedule"/> = false) keeps. With the schedule
    /// on, the rate ramps linearly over the first <see cref="MetaSGDOptions{T, TInput, TOutput}.ScheduleWarmupEpisodes"/>
    /// iterations (Goyal et al. 2017) and then follows <see cref="MetaSGDOptions{T, TInput, TOutput}.LearningRateSchedule"/>
    /// over the remaining iterations up to <see cref="MetaSGDOptions{T, TInput, TOutput}.NumMetaIterations"/>, with
    /// progress p in [0, 1]:
    /// Constant keeps the rate; StepDecay halves it at each quarter (gamma 0.5, PyTorch StepLR); Exponential
    /// decays it to 1% (rate * 0.01^p, PyTorch ExponentialLR); CosineAnnealing is rate * (1 + cos(pi p)) / 2
    /// (Loshchilov and Hutter 2017, no restarts); Cyclical is the triangular policy of Smith (2017) between
    /// rate / 10 and rate with four cycles.
    /// </para>
    /// </remarks>
    internal double ScheduledMetaLearningRate(int iteration)
    {
        double rate = _metaSGDOptions.OuterLearningRate;
        if (!_metaSGDOptions.UseLearningRateSchedule) return rate;

        int warmup = Math.Max(0, _metaSGDOptions.ScheduleWarmupEpisodes);
        if (iteration < warmup) return rate * (iteration + 1) / warmup;

        int span = Math.Max(1, _metaSGDOptions.NumMetaIterations - warmup);
        double progress = Math.Min(1.0, Math.Max(0.0, (double)(iteration - warmup) / span));
        switch (_metaSGDOptions.LearningRateSchedule)
        {
            case MetaSGDLearningRateScheduleType.StepDecay:
                return rate * Math.Pow(StepDecayFactor,
                    Math.Min(ScheduleIntervals - 1, Math.Floor(ScheduleIntervals * progress)));
            case MetaSGDLearningRateScheduleType.Exponential:
                return rate * Math.Pow(ExponentialFinalFraction, progress);
            case MetaSGDLearningRateScheduleType.CosineAnnealing:
                return rate * 0.5 * (1 + Math.Cos(Math.PI * progress));
            case MetaSGDLearningRateScheduleType.Cyclical:
            {
                double phase = ScheduleIntervals * progress - Math.Floor(ScheduleIntervals * progress);
                double triangle = 1 - Math.Abs(2 * phase - 1);
                return rate * (CyclicalMinimumFraction + (1 - CyclicalMinimumFraction) * triangle);
            }
            default:
                return rate;
        }
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
    ///     update = α ⊙ d ⊙ gradients + β ⊙ velocity
    ///     params = params - update
    /// </code>
    /// Where α, d, β are the learned per-parameter coefficients.
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
    private sealed class Coefficients
    {
        public Coefficients(
            Vector<T> rate, Vector<T> momentum, Vector<T> direction, Vector<T> beta1, Vector<T> beta2, Vector<T> epsilon)
        {
            Rate = rate;
            Momentum = momentum;
            Direction = direction;
            Beta1 = beta1;
            Beta2 = beta2;
            Epsilon = epsilon;
        }

        public Vector<T> Rate { get; }
        public Vector<T> Momentum { get; }
        public Vector<T> Direction { get; }
        public Vector<T> Beta1 { get; }
        public Vector<T> Beta2 { get; }
        public Vector<T> Epsilon { get; }
    }

    /// <summary>The inner optimizer's per-parameter state before one step. Steps replace it; they never mutate it.</summary>
    private sealed class OptimizerState
    {
        public OptimizerState(Vector<T> velocity, Vector<T> firstMoment, Vector<T> secondMoment, Vector<T> accumulatedDelta)
        {
            Velocity = velocity;
            FirstMoment = firstMoment;
            SecondMoment = secondMoment;
            AccumulatedDelta = accumulatedDelta;
        }

        public static OptimizerState Zero(int n)
            => new OptimizerState(new Vector<T>(n), new Vector<T>(n), new Vector<T>(n), new Vector<T>(n));

        public Vector<T> Velocity { get; }
        public Vector<T> FirstMoment { get; }
        public Vector<T> SecondMoment { get; }
        public Vector<T> AccumulatedDelta { get; }
    }

    /// <summary>What one inner step recorded for the reverse pass: where it started and what it saw.</summary>
    private sealed class InnerStep
    {
        public InnerStep(Vector<T> parameters, Vector<T> rawGradient, T clipScale, bool clipped, OptimizerState state)
        {
            Parameters = parameters;
            RawGradient = rawGradient;
            ClipScale = clipScale;
            Clipped = clipped;
            State = state;
        }

        public Vector<T> Parameters { get; }
        public Vector<T> RawGradient { get; }
        public T ClipScale { get; }
        public bool Clipped { get; }
        public OptimizerState State { get; }
    }

    /// <summary>The reverse pass's running adjoints and coefficient gradients.</summary>
    private sealed class Adjoint
    {
        public Adjoint(Vector<T> theta)
        {
            int n = theta.Length;
            Theta = theta;
            Velocity = new Vector<T>(n);
            FirstMoment = new Vector<T>(n);
            SecondMoment = new Vector<T>(n);
            AccumulatedDelta = new Vector<T>(n);
            Rate = new Vector<T>(n);
            Momentum = new Vector<T>(n);
            Direction = new Vector<T>(n);
            Beta1 = new Vector<T>(n);
            Beta2 = new Vector<T>(n);
            Epsilon = new Vector<T>(n);
        }

        public Vector<T> Theta { get; set; }
        public Vector<T> Velocity { get; set; }
        public Vector<T> FirstMoment { get; set; }
        public Vector<T> SecondMoment { get; set; }
        public Vector<T> AccumulatedDelta { get; set; }
        public Vector<T> Rate { get; set; }
        public Vector<T> Momentum { get; set; }
        public Vector<T> Direction { get; set; }
        public Vector<T> Beta1 { get; set; }
        public Vector<T> Beta2 { get; set; }
        public Vector<T> Epsilon { get; set; }
    }

    private static Coefficients ReadCoefficients(PerParameterOptimizer<T, TInput, TOutput> optimizer)
        => new Coefficients(
            optimizer.LearningRates.Clone(), optimizer.Momentums.Clone(), optimizer.Directions.Clone(),
            optimizer.AdamBeta1.Clone(), optimizer.AdamBeta2.Clone(), optimizer.AdamEpsilon.Clone());

    /// <summary>
    /// One task's query loss and the exact gradients of that loss with respect to theta and to every learned
    /// coefficient, in the layout <see cref="PerParameterOptimizer{T, TInput, TOutput}.UpdateMetaParameters(Vector{T})"/> reads.
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
    private (T Loss, Vector<T> ThetaGradient, Vector<T> CoefficientGradient) Episode(
        IMetaLearningTask<T, TInput, TOutput> task, Coefficients coefficients)
    {
        var model = CloneModel();
        var (_, trace) = RunInnerLoop(
            model, ParamModel.GetParameters(), task.SupportInput, task.SupportOutput,
            _metaSGDOptions.InnerSteps, coefficients, record: true);

        T loss = ComputeLossFromOutput(model.Predict(task.QueryInput), task.QueryOutput);
        var queryGradient = ComputeGradients(model, task.QueryInput, task.QueryOutput).Clone();

        var adjoint = Backpropagate(
            model, trace ?? new List<InnerStep>(), queryGradient, task.SupportInput, task.SupportOutput, coefficients);

        return (loss, adjoint.Theta, PackCoefficientGradients(adjoint));
    }

    /// <summary>
    /// The coefficient gradients in <see cref="PerParameterOptimizer{T, TInput, TOutput}"/>'s layout: the learned
    /// rates, momentums and directions as blocks, then (beta1, beta2, epsilon) interleaved per parameter.
    /// </summary>
    private Vector<T> PackCoefficientGradients(Adjoint adjoint)
    {
        var blocks = new List<Tensor<T>>();
        if (_metaSGDOptions.LearnLearningRate) blocks.Add(Tensor<T>.FromVector(adjoint.Rate));
        if (_metaSGDOptions.LearnMomentum) blocks.Add(Tensor<T>.FromVector(adjoint.Momentum));
        if (_metaSGDOptions.LearnDirection) blocks.Add(Tensor<T>.FromVector(adjoint.Direction));
        if (_metaSGDOptions.UpdateRuleType == MetaSGDUpdateRuleType.Adam && _metaSGDOptions.LearnAdamBetas)
        {
            var interleaved = Engine.TensorStack(new[]
            {
                Tensor<T>.FromVector(adjoint.Beta1), Tensor<T>.FromVector(adjoint.Beta2), Tensor<T>.FromVector(adjoint.Epsilon),
            }, 1);
            blocks.Add(interleaved.Reshape(interleaved.Length));
        }

        return blocks.Count == 0 ? new Vector<T>(0) : Engine.Concat(blocks, 0).ToVector();
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
        var theta = start.Clone();
        var state = OptimizerState.Zero(start.Length);
        var trace = record ? new List<InnerStep>(steps) : null;

        for (int k = 0; k < steps; k++)
        {
            parameters.SetParameters(theta);
            var raw = ComputeGradients(model, input, target).Clone();
            var (scale, clipped) = ClipScale(raw);

            trace?.Add(new InnerStep(theta, raw, scale, clipped, state));

            var (update, next) = StepForward(k + 1, Engine.Multiply(raw, scale), state, c);
            state = next;
            theta = Engine.Subtract(theta, update);
        }

        parameters.SetParameters(theta);
        return (theta, trace);
    }

    /// <summary>Global-norm clipping of the inner gradient (the factor multiplying it; 1 when not clipped).</summary>
    private (T Scale, bool Clipped) ClipScale(Vector<T> gradient)
    {
        if (!_metaSGDOptions.GradientClipThreshold.HasValue) return (NumOps.One, false);
        T norm = Engine.Norm(gradient);
        T threshold = NumOps.FromDouble(_metaSGDOptions.GradientClipThreshold.Value);
        return NumOps.GreaterThan(norm, threshold) && NumOps.GreaterThan(norm, NumOps.Zero)
            ? (NumOps.Divide(threshold, norm), true)
            : (NumOps.One, false);
    }

    private const double RmsDecay = 0.9;
    private const double RmsEpsilon = 1e-6;
    private const double AdaDeltaRho = 0.95;
    private const double AdaDeltaEpsilon = 1e-6;

    // Outer-rate schedule policy (see ScheduledMetaLearningRate): StepDecay halves the rate at each of four equal
    // intervals (PyTorch StepLR, gamma 0.5) and Cyclical runs four triangular cycles; Exponential ends at 1% of the
    // rate; Cyclical's trough is 10% of it (Smith 2017).
    private const int ScheduleIntervals = 4;
    private const double StepDecayFactor = 0.5;
    private const double ExponentialFinalFraction = 0.01;
    private const double CyclicalMinimumFraction = 0.1;

    // Relative finite-difference step of the Hessian-vector product: h = HessianProbeScale (1 + |at|) / |v|.
    private const double HessianProbeScale = 1e-5;

    // Starting value of every learned momentum coefficient (the common SGD-momentum default).
    private const double InitialMomentum = 0.9;

    // Layer-wise decay treats the flat parameter vector as this many equal layers: rate * decay^(L i / n).
    private const double ApproximateLayerCount = 10.0;

    /// <summary>1 - v, element-wise.</summary>
    private Vector<T> OneMinus(Vector<T> v) => Engine.Negate(v) + NumOps.One;

    /// <summary>The learned step scale: alpha, or alpha * d when directions are learned.</summary>
    private Vector<T> StepScale(Coefficients c)
        => _metaSGDOptions.LearnDirection ? Engine.Multiply(c.Rate, c.Direction) : c.Rate;

    /// <summary>
    /// <paramref name="numerator"/> * 0.5 / <paramref name="root"/> where the root is positive and 0 where it is
    /// not: the derivative of sqrt(max(0, s)) through s, which is 0 at s = 0.
    /// </summary>
    private Vector<T> HalfOverPositiveRoot(Vector<T> numerator, Vector<T> root)
    {
        var positive = Engine.TensorGreaterThan(Tensor<T>.FromVector(root), NumOps.Zero).ToVector();
        var safeRoot = Engine.Add(root, OneMinus(positive));
        return Engine.Divide(Engine.Multiply(Engine.Multiply(numerator, positive), NumOps.FromDouble(0.5)), safeRoot);
    }

    private Vector<T> NonNegative(Vector<T> v) => Engine.Max(v, new Vector<T>(v.Length));

    /// <summary>
    /// One inner step of the configured update rule for every parameter at once: returns the update u
    /// (theta' = theta - u) and the advanced optimizer state.
    /// </summary>
    private (Vector<T> Update, OptimizerState Next) StepForward(int t, Vector<T> g, OptimizerState s, Coefficients c)
    {
        switch (_metaSGDOptions.UpdateRuleType)
        {
            case MetaSGDUpdateRuleType.SGD:
                return (Engine.Multiply(StepScale(c), g), s);
            case MetaSGDUpdateRuleType.SGDWithMomentum:
            {
                var velocity = Engine.Add(Engine.Multiply(c.Momentum, s.Velocity), Engine.Multiply(StepScale(c), g));
                return (velocity, new OptimizerState(velocity, s.FirstMoment, s.SecondMoment, s.AccumulatedDelta));
            }
            case MetaSGDUpdateRuleType.Adam:
            {
                var first = Engine.Add(Engine.Multiply(c.Beta1, s.FirstMoment), Engine.Multiply(OneMinus(c.Beta1), g));
                var second = Engine.Add(
                    Engine.Multiply(c.Beta2, s.SecondMoment), Engine.Multiply(OneMinus(c.Beta2), Engine.Multiply(g, g)));
                T step = NumOps.FromDouble(t);
                var firstHat = Engine.Divide(first, OneMinus(Engine.Power(c.Beta1, step)));
                var secondHat = Engine.Divide(second, OneMinus(Engine.Power(c.Beta2, step)));
                var denominator = Engine.Add(Engine.Sqrt(NonNegative(secondHat)), c.Epsilon);
                var update = Engine.Divide(Engine.Multiply(c.Rate, firstHat), denominator);
                return (update, new OptimizerState(s.Velocity, first, second, s.AccumulatedDelta));
            }
            case MetaSGDUpdateRuleType.RMSprop:
            case MetaSGDUpdateRuleType.AdaGrad:
            {
                bool rms = _metaSGDOptions.UpdateRuleType == MetaSGDUpdateRuleType.RMSprop;
                var squared = Engine.Multiply(g, g);
                var second = rms
                    ? Engine.Add(Engine.Multiply(s.SecondMoment, NumOps.FromDouble(RmsDecay)),
                        Engine.Multiply(squared, NumOps.FromDouble(1 - RmsDecay)))
                    : Engine.Add(s.SecondMoment, squared);
                var denominator = Engine.Sqrt(NonNegative(second)) + NumOps.FromDouble(RmsEpsilon);
                var update = Engine.Divide(Engine.Multiply(c.Rate, g), denominator);
                return (update, new OptimizerState(s.Velocity, s.FirstMoment, second, s.AccumulatedDelta));
            }
            case MetaSGDUpdateRuleType.AdaDelta:
            {
                // Zeiler 2012, Algorithm 1: RMS[x] = sqrt(E[x^2] + eps).
                var second = Engine.Add(Engine.Multiply(s.SecondMoment, NumOps.FromDouble(AdaDeltaRho)),
                    Engine.Multiply(Engine.Multiply(g, g), NumOps.FromDouble(1 - AdaDeltaRho)));
                var rmsGradient = Engine.Sqrt(second + NumOps.FromDouble(AdaDeltaEpsilon));
                var rmsDelta = Engine.Sqrt(s.AccumulatedDelta + NumOps.FromDouble(AdaDeltaEpsilon));
                var update = Engine.Multiply(Engine.Divide(rmsDelta, rmsGradient), g);
                var delta = Engine.Add(Engine.Multiply(s.AccumulatedDelta, NumOps.FromDouble(AdaDeltaRho)),
                    Engine.Multiply(Engine.Multiply(update, update), NumOps.FromDouble(1 - AdaDeltaRho)));
                return (update, new OptimizerState(s.Velocity, s.FirstMoment, second, delta));
            }
            default:
                return (Engine.Multiply(c.Rate, g), s);
        }
    }

    /// <summary>
    /// Reverse pass through the recorded inner loop: the exact gradient of the query loss with respect to theta
    /// (through every step's support gradient, unless first-order) and to every learned coefficient.
    /// </summary>
    private Adjoint Backpropagate(IFullModel<T, TInput, TOutput> model, List<InnerStep> trace, Vector<T> queryGradient,
        TInput supportInput, TOutput supportTarget, Coefficients c)
    {
        var adjoint = new Adjoint(queryGradient);

        for (int k = trace.Count - 1; k >= 0; k--)
        {
            var step = trace[k];
            var lambdaG = StepBackward(k + 1, step, c, adjoint);

            // Through the clip: g_used = s(g) g with s = threshold / |g| when clipped.
            if (step.Clipped) lambdaG = ClipBackward(step, lambdaG);

            if (!_metaSGDOptions.UseFirstOrder)
            {
                adjoint.Theta = Engine.Add(
                    adjoint.Theta, HessianVectorProduct(model, step.Parameters, lambdaG, supportInput, supportTarget));
            }
        }

        return adjoint;
    }

    /// <summary>
    /// The adjoint of the clipped gradient g s(g), s = threshold / |g|:
    /// s (lambda - g (g . lambda) / |g|^2).
    /// </summary>
    private Vector<T> ClipBackward(InnerStep step, Vector<T> lambdaG)
    {
        var raw = step.RawGradient;
        T projection = NumOps.Divide(Engine.DotProduct(raw, lambdaG), Engine.DotProduct(raw, raw));
        return Engine.Multiply(Engine.Subtract(lambdaG, Engine.Multiply(raw, projection)), step.ClipScale);
    }

    /// <summary>
    /// One inner step of the reverse pass for every parameter at once: consumes the update's adjoint
    /// -lambda_theta, carries the optimizer-state adjoints back one step, accumulates the coefficient
    /// gradients, and returns the adjoint of the (clipped) gradient the step used.
    /// </summary>
    private Vector<T> StepBackward(int t, InnerStep step, Coefficients c, Adjoint a)
    {
        var g = Engine.Multiply(step.RawGradient, step.ClipScale);
        var s = step.State;
        var lambdaU = Engine.Negate(a.Theta);
        bool useDirection = _metaSGDOptions.LearnDirection;
        var scale = StepScale(c);

        switch (_metaSGDOptions.UpdateRuleType)
        {
            case MetaSGDUpdateRuleType.SGD:
            {
                var directionFactor = useDirection ? c.Direction : null;
                var lambdaUg = Engine.Multiply(lambdaU, g);
                a.Rate = Engine.Add(a.Rate, directionFactor is null ? lambdaUg : Engine.Multiply(lambdaUg, directionFactor));
                if (useDirection) a.Direction = Engine.Add(a.Direction, Engine.Multiply(lambdaUg, c.Rate));
                return Engine.Multiply(lambdaU, scale);
            }

            case MetaSGDUpdateRuleType.SGDWithMomentum:
            {
                // v' = mu v + scale g, u = v'
                var lv = Engine.Add(a.Velocity, lambdaU);
                var lvg = Engine.Multiply(lv, g);
                a.Momentum = Engine.Add(a.Momentum, Engine.Multiply(lv, s.Velocity));
                a.Rate = Engine.Add(a.Rate, useDirection ? Engine.Multiply(lvg, c.Direction) : lvg);
                if (useDirection) a.Direction = Engine.Add(a.Direction, Engine.Multiply(lvg, c.Rate));
                a.Velocity = Engine.Multiply(lv, c.Momentum);
                return Engine.Multiply(lv, scale);
            }

            case MetaSGDUpdateRuleType.Adam:
            {
                T stepT = NumOps.FromDouble(t);
                var squared = Engine.Multiply(g, g);
                var firstNew = Engine.Add(Engine.Multiply(c.Beta1, s.FirstMoment), Engine.Multiply(OneMinus(c.Beta1), g));
                var secondNew = Engine.Add(Engine.Multiply(c.Beta2, s.SecondMoment), Engine.Multiply(OneMinus(c.Beta2), squared));
                var c1 = OneMinus(Engine.Power(c.Beta1, stepT));
                var c2 = OneMinus(Engine.Power(c.Beta2, stepT));
                var firstHat = Engine.Divide(firstNew, c1);
                var secondHat = Engine.Divide(secondNew, c2);
                var root = Engine.Sqrt(NonNegative(secondHat));
                var denominator = Engine.Add(root, c.Epsilon);

                // d u / d den = -rate mHat / den^2
                var lambdaDenominator = Engine.Multiply(lambdaU,
                    Engine.Negate(Engine.Divide(Engine.Multiply(c.Rate, firstHat), Engine.Multiply(denominator, denominator))));
                a.Rate = Engine.Add(a.Rate, Engine.Divide(Engine.Multiply(lambdaU, firstHat), denominator));
                a.Epsilon = Engine.Add(a.Epsilon, lambdaDenominator);
                var lambdaFirstHat = Engine.Divide(Engine.Multiply(lambdaU, c.Rate), denominator);
                var lambdaSecondHat = HalfOverPositiveRoot(lambdaDenominator, root);

                // Bias corrections: d(1 - b^t)/db = -t b^(t-1), so d(m / c1)/db1 = m t b1^(t-1) / c1^2.
                T stepMinusOne = NumOps.FromDouble(t - 1);
                var lambdaFirstNew = Engine.Add(a.FirstMoment, Engine.Divide(lambdaFirstHat, c1));
                var lambdaSecondNew = Engine.Add(a.SecondMoment, Engine.Divide(lambdaSecondHat, c2));
                var beta1Correction = Engine.Divide(
                    Engine.Multiply(Engine.Multiply(lambdaFirstHat, firstNew), Engine.Multiply(Engine.Power(c.Beta1, stepMinusOne), stepT)),
                    Engine.Multiply(c1, c1));
                var beta2Correction = Engine.Divide(
                    Engine.Multiply(Engine.Multiply(lambdaSecondHat, secondNew), Engine.Multiply(Engine.Power(c.Beta2, stepMinusOne), stepT)),
                    Engine.Multiply(c2, c2));
                a.Beta1 = Engine.Add(Engine.Add(a.Beta1, beta1Correction),
                    Engine.Multiply(lambdaFirstNew, Engine.Subtract(s.FirstMoment, g)));
                a.Beta2 = Engine.Add(Engine.Add(a.Beta2, beta2Correction),
                    Engine.Multiply(lambdaSecondNew, Engine.Subtract(s.SecondMoment, squared)));
                a.FirstMoment = Engine.Multiply(lambdaFirstNew, c.Beta1);
                a.SecondMoment = Engine.Multiply(lambdaSecondNew, c.Beta2);
                return Engine.Add(
                    Engine.Multiply(lambdaFirstNew, OneMinus(c.Beta1)),
                    Engine.Multiply(Engine.Multiply(lambdaSecondNew, OneMinus(c.Beta2)), Engine.Multiply(g, NumOps.FromDouble(2))));
            }

            case MetaSGDUpdateRuleType.RMSprop:
            case MetaSGDUpdateRuleType.AdaGrad:
            {
                bool rms = _metaSGDOptions.UpdateRuleType == MetaSGDUpdateRuleType.RMSprop;
                T decay = NumOps.FromDouble(rms ? RmsDecay : 1.0);
                T gain = NumOps.FromDouble(rms ? 1 - RmsDecay : 1.0);
                var secondNew = Engine.Add(Engine.Multiply(s.SecondMoment, decay), Engine.Multiply(Engine.Multiply(g, g), gain));
                var root = Engine.Sqrt(NonNegative(secondNew));
                var denominator = root + NumOps.FromDouble(RmsEpsilon);

                a.Rate = Engine.Add(a.Rate, Engine.Divide(Engine.Multiply(lambdaU, g), denominator));
                var lambdaG = Engine.Divide(Engine.Multiply(lambdaU, c.Rate), denominator);
                var lambdaDenominator = Engine.Multiply(lambdaU,
                    Engine.Negate(Engine.Divide(Engine.Multiply(c.Rate, g), Engine.Multiply(denominator, denominator))));
                var lambdaSecondNew = Engine.Add(a.SecondMoment, HalfOverPositiveRoot(lambdaDenominator, root));
                a.SecondMoment = Engine.Multiply(lambdaSecondNew, decay);
                return Engine.Add(lambdaG,
                    Engine.Multiply(Engine.Multiply(lambdaSecondNew, g), NumOps.Multiply(gain, NumOps.FromDouble(2))));
            }

            case MetaSGDUpdateRuleType.AdaDelta:
            {
                T rho = NumOps.FromDouble(AdaDeltaRho);
                T oneMinusRho = NumOps.FromDouble(1 - AdaDeltaRho);
                var secondNew = Engine.Add(Engine.Multiply(s.SecondMoment, rho), Engine.Multiply(Engine.Multiply(g, g), oneMinusRho));
                var rmsGradient = Engine.Sqrt(secondNew + NumOps.FromDouble(AdaDeltaEpsilon));
                var rmsDelta = Engine.Sqrt(s.AccumulatedDelta + NumOps.FromDouble(AdaDeltaEpsilon));
                var u = Engine.Multiply(Engine.Divide(rmsDelta, rmsGradient), g);

                // The accumulated update after this step depends on u, so its adjoint flows into u first.
                var lu = Engine.Add(lambdaU,
                    Engine.Multiply(Engine.Multiply(a.AccumulatedDelta, u), NumOps.Multiply(oneMinusRho, NumOps.FromDouble(2))));
                var lambdaG = Engine.Divide(Engine.Multiply(lu, rmsDelta), rmsGradient);
                var lambdaRmsDelta = Engine.Divide(Engine.Multiply(lu, g), rmsGradient);
                var lambdaRmsGradient = Engine.Negate(
                    Engine.Divide(Engine.Multiply(Engine.Multiply(lu, rmsDelta), g), Engine.Multiply(rmsGradient, rmsGradient)));
                var lambdaSecondNew = Engine.Add(a.SecondMoment,
                    Engine.Divide(Engine.Multiply(lambdaRmsGradient, NumOps.FromDouble(0.5)), rmsGradient));
                a.SecondMoment = Engine.Multiply(lambdaSecondNew, rho);
                a.AccumulatedDelta = Engine.Add(Engine.Multiply(a.AccumulatedDelta, rho),
                    Engine.Divide(Engine.Multiply(lambdaRmsDelta, NumOps.FromDouble(0.5)), rmsDelta));
                return Engine.Add(lambdaG,
                    Engine.Multiply(Engine.Multiply(lambdaSecondNew, g), NumOps.Multiply(oneMinusRho, NumOps.FromDouble(2))));
            }

            default:
                a.Rate = Engine.Add(a.Rate, Engine.Multiply(lambdaU, g));
                return Engine.Multiply(lambdaU, c.Rate);
        }
    }

    /// <summary>
    /// H v at <paramref name="at"/> by a central difference of support gradients, with a step scaled to the
    /// parameter and direction norms.
    /// </summary>
    private Vector<T> HessianVectorProduct(
        IFullModel<T, TInput, TOutput> model, Vector<T> at, Vector<T> v, TInput input, TOutput target)
    {
        T vNorm = Engine.Norm(v);
        if (NumOps.Equals(vNorm, NumOps.Zero)) return new Vector<T>(v.Length);

        T h = NumOps.Divide(
            NumOps.Multiply(NumOps.FromDouble(HessianProbeScale), NumOps.Add(NumOps.One, Engine.Norm(at))), vNorm);
        var offset = Engine.Multiply(v, h);

        var parameters = InterfaceGuard.Parameterizable(model);
        parameters.SetParameters(Engine.Add(at, offset));
        var gradientPlus = ComputeGradients(model, input, target).Clone();
        parameters.SetParameters(Engine.Subtract(at, offset));
        var gradientMinus = ComputeGradients(model, input, target).Clone();

        return Engine.Divide(Engine.Subtract(gradientPlus, gradientMinus), NumOps.Multiply(NumOps.FromDouble(2), h));
    }

    /// <summary>The per-parameter optimizer holding the learned coefficients (for tests and inspection).</summary>
    internal PerParameterOptimizer<T, TInput, TOutput> LearnedOptimizer => _optimizer;

    /// <summary>One task's query loss and its exact gradients, for gradient checks.</summary>
    internal (T Loss, Vector<T> ThetaGradient, Vector<T> CoefficientGradient) EpisodeGradientForTesting(
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

        // Initialize learning rates based on initialization strategy. Random draws one value per parameter, in
        // parameter order, from the learner's seeded stream.
        if (_metaSGDOptions.LearningRateInitialization == MetaSGDLearningRateInitialization.Random)
        {
            for (int i = 0; i < numParams; i++)
            {
                _optimizer.SetLearningRate(i, GetInitialLearningRate(i, numParams));
            }
        }
        else
        {
            _optimizer.LearningRates = InitialLearningRates(numParams);
        }

        // Initialize momentum if enabled
        if (_metaSGDOptions.LearnMomentum)
        {
            _optimizer.Momentums = Vector<T>.CreateDefault(numParams, NumOps.FromDouble(InitialMomentum));
        }

        // Initialize direction if enabled
        if (_metaSGDOptions.LearnDirection)
        {
            _optimizer.Directions = Vector<T>.CreateDefault(numParams, NumOps.One);
        }

        // Initialize Adam parameters if using Adam
        if (_metaSGDOptions.UpdateRuleType == MetaSGDUpdateRuleType.Adam)
        {
            _optimizer.AdamBeta1 = Vector<T>.CreateDefault(numParams, NumOps.FromDouble(_metaSGDOptions.AdamBeta1Init));
            _optimizer.AdamBeta2 = Vector<T>.CreateDefault(numParams, NumOps.FromDouble(_metaSGDOptions.AdamBeta2Init));
            _optimizer.AdamEpsilon = Vector<T>.CreateDefault(numParams, NumOps.FromDouble(_metaSGDOptions.AdamEpsilonInit));
        }
    }

    /// <summary>
    /// The deterministic initial learning rates: the base rate, scaled by 1/sqrt(n) for Xavier, or decayed by
    /// LayerDecayFactor^(10 i / n) along the parameter index for layer-based initialization with layer-wise decay.
    /// </summary>
    private Vector<T> InitialLearningRates(int totalParams)
    {
        double baseLR = _metaSGDOptions.InnerLearningRate;
        switch (_metaSGDOptions.LearningRateInitialization)
        {
            case MetaSGDLearningRateInitialization.LayerBased when _metaSGDOptions.UseLayerWiseDecay:
            {
                // Approximate layer from parameter index: baseLR * decay^(L i / n) = baseLR * exp(i * L ln(decay) / n)
                var exponent = Engine.TensorMultiplyScalar(
                    Engine.TensorArange<T>(0, totalParams, 1),
                    NumOps.FromDouble(ApproximateLayerCount * Math.Log(_metaSGDOptions.LayerDecayFactor) / totalParams));
                return Engine.Multiply(Engine.TensorExp(exponent).ToVector(), NumOps.FromDouble(baseLR));
            }

            case MetaSGDLearningRateInitialization.Xavier:
                // Xavier-inspired: scale by 1/sqrt(n)
                return Vector<T>.CreateDefault(totalParams, NumOps.FromDouble(baseLR / Math.Sqrt(totalParams)));

            default:
                return Vector<T>.CreateDefault(totalParams, NumOps.FromDouble(baseLR));
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
        return value.Clone();
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
                T stepCount = NumOps.FromDouble(adamStep);
                T biasCorrectedFirst = NumOps.Divide(_firstMoments[parameterIndex],
                    NumOps.Subtract(NumOps.One, NumOps.Power(_adamBeta1[parameterIndex], stepCount)));
                T biasCorrectedSecond = NumOps.Divide(_secondMoments[parameterIndex],
                    NumOps.Subtract(NumOps.One, NumOps.Power(_adamBeta2[parameterIndex], stepCount)));

                // Compute update
                T sqrtSecond = SqrtOfNonNegative(biasCorrectedSecond);
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

                T sqrtSecondRms = SqrtOfNonNegative(_secondMoments[parameterIndex]);
                T denomRms = NumOps.Add(sqrtSecondRms, NumOps.FromDouble(1e-6));
                update = NumOps.Divide(NumOps.Multiply(lr, gradient), denomRms);
                break;

            case MetaSGDUpdateRuleType.AdaGrad:
                // AdaGrad update
                _secondMoments[parameterIndex] = NumOps.Add(
                    _secondMoments[parameterIndex],
                    NumOps.Multiply(gradient, gradient));

                T sqrtAccum = SqrtOfNonNegative(_secondMoments[parameterIndex]);
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
                T rmsGrad = NumOps.Sqrt(NumOps.Add(_secondMoments[parameterIndex], NumOps.FromDouble(1e-6)));

                // Use velocity as accumulated parameter updates
                T rmsDelta = NumOps.Sqrt(NumOps.Add(_velocities[parameterIndex], NumOps.FromDouble(1e-6)));

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
        => UpdateMetaParameters(metaGradients, _options.OuterLearningRate);

    /// <summary>sqrt(max(0, x)).</summary>
    private static T SqrtOfNonNegative(T x) => NumOps.Sqrt(NumOps.GreaterThan(x, NumOps.Zero) ? x : NumOps.Zero);

    /// <summary>
    /// Updates the meta-parameters (learned coefficients) with an explicit meta learning rate, such as the
    /// scheduled rate of the current meta-iteration.
    /// </summary>
    /// <param name="metaGradients">Gradients for the meta-parameters, in <see cref="GetMetaParameterCount"/> layout.</param>
    /// <param name="metaLearningRate">The step size of this update; zero (a schedule's endpoint) is allowed.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="metaLearningRate"/> is not finite.</exception>
    internal void UpdateMetaParameters(Vector<T> metaGradients, double metaLearningRate)
    {
        if (metaGradients is null) throw new ArgumentNullException(nameof(metaGradients));
        if (double.IsNaN(metaLearningRate) || double.IsInfinity(metaLearningRate))
            throw new ArgumentOutOfRangeException(nameof(metaLearningRate), metaLearningRate,
                "The meta learning rate must be finite.");
        int n = _numParameters;
        int index = 0;
        T metaLR = NumOps.FromDouble(metaLearningRate);

        // coefficient -= metaLR * gradient, for the block of the gradient that belongs to it
        Vector<T> Step(Vector<T> coefficients, Vector<T> gradient)
            => _engine.Subtract(coefficients, _engine.Multiply(gradient, metaLR));

        Vector<T> NextBlock(int length)
        {
            var block = metaGradients.GetSubVector(index, length);
            index += length;
            return block;
        }

        // Update learning rates
        if (_options.LearnLearningRate) _learningRates = Step(_learningRates, NextBlock(n));

        // Update momentums
        if (_options.LearnMomentum) _momentums = Step(_momentums, NextBlock(n));

        // Update directions
        if (_options.LearnDirection) _directions = Step(_directions, NextBlock(n));

        // Update Adam parameters: (beta1, beta2, epsilon) interleaved per parameter
        if (_options.UpdateRuleType == MetaSGDUpdateRuleType.Adam && _options.LearnAdamBetas)
        {
            var rows = Tensor<T>.FromVector(NextBlock(3 * n)).Reshape(n, 3).ToMatrix();
            _adamBeta1 = Step(_adamBeta1, rows.GetColumn(0));
            _adamBeta2 = Step(_adamBeta2, rows.GetColumn(1));
            _adamEpsilon = Step(_adamEpsilon, rows.GetColumn(2));
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

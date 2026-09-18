using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Models;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Data.Structures;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of Meta-learning with Differentiable Convex Optimization (MetaOptNet) algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// MetaOptNet (Lee et al. 2019) uses a regularized linear classifier as the base learner: the embedding network
/// produces features, a convex solver fits a classifier to the support set, and the query loss of that classifier
/// trains the embedding. The paper's base learner is the Crammer and Singer multi-class SVM in its dual form
/// (eq. 10), with ridge regression and logistic regression as the other convex choices.
/// </para>
/// <para>
/// <b>The gradient through the solver.</b> The paper differentiates the base learner by applying the implicit
/// function theorem to the solver's KKT conditions (section 3.2), not by differentiating its iterations. Each
/// solver here therefore runs numerically, off the tape, and its solution enters the tape as
/// <c>theta = theta* - Jinv F(theta*, Z)</c>: the KKT residual <c>F</c> is rebuilt from the embeddings with engine
/// ops, and <c>Jinv</c> - the inverse KKT Jacobian at the solution - is a constant. The value is the solution
/// (the residual vanishes there) and the derivative is exactly the implicit one, so the embedding network's
/// meta-gradient runs through the base learner at the cost of one multiplication by a constant matrix.
/// </para>
/// <para>
/// <b>What this replaced.</b> "SVM" was ridge regression on +/-1 labels; the linear system was solved by
/// Gauss-Seidel iteration; logistic regression took fixed 0.1-sized steps and called them Newton's method; and the
/// encoder's meta-gradient was the configured loss of its raw output against the labels, which never involved the
/// solver at all. Query logits also flattened the batch into one vector.
/// </para>
/// <para>
/// <b>For Beginners:</b> MetaOptNet learns a feature extractor that produces embeddings where simple classifiers
/// work well. The convex solver finds the best simple classifier for the few labelled examples, and the feature
/// extractor is updated so that classifier does better on the rest of the task.
/// </para>
/// <para>
/// Reference: Lee, K., Maji, S., Ravichandran, A., &amp; Soatto, S. (2019).
/// Meta-Learning with Differentiable Convex Optimization. CVPR 2019.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Meta-Learning with Differentiable Convex Optimization",
    "https://arxiv.org/abs/1904.03758",
    Year = 2019,
    Authors = "Kwonjoon Lee, Subhransu Maji, Avinash Ravichandran, Stefano Soatto")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class MetaOptNetAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly MetaOptNetOptions<T, TInput, TOutput> _metaOptNetOptions;

    /// <summary>
    /// The learnable scale that multiplies the logits (eq. 12's gamma), one value; learned unless
    /// <see cref="MetaOptNetOptions{T,TInput,TOutput}.UseLearnedTemperature"/> is off.
    /// </summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _logitScale;

    /// <summary>
    /// Initializes a new instance of the MetaOptNetAlgorithm class.
    /// </summary>
    /// <param name="options">MetaOptNet configuration options containing the model and all hyperparameters.</param>
    /// <exception cref="ArgumentNullException">Thrown when options is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the configuration is invalid.</exception>
    public MetaOptNetAlgorithm(MetaOptNetOptions<T, TInput, TOutput> options)
        : base(
            options?.MetaModel ?? throw new ArgumentNullException(nameof(options), "MetaModel must be set in options."),
            options.LossFunction ?? new CrossEntropyWithLogitsLoss<T>(),
            options,
            options.DataLoader,
            options.MetaOptimizer,
            options.InnerOptimizer)
    {
        _metaOptNetOptions = options;
        if (!options.IsValid())
        {
            throw new ArgumentException("MetaOptNet configuration is invalid. Check all parameters.", nameof(options));
        }

        _logitScale = new Vector<T>(1);
        _logitScale[0] = NumOps.FromDouble(options.InitialTemperature);
    }

    /// <summary>
    /// Gets the algorithm type identifier for this meta-learner.
    /// </summary>
    /// <value>Returns <see cref="MetaLearningAlgorithmType.MetaOptNet"/>.</value>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.MetaOptNet;

    /// <summary>
    /// Performs one meta-training step: each task's query loss under its solved base learner, differentiated through
    /// the solver into the embedding network and the logit scale.
    /// </summary>
    /// <param name="taskBatch">A batch of tasks to meta-train on, each containing support and query sets.</param>
    /// <returns>The average query loss across the batch.</returns>
    /// <exception cref="ArgumentException">Thrown when the task batch is null or empty.</exception>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null || taskBatch.BatchSize == 0)
        {
            throw new ArgumentException("Task batch cannot be null or empty.", nameof(taskBatch));
        }

        var body = ParamModel.GetParameters();
        Vector<T>? bodyGradient = null, scaleGradient = null;
        T totalLoss = NumOps.Zero;
        foreach (var task in taskBatch.Tasks)
        {
            var (loss, taskBody, scale) = EpisodeGradient(task);
            totalLoss = NumOps.Add(totalLoss, loss);
            bodyGradient = Accumulate(bodyGradient, taskBody);
            scaleGradient = Accumulate(scaleGradient, scale);
        }

        T batchSize = NumOps.FromDouble(taskBatch.BatchSize);
        bodyGradient = Scale(bodyGradient ?? new Vector<T>(body.Length), batchSize);
        scaleGradient = Scale(scaleGradient ?? new Vector<T>(1), batchSize);

        if (_metaOptNetOptions.EncoderL2Regularization > 0)
        {
            T decay = NumOps.FromDouble(_metaOptNetOptions.EncoderL2Regularization);
            for (int i = 0; i < bodyGradient.Length; i++)
            {
                bodyGradient[i] = NumOps.Add(bodyGradient[i], NumOps.Multiply(decay, body[i]));
            }
        }

        if (_metaOptNetOptions.GradientClipThreshold.HasValue && _metaOptNetOptions.GradientClipThreshold.Value > 0)
        {
            bodyGradient = ClipGradients(bodyGradient, _metaOptNetOptions.GradientClipThreshold.Value);
        }

        double beta = _metaOptNetOptions.OuterLearningRate;
        ParamModel.SetParameters(ApplyGradients(body, bodyGradient, beta));
        if (_metaOptNetOptions.UseLearnedTemperature)
        {
            _logitScale = ApplyGradients(_logitScale, scaleGradient, beta);
        }

        return NumOps.Divide(totalLoss, batchSize);
    }

    /// <summary>
    /// Adapts to a new task by solving the convex base learner on its support set.
    /// </summary>
    /// <param name="task">The new task containing support set examples for adaptation.</param>
    /// <returns>The embedding network's own copy with the classifier the solver produced.</returns>
    /// <exception cref="ArgumentNullException">Thrown when task is null.</exception>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null)
        {
            throw new ArgumentNullException(nameof(task));
        }

        var episode = Episode(task);
        var (support, _) = Embed(task, episode);
        var solved = Solve(support, episode);

        // One row per class of NumClasses, in class order: a class the task did not contain keeps a zero row, which
        // scores zero for every example.
        var weights = new Matrix<T>(_metaOptNetOptions.NumClasses, _metaOptNetOptions.EmbeddingDimension);
        var primal = Primal(solved.Coefficients, support);
        for (int c = 0; c < episode.ClassSlots.Length; c++)
        {
            for (int j = 0; j < _metaOptNetOptions.EmbeddingDimension; j++)
            {
                weights[episode.ClassSlots[c], j] = primal[c * _metaOptNetOptions.EmbeddingDimension + j];
            }
        }

        return new MetaOptNetModel<T, TInput, TOutput>(MetaModel, weights, _logitScale[0], _metaOptNetOptions);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// The adapted model returns one score row per example, so this is the configured loss of those scores against
    /// the class indices. A Vector output carries one predicted class per example, and its loss is the error rate.
    /// </remarks>
    protected override T ComputeLossFromOutput(TOutput predictions, TOutput expectedOutput)
        => ClassifierOutputs<T>.ScoreLoss(LossFunction, predictions, expectedOutput, _metaOptNetOptions.NumClasses);

    #region Episode

    /// <summary>A solved base learner: its coefficients and the constant that makes them differentiable.</summary>
    private readonly struct Solved
    {
        public Solved(Tensor<T> coefficients, Tensor<T> dual, bool isDual)
        {
            Coefficients = coefficients;
            Dual = dual;
            IsDual = isDual;
        }

        /// <summary>The classifier: dual coefficients <c>[support, classes]</c>, or weights <c>[classes, width]</c>.</summary>
        public Tensor<T> Coefficients { get; }

        /// <summary>The dual coefficients when the solver works in the dual, else the weights again.</summary>
        public Tensor<T> Dual { get; }

        /// <summary>Whether <see cref="Coefficients"/> are dual coefficients over the support set.</summary>
        public bool IsDual { get; }
    }

    /// <summary>
    /// One task's query loss and the exact gradient of that loss with respect to the embedding network and the logit
    /// scale.
    /// </summary>
    private (T Loss, Vector<T> Body, Vector<T> Scale) EpisodeGradient(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var episode = Episode(task);
        var stackedInput = ClassifierOutputs<T>.StackRows(task.SupportInput, task.QueryInput);
        var stackedTarget = ClassifierOutputs<T>.ToOutput<TOutput>(new Tensor<T>(new[] { episode.Rows, 1 }));
        var scaleTensor = Tensor<T>.FromVector(_logitScale);

        var composed = new EmbeddingObjectiveLoss<T>(embeddings =>
        {
            var engine = AiDotNetEngine.Current;
            var support = CheckWidth(engine.TensorMatMul(episode.SupportSelector, embeddings));
            var query = engine.TensorMatMul(episode.QuerySelector, embeddings);
            return QueryLoss(support, query, episode, scaleTensor);
        });
        var bodyGradient = ComputeGradients(MetaModel, stackedInput, stackedTarget, composed);

        var (fixedSupport, fixedQuery) = Embed(task, episode);
        T loss;
        var scaleGradient = new Vector<T>(1);
        using (var tape = new GradientTape<T>(new GradientTapeOptions { Persistent = true }))
        {
            var objective = QueryLoss(fixedSupport, fixedQuery, episode, scaleTensor);
            loss = objective[0];
            var gradients = tape.ComputeGradients(objective, new[] { scaleTensor });
            if (gradients.TryGetValue(scaleTensor, out var g)) scaleGradient[0] = g[0];
        }

        return (loss, bodyGradient, scaleGradient);
    }

    /// <summary>The task's query loss with the base learner solved on its support set (eq. 12).</summary>
    private Tensor<T> QueryLoss(Tensor<T> support, Tensor<T> query, PrototypeEpisode<T> episode, Tensor<T> scale)
    {
        var engine = AiDotNetEngine.Current;
        var solved = Solve(support, episode);
        var logits = solved.IsDual
            ? engine.TensorMatMul(engine.TensorMatMul(query, engine.TensorTranspose(support)), solved.Coefficients)
            : engine.TensorMatMul(query, engine.TensorTranspose(solved.Coefficients));
        logits = engine.TensorMultiply(logits, engine.Reshape(scale, new[] { 1, 1 }));
        return SmoothedLoss(logits, episode.QueryTarget, episode.ClassSlots.Length);
    }

    /// <summary>
    /// The configured loss of the logits against the class indices, or - with
    /// <see cref="MetaOptNetOptions{T,TInput,TOutput}.LabelSmoothing"/> - the cross-entropy against the smoothed
    /// label distribution the paper uses for miniImageNet.
    /// </summary>
    private Tensor<T> SmoothedLoss(Tensor<T> logits, Tensor<T> target, int classes)
    {
        double epsilon = _metaOptNetOptions.LabelSmoothing;
        if (epsilon <= 0) return Scalar(LossFunction.ComputeTapeLoss(logits, target));

        var engine = AiDotNetEngine.Current;
        var max = engine.ReduceMax(logits, new[] { 1 }, keepDims: true, out _);
        var shifted = engine.TensorAdd(logits, engine.TensorNegate(engine.StopGradient(max)));
        var logSum = engine.TensorLog(engine.ReduceSum(engine.TensorExp(shifted), new[] { 1 }, keepDims: true));
        var logProbabilities = engine.TensorAdd(shifted, engine.TensorNegate(logSum));

        var smoothed = new Tensor<T>(new[] { target.Length, classes });
        T off = Ops.FromDouble(epsilon / classes);
        T on = Ops.FromDouble(1.0 - epsilon + epsilon / classes);
        for (int r = 0; r < target.Length; r++)
        {
            int label = (int)Math.Round(Ops.ToDouble(target[r]));
            for (int c = 0; c < classes; c++) smoothed[r * classes + c] = c == label ? on : off;
        }

        var product = engine.TensorMultiply(logProbabilities, smoothed);
        return engine.TensorMultiplyScalar(
            Scalar(engine.ReduceSum(product, null)), Ops.FromDouble(-1.0 / Math.Max(1, target.Length)));
    }

    /// <summary>
    /// Solves the configured base learner on the support set and returns its solution as a tape tensor whose
    /// derivative is the implicit one.
    /// </summary>
    private Solved Solve(Tensor<T> support, PrototypeEpisode<T> episode)
    {
        var engine = AiDotNetEngine.Current;
        int n = support.Shape[0], d = support.Shape[1], classes = episode.ClassSlots.Length;
        var labels = SupportLabels(episode, n);
        var oneHot = new double[n, classes];
        for (int i = 0; i < n; i++) oneHot[i, labels[i]] = 1.0;

        if (_metaOptNetOptions.SolverType == ConvexSolverType.LogisticRegression)
        {
            var features = ToDoubles(support);
            double logisticLambda = _metaOptNetOptions.RegularizationStrength ?? _metaOptNetOptions.LogisticLambda;
            var (weights, hessianInverse) = ConvexBaseLearners.SolveLogistic(
                features, oneHot, logisticLambda,
                _metaOptNetOptions.MaxSolverIterations, _metaOptNetOptions.SolverTolerance);

            // F(W) = Z'(softmax(Z W') - Y) / n + lambda W, zero at the solution; W = W* - Hinv F(W; Z).
            var solution = FromDoubles(weights, classes, d);
            var probabilities = engine.Softmax(
                engine.TensorMatMul(support, engine.TensorTranspose(solution)), axis: 1);
            var residual = engine.TensorAdd(
                engine.TensorMultiplyScalar(
                    engine.TensorMatMul(engine.TensorTranspose(
                        engine.TensorAdd(probabilities, engine.TensorNegate(FromDoubles(oneHot, n, classes)))), support),
                    Ops.FromDouble(1.0 / n)),
                engine.TensorMultiplyScalar(solution, Ops.FromDouble(logisticLambda)));
            return new Solved(Implicit(solution, residual, hessianInverse, classes, d), solution, isDual: false);
        }

        var kernelTensor = engine.TensorMatMul(support, engine.TensorTranspose(support));
        var kernel = ToDoubles(kernelTensor);

        if (_metaOptNetOptions.SolverType == ConvexSolverType.SVM)
        {
            double cost = _metaOptNetOptions.RegularizationStrength ?? _metaOptNetOptions.SvmCost;
            var alpha = ConvexBaseLearners.SolveCrammerSingerDual(
                kernel, labels, classes, cost, _metaOptNetOptions.MaxSolverIterations, _metaOptNetOptions.SolverTolerance);
            var jacobianInverse = ConvexBaseLearners.CrammerSingerJacobianInverse(
                kernel, alpha, labels, classes, cost, _metaOptNetOptions.SolverTolerance);

            // Stationarity: (K alpha)_nk - [k = y_n] + lambda_nk + nu_n = 0, zero at the solution.
            var multipliers = StationarityConstant(kernel, alpha, labels, classes, cost);
            var solution = FromDoubles(alpha, n, classes);
            var residual = engine.TensorAdd(engine.TensorMatMul(kernelTensor, solution), FromDoubles(multipliers, n, classes));
            return new Solved(Implicit(solution, residual, jacobianInverse, n, classes), solution, isDual: true);
        }

        double ridge = _metaOptNetOptions.RegularizationStrength ?? _metaOptNetOptions.RidgeLambda;
        var (dual, inverse) = ConvexBaseLearners.SolveRidgeDual(kernel, oneHot, ridge);

        // (K + lambda I) M - Y = 0 at the solution.
        var dualTensor = FromDoubles(dual, n, classes);
        var ridgeResidual = engine.TensorAdd(
            engine.TensorAdd(engine.TensorMatMul(kernelTensor, dualTensor),
                engine.TensorMultiplyScalar(dualTensor, Ops.FromDouble(ridge))),
            engine.TensorNegate(FromDoubles(oneHot, n, classes)));

        // (K + lambda I) acts on each class column independently, so the implicit correction is that one [n, n]
        // inverse applied to the residual's columns - not a matrix over the flattened solution as the SVM and
        // logistic Jacobians are.
        var ridgeCorrection = engine.TensorMatMul(FromDoubles(inverse, n, n), ridgeResidual);
        return new Solved(
            engine.TensorAdd(dualTensor, engine.TensorNegate(ridgeCorrection)), dualTensor, isDual: true);
    }

    /// <summary>
    /// <c>theta* - Jinv vec(residual)</c>: the solution's value with the implicit function theorem's derivative.
    /// </summary>
    private static Tensor<T> Implicit(Tensor<T> solution, Tensor<T> residual, double[,] jacobianInverse, int rows, int columns)
    {
        var engine = AiDotNetEngine.Current;
        int size = rows * columns;
        var constant = new Tensor<T>(new[] { size, size });
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++) constant[i * size + j] = Ops.FromDouble(jacobianInverse[i, j]);

        var correction = engine.TensorMatMul(constant, engine.Reshape(residual, new[] { size, 1 }));
        return engine.TensorAdd(solution, engine.TensorNegate(engine.Reshape(correction, new[] { rows, columns })));
    }

    /// <summary>
    /// The parts of the SVM's stationarity residual that do not depend on the embeddings: <c>-onehot</c> plus the
    /// multipliers of the constraints active at the solution.
    /// </summary>
    private static double[,] StationarityConstant(double[,] kernel, double[,] alpha, int[] labels, int classes, double cost)
    {
        int n = labels.Length;
        var product = new double[n, classes];
        for (int i = 0; i < n; i++)
            for (int k = 0; k < classes; k++)
            {
                double sum = 0;
                for (int j = 0; j < n; j++) sum += kernel[i, j] * alpha[j, k];
                product[i, k] = sum - (k == labels[i] ? 1.0 : 0.0);
            }

        var constant = new double[n, classes];
        for (int i = 0; i < n; i++)
        {
            // nu_n comes from any entry off its bound; with every entry pinned the row's multiplier is free.
            double nu = 0;
            for (int k = 0; k < classes; k++)
            {
                double upper = k == labels[i] ? cost : 0.0;
                if (upper - alpha[i, k] > 1e-9) { nu = -product[i, k]; break; }
            }

            for (int k = 0; k < classes; k++)
            {
                double upper = k == labels[i] ? cost : 0.0;
                double lambda = upper - alpha[i, k] > 1e-9 ? 0.0 : -(product[i, k] + nu);
                constant[i, k] = -(k == labels[i] ? 1.0 : 0.0) + lambda + nu;
            }
        }

        return constant;
    }

    /// <summary>The primal weights <c>[classes * width]</c> of a solved base learner.</summary>
    private Vector<T> Primal(Tensor<T> coefficients, Tensor<T> support)
    {
        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        var weights = _metaOptNetOptions.SolverType == ConvexSolverType.LogisticRegression
            ? coefficients
            : engine.TensorMatMul(engine.TensorTranspose(coefficients), support);
        return weights.ToVector();
    }

    private PrototypeEpisode<T> Episode(IMetaLearningTask<T, TInput, TOutput> task)
        => PrototypeEpisode<T>.Build(ReadLabels(task.SupportOutput), ReadLabels(task.QueryOutput));

    private int[] ReadLabels(TOutput labels)
    {
        var tensor = ClassifierOutputs<T>.Labels(labels, _metaOptNetOptions.NumClasses);
        var indices = new int[tensor.Length];
        for (int i = 0; i < indices.Length; i++) indices[i] = (int)Math.Round(NumOps.ToDouble(tensor[i]));
        return indices;
    }

    private static int[] SupportLabels(PrototypeEpisode<T> episode, int rows)
    {
        var labels = new int[rows];
        for (int c = 0; c < episode.ClassSlots.Length; c++)
            for (int r = 0; r < rows; r++)
                if (Ops.ToDouble(episode.Membership[c * rows + r]) > 0.5) labels[r] = c;
        return labels;
    }

    private (Tensor<T> Support, Tensor<T> Query) Embed(IMetaLearningTask<T, TInput, TOutput> task, PrototypeEpisode<T> episode)
    {
        Tensor<T> embeddings;
        using (new NoGradScope<T>())
        {
            embeddings = ClassifierOutputs<T>.AsRows(
                MetaModel.Predict(ClassifierOutputs<T>.StackRows(task.SupportInput, task.QueryInput)));
        }

        var engine = AiDotNetEngine.Current;
        return (CheckWidth(engine.TensorMatMul(episode.SupportSelector, embeddings)),
            engine.TensorMatMul(episode.QuerySelector, embeddings));
    }

    private Tensor<T> CheckWidth(Tensor<T> embeddings)
    {
        var normalized = _metaOptNetOptions.NormalizeEmbeddings
            ? PrototypeMetric<T>.Normalized(embeddings, normalize: true)
            : embeddings;
        if (normalized.Shape[1] != _metaOptNetOptions.EmbeddingDimension)
        {
            throw new InvalidOperationException(
                $"The embedding network emits {normalized.Shape[1]}-wide embeddings per example but "
                + $"EmbeddingDimension is {_metaOptNetOptions.EmbeddingDimension}. EmbeddingDimension is the width "
                + "of the representation the base learner classifies - the network's per-example output width.");
        }

        return normalized;
    }

    #endregion

    #region Test hooks

    /// <summary>One task's query loss and exact gradient from the current state, for gradient checks.</summary>
    internal (T Loss, Vector<T> Body, Vector<T> Scale) EpisodeGradientForTesting(IMetaLearningTask<T, TInput, TOutput> task)
        => EpisodeGradient(task);

    /// <summary>One task's query loss from the current state.</summary>
    internal T EpisodeLossForTesting(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var episode = Episode(task);
        var (support, query) = Embed(task, episode);
        return QueryLoss(support, query, episode, Tensor<T>.FromVector(_logitScale))[0];
    }

    /// <summary>The solved base learner's coefficients for one task.</summary>
    internal Vector<T> CoefficientsForTesting(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var episode = Episode(task);
        var (support, _) = Embed(task, episode);
        using var noGrad = new NoGradScope<T>();
        return Solve(support, episode).Dual.ToVector();
    }

    /// <summary>Gets or sets the learned logit scale (for tests).</summary>
    internal Vector<T> LogitScaleForTesting
    {
        get { var copy = new Vector<T>(1); copy[0] = _logitScale[0]; return copy; }
        set { _logitScale = new Vector<T>(1); _logitScale[0] = value[0]; }
    }

    #endregion

    #region Helpers

    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    private static double[,] ToDoubles(Tensor<T> tensor)
    {
        int rows = tensor.Shape[0], columns = tensor.Shape[1];
        var values = new double[rows, columns];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++) values[i, j] = Ops.ToDouble(tensor[i * columns + j]);
        return values;
    }

    private static Tensor<T> FromDoubles(double[,] values, int rows, int columns)
    {
        var tensor = new Tensor<T>(new[] { rows, columns });
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++) tensor[i * columns + j] = Ops.FromDouble(values[i, j]);
        return tensor;
    }

    private static Tensor<T> Scalar(Tensor<T> value) => AiDotNetEngine.Current.Reshape(value, new[] { 1 });

    private static Vector<T> Accumulate(Vector<T>? sum, Vector<T> values)
    {
        if (sum is null)
        {
            var copy = new Vector<T>(values.Length);
            for (int i = 0; i < values.Length; i++) copy[i] = values[i];
            return copy;
        }

        for (int i = 0; i < sum.Length; i++) sum[i] = NumOps.Add(sum[i], values[i]);
        return sum;
    }

    private static Vector<T> Scale(Vector<T> values, T divisor)
    {
        for (int i = 0; i < values.Length; i++) values[i] = NumOps.Divide(values[i], divisor);
        return values;
    }

    #endregion
}

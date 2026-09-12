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
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Data.Structures;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of Almost No Inner Loop (ANIL) meta-learning algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// ANIL (Raghu et al. 2020) is MAML with the inner loop removed from every layer but the head: each task adapts
/// only the final linear classifier on its support set, while the body (the meta-model, which embeds each
/// example) is updated only by the outer loop. The outer loop meta-learns the body and the head's
/// initialisation together from the query loss after adaptation.
/// </para>
/// <para>
/// <b>Per example.</b> The body embeds every row of the input into a <see cref="ANILOptions{T,TInput,TOutput}.FeatureDimension"/>-wide
/// representation, and the head scores every row: <c>logits = h W^T + b</c>, one row of
/// <see cref="ANILOptions{T,TInput,TOutput}.NumClasses"/> scores per example. The loss is the configured loss on
/// those logits against one class index per example - cross-entropy by default, the paper's classification loss.
/// </para>
/// <para>
/// <b>Exact meta-gradient.</b> "We do not remove second order terms in ANIL (unlike in first-order MAML); second
/// order terms still persist through the derivative of the inner loop update for the head parameters" (Raghu et
/// al. 2020, App. C.1). The head's initialisation receives <c>prod_k (I - alpha H_k)</c> applied to the query
/// gradient, and the body receives, besides its direct query gradient, the cross term through each inner step's
/// support gradient. Both second-order pieces are Hessian-vector products taken as central differences of
/// gradients along the head direction, so they cost two gradient evaluations per inner step.
/// <see cref="ANILOptions{T,TInput,TOutput}.UseFirstOrder"/> drops them.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of a neural network as having two parts: a body that turns each example into a
/// description (an embedding), and a head that turns a description into a score for each class. ANIL keeps the
/// body fixed while it adapts to a new task and only retrains the small head; the body improves across tasks.
/// </para>
/// <para>
/// Reference: Raghu, A., Raghu, M., Bengio, S., &amp; Vinyals, O. (2020).
/// Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML",
    "https://arxiv.org/abs/1909.09157",
    Year = 2020,
    Authors = "Aniruddh Raghu, Maithra Raghu, Samy Bengio, Oriol Vinyals")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class ANILAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly ANILOptions<T, TInput, TOutput> _anilOptions;

    /// <summary>The head's meta-learned initial weights W, <c>[NumClasses x FeatureDimension]</c> row-major.</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _headWeights;

    /// <summary>The head's meta-learned initial bias b, <c>[NumClasses]</c>; empty when UseHeadBias is off.</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _headBias;

    /// <summary>
    /// Initializes a new instance of the ANILAlgorithm class.
    /// </summary>
    /// <param name="options">ANIL configuration options containing the model and all hyperparameters.</param>
    /// <exception cref="ArgumentNullException">Thrown when options is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the class count or feature width is not positive.</exception>
    /// <example>
    /// <code>
    /// // A body that embeds each example into 64 values, and a 5-way head over them
    /// var options = new ANILOptions&lt;double, Matrix&lt;double&gt;, Tensor&lt;double&gt;&gt;(body)
    /// {
    ///     NumClasses = 5,
    ///     FeatureDimension = 64
    /// };
    /// var anil = new ANILAlgorithm&lt;double, Matrix&lt;double&gt;, Tensor&lt;double&gt;&gt;(options);
    /// </code>
    /// </example>
    public ANILAlgorithm(ANILOptions<T, TInput, TOutput> options)
        : base(
            options?.MetaModel ?? throw new ArgumentNullException(nameof(options), "MetaModel must be set in options."),
            options.LossFunction ?? new CrossEntropyWithLogitsLoss<T>(),
            options,
            options.DataLoader,
            options.MetaOptimizer,
            options.InnerOptimizer)
    {
        _anilOptions = options;
        if (options.NumClasses <= 0)
            throw new ArgumentException("NumClasses must be positive.", nameof(options));
        if (options.FeatureDimension <= 0)
            throw new ArgumentException("FeatureDimension must be positive.", nameof(options));

        _headWeights = InitializeHeadWeights();
        _headBias = new Vector<T>(options.UseHeadBias ? options.NumClasses : 0);
    }

    /// <summary>
    /// Gets the algorithm type identifier for this meta-learner.
    /// </summary>
    /// <value>Returns <see cref="MetaLearningAlgorithmType.ANIL"/>.</value>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.ANIL;

    /// <summary>
    /// Performs one meta-training step: head-only adaptation per task, then one update of the body and the head's
    /// initialisation from the query losses.
    /// </summary>
    /// <param name="taskBatch">A batch of tasks to meta-train on, each containing support and query sets.</param>
    /// <returns>The average query loss across the batch, after adaptation.</returns>
    /// <exception cref="ArgumentException">Thrown when the task batch is null or empty.</exception>
    /// <remarks>
    /// <para>
    /// This used to treat each task's whole support set as one example (the body's flattened output was the
    /// "feature vector"), estimate head gradients by one-sided finite differences, and compute the body gradient by
    /// writing the head's weights into the tail of the body's own parameter vector and differentiating the
    /// configured loss on the body's raw output - so the head never reached the body's objective, and the body's
    /// last parameters were overwritten by head weights on every step. <see cref="ANILOptions{T,TInput,TOutput}.UseFirstOrder"/>
    /// was ignored.
    /// </para>
    /// </remarks>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null || taskBatch.BatchSize == 0)
        {
            throw new ArgumentException("Task batch cannot be null or empty.", nameof(taskBatch));
        }

        var body = ParamModel.GetParameters();
        double alpha = _anilOptions.InnerLearningRate;
        Vector<T>? bodyGradient = null;
        Vector<T>? headWeightGradient = null;
        Vector<T>? headBiasGradient = null;
        T totalLoss = NumOps.Zero;

        foreach (var task in taskBatch.Tasks)
        {
            var (queryLoss, taskBodyGradient, lambdaWeights, lambdaBias) = TaskMetaGradient(task, alpha);
            totalLoss = NumOps.Add(totalLoss, queryLoss);
            bodyGradient = Accumulate(bodyGradient, taskBodyGradient);
            headWeightGradient = Accumulate(headWeightGradient, lambdaWeights);
            headBiasGradient = Accumulate(headBiasGradient, lambdaBias);
        }

        T batchSize = NumOps.FromDouble(taskBatch.BatchSize);
        bodyGradient = Scale(bodyGradient ?? new Vector<T>(body.Length), batchSize);
        headWeightGradient = Scale(headWeightGradient ?? new Vector<T>(_headWeights.Length), batchSize);
        headBiasGradient = Scale(headBiasGradient ?? new Vector<T>(_headBias.Length), batchSize);

        if (_anilOptions.GradientClipThreshold.HasValue && _anilOptions.GradientClipThreshold.Value > 0)
        {
            bodyGradient = ClipGradients(bodyGradient, _anilOptions.GradientClipThreshold.Value);
            headWeightGradient = ClipGradients(headWeightGradient, _anilOptions.GradientClipThreshold.Value);
            if (headBiasGradient.Length > 0)
                headBiasGradient = ClipGradients(headBiasGradient, _anilOptions.GradientClipThreshold.Value);
        }

        ParamModel.SetParameters(ApplyGradients(body, bodyGradient, _anilOptions.OuterLearningRate));
        _headWeights = ApplyGradients(_headWeights, headWeightGradient, _anilOptions.OuterLearningRate);
        if (_headBias.Length > 0)
            _headBias = ApplyGradients(_headBias, headBiasGradient, _anilOptions.OuterLearningRate);

        return NumOps.Divide(totalLoss, batchSize);
    }

    /// <summary>
    /// One task's query loss after head-only adaptation, and the exact gradient of that loss with respect to the body
    /// and to the head's initialisation.
    /// </summary>
    private (T Loss, Vector<T> Body, Vector<T> Weights, Vector<T> Bias) TaskMetaGradient(
        IMetaLearningTask<T, TInput, TOutput> task, double alpha)
    {
        var supportLabels = ClassifierOutputs<T>.Labels(task.SupportOutput, _anilOptions.NumClasses);
        var queryLabels = ClassifierOutputs<T>.Labels(task.QueryOutput, _anilOptions.NumClasses);
        var supportEmbeddings = Embed(MetaModel, task.SupportInput);
        var queryEmbeddings = Embed(MetaModel, task.QueryInput);

        var (weights, bias, trace) = AdaptHead(
            supportEmbeddings, supportLabels, _headWeights, _headBias, _anilOptions.AdaptationSteps,
            record: !_anilOptions.UseFirstOrder);

        // Query loss after adaptation, and its gradient with respect to the adapted head. The head's L2 penalty
        // regularises adaptation only; it is not part of the meta-objective.
        var (queryLoss, lambdaWeights, lambdaBias) = HeadLossAndGradient(
            queryEmbeddings, queryLabels, weights, bias, regularize: false);

        var bodyGradient = BodyGradient(task.QueryInput, task.QueryOutput, weights, bias);

        if (!_anilOptions.UseFirstOrder)
        {
            // Walk the inner loop back. head_{k+1} = head_k - alpha grad_head L_s(head_k, body), so the adjoint lambda
            // of head_{k+1} sends -alpha * d/dbody (grad_head L_s . lambda) to the body and (I - alpha H_k) lambda back
            // to head_k. Both are central differences along lambda.
            for (int k = trace.Count - 1; k >= 0; k--)
            {
                var (stepWeights, stepBias) = trace[k];
                double eps = DifferenceStep(stepWeights, stepBias, lambdaWeights, lambdaBias);
                if (eps == 0) break;

                var plusWeights = Shift(stepWeights, lambdaWeights, eps);
                var plusBias = Shift(stepBias, lambdaBias, eps);
                var minusWeights = Shift(stepWeights, lambdaWeights, -eps);
                var minusBias = Shift(stepBias, lambdaBias, -eps);

                var bodyPlus = BodyGradient(task.SupportInput, task.SupportOutput, plusWeights, plusBias);
                var bodyMinus = BodyGradient(task.SupportInput, task.SupportOutput, minusWeights, minusBias);
                bodyGradient = Combine(bodyGradient, bodyPlus, bodyMinus, -alpha / (2 * eps));

                var (_, headPlusWeights, headPlusBias) = HeadLossAndGradient(
                    supportEmbeddings, supportLabels, plusWeights, plusBias, regularize: true);
                var (_, headMinusWeights, headMinusBias) = HeadLossAndGradient(
                    supportEmbeddings, supportLabels, minusWeights, minusBias, regularize: true);
                lambdaWeights = Combine(lambdaWeights, headPlusWeights, headMinusWeights, -alpha / (2 * eps));
                lambdaBias = Combine(lambdaBias, headPlusBias, headMinusBias, -alpha / (2 * eps));
            }
        }

        return (queryLoss, bodyGradient, lambdaWeights, lambdaBias);
    }

    /// <summary>One task's query loss and exact meta-gradient, for gradient checks.</summary>
    internal (T Loss, Vector<T> Body, Vector<T> Weights, Vector<T> Bias) TaskMetaGradientForTesting(
        IMetaLearningTask<T, TInput, TOutput> task)
        => TaskMetaGradient(task, _anilOptions.InnerLearningRate);

    /// <summary>One task's query loss after adaptation from the current body and head initialisation.</summary>
    internal T TaskLossForTesting(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var (weights, bias, _) = AdaptHead(
            Embed(MetaModel, task.SupportInput),
            ClassifierOutputs<T>.Labels(task.SupportOutput, _anilOptions.NumClasses),
            _headWeights, _headBias, _anilOptions.AdaptationSteps, record: false);
        return HeadLossAndGradient(
            Embed(MetaModel, task.QueryInput),
            ClassifierOutputs<T>.Labels(task.QueryOutput, _anilOptions.NumClasses),
            weights, bias, regularize: false).Loss;
    }

    /// <summary>Gets or sets a copy of the head's initial weights (for tests).</summary>
    internal Vector<T> HeadWeightsForTesting
    {
        get => Copy(_headWeights);
        set => _headWeights = Copy(value);
    }

    /// <summary>Gets or sets a copy of the head's initial bias (for tests).</summary>
    internal Vector<T> HeadBiasForTesting
    {
        get => Copy(_headBias);
        set => _headBias = Copy(value);
    }

    /// <summary>
    /// Adapts the meta-learned model to a new task by only updating the classification head.
    /// </summary>
    /// <param name="task">The new task containing support set examples for adaptation.</param>
    /// <returns>A model with its own copy of the body and the head adapted to this task.</returns>
    /// <exception cref="ArgumentNullException">Thrown when task is null.</exception>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null)
        {
            throw new ArgumentNullException(nameof(task));
        }

        var body = CloneModel();
        var initialWeights = _anilOptions.ReinitializeHead ? InitializeHeadWeights() : Copy(_headWeights);
        var initialBias = _anilOptions.ReinitializeHead ? new Vector<T>(_headBias.Length) : Copy(_headBias);

        var (weights, bias, _) = AdaptHead(
            Embed(body, task.SupportInput),
            ClassifierOutputs<T>.Labels(task.SupportOutput, _anilOptions.NumClasses),
            initialWeights, initialBias, _anilOptions.AdaptationSteps, record: false);

        return new ANILModel<T, TInput, TOutput>(body, weights, bias.Length > 0 ? bias : null, _anilOptions);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// The configured loss on <c>[rows, NumClasses]</c> scores against one class index per example. The base's
    /// version flattened both to vectors, which for cross-entropy scores the whole batch as ONE distribution.
    /// </remarks>
    protected override T ComputeLossFromOutput(TOutput predictions, TOutput expectedOutput)
    {
        var labels = ClassifierOutputs<T>.Labels(expectedOutput, _anilOptions.NumClasses);
        var scores = ClassifierOutputs<T>.ScoreRows(predictions, labels.Length);
        using var noGrad = new NoGradScope<T>();
        return LossFunction.ComputeTapeLoss(scores, labels)[0];
    }

    #region Head

    /// <summary>The body's per-example embeddings, <c>[rows, FeatureDimension]</c>.</summary>
    private Tensor<T> Embed(IFullModel<T, TInput, TOutput> body, TInput input)
    {
        Tensor<T> embeddings;
        using (new NoGradScope<T>())
        {
            embeddings = ClassifierOutputs<T>.AsRows(body.Predict(input));
        }

        if (embeddings.Shape[1] != _anilOptions.FeatureDimension)
        {
            throw new InvalidOperationException(
                $"The body emits {embeddings.Shape[1]}-wide embeddings per example but FeatureDimension is "
                + $"{_anilOptions.FeatureDimension}. FeatureDimension is the width of the representation the head "
                + "reads - the body's per-example output width.");
        }

        return embeddings;
    }

    private Tensor<T> WeightsTensor(Vector<T> weights)
        => Tensor<T>.FromVector(weights).Reshape(_anilOptions.NumClasses, _anilOptions.FeatureDimension);

    private static Tensor<T>? BiasTensor(Vector<T> bias) => bias.Length > 0 ? Tensor<T>.FromVector(bias) : null;

    /// <summary>
    /// The configured loss of the head on fixed embeddings, and its exact gradient with respect to W and b.
    /// </summary>
    /// <remarks>
    /// dL/dlogits comes from differentiating the loss on the tape, so any configured loss works; the chain rule
    /// through <c>logits = h W^T + b</c> is then exact. It replaces a one-sided finite difference per head weight.
    /// </remarks>
    private (T Loss, Vector<T> Weights, Vector<T> Bias) HeadLossAndGradient(
        Tensor<T> embeddings, Tensor<T> labels, Vector<T> weights, Vector<T> bias, bool regularize)
    {
        Tensor<T> logits;
        using (new NoGradScope<T>())
        {
            logits = EmbeddingClassificationLoss<T>.LinearHead(embeddings, WeightsTensor(weights), BiasTensor(bias));
        }

        var (loss, logitGradient) = LossFunctionExtensions.ComputeLossAndGradient(LossFunction, logits, labels);

        int rows = embeddings.Shape[0];
        int classes = _anilOptions.NumClasses;
        int width = _anilOptions.FeatureDimension;
        var weightGradient = new Vector<T>(weights.Length);
        var biasGradient = new Vector<T>(bias.Length);
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < classes; c++)
            {
                T g = logitGradient[r * classes + c];
                for (int f = 0; f < width; f++)
                {
                    weightGradient[c * width + f] = NumOps.Add(
                        weightGradient[c * width + f], NumOps.Multiply(g, embeddings[r * width + f]));
                }

                if (bias.Length > 0) biasGradient[c] = NumOps.Add(biasGradient[c], g);
            }
        }

        if (regularize && _anilOptions.HeadL2Regularization > 0)
        {
            T strength = NumOps.FromDouble(_anilOptions.HeadL2Regularization);
            T penalty = NumOps.Zero;
            for (int i = 0; i < weights.Length; i++)
            {
                penalty = NumOps.Add(penalty, NumOps.Multiply(weights[i], weights[i]));
                weightGradient[i] = NumOps.Add(weightGradient[i], NumOps.Multiply(strength, weights[i]));
            }

            loss = NumOps.Add(loss, NumOps.Multiply(NumOps.FromDouble(0.5 * _anilOptions.HeadL2Regularization), penalty));
        }

        return (loss, weightGradient, biasGradient);
    }

    /// <summary>
    /// The body's gradient of the configured loss through a fixed head: the head is composed into the loss the
    /// body differentiates, so the chain rule runs through it to every example's embedding.
    /// </summary>
    private Vector<T> BodyGradient(TInput input, TOutput labels, Vector<T> weights, Vector<T> bias)
    {
        var headWeights = WeightsTensor(weights);
        var headBias = BiasTensor(bias);
        var composed = new EmbeddingClassificationLoss<T>(
            embeddings => EmbeddingClassificationLoss<T>.LinearHead(embeddings, headWeights, headBias), LossFunction);
        return ComputeGradients(MetaModel, input, labels, composed);
    }

    /// <summary>Head-only inner loop: SGD on the support loss, optionally recording each step's starting head.</summary>
    private (Vector<T> Weights, Vector<T> Bias, List<(Vector<T> Weights, Vector<T> Bias)> Trace) AdaptHead(
        Tensor<T> supportEmbeddings, Tensor<T> supportLabels, Vector<T> initialWeights, Vector<T> initialBias,
        int steps, bool record)
    {
        var weights = Copy(initialWeights);
        var bias = Copy(initialBias);
        var trace = new List<(Vector<T> Weights, Vector<T> Bias)>(record ? steps : 0);
        for (int step = 0; step < steps; step++)
        {
            if (record) trace.Add((weights, bias));
            var (_, weightGradient, biasGradient) = HeadLossAndGradient(
                supportEmbeddings, supportLabels, weights, bias, regularize: true);
            weights = ApplyGradients(weights, weightGradient, _anilOptions.InnerLearningRate);
            if (bias.Length > 0) bias = ApplyGradients(bias, biasGradient, _anilOptions.InnerLearningRate);
        }

        return (weights, bias, trace);
    }

    /// <summary>
    /// The central-difference step along the head direction: small relative to the head, and zero when there is
    /// no direction to differentiate along.
    /// </summary>
    private static double DifferenceStep(Vector<T> weights, Vector<T> bias, Vector<T> directionWeights, Vector<T> directionBias)
    {
        double directionNorm = 0, headNorm = 0;
        for (int i = 0; i < weights.Length; i++)
        {
            double d = NumOps.ToDouble(directionWeights[i]);
            double w = NumOps.ToDouble(weights[i]);
            directionNorm += d * d;
            headNorm += w * w;
        }

        for (int i = 0; i < bias.Length; i++)
        {
            double d = NumOps.ToDouble(directionBias[i]);
            double b = NumOps.ToDouble(bias[i]);
            directionNorm += d * d;
            headNorm += b * b;
        }

        directionNorm = Math.Sqrt(directionNorm);
        return directionNorm == 0 ? 0 : 1e-5 * (1.0 + Math.Sqrt(headNorm)) / directionNorm;
    }

    private static Vector<T> Shift(Vector<T> values, Vector<T> direction, double step)
    {
        var shifted = new Vector<T>(values.Length);
        for (int i = 0; i < values.Length; i++)
        {
            shifted[i] = NumOps.Add(values[i], NumOps.Multiply(NumOps.FromDouble(step), direction[i]));
        }

        return shifted;
    }

    /// <summary><c>baseline + scale * (plus - minus)</c>.</summary>
    private static Vector<T> Combine(Vector<T> baseline, Vector<T> plus, Vector<T> minus, double scale)
    {
        var combined = new Vector<T>(baseline.Length);
        T s = NumOps.FromDouble(scale);
        for (int i = 0; i < baseline.Length; i++)
        {
            combined[i] = NumOps.Add(baseline[i], NumOps.Multiply(s, NumOps.Subtract(plus[i], minus[i])));
        }

        return combined;
    }

    private static Vector<T> Accumulate(Vector<T>? sum, Vector<T> values)
    {
        if (sum is null) return Copy(values);
        for (int i = 0; i < sum.Length; i++) sum[i] = NumOps.Add(sum[i], values[i]);
        return sum;
    }

    private static Vector<T> Scale(Vector<T> values, T divisor)
    {
        for (int i = 0; i < values.Length; i++) values[i] = NumOps.Divide(values[i], divisor);
        return values;
    }

    private static Vector<T> Copy(Vector<T> source)
    {
        var copy = new Vector<T>(source.Length);
        for (int i = 0; i < source.Length; i++) copy[i] = source[i];
        return copy;
    }

    /// <summary>
    /// Initial head weights, uniform in <c>[-s, s]</c> with <c>s = sqrt(2 / FeatureDimension)</c>.
    /// </summary>
    private Vector<T> InitializeHeadWeights()
    {
        int size = _anilOptions.FeatureDimension * _anilOptions.NumClasses;
        var weights = new Vector<T>(size);
        double scale = Math.Sqrt(2.0 / _anilOptions.FeatureDimension);
        for (int i = 0; i < size; i++)
        {
            weights[i] = NumOps.FromDouble((RandomGenerator.NextDouble() * 2 - 1) * scale);
        }

        return weights;
    }

    #endregion
}

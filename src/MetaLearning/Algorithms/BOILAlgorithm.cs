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
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Data.Structures;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of BOIL (Body Only update in Inner Loop) meta-learning algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// BOIL (Oh et al., ICLR 2021) is ANIL's mirror image: each task adapts only the body - the meta-model, which
/// embeds every example - while the head (the final linear classifier) stays frozen in the inner loop.
/// "MAML usually sets alpha = alpha_b = alpha_h (!= 0), ANIL sets alpha_b = 0 and alpha_h != 0, and BOIL sets
/// alpha_b != 0 and alpha_h = 0" (Sec. 3.2); the outer loop meta-updates body and head together. The frozen head
/// forces representation change: to solve a new task the body must move the features, not the head the
/// decision boundary.
/// </para>
/// <para>
/// <b>Per example, exact.</b> The body embeds each row into a FeatureDimension-wide representation and the head
/// scores each row. The inner loop steps the body on the support loss through the frozen head; the outer loop
/// differentiates the query loss through every inner step - body Hessian-vector products, and the head's cross
/// term through each step's support gradient - taken as central differences of gradients, two evaluations per
/// step. UseFirstOrder drops the second-order terms.
/// </para>
/// <para>
/// <b>Beyond the paper, off by default.</b> <see cref="BOILOptions{T,TInput,TOutput}.BodyAdaptationFraction"/>
/// adapts only the top fraction of the body's layers - BOIL's own analysis finds representation change in the
/// high-level body and reuse in the low and middle layers. <see cref="BOILOptions{T,TInput,TOutput}.UseLayerwiseLearningRates"/>
/// meta-learns one inner learning rate per body layer and per inner step, MAML++'s LSLR (Antoniou et al. 2019),
/// on the same exact meta-gradient.
/// </para>
/// <para>
/// <b>For Beginners:</b> BOIL keeps the "decision maker" (the head) fixed while it adapts to a new task and
/// re-tunes the "feature extractor" (the body) instead, so each task teaches the model new ways of looking at
/// its inputs.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("BOIL: Towards Representation Change for Few-shot Learning",
    "https://arxiv.org/abs/2008.08882",
    Year = 2021,
    Authors = "Jaehoon Oh, Hyungjun Yoo, ChangHwan Kim, Se-Young Yun")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class BOILAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly BOILOptions<T, TInput, TOutput> _boilOptions;

    /// <summary>The head's weights W, <c>[NumClasses x FeatureDimension]</c> row-major: frozen per task, meta-learned.</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _headWeights;

    /// <summary>The head's bias b, <c>[NumClasses]</c>.</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _headBias;

    /// <summary>
    /// Learned inner learning rates, one per body layer and inner step (<c>[AdaptationSteps x layers]</c>); empty
    /// when UseLayerwiseLearningRates is off.
    /// </summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _layerStepRates;

    /// <summary>The body's layers as contiguous ranges of its flat parameter vector.</summary>
    private readonly (int Offset, int Count)[] _layers;

    /// <summary>Which of <see cref="_layers"/> the inner loop adapts: the top BodyAdaptationFraction of them.</summary>
    private readonly bool[] _adaptedLayers;

    /// <summary>
    /// Initializes a new instance of the BOILAlgorithm class.
    /// </summary>
    /// <param name="options">BOIL configuration options containing the model and all hyperparameters.</param>
    /// <exception cref="ArgumentNullException">Thrown when options is null.</exception>
    /// <exception cref="ArgumentException">Thrown when a size or the adaptation fraction is out of range.</exception>
    public BOILAlgorithm(BOILOptions<T, TInput, TOutput> options)
        : base(
            options?.MetaModel ?? throw new ArgumentNullException(nameof(options), "MetaModel must be set in options."),
            options.LossFunction ?? new CrossEntropyWithLogitsLoss<T>(),
            options,
            options.DataLoader,
            options.MetaOptimizer,
            options.InnerOptimizer)
    {
        _boilOptions = options;
        if (options.NumClasses <= 0)
            throw new ArgumentException("NumClasses must be positive.", nameof(options));
        if (options.FeatureDimension <= 0)
            throw new ArgumentException("FeatureDimension must be positive.", nameof(options));
        if (options.BodyAdaptationFraction <= 0 || options.BodyAdaptationFraction > 1)
            throw new ArgumentException("BodyAdaptationFraction must lie in (0, 1].", nameof(options));

        _headWeights = InitializeHeadWeights();
        _headBias = new Vector<T>(options.NumClasses);
        _layers = BodyLayers(options.MetaModel, InterfaceGuard.Parameterizable(options.MetaModel).GetParameters().Length);
        _adaptedLayers = TopLayers(_layers.Length, options.BodyAdaptationFraction);
        _layerStepRates = options.UseLayerwiseLearningRates ? InitialLayerStepRates() : new Vector<T>(0);
    }

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.BOIL;

    /// <summary>
    /// Performs one meta-training step: body-only adaptation per task, then one update of the body, the head and
    /// (when learned) the per-layer rates from the query losses.
    /// </summary>
    /// <param name="taskBatch">A batch of tasks to meta-train on, each containing support and query sets.</param>
    /// <returns>The average query loss across the batch, after adaptation.</returns>
    /// <exception cref="ArgumentException">Thrown when the task batch is null or empty.</exception>
    /// <remarks>
    /// This used to treat each task's whole batch as one example, estimate body gradients by one-sided finite
    /// differences on at most 100 sampled parameters scaled up to stand for the rest, estimate head gradients the
    /// same way, and compute its "second-order" gradient by re-running the whole adaptation once per sampled
    /// parameter. Its layer-wise rates assumed the first half of the flat parameter vector was the early layers.
    /// </remarks>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null || taskBatch.BatchSize == 0)
        {
            throw new ArgumentException("Task batch cannot be null or empty.", nameof(taskBatch));
        }

        var metaBody = ParamModel.GetParameters();
        Vector<T>? bodyGradient = null, weightGradient = null, biasGradient = null, rateGradient = null;
        T totalLoss = NumOps.Zero;
        try
        {
            foreach (var task in taskBatch.Tasks)
            {
                var (loss, body, weights, bias, rates) = TaskMetaGradient(task, metaBody);
                totalLoss = NumOps.Add(totalLoss, loss);
                bodyGradient = Accumulate(bodyGradient, body);
                weightGradient = Accumulate(weightGradient, weights);
                biasGradient = Accumulate(biasGradient, bias);
                rateGradient = Accumulate(rateGradient, rates);
            }
        }
        finally
        {
            ParamModel.SetParameters(metaBody);
        }

        T batchSize = NumOps.FromDouble(taskBatch.BatchSize);
        bodyGradient = Scale(bodyGradient ?? new Vector<T>(metaBody.Length), batchSize);
        weightGradient = Scale(weightGradient ?? new Vector<T>(_headWeights.Length), batchSize);
        biasGradient = Scale(biasGradient ?? new Vector<T>(_headBias.Length), batchSize);
        rateGradient = Scale(rateGradient ?? new Vector<T>(_layerStepRates.Length), batchSize);

        if (_boilOptions.GradientClipThreshold.HasValue && _boilOptions.GradientClipThreshold.Value > 0)
        {
            double threshold = _boilOptions.GradientClipThreshold.Value;
            bodyGradient = ClipGradients(bodyGradient, threshold);
            weightGradient = ClipGradients(weightGradient, threshold);
            biasGradient = ClipGradients(biasGradient, threshold);
            if (rateGradient.Length > 0) rateGradient = ClipGradients(rateGradient, threshold);
        }

        double beta = _boilOptions.OuterLearningRate;
        ParamModel.SetParameters(ApplyGradients(metaBody, bodyGradient, beta));
        _headWeights = ApplyGradients(_headWeights, weightGradient, beta);
        _headBias = ApplyGradients(_headBias, biasGradient, beta);
        if (_layerStepRates.Length > 0) _layerStepRates = ApplyGradients(_layerStepRates, rateGradient, beta);

        return NumOps.Divide(totalLoss, batchSize);
    }

    /// <summary>
    /// One task's query loss after body-only adaptation, and the exact gradient of that loss with respect to the
    /// body's initialisation, the head, and the per-layer per-step rates.
    /// </summary>
    private (T Loss, Vector<T> Body, Vector<T> Weights, Vector<T> Bias, Vector<T> Rates) TaskMetaGradient(
        IMetaLearningTask<T, TInput, TOutput> task, Vector<T> metaBody)
    {
        var supportLabels = ClassifierOutputs<T>.Labels(task.SupportOutput, _boilOptions.NumClasses);
        var queryLabels = ClassifierOutputs<T>.Labels(task.QueryOutput, _boilOptions.NumClasses);

        var (adapted, trace) = AdaptBody(metaBody, task.SupportInput, task.SupportOutput, record: true);

        var (loss, weights, bias) = HeadLossAndGradient(Embed(adapted, task.QueryInput), queryLabels, _headWeights, _headBias);
        var lambda = BodyGradient(adapted, task.QueryInput, task.QueryOutput, regularize: false);
        var rates = new Vector<T>(_layerStepRates.Length);

        // Walk the inner loop back. body_{k+1} = body_k - A_k o g_k with g_k the support gradient; the adjoint lambda
        // of body_{k+1} gives each learned rate -sum_layer(g_k o lambda), sends -d/dhead (g_k . A_k o lambda) to the
        // head, and (I - H_k A_k) lambda back to body_k.
        for (int k = trace.Count - 1; k >= 0; k--)
        {
            var (stepBody, stepGradient) = trace[k];
            if (_layerStepRates.Length > 0)
            {
                for (int l = 0; l < _layers.Length; l++)
                {
                    if (!_adaptedLayers[l]) continue;
                    var (offset, count) = _layers[l];
                    T dot = NumOps.Zero;
                    for (int i = offset; i < offset + count; i++)
                    {
                        dot = NumOps.Add(dot, NumOps.Multiply(stepGradient[i], lambda[i]));
                    }

                    rates[k * _layers.Length + l] = NumOps.Negate(dot);
                }
            }

            if (_boilOptions.UseFirstOrder) continue;

            var direction = Hadamard(RateVector(k), lambda);
            double eps = DifferenceStep(stepBody, direction);
            if (eps == 0) continue;

            var plus = Shift(stepBody, direction, eps);
            var minus = Shift(stepBody, direction, -eps);

            var (_, headPlusWeights, headPlusBias) = HeadLossAndGradient(
                Embed(plus, task.SupportInput), supportLabels, _headWeights, _headBias);
            var (_, headMinusWeights, headMinusBias) = HeadLossAndGradient(
                Embed(minus, task.SupportInput), supportLabels, _headWeights, _headBias);
            weights = Combine(weights, headPlusWeights, headMinusWeights, -1.0 / (2 * eps));
            bias = Combine(bias, headPlusBias, headMinusBias, -1.0 / (2 * eps));

            var bodyPlus = BodyGradient(plus, task.SupportInput, task.SupportOutput, regularize: true);
            var bodyMinus = BodyGradient(minus, task.SupportInput, task.SupportOutput, regularize: true);
            lambda = Combine(lambda, bodyPlus, bodyMinus, -1.0 / (2 * eps));
        }

        return (loss, lambda, weights, bias, rates);
    }

    /// <summary>
    /// Adapts the meta-learned model to a new task by only updating the body.
    /// </summary>
    /// <param name="task">The new task containing support set examples for adaptation.</param>
    /// <returns>A model with its own copy of the adapted body and the frozen head.</returns>
    /// <exception cref="ArgumentNullException">Thrown when task is null.</exception>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null)
        {
            throw new ArgumentNullException(nameof(task));
        }

        var metaBody = ParamModel.GetParameters();
        try
        {
            var (adapted, _) = AdaptBody(metaBody, task.SupportInput, task.SupportOutput, record: false);
            return new BOILModel<T, TInput, TOutput>(
                MetaModel, adapted, CloneVector(_headWeights), CloneVector(_headBias), _boilOptions);
        }
        finally
        {
            ParamModel.SetParameters(metaBody);
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// The configured loss on <c>[rows, NumClasses]</c> scores against one class index per example; the base's
    /// version flattened both to vectors, which for cross-entropy scores the whole batch as ONE distribution.
    /// </remarks>
    protected override T ComputeLossFromOutput(TOutput predictions, TOutput expectedOutput)
    {
        var labels = ClassifierOutputs<T>.Labels(expectedOutput, _boilOptions.NumClasses);
        var scores = ClassifierOutputs<T>.ScoreRows(predictions, labels.Length);
        using var noGrad = new NoGradScope<T>();
        return LossFunction.ComputeTapeLoss(scores, labels)[0];
    }

    #region Test hooks

    /// <summary>One task's query loss and exact meta-gradient from the current state, for gradient checks.</summary>
    internal (T Loss, Vector<T> Body, Vector<T> Weights, Vector<T> Bias, Vector<T> Rates) TaskMetaGradientForTesting(
        IMetaLearningTask<T, TInput, TOutput> task)
    {
        var metaBody = ParamModel.GetParameters();
        try
        {
            return TaskMetaGradient(task, metaBody);
        }
        finally
        {
            ParamModel.SetParameters(metaBody);
        }
    }

    /// <summary>One task's query loss after adaptation from the current state.</summary>
    internal T TaskLossForTesting(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var metaBody = ParamModel.GetParameters();
        try
        {
            var (adapted, _) = AdaptBody(metaBody, task.SupportInput, task.SupportOutput, record: false);
            return HeadLossAndGradient(
                Embed(adapted, task.QueryInput),
                ClassifierOutputs<T>.Labels(task.QueryOutput, _boilOptions.NumClasses),
                _headWeights, _headBias).Loss;
        }
        finally
        {
            ParamModel.SetParameters(metaBody);
        }
    }

    /// <summary>Gets or sets a copy of the head's weights (for tests).</summary>
    internal Vector<T> HeadWeightsForTesting { get => CloneVector(_headWeights); set => _headWeights = CloneVector(value); }

    /// <summary>Gets or sets a copy of the head's bias (for tests).</summary>
    internal Vector<T> HeadBiasForTesting { get => CloneVector(_headBias); set => _headBias = CloneVector(value); }

    /// <summary>Gets or sets a copy of the learned per-layer per-step rates (for tests).</summary>
    internal Vector<T> LayerStepRatesForTesting { get => CloneVector(_layerStepRates); set => _layerStepRates = CloneVector(value); }

    #endregion

    #region Body

    /// <summary>
    /// Body-only inner loop: SGD on the support loss through the frozen head, from <paramref name="start"/>. With
    /// <paramref name="record"/> it keeps each step's starting body and support gradient for the reverse pass.
    /// </summary>
    private (Vector<T> Body, List<(Vector<T> Body, Vector<T> Gradient)> Trace) AdaptBody(
        Vector<T> start, TInput supportInput, TOutput supportLabels, bool record)
    {
        var body = CloneVector(start);
        var trace = new List<(Vector<T> Body, Vector<T> Gradient)>(record ? _boilOptions.AdaptationSteps : 0);
        for (int k = 0; k < _boilOptions.AdaptationSteps; k++)
        {
            var gradient = BodyGradient(body, supportInput, supportLabels, regularize: true);
            if (record) trace.Add((body, gradient));
            body = Subtract(body, Hadamard(RateVector(k), gradient));
        }

        return (body, trace);
    }

    /// <summary>
    /// The body's gradient of the configured loss through the head at the given body parameters, plus the body's
    /// L2 penalty when asked. The head is composed into the loss the body differentiates.
    /// </summary>
    private Vector<T> BodyGradient(Vector<T> body, TInput input, TOutput labels, bool regularize)
    {
        ParamModel.SetParameters(body);
        var weights = WeightsTensor(_headWeights);
        var bias = Tensor<T>.FromVector(_headBias);
        var composed = new EmbeddingClassificationLoss<T>(
            embeddings => EmbeddingClassificationLoss<T>.LinearHead(CheckWidth(embeddings), weights, bias), LossFunction);
        var gradient = ComputeGradients(MetaModel, input, labels, composed);

        if (regularize && _boilOptions.BodyL2Regularization > 0)
        {
            T strength = NumOps.FromDouble(_boilOptions.BodyL2Regularization);
            for (int i = 0; i < gradient.Length; i++)
            {
                gradient[i] = NumOps.Add(gradient[i], NumOps.Multiply(strength, body[i]));
            }
        }

        return gradient;
    }

    /// <summary>The body's per-example embeddings at the given body parameters, <c>[rows, FeatureDimension]</c>.</summary>
    private Tensor<T> Embed(Vector<T> body, TInput input)
    {
        ParamModel.SetParameters(body);
        Tensor<T> embeddings;
        using (new NoGradScope<T>())
        {
            embeddings = ClassifierOutputs<T>.AsRows(MetaModel.Predict(input));
        }

        return CheckWidth(embeddings);
    }

    /// <summary>
    /// The embeddings, once they are known to be as wide as the head reads. Checked wherever the head meets them - the
    /// composed loss reaches the head before any plain forward pass does.
    /// </summary>
    /// <exception cref="InvalidOperationException">The body's per-example width is not FeatureDimension.</exception>
    private Tensor<T> CheckWidth(Tensor<T> embeddings)
    {
        int width = embeddings.Shape.Length == 2 ? embeddings.Shape[1] : -1;
        if (width != _boilOptions.FeatureDimension)
        {
            throw new InvalidOperationException(
                $"The body emits {width}-wide embeddings per example but FeatureDimension is "
                + $"{_boilOptions.FeatureDimension}. FeatureDimension is the width of the representation the head "
                + "reads - the body's per-example output width.");
        }

        return embeddings;
    }

    /// <summary>
    /// The per-parameter inner learning rate at step <paramref name="step"/>: the layer's learned rate (or the
    /// shared inner rate) inside adapted layers, zero in layers the inner loop leaves alone.
    /// </summary>
    private Vector<T> RateVector(int step)
    {
        int length = _layers.Length == 0 ? 0 : _layers[_layers.Length - 1].Offset + _layers[_layers.Length - 1].Count;
        var rates = new Vector<T>(length);
        T shared = NumOps.FromDouble(_boilOptions.InnerLearningRate);
        for (int l = 0; l < _layers.Length; l++)
        {
            if (!_adaptedLayers[l]) continue;
            T rate = _layerStepRates.Length > 0 ? _layerStepRates[step * _layers.Length + l] : shared;
            var (offset, count) = _layers[l];
            for (int i = offset; i < offset + count; i++) rates[i] = rate;
        }

        return rates;
    }

    /// <summary>
    /// The body's layers as ranges of its flat parameter vector: the trainable layers an ILayeredModel reports
    /// when they tile the vector exactly, otherwise the whole body as one layer.
    /// </summary>
    private static (int Offset, int Count)[] BodyLayers(IFullModel<T, TInput, TOutput> body, int length)
    {
        if (body is ILayeredModel<T> layered)
        {
            var layers = layered.GetAllLayerInfo()
                .Where(info => info.IsTrainable && info.ParameterCount > 0)
                .OrderBy(info => info.ParameterOffset)
                .Select(info => (Offset: info.ParameterOffset, Count: (int)info.ParameterCount))
                .ToArray();

            int next = 0;
            bool tiles = layers.Length > 0;
            foreach (var (offset, count) in layers)
            {
                if (offset != next) { tiles = false; break; }
                next += count;
            }

            if (tiles && next == length) return layers;
        }

        return new[] { (0, length) };
    }

    /// <summary>The top <paramref name="fraction"/> of <paramref name="layerCount"/> layers, at least one.</summary>
    private static bool[] TopLayers(int layerCount, double fraction)
    {
        var adapted = new bool[layerCount];
        int top = Math.Max(1, (int)Math.Ceiling(fraction * layerCount - 1e-9));
        for (int l = layerCount - top; l < layerCount; l++) adapted[l] = true;
        return adapted;
    }

    /// <summary>
    /// Initial per-layer per-step rates: the inner learning rate, scaled by EarlyLayerLrMultiplier in the lower half
    /// of the layers. From there the outer loop learns them.
    /// </summary>
    private Vector<T> InitialLayerStepRates()
    {
        int steps = _boilOptions.AdaptationSteps;
        var rates = new Vector<T>(steps * _layers.Length);
        for (int k = 0; k < steps; k++)
        {
            for (int l = 0; l < _layers.Length; l++)
            {
                bool early = _layers.Length > 1 && l < _layers.Length / 2;
                double rate = _boilOptions.InnerLearningRate * (early ? _boilOptions.EarlyLayerLrMultiplier : 1.0);
                rates[k * _layers.Length + l] = NumOps.FromDouble(rate);
            }
        }

        return rates;
    }

    #endregion

    #region Head

    private Tensor<T> WeightsTensor(Vector<T> weights)
        => Tensor<T>.FromVector(weights).Reshape(_boilOptions.NumClasses, _boilOptions.FeatureDimension);

    /// <summary>The configured loss of the head on fixed embeddings, and its exact gradient with respect to W and b.</summary>
    private (T Loss, Vector<T> Weights, Vector<T> Bias) HeadLossAndGradient(
        Tensor<T> embeddings, Tensor<T> labels, Vector<T> weights, Vector<T> bias)
    {
        Tensor<T> logits;
        using (new NoGradScope<T>())
        {
            logits = EmbeddingClassificationLoss<T>.LinearHead(embeddings, WeightsTensor(weights), Tensor<T>.FromVector(bias));
        }

        var (loss, logitGradient) = LossFunctionExtensions.ComputeLossAndGradient(LossFunction, logits, labels);

        int rows = embeddings.Shape[0];
        int classes = _boilOptions.NumClasses;
        int width = _boilOptions.FeatureDimension;
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

                biasGradient[c] = NumOps.Add(biasGradient[c], g);
            }
        }

        return (loss, weightGradient, biasGradient);
    }

    /// <summary>Initial head weights, uniform in <c>[-s, s]</c> with <c>s = sqrt(2 / FeatureDimension)</c>.</summary>
    private Vector<T> InitializeHeadWeights()
    {
        int size = _boilOptions.FeatureDimension * _boilOptions.NumClasses;
        var weights = new Vector<T>(size);
        double scale = Math.Sqrt(2.0 / _boilOptions.FeatureDimension);
        for (int i = 0; i < size; i++)
        {
            weights[i] = NumOps.FromDouble((RandomGenerator.NextDouble() * 2 - 1) * scale);
        }

        return weights;
    }

    #endregion

    #region Vector arithmetic

    /// <summary>The central-difference step along a direction: small relative to the point, zero with no direction.</summary>
    private static double DifferenceStep(Vector<T> point, Vector<T> direction)
    {
        double directionNorm = 0, pointNorm = 0;
        for (int i = 0; i < point.Length; i++)
        {
            double d = NumOps.ToDouble(direction[i]);
            double p = NumOps.ToDouble(point[i]);
            directionNorm += d * d;
            pointNorm += p * p;
        }

        directionNorm = Math.Sqrt(directionNorm);
        return directionNorm == 0 ? 0 : 1e-5 * (1.0 + Math.Sqrt(pointNorm)) / directionNorm;
    }

    private static Vector<T> Hadamard(Vector<T> a, Vector<T> b)
    {
        var product = new Vector<T>(a.Length);
        for (int i = 0; i < a.Length; i++) product[i] = NumOps.Multiply(a[i], b[i]);
        return product;
    }

    private static Vector<T> Subtract(Vector<T> a, Vector<T> b)
    {
        var difference = new Vector<T>(a.Length);
        for (int i = 0; i < a.Length; i++) difference[i] = NumOps.Subtract(a[i], b[i]);
        return difference;
    }

    private static Vector<T> Shift(Vector<T> values, Vector<T> direction, double step)
    {
        var shifted = new Vector<T>(values.Length);
        T s = NumOps.FromDouble(step);
        for (int i = 0; i < values.Length; i++) shifted[i] = NumOps.Add(values[i], NumOps.Multiply(s, direction[i]));
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

    private Vector<T> Accumulate(Vector<T>? sum, Vector<T> values)
    {
        if (sum is null) return CloneVector(values);
        for (int i = 0; i < sum.Length; i++) sum[i] = NumOps.Add(sum[i], values[i]);
        return sum;
    }

    private static Vector<T> Scale(Vector<T> values, T divisor)
    {
        for (int i = 0; i < values.Length; i++) values[i] = NumOps.Divide(values[i], divisor);
        return values;
    }

    /// <summary>Clones a vector.</summary>
    /// <remarks>An instance member: the coverage suite reaches it by reflection over instance methods.</remarks>
    private Vector<T> CloneVector(Vector<T> source)
    {
        var clone = new Vector<T>(source.Length);
        for (int i = 0; i < source.Length; i++) clone[i] = source[i];
        return clone;
    }

    #endregion
}

using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Extensions;
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
/// Implementation of Latent Embedding Optimization (LEO) meta-learning algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// LEO (Rusu et al. 2019) generates the weights of a linear softmax classifier from a low-dimensional latent code
/// and adapts the code rather than the weights. For each task:
/// </para>
/// <list type="number">
/// <item>A linear encoder maps each support embedding to a code, and a relation network over all pairs of support
/// codes, averaged per class, gives each class a Gaussian <c>N(mu_n, sigma_n^2)</c> over its latent code (eq. 3).</item>
/// <item>A linear decoder maps each class code to a Gaussian over that class's classifier weights (eq. 4).</item>
/// <item>The inner loop takes gradient steps on the codes against the support loss, re-decoding the weights after
/// each (eq. 5, Algorithm 1), then fine-tunes the weights themselves for a few steps (section 4.2.3). Both step
/// sizes are learned per dimension.</item>
/// <item>The outer loop minimises the query loss plus a KL term toward <c>N(0, I)</c>, a penalty pulling the
/// initial codes toward the adapted ones, L2 on every weight and a decoder orthogonality penalty (eq. 6, 7).</item>
/// </list>
/// <para>
/// <b>Exact meta-gradients.</b> Every inner step differentiates the support loss on a nested gradient tape with
/// <c>createGraph</c>, so the step itself is part of the outer graph and the outer loop differentiates through it -
/// the second-order terms included - into the encoder, relation network, decoder, learned step sizes and the
/// feature encoder. <see cref="LEOOptions{T,TInput,TOutput}.UseFirstOrder"/> stops the gradient at each inner
/// gradient. This replaced forward differences over at most 500 weights per network, scaled up as if the rest behaved
/// the same, and a feature-encoder update that differentiated its raw output against the labels.
/// </para>
/// <para>
/// Details the paper leaves open follow DeepMind's reference implementation: the relation network is three
/// bias-free layers with ReLU between them, a Gaussian's scale is <c>max(exp(u) - 1, 1e-10)</c>, the KL term is the
/// mean over sampled codes of <c>log q(z) - log p(z)</c>, the encoder penalty is a mean squared error, the
/// orthogonality penalty is the mean squared deviation of the decoder's row correlations from the identity, dropout
/// acts on the feature embeddings, and evaluation uses means instead of samples.
/// </para>
/// <para>
/// <b>For Beginners:</b> LEO learns to write a small classifier for any new task: it summarises each class of the
/// task in a short code, turns the codes into classifier weights, and nudges the codes (not the weights) until the
/// classifier fits the few examples it was given.
/// </para>
/// <para>
/// Reference: Rusu, A. A., Rao, D., Sygnowski, J., et al. (2019).
/// Meta-Learning with Latent Embedding Optimization. ICLR 2019.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Meta-Learning with Latent Embedding Optimization",
    "https://arxiv.org/abs/1807.05960",
    Year = 2019,
    Authors = "Andrei A. Rusu, Dushyant Rao, Jakub Sygnowski, et al.")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class LEOAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly LEOOptions<T, TInput, TOutput> _leoOptions;

    /// <summary>
    /// The encoder <c>g_e</c>: <c>[codeWidth, EmbeddingDimension]</c>, or one such block per class slot when the
    /// encoder is not shared. The code is LatentDimension wide with a relation network and twice that without one.
    /// </summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _encoderWeights;

    /// <summary>The relation network <c>g_r</c>'s three layers, first to last; empty without a relation network.</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _relationWeights;

    /// <summary>The decoder <c>g_d</c>: <c>[2 * EmbeddingDimension, LatentDimension]</c>, means then scales.</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _decoderWeights;

    /// <summary>The learned latent step size, one per latent dimension (the reference's <c>[1, 1, nz]</c>).</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _latentRates;

    /// <summary>The learned fine-tuning step size, one per embedding dimension (the reference's <c>[1, 1, d]</c>).</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _fineTuningRates;

    /// <summary>
    /// Initializes a new instance of the LEOAlgorithm class.
    /// </summary>
    /// <param name="options">LEO configuration options containing the model and all hyperparameters.</param>
    /// <exception cref="ArgumentNullException">Thrown when options is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the configuration is invalid.</exception>
    public LEOAlgorithm(LEOOptions<T, TInput, TOutput> options)
        : base(
            options?.MetaModel ?? throw new ArgumentNullException(nameof(options), "MetaModel must be set in options."),
            options.LossFunction ?? new CrossEntropyWithLogitsLoss<T>(),
            options,
            options.DataLoader,
            options.MetaOptimizer,
            options.InnerOptimizer)
    {
        _leoOptions = options;
        if (!options.IsValid())
        {
            throw new ArgumentException("LEO configuration is invalid. Check all parameters.", nameof(options));
        }

        int d = options.EmbeddingDimension, nz = options.LatentDimension, h = options.HiddenDimension;
        int codeWidth = CodeWidth;
        int encoderBlocks = options.ShareEncoder ? 1 : options.NumClasses;
        _encoderWeights = new Vector<T>(encoderBlocks * codeWidth * d);
        GlorotUniform(_encoderWeights, 0, encoderBlocks * codeWidth * d, d, codeWidth);

        if (options.UseRelationEncoder)
        {
            _relationWeights = new Vector<T>(h * 2 * nz + h * h + 2 * nz * h);
            GlorotUniform(_relationWeights, 0, h * 2 * nz, 2 * nz, h);
            GlorotUniform(_relationWeights, h * 2 * nz, h * h, h, h);
            GlorotUniform(_relationWeights, h * 2 * nz + h * h, 2 * nz * h, h, 2 * nz);
        }
        else
        {
            _relationWeights = new Vector<T>(0);
        }

        _decoderWeights = new Vector<T>(2 * d * nz);
        if (options.UseOrthogonalInit) OrthogonalDecoder();
        else GlorotUniform(_decoderWeights, 0, 2 * d * nz, nz, 2 * d);

        _latentRates = Filled(nz, options.InnerLearningRate);
        _fineTuningRates = Filled(d, options.FineTuningLearningRate);
    }

    /// <summary>
    /// Gets the algorithm type identifier for this meta-learner.
    /// </summary>
    /// <value>Returns <see cref="MetaLearningAlgorithmType.LEO"/>.</value>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.LEO;

    /// <summary>
    /// Performs one meta-training step: each task's objective (eq. 6), differentiated exactly through its inner loop,
    /// averaged over the batch.
    /// </summary>
    /// <param name="taskBatch">A batch of tasks to meta-train on, each containing support and query sets.</param>
    /// <returns>The average objective across the batch.</returns>
    /// <exception cref="ArgumentException">Thrown when the task batch is null or empty.</exception>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null || taskBatch.BatchSize == 0)
        {
            throw new ArgumentException("Task batch cannot be null or empty.", nameof(taskBatch));
        }

        var body = ParamModel.GetParameters();
        var sums = new Vector<T>?[6];
        T total = NumOps.Zero;
        foreach (var task in taskBatch.Tasks)
        {
            var gradient = EpisodeGradient(task, training: true, RandomGenerator);
            total = NumOps.Add(total, gradient.Loss);
            sums[0] = Accumulate(sums[0], gradient.Body);
            sums[1] = Accumulate(sums[1], gradient.Encoder);
            sums[2] = Accumulate(sums[2], gradient.Relation);
            sums[3] = Accumulate(sums[3], gradient.Decoder);
            sums[4] = Accumulate(sums[4], gradient.LatentRates);
            sums[5] = Accumulate(sums[5], gradient.FineTuningRates);
        }

        T batchSize = NumOps.FromDouble(taskBatch.BatchSize);
        var lengths = new[]
        {
            body.Length, _encoderWeights.Length, _relationWeights.Length, _decoderWeights.Length,
            _latentRates.Length, _fineTuningRates.Length,
        };
        var gradients = new Vector<T>[6];
        for (int i = 0; i < 6; i++)
        {
            gradients[i] = Scale(sums[i] ?? new Vector<T>(lengths[i]), batchSize);
            if (_leoOptions.GradientClipThreshold.HasValue && _leoOptions.GradientClipThreshold.Value > 0 && gradients[i].Length > 0)
            {
                gradients[i] = ClipGradients(gradients[i], _leoOptions.GradientClipThreshold.Value);
            }
        }

        double beta = _leoOptions.OuterLearningRate;
        ParamModel.SetParameters(ApplyGradients(body, gradients[0], beta));
        _encoderWeights = ApplyGradients(_encoderWeights, gradients[1], beta);
        if (_relationWeights.Length > 0) _relationWeights = ApplyGradients(_relationWeights, gradients[2], beta);
        _decoderWeights = ApplyGradients(_decoderWeights, gradients[3], beta);
        _latentRates = ApplyGradients(_latentRates, gradients[4], beta);
        _fineTuningRates = ApplyGradients(_fineTuningRates, gradients[5], beta);

        return NumOps.Divide(total, batchSize);
    }

    /// <summary>
    /// Adapts to a new task: encode the support set to latent means, take the latent steps and the fine-tuning steps,
    /// and keep the resulting classifier. Adaptation uses means rather than samples and no dropout, as the reference
    /// implementation does outside meta-training.
    /// </summary>
    /// <param name="task">The new task containing support set examples for adaptation.</param>
    /// <returns>The adapted classifier with its own copy of the feature encoder.</returns>
    /// <exception cref="ArgumentNullException">Thrown when task is null.</exception>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null)
        {
            throw new ArgumentNullException(nameof(task));
        }

        var episode = Episode(task);
        var (support, query) = Embed(task, episode);
        var weights = new LeoWeights(this);
        var run = Run(support, query, episode, weights, Noise.None);
        return new LEOModel<T, TInput, TOutput>(
            MetaModel, run.Classifier.ToVector(), run.Latents.ToVector(), _leoOptions, episode.ClassSlots);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// The adapted model returns class probabilities, so this is the configured loss of their logarithm against the
    /// class indices - with the default cross-entropy, the paper's classification loss. A Vector output carries one
    /// predicted class per example, and its loss is the classification error rate.
    /// </remarks>
    protected override T ComputeLossFromOutput(TOutput predictions, TOutput expectedOutput)
        => ClassifierOutputs<T>.ProbabilityLoss(LossFunction, predictions, expectedOutput);

    #region Episode

    private int CodeWidth => _leoOptions.UseRelationEncoder ? _leoOptions.LatentDimension : 2 * _leoOptions.LatentDimension;

    private readonly struct Gradient
    {
        public Gradient(T loss, Vector<T> body, Vector<T> encoder, Vector<T> relation, Vector<T> decoder,
            Vector<T> latentRates, Vector<T> fineTuningRates)
        {
            Loss = loss;
            Body = body;
            Encoder = encoder;
            Relation = relation;
            Decoder = decoder;
            LatentRates = latentRates;
            FineTuningRates = fineTuningRates;
        }

        public T Loss { get; }
        public Vector<T> Body { get; }
        public Vector<T> Encoder { get; }
        public Vector<T> Relation { get; }
        public Vector<T> Decoder { get; }
        public Vector<T> LatentRates { get; }
        public Vector<T> FineTuningRates { get; }
    }

    /// <summary>
    /// One task's objective and its exact gradient with respect to the feature encoder and every LEO weight. The
    /// sampling noise and dropout masks are drawn once, so both passes differentiate the same function.
    /// </summary>
    private Gradient EpisodeGradient(IMetaLearningTask<T, TInput, TOutput> task, bool training, Random random)
    {
        var episode = Episode(task);
        int supportRows = episode.SupportSelector.Shape[0], queryRows = episode.QuerySelector.Shape[0];
        var noise = training
            ? Noise.Draw(random, _leoOptions, episode.ClassSlots.Length, supportRows, queryRows)
            : Noise.None;
        var stackedInput = ClassifierOutputs<T>.StackRows(task.SupportInput, task.QueryInput);
        var stackedTarget = ClassifierOutputs<T>.ToOutput<TOutput>(new Tensor<T>(new[] { episode.Rows, 1 }));

        // The feature encoder's gradient: the whole objective rebuilt from its embeddings on the tape, LEO's own
        // weights held constant.
        var frozen = new LeoWeights(this);
        var composed = new EmbeddingObjectiveLoss<T>(embeddings =>
        {
            var engine = AiDotNetEngine.Current;
            var s = CheckWidth(engine.TensorMatMul(episode.SupportSelector, embeddings));
            var q = engine.TensorMatMul(episode.QuerySelector, embeddings);
            return Run(s, q, episode, frozen, noise).Objective;
        });
        var bodyGradient = ComputeGradients(MetaModel, stackedInput, stackedTarget, composed);

        // LEO's gradient: the same objective from fixed embeddings, against its own weights.
        var (support, query) = Embed(task, episode);
        var weights = new LeoWeights(this);
        T loss;
        var encoder = new Vector<T>(_encoderWeights.Length);
        var relation = new Vector<T>(_relationWeights.Length);
        var decoder = new Vector<T>(_decoderWeights.Length);
        var latentRates = new Vector<T>(_latentRates.Length);
        var fineTuningRates = new Vector<T>(_fineTuningRates.Length);
        using (var tape = new GradientTape<T>(new GradientTapeOptions { Persistent = true }))
        {
            var objective = Run(support, query, episode, weights, noise).Objective;
            loss = objective[0];
            var gradients = tape.ComputeGradients(objective, weights.Leaves);
            weights.CopyGradients(gradients, encoder, relation, decoder, latentRates, fineTuningRates);
        }

        return new Gradient(loss, bodyGradient, encoder, relation, decoder, latentRates, fineTuningRates);
    }

    private sealed class RunResult
    {
        public RunResult(Tensor<T> objective, Tensor<T> classifier, Tensor<T> latents, Tensor<T> initial,
            Tensor<T> latentScales, Tensor<T> weightScales)
        {
            Objective = objective;
            Classifier = classifier;
            Latents = latents;
            Initial = initial;
            LatentScales = latentScales;
            WeightScales = weightScales;
        }

        /// <summary>The encoder's Gaussian scales over the latent codes.</summary>
        public Tensor<T> LatentScales { get; }

        /// <summary>The decoder's Gaussian scales over the classifier weights.</summary>
        public Tensor<T> WeightScales { get; }

        public Tensor<T> Objective { get; }
        public Tensor<T> Classifier { get; }

        /// <summary>The codes after the latent steps.</summary>
        public Tensor<T> Latents { get; }

        /// <summary>The codes the encoder produced, before any latent step.</summary>
        public Tensor<T> Initial { get; }
    }

    /// <summary>
    /// Algorithm 1 for one task from its support and query embeddings: encode, sample, decode, latent steps,
    /// fine-tuning steps, and the eq. 6 objective. Every op is an engine op, so a live tape differentiates it.
    /// </summary>
    private RunResult Run(Tensor<T> support, Tensor<T> query, PrototypeEpisode<T> episode, LeoWeights weights, Noise noise)
    {
        var engine = AiDotNetEngine.Current;
        int nz = _leoOptions.LatentDimension, d = _leoOptions.EmbeddingDimension;
        int classes = episode.ClassSlots.Length, supportRows = support.Shape[0];
        var supportTarget = SupportTarget(episode, supportRows);

        // Eq. 3: codes per example, pair-wise relations averaged per class, a Gaussian per class.
        var codes = Encode(Drop(support, noise.EncoderMask), episode, weights);
        Tensor<T> parameters;
        if (weights.Relation1 is not null && weights.Relation2 is not null && weights.Relation3 is not null)
        {
            var pairs = Pairs(codes);
            var hidden = engine.ReLU(engine.TensorMatMul(pairs, engine.TensorTranspose(weights.Relation1)));
            hidden = engine.ReLU(engine.TensorMatMul(hidden, engine.TensorTranspose(weights.Relation2)));
            var relations = engine.TensorMatMul(hidden, engine.TensorTranspose(weights.Relation3));
            parameters = engine.TensorMatMul(PairAverager(episode, supportRows), relations);
        }
        else
        {
            parameters = engine.TensorMatMul(ClassAverager(episode, supportRows), codes);
        }

        var means = engine.TensorMatMul(parameters, Half(2 * nz, nz, 0));
        var scales = Scale(engine.TensorMatMul(parameters, Half(2 * nz, nz, nz)));
        var initial = noise.LatentNoise is null
            ? means
            : engine.TensorAdd(means, engine.TensorMultiply(scales, noise.LatentNoise));

        // Latent steps (eq. 5, Algorithm 1 lines 10-14): the support loss's gradient with respect to the codes,
        // through the decoder, on a nested tape whose backward the outer tape records.
        var latents = initial;
        for (int step = 0; step < _leoOptions.AdaptationSteps; step++)
        {
            var current = latents;
            var gradient = InnerGradient(current, () =>
            {
                var w = Decode(current, weights, noise.WeightNoise(step)).Weights;
                return SupportLoss(support, w, supportTarget, noise.InnerMask(step));
            });
            latents = engine.TensorAdd(current, engine.TensorNegate(engine.TensorMultiply(weights.LatentRates, gradient)));
        }

        var decoded = Decode(latents, weights, noise.WeightNoise(_leoOptions.AdaptationSteps));
        var classifier = decoded.Weights;

        // Fine-tuning in parameter space (section 4.2.3), from the weights LEO generated.
        for (int step = 0; step < _leoOptions.FineTuningSteps; step++)
        {
            var current = classifier;
            var gradient = InnerGradient(current,
                () => SupportLoss(support, current, supportTarget, noise.FineTuningMask(step)));
            classifier = engine.TensorAdd(current, engine.TensorNegate(engine.TensorMultiply(weights.FineTuningRates, gradient)));
        }

        // Eq. 6: validation loss, KL toward N(0, I), the encoder penalty, then eq. 7's regularisers.
        var logits = engine.TensorMatMul(Drop(query, noise.QueryMask), engine.TensorTranspose(classifier));
        var objective = Scalar(LossFunction.ComputeTapeLoss(logits, episode.QueryTarget));

        if (noise.LatentNoise is not null && _leoOptions.KLWeight > 0)
        {
            var standardized = engine.TensorDivide(engine.TensorAdd(initial, engine.TensorNegate(means)), scales);
            var logRatio = engine.TensorAdd(
                engine.TensorAdd(engine.TensorNegate(engine.TensorLog(scales)),
                    engine.TensorMultiplyScalar(engine.TensorMultiply(standardized, standardized), Ops.FromDouble(-0.5))),
                engine.TensorMultiplyScalar(engine.TensorMultiply(initial, initial), Ops.FromDouble(0.5)));
            objective = AddWeighted(objective, Mean(logRatio), _leoOptions.KLWeight);
        }

        if (_leoOptions.EncoderPenaltyWeight > 0)
        {
            var gap = engine.TensorAdd(engine.StopGradient(latents), engine.TensorNegate(initial));
            objective = AddWeighted(objective, Mean(engine.TensorMultiply(gap, gap)), _leoOptions.EncoderPenaltyWeight);
        }

        if (noise.LatentNoise is not null && _leoOptions.EntropyWeight > 0)
        {
            // Extension: the entropy of the decoder's weight distribution is sum(log sigma) plus a constant; a bonus
            // on it keeps the generator from collapsing to a point mass.
            var entropy = Mean(engine.TensorLog(Decode(initial, weights, null).Scales));
            objective = AddWeighted(objective, entropy, -_leoOptions.EntropyWeight);
        }

        if (_leoOptions.L2Regularization > 0)
        {
            Tensor<T>? squares = null;
            foreach (var tensor in weights.NetworkWeights)
            {
                var sum = Sum(engine.TensorMultiply(tensor, tensor));
                squares = squares is null ? sum : engine.TensorAdd(squares, sum);
            }

            if (squares is not null) objective = AddWeighted(objective, squares, 0.5 * _leoOptions.L2Regularization);
        }

        if (_leoOptions.OrthogonalityWeight > 0)
        {
            objective = AddWeighted(objective, OrthogonalityPenalty(weights.Decoder), _leoOptions.OrthogonalityWeight);
        }

        return new RunResult(objective, classifier, latents, initial, scales, decoded.Scales);
    }

    /// <summary>
    /// The gradient of an inner loss with respect to <paramref name="source"/>, computed on a nested tape. With
    /// createGraph the backward is recorded on the enclosing tape, so outer differentiation reaches through it.
    /// </summary>
    private Tensor<T> InnerGradient(Tensor<T> source, Func<Tensor<T>> innerLoss)
    {
        // The tape is deliberately not disposed: inside a TensorArena, disposing it returns tensors the enclosing
        // computation still holds - the next step then reads a recycled buffer and fails on its shape. Left to the
        // collector, the step's graph stays valid for the outer backward that differentiates through it.
        var inner = new GradientTape<T>(new GradientTapeOptions { Persistent = true });
        var loss = innerLoss();
        var gradients = inner.ComputeGradients(loss, new[] { source }, createGraph: !_leoOptions.UseFirstOrder);
        var gradient = gradients.TryGetValue(source, out var g) ? g : new Tensor<T>(source._shape);
        return _leoOptions.UseFirstOrder ? AiDotNetEngine.Current.StopGradient(gradient) : gradient;
    }

    private Tensor<T> SupportLoss(Tensor<T> support, Tensor<T> classifier, Tensor<T> target, Tensor<T>? mask)
    {
        var engine = AiDotNetEngine.Current;
        var logits = engine.TensorMatMul(Drop(support, mask), engine.TensorTranspose(classifier));
        return Scalar(LossFunction.ComputeTapeLoss(logits, target));
    }

    /// <summary>Eq. 4: each class code to a Gaussian over that class's weights; a sample while training.</summary>
    private (Tensor<T> Weights, Tensor<T> Scales) Decode(Tensor<T> latents, LeoWeights weights, Tensor<T>? noise)
    {
        var engine = AiDotNetEngine.Current;
        int d = _leoOptions.EmbeddingDimension;
        var outputs = engine.TensorMatMul(latents, engine.TensorTranspose(weights.Decoder));
        var means = engine.TensorMatMul(outputs, Half(2 * d, d, 0));
        var scales = Scale(engine.TensorMatMul(outputs, Half(2 * d, d, d)));
        var sample = noise is null ? means : engine.TensorAdd(means, engine.TensorMultiply(scales, noise));
        return (sample, scales);
    }

    /// <summary>The reference implementation's scale: <c>max(exp(u) - 1, 1e-10)</c>.</summary>
    private static Tensor<T> Scale(Tensor<T> unnormalized)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorClampMin(engine.TensorAddScalar(engine.TensorExp(unnormalized), Ops.FromDouble(-1.0)),
            Ops.FromDouble(1e-10));
    }

    /// <summary>Per-example codes: the shared encoder, or each example's class-slot encoder.</summary>
    private Tensor<T> Encode(Tensor<T> support, PrototypeEpisode<T> episode, LeoWeights weights)
    {
        var engine = AiDotNetEngine.Current;
        int codeWidth = CodeWidth;
        if (_leoOptions.ShareEncoder)
        {
            return engine.TensorMatMul(support, engine.TensorTranspose(weights.Encoder));
        }

        // Extension: one encoder per class slot, each applied to the examples of its class.
        int rows = support.Shape[0], blocks = _leoOptions.NumClasses;
        Tensor<T>? codes = null;
        for (int c = 0; c < episode.ClassSlots.Length; c++)
        {
            int slot = episode.ClassSlots[c];
            var pick = new Tensor<T>(new[] { codeWidth, blocks * codeWidth });
            for (int i = 0; i < codeWidth; i++) pick[i * blocks * codeWidth + slot * codeWidth + i] = Ops.One;
            var block = engine.TensorMatMul(pick, weights.Encoder);
            var rowMask = new Tensor<T>(new[] { rows, 1 });
            for (int r = 0; r < rows; r++) rowMask[r] = episode.Membership[c * rows + r];
            var part = engine.TensorMultiply(engine.TensorMatMul(support, engine.TensorTranspose(block)), rowMask);
            codes = codes is null ? part : engine.TensorAdd(codes, part);
        }

        return codes ?? new Tensor<T>(new[] { rows, codeWidth });
    }

    /// <summary>Every ordered pair of codes side by side, <c>[rows * rows, 2 * width]</c>; row <c>i * rows + j</c>.</summary>
    private static Tensor<T> Pairs(Tensor<T> codes)
    {
        var engine = AiDotNetEngine.Current;
        int rows = codes.Shape[0], width = codes.Shape[1], pairs = rows * rows;
        var left = new Tensor<T>(new[] { pairs, rows });
        var right = new Tensor<T>(new[] { pairs, rows });
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < rows; j++)
            {
                left[(i * rows + j) * rows + i] = Ops.One;
                right[(i * rows + j) * rows + j] = Ops.One;
            }
        }

        var placeLeft = new Tensor<T>(new[] { width, 2 * width });
        var placeRight = new Tensor<T>(new[] { width, 2 * width });
        for (int k = 0; k < width; k++)
        {
            placeLeft[k * 2 * width + k] = Ops.One;
            placeRight[k * 2 * width + width + k] = Ops.One;
        }

        return engine.TensorAdd(
            engine.TensorMatMul(engine.TensorMatMul(left, codes), placeLeft),
            engine.TensorMatMul(engine.TensorMatMul(right, codes), placeRight));
    }

    /// <summary>
    /// <c>[classes, rows * rows]</c>: eq. 3's average, over a class's examples and every example they are paired with.
    /// </summary>
    private static Tensor<T> PairAverager(PrototypeEpisode<T> episode, int rows)
    {
        int classes = episode.ClassSlots.Length;
        var averager = new Tensor<T>(new[] { classes, rows * rows });
        for (int c = 0; c < classes; c++)
        {
            double members = 0;
            for (int i = 0; i < rows; i++) members += Ops.ToDouble(episode.Membership[c * rows + i]);
            T share = Ops.FromDouble(1.0 / (members * rows));
            for (int i = 0; i < rows; i++)
            {
                if (Ops.ToDouble(episode.Membership[c * rows + i]) < 0.5) continue;
                for (int j = 0; j < rows; j++) averager[c * rows * rows + i * rows + j] = share;
            }
        }

        return averager;
    }

    /// <summary><c>[classes, rows]</c>: the mean over each class's examples.</summary>
    private static Tensor<T> ClassAverager(PrototypeEpisode<T> episode, int rows)
    {
        int classes = episode.ClassSlots.Length;
        var averager = new Tensor<T>(new[] { classes, rows });
        for (int c = 0; c < classes; c++)
        {
            double members = 0;
            for (int i = 0; i < rows; i++) members += Ops.ToDouble(episode.Membership[c * rows + i]);
            for (int i = 0; i < rows; i++)
            {
                averager[c * rows + i] = Ops.FromDouble(Ops.ToDouble(episode.Membership[c * rows + i]) / members);
            }
        }

        return averager;
    }

    /// <summary>
    /// The reference orthogonality penalty: the mean squared deviation from the identity of the correlations between
    /// the decoder's latent rows.
    /// </summary>
    private static Tensor<T> OrthogonalityPenalty(Tensor<T> decoder)
    {
        var engine = AiDotNetEngine.Current;
        var rows = engine.TensorTranspose(decoder);
        var products = engine.TensorMatMul(rows, engine.TensorTranspose(rows));
        var norms = engine.TensorAddScalar(
            engine.TensorSqrt(engine.ReduceSum(engine.TensorMultiply(rows, rows), new[] { 1 }, keepDims: true)),
            Ops.FromDouble(1e-32));
        var correlation = engine.TensorDivide(products, engine.TensorMatMul(norms, engine.TensorTranspose(norms)));
        int n = correlation.Shape[0];
        var identity = new Tensor<T>(new[] { n, n });
        for (int i = 0; i < n; i++) identity[i * n + i] = Ops.One;
        var deviation = engine.TensorAdd(correlation, engine.TensorNegate(identity));
        return Mean(engine.TensorMultiply(deviation, deviation));
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
        var support = CheckWidth(engine.TensorMatMul(episode.SupportSelector, embeddings));
        return (support, engine.TensorMatMul(episode.QuerySelector, embeddings));
    }

    private Tensor<T> CheckWidth(Tensor<T> embeddings)
    {
        if (embeddings.Shape[1] != _leoOptions.EmbeddingDimension)
        {
            throw new InvalidOperationException(
                $"The feature encoder emits {embeddings.Shape[1]}-wide embeddings per example but EmbeddingDimension is "
                + $"{_leoOptions.EmbeddingDimension}. EmbeddingDimension is the width of the representation LEO "
                + "encodes and classifies - the feature encoder's per-example output width.");
        }

        return embeddings;
    }

    private PrototypeEpisode<T> Episode(IMetaLearningTask<T, TInput, TOutput> task)
        => PrototypeEpisode<T>.Build(ReadLabels(task.SupportOutput), ReadLabels(task.QueryOutput));

    private int[] ReadLabels(TOutput labels)
    {
        var tensor = ClassifierOutputs<T>.Labels(labels, _leoOptions.NumClasses);
        var indices = new int[tensor.Length];
        for (int i = 0; i < indices.Length; i++) indices[i] = (int)Math.Round(NumOps.ToDouble(tensor[i]));
        return indices;
    }

    /// <summary>Each support row's class column, <c>[rows]</c>.</summary>
    private static Tensor<T> SupportTarget(PrototypeEpisode<T> episode, int rows)
    {
        var target = new Tensor<T>(new[] { rows });
        for (int c = 0; c < episode.ClassSlots.Length; c++)
        {
            for (int r = 0; r < rows; r++)
            {
                if (Ops.ToDouble(episode.Membership[c * rows + r]) > 0.5) target[r] = Ops.FromDouble(c);
            }
        }

        return target;
    }

    #endregion

    #region Weights and noise

    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    /// <summary>LEO's weights as tape leaves, with the flat layout each came from.</summary>
    private sealed class LeoWeights
    {
        private readonly List<Tensor<T>> _leaves = new List<Tensor<T>>();

        public LeoWeights(LEOAlgorithm<T, TInput, TOutput> owner)
        {
            var o = owner._leoOptions;
            int d = o.EmbeddingDimension, nz = o.LatentDimension, h = o.HiddenDimension, code = owner.CodeWidth;
            int blocks = o.ShareEncoder ? 1 : o.NumClasses;
            Encoder = Leaf(owner._encoderWeights, 0, blocks * code, d);
            if (owner._relationWeights.Length > 0)
            {
                Relation1 = Leaf(owner._relationWeights, 0, h, 2 * nz);
                Relation2 = Leaf(owner._relationWeights, h * 2 * nz, h, h);
                Relation3 = Leaf(owner._relationWeights, h * 2 * nz + h * h, 2 * nz, h);
            }

            Decoder = Leaf(owner._decoderWeights, 0, 2 * d, nz);
            LatentRates = Leaf(owner._latentRates, 0, 1, nz);
            FineTuningRates = Leaf(owner._fineTuningRates, 0, 1, d);
        }

        public Tensor<T> Encoder { get; }
        public Tensor<T>? Relation1 { get; }
        public Tensor<T>? Relation2 { get; }
        public Tensor<T>? Relation3 { get; }
        public Tensor<T> Decoder { get; }
        public Tensor<T> LatentRates { get; }
        public Tensor<T> FineTuningRates { get; }

        /// <summary>Every leaf, in the order <see cref="CopyGradients"/> reads them.</summary>
        public IReadOnlyList<Tensor<T>> Leaves => _leaves;

        /// <summary>The encoder, relation and decoder weights - what eq. 7's L2 term regularises.</summary>
        public IEnumerable<Tensor<T>> NetworkWeights
        {
            get
            {
                yield return Encoder;
                if (Relation1 is not null) yield return Relation1;
                if (Relation2 is not null) yield return Relation2;
                if (Relation3 is not null) yield return Relation3;
                yield return Decoder;
            }
        }

        public void CopyGradients(Dictionary<Tensor<T>, Tensor<T>> gradients, Vector<T> encoder, Vector<T> relation,
            Vector<T> decoder, Vector<T> latentRates, Vector<T> fineTuningRates)
        {
            Copy(gradients, Encoder, encoder, 0);
            if (Relation1 is not null) Copy(gradients, Relation1, relation, 0);
            if (Relation2 is not null && Relation1 is not null) Copy(gradients, Relation2, relation, Relation1.Length);
            if (Relation3 is not null && Relation1 is not null && Relation2 is not null)
                Copy(gradients, Relation3, relation, Relation1.Length + Relation2.Length);
            Copy(gradients, Decoder, decoder, 0);
            Copy(gradients, LatentRates, latentRates, 0);
            Copy(gradients, FineTuningRates, fineTuningRates, 0);
        }

        private Tensor<T> Leaf(Vector<T> source, int offset, int rows, int columns)
        {
            var leaf = new Tensor<T>(new[] { rows, columns });
            for (int i = 0; i < leaf.Length; i++) leaf[i] = source[offset + i];
            _leaves.Add(leaf);
            return leaf;
        }

        private static void Copy(Dictionary<Tensor<T>, Tensor<T>> gradients, Tensor<T> leaf, Vector<T> into, int offset)
        {
            if (!gradients.TryGetValue(leaf, out var gradient)) return;
            for (int i = 0; i < leaf.Length; i++) into[offset + i] = gradient[i];
        }
    }

    /// <summary>
    /// One task's sampling noise and dropout masks, drawn up front so every pass over the task sees the same ones.
    /// </summary>
    private sealed class Noise
    {
        private readonly Tensor<T>?[] _weightNoise;
        private readonly Tensor<T>?[] _innerMasks;
        private readonly Tensor<T>?[] _fineTuningMasks;

        private Noise(Tensor<T>? latentNoise, Tensor<T>?[] weightNoise, Tensor<T>? encoderMask, Tensor<T>?[] innerMasks,
            Tensor<T>?[] fineTuningMasks, Tensor<T>? queryMask)
        {
            LatentNoise = latentNoise;
            _weightNoise = weightNoise;
            EncoderMask = encoderMask;
            _innerMasks = innerMasks;
            _fineTuningMasks = fineTuningMasks;
            QueryMask = queryMask;
        }

        /// <summary>No sampling and no dropout: evaluation, as the reference implementation evaluates.</summary>
        public static Noise None { get; } = new Noise(null, Array.Empty<Tensor<T>?>(), null, Array.Empty<Tensor<T>?>(),
            Array.Empty<Tensor<T>?>(), null);

        public Tensor<T>? LatentNoise { get; }
        public Tensor<T>? EncoderMask { get; }
        public Tensor<T>? QueryMask { get; }

        public Tensor<T>? WeightNoise(int decode) => decode < _weightNoise.Length ? _weightNoise[decode] : null;
        public Tensor<T>? InnerMask(int step) => step < _innerMasks.Length ? _innerMasks[step] : null;
        public Tensor<T>? FineTuningMask(int step) => step < _fineTuningMasks.Length ? _fineTuningMasks[step] : null;

        public static Noise Draw(Random random, LEOOptions<T, TInput, TOutput> options, int classes, int supportRows, int queryRows)
        {
            int nz = options.LatentDimension, d = options.EmbeddingDimension;
            Tensor<T> Gaussian(int rows, int columns)
            {
                var t = new Tensor<T>(new[] { rows, columns });
                for (int i = 0; i < t.Length; i++) t[i] = Ops.FromDouble(random.NextGaussian());
                return t;
            }

            Tensor<T>? Mask(int rows)
            {
                if (options.DropoutRate <= 0) return null;
                var t = new Tensor<T>(new[] { rows, d });
                T keep = Ops.FromDouble(1.0 / (1.0 - options.DropoutRate));
                for (int i = 0; i < t.Length; i++) t[i] = random.NextDouble() >= options.DropoutRate ? keep : Ops.Zero;
                return t;
            }

            var weightNoise = new Tensor<T>?[options.AdaptationSteps + 1];
            for (int i = 0; i < weightNoise.Length; i++) weightNoise[i] = Gaussian(classes, d);
            var inner = new Tensor<T>?[options.AdaptationSteps];
            for (int i = 0; i < inner.Length; i++) inner[i] = Mask(supportRows);
            var fine = new Tensor<T>?[options.FineTuningSteps];
            for (int i = 0; i < fine.Length; i++) fine[i] = Mask(supportRows);
            return new Noise(Gaussian(classes, nz), weightNoise, Mask(supportRows), inner, fine, Mask(queryRows));
        }
    }

    #endregion

    #region Test hooks

    /// <summary>
    /// One task's objective and exact gradient, with sampling and dropout drawn from a fixed seed while
    /// <paramref name="training"/> - the same function on every call, for gradient checks.
    /// </summary>
    internal (T Loss, Vector<T> Body, Vector<T> Encoder, Vector<T> Relation, Vector<T> Decoder, Vector<T> LatentRates,
        Vector<T> FineTuningRates) EpisodeGradientForTesting(IMetaLearningTask<T, TInput, TOutput> task, bool training)
    {
        var g = EpisodeGradient(task, training, RandomHelper.CreateSeededRandom(TestNoiseSeed));
        return (g.Loss, g.Body, g.Encoder, g.Relation, g.Decoder, g.LatentRates, g.FineTuningRates);
    }

    /// <summary>One task's objective with the same fixed-seed noise as <see cref="EpisodeGradientForTesting"/>.</summary>
    internal T EpisodeLossForTesting(IMetaLearningTask<T, TInput, TOutput> task, bool training)
    {
        var episode = Episode(task);
        var noise = training
            ? Noise.Draw(RandomHelper.CreateSeededRandom(TestNoiseSeed), _leoOptions, episode.ClassSlots.Length,
                episode.SupportSelector.Shape[0], episode.QuerySelector.Shape[0])
            : Noise.None;
        var (support, query) = Embed(task, episode);
        return Run(support, query, episode, new LeoWeights(this), noise).Objective[0];
    }

    /// <summary>
    /// One task's Gaussian scales, the encoder's then the decoder's. A scale sitting at the floor of
    /// <c>max(exp(u) - 1, 1e-10)</c> is a kink: the objective is flat in it, which a central difference across the
    /// floor does not see.
    /// </summary>
    internal Vector<T> ScalesForTesting(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var episode = Episode(task);
        var (support, query) = Embed(task, episode);
        var run = Run(support, query, episode, new LeoWeights(this), Noise.None);
        var latent = run.LatentScales.ToVector();
        var weights = run.WeightScales.ToVector();
        var all = new Vector<T>(latent.Length + weights.Length);
        for (int i = 0; i < latent.Length; i++) all[i] = latent[i];
        for (int i = 0; i < weights.Length; i++) all[latent.Length + i] = weights[i];
        return all;
    }

    /// <summary>One task's latent codes before and after the latent steps, without sampling or dropout.</summary>
    internal (Vector<T> Initial, Vector<T> Adapted) LatentCodesForTesting(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var episode = Episode(task);
        var (support, query) = Embed(task, episode);
        var run = Run(support, query, episode, new LeoWeights(this), Noise.None);
        return (run.Initial.ToVector(), run.Latents.ToVector());
    }

    private const int TestNoiseSeed = 20190501;

    /// <summary>Gets or sets a copy of the encoder weights (for tests).</summary>
    internal Vector<T> EncoderWeightsForTesting { get => CloneVector(_encoderWeights); set => _encoderWeights = CloneVector(value); }

    /// <summary>Gets or sets a copy of the relation network weights (for tests).</summary>
    internal Vector<T> RelationWeightsForTesting { get => CloneVector(_relationWeights); set => _relationWeights = CloneVector(value); }

    /// <summary>Gets or sets a copy of the decoder weights (for tests).</summary>
    internal Vector<T> DecoderWeightsForTesting { get => CloneVector(_decoderWeights); set => _decoderWeights = CloneVector(value); }

    /// <summary>Gets or sets a copy of the learned latent step sizes (for tests).</summary>
    internal Vector<T> LatentRatesForTesting { get => CloneVector(_latentRates); set => _latentRates = CloneVector(value); }

    /// <summary>Gets or sets a copy of the learned fine-tuning step sizes (for tests).</summary>
    internal Vector<T> FineTuningRatesForTesting { get => CloneVector(_fineTuningRates); set => _fineTuningRates = CloneVector(value); }

    #endregion

    #region Helpers

    /// <summary>Glorot-uniform initialisation, U(-sqrt(6 / (fan_in + fan_out)), +...), the reference's initializer.</summary>
    private void GlorotUniform(Vector<T> weights, int offset, int count, int fanIn, int fanOut)
    {
        double bound = Math.Sqrt(6.0 / (fanIn + fanOut));
        for (int i = 0; i < count; i++)
        {
            weights[offset + i] = NumOps.FromDouble((2.0 * RandomGenerator.NextDouble() - 1.0) * bound);
        }
    }

    /// <summary>
    /// Extension: an orthogonal decoder - Gram-Schmidt over Gaussian latent rows - so the decoder starts where the
    /// orthogonality penalty wants it. With more latent rows than a row is long, the columns are made orthonormal.
    /// </summary>
    private void OrthogonalDecoder()
    {
        int d2 = 2 * _leoOptions.EmbeddingDimension, nz = _leoOptions.LatentDimension;
        bool byRows = nz <= d2;
        int vectors = byRows ? nz : d2, length = byRows ? d2 : nz;
        var basis = new double[vectors][];
        for (int v = 0; v < vectors; v++)
        {
            var candidate = new double[length];
            for (int i = 0; i < length; i++) candidate[i] = RandomGenerator.NextGaussian();
            for (int u = 0; u < v; u++)
            {
                double dot = 0;
                for (int i = 0; i < length; i++) dot += candidate[i] * basis[u][i];
                for (int i = 0; i < length; i++) candidate[i] -= dot * basis[u][i];
            }

            double norm = Math.Sqrt(candidate.Sum(x => x * x));
            if (norm < 1e-12) norm = 1e-12;
            for (int i = 0; i < length; i++) candidate[i] /= norm;
            basis[v] = candidate;
        }

        // Stored as [2d, nz]: entry (row r of the output, latent k).
        for (int r = 0; r < d2; r++)
        {
            for (int k = 0; k < nz; k++)
            {
                double value = byRows ? basis[k][r] : basis[r][k];
                _decoderWeights[r * nz + k] = NumOps.FromDouble(value);
            }
        }
    }

    private static Vector<T> Filled(int length, double value)
    {
        var vector = new Vector<T>(length);
        for (int i = 0; i < length; i++) vector[i] = Ops.FromDouble(value);
        return vector;
    }

    /// <summary><c>[width, half]</c>: picks <paramref name="half"/> columns from <paramref name="start"/>.</summary>
    private static Tensor<T> Half(int width, int half, int start)
    {
        var select = new Tensor<T>(new[] { width, half });
        for (int i = 0; i < half; i++) select[(start + i) * half + i] = Ops.One;
        return select;
    }

    private static Tensor<T> Drop(Tensor<T> rows, Tensor<T>? mask)
        => mask is null ? rows : AiDotNetEngine.Current.TensorMultiply(rows, mask);

    private static Tensor<T> Scalar(Tensor<T> value) => AiDotNetEngine.Current.Reshape(value, new[] { 1 });

    private static Tensor<T> Sum(Tensor<T> value) => Scalar(AiDotNetEngine.Current.ReduceSum(value, null));

    private static Tensor<T> Mean(Tensor<T> value)
        => AiDotNetEngine.Current.TensorMultiplyScalar(Sum(value), Ops.FromDouble(1.0 / value.Length));

    private static Tensor<T> AddWeighted(Tensor<T> total, Tensor<T> term, double weight)
        => AiDotNetEngine.Current.TensorAdd(total, AiDotNetEngine.Current.TensorMultiplyScalar(term, Ops.FromDouble(weight)));

    private static Vector<T> Accumulate(Vector<T>? sum, Vector<T> values)
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

    private static Vector<T> CloneVector(Vector<T> source)
    {
        var clone = new Vector<T>(source.Length);
        for (int i = 0; i < source.Length; i++) clone[i] = source[i];
        return clone;
    }

    #endregion
}

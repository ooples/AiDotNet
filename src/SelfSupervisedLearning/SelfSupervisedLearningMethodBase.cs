using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.SelfSupervisedLearning;

/// <summary>
/// Abstract base class for self-supervised learning methods.
/// Extends <see cref="ModelBase{T, TInput, TOutput}"/> for unified model framework participation.
/// </summary>
/// <typeparam name="T">The numeric type used for computations (typically float or double).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This base class provides common functionality shared by all
/// SSL methods, including parameter management, training mode control, and configuration handling.</para>
///
/// <para>Derived classes (SimCLR, MoCo, BYOL, etc.) implement the specific training logic
/// in the <see cref="TrainStepCore"/> method.</para>
/// </remarks>
public abstract partial class SelfSupervisedLearningMethodBase<T> : ModelBase<T, Tensor<T>, Tensor<T>>, ISelfSupervisedLearningMethod<T>
{
    // NumOps and Engine are inherited from ModelBase

    /// <summary>
    /// The main encoder neural network.
    /// </summary>
    [TrainableParameter]
    protected readonly INeuralNetwork<T> _encoder;

    /// <summary>
    /// The projection head for SSL embeddings.
    /// </summary>
    [TrainableParameter(Optional = true)]
    protected readonly IProjectorHead<T>? _projector;

    /// <summary>
    /// Gets the projector, throwing if not initialized.
    /// </summary>
    [ParameterAlias(nameof(_projector))]
    protected IProjectorHead<T> Projector => _projector ?? throw new InvalidOperationException(
        $"{GetType().Name}: Projector head not initialized. Ensure a projector was provided during construction.");

    /// <summary>
    /// The SSL configuration.
    /// </summary>
    protected readonly SelfSupervisedLearningConfig<T> _config;

    /// <summary>
    /// Whether the method is in training mode.
    /// </summary>
    protected bool _isTraining = true;

    /// <summary>
    /// Current training step counter.
    /// </summary>
    protected int _currentStep;

    /// <summary>
    /// Current epoch counter.
    /// </summary>
    protected int _currentEpoch;

    /// <inheritdoc />
    public abstract string Name { get; }

    /// <inheritdoc />
    public abstract SelfSupervisedLearningMethodCategory Category { get; }

    /// <inheritdoc />
    public abstract bool RequiresMemoryBank { get; }

    /// <inheritdoc />
    public abstract bool UsesMomentumEncoder { get; }

    /// <summary>
    /// Initializes a new instance of the SelfSupervisedLearningMethodBase class.
    /// </summary>
    /// <param name="encoder">The encoder neural network.</param>
    /// <param name="projector">Optional projection head.</param>
    /// <param name="config">SSL configuration.</param>
    protected SelfSupervisedLearningMethodBase(
        INeuralNetwork<T> encoder,
        IProjectorHead<T>? projector,
        SelfSupervisedLearningConfig<T>? config)
    {
        Guard.NotNull(encoder);
        _encoder = encoder;
        _projector = projector;
        _config = config ?? new SelfSupervisedLearningConfig<T>();
    }

    /// <inheritdoc />
    public INeuralNetwork<T> GetEncoder() => _encoder;

    /// <inheritdoc />
    public SelfSupervisedLearningStepResult<T> TrainStep(Tensor<T> batch, SelfSupervisedLearningAugmentationContext<T>? augmentationContext = null)
    {
        if (batch is null)
        {
            throw new ArgumentNullException(nameof(batch));
        }

        _currentStep++;

        // Set training mode
        _encoder.SetTrainingMode(true);
        _projector?.SetTrainingMode(true);

        // Delegate to implementation-specific training step
        return TrainStepCore(batch, augmentationContext);
    }

    /// <summary>
    /// Implementation-specific training step logic.
    /// </summary>
    /// <param name="batch">The input batch tensor.</param>
    /// <param name="augmentationContext">Optional augmentation context.</param>
    /// <returns>The result of the training step.</returns>
    /// <remarks>
    /// <para>
    /// One tape-based update: the two views are built once (outside the objective, so a re-evaluation replays the same
    /// draws), the method's <see cref="ComputeObjective"/> is recorded on the gradient tape, the online network's
    /// tensors (<see cref="GetOnlineTrainableTensors"/>) are stepped by the optimizer, and
    /// <see cref="AfterOptimizerStep"/> updates whatever follows the online network (a momentum target, a queue, a
    /// center).
    /// </para>
    /// <para>
    /// This used to return a zero loss and change nothing: no method overrode it, so every self-supervised method,
    /// and the pretraining session and pretrainer that call TrainStep, silently skipped training.
    /// </para>
    /// </remarks>
    protected virtual SelfSupervisedLearningStepResult<T> TrainStepCore(Tensor<T> batch, SelfSupervisedLearningAugmentationContext<T>? augmentationContext)
    {
        var (view1, view2) = CreateViews(batch, augmentationContext);
        var optimizer = _trainingOptimizer ??= CreateOptimizer();
        // The configured schedule (linear scaling, warmup, cosine decay) sets this step's rate.
        double learningRate = GetEffectiveLearningRate();
        if (optimizer is AiDotNet.Optimizers.GradientBasedOptimizerBase<T, Tensor<T>, Tensor<T>> scheduled)
            scheduled.SetScheduledLearningRate(learningRate);

        // SGD and Adam take weight decay as an L2 term in the gradient (torch.optim's weight_decay), which is the
        // gradient of (wd / 2) * ||w||^2 added to the objective; the decoupled optimizers apply it themselves.
        double coupledDecay = CoupledWeightDecay();
        T penalty = NumOps.Zero;
        T loss = AiDotNet.Training.TapeTrainingStepper<T>.EagerObjectiveStep(
            GetOnlineTrainableTensors,
            () =>
            {
                var objective = ComputeObjective(view1, view2);
                if (coupledDecay <= 0) return objective;
                var l2 = L2Penalty(GetOnlineTrainableTensors(), coupledDecay);
                penalty = l2.Length > 0 ? l2.GetFlat(0) : NumOps.Zero;
                return Engine.TensorAdd(objective, l2);
            },
            optimizer);
        AfterOptimizerStep();
        // The reported loss is the method's objective, without the weight-decay term.
        return CreateStepResult(NumOps.Subtract(loss, penalty));
    }

    // The weight decay the objective carries: the configured decay for SGD and Adam, which have no decay of their own.
    private double CoupledWeightDecay()
    {
        double wd = _config.WeightDecay ?? 0.0;
        if (wd < 0 || double.IsNaN(wd)) throw new InvalidOperationException($"Weight decay must be non-negative; got {wd}.");
        var type = _config.OptimizerType ?? DefaultOptimizerType;
        return type == SelfSupervisedLearningOptimizerType.SGD || type == SelfSupervisedLearningOptimizerType.Adam ? wd : 0.0;
    }

    // (wd / 2) * sum ||w||^2 over the given tensors, as a one-element tensor the tape records.
    private Tensor<T> L2Penalty(IReadOnlyList<Tensor<T>> tensors, double weightDecay)
    {
        Tensor<T>? total = null;
        foreach (var tensor in tensors)
        {
            var sum = Engine.ReduceSum(Engine.TensorMultiply(tensor, tensor), Enumerable.Range(0, tensor.Rank).ToArray(), keepDims: false);
            var flat = Engine.Reshape(sum, new[] { 1 });
            total = total is null ? flat : Engine.TensorAdd(total, flat);
        }
        return total is null
            ? new Tensor<T>(new[] { 1 })
            : Engine.TensorMultiplyScalar(total, NumOps.FromDouble(weightDecay / 2.0));
    }

    // Created on the first training step, so a method configured after construction still gets its settings.
    [Scratch]
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _trainingOptimizer;

    /// <summary>The rate the training optimizer last stepped at (NaN before the first step).</summary>
    internal double TrainingOptimizerLearningRate => _trainingOptimizer?.GetCurrentLearningRate() ?? double.NaN;

    [Scratch]
    private SelfSupervisedLearningAugmentationPolicies<T>? _viewPolicies;

    /// <summary>
    /// The two views a training step compares. A context that supplies <see cref="SelfSupervisedLearningAugmentationContext{T}.PrecomputedView"/>
    /// pairs the batch with it; otherwise the method's augmentation policy makes both views.
    /// </summary>
    protected virtual (Tensor<T> View1, Tensor<T> View2) CreateViews(
        Tensor<T> batch, SelfSupervisedLearningAugmentationContext<T>? augmentationContext)
    {
        if (augmentationContext?.PrecomputedView is { } precomputed)
            return (batch, precomputed);
        var policies = _viewPolicies ??= new SelfSupervisedLearningAugmentationPolicies<T>(_config.Seed);
        return AugmentViews(policies, batch);
    }

    /// <summary>The method's own augmentation pipeline (SimCLR's by default).</summary>
    protected virtual (Tensor<T> View1, Tensor<T> View2) AugmentViews(
        SelfSupervisedLearningAugmentationPolicies<T> policies, Tensor<T> batch) => policies.ApplySimCLR(batch);

    /// <summary>
    /// The method's training loss for two views of a batch, built from the live online parameters so the tape records
    /// it. Each method states its paper's objective here.
    /// </summary>
    /// <exception cref="NotSupportedException">The method defines no training objective.</exception>
    protected virtual Tensor<T> ComputeObjective(Tensor<T> view1, Tensor<T> view2) =>
        throw new NotSupportedException($"{Name} defines no training objective.");

    /// <summary>
    /// The tensors the optimizer steps: the online encoder's and the projector's. A method with further online heads
    /// (a predictor) adds them; momentum targets are never included, they follow by EMA in <see cref="AfterOptimizerStep"/>.
    /// </summary>
    protected virtual IReadOnlyList<Tensor<T>> GetOnlineTrainableTensors()
    {
        var tensors = new List<Tensor<T>>(EncoderTrainableTensors(_encoder));
        if (_projector is not null) tensors.AddRange(_projector.GetTrainableTensors());
        return tensors;
    }

    /// <summary>
    /// The encoder's training forward, recorded on the gradient tape (Predict runs without it, for inference).
    /// </summary>
    protected Tensor<T> EncodeForTraining(Tensor<T> input) => EncodeForTraining(_encoder, input);

    /// <summary>A network's training forward, recorded on the gradient tape.</summary>
    protected static Tensor<T> EncodeForTraining(INeuralNetwork<T> network, Tensor<T> input) =>
        network is AiDotNet.NeuralNetworks.NeuralNetworkBase<T> trainable
            ? trainable.ForwardForTraining(input)
            : network.ForwardWithMemory(input);

    /// <summary>The trainable tensors of an encoder network, the live objects its forward reads.</summary>
    /// <exception cref="NotSupportedException">The encoder is not a network whose tensors can be stepped.</exception>
    protected static IReadOnlyList<Tensor<T>> EncoderTrainableTensors(INeuralNetwork<T> encoder) =>
        encoder is AiDotNet.NeuralNetworks.NeuralNetworkBase<T> network
            ? network.CollectModelTrainableTensors()
            : throw new NotSupportedException(
                $"A {encoder.GetType().Name} encoder exposes no trainable tensors; self-supervised training needs a NeuralNetworkBase encoder.");

    /// <summary>Runs after the optimizer step: momentum targets, queues and centers update here.</summary>
    protected virtual void AfterOptimizerStep()
    {
    }

    /// <summary>The optimizer the paper trains this method with, unless the configuration names another.</summary>
    protected virtual SelfSupervisedLearningOptimizerType DefaultOptimizerType => SelfSupervisedLearningOptimizerType.AdamW;

    /// <summary>The learning rate when the configuration sets none.</summary>
    protected virtual double DefaultLearningRate => 1e-3;

    /// <summary>Creates the training optimizer from the configuration.</summary>
    protected virtual IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreateOptimizer()
    {
        // The first step's rate; TrainStepCore sets each step's from GetEffectiveLearningRate. SGD and Adam take the
        // weight decay through the objective (see TrainStepCore).
        double lr = GetEffectiveLearningRate();
        double wd = _config.WeightDecay ?? 0.0;
        switch (_config.OptimizerType ?? DefaultOptimizerType)
        {
            case SelfSupervisedLearningOptimizerType.SGD:
                return new AiDotNet.Optimizers.MomentumOptimizer<T, Tensor<T>, Tensor<T>>(null,
                    new AiDotNet.Models.Options.MomentumOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = lr });
            case SelfSupervisedLearningOptimizerType.Adam:
                return new AiDotNet.Optimizers.AdamOptimizer<T, Tensor<T>, Tensor<T>>(null,
                    new AiDotNet.Models.Options.AdamOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = lr });
            case SelfSupervisedLearningOptimizerType.LARS:
                return new AiDotNet.Optimizers.LARSOptimizer<T, Tensor<T>, Tensor<T>>(null,
                    new AiDotNet.Models.Options.LARSOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = lr, WeightDecay = wd });
            case SelfSupervisedLearningOptimizerType.LAMB:
                return new AiDotNet.Optimizers.LAMBOptimizer<T, Tensor<T>, Tensor<T>>(null,
                    new AiDotNet.Models.Options.LAMBOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = lr, WeightDecay = wd });
            default:
                return new AiDotNet.Optimizers.AdamWOptimizer<T, Tensor<T>, Tensor<T>>(null,
                    new AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = lr, WeightDecay = wd });
        }
    }

    /// <inheritdoc />
    public override Tensor<T> Predict(Tensor<T> input) => Encode(input);

    /// <inheritdoc />
    public virtual Tensor<T> Encode(Tensor<T> input)
    {
        if (input is null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        // Set evaluation mode for encoding
        _encoder.SetTrainingMode(false);

        // Get encoder output
        var encoded = _encoder.Predict(input);

        return encoded;
    }

    /// <summary>
    /// Encodes input and projects it to the SSL embedding space.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>The projected embedding.</returns>
    protected virtual Tensor<T> EncodeAndProject(Tensor<T> input)
    {
        var encoded = _encoder.ForwardWithMemory(input);

        if (_projector is not null)
        {
            return _projector.Project(encoded);
        }

        return encoded;
    }

    /// <inheritdoc />
    public virtual void Reset()
    {
        _currentStep = 0;
        _currentEpoch = 0;
        _projector?.Reset();
    }

    /// <summary>
    /// Sets the training mode for the SSL method.
    /// </summary>
    /// <param name="isTraining">True for training mode, false for evaluation.</param>
    public virtual void SetTrainingMode(bool isTraining)
    {
        _isTraining = isTraining;
        _encoder.SetTrainingMode(isTraining);
        _projector?.SetTrainingMode(isTraining);
    }

    /// <summary>
    /// Signals the start of a new epoch.
    /// </summary>
    /// <param name="epochNumber">The current epoch number.</param>
    public virtual void OnEpochStart(int epochNumber)
    {
        _currentEpoch = epochNumber;
    }

    /// <summary>
    /// Signals the end of an epoch.
    /// </summary>
    /// <param name="epochNumber">The completed epoch number.</param>
    public virtual void OnEpochEnd(int epochNumber)
    {
        // Subclasses can override for epoch-end operations
    }

    /// <summary>
    /// Gets the effective temperature based on configuration and scheduling.
    /// </summary>
    /// <returns>The current temperature value.</returns>
    protected virtual double GetEffectiveTemperature()
    {
        return _config.Temperature ?? 0.07;
    }

    /// <summary>
    /// Gets the effective learning rate based on configuration and scheduling.
    /// </summary>
    /// <returns>The current learning rate.</returns>
    public virtual double GetEffectiveLearningRate()
    {
        // The base rate is per 256 samples (the linear scaling rule of the papers: SimCLR's 0.3, MoCo v3's 1.5e-4).
        double baseLr = _config.LearningRate ?? DefaultLearningRate;
        int batchSize = _config.BatchSize ?? 256;
        double effectiveLr = baseLr * batchSize / 256.0;

        // Linear warmup over the first WarmupEpochs, by epoch: epoch k of W runs at (k + 1) / W of the rate, so no
        // epoch trains at zero.
        int warmupEpochs = Math.Max(0, _config.WarmupEpochs ?? 10);
        if (_currentEpoch < warmupEpochs)
            return effectiveLr * (_currentEpoch + 1) / warmupEpochs;

        // Then cosine decay to zero over the remaining epochs (MoCo v3's schedule).
        if (_config.UseCosineDecay ?? true)
        {
            int totalEpochs = _config.PretrainingEpochs ?? 100;
            int decayEpochs = Math.Max(1, totalEpochs - warmupEpochs);
            double progress = Math.Min(1.0, (double)(_currentEpoch - warmupEpochs) / decayEpochs);
            effectiveLr *= 0.5 * (1.0 + Math.Cos(Math.PI * progress));
        }

        return effectiveLr;
    }

    /// <summary>
    /// Creates a default step result with common metrics.
    /// </summary>
    /// <param name="loss">The loss value.</param>
    /// <returns>A step result with populated common fields.</returns>
    protected SelfSupervisedLearningStepResult<T> CreateStepResult(T loss)
    {
        return new SelfSupervisedLearningStepResult<T>
        {
            Loss = loss,
            CurrentLearningRate = GetEffectiveLearningRate(),
            CurrentTemperature = GetEffectiveTemperature(),
            Metrics = []
        };
    }

    /// <summary>
    /// Computes cosine similarity between two tensors.
    /// </summary>
    /// <param name="a">First tensor [batch, dim].</param>
    /// <param name="b">Second tensor [batch, dim].</param>
    /// <returns>Cosine similarity values [batch].</returns>
    protected virtual Tensor<T> CosineSimilarity(Tensor<T> a, Tensor<T> b)
    {
        // Normalize both tensors
        var aNorm = L2Normalize(a);
        var bNorm = L2Normalize(b);

        // Compute dot product along last dimension: element-wise multiply then ReduceSum along dim 1
        var product = Engine.TensorMultiply(aNorm, bNorm);
        return Engine.ReduceSum(product, [1], keepDims: false);
    }

    /// <summary>
    /// L2-normalizes a tensor along the last dimension.
    /// </summary>
    /// <param name="tensor">The tensor to normalize [batch, dim].</param>
    /// <returns>The normalized tensor.</returns>
    protected virtual Tensor<T> L2Normalize(Tensor<T> tensor)
    {
        var batchSize = tensor.Shape[0];
        var dim = tensor.Shape[1];
        var result = new T[batchSize * dim];

        for (int i = 0; i < batchSize; i++)
        {
            // Use Engine for accelerated vector operations when available
            var row = new T[dim];
            for (int j = 0; j < dim; j++)
            {
                row[j] = tensor[i, j];
            }

            var vec = new Vector<T>(row);

            // Compute L2 norm using engine's dot product
            var normSquared = Engine.DotProduct(vec, vec);
            var norm = NumOps.Sqrt(NumOps.Add(normSquared, NumOps.FromDouble(1e-8)));

            // Normalize
            for (int j = 0; j < dim; j++)
            {
                result[i * dim + j] = NumOps.Divide(tensor[i, j], norm);
            }
        }

        return new Tensor<T>(result, [batchSize, dim]);
    }

    /// <summary>
    /// Computes matrix multiplication with engine-accelerated dot products.
    /// </summary>
    /// <param name="a">First matrix [M, K].</param>
    /// <param name="b">Second matrix [K, N].</param>
    /// <returns>Result matrix [M, N].</returns>
    protected virtual Tensor<T> MatMul(Tensor<T> a, Tensor<T> b)
    {
        var m = a.Shape[0];
        var k = a.Shape[1];
        var n = b.Shape[1];
        var result = new T[m * n];

        // Transpose b for better cache locality: [K, N] -> column vectors of length K
        for (int i = 0; i < m; i++)
        {
            // Extract row i from a
            var rowA = new T[k];
            for (int j = 0; j < k; j++)
            {
                rowA[j] = a[i, j];
            }
            var vecA = new Vector<T>(rowA);

            for (int j = 0; j < n; j++)
            {
                // Extract column j from b
                var colB = new T[k];
                for (int l = 0; l < k; l++)
                {
                    colB[l] = b[l, j];
                }
                var vecB = new Vector<T>(colB);

                // Use engine for accelerated dot product
                result[i * n + j] = Engine.DotProduct(vecA, vecB);
            }
        }

        return new Tensor<T>(result, [m, n]);
    }

    /// <summary>
    /// Computes similarity matrix between two sets of embeddings.
    /// </summary>
    /// <param name="embeddings1">First set of embeddings [N, D].</param>
    /// <param name="embeddings2">Second set of embeddings [M, D].</param>
    /// <param name="normalize">Whether to L2-normalize before computing similarity.</param>
    /// <returns>Similarity matrix [N, M].</returns>
    protected virtual Tensor<T> ComputeSimilarityMatrix(
        Tensor<T> embeddings1,
        Tensor<T> embeddings2,
        bool normalize = true)
    {
        var n = embeddings1.Shape[0];
        var m = embeddings2.Shape[0];
        var d = embeddings1.Shape[1];

        var e1 = normalize ? L2Normalize(embeddings1) : embeddings1;
        var e2 = normalize ? L2Normalize(embeddings2) : embeddings2;

        var result = new T[n * m];

        // Compute dot products using engine
        for (int i = 0; i < n; i++)
        {
            var row1 = new T[d];
            for (int k = 0; k < d; k++)
            {
                row1[k] = e1[i, k];
            }
            var vec1 = new Vector<T>(row1);

            for (int j = 0; j < m; j++)
            {
                var row2 = new T[d];
                for (int k = 0; k < d; k++)
                {
                    row2[k] = e2[j, k];
                }
                var vec2 = new Vector<T>(row2);

                result[i * m + j] = Engine.DotProduct(vec1, vec2);
            }
        }

        return new Tensor<T>(result, [n, m]);
    }

    /// <summary>
    /// Computes the pairwise squared distances between embeddings.
    /// </summary>
    /// <param name="embeddings">Embeddings [N, D].</param>
    /// <returns>Distance matrix [N, N].</returns>
    protected virtual Tensor<T> ComputePairwiseDistances(Tensor<T> embeddings)
    {
        var n = embeddings.Shape[0];
        var d = embeddings.Shape[1];
        var result = new T[n * n];

        // ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a.b
        // First compute norms using engine
        var norms = new T[n];
        for (int i = 0; i < n; i++)
        {
            var row = new T[d];
            for (int j = 0; j < d; j++)
            {
                row[j] = embeddings[i, j];
            }
            var vec = new Vector<T>(row);
            norms[i] = Engine.DotProduct(vec, vec);
        }

        // Compute similarity matrix using engine
        var similarity = ComputeSimilarityMatrix(embeddings, embeddings, normalize: false);

        // Compute distances: ||a||^2 + ||b||^2 - 2 * a.b
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                var dist = NumOps.Subtract(
                    NumOps.Add(norms[i], norms[j]),
                    NumOps.Multiply(NumOps.FromDouble(2.0), similarity[i, j]));
                // Clamp negative values due to floating point errors
                result[i * n + j] = NumOps.GreaterThan(dist, NumOps.Zero) ? dist : NumOps.Zero;
            }
        }

        return new Tensor<T>(result, [n, n]);
    }

    // === ModelBase abstract implementations ===

    /// <summary>
    /// Trains the SSL method. For self-supervised learning, expectedOutput is ignored
    /// since learning is unsupervised (the model creates its own supervision).
    /// </summary>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        TrainStep(input);
    }

    /// <inheritdoc/>
    public override ILossFunction<T> DefaultLossFunction =>
        new MeanSquaredErrorLoss<T>();

    /// <inheritdoc/>
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        return (SelfSupervisedLearningMethodBase<T>)MemberwiseClone();
    }

    /// <inheritdoc/>
    public override IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters)
    {
        var clone = DeepCopy();
        ((IParameterizable<T, Tensor<T>, Tensor<T>>)clone).SetParameters(parameters);
        return clone;
    }
}

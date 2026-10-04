using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.ComputerVision;

/// <summary>
/// The training and state plumbing the vision task families share: object detectors, text detectors and text
/// recognisers (<c>ObjectDetectorBase</c>, <c>TextDetectorBase</c>, <c>OCRBase</c>).
/// </summary>
/// <remarks>
/// <para>
/// Each family used to carry its own copy of the following:
/// <list type="bullet">
/// <item>The raw-output training step.</item>
/// <item>The typed-target step: annotations, the model's target assignment and the model's loss.</item>
/// <item>The last-loss record.</item>
/// <item>The training-mode flag.</item>
/// <item>The input-shape memory that lets a rebuilt copy size its lazy layers before its state is
/// restored.</item>
/// </list>
/// Only the object detectors had the typed step, so text detection and recognition could not train on
/// annotations at all.
/// </para>
/// <para>
/// A family supplies two hooks. <see cref="TrainingForward"/> is the forward a raw step regresses; it
/// defaults to <see cref="ModelBase{T, TInput, TOutput}.Predict"/>, and recognisers use their logits.
/// <see cref="DeferredParameterProbeShape"/> is the zero input that resolves shape-deferred layers.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class VisionTaskModelBase<T> : ModelBase<T, Tensor<T>, Tensor<T>>
{
    /// <summary>Whether the model is in training mode (dropout and batch statistics active).</summary>
    protected bool IsTrainingMode;

    /// <summary>
    /// The shape of the first input this model's forward pass ran on. Its lazily-shaped layers sized their
    /// weights from it, so replaying it on a rebuilt copy reproduces the same parameter topology. Scratch:
    /// never persisted, and rebuilt copies record their own.
    /// </summary>
    [AiDotNet.Attributes.Scratch]
    private int[]? _resolvedInputShape;

    /// <summary>The loss of the most recent training step, measured before its update.</summary>
    [AiDotNet.Attributes.Scratch]
    private T _lastTrainingLoss = MathHelper.GetNumericOperations<T>().Zero;

    /// <summary>Switches training mode. Families override this to propagate the mode to their sub-networks.</summary>
    /// <param name="training">True for training mode, false for inference.</param>
    public virtual void SetTrainingMode(bool training)
    {
        IsTrainingMode = training;
    }

    /// <summary>The step size of the plain-SGD update used when <see cref="CreateTrainingOptimizer"/> returns null. Defaults to 0.001.</summary>
    protected virtual double TrainingLearningRate => 0.001;

    /// <summary>
    /// The optimizer the training steps apply, created on first use and kept for the life of the model, so its
    /// state (momentum, Adam moments) carries from step to step. Scratch: a rebuilt copy starts fresh.
    /// </summary>
    [AiDotNet.Attributes.Scratch]
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _trainingOptimizer;

    /// <summary>
    /// The model's training optimizer, which should be its paper's. Null means plain SGD at
    /// <see cref="TrainingLearningRate"/>. Use <see cref="PaperAdam"/>, <see cref="PaperAdamW"/>,
    /// <see cref="PaperSgdMomentum"/> or <see cref="PaperAdadelta"/>: they build the optimizer as the paper
    /// states it, with no adaptive learning rate, adaptive betas or gradient clipping, none of which the
    /// papers use.
    /// </summary>
    protected virtual IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? CreateTrainingOptimizer() => null;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer => _trainingOptimizer ??= CreateTrainingOptimizer();

    /// <summary>Adam (Kingma and Ba 2015) with betas (0.9, 0.999) and epsilon 1e-8.</summary>
    protected IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> PaperAdam(double learningRate)
        => new AiDotNet.Optimizers.AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AiDotNet.Models.Options.AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = learningRate,
                UseAMSGrad = false,
                UseAdaptiveBetas = false,
                UseAdaptiveLearningRate = false,
                EnableGradientClipping = false,
            });

    /// <summary>AdamW (Loshchilov and Hutter 2019) with betas (0.9, 0.999), epsilon 1e-8 and decoupled weight decay.</summary>
    protected IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> PaperAdamW(double learningRate, double weightDecay)
        => new AiDotNet.Optimizers.AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = learningRate,
                WeightDecay = weightDecay,
                UseAMSGrad = false,
                UseAdaptiveBetas = false,
                UseAdaptiveLearningRate = false,
                EnableGradientClipping = false,
            });

    /// <summary>SGD with heavy-ball momentum at a fixed coefficient.</summary>
    protected IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> PaperSgdMomentum(double learningRate, double momentum)
        => new AiDotNet.Optimizers.MomentumOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AiDotNet.Models.Options.MomentumOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = learningRate,
                InitialMomentum = momentum,
                UseAdaptiveMomentum = false,
                UseAdaptiveLearningRate = false,
                EnableGradientClipping = false,
            });

    /// <summary>ADADELTA (Zeiler 2012) at a fixed decay <paramref name="rho"/>. It has no learning rate (1.0).</summary>
    protected IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> PaperAdadelta(double rho, double epsilon)
        => new AiDotNet.Optimizers.AdaDeltaOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AiDotNet.Models.Options.AdaDeltaOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = 1.0,
                Rho = rho,
                Epsilon = epsilon,
                UseAdaptiveRho = false,
                UseAdaptiveLearningRate = false,
                EnableGradientClipping = false,
            });

    /// <summary>Gets the number of channels in the images this model reads. RGB unless a model overrides it.</summary>
    protected virtual int InputChannels => 3;

    /// <summary>The forward a raw-output <see cref="Train"/> step regresses. Defaults to <c>Predict</c>.</summary>
    protected virtual Tensor<T> TrainingForward(Tensor<T> input) => Predict(input);

    /// <summary>
    /// The zero input, <c>[1, channels, height, width]</c>, whose forward pass resolves this model's
    /// shape-deferred layers exactly as its first real input would.
    /// </summary>
    protected abstract int[] DeferredParameterProbeShape { get; }

    /// <summary>
    /// Runs one training step against the raw output of <see cref="TrainingForward"/>.
    /// </summary>
    /// <param name="input">The training image.</param>
    /// <param name="expectedOutput">The desired output, shaped like the training forward's output.</param>
    /// <remarks>
    /// The step records the forward pass on a gradient tape, takes mean squared error against
    /// <paramref name="expectedOutput"/>, and applies a stochastic-gradient update to every trainable
    /// tensor reachable from this model. This is raw-output regression, not task training. A family's
    /// typed API (annotations, the model's target assignment and its paper's loss) uses
    /// <see cref="TrainWithTargets{TPrediction, TTarget}"/>.
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (expectedOutput is null) throw new ArgumentNullException(nameof(expectedOutput));
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(true);
        try
        {
            RecordTrainingLoss(TensorModelTrainer<T>.Step(
                this, input, expectedOutput, NumOps.FromDouble(TrainingLearningRate), TrainingForward, optimizer: TrainingOptimizer));
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <summary>
    /// One training step on typed targets. <paramref name="forward"/> produces the heads the loss reads, and
    /// <paramref name="loss"/> scores them against <paramref name="targets"/>. The step uses the same single
    /// tape-and-update path as <see cref="Train"/>.
    /// </summary>
    /// <remarks>The derived model validates its task targets before calling this method.</remarks>
    protected void TrainWithTargets<TPrediction, TTarget>(Tensor<T> input, TTarget targets,
        Func<Tensor<T>, TPrediction> forward, Func<TPrediction, TTarget, Tensor<T>> loss) where TTarget : class
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (forward is null) throw new ArgumentNullException(nameof(forward));
        if (loss is null) throw new ArgumentNullException(nameof(loss));
        NoteResolvedInput(input);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(true);
        try
        {
            RecordTrainingLoss(TensorModelTrainer<T>.StepWithTargets(
                this, input, targets, NumOps.FromDouble(TrainingLearningRate), forward, loss, TrainingOptimizer));
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <inheritdoc />
    public override ILossFunction<T> DefaultLossFunction => new MeanSquaredErrorLoss<T>();

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters)
    {
        var copy = DeepCopy();
        InterfaceGuard.Parameterizable(copy).SetParameters(parameters);
        return copy;
    }

    // DeepCopy is deliberately NOT overridden. MemberwiseClone gave a shallow copy that shared every layer with
    // the original, so fine-tuning a clone rewrote the source's weights. ModelBase rebuilds the model from its
    // recorded constructor and reloads state, which is correct.

    /// <summary>Records the input shape on the first forward pass.</summary>
    protected void NoteResolvedInput(Tensor<T> input)
    {
        if (_resolvedInputShape is not null || input is null)
        {
            return;
        }

        var shape = new int[input.Shape.Length];
        for (int i = 0; i < shape.Length; i++)
        {
            shape[i] = input.Shape[i];
        }

        _resolvedInputShape = shape;
    }

    /// <inheritdoc />
    /// <remarks>
    /// Runs the copy once on a zero input of the shape this model has already processed, so its lazily-shaped
    /// layers size their weights exactly as this model's did before its state is loaded into them.
    /// </remarks>
    protected override void PrepareCopyForStateRestore(ModelBase<T, Tensor<T>, Tensor<T>> copy)
    {
        if (_resolvedInputShape is not null && copy is VisionTaskModelBase<T> rebuilt)
        {
            var shape = (int[])_resolvedInputShape.Clone();
            shape[0] = 1;
            rebuilt.Predict(new Tensor<T>(shape));
        }
    }

    /// <inheritdoc />
    /// <remarks>
    /// Several layers size their weights on their first forward pass. Until then the model reports its
    /// parameters as shape-deferred, which made <see cref="Serialize"/> (and therefore <c>Clone</c>) throw on a
    /// freshly constructed model. Running the network once on <see cref="DeferredParameterProbeShape"/>
    /// resolves exactly the shapes the first real input would.
    /// </remarks>
    public override byte[] Serialize()
    {
        if (_resolvedInputShape is null)
            Predict(new Tensor<T>(DeferredParameterProbeShape));
        return base.Serialize();
    }

    /// <summary>
    /// Gets the loss of the most recent training step, measured on that step's input before its update (zero
    /// before the first step).
    /// </summary>
    /// <returns>The training objective's value: mean squared error, or the model's own loss where it has one.</returns>
    /// <remarks>Same contract as <c>INeuralNetwork&lt;T&gt;.GetLastLoss</c>.</remarks>
    public T GetLastLoss() => _lastTrainingLoss;

    /// <summary>Records the loss a training step reported.</summary>
    /// <param name="loss">The step's loss.</param>
    protected void RecordTrainingLoss(T loss) => _lastTrainingLoss = loss;
}

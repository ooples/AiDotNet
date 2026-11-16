using AiDotNet.Attributes;
using AiDotNet.Augmentation.Tabular;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AdversarialRobustness.Attacks;

/// <summary>
/// Implements AutoAttack, the parameter-free attack ensemble of Croce and Hein (2020).
/// </summary>
/// <remarks>
/// <para>
/// The standard version runs APGD-CE (auto-PGD on the cross-entropy loss), APGD-T (targeted auto-PGD on the
/// DLR loss, over the top 9 target classes), FAB-T (targeted fast adaptive boundary attack) and the Square
/// attack (5000 queries). The plus version runs APGD-CE and APGD-DLR with 5 restarts, FAB with 5 restarts,
/// Square, APGD-T and FAB-T. The attacks run in that order, and each one only attacks an input that every
/// earlier attack failed on: an input is robust only if every attack fails (worst case over attacks), as in
/// the reference implementation (fra31/auto-attack).
/// </para>
/// <para><b>For Beginners:</b> AutoAttack is like having multiple expert attackers work together.
/// Each attacker tries in turn; as soon as one of them fools the model, that adversarial example is
/// returned. If your model resists all of them, it is genuinely robust.</para>
/// <para>
/// AutoAttack is an untargeted evaluation: its targeted members only use target classes to search for any
/// misclassification, and inputs the model already misclassifies are returned unchanged. With
/// <see cref="AdversarialAttackOptions{T}.IsTargeted"/> set, it runs its targeted members (APGD-T, then FAB-T)
/// against <see cref="AdversarialAttackOptions{T}.TargetClass"/> only, an attack succeeds when the model
/// predicts that class, and inputs already predicted as it are returned unchanged.
/// <see cref="AdversarialAttackOptions{T}.UseRandomStart"/> chooses whether APGD starts from a random point of
/// the ball (the reference default) or from the clean input. The step size is APGD's own (2 epsilon, halved
/// adaptively), so <see cref="AdversarialAttackOptions{T}.StepSize"/> is not used. For inputs without spatial
/// layout (vectors), the Square attack uses intervals covering the same fraction of the input instead of squares.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type for the model.</typeparam>
/// <typeparam name="TOutput">The output data type for the model.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelCategory(ModelCategory.AnomalyDetection)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-Free Attacks", "https://arxiv.org/abs/2003.01690", Year = 2020, Authors = "Francesco Croce, Matthias Hein")]
public partial class AutoAttack<T, TInput, TOutput> : AdversarialAttackBase<T, TInput, TOutput>
{
    /// <summary>
    /// The attack's settings: the options object that GetOptions returns and that persistence writes, so a
    /// change made through either is the one the attack runs with.
    /// </summary>
    private AutoAttackOptions<T> Settings => Options as AutoAttackOptions<T> ?? AsAutoAttackOptions(Options);

    /// <summary>
    /// Initializes a new instance of AutoAttack.
    /// </summary>
    /// <param name="options">
    /// The configuration options. Pass an <see cref="AutoAttackOptions{T}"/> to choose the version and the
    /// per-attack budgets; plain attack options use the paper's budgets with their epsilon, norm, iteration
    /// count and seed.
    /// </param>
    public AutoAttack(AdversarialAttackOptions<T> options) : base(AsAutoAttackOptions(options))
    {
    }

    /// <summary>
    /// The options as AutoAttack options: the same object when they already are, otherwise one copy, made once
    /// at construction, with the paper's budgets and every shared attack setting of <paramref name="options"/>.
    /// </summary>
    private static AutoAttackOptions<T> AsAutoAttackOptions(AdversarialAttackOptions<T> options)
    {
        if (options is null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        return options as AutoAttackOptions<T> ?? new AutoAttackOptions<T>
        {
            Epsilon = options.Epsilon,
            StepSize = options.StepSize,
            Iterations = options.Iterations,
            NormType = options.NormType,
            IsTargeted = options.IsTargeted,
            TargetClass = options.TargetClass,
            UseRandomStart = options.UseRandomStart,
            RandomSeed = options.RandomSeed,
            CWConstant = options.CWConstant,
            CWConfidence = options.CWConfidence,
            CWLearningRate = options.CWLearningRate
        };
    }

    /// <summary>The attacks of the ensemble.</summary>
    private enum Stage
    {
        ApgdCrossEntropy,
        ApgdDlr,
        ApgdTargeted,
        Fab,
        FabTargeted,
        Square
    }

    /// <summary>The losses APGD maximizes.</summary>
    internal enum LossKind
    {
        CrossEntropy,
        Dlr,
        TargetedDlr
    }

    /// <summary>
    /// Generates an adversarial example using the AutoAttack ensemble.
    /// </summary>
    /// <param name="input">The clean input to perturb.</param>
    /// <param name="trueLabel">The correct label for the input.</param>
    /// <param name="targetModel">The model to attack.</param>
    /// <returns>The first successful adversarial example, or the input when every attack fails.</returns>
    public override TInput GenerateAdversarialExample(TInput input, TOutput trueLabel, IFullModel<T, TInput, TOutput> targetModel)
    {
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        if (targetModel == null)
        {
            throw new ArgumentNullException(nameof(targetModel));
        }

        var vectorInput = ConversionsHelper.ConvertToVector<T, TInput>(input);
        int label = GetClassIndex(ConversionsHelper.ConvertToVector<T, TOutput>(trueLabel));
        var point = new AttackPoint(vectorInput, label, input, targetModel, input as Tensor<T>);
        var cleanLogits = point.Logits(vectorInput);
        if (Settings.IsTargeted)
        {
            if (Settings.TargetClass < 0 || Settings.TargetClass >= cleanLogits.Length)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(AdversarialAttackOptions<T>.TargetClass),
                    Settings.TargetClass,
                    "The target class must be a class index of the model output.");
            }

            // The targeted losses compare the target with the source class. Callers that follow the targeted
            // convention of the other attacks pass the target as the label, which made APGD-T's z_y - z_t and
            // FAB-T's boundary normal identically zero; the source class is the clean prediction.
            point.Label = ArgMaxIndex(cleanLogits);
            point.Goal = Settings.TargetClass;
        }

        if (Fools(cleanLogits, point))
        {
            // Already misclassified (or already the target): nothing to attack.
            return input;
        }

        foreach (var stage in Schedule())
        {
            var adversarial = RunStage(stage, point, cleanLogits);
            if (adversarial is not null)
            {
                return ConversionsHelper.ConvertVectorToInput<T, TInput>(adversarial, input);
            }
        }

        return input;
    }

    /// <summary>
    /// The attack order of the chosen version (autoattack.py, run_standard_evaluation).
    /// </summary>
    private Stage[] Schedule()
    {
        if (Settings.IsTargeted)
        {
            return new[] { Stage.ApgdTargeted, Stage.FabTargeted };
        }

        return Settings.Version == AutoAttackVersion.Plus
            ? new[] { Stage.ApgdCrossEntropy, Stage.ApgdDlr, Stage.Fab, Stage.Square, Stage.ApgdTargeted, Stage.FabTargeted }
            : new[] { Stage.ApgdCrossEntropy, Stage.ApgdTargeted, Stage.FabTargeted, Stage.Square };
    }

    /// <summary>
    /// Runs one attack of the ensemble; returns an adversarial example that fools the model, or null.
    /// </summary>
    private Vector<T>? RunStage(Stage stage, AttackPoint point, Vector<T> cleanLogits)
    {
        // Restarts as configured by autoattack.py: 5 for the untargeted APGD/FAB in the plus version, else 1.
        int restarts = Settings.Version == AutoAttackVersion.Plus ? 5 : 1;
        switch (stage)
        {
            case Stage.ApgdCrossEntropy:
                return Apgd(point, LossKind.CrossEntropy, -1, restarts);
            case Stage.ApgdDlr:
                return Apgd(point, LossKind.Dlr, -1, restarts);
            case Stage.ApgdTargeted:
                return ApgdTargeted(point, cleanLogits);
            case Stage.Fab:
                return Fab(point, cleanLogits, targeted: false, restarts);
            case Stage.FabTargeted:
                return Fab(point, cleanLogits, targeted: true, 1);
            default:
                return Square(point);
        }
    }

    /// <summary>
    /// The highest-scoring wrong classes of the clean input, in decreasing order of their logits (ties by
    /// class index), or the configured target class of a targeted attack (<paramref name="goal"/> >= 0).
    /// </summary>
    /// <remarks>
    /// The true label is excluded by index, not by position: when another class ties with it at the top, the
    /// label need not come first in the ranking.
    /// </remarks>
    internal int[] TargetClassesOf(Vector<T> cleanLogits, int label, int goal)
    {
        if (goal >= 0)
        {
            return new[] { goal };
        }

        return RankDescending(cleanLogits).Where(c => c != label).Take(Math.Max(Settings.TargetClasses, 0)).ToArray();
    }

    #region Shared helpers

    /// <summary>
    /// The clean input, its label and the model under attack.
    /// </summary>
    private sealed class AttackPoint
    {
        private readonly TInput _reference;

        public AttackPoint(Vector<T> clean, int label, TInput reference, IFullModel<T, TInput, TOutput> model, Tensor<T>? shaped)
        {
            Clean = clean;
            Label = label;
            _reference = reference;
            Model = model;

            // Spatial layout for the Square attack: the last two axes are height and width; the rest are
            // channels. Inputs without at least two axes are treated as one row.
            Channels = 1;
            Height = 1;
            Width = clean.Length;
            if (shaped is not null && shaped.Rank >= 2 && shaped.Length == clean.Length)
            {
                var shape = shaped._shape;
                Height = shape[shape.Length - 2];
                Width = shape[shape.Length - 1];
                Channels = Math.Max(clean.Length / Math.Max(Height * Width, 1), 1);
            }
        }

        public Vector<T> Clean { get; }

        /// <summary>The source class: the true label, or the clean prediction in a targeted run.</summary>
        public int Label { get; set; }

        /// <summary>The target class of a targeted run, or -1.</summary>
        public int Goal { get; set; } = -1;

        public IFullModel<T, TInput, TOutput> Model { get; }

        public int Channels { get; }

        public int Height { get; }

        public int Width { get; }

        public TInput ToInput(Vector<T> vector) => ConversionsHelper.ConvertVectorToInput<T, TInput>(vector, _reference);

        public TInput Reference => _reference;

        public Vector<T> Logits(Vector<T> vector)
            => ConversionsHelper.ConvertToVector<T, TOutput>(Model.Predict(ToInput(vector)));
    }

    private T Epsilon => NumOps.FromDouble(Settings.Epsilon);

    private bool IsL2 => Settings.NormType == AdversarialNormType.L2;

    /// <summary>
    /// A small positive constant of T: the requested value, or the library's stability epsilon where the
    /// value underflows to zero in T (Half).
    /// </summary>
    private static T Tiny(double value)
    {
        var tiny = NumOps.FromDouble(value);
        return NumOps.GreaterThan(tiny, NumOps.Zero) ? tiny : NumOps.FromDouble(NumericalStabilityHelper.DefaultEpsilon);
    }

    private static T MaxOf(T a, T b) => NumOps.GreaterThan(a, b) ? a : b;

    private static T MinOf(T a, T b) => NumOps.LessThan(a, b) ? a : b;

    private static T At(Vector<T> vector, int index) => vector.AsSpan()[index];

    /// <summary>
    /// Whether the logits meet the attack's goal: a class other than the label, or the target of a targeted run.
    /// </summary>
    private bool Fools(Vector<T> logits, AttackPoint point)
    {
        int predicted = ArgMaxIndex(logits);
        return point.Goal >= 0 ? predicted == point.Goal : predicted != point.Label;
    }

    /// <summary>
    /// Class indices ordered by decreasing logit; ties keep increasing class index (the engine sort is not
    /// stable).
    /// </summary>
    private static int[] RankDescending(Vector<T> logits)
    {
        return TabularEngineOps<T>.StableOrder(logits.AsTensor(), true);
    }

    /// <summary>
    /// The norm of the threat model: L-infinity or L2.
    /// </summary>
    private T ThreatNorm(Vector<T> vector)
    {
        return IsL2 ? ComputeL2Norm(vector) : ComputeLInfinityNorm(vector);
    }

    /// <summary>
    /// v / (||v|| + 1e-12) in the threat-model norm (autoattack normalize).
    /// </summary>
    /// <remarks>
    /// The components are divided by the guarded norm; its reciprocal is not formed first. In Half the guard is
    /// 1e-7, whose reciprocal is infinite, so a zero vector became 0 * infinity = NaN instead of zero.
    /// </remarks>
    internal Vector<T> NormalizeThreat(Vector<T> vector)
    {
        return Engine.Divide<T>(vector, NumOps.Add(ThreatNorm(vector), Tiny(1e-12)));
    }

    /// <summary>
    /// Projects onto the epsilon-ball around the clean input intersected with the box [0, 1].
    /// </summary>
    private Vector<T> Project(Vector<T> clean, Vector<T> candidate)
    {
        if (IsL2)
        {
            var delta = Engine.Subtract<T>(candidate, clean);
            T norm = ComputeL2Norm(delta);
            T scale = NumOps.Divide(MinOf(Epsilon, norm), NumOps.Add(norm, Tiny(1e-12)));
            return Engine.Clamp<T>(Engine.Add<T>(clean, Engine.Multiply<T>(delta, scale)), NumOps.Zero, NumOps.One);
        }

        var lower = Engine.Subtract<T>(clean, Engine.Fill<T>(clean.Length, Epsilon));
        var upper = Engine.Add<T>(clean, Engine.Fill<T>(clean.Length, Epsilon));
        var inBall = Engine.Min<T>(Engine.Max<T>(candidate, lower), upper);
        return Engine.Clamp<T>(inBall, NumOps.Zero, NumOps.One);
    }

    /// <summary>
    /// d (weights . logits(x)) / dx: the input gradient of a fixed linear combination of the logits.
    /// </summary>
    /// <remarks>
    /// With weights = d loss / d logits this is the chain rule for the loss gradient. Neural networks are
    /// differentiated on the tape; other models with central finite differences.
    /// </remarks>
    private Vector<T> InputGradient(AttackPoint point, Vector<T> vector, Vector<T> weights)
    {
        if (point.Model is NeuralNetworks.NeuralNetworkBase<T> network)
        {
            var inputTensor = point.ToInput(vector) as Tensor<T> ?? Tensor<T>.FromVector(vector);
            using var tape = new GradientTape<T>();
            var output = network.ForwardForTraining(inputTensor);
            var shapedWeights = Engine.Reshape(Tensor<T>.FromVector(weights), output.Shape.ToArray());
            var product = Engine.TensorMultiply(output, shapedWeights);
            var allAxes = Enumerable.Range(0, product.Shape.Length).ToArray();
            var loss = Engine.ReduceSum(product, allAxes, keepDims: false);
            var gradients = tape.ComputeGradients(loss, [inputTensor]);
            if (!gradients.TryGetValue(inputTensor, out var gradient))
            {
                throw new InvalidOperationException("AutoAttack: gradient computation failed - no gradient for the input tensor.");
            }

            return gradient.ToVector();
        }

        return EstimateInputGradient(
            vector,
            point.Reference,
            point.Model,
            output => Engine.DotProduct<T>(output, weights),
            NumOps.FromDouble(1e-4));
    }

    /// <summary>
    /// The value of an APGD loss and its gradient with respect to the logits.
    /// </summary>
    internal (T Value, Vector<T> LogitGradient) Loss(Vector<T> logits, LossKind kind, int label, int target)
    {
        int n = logits.Length;
        if (kind == LossKind.CrossEntropy)
        {
            // CE = logsumexp(z) - z_y; d CE / d z = softmax(z) - e_y.
            T logSumExp = Engine.TensorLogSumExp(logits.AsTensor()).GetFlat(0);
            T value = NumOps.Subtract(logSumExp, At(logits, label));
            var gradient = Engine.Subtract<T>(Engine.Softmax<T>(logits), Vector<T>.CreateStandardBasis(n, label));
            return (value, gradient);
        }

        var ranking = RankDescending(logits);
        int Rank(int position) => ranking[Math.Min(position, n - 1)];
        int top = Rank(0);

        Vector<T> numeratorGradient;
        Vector<T> denominatorGradient;
        T numerator;
        T denominator;
        if (kind == LossKind.Dlr)
        {
            // DLR = -(z_y - max_{i != y} z_i) / (z_pi1 - z_pi3 + 1e-12) (Croce and Hein 2020, Eq. 5).
            int other = top == label ? Rank(1) : top;
            int third = Rank(2);
            numerator = NumOps.Subtract(At(logits, label), At(logits, other));
            denominator = NumOps.Add(NumOps.Subtract(At(logits, top), At(logits, third)), Tiny(1e-12));
            numeratorGradient = Engine.Subtract<T>(Vector<T>.CreateStandardBasis(n, label), Vector<T>.CreateStandardBasis(n, other));
            denominatorGradient = Engine.Subtract<T>(Vector<T>.CreateStandardBasis(n, top), Vector<T>.CreateStandardBasis(n, third));
        }
        else
        {
            // Targeted DLR = -(z_y - z_t) / (z_pi1 - (z_pi3 + z_pi4) / 2 + 1e-12) (Eq. 6).
            int third = Rank(2);
            int fourth = Rank(3);
            T half = NumOps.FromDouble(0.5);
            numerator = NumOps.Subtract(At(logits, label), At(logits, target));
            T tail = NumOps.Multiply(half, NumOps.Add(At(logits, third), At(logits, fourth)));
            denominator = NumOps.Add(NumOps.Subtract(At(logits, top), tail), Tiny(1e-12));
            numeratorGradient = Engine.Subtract<T>(Vector<T>.CreateStandardBasis(n, label), Vector<T>.CreateStandardBasis(n, target));
            var tailGradient = Engine.Multiply<T>(
                Engine.Add<T>(Vector<T>.CreateStandardBasis(n, third), Vector<T>.CreateStandardBasis(n, fourth)), half);
            denominatorGradient = Engine.Subtract<T>(Vector<T>.CreateStandardBasis(n, top), tailGradient);
        }

        // L = -N / D; dL/dz = -dN / D + (N / D) / D * dD (the ratio first, so N / D^2 cannot overflow).
        T ratio = NumOps.Divide(numerator, denominator);
        T lossValue = NumOps.Negate(ratio);
        var logitGradient = Engine.Add<T>(
            Engine.Multiply<T>(numeratorGradient, NumOps.Negate(NumOps.Divide(NumOps.One, denominator))),
            Engine.Multiply<T>(denominatorGradient, NumOps.Divide(ratio, denominator)));
        return (lossValue, logitGradient);
    }

    /// <summary>
    /// Draws signs (-1 or +1) of the given shape from the attack's seeded stream (autoattack random_choice).
    /// </summary>
    private Tensor<T> RandomSigns(int[] shape)
    {
        var uniform = Engine.TensorRandomUniformRange(shape, NumOps.Negate(NumOps.One), NumOps.One, Random.Next());
        return Engine.TensorSign(uniform);
    }

    #endregion

    #region APGD

    /// <summary>
    /// APGD with restarts: the first restart that fools the model wins.
    /// </summary>
    private Vector<T>? Apgd(AttackPoint point, LossKind loss, int target, int restarts)
    {
        for (int restart = 0; restart < restarts; restart++)
        {
            var adversarial = RunApgd(point, loss, target);
            if (adversarial is not null)
            {
                return adversarial;
            }
        }

        return null;
    }

    /// <summary>
    /// APGD-T: targeted DLR APGD against each of the top target classes in turn.
    /// </summary>
    private Vector<T>? ApgdTargeted(AttackPoint point, Vector<T> cleanLogits)
    {
        // Lazily: the first target that yields an adversarial example ends the search.
        return TargetClassesOf(cleanLogits, point.Label, point.Goal)
            .Select(target => Apgd(point, LossKind.TargetedDlr, target, 1))
            .FirstOrDefault(adversarial => adversarial is not null);
    }

    private Vector<T>? RunApgd(AttackPoint point, LossKind loss, int target)
    {
        var run = new ApgdRun(this, point, loss, target, Settings.Iterations);
        for (int iteration = 0; iteration < Settings.Iterations; iteration++)
        {
            run.Iterate(iteration);
        }

        return run.Adversarial;
    }

    /// <summary>
    /// The APGD random start: the clean input plus epsilon times a normalized uniform (L-infinity) or
    /// Gaussian (L2) direction, clipped to [0, 1].
    /// </summary>
    private Vector<T> ApgdRandomStart(Vector<T> clean)
    {
        var direction = IsL2
            ? Engine.GenerateGaussianNoise<T>(clean.Length, NumOps.Zero, NumOps.One, Random.Next())
            : Engine.TensorRandomUniformRange(new[] { clean.Length }, NumOps.Negate(NumOps.One), NumOps.One, Random.Next()).ToVector();
        var start = Engine.Add<T>(clean, Engine.Multiply<T>(NormalizeThreat(direction), Epsilon));
        return Engine.Clamp<T>(start, NumOps.Zero, NumOps.One);
    }

    /// <summary>
    /// The steepest ascent direction of the threat model times the step size.
    /// </summary>
    private Vector<T> SteepestStep(Vector<T> gradient, T stepSize)
    {
        return IsL2
            ? Engine.Multiply<T>(NormalizeThreat(gradient), stepSize)
            : Engine.Multiply<T>(SignVector(gradient), stepSize);
    }

    /// <summary>
    /// One run of APGD (Croce and Hein 2020, Algorithm 1; autoattack APGDAttack.attack_single_run).
    /// </summary>
    /// <remarks>
    /// Step size 2 epsilon, momentum 0.75 after the first step, checkpoints every 0.22 N_iter iterations
    /// shrinking by 0.03 N_iter down to 0.06 N_iter, and step halving with a restart from the best point when
    /// fewer than 75% of the steps since the last checkpoint increased the loss, or when the best loss did
    /// not improve after a checkpoint that did not halve.
    /// </remarks>
    private sealed class ApgdRun
    {
        private const double Momentum = 0.75;
        private const double OscillationFraction = 0.75;

        private readonly AutoAttack<T, TInput, TOutput> _attack;
        private readonly AttackPoint _point;
        private readonly LossKind _loss;
        private readonly int _target;
        private readonly int _iterations;
        private readonly int _sizeDecrease;
        private readonly int _minimumCheckpoint;
        private readonly Vector<T> _lossSteps;
        private Vector<T> _current;
        private Vector<T> _previous;
        private Vector<T> _gradient;
        private Vector<T> _best;
        private Vector<T> _bestGradient;
        private T _bestLoss;
        private T _bestLossAtLastCheck;
        private T _stepSize;
        private bool _reducedAtLastCheck = true;
        private int _checkpointLength;
        private int _sinceCheckpoint;

        public ApgdRun(AutoAttack<T, TInput, TOutput> attack, AttackPoint point, LossKind loss, int target, int iterations)
        {
            _attack = attack;
            _point = point;
            _loss = loss;
            _target = target;
            _iterations = Math.Max(iterations, 1);
            _checkpointLength = Math.Max((int)(0.22 * iterations), 1);
            _minimumCheckpoint = Math.Max((int)(0.06 * iterations), 1);
            _sizeDecrease = Math.Max((int)(0.03 * iterations), 1);
            _lossSteps = new Vector<T>(_iterations);

            // autoattack APGDAttack use_rs: a random start in the ball, or the clean input.
            _current = attack.Settings.UseRandomStart ? attack.ApgdRandomStart(point.Clean) : point.Clean;
            var logits = point.Logits(_current);
            var (value, logitGradient) = attack.Loss(logits, loss, point.Label, target);
            _gradient = attack.InputGradient(point, _current, logitGradient);
            if (attack.Fools(logits, point))
            {
                Adversarial = _current;
            }

            _best = _current;
            _bestGradient = _gradient;
            _bestLoss = value;
            _bestLossAtLastCheck = value;
            _previous = _current;
            _stepSize = NumOps.Multiply(NumOps.FromDouble(2.0), attack.Epsilon);
        }

        /// <summary>The last iterate that fooled the model, or null.</summary>
        public Vector<T>? Adversarial { get; private set; }

        public void Iterate(int iteration)
        {
            var engine = AiDotNetEngine.Current;
            T a = NumOps.FromDouble(iteration > 0 ? Momentum : 1.0);
            var momentumTerm = engine.Subtract<T>(_current, _previous);
            _previous = _current;

            // z = P(x_k + eta * step(grad)); x_{k+1} = P(x_k + a (z - x_k) + (1 - a)(x_k - x_{k-1})).
            var z = _attack.Project(_point.Clean, engine.Add<T>(_current, _attack.SteepestStep(_gradient, _stepSize)));
            var blended = engine.Add<T>(
                engine.Add<T>(_current, engine.Multiply<T>(engine.Subtract<T>(z, _current), a)),
                engine.Multiply<T>(momentumTerm, NumOps.Subtract(NumOps.One, a)));
            _current = _attack.Project(_point.Clean, blended);

            var logits = _point.Logits(_current);
            var (value, logitGradient) = _attack.Loss(logits, _loss, _point.Label, _target);
            _gradient = _attack.InputGradient(_point, _current, logitGradient);
            if (_attack.Fools(logits, _point))
            {
                Adversarial = _current;
            }

            _lossSteps.AsWritableSpan()[iteration] = value;
            if (NumOps.GreaterThan(value, _bestLoss))
            {
                _best = _current;
                _bestGradient = _gradient;
                _bestLoss = value;
            }

            _sinceCheckpoint++;
            if (_sinceCheckpoint == _checkpointLength)
            {
                Checkpoint(iteration);
            }
        }

        private void Checkpoint(int iteration)
        {
            var engine = AiDotNetEngine.Current;
            int k = _checkpointLength;

            // Count of loss increases over the last k steps; index -1 wraps to the (still zero) last slot,
            // as in autoattack check_oscillation.
            int iterations = _iterations;
            var indices = Enumerable.Range(iteration - k, k + 1).Select(i => ((i % iterations) + iterations) % iterations).ToArray();
            var window = engine.TensorGather(_lossSteps.AsTensor(), new Tensor<int>(indices, new[] { k + 1 }), 0);
            var later = engine.TensorSlice(window, new[] { 1 }, new[] { k });
            var earlier = engine.TensorSlice(window, new[] { 0 }, new[] { k });
            T increases = engine.TensorSum(engine.TensorGreaterThan(later, earlier));
            bool oscillating = NumOps.LessThanOrEquals(increases, NumOps.FromDouble(k * OscillationFraction));
            bool stalled = !_reducedAtLastCheck && NumOps.GreaterThanOrEquals(_bestLossAtLastCheck, _bestLoss);
            bool reduce = oscillating || stalled;
            _reducedAtLastCheck = reduce;
            _bestLossAtLastCheck = _bestLoss;
            if (reduce)
            {
                _stepSize = NumOps.Divide(_stepSize, NumOps.FromDouble(2.0));
                _current = _best;
                _gradient = _bestGradient;
            }

            _checkpointLength = Math.Max(k - _sizeDecrease, _minimumCheckpoint);
            _sinceCheckpoint = 0;
        }
    }

    /// <summary>
    /// The APGD checkpoint iterations w_j for a budget (Croce and Hein 2020, Sec. 3.2): p_0 = 0, p_1 = 0.22,
    /// p_{j+1} = p_j + max(p_j - p_{j-1} - 0.03, 0.06), as realized by the reference implementation's
    /// shrinking checkpoint length.
    /// </summary>
    internal static int[] ApgdCheckpoints(int iterations)
    {
        int length = Math.Max((int)(0.22 * iterations), 1);
        int minimum = Math.Max((int)(0.06 * iterations), 1);
        int decrease = Math.Max((int)(0.03 * iterations), 1);
        var checkpoints = new List<int>();
        int next = length;
        while (next <= iterations)
        {
            checkpoints.Add(next);
            length = Math.Max(length - decrease, minimum);
            next += length;
        }

        return checkpoints.ToArray();
    }

    #endregion

    #region FAB

    /// <summary>
    /// FAB (untargeted) or FAB-T: every target class in turn, each with its restarts (random start from the
    /// second restart on); a candidate counts only when it fools the model within epsilon.
    /// </summary>
    private Vector<T>? Fab(AttackPoint point, Vector<T> cleanLogits, bool targeted, int restarts)
    {
        var targets = targeted ? TargetClassesOf(cleanLogits, point.Label, point.Goal) : new[] { -1 };
        foreach (int target in targets)
        {
            for (int restart = 0; restart < restarts; restart++)
            {
                var candidate = RunFab(point, target, restart > 0);
                if (FabSucceeded(point, candidate))
                {
                    return candidate;
                }
            }
        }

        return null;
    }

    private Vector<T> RunFab(AttackPoint point, int target, bool randomStart)
    {
        var run = new FabRun(this, point, target, randomStart);
        for (int iteration = 0; iteration < Settings.Iterations; iteration++)
        {
            run.Iterate();
        }

        return run.Result;
    }

    private bool FabSucceeded(AttackPoint point, Vector<T> candidate)
    {
        return Fools(point.Logits(candidate), point)
            && NumOps.LessThanOrEquals(ThreatNorm(Engine.Subtract<T>(candidate, point.Clean)), Epsilon);
    }

    /// <summary>
    /// The linearized decision boundary at x: f = z_t - z_y and its input gradient. Untargeted FAB picks the
    /// class whose boundary is closest in the dual norm, |f_c| / ||grad f_c||_q (fab_pt get_diff_logits_grads_batch).
    /// </summary>
    /// <remarks>
    /// The boundary is returned scaled so the largest entry of its normal is 1 (see <see cref="ClosestBoundary"/>).
    /// </remarks>
    private (T Difference, Vector<T> Normal) Hyperplane(AttackPoint point, Vector<T> vector, int target)
    {
        var logits = point.Logits(vector);
        int n = logits.Length;
        int label = point.Label;
        int d = vector.Length;
        if (target >= 0)
        {
            var weights = Engine.Subtract<T>(Vector<T>.CreateStandardBasis(n, target), Vector<T>.CreateStandardBasis(n, label));
            var difference = new Vector<T>(new[] { NumOps.Subtract(At(logits, target), At(logits, label)) });
            var gradient = Engine.Reshape(InputGradient(point, vector, weights).AsTensor(), new[] { 1, d });
            var (scaledDifferences, scaledNormals) = ScaleRows(difference, gradient);
            return (At(scaledDifferences, 0), scaledNormals.ToVector());
        }

        var rows = new Tensor<T>[n];
        for (int c = 0; c < n; c++)
        {
            rows[c] = InputGradient(point, vector, Vector<T>.CreateStandardBasis(n, c)).AsTensor();
        }

        var jacobian = Engine.TensorStack(rows, 0);
        var labelRow = Engine.TensorSlice(jacobian, new[] { label, 0 }, new[] { 1, d });
        var normals = Engine.TensorSubtract(jacobian, Engine.TensorBroadcastTo(labelRow, new[] { n, d }));
        var differences = Engine.Subtract<T>(logits, Engine.Fill<T>(n, At(logits, label)));
        var closest = ClosestBoundary(differences, normals, label);
        return (closest.Difference, closest.Normal);
    }

    /// <summary>
    /// The boundary c != label closest in the dual norm, |f_c| / ||g_c||_q, with its difference and normal scaled
    /// by the largest absolute entry of the normal.
    /// </summary>
    /// <remarks>
    /// Scaling a row by a positive factor leaves its hyperplane {z : g . z = g . x - f} and the distance
    /// |f| / ||g||_q unchanged, and the FAB projections are invariant to it. Unscaled, in Half the squares of
    /// the L2 dual norm overflow for gradient entries above 256, which made such a boundary look as if it were
    /// at distance 0, and the offset g . x overflows for large gradients.
    /// </remarks>
    internal (int Index, T Difference, Vector<T> Normal) ClosestBoundary(Vector<T> differences, Tensor<T> normals, int label)
    {
        int n = normals._shape[0];
        int d = normals._shape[1];
        var (scaledDifferences, scaledNormals) = ScaleRows(differences, normals);
        var dualNorms = IsL2
            ? Engine.TensorNorm(scaledNormals, 1, false)
            : Engine.ReduceSum(Engine.TensorAbs(scaledNormals), new[] { 1 }, false);
        var distances = Engine.TensorDivide(
            Engine.TensorAbs(scaledDifferences.AsTensor()),
            Engine.TensorAddScalar(dualNorms, Tiny(1e-12)));
        var isLabel = Vector<T>.CreateStandardBasis(n, label).AsTensor();
        var excluded = Engine.TensorAddScalar(Engine.TensorZerosLike(distances), NumOps.MaxValue);
        distances = Engine.TensorWhere(isLabel, excluded, distances);
        int closest = Engine.TensorArgMin(distances, 0).ToArray()[0];
        var normal = Engine.TensorSlice(scaledNormals, new[] { closest, 0 }, new[] { 1, d }).ToVector();
        return (closest, At(scaledDifferences, closest), normal);
    }

    /// <summary>
    /// Divides each row of the normals, and its difference, by the row's largest absolute entry (rows of zeros
    /// are left unchanged).
    /// </summary>
    private (Vector<T> Differences, Tensor<T> Normals) ScaleRows(Vector<T> differences, Tensor<T> normals)
    {
        int n = normals._shape[0];
        int d = normals._shape[1];
        var scale = Engine.ReduceMax(Engine.TensorAbs(normals), new[] { 1 }, true);
        scale = Engine.TensorWhere(Engine.TensorEquals(scale, NumOps.Zero), Engine.TensorOnesLike(scale), scale);
        var scaledNormals = Engine.TensorDivide(normals, Engine.TensorBroadcastTo(scale, new[] { n, d }));
        var scaledDifferences = Engine.TensorDivide(differences.AsTensor(), Engine.Reshape(scale, new[] { n })).ToVector();
        return (scaledDifferences, scaledNormals);
    }

    /// <summary>
    /// The same hyperplane {z : normal . z = offset} with the normal scaled so its largest absolute entry is 1.
    /// </summary>
    private (Vector<T> Normal, T Offset) ScaledHyperplane(Vector<T> normal, T offset)
    {
        var (offsets, normals) = ScaleRows(new Vector<T>(new[] { offset }), Engine.Reshape(normal.AsTensor(), new[] { 1, normal.Length }));
        return (normals.ToVector(), At(offsets, 0));
    }

    /// <summary>
    /// The minimal step d with normal . (point + d) = offset and point + d in [0, 1], in the threat norm.
    /// </summary>
    private Vector<T> ProjectOntoHyperplane(Vector<T> point, Vector<T> normal, T offset)
    {
        return IsL2 ? ProjectL2Hyperplane(point, normal, offset) : ProjectLInfinityHyperplane(point, normal, offset);
    }

    /// <summary>
    /// Box-constrained L-infinity projection onto a hyperplane (autoattack fab_projections.projection_linf).
    /// </summary>
    /// <remarks>
    /// The hyperplane is first scaled to a largest normal entry of 1, which leaves the projection unchanged and
    /// keeps its sums finite in Half.
    /// </remarks>
    internal Vector<T> ProjectLInfinityHyperplane(Vector<T> point, Vector<T> normal, T offset)
    {
        (normal, offset) = ScaledHyperplane(normal, offset);
        int n = point.Length;
        var ones = Engine.Fill<T>(n, NumOps.One);
        T side = NumOps.Subtract(Engine.DotProduct<T>(normal, point), offset);
        T sign = NumOps.GreaterThanOrEquals(side, NumOps.Zero) ? NumOps.One : NumOps.Negate(NumOps.One);
        var w = Engine.Multiply<T>(normal, sign);
        T b = NumOps.Multiply(offset, sign);

        var negative = Engine.TensorLessThan(w.AsTensor(), NumOps.Zero).ToVector();
        var nonzero = Engine.TensorNotEquals(w.AsTensor(), NumOps.Zero).ToVector();
        var d = Engine.Multiply<T>(Engine.Subtract<T>(negative, point), nonzero);
        var twoAMinusOne = Engine.Subtract<T>(Engine.Multiply<T>(negative, NumOps.FromDouble(2.0)), ones);
        var p = Engine.Subtract<T>(negative, Engine.Multiply<T>(point, twoAMinusOne));
        int[] ascending = Engine.TensorArgsort(p.AsTensor(), 0, false).ToArray();
        T residual = NumOps.Subtract(b, Engine.DotProduct<T>(w, point));
        T b0 = Engine.DotProduct<T>(w, d);

        var order = new Tensor<int>(Enumerable.Reverse(ascending).ToArray(), new[] { n });
        var ws = Engine.TensorGather(w.AsTensor(), order, 0);
        var bs2 = Engine.TensorNegate(Engine.TensorMultiply(ws, Engine.TensorGather(d.AsTensor(), order, 0)));
        var s = Engine.TensorCumSum(Engine.TensorAbs(ws), 0).ToVector();
        var sb = Engine.Add<T>(Engine.TensorCumSum(bs2, 0).ToVector(), Engine.Fill<T>(n, b0));

        T b2 = NumOps.Subtract(At(sb, n - 1), NumOps.Multiply(At(s, n - 1), At(p, ascending[0])));
        bool beyondAll = NumOps.GreaterThan(NumOps.Subtract(residual, b2), NumOps.Zero);
        bool bracketed = !beyondAll && NumOps.GreaterThan(NumOps.Subtract(residual, b0), NumOps.Zero);
        if (beyondAll)
        {
            T lambda = MaxOf(NumOps.Divide(NumOps.Subtract(residual, At(sb, n - 1)), NumOps.Negate(At(s, n - 1))), NumOps.Zero);
            d = Engine.Multiply<T>(twoAMinusOne, lambda);
        }
        else if (bracketed)
        {
            int lb = BisectLInfinity(ascending, sb, s, p, residual);
            T lambda = MaxOf(NumOps.Divide(NumOps.Subtract(residual, At(sb, lb)), NumOps.Negate(At(s, lb))), NumOps.Zero);
            var lambdas = Engine.Fill<T>(n, lambda);
            var positivePart = Engine.Multiply<T>(Engine.Min<T>(lambdas, d), negative);
            var negativePart = Engine.Multiply<T>(Engine.Max<T>(Engine.Negate<T>(lambdas), d), Engine.Subtract<T>(ones, negative));
            d = Engine.Add<T>(positivePart, negativePart);
        }

        return Engine.Multiply<T>(d, nonzero);
    }

    private static int CeilLog2(int n)
    {
        int steps = 0;
        while ((1L << steps) < n)
        {
            steps++;
        }

        return steps;
    }

    private static int BisectLInfinity(int[] ascending, Vector<T> sb, Vector<T> s, Vector<T> p, T residual)
    {
        int n = ascending.Length;
        int lb = 0;
        int ub = n - 1;
        int steps = CeilLog2(n);
        for (int step = 0; step < steps; step++)
        {
            int mid = (lb + ub) / 2;
            if (LInfinityBelow(residual, sb, s, p, ascending[n - 1 - mid], mid))
            {
                lb = mid;
            }
            else
            {
                ub = mid;
            }
        }

        return lb;
    }

    private static bool LInfinityBelow(T residual, Vector<T> sb, Vector<T> s, Vector<T> p, int index, int position)
    {
        T b2 = NumOps.Subtract(At(sb, position), NumOps.Multiply(At(s, position), At(p, index)));
        return NumOps.GreaterThan(NumOps.Subtract(residual, b2), NumOps.Zero);
    }

    /// <summary>
    /// Box-constrained L2 projection onto a hyperplane (autoattack fab_projections.projection_l2).
    /// </summary>
    /// <remarks>
    /// The hyperplane is first scaled to a largest normal entry of 1, which leaves the projection unchanged: in
    /// Half the squared normal entries overflowed above 256. The 1e-8 threshold for a zero entry is therefore
    /// relative to the largest entry.
    /// </remarks>
    internal Vector<T> ProjectL2Hyperplane(Vector<T> point, Vector<T> normal, T offset)
    {
        (normal, offset) = ScaledHyperplane(normal, offset);
        int n = point.Length;
        T c = NumOps.Subtract(Engine.DotProduct<T>(normal, point), offset);
        T sign = NumOps.GreaterThanOrEquals(c, NumOps.Zero) ? NumOps.One : NumOps.Negate(NumOps.One);
        var w = Engine.Multiply<T>(normal, sign).AsTensor();
        c = NumOps.Multiply(c, sign);

        T big = NumOps.FromDouble(1e12);
        var t = point.AsTensor();
        var zeros = Engine.TensorZerosLike(w);
        var ones = Engine.TensorOnesLike(w);
        var bigs = Engine.TensorAddScalar(zeros, big);
        var small = Engine.TensorLessThan(Engine.TensorAbs(w), Tiny(1e-8));
        var safeW = Engine.TensorWhere(small, ones, w);
        var r = Engine.TensorMax(Engine.TensorDivide(t, safeW), Engine.TensorDivide(Engine.TensorSubtract(t, ones), safeW));
        r = Engine.TensorClamp(r, NumOps.Negate(big), big);
        r = Engine.TensorWhere(small, bigs, r);
        r = Engine.TensorWhere(Engine.TensorEquals(r, NumOps.Negate(big)), bigs, r);

        int[] ascending = Engine.TensorArgsort(r, 0, false).ToArray();
        var order = new Tensor<int>(ascending, new[] { n });
        var rs = Engine.TensorGather(r, order, 0);
        var rs2 = n > 1
            ? Engine.TensorConcatenate(new[] { Engine.TensorSlice(rs, new[] { 1 }, new[] { n - 1 }), new Tensor<T>(new[] { 1 }) }, 0)
            : new Tensor<T>(new[] { 1 });
        rs = Engine.TensorWhere(Engine.TensorEquals(rs, big), zeros, rs);
        rs2 = Engine.TensorWhere(Engine.TensorEquals(rs2, big), zeros, rs2);

        var w3s = Engine.TensorGather(Engine.TensorSquare(w), order, 0);
        T w5 = Engine.TensorSum(w3s);
        var ws = Engine.ScalarMinusTensor(w5, Engine.TensorCumSum(w3s, 0)).ToVector();
        var d = Engine.TensorWhere(small, zeros, Engine.TensorNegate(Engine.TensorMultiply(r, w)));

        T rs0 = rs.GetFlat(0);
        T s0 = NumOps.Negate(NumOps.Multiply(w5, rs0));
        var sTail = Engine.TensorAddScalar(Engine.TensorCumSum(Engine.TensorMultiply(Engine.TensorSubtract(rs, rs2), ws.AsTensor()), 0), s0).ToVector();
        bool c4 = NumOps.LessThan(NumOps.Add(s0, c), NumOps.Zero);
        bool c3 = NumOps.GreaterThan(NumOps.Add(Engine.TensorSum(Engine.TensorMultiply(d, w)), c), NumOps.Zero);
        var wVector = w.ToVector();
        if (c4)
        {
            T alpha = NumOps.Divide(c, w5);
            d = Engine.Multiply<T>(wVector, NumOps.Negate(alpha)).AsTensor();
        }
        else if (!c3)
        {
            int lb = BisectL2(n, s0, sTail, c);
            T wsLb = At(ws, lb);
            T alpha = !NumOps.GreaterThan(wsLb, NumOps.Zero) && !NumOps.LessThan(wsLb, NumOps.Zero)
                ? NumOps.Zero
                : NumOps.Add(NumOps.Divide(NumOps.Add(SAt(s0, sTail, lb), c), wsLb), rs.GetFlat(lb));
            var keep = Engine.TensorGreaterThan(Engine.TensorAddScalar(zeros, alpha), r);
            d = Engine.TensorWhere(keep, d, Engine.Multiply<T>(wVector, NumOps.Negate(alpha)).AsTensor());
        }

        return Engine.TensorWhere(small, zeros, d).ToVector();
    }

    private static T SAt(T s0, Vector<T> sTail, int position) => position == 0 ? s0 : At(sTail, position - 1);

    private static int BisectL2(int n, T s0, Vector<T> sTail, T c)
    {
        int lb = 0;
        int ub = n - 1;
        int steps = CeilLog2(n);
        for (int step = 0; step < steps; step++)
        {
            int mid = (lb + ub) / 2;
            if (L2Above(s0, sTail, c, mid))
            {
                lb = mid;
            }
            else
            {
                ub = mid;
            }
        }

        return lb;
    }

    private static bool L2Above(T s0, Vector<T> sTail, T c, int position)
        => NumOps.GreaterThan(NumOps.Add(SAt(s0, sTail, position), c), NumOps.Zero);

    /// <summary>
    /// One run of FAB (Croce and Hein 2020b, Algorithm 1; autoattack FABAttack attack_single_run).
    /// </summary>
    private sealed class FabRun
    {
        private readonly AutoAttack<T, TInput, TOutput> _attack;
        private readonly AttackPoint _point;
        private readonly int _target;
        private readonly T _alphaMax;
        private readonly T _eta;
        private readonly T _beta;
        private Vector<T> _x;
        private T _bestDistance;
        private bool _found;

        public FabRun(AutoAttack<T, TInput, TOutput> attack, AttackPoint point, int target, bool randomStart)
        {
            _attack = attack;
            _point = point;
            _target = target;
            _alphaMax = NumOps.FromDouble(attack.Settings.FabAlphaMax);
            _eta = NumOps.FromDouble(attack.Settings.FabEta);
            _beta = NumOps.FromDouble(attack.Settings.FabBeta);
            _bestDistance = NumOps.Zero;
            Result = point.Clean;
            _x = randomStart ? attack.FabRandomStart(point.Clean) : point.Clean;
        }

        /// <summary>The smallest successful perturbation found, or the clean input.</summary>
        public Vector<T> Result { get; private set; }

        public void Iterate()
        {
            var engine = AiDotNetEngine.Current;
            var clean = _point.Clean;
            var (difference, normal) = _attack.Hyperplane(_point, _x, _target);
            T offset = NumOps.Add(NumOps.Negate(difference), engine.DotProduct<T>(normal, _x));
            var d1 = _attack.ProjectOntoHyperplane(_x, normal, offset);
            var d2 = _attack.ProjectOntoHyperplane(clean, normal, offset);
            T floor = Tiny(1e-8);
            T a1 = MaxOf(_attack.ThreatNorm(d1), floor);
            T a2 = MaxOf(_attack.ThreatNorm(d2), floor);
            T alpha = MinOf(MaxOf(NumOps.Divide(a1, NumOps.Add(a1, a2)), NumOps.Zero), _alphaMax);
            var fromCurrent = engine.Multiply<T>(engine.Add<T>(_x, engine.Multiply<T>(d1, _eta)), NumOps.Subtract(NumOps.One, alpha));
            var fromClean = engine.Multiply<T>(engine.Add<T>(clean, engine.Multiply<T>(d2, _eta)), alpha);
            _x = engine.Clamp<T>(engine.Add<T>(fromCurrent, fromClean), NumOps.Zero, NumOps.One);

            if (_attack.Fools(_point.Logits(_x), _point))
            {
                T distance = _attack.ThreatNorm(engine.Subtract<T>(_x, clean));
                if (!_found || NumOps.LessThan(distance, _bestDistance))
                {
                    Result = _x;
                    _bestDistance = distance;
                    _found = true;
                }

                // Step back towards the clean input to look for a smaller perturbation.
                _x = engine.Add<T>(clean, engine.Multiply<T>(engine.Subtract<T>(_x, clean), _beta));
            }
        }
    }

    /// <summary>
    /// FAB's random restart: half an epsilon-ball step in a normalized uniform (L-infinity) or Gaussian (L2) direction.
    /// </summary>
    private Vector<T> FabRandomStart(Vector<T> clean)
    {
        var direction = IsL2
            ? Engine.GenerateGaussianNoise<T>(clean.Length, NumOps.Zero, NumOps.One, Random.Next())
            : Engine.TensorRandomUniformRange(new[] { clean.Length }, NumOps.Negate(NumOps.One), NumOps.One, Random.Next()).ToVector();
        T scale = NumOps.Divide(NumOps.Multiply(Epsilon, NumOps.FromDouble(0.5)), ThreatNorm(direction));
        return Engine.Clamp<T>(Engine.Add<T>(clean, Engine.Multiply<T>(direction, scale)), NumOps.Zero, NumOps.One);
    }

    #endregion

    #region Square

    /// <summary>
    /// The Square attack (Andriushchenko et al. 2020; autoattack SquareAttack with the margin loss): a
    /// random search that keeps a window update whenever it lowers the margin z_y - max_{i != y} z_i.
    /// </summary>
    private Vector<T>? Square(AttackPoint point)
    {
        var run = new SquareRun(this, point);
        for (int query = 0; query < Settings.SquareQueries && !run.Fooled; query++)
        {
            run.Iterate(query);
        }

        return run.Fooled && Fools(point.Logits(run.Best), point) ? run.Best : null;
    }

    /// <summary>
    /// The margin z_y - max_{i != y} z_i; it is at most zero once the model is fooled.
    /// </summary>
    private T Margin(Vector<T> logits, int label)
    {
        var ranking = RankDescending(logits);
        int other = ranking[0] == label && ranking.Length > 1 ? ranking[1] : ranking[0];
        return NumOps.Subtract(At(logits, label), At(logits, other));
    }

    /// <summary>
    /// The fraction p of the input one window covers at a query (autoattack p_selection, rescaled to 10000 queries).
    /// </summary>
    internal static double SquareFraction(double initial, int query, int totalQueries)
    {
        int it = (int)((double)query / Math.Max(totalQueries, 1) * 10000);
        double[] bounds = { 10, 50, 200, 500, 1000, 2000, 4000, 6000, 8000 };
        // The fraction halves once per schedule bound the rescaled iteration has passed.
        double divisor = Math.Pow(2, bounds.Count(bound => it > bound));
        return initial / divisor;
    }

    /// <summary>
    /// The window size for a fraction p: sqrt(p * h * w) for images (at least 1 for L-infinity, at least 3 and
    /// odd for L2), or p * w for one-row inputs.
    /// </summary>
    internal static (int Rows, int Cols) SquareWindow(double p, int height, int width, bool l2)
    {
        int minimum = l2 ? 3 : 1;
        int size = height == 1
            ? Math.Max((int)Math.Round(p * width), minimum)
            : Math.Max((int)Math.Round(Math.Sqrt(p * height * width)), minimum);
        if (l2 && size % 2 == 0)
        {
            size++;
        }

        return (Math.Min(height == 1 ? 1 : size, height), Math.Min(size, width));
    }

    /// <summary>
    /// A uniformly random first row (or column) of a window of <paramref name="size"/> cells along an axis of
    /// <paramref name="extent"/> cells: one of 0 .. extent - size, both ends included.
    /// </summary>
    /// <remarks>
    /// The Square Attack paper (Andriushchenko et al. 2020, Alg. 2 and 3) draws the window position uniformly
    /// from {0, ..., w - h}. Random.Next excludes its upper bound, so the bound is extent - size + 1; with
    /// extent - size the last position, and with one-cell windows the last coordinate, was never updated.
    /// </remarks>
    internal static int SquareWindowStart(Random random, int extent, int size)
    {
        return random.Next(0, Math.Max(extent - size, 0) + 1);
    }

    /// <summary>
    /// eta_rectangles of the L2 Square attack: concentric rectangles weighted 1/(k+1)^2, unit L2 norm.
    /// </summary>
    /// <remarks>
    /// A cell at Chebyshev distance d from the center lies in the rectangles k = d .. K-1, so its value is the
    /// tail sum of 1/(k+1)^2 from d.
    /// </remarks>
    private Tensor<T> EtaRectangles(int rows, int cols)
    {
        int centerRow = rows / 2;
        int centerCol = cols / 2;
        int levels = Math.Max(rows / 2 + 1, cols / 2 + 1);
        var k = Engine.TensorArange<T>(1, levels + 1, 1);
        var weights = Engine.TensorReciprocal(Engine.TensorSquare(k));
        var tail = Engine.TensorFlip(Engine.TensorCumSum(Engine.TensorFlip(weights, new[] { 0 }), 0), new[] { 0 });
        var distance = Enumerable.Range(0, rows * cols)
            .Select(i => Math.Max(Math.Abs(i / cols - centerRow), Math.Abs(i % cols - centerCol)))
            .ToArray();
        var values = Engine.TensorGather(tail, new Tensor<int>(distance, new[] { rows * cols }), 0);
        var normalized = Engine.TensorMultiplyScalar(values, NumOps.Divide(NumOps.One, Engine.Norm<T>(values.ToVector())));
        return Engine.Reshape(normalized, new[] { rows, cols });
    }

    /// <summary>
    /// meta_pseudo_gaussian_pert: +eta on one half of the window and -eta on the other, unit L2 norm, transposed
    /// with probability 1/2. One-row windows are split along their length.
    /// </summary>
    private Tensor<T> PseudoGaussianPerturbation(int rows, int cols)
    {
        bool transpose = Random.NextDouble() > 0.5;
        Tensor<T> delta;
        if (rows == 1)
        {
            int half = cols / 2;
            delta = Engine.TensorConcatenate(
                new[] { EtaRectangles(1, half), Engine.TensorNegate(EtaRectangles(1, cols - half)) }, 1);
        }
        else
        {
            int half = rows / 2;
            delta = Engine.TensorConcatenate(
                new[] { EtaRectangles(half, cols), Engine.TensorNegate(EtaRectangles(rows - half, cols)) }, 0);
        }

        delta = Engine.TensorMultiplyScalar(delta, NumOps.Divide(NumOps.One, Engine.Norm<T>(delta.ToVector())));
        return transpose && rows == cols ? Engine.TensorTranspose(delta) : delta;
    }

    /// <summary>
    /// A [channels, rows, cols] patch: the per-channel sign times the given [rows, cols] pattern.
    /// </summary>
    private Tensor<T> SignedPatch(Tensor<T> pattern, int channels)
    {
        int rows = pattern.Shape[0];
        int cols = pattern.Shape[1];
        var signs = Engine.TensorBroadcastTo(RandomSigns(new[] { channels, 1, 1 }), new[] { channels, rows, cols });
        var tiled = Engine.TensorBroadcastTo(Engine.Reshape(pattern, new[] { 1, rows, cols }), new[] { channels, rows, cols });
        return Engine.TensorMultiply(signs, tiled);
    }

    private Tensor<T> AddWindow(Tensor<T> image, Tensor<T> patch, int row, int col)
    {
        var current = Engine.TensorSlice(image, new[] { 0, row, col }, patch.Shape.ToArray());
        return Engine.TensorSetSlice(image.Clone(), Engine.TensorAdd(current, patch), new[] { 0, row, col });
    }

    private Tensor<T> SetWindow(Tensor<T> image, Tensor<T> patch, int row, int col)
    {
        return Engine.TensorSetSlice(image.Clone(), patch, new[] { 0, row, col });
    }

    private Tensor<T> Window(Tensor<T> image, int row, int col, int rows, int cols)
    {
        return Engine.TensorSlice(image, new[] { 0, row, col }, new[] { image.Shape[0], rows, cols });
    }

    /// <summary>
    /// The L2 Square initialization: a grid of signed pseudo-Gaussian tiles of side h/5, scaled to norm epsilon.
    /// </summary>
    private Tensor<T> SquareL2Initialization(AttackPoint point)
    {
        int c = point.Channels;
        int h = point.Height;
        int w = point.Width;
        bool oneRow = h == 1;
        int size = Math.Max((oneRow ? w : h) / 5, 1);
        int tileRows = oneRow ? 1 : size;
        int startRow = oneRow ? 0 : (h - size * 5) / 2;
        int startCol = (w - size * 5) / 2;
        startRow = Math.Max(startRow, 0);
        startCol = Math.Max(startCol, 0);
        int rowTiles = oneRow ? 1 : Math.Min(h / size, (h - startRow) / size);
        int colTiles = Math.Min(w / size, (w - startCol) / size);
        var delta = new Tensor<T>(new[] { c, h, w });
        for (int tr = 0; tr < rowTiles; tr++)
        {
            for (int tc = 0; tc < colTiles; tc++)
            {
                var patch = SignedPatch(PseudoGaussianPerturbation(tileRows, size), c);
                delta = AddWindow(delta, patch, startRow + tr * tileRows, startCol + tc * size);
            }
        }

        return delta;
    }

    /// <summary>
    /// The state of one Square attack run.
    /// </summary>
    private sealed class SquareRun
    {
        private readonly AutoAttack<T, TInput, TOutput> _attack;
        private readonly AttackPoint _point;
        private readonly Tensor<T> _clean;
        private readonly int[] _shape;
        private T _bestMargin;

        public SquareRun(AutoAttack<T, TInput, TOutput> attack, AttackPoint point)
        {
            var engine = AiDotNetEngine.Current;
            _attack = attack;
            _point = point;
            _shape = new[] { point.Channels, point.Height, point.Width };
            _clean = Tensor<T>.FromVector(point.Clean, _shape);
            T eps = attack.Epsilon;
            Tensor<T> start;
            if (attack.IsL2)
            {
                var delta = attack.SquareL2Initialization(point);
                start = engine.TensorAdd(_clean, attack.ScaleToEpsilon(delta));
            }
            else
            {
                // Vertical stripes: one sign per channel and column.
                var stripes = attack.RandomSigns(new[] { point.Channels, 1, point.Width });
                start = engine.TensorAdd(_clean, engine.TensorMultiplyScalar(engine.TensorBroadcastTo(stripes, _shape), eps));
            }

            Best = engine.TensorClamp(start, NumOps.Zero, NumOps.One).ToVector();
            var logits = point.Logits(Best);
            _bestMargin = attack.Margin(logits, point.Label);

            // A zero margin can be a correctly classified tie (the label wins the argmax), so the search stops
            // only once the retained candidate is misclassified; the margin only ranks the candidates.
            Fooled = attack.Fools(logits, point);
        }

        public Vector<T> Best { get; private set; }

        public bool Fooled { get; private set; }

        public void Iterate(int query)
        {
            int h = _point.Height;
            int w = _point.Width;
            double p = SquareFraction(_attack.Settings.SquareInitialFraction, query, _attack.Settings.SquareQueries);
            var (rows, cols) = SquareWindow(p, h, w, _attack.IsL2);
            var best = Tensor<T>.FromVector(Best, _shape);
            int row = SquareWindowStart(_attack.Random, h, rows);
            int col = SquareWindowStart(_attack.Random, w, cols);
            Tensor<T> candidate = _attack.IsL2
                ? L2Candidate(best, row, col, rows, cols)
                : LInfinityCandidate(best, row, col, rows, cols);

            var vector = candidate.ToVector();
            var logits = _point.Logits(vector);
            T margin = _attack.Margin(logits, _point.Label);
            if (NumOps.LessThan(margin, _bestMargin))
            {
                _bestMargin = margin;
                Best = vector;
                Fooled = _attack.Fools(logits, _point);
            }
        }

        private Tensor<T> LInfinityCandidate(Tensor<T> best, int row, int col, int rows, int cols)
        {
            var engine = AiDotNetEngine.Current;
            T eps = _attack.Epsilon;
            var signs = _attack.RandomSigns(new[] { _point.Channels, 1, 1 });
            var patch = engine.TensorMultiplyScalar(
                engine.TensorBroadcastTo(signs, new[] { _point.Channels, rows, cols }),
                NumOps.Multiply(NumOps.FromDouble(2.0), eps));
            var candidate = _attack.AddWindow(best, patch, row, col);
            var lower = engine.TensorAddScalar(_clean, NumOps.Negate(eps));
            var upper = engine.TensorAddScalar(_clean, eps);
            return engine.TensorClamp(engine.TensorClampTensor(candidate, lower, upper), NumOps.Zero, NumOps.One);
        }

        private Tensor<T> L2Candidate(Tensor<T> best, int row, int col, int rows, int cols)
        {
            var engine = AiDotNetEngine.Current;
            int c = _point.Channels;
            int h = _point.Height;
            int w = _point.Width;
            T eps = _attack.Epsilon;
            var delta = engine.TensorSubtract(best, _clean);
            var windowNorms = ChannelNorms(_attack.Window(delta, row, col, rows, cols));
            int row2 = SquareWindowStart(_attack.Random, h, rows);
            int col2 = SquareWindowStart(_attack.Random, w, cols);

            var ones = Tensor<T>.CreateOnes(new[] { c, rows, cols });
            var mask = _attack.SetWindow(_attack.SetWindow(new Tensor<T>(_shape), ones, row, col), ones, row2, col2);
            T imageNorm = _attack.Engine.Norm<T>(delta.ToVector());
            T windowsNorm = _attack.Engine.Norm<T>(engine.TensorMultiply(delta, mask).ToVector());

            var patch = _attack.SignedPatch(_attack.PseudoGaussianPerturbation(rows, cols), c);
            var old = engine.TensorDivide(
                _attack.Window(delta, row, col, rows, cols),
                engine.TensorBroadcastTo(engine.TensorAddScalar(windowNorms, Tiny(1e-12)), new[] { c, rows, cols }));
            patch = engine.TensorAdd(patch, old);

            // Rescale so the image keeps norm epsilon: sqrt(max(eps^2 - ||delta||^2, 0) / c + ||delta on windows||^2).
            T budget = MaxOf(NumOps.Subtract(NumOps.Multiply(eps, eps), NumOps.Multiply(imageNorm, imageNorm)), NumOps.Zero);
            T target = NumOps.Sqrt(NumOps.Add(NumOps.Divide(budget, NumOps.FromDouble(c)), NumOps.Multiply(windowsNorm, windowsNorm)));
            patch = engine.TensorMultiplyScalar(
                engine.TensorDivide(patch, engine.TensorBroadcastTo(ChannelNorms(patch), new[] { c, rows, cols })),
                target);

            delta = _attack.SetWindow(delta, new Tensor<T>(new[] { c, rows, cols }), row2, col2);
            delta = _attack.SetWindow(delta, patch, row, col);
            return engine.TensorClamp(engine.TensorAdd(_clean, _attack.ScaleToEpsilon(delta)), NumOps.Zero, NumOps.One);
        }

        private static Tensor<T> ChannelNorms(Tensor<T> window)
        {
            var engine = AiDotNetEngine.Current;
            return engine.TensorSqrt(engine.ReduceSum(engine.TensorSquare(window), new[] { 1, 2 }, true));
        }
    }

    /// <summary>
    /// delta / (||delta||_2 + 1e-12) * epsilon.
    /// </summary>
    /// <remarks>
    /// Divided before it is scaled: epsilon / 1e-7 (the Half guard) overflows, which turned a zero delta into NaN.
    /// </remarks>
    internal Tensor<T> ScaleToEpsilon(Tensor<T> delta)
    {
        T norm = Engine.Norm<T>(delta.ToVector());
        return Engine.TensorMultiplyScalar(Engine.TensorDivideScalar(delta, NumOps.Add(norm, Tiny(1e-12))), Epsilon);
    }

    #endregion

    /// <summary>
    /// Gets the class index from a label vector (argmax for one-hot or probability vectors).
    /// </summary>
    private int GetClassIndex(Vector<T> label)
    {
        if (label == null || label.Length == 0)
        {
            return 0;
        }

        return ArgMaxIndex(label);
    }

    /// <inheritdoc/>
    public override TInput CalculatePerturbation(TInput original, TInput adversarial)
    {
        if (original == null)
        {
            throw new ArgumentNullException(nameof(original));
        }

        if (adversarial == null)
        {
            throw new ArgumentNullException(nameof(adversarial));
        }

        var originalVector = ConversionsHelper.ConvertToVector<T, TInput>(original);
        var adversarialVector = ConversionsHelper.ConvertToVector<T, TInput>(adversarial);

        if (originalVector.Length != adversarialVector.Length)
        {
            throw new ArgumentException("Original and adversarial examples must have the same length.");
        }

        // Use vectorized subtraction
        var perturbation = Engine.Subtract<T>(adversarialVector, originalVector);

        return ConversionsHelper.ConvertVectorToInput<T, TInput>(perturbation, original);
    }
}

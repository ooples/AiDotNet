using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AdversarialRobustness.Attacks;

/// <summary>
/// Implements the Carlini and Wagner (C and W) attack.
/// </summary>
/// <remarks>
/// <para>
/// C and W is an optimization-based attack that formulates adversarial example generation as
/// an optimization problem, typically producing stronger attacks than gradient-based methods.
/// </para>
/// <para><b>For Beginners:</b> C and W is one of the most sophisticated attacks. Instead of
/// following gradients, it treats creating adversarial examples as a carefully crafted
/// optimization problem. It's slower than FGSM or PGD but often finds adversarial examples
/// that are more subtle and harder to defend against.</para>
/// <para>
/// Original paper: "Towards Evaluating the Robustness of Neural Networks"
/// by Carlini and Wagner (2017)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type for the model.</typeparam>
/// <typeparam name="TOutput">The output data type for the model.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Towards Evaluating the Robustness of Neural Networks", "https://arxiv.org/abs/1608.04644", Year = 2017, Authors = "Nicholas Carlini, David Wagner")]
public class CWAttack<T, TInput, TOutput> : AdversarialAttackBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the C and W attack.
    /// </summary>
    /// <param name="options">The configuration options for the attack.</param>
    public CWAttack(AdversarialAttackOptions<T> options) : base(options)
    {
    }

    /// <summary>
    /// Generates an adversarial example using the C and W L2 attack.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The C and W attack solves:
    /// minimize ||delta||_2 + c * f(x + delta)
    /// where f measures how well the attack succeeds.
    /// </para>
    /// <para><b>For Beginners:</b> This method tries to find the smallest possible change
    /// that will fool the model. It balances making the change small (hard to detect) with
    /// making the attack successful (fooling the model).</para>
    /// </remarks>
    /// <param name="input">The clean input to perturb.</param>
    /// <param name="trueLabel">The correct label for the input.</param>
    /// <param name="targetModel">The model to attack.</param>
    /// <returns>The adversarial example.</returns>
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
        var vectorLabel = ConversionsHelper.ConvertToVector<T, TOutput>(trueLabel);
        int trueLabelIndex = ArgMaxIndex(vectorLabel);

        var state = new OptimizationState(this, vectorInput);
        int abortCheckInterval = Math.Max(Options.Iterations / 10, 1);

        for (int iteration = 0; iteration < Options.Iterations; iteration++)
        {
            // x' = (tanh(w) + 1) / 2 keeps every candidate inside the [0, 1] box.
            var adversarial = state.CurrentInput();
            var outputVector = ConversionsHelper.ConvertToVector<T, TOutput>(
                targetModel.Predict(ConversionsHelper.ConvertVectorToInput<T, TInput>(adversarial, input)));
            int realIndex = RealClassIndex(outputVector.Length, trueLabelIndex);

            var attackLossGradient = ComputeAttackLossInputGradient(adversarial, outputVector, realIndex, input, targetModel);
            T cost = state.Step(adversarial, outputVector, realIndex, attackLossGradient);

            state.TrackBest(adversarial, IsSuccessfulAttack(outputVector, trueLabelIndex, realIndex));

            // torchattacks CW: every tenth of the run, stop once the objective stops decreasing.
            if (iteration % abortCheckInterval == 0 && state.ObjectiveStoppedDecreasing(cost))
            {
                break;
            }
        }

        // Each iteration scores the candidate before its Adam step, so the candidate of the last step was never
        // checked (with one iteration the attack could only return the clean input). Score it as well.
        if (Options.Iterations > 0)
        {
            var finalCandidate = state.CurrentInput();
            var finalOutput = ConversionsHelper.ConvertToVector<T, TOutput>(
                targetModel.Predict(ConversionsHelper.ConvertVectorToInput<T, TInput>(finalCandidate, input)));
            int finalRealIndex = RealClassIndex(finalOutput.Length, trueLabelIndex);
            state.TrackBest(finalCandidate, IsSuccessfulAttack(finalOutput, trueLabelIndex, finalRealIndex));
        }

        return ConversionsHelper.ConvertVectorToInput<T, TInput>(state.BestAdversarial, input);
    }

    /// <summary>
    /// The logit the attack loss pushes down (untargeted: the true class) or up (targeted: the target class).
    /// </summary>
    private int RealClassIndex(int classCount, int trueLabelIndex)
    {
        return Options.IsTargeted
            ? MathHelper.Clamp(Options.TargetClass, 0, Math.Max(classCount - 1, 0))
            : trueLabelIndex;
    }

    /// <summary>
    /// The paper's attack loss f_6 on the logits Z.
    /// </summary>
    /// <remarks>
    /// With other = max over i != real of Z_i: untargeted f = max(Z_real - other, -kappa), targeted
    /// f = max(other - Z_target, -kappa) (Carlini and Wagner 2017, f_6; torchattacks CW.f).
    /// Minimizing f lowers the true logit below the best other one (untargeted) or raises the target
    /// logit above every other one (targeted).
    /// </remarks>
    private T ComputeAttackLoss(Vector<T> logits, int realIndex)
    {
        T margin = Margin(logits, realIndex, out _);
        T floor = NumOps.Negate(NumOps.FromDouble(Options.CWConfidence));
        return NumOps.GreaterThan(margin, floor) ? margin : floor;
    }

    /// <summary>
    /// The unclipped margin of f: Z_real - other (untargeted) or other - Z_target (targeted).
    /// </summary>
    private T Margin(Vector<T> logits, int realIndex, out int otherIndex)
    {
        otherIndex = realIndex;
        if (logits.Length < 2)
        {
            return NumOps.Zero;
        }

        // Largest logit among the other classes (first index on ties): mask the real class to the lowest value.
        var tensor = logits.AsTensor();
        var isReal = Vector<T>.CreateStandardBasis(logits.Length, realIndex).AsTensor();
        var lowest = Engine.TensorAddScalar(new Tensor<T>(tensor.Shape.ToArray()), NumOps.MinValue);
        otherIndex = Engine.TensorArgMax(Engine.TensorWhere(isReal, lowest, tensor), 0).Data.Span[0];

        var span = logits.AsSpan();
        T realLogit = span[realIndex];
        T otherLogit = span[otherIndex];
        return Options.IsTargeted
            ? NumOps.Subtract(otherLogit, realLogit)
            : NumOps.Subtract(realLogit, otherLogit);
    }

    /// <summary>
    /// d f / d x' at the current candidate.
    /// </summary>
    /// <remarks>
    /// f is piecewise linear in the logits: above its -kappa floor, d f / d Z is +1 and -1 on the two
    /// logits of the margin. Neural networks back-propagate that vector through a recorded forward pass;
    /// other models are differentiated with central finite differences of f.
    /// </remarks>
    private Vector<T> ComputeAttackLossInputGradient(
        Vector<T> adversarial,
        Vector<T> logits,
        int realIndex,
        TInput referenceInput,
        IFullModel<T, TInput, TOutput> targetModel)
    {
        if (targetModel is NeuralNetworks.NeuralNetworkBase<T> nnModel)
        {
            var logitGradient = AttackLossLogitGradient(logits, realIndex);
            var inputTensor = Tensor<T>.FromVector(adversarial);
            using var tape = new GradientTape<T>();
            var outputTensor = nnModel.ForwardForTraining(inputTensor);
            var weights = Engine.Reshape(Tensor<T>.FromVector(logitGradient), outputTensor.Shape.ToArray());
            var product = Engine.TensorMultiply(outputTensor, weights);
            var allAxes = Enumerable.Range(0, product.Shape.Length).ToArray();
            var loss = Engine.ReduceSum(product, allAxes, keepDims: false);
            var grads = tape.ComputeGradients(loss, [inputTensor]);
            if (!grads.TryGetValue(inputTensor, out var gradient))
            {
                throw new InvalidOperationException(
                    "C&W: gradient computation failed - no gradient for input tensor.");
            }

            return gradient.ToVector();
        }

        return EstimateInputGradient(
            adversarial,
            referenceInput,
            targetModel,
            output => ComputeAttackLoss(output, realIndex),
            NumOps.FromDouble(1e-4));
    }

    /// <summary>
    /// d f / d Z: zero on the -kappa floor, otherwise +1 / -1 on the two logits of the margin.
    /// </summary>
    private Vector<T> AttackLossLogitGradient(Vector<T> logits, int realIndex)
    {
        T margin = Margin(logits, realIndex, out int otherIndex);
        T floor = NumOps.Negate(NumOps.FromDouble(Options.CWConfidence));
        if (otherIndex == realIndex || !NumOps.GreaterThan(margin, floor))
        {
            return new Vector<T>(logits.Length);
        }

        var real = Vector<T>.CreateStandardBasis(logits.Length, realIndex);
        var other = Vector<T>.CreateStandardBasis(logits.Length, otherIndex);
        return Options.IsTargeted ? Engine.Subtract<T>(other, real) : Engine.Subtract<T>(real, other);
    }

    /// <summary>
    /// Checks if an attack was successful based on the model's output.
    /// </summary>
    /// <remarks>
    /// A targeted attack succeeds when the prediction is the class its loss drives up
    /// (<paramref name="realIndex"/>, the target class clamped to the model's classes), so the success test
    /// and the loss always agree on the target.
    /// </remarks>
    private bool IsSuccessfulAttack(Vector<T> output, int trueLabel, int realIndex)
    {
        int predictedClass = ArgMaxIndex(output);
        return Options.IsTargeted
            ? predictedClass == realIndex
            : predictedClass != trueLabel;
    }

    /// <summary>
    /// Adam optimization of the tanh-space variable w, plus the best successful candidate so far.
    /// </summary>
    private sealed class OptimizationState
    {
        private const double Beta1 = 0.9;
        private const double Beta2 = 0.999;
        private const double AdamEpsilon = 1e-8;
        private static readonly double SqrtBeta2 = Math.Sqrt(Beta2);
        private static readonly double SqrtOneMinusBeta2 = Math.Sqrt(1.0 - Beta2);

        private readonly CWAttack<T, TInput, TOutput> _attack;
        private readonly Vector<T> _original;
        private readonly T _c;
        private readonly T _learningRate;
        private Vector<T> _w;
        private Vector<T> _firstMoment;
        private Vector<T> _rootSecondMoment;
        private readonly T _adamEpsilon;
        private int _step;
        private T _bestSquaredDistance;
        private T _previousCost;
        private bool _hasPreviousCost;

        public OptimizationState(CWAttack<T, TInput, TOutput> attack, Vector<T> original)
        {
            _attack = attack;
            _original = original;
            _c = NumOps.FromDouble(attack.Options.CWConstant);
            _learningRate = NumOps.FromDouble(attack.Options.CWLearningRate);

            // w = atanh(2x - 1), clipped away from +-1 so pixels at the box edges stay finite. In a type too
            // coarse to hold 0.9999 below 1 (Half rounds it to 1, where atanh is infinite and the pixel could
            // never move) the bound is the largest value below 1 instead.
            var engine = attack.Engine;
            var bound = NumOps.FromDouble(Math.Min(0.9999, 1.0 - RelativeResolution / 2));
            var scaled = engine.Subtract<T>(
                engine.Multiply<T>(original, NumOps.FromDouble(2.0)),
                engine.Fill<T>(original.Length, NumOps.One));
            _w = engine.Atanh<T>(engine.Clamp<T>(scaled, NumOps.Negate(bound), bound));
            _firstMoment = new Vector<T>(original.Length);
            _rootSecondMoment = new Vector<T>(original.Length);
            _adamEpsilon = PositiveConstant(AdamEpsilon);
            BestAdversarial = original.Clone();
            _bestSquaredDistance = NumOps.MaxValue;
            _previousCost = NumOps.MaxValue;
        }

        public Vector<T> BestAdversarial { get; private set; }

        /// <summary>x' = (tanh(w) + 1) / 2.</summary>
        public Vector<T> CurrentInput()
        {
            var engine = _attack.Engine;
            var shifted = engine.Add<T>(engine.Tanh<T>(_w), engine.Fill<T>(_w.Length, NumOps.One));
            return engine.Multiply<T>(shifted, NumOps.FromDouble(0.5));
        }

        /// <summary>
        /// One Adam step on ||x' - x||^2 + c * f(x'); returns the objective at the pre-step point.
        /// </summary>
        public T Step(Vector<T> adversarial, Vector<T> logits, int realIndex, Vector<T> attackLossGradient)
        {
            var engine = _attack.Engine;
            var diff = engine.Subtract<T>(adversarial, _original);
            T attackLoss = _attack.ComputeAttackLoss(logits, realIndex);
            T cost = NumOps.Add(engine.DotProduct<T>(diff, diff), NumOps.Multiply(_c, attackLoss));

            // d/dx' = 2 (x' - x) + c * df/dx';  dx'/dw = (1 - tanh(w)^2) / 2.
            var inputGradient = engine.Add<T>(
                engine.Multiply<T>(diff, NumOps.FromDouble(2.0)),
                engine.Multiply<T>(attackLossGradient, _c));
            var tanhW = engine.Tanh<T>(_w);
            var jacobian = engine.Multiply<T>(
                engine.Subtract<T>(engine.Fill<T>(_w.Length, NumOps.One), engine.Multiply<T>(tanhW, tanhW)),
                NumOps.FromDouble(0.5));
            var gradient = engine.Multiply<T>(inputGradient, jacobian);

            // Adam (Kingma and Ba 2015) with PyTorch's defaults, as torchattacks CW uses.
            _step++;
            _firstMoment = engine.Add<T>(
                engine.Multiply<T>(_firstMoment, NumOps.FromDouble(Beta1)),
                engine.Multiply<T>(gradient, NumOps.FromDouble(1.0 - Beta1)));
            _rootSecondMoment = RootSecondMoment(engine, gradient);
            T firstCorrection = NumOps.FromDouble(1.0 - Math.Pow(Beta1, _step));
            T rootSecondCorrection = NumOps.FromDouble(Math.Sqrt(1.0 - Math.Pow(Beta2, _step)));
            var denominator = engine.Add<T>(
                engine.Divide<T>(_rootSecondMoment, rootSecondCorrection),
                engine.Fill<T>(_w.Length, _adamEpsilon));
            var update = engine.Divide<T>(engine.Divide<T>(_firstMoment, firstCorrection), denominator);
            _w = engine.Subtract<T>(_w, engine.Multiply<T>(update, _learningRate));
            return cost;
        }

        /// <summary>
        /// sqrt(v_t) for Adam's v_t = beta2 v_{t-1} + (1 - beta2) g^2, kept as
        /// hypot(sqrt(beta2) sqrt(v_{t-1}), sqrt(1 - beta2) |g|).
        /// </summary>
        /// <remarks>
        /// The same quantity as the square root of PyTorch's exp_avg_sq, but the squares are never formed:
        /// in Half, (1 - beta2) g^2 underflows to zero for |g| below about 0.008, which leaves v = 0 and turns
        /// the update m / (sqrt(v) + eps) into a jump of g / eps.
        /// </remarks>
        private Vector<T> RootSecondMoment(IEngine engine, Vector<T> gradient)
        {
            var decayed = engine.TensorMultiplyScalar(_rootSecondMoment.AsTensor(), NumOps.FromDouble(SqrtBeta2));
            var fresh = engine.TensorMultiplyScalar(
                engine.TensorAbs(gradient.AsTensor()), NumOps.FromDouble(SqrtOneMinusBeta2));
            var larger = engine.TensorMax(decayed, fresh);
            var smaller = engine.TensorMin(decayed, fresh);
            var positive = engine.TensorGreaterThan(larger, NumOps.Zero);
            var zeros = new Tensor<T>(larger.Shape.ToArray());
            var safeLarger = engine.TensorWhere(positive, larger, engine.TensorAddScalar(zeros, NumOps.One));
            var ratio = engine.TensorWhere(positive, engine.TensorDivide(smaller, safeLarger), zeros);
            var root = engine.TensorSqrt(engine.TensorAddScalar(engine.TensorMultiply(ratio, ratio), NumOps.One));
            return engine.TensorMultiply(larger, root).ToVector();
        }

        /// <summary>Keeps the successful candidate with the smallest L2 distance.</summary>
        public void TrackBest(Vector<T> adversarial, bool successful)
        {
            if (!successful)
            {
                return;
            }

            var diff = _attack.Engine.Subtract<T>(adversarial, _original);
            T squaredDistance = _attack.Engine.DotProduct<T>(diff, diff);
            if (NumOps.LessThan(squaredDistance, _bestSquaredDistance))
            {
                _bestSquaredDistance = squaredDistance;
                BestAdversarial = adversarial.Clone();
            }
        }

        /// <summary>True when the objective rose since the previous check (the torchattacks early abort).</summary>
        public bool ObjectiveStoppedDecreasing(T cost)
        {
            if (_hasPreviousCost && NumOps.GreaterThan(cost, _previousCost))
            {
                return true;
            }

            _previousCost = cost;
            _hasPreviousCost = true;
            return false;
        }
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

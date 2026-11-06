using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.CreditAssignment;

/// <summary>
/// <b>Layerwise distillation as local credit</b>: every hidden layer of the student is trained, with its input detached,
/// to reproduce one hidden state of a frozen <b>teacher</b> model through a learned linear map
/// <c>P_k : student width → teacher width</c>; the readout is trained by a blend of the hard-label loss and the
/// teacher's soft logits (Hinton et al., 2015). Combines local credit (no gradient crosses a layer boundary below the
/// readout) with knowledge transfer from an established open-weight model (layer-to-layer hint learning, Romero et al.
/// 2015, FitNets).
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
/// <remarks>
/// <para>
/// <b>Layer map.</b> The teacher supplies <c>L_t + 1</c> hidden states per step (index 0 = its embedding output,
/// index <c>j</c> = output of its block <c>j</c>). Student hidden layer <c>k</c> of <c>H</c> is matched to teacher
/// state <c>round(k · L_t / (H − 1))</c>, so the student embedding meets the teacher embedding and the last student
/// hidden layer meets the teacher's last block, whatever the two depths are.
/// </para>
/// <para>
/// <b>Target scale.</b> Each teacher state is divided by its per-token RMS before regression, so blocks whose residual
/// norm grows with depth do not dominate the sum. The local loss is the mean squared error
/// <c>mean((h_k P_k − t̂_j)²)</c>, weighted by <c>hiddenWeight</c>.
/// </para>
/// <para>
/// <b>Projections.</b> <c>P_k</c> are owned by the rule (they are not model parameters, so they never count toward the
/// student's size). They are created on first use (scaled Gaussian, seeded) and trained by plain SGD from their own
/// gradients each step.
/// </para>
/// <para>
/// <b>For Beginners:</b> a big trained model (the teacher) shows the small model what each of its layers "looks like"
/// on the same text, and each small-model layer practises imitating one teacher layer on its own. The final layer
/// also learns from the teacher's guesses about the next word, not just the single correct word.
/// </para>
/// </remarks>
public sealed class LayerwiseDistillationCreditRule<T> : ILocalObjectiveCreditRule<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();
    private readonly Func<Tensor<T>, IReadOnlyList<Tensor<T>>> _teacherHidden;
    private readonly Func<Tensor<T>, Tensor<T>>? _teacherLogits;
    private readonly double _hiddenWeight, _kdAlpha, _temperature, _projectionLr, _localTaskWeight;
    private readonly Random _random;
    private readonly List<Tensor<T>> _projections = new();
    private readonly Dictionary<int, Tensor<T>> _projectionOf = new();
    private Tensor<T>? _stepInput;
    private IReadOnlyList<Tensor<T>>? _stepTeacher;

    /// <summary>Creates the rule.</summary>
    /// <param name="teacherHidden">Frozen teacher: input → its <c>L_t + 1</c> hidden states, each <c>[batch, sequence, teacherWidth]</c>.</param>
    /// <param name="teacherLogits">Frozen teacher logits for the same input (<c>[batch, sequence, vocabulary]</c>), or null for hidden-state distillation only.</param>
    /// <param name="readoutLayerCount">Trailing student layers forming the readout (language model: norm + head = 2).</param>
    /// <param name="hiddenWeight">Weight of each hidden layer's regression loss.</param>
    /// <param name="kdAlpha">Weight of the soft-logit term at the readout (0 = hard labels only).</param>
    /// <param name="temperature">Softmax temperature of the soft-logit term (the term is scaled by T²).</param>
    /// <param name="projectionLearningRate">SGD step size of the per-layer projections.</param>
    /// <param name="seed">Seed for projection initialisation.</param>
    /// <param name="localTaskWeight">Weight of each hidden layer's own task loss through the shared readout
    /// (<c>loss(readout(h_k), target)</c>, as <see cref="TiedReadoutLocalErrorCreditRule{T}"/>). 0 (the default) trains
    /// every detached hidden layer by teacher regression alone, so no task signal reaches it; a positive weight makes
    /// the teacher a per-layer regulariser on top of local task credit.</param>
    public LayerwiseDistillationCreditRule(
        Func<Tensor<T>, IReadOnlyList<Tensor<T>>> teacherHidden,
        Func<Tensor<T>, Tensor<T>>? teacherLogits = null,
        int readoutLayerCount = 2,
        double hiddenWeight = 1.0,
        double kdAlpha = 0.5,
        double temperature = 2.0,
        double projectionLearningRate = 1e-3,
        int? seed = null,
        double localTaskWeight = 0.0)
    {
        _teacherHidden = teacherHidden ?? throw new ArgumentNullException(nameof(teacherHidden));
        if (readoutLayerCount < 1) throw new ArgumentOutOfRangeException(nameof(readoutLayerCount), "Must be at least 1.");
        if (!(hiddenWeight >= 0)) throw new ArgumentOutOfRangeException(nameof(hiddenWeight), "Must be non-negative.");
        if (!(kdAlpha >= 0 && kdAlpha <= 1)) throw new ArgumentOutOfRangeException(nameof(kdAlpha), "Must be in [0, 1].");
        if (!(temperature > 0)) throw new ArgumentOutOfRangeException(nameof(temperature), "Must be positive.");
        if (!(projectionLearningRate > 0)) throw new ArgumentOutOfRangeException(nameof(projectionLearningRate), "Must be positive.");
        if (!(localTaskWeight >= 0)) throw new ArgumentOutOfRangeException(nameof(localTaskWeight), "Must be non-negative.");
        if (kdAlpha > 0 && teacherLogits is null) throw new ArgumentException("kdAlpha > 0 needs teacherLogits.", nameof(teacherLogits));
        _teacherLogits = teacherLogits;
        ReadoutLayerCount = readoutLayerCount;
        _hiddenWeight = hiddenWeight; _kdAlpha = kdAlpha; _temperature = temperature; _projectionLr = projectionLearningRate;
        _localTaskWeight = localTaskWeight;
        _random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed ?? 12345);
    }

    /// <inheritdoc />
    public string Name => "LayerwiseDistillation";

    /// <inheritdoc />
    public bool IsExactBackprop => false;

    /// <inheritdoc />
    public int ReadoutLayerCount { get; }

    /// <inheritdoc />
    public IReadOnlyList<Tensor<T>> AuxiliaryParameters => _projections;

    /// <inheritdoc />
    public void Initialize(ICreditAssignmentContext<T> context) { }

    /// <inheritdoc />
    public void ComputeTeachingSignals(ICreditAssignmentContext<T> context) =>
        throw new NotSupportedException($"{Name} is a local-objective rule; it is driven through {nameof(ILocalObjectiveCreditRule<T>)}.");

    private IReadOnlyList<Tensor<T>> TeacherStates(Tensor<T> input)
    {
        if (!ReferenceEquals(input, _stepInput) || _stepTeacher is null)
        {
            var states = _teacherHidden(input);
            if (states is null || states.Count < 2)
                throw new InvalidOperationException("The teacher must return its embedding output plus at least one block output.");
            var normalized = new Tensor<T>[states.Count];
            for (int j = 0; j < states.Count; j++) normalized[j] = RmsNormalizeConst(states[j]);
            _stepTeacher = normalized;
            _stepInput = input;
        }
        return _stepTeacher;
    }

    /// <inheritdoc />
    public Tensor<T>? BuildLocalObjective(LocalObjectiveContext<T> context)
    {
        var s = context.Step;
        // The last hidden layer feeds the readout undetached, so the output loss already gives it task credit.
        Tensor<T>? task = null;
        if (_localTaskWeight > 0 && context.HiddenIndex != s.HiddenLayerCount - 1)
        {
            task = s.ReadoutLoss(context.Output, s.Target);
            if (_localTaskWeight != 1.0) task = s.Engine.TensorMultiplyScalar(task, Ops.FromDouble(_localTaskWeight));
        }
        if (_hiddenWeight == 0) return task;
        var teacher = TeacherStates(s.Input);
        int lt = teacher.Count - 1, h = s.HiddenLayerCount;
        int j = h <= 1 ? lt : (int)Math.Round((double)context.HiddenIndex * lt / (h - 1));
        var target = teacher[j];
        var output = context.Output;
        int es = output.Shape[output.Shape.Length - 1], et = target.Shape[target.Shape.Length - 1];
        if (!SameLeading(output, target))
            throw new InvalidOperationException(
                $"Teacher state {j} shape [{string.Join(",", target.Shape.ToArray())}] does not match student layer " +
                $"{context.NetworkLayerIndex} output [{string.Join(",", output.Shape.ToArray())}] in its leading dimensions.");
        var p = Projection(context.HiddenIndex, es, et);
        var diff = s.Engine.TensorSubtract(s.Engine.TensorMatMul(output, p), target);
        var axes = Enumerable.Range(0, diff.Shape.Length).ToArray();
        var mse = s.Engine.ReduceMean(s.Engine.TensorSquare(diff), axes, keepDims: false);
        var regression = _hiddenWeight == 1.0 ? mse : s.Engine.TensorMultiplyScalar(mse, Ops.FromDouble(_hiddenWeight));
        return task is null ? regression : s.Engine.TensorAdd(task, regression);
    }

    /// <inheritdoc />
    public Tensor<T>? BuildOutputObjective(LocalObjectiveStep<T> step, Tensor<T> prediction, Tensor<T> defaultLoss)
    {
        if (_kdAlpha == 0 || _teacherLogits is null) return null;
        var e = step.Engine;
        int axis = prediction.Shape.Length - 1;
        var teacherLogits = _teacherLogits(step.Input);
        if (!teacherLogits.Shape.ToArray().SequenceEqual(prediction.Shape.ToArray()))
            throw new InvalidOperationException(
                $"Teacher logits [{string.Join(",", teacherLogits.Shape.ToArray())}] do not match student logits [{string.Join(",", prediction.Shape.ToArray())}].");
        // Soft term: T² · mean over positions of KL(p_t || p_s) = T² · mean(Σ p_t (log p_t − log p_s)); the p_t log p_t part is
        // constant in the student, so the student gradient is that of −T² · mean(Σ p_t log p_s).
        var invT = Ops.FromDouble(1.0 / _temperature);
        var pt = SoftmaxConst(teacherLogits, _temperature);
        var logPs = e.TensorLogSoftmax(e.TensorMultiplyScalar(prediction, invT), axis);
        var cross = e.ReduceSum(e.TensorMultiply(logPs, pt), new[] { axis }, keepDims: false);
        var positions = Enumerable.Range(0, cross.Shape.Length).ToArray();
        var soft = e.TensorMultiplyScalar(e.ReduceMean(cross, positions, keepDims: false), Ops.FromDouble(-_temperature * _temperature));
        var hard = e.TensorMultiplyScalar(defaultLoss, Ops.FromDouble(1 - _kdAlpha));
        return e.TensorAdd(hard, e.TensorMultiplyScalar(soft, Ops.FromDouble(_kdAlpha)));
    }

    /// <inheritdoc />
    public void ApplyAuxiliaryGradients(IReadOnlyDictionary<Tensor<T>, Tensor<T>> gradients)
    {
        var lr = _projectionLr;
        foreach (var p in _projections)
        {
            if (!gradients.TryGetValue(p, out var g) || g.Length != p.Length) continue;
            for (int i = 0; i < p.Length; i++)
                p[i] = Ops.Subtract(p[i], Ops.Multiply(Ops.FromDouble(lr), g[i]));
        }
        _stepInput = null; _stepTeacher = null;
    }

    private Tensor<T> Projection(int hiddenIndex, int es, int et)
    {
        if (_projectionOf.TryGetValue(hiddenIndex, out var existing) && existing.Shape[0] == es && existing.Shape[1] == et)
            return existing;
        var p = new Tensor<T>(new[] { es, et });
        double scale = 1.0 / Math.Sqrt(es);
        for (int i = 0; i < p.Length; i++)
        {
            double u1 = 1.0 - _random.NextDouble(), u2 = _random.NextDouble();
            p[i] = Ops.FromDouble(scale * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
        }
        if (existing is not null) _projections.Remove(existing);
        _projectionOf[hiddenIndex] = p;
        _projections.Add(p);
        return p;
    }

    private static bool SameLeading(Tensor<T> a, Tensor<T> b)
    {
        if (a.Shape.Length != b.Shape.Length) return false;
        for (int i = 0; i < a.Shape.Length - 1; i++) if (a.Shape[i] != b.Shape[i]) return false;
        return true;
    }

    private static Tensor<T> RmsNormalizeConst(Tensor<T> t)
    {
        int w = t.Shape[t.Shape.Length - 1], rows = t.Length / w;
        var v = t.ToVector(); var r = new Vector<T>(v.Length);
        for (int row = 0; row < rows; row++)
        {
            double ss = 0;
            for (int c = 0; c < w; c++) { double x = Ops.ToDouble(v[row * w + c]); ss += x * x; }
            double inv = 1.0 / Math.Sqrt(ss / w + 1e-6);
            for (int c = 0; c < w; c++) r[row * w + c] = Ops.FromDouble(Ops.ToDouble(v[row * w + c]) * inv);
        }
        return new Tensor<T>(t.Shape.ToArray(), r);
    }

    private static Tensor<T> SoftmaxConst(Tensor<T> logits, double temperature)
    {
        int w = logits.Shape[logits.Shape.Length - 1], rows = logits.Length / w;
        var v = logits.ToVector(); var r = new Vector<T>(v.Length);
        for (int row = 0; row < rows; row++)
        {
            double max = double.NegativeInfinity;
            for (int c = 0; c < w; c++) max = Math.Max(max, Ops.ToDouble(v[row * w + c]) / temperature);
            double sum = 0;
            for (int c = 0; c < w; c++) sum += Math.Exp(Ops.ToDouble(v[row * w + c]) / temperature - max);
            for (int c = 0; c < w; c++) r[row * w + c] = Ops.FromDouble(Math.Exp(Ops.ToDouble(v[row * w + c]) / temperature - max) / sum);
        }
        return new Tensor<T>(logits.Shape.ToArray(), r);
    }
}

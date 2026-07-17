using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation;

/// <summary>
/// Presents an <see cref="IDistillationStrategy{T}"/> as a tape-capable loss so the standard gradient
/// path can train a student with distillation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This is the "tape-aware wrapper around the standard loss" knowledge distillation needs. The
/// distillation loss combines a soft term against the teacher's logits with a hard term against the
/// true label; that signal cannot travel through the optimizer's (predicted, actual) loss contract,
/// which carries no teacher. The strategy already computes the loss and its gradient with respect to
/// the student output, but on plain matrices — off the autodiff tape.
/// </para>
/// <para>
/// <b>How the gradient reaches the parameters.</b> The gradient-computable path backpropagates through
/// <see cref="ComputeTapeLoss"/>, so returning the strategy's gradient from <c>CalculateDerivative</c>
/// is not enough — the fallback builds a tensor disconnected from the tape, whose parameter gradients
/// are zero. Instead this uses the standard surrogate-loss construction: with the strategy's gradient
/// <c>g</c> treated as a constant, the scalar <c>L = Σ (g · ŷ)</c> is a genuine tape expression over
/// the prediction <c>ŷ</c>, and <c>∂L/∂ŷ = g</c> exactly. The tape then carries <c>g</c> back through
/// the student to real parameter gradients. The scalar's VALUE is meaningless (it is a surrogate, not
/// the distillation loss); only its gradient matters, which is the point.
/// </para>
/// </remarks>
public sealed class DistillationLossAdapter<T> : LossFunctionBase<T>
{
    private readonly IDistillationStrategy<T> _strategy;

    /// <summary>Creates an adapter over a distillation strategy.</summary>
    /// <param name="strategy">The strategy that computes the distillation loss and its gradient.</param>
    public DistillationLossAdapter(IDistillationStrategy<T> strategy)
        => _strategy = strategy ?? throw new ArgumentNullException(nameof(strategy));

    /// <summary>
    /// The true label for the sample currently being trained, or null when labels are unavailable.
    /// </summary>
    /// <remarks>Set by the training loop before each gradient step; the hard-label term uses it.</remarks>
    public Vector<T>? CurrentTrueLabel { get; set; }

    /// <inheritdoc />
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
        => _strategy.ComputeLoss(ToRow(predicted), ToRow(actual), LabelRow());

    /// <inheritdoc />
    /// <remarks>
    /// <paramref name="predicted"/> is the student output, <paramref name="actual"/> the teacher
    /// output. Returns the distillation gradient with respect to the student output.
    /// </remarks>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        var gradient = _strategy.ComputeGradient(ToRow(predicted), ToRow(actual), LabelRow());
        var result = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            result[i] = gradient[0, i];
        }

        return result;
    }

    /// <inheritdoc />
    /// <remarks>
    /// Builds the surrogate <c>Σ (g · ŷ)</c> where <c>g</c> is the strategy's gradient at the current
    /// prediction values, held as a constant. Differentiating this on the tape yields <c>g</c> as the
    /// prediction cotangent, which is exactly what distillation needs to propagate.
    /// </remarks>
    public override Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
    {
        // g = strategy gradient at the CURRENT prediction/teacher values (detached — plain numbers).
        var predVec = predicted.ToVector();
        var teacherVec = target.ToVector();
        var g = _strategy.ComputeGradient(ToRow(predVec), ToRow(teacherVec), LabelRow());

        var gConst = new Tensor<T>(predicted._shape);
        int cols = predVec.Length;
        for (int i = 0; i < cols; i++)
        {
            gConst[i] = g[0, i];
        }

        // Σ (g · ŷ): a tape expression over `predicted`, so backprop seeds the prediction cotangent
        // with g and carries it into the student's parameters.
        var product = Engine.TensorMultiply(gConst, predicted);
        var allAxes = Enumerable.Range(0, product.Shape.Length).ToArray();
        return Engine.ReduceSum(product, allAxes, keepDims: false);
    }

    private Matrix<T> LabelRow() => CurrentTrueLabel is null ? null! : ToRow(CurrentTrueLabel);

    private static Matrix<T> ToRow(Vector<T> v)
    {
        var m = new Matrix<T>(1, v.Length);
        for (int i = 0; i < v.Length; i++)
        {
            m[0, i] = v[i];
        }

        return m;
    }
}

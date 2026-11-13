using AiDotNet.Interfaces;

namespace AiDotNet.ReinforcementLearning.Agents.DecisionTransformer;

/// <summary>
/// The Decision Transformer action objective: the loss between the action predicted at each real state token
/// and the dataset action, ignoring every other token of the sequence.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The transformer emits one output per token, three per timestep (return-to-go, state, action). Chen et al.
/// 2021 (Sec. 3) predict the action a_t from the output at the state token s_t only, and train with
/// mean-squared error for continuous actions or cross-entropy for discrete ones, averaged over the real
/// timesteps of each segment. Padding positions and the return and action token outputs contribute nothing.
/// </para>
/// <para>
/// Network outputs and targets are both laid out <c>[batch, tokens, actionSize]</c>. The rows that carry an
/// action label are given once, as indices into the flattened <c>[batch * tokens]</c> axis, and are gathered
/// with an engine op so the objective stays on the gradient tape.
/// </para>
/// </remarks>
internal sealed class DecisionTransformerActionLoss<T> : ILossFunction<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private static IEngine Engine => AiDotNetEngine.Current;

    private readonly int[] _actionRows;
    private readonly int _actionSize;
    private readonly bool _continuous;
    private readonly ILossFunction<T> _continuousLoss;

    /// <summary>
    /// Creates the objective for one batch.
    /// </summary>
    /// <param name="actionRows">Flattened <c>[batch * tokens]</c> indices of the state tokens of real timesteps.</param>
    /// <param name="actionSize">The action width (continuous) or the number of actions (discrete).</param>
    /// <param name="continuous">True for the continuous regression objective; false for cross-entropy.</param>
    /// <param name="continuousLoss">The regression loss applied to the gathered continuous actions (MSE in the paper).</param>
    public DecisionTransformerActionLoss(int[] actionRows, int actionSize, bool continuous, ILossFunction<T> continuousLoss)
    {
        if (actionRows is null) throw new ArgumentNullException(nameof(actionRows));
        if (actionRows.Length == 0)
            throw new ArgumentException("At least one labelled timestep is required.", nameof(actionRows));
        if (actionSize <= 0) throw new ArgumentOutOfRangeException(nameof(actionSize));
        if (continuousLoss is null) throw new ArgumentNullException(nameof(continuousLoss));

        _actionRows = (int[])actionRows.Clone();
        _actionSize = actionSize;
        _continuous = continuous;
        _continuousLoss = continuousLoss;
    }

    /// <inheritdoc/>
    public Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
    {
        if (predicted is null) throw new ArgumentNullException(nameof(predicted));
        if (target is null) throw new ArgumentNullException(nameof(target));
        int rows = RowCount(predicted.Length);
        if (target.Length != predicted.Length)
            throw new ArgumentException("Predicted and target tensors must have the same length.", nameof(target));

        var indices = new Tensor<int>((int[])_actionRows.Clone(), [_actionRows.Length]);
        var predictedRows = Engine.TensorGather(Engine.Reshape(predicted, [rows, _actionSize]), indices, 0);
        var targetRows = Engine.TensorGather(Engine.Reshape(target, [rows, _actionSize]), indices, 0);

        if (_continuous)
            return _continuousLoss.ComputeTapeLoss(predictedRows, targetRows);

        // Cross-entropy -mean_i sum_a y_ia log softmax(z_i)_a, with one-hot targets y. log_softmax is taken directly
        // (z - logsumexp z), as torch's cross_entropy does: log(softmax + eps) caps the loss at -log(eps) and shrinks the
        // gradient of a confidently wrong row by p / (p + eps), so the worst-predicted actions barely learned.
        var logProbabilities = Engine.TensorLogSoftmax(predictedRows, axis: -1);
        var labelled = Engine.ReduceSum(Engine.TensorMultiply(logProbabilities, targetRows), [1], keepDims: false);
        return Engine.TensorNegate(Engine.ReduceMean(labelled, [0], keepDims: false));
    }

    /// <inheritdoc/>
    public T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        if (predicted is null) throw new ArgumentNullException(nameof(predicted));
        if (actual is null) throw new ArgumentNullException(nameof(actual));
        if (predicted.Length != actual.Length)
            throw new ArgumentException("Predicted and actual vectors must have the same length.", nameof(actual));
        RowCount(predicted.Length);

        var gatheredPredicted = GatherRows(predicted);
        var gatheredActual = GatherRows(actual);
        if (_continuous) return _continuousLoss.CalculateLoss(gatheredPredicted, gatheredActual);

        // -mean_i sum_a y_ia log softmax(z_i)_a
        var logProbabilities = LogSoftmaxRows(gatheredPredicted);
        T total = Engine.TensorSum(Engine.TensorMultiply(logProbabilities, LabelledRows(gatheredActual)));

        return NumOps.Negate(NumOps.Divide(total, NumOps.FromDouble(_actionRows.Length)));
    }

    /// <inheritdoc/>
    public (T Loss, Tensor<T> Gradient) CalculateLossAndGradientGpu(Tensor<T> predicted, Tensor<T> actual)
    {
        if (predicted is null) throw new ArgumentNullException(nameof(predicted));
        if (actual is null) throw new ArgumentNullException(nameof(actual));
        if (predicted.Length != actual.Length)
            throw new ArgumentException("Predicted and actual tensors must have the same length.", nameof(actual));
        RowCount(predicted.Length);

        var predictedFlat = predicted.ToVector();
        var actualFlat = actual.ToVector();
        var gatheredPredicted = GatherRows(predictedFlat);
        var gatheredActual = GatherRows(actualFlat);
        T loss;
        Tensor<T> labelledGradient;

        if (_continuous)
        {
            var shape = new[] { _actionRows.Length, _actionSize };
            var (innerLoss, innerGradient) = _continuousLoss.CalculateLossAndGradientGpu(
                LabelledRows(gatheredPredicted), LabelledRows(gatheredActual));
            loss = innerLoss;
            labelledGradient = Engine.Reshape(innerGradient, shape);
        }
        else
        {
            // d(mean cross-entropy)/dz = (softmax(z) - y) / N for one-hot y.
            loss = CalculateLoss(predictedFlat, actualFlat);
            var probabilities = Engine.TensorExp(LogSoftmaxRows(gatheredPredicted));
            labelledGradient = Engine.TensorDivideScalar(
                Engine.TensorSubtract(probabilities, LabelledRows(gatheredActual)), NumOps.FromDouble(_actionRows.Length));
        }

        // Each labelled row's gradient lands back on its token row (repeated rows accumulate); the rest stay zero.
        int rows = predicted.Length / _actionSize;
        var gradient = Engine.TensorScatterAdd(new Tensor<T>([rows, _actionSize]), ActionRowIndices(), labelledGradient, 0);
        return (loss, Engine.Reshape(gradient, predicted.Shape.ToArray()));
    }

    /// <summary>Number of token rows in a flattened <c>[batch, tokens, actionSize]</c> buffer, validating the labelled rows.</summary>
    private int RowCount(int length)
    {
        if (length % _actionSize != 0)
            throw new ArgumentException(
                $"The output length {length} is not a whole number of {_actionSize}-wide action rows.");

        int rows = length / _actionSize;
        foreach (int row in _actionRows)
        {
            if ((uint)row >= (uint)rows)
                throw new ArgumentException($"Labelled row {row} is outside the {rows} token rows of the output.");
        }

        return rows;
    }

    /// <summary>Copies the labelled rows of a flattened <c>[rows, actionSize]</c> buffer into a compact vector.</summary>
    private Vector<T> GatherRows(Vector<T> flat)
    {
        var table = Engine.Reshape(flat.AsTensor(), [flat.Length / _actionSize, _actionSize]);
        return Engine.TensorGather(table, ActionRowIndices(), 0).ToVector();
    }

    /// <summary>The labelled rows as a [labelled, action] tensor.</summary>
    private Tensor<T> LabelledRows(Vector<T> gathered)
        => Engine.Reshape(gathered.AsTensor(), [_actionRows.Length, _actionSize]);

    /// <summary>log softmax over each labelled row (z - logsumexp z).</summary>
    private Tensor<T> LogSoftmaxRows(Vector<T> gathered)
        => Engine.TensorLogSoftmax(LabelledRows(gathered), axis: -1);

    /// <summary>The labelled token rows as a gather/scatter index tensor.</summary>
    private Tensor<int> ActionRowIndices()
        => new Tensor<int>((int[])_actionRows.Clone(), [_actionRows.Length]);
}

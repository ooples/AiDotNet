using System;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation;

/// <summary>
/// Presents an <see cref="IDistillationStrategy{T}"/> as a tape-capable <see cref="LossFunctionBase{T}"/> so both
/// reverse-mode autodiff and non-backprop <see cref="ICreditRule{T}"/> trainers can optimize students over mini-batches.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public sealed class DistillationLossAdapter<T> : LossFunctionBase<T>
{
    private readonly IDistillationStrategy<T> _strategy;

    /// <summary>Creates an adapter over a distillation strategy.</summary>
    /// <param name="strategy">The strategy that computes the distillation loss and its gradient.</param>
    public DistillationLossAdapter(IDistillationStrategy<T> strategy)
        => _strategy = strategy ?? throw new ArgumentNullException(nameof(strategy));

    /// <summary>
    /// Gets the underlying distillation strategy wrapped by this adapter.
    /// </summary>
    public IDistillationStrategy<T> Strategy => _strategy;

    /// <summary>
    /// The true label vector for a single sample, or null when labels are unavailable or supplied via <see cref="CurrentTrueLabelBatch"/>.
    /// </summary>
    public Vector<T>? CurrentTrueLabel { get; set; }

    /// <summary>
    /// The true label tensor for the current mini-batch (either <c>[batchSize, numClasses]</c> probabilities/one-hot,
    /// <c>[batchSize, seqLen, numClasses]</c>, or integer class indices of length <c>rows</c>), or null when distilling from teacher logits alone.
    /// </summary>
    public Tensor<T>? CurrentTrueLabelBatch { get; set; }

    /// <inheritdoc />
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        using var _ = new NoGradScope<T>();
        var predRow = ToSingleRow(predicted);
        var actualRow = ToSingleRow(actual);
        return _strategy.ComputeLoss(predRow, actualRow, ResolveLabelsMatrix(rows: 1, numClasses: predRow.Columns));
    }

    /// <summary>
    /// Computes the true distillation loss over a mini-batch tensor without reshaping multiple rows into one class axis.
    /// </summary>
    public T CalculateBatchLoss(Tensor<T> predicted, Tensor<T> teacherOutput)
    {
        if (predicted is null) throw new ArgumentNullException(nameof(predicted));
        if (teacherOutput is null) throw new ArgumentNullException(nameof(teacherOutput));

        using var _ = new NoGradScope<T>();
        var predMat = ToBatchMatrix(predicted);
        var teacherMat = ToBatchMatrix(teacherOutput);
        return _strategy.ComputeLoss(predMat, teacherMat, ResolveLabelsMatrix(predMat.Rows, predMat.Columns));
    }

    /// <inheritdoc />
    public override Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
    {
        if (predicted is null) throw new ArgumentNullException(nameof(predicted));
        if (target is null) throw new ArgumentNullException(nameof(target));

        Matrix<T> g;
        using (new NoGradScope<T>())
        {
            var predMat = ToBatchMatrix(predicted);
            var teacherMat = ToBatchMatrix(target);
            g = _strategy.ComputeGradient(predMat, teacherMat, ResolveLabelsMatrix(predMat.Rows, predMat.Columns));
        }

        var gConst = Engine.Reshape(Tensor<T>.FromMatrix(g), predicted.Shape.ToArray());
        var product = Engine.TensorMultiply(gConst, predicted);
        var allAxes = Enumerable.Range(0, product.Shape.Length).ToArray();
        return Engine.ReduceSum(product, allAxes, keepDims: false);
    }

    private Matrix<T> ToBatchMatrix(Tensor<T> tensor)
    {
        if (tensor.Shape.Length <= 1)
        {
            return Engine.Reshape(tensor, [1, Math.Max(1, tensor.Length)]).ToMatrix();
        }

        if (tensor.Shape.Length == 2)
        {
            return tensor.ToMatrix();
        }

        int cols = Math.Max(1, tensor.Shape[tensor.Shape.Length - 1]);
        int rows = Math.Max(1, tensor.Length / cols);
        return Engine.Reshape(tensor, [rows, cols]).ToMatrix();
    }

    private Matrix<T>? ResolveLabelsMatrix(int rows, int numClasses)
    {
        if (CurrentTrueLabelBatch is { Length: > 0 } batchLabels)
        {
            if (batchLabels.Length == rows * numClasses)
            {
                return batchLabels.Shape.Length == 2 && batchLabels.Shape[0] == rows && batchLabels.Shape[1] == numClasses
                    ? batchLabels.ToMatrix()
                    : Engine.Reshape(batchLabels, [rows, numClasses]).ToMatrix();
            }

            if (numClasses > 1 && batchLabels.Length == rows)
            {
                var flatIndices = Engine.Reshape(batchLabels, [rows]).Cast<int>();
                return Engine.TensorOneHot<T>(flatIndices, numClasses).ToMatrix();
            }
        }

        if (CurrentTrueLabel is { Length: > 0 } singleLabel)
        {
            var labelTensor = Tensor<T>.FromVector(singleLabel);
            if (singleLabel.Length == rows * numClasses)
            {
                return Engine.Reshape(labelTensor, [rows, numClasses]).ToMatrix();
            }

            if (numClasses > 1 && singleLabel.Length == rows)
            {
                var flatIndices = Engine.Reshape(labelTensor, [rows]).Cast<int>();
                return Engine.TensorOneHot<T>(flatIndices, numClasses).ToMatrix();
            }

            if (rows == 1)
            {
                return ToSingleRow(singleLabel);
            }
        }

        return null;
    }

    private Matrix<T> ToSingleRow(Vector<T> v)
        => Engine.Reshape(Tensor<T>.FromVector(v), [1, v.Length]).ToMatrix();
}

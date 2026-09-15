using System;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Shape conventions shared by the classifier meta-learners: one row per example in, one score row per example out.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// A classifier meta-learner reads one embedding per example and emits one score per class for each example. Its
/// body can hand the embeddings back as a <c>[rows, width]</c> tensor, a matrix, or - for a body that emits one
/// value per example - a vector; its labels arrive as class indices in whichever of those the task uses. These
/// helpers put all of them into the two shapes the algorithms compute with, and put the scores back into the
/// learner's output type.
/// </para>
/// <para>
/// Several classifiers used to flatten a whole batch into one vector and treat it as ONE example's features, so a
/// batch of four examples produced one score per class instead of four. These helpers never flatten across
/// examples.
/// </para>
/// </remarks>
internal static class ClassifierOutputs<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// A model output as one row per example, <c>[rows, width]</c>. A vector or a rank-1 tensor holds one scalar
    /// per example, so it becomes <c>[rows, 1]</c>.
    /// </summary>
    internal static Tensor<T> AsRows(object? output)
    {
        switch (output)
        {
            case Tensor<T> tensor when tensor.Shape.Length == 2:
                return tensor;
            case Tensor<T> tensor when tensor.Shape.Length == 1:
                return tensor.Reshape(tensor.Shape[0], 1);
            case Tensor<T> tensor when tensor.Shape.Length > 2:
                // [rows, ...features]: everything after the first axis is one example's representation.
                return tensor.Reshape(tensor.Shape[0], tensor.Length / tensor.Shape[0]);
            case Vector<T> vector:
                return Tensor<T>.FromVector(vector).Reshape(vector.Length, 1);
            case Matrix<T> matrix:
            {
                var rows = new Tensor<T>(new[] { matrix.Rows, matrix.Columns });
                for (int r = 0; r < matrix.Rows; r++)
                {
                    for (int c = 0; c < matrix.Columns; c++) rows[r * matrix.Columns + c] = matrix[r, c];
                }

                return rows;
            }

            default:
                throw new NotSupportedException(
                    $"A classifier meta-learner needs one row per example, but the model produced "
                    + $"'{output?.GetType().Name ?? "null"}'. Use Tensor<T>, Matrix<T> or Vector<T> outputs.");
        }
    }

    /// <summary>
    /// Class indices as a <c>[rows]</c> tensor, validated to be whole numbers in <c>[0, numClasses)</c>.
    /// </summary>
    internal static Tensor<T> Labels(object? labels, int numClasses)
    {
        Tensor<T> flat = labels switch
        {
            Tensor<T> tensor => tensor.Reshape(tensor.Length),
            Vector<T> vector => Tensor<T>.FromVector(vector),
            Matrix<T> matrix => AsRows(matrix).Reshape(matrix.Rows * matrix.Columns),
            _ => throw new NotSupportedException(
                $"Class labels must be a Tensor<T>, Vector<T> or Matrix<T> of class indices, not "
                + $"'{labels?.GetType().Name ?? "null"}'."),
        };

        for (int i = 0; i < flat.Length; i++)
        {
            double value = Ops.ToDouble(flat[i]);
            if (value < 0 || value >= numClasses || Math.Abs(value - Math.Round(value)) > 1e-9)
            {
                throw new ArgumentException(
                    $"Label {value} at position {i} is not a class index in [0, {numClasses}). Classifier "
                    + "meta-learners take one class index per example.", nameof(labels));
            }
        }

        return flat;
    }

    /// <summary>
    /// A learner's scores back as <c>[rows, classes]</c>, whatever its output type flattened them to.
    /// </summary>
    internal static Tensor<T> ScoreRows(object? scores, int rows)
    {
        var asRows = AsRows(scores);
        if (asRows.Shape[0] == rows) return asRows;
        if (rows > 0 && asRows.Length % rows == 0) return asRows.Reshape(rows, asRows.Length / rows);
        throw new ArgumentException(
            $"{asRows.Length} scores do not divide into {rows} examples.", nameof(scores));
    }

    /// <summary>
    /// Support rows followed by query rows: one input an embedding network processes in one pass, so an episode's
    /// support and query embeddings share one tape.
    /// </summary>
    /// <exception cref="NotSupportedException">The inputs are neither matrices nor tensors.</exception>
    internal static TInput StackRows<TInput>(TInput support, TInput query)
    {
        if (support is Matrix<T> supportMatrix && query is Matrix<T> queryMatrix)
        {
            var stacked = new Matrix<T>(supportMatrix.Rows + queryMatrix.Rows, supportMatrix.Columns);
            for (int r = 0; r < supportMatrix.Rows; r++)
                for (int c = 0; c < supportMatrix.Columns; c++) stacked[r, c] = supportMatrix[r, c];
            for (int r = 0; r < queryMatrix.Rows; r++)
                for (int c = 0; c < queryMatrix.Columns; c++) stacked[supportMatrix.Rows + r, c] = queryMatrix[r, c];
            return (TInput)(object)stacked;
        }

        if (support is Tensor<T> supportTensor && query is Tensor<T> queryTensor)
        {
            var shape = supportTensor.Shape.ToArray();
            shape[0] += queryTensor.Shape[0];
            var stacked = new Tensor<T>(shape);
            for (int i = 0; i < supportTensor.Length; i++) stacked[i] = supportTensor[i];
            for (int i = 0; i < queryTensor.Length; i++) stacked[supportTensor.Length + i] = queryTensor[i];
            return (TInput)(object)stacked;
        }

        throw new NotSupportedException(
            "A classifier meta-learner embeds support and query examples in one pass, which needs Matrix<T> or "
            + $"Tensor<T> inputs, not {typeof(TInput).Name}.");
    }

    /// <summary>
    /// The loss of an adapted classifier's predictions. Score rows are class probabilities: the configured loss of
    /// their logarithm against the class indices - with cross-entropy, <c>-log p(true class)</c>. A Vector output
    /// carries one predicted class per example instead, and its loss is the classification error rate.
    /// </summary>
    internal static T ProbabilityLoss(ILossFunction<T> loss, object? predictions, object? expected)
    {
        var labels = Labels(expected, int.MaxValue);
        if (predictions is Vector<T> predictedClasses)
        {
            int wrong = 0;
            for (int i = 0; i < labels.Length; i++)
            {
                if (i >= predictedClasses.Length || Math.Abs(Ops.ToDouble(predictedClasses[i]) - Ops.ToDouble(labels[i])) > 0.5)
                    wrong++;
            }

            return Ops.FromDouble(labels.Length == 0 ? 0 : (double)wrong / labels.Length);
        }

        var probabilities = ScoreRows(predictions, labels.Length);
        using var noGrad = new NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        var logProbabilities = engine.TensorLog(engine.TensorClampMin(probabilities, Ops.FromDouble(1e-12)));

        // The target is one-hot rather than a vector of class indices. A loss defined over indices - cross
        // entropy with logits - accepts both forms, but one defined over a same-shape target - categorical
        // cross-entropy, or the squared error that every model's DefaultLossFunction returns - subtracts the
        // two elementwise and threw "Tensor shapes must match. Got [rows, classes] and [rows]". One-hot is the
        // form every loss here understands, and for the index-consuming case it is the same number:
        // -sum_c y_c log p_c collapses to -log p(true class).
        int classes = logProbabilities.Shape.Length > 1 ? logProbabilities.Shape[1] : 1;
        var oneHot = new Tensor<T>(new[] { labels.Length, classes });
        for (int row = 0; row < labels.Length; row++)
        {
            int column = (int)Math.Round(Ops.ToDouble(labels[row]));
            if (column >= 0 && column < classes) oneHot[row * classes + column] = Ops.One;
        }

        return loss.ComputeTapeLoss(logProbabilities, oneHot)[0];
    }

    /// <summary>
    /// The loss of an adapted classifier whose output is scores rather than probabilities: the configured loss of
    /// the score rows against the class indices, or the error rate of a Vector of predicted classes.
    /// </summary>
    internal static T ScoreLoss(ILossFunction<T> loss, object? predictions, object? expected, int numClasses)
    {
        var labels = Labels(expected, numClasses);
        if (predictions is Vector<T> predictedClasses)
        {
            int wrong = 0;
            for (int i = 0; i < labels.Length; i++)
            {
                if (i >= predictedClasses.Length || Math.Abs(Ops.ToDouble(predictedClasses[i]) - Ops.ToDouble(labels[i])) > 0.5)
                    wrong++;
            }

            return Ops.FromDouble(labels.Length == 0 ? 0 : (double)wrong / labels.Length);
        }

        using var noGrad = new NoGradScope<T>();
        return loss.ComputeTapeLoss(ScoreRows(predictions, labels.Length), labels)[0];
    }

    /// <summary><c>[rows, classes]</c> scores as the learner's output type.</summary>
    internal static TOutput ToOutput<TOutput>(Tensor<T> scores)
    {
        if (typeof(TOutput) == typeof(Tensor<T>)) return (TOutput)(object)scores;
        if (typeof(TOutput) == typeof(Vector<T>)) return (TOutput)(object)scores.ToVector();
        if (typeof(TOutput) == typeof(T[])) return (TOutput)(object)scores.ToVector().ToArray();
        if (typeof(TOutput) == typeof(Matrix<T>))
        {
            int rows = scores.Shape[0], classes = scores.Shape[1];
            var matrix = new Matrix<T>(rows, classes);
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < classes; c++) matrix[r, c] = scores[r * classes + c];
            }

            return (TOutput)(object)matrix;
        }

        throw new NotSupportedException(
            $"Cannot return classifier scores as {typeof(TOutput).Name}. Supported: Tensor<T>, Vector<T>, Matrix<T>, T[].");
    }
}

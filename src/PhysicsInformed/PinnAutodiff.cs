using System;
using System.Collections.Generic;
using AiDotNet.PhysicsInformed.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.PhysicsInformed
{
    /// <summary>
    /// Derivatives of a network's outputs with respect to its inputs, taken through the gradient tape.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <remarks>
    /// <para>
    /// Raissi et al. (2019), Sec. 2, evaluate the PDE residual f := u_t + N[u] with automatic differentiation of the
    /// network u(t, x) with respect to its inputs, not with finite differences. The gradient of sum_i u_k(x_i) with
    /// respect to the batch X has row i equal to du_k/dx (x_i), because each row of the output depends only on its own
    /// input row; applying the same step to each gradient column gives the Hessian, and once more the third
    /// derivatives. Every step records its backward with <c>createGraph</c>, so the derivatives stay differentiable
    /// and a training objective built from them reaches the network's weights.
    /// </para>
    /// <para>
    /// The derivatives are returned batched: first [N, out, in], second [N, out, in, in], third
    /// [N, out, in, in, in], the layout <see cref="PDEs.PDESpecificationBase{T}.ComputeTapeResidual"/> reads.
    /// </para>
    /// </remarks>
    internal static class PinnAutodiff<T>
    {
        /// <summary>
        /// Evaluates <paramref name="forward"/> at <paramref name="points"/> and its input derivatives up to
        /// <paramref name="order"/>.
        /// </summary>
        /// <param name="forward">The network: [N, in] to [N, out], each output row a function of its own input row.</param>
        /// <param name="points">The evaluation points [N, in].</param>
        /// <param name="order">The highest derivative order to compute (0 to 3).</param>
        /// <param name="derivatives">The batched derivatives.</param>
        /// <returns>The network outputs [N, out].</returns>
        public static Tensor<T> Evaluate(
            Func<Tensor<T>, Tensor<T>> forward,
            Tensor<T> points,
            int order,
            out PDEDerivatives<T> derivatives)
        {
            if (forward is null)
            {
                throw new ArgumentNullException(nameof(forward));
            }

            if (points is null)
            {
                throw new ArgumentNullException(nameof(points));
            }

            if (points.Rank != 2)
            {
                throw new ArgumentException("Points must be a [N, inputDim] tensor.", nameof(points));
            }

            if (order < 0 || order > 3)
            {
                throw new ArgumentOutOfRangeException(nameof(order), "Derivative order must be between 0 and 3.");
            }

            var engine = AiDotNetEngine.Current;
            derivatives = new PDEDerivatives<T>();
            if (order == 0)
            {
                return forward(points);
            }

            int n = points.Shape[0];
            int inputDim = points.Shape[1];

            // A fresh leaf: the tape differentiates with respect to exactly the tensor the forward reads.
            var x = new Tensor<T>(points.Shape.ToArray());
            points.AsSpan().CopyTo(x.AsWritableSpan());

            // Outside any step arena: the derivatives outlive this tape, and an arena may reissue a step's tensors
            // at a tape boundary (the same reason WGAN-GP's penalty suspends it).
            using var noArena = TensorArena.Suspend();
            using var tape = new GradientTape<T>(new GradientTapeOptions { Persistent = true });

            var outputs = forward(x);
            if (outputs.Rank != 2 || outputs.Shape[0] != n)
            {
                throw new InvalidOperationException(
                    $"The network must map [N, {inputDim}] points to [N, outputDim] values; got rank {outputs.Rank}.");
            }

            int outputDim = outputs.Shape[1];

            // Level 1: one gradient per output component, each [N, in].
            var first = new Tensor<T>[outputDim];
            for (int k = 0; k < outputDim; k++)
            {
                first[k] = InputGradient(tape, engine, Column(engine, outputs, k), x);
            }

            derivatives.FirstDerivatives = engine.TensorStack(first, 1);
            if (order == 1)
            {
                return outputs;
            }

            // Level 2: d/dx of every first-derivative column, [N, in] per (k, i).
            var second = new Tensor<T>[outputDim];
            var secondColumns = new Tensor<T>[outputDim][];
            for (int k = 0; k < outputDim; k++)
            {
                var rows = new Tensor<T>[inputDim];
                for (int i = 0; i < inputDim; i++)
                {
                    rows[i] = InputGradient(tape, engine, Column(engine, first[k], i), x);
                }

                secondColumns[k] = rows;
                second[k] = engine.TensorStack(rows, 1);
            }

            derivatives.SecondDerivatives = engine.TensorStack(second, 1);
            if (order == 2)
            {
                return outputs;
            }

            // Level 3: d/dx of every Hessian column, [N, in] per (k, i, j).
            var third = new Tensor<T>[outputDim];
            for (int k = 0; k < outputDim; k++)
            {
                var planes = new Tensor<T>[inputDim];
                for (int i = 0; i < inputDim; i++)
                {
                    var rows = new Tensor<T>[inputDim];
                    for (int j = 0; j < inputDim; j++)
                    {
                        rows[j] = InputGradient(tape, engine, Column(engine, secondColumns[k][i], j), x);
                    }

                    planes[i] = engine.TensorStack(rows, 1);
                }

                third[k] = engine.TensorStack(planes, 1);
            }

            derivatives.ThirdDerivatives = engine.TensorStack(third, 1);
            return outputs;
        }

        /// <summary>
        /// Returns the [N] column <paramref name="index"/> of a [N, M] tensor.
        /// </summary>
        internal static Tensor<T> Column(IEngine engine, Tensor<T> matrix, int index)
        {
            int n = matrix.Shape[0];
            return engine.Reshape(engine.TensorSlice(matrix, new[] { 0, index }, new[] { n, 1 }), new[] { n });
        }

        private static Tensor<T> InputGradient(GradientTape<T> tape, IEngine engine, Tensor<T> column, Tensor<T> x)
        {
            var total = engine.ReduceSum(column, new[] { 0 }, keepDims: false);
            var gradients = tape.ComputeGradients(total, new[] { x }, createGraph: true);
            return gradients.TryGetValue(x, out var gradient)
                ? gradient
                : new Tensor<T>(x.Shape.ToArray());
        }
    }
}

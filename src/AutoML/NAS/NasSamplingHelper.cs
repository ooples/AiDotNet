using System;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.AutoML.NAS
{
    internal static class NasSamplingHelper
    {
        // Gumbel(0, 1) noise is -log(-log(u)) with u ~ U(0, 1) (Gumbel 1954; Jang et al. 2017, Eq. 1). u is only
        // kept off the endpoints 0 and 1 where the transform is infinite; master clipped u to [0.005, 0.995],
        // which truncated the noise to [-1.67, 5.30] and biased the categorical sample toward the argmax.
        private const double GumbelMinU = 1e-10;
        private const double GumbelRangeU = 1.0 - 2e-10;

        internal static Vector<T> Softmax<T>(Vector<T> logits, INumericOperations<T> ops)
        {
            if (logits.Length == 0)
            {
                return new Vector<T>(0);
            }

            // Max-shifted softmax: the shift makes the denominator >= 1, so master's 1e-12 floor never applied.
            return AiDotNetEngine.Current.Softmax(logits);
        }

        internal static Vector<T> SoftmaxWithTemperature<T>(Vector<T> logits, T temperature, INumericOperations<T> ops)
        {
            if (logits.Length == 0)
            {
                return new Vector<T>(0);
            }

            var engine = AiDotNetEngine.Current;
            return engine.Softmax(engine.Divide(logits, temperature));
        }

        internal static Matrix<T> SoftmaxRowsWithTemperature<T>(Matrix<T> logits, T temperature, INumericOperations<T> ops)
        {
            if (logits.Rows == 0 || logits.Columns == 0)
            {
                return new Matrix<T>(logits.Rows, logits.Columns);
            }

            var engine = AiDotNetEngine.Current;
            return engine.TensorSoftmax(engine.TensorDivideScalar(Tensor<T>.FromMatrix(logits), temperature), 1).ToMatrix();
        }

        internal static Matrix<T> SoftmaxRows<T>(Matrix<T> logits, INumericOperations<T> ops)
        {
            if (logits.Rows == 0 || logits.Columns == 0)
            {
                return new Matrix<T>(logits.Rows, logits.Columns);
            }

            return AiDotNetEngine.Current.TensorSoftmax(Tensor<T>.FromMatrix(logits), 1).ToMatrix();
        }

        /// <summary>
        /// Argmax of every row of a [rows, cols] tensor, ties going to the lowest column like a strict
        /// greater-than scan. The tie-break is explicit (the maxima are ranked by reversed column index) so it
        /// does not depend on a backend's argmax tie order.
        /// </summary>
        internal static int[] ArgMaxRows<T>(Tensor<T> values, out T[] rowMax)
        {
            var engine = AiDotNetEngine.Current;
            int rows = values.Shape[0];
            int cols = values.Shape[1];
            var shape = new[] { rows, cols };
            var max = engine.ReduceMax(values, new[] { 1 }, true);
            var isMax = engine.TensorEquals(values, engine.TensorBroadcastTo(max, shape));
            var columnIndex = engine.TensorArange<T>(0, cols, 1);
            var reversedRank = engine.TensorAddScalar(engine.TensorNegate(columnIndex),
                MathHelper.GetNumericOperations<T>().FromDouble(cols));
            var key = engine.TensorMultiply(isMax, engine.TensorBroadcastTo(reversedRank.Reshape(new[] { 1, cols }), shape));
            rowMax = max.ToArray();
            return engine.TensorArgMax(key, 1).ToArray();
        }

        /// <summary>
        /// First-index row argmax of a matrix as a [rows] index tensor (see <see cref="ArgMaxRows{T}(Tensor{T}, out T[])"/>).
        /// </summary>
        internal static Tensor<int> ArgMaxRowsTensor<T>(Matrix<T> values)
        {
            int[] indices = ArgMaxRows(Tensor<T>.FromMatrix(values), out _);
            return Tensor<int>.FromVector(Vector<int>.FromArray(indices));
        }

        internal static int[] ArgMaxRows<T>(Matrix<T> values, out T[] rowMax)
        {
            return ArgMaxRows(Tensor<T>.FromMatrix(values), out rowMax);
        }

        internal static int ArgMax<T>(Vector<T> values)
        {
            return ArgMaxRows(values.AsTensor().Reshape(new[] { 1, values.Length }), out _)[0];
        }

        internal static Vector<T> GumbelSoftmax<T>(Vector<T> logits, T temperature, INumericOperations<T> ops, Random random, bool hard = false)
        {
            if (logits.Length == 0)
            {
                return new Vector<T>(0);
            }

            var gumbelLogits = new Vector<T>(logits.Length);
            for (int i = 0; i < logits.Length; i++)
            {
                double u = random.NextDouble() * GumbelRangeU + GumbelMinU;
                T gumbel = ops.FromDouble(-Math.Log(-Math.Log(u)));
                gumbelLogits[i] = ops.Divide(ops.Add(logits[i], gumbel), temperature);
            }

            var probabilities = Softmax(gumbelLogits, ops);

            if (!hard)
            {
                return probabilities;
            }

            int maxIdx = 0;
            T maxVal = probabilities[0];
            for (int i = 1; i < probabilities.Length; i++)
            {
                if (ops.GreaterThan(probabilities[i], maxVal))
                {
                    maxVal = probabilities[i];
                    maxIdx = i;
                }
            }

            var oneHot = new Vector<T>(probabilities.Length);
            for (int i = 0; i < oneHot.Length; i++)
            {
                oneHot[i] = i == maxIdx ? ops.One : ops.Zero;
            }

            return oneHot;
        }

        internal static Matrix<T> GumbelSoftmaxRows<T>(Matrix<T> logits, T temperature, INumericOperations<T> ops, Random random, bool hard = false)
        {
            var result = new Matrix<T>(logits.Rows, logits.Columns);

            if (logits.Rows == 0 || logits.Columns == 0)
            {
                return result;
            }

            for (int row = 0; row < logits.Rows; row++)
            {
                // Rows are drawn in order so the caller's Random stream matches the per-row scalar sampler.
                result.SetRow(row, GumbelSoftmax(logits.GetRow(row), temperature, ops, random, hard));
            }

            return result;
        }
    }
}

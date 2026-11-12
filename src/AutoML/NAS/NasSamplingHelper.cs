using System;
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

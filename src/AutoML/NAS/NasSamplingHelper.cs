using System;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.AutoML.NAS
{
    internal static class NasSamplingHelper
    {
        private const double SoftmaxEpsilon = 1e-12;
        private const double GumbelMinU = 0.005;
        private const double GumbelRangeU = 0.99;

        internal static Vector<T> Softmax<T>(Vector<T> logits, INumericOperations<T> ops)
        {
            if (logits.Length == 0)
            {
                return new Vector<T>(0);
            }

            T maxVal = logits[0];
            for (int i = 1; i < logits.Length; i++)
            {
                if (ops.GreaterThan(logits[i], maxVal))
                {
                    maxVal = logits[i];
                }
            }

            var expValues = new Vector<T>(logits.Length);
            T sumExp = ops.Zero;
            for (int i = 0; i < logits.Length; i++)
            {
                expValues[i] = ops.Exp(ops.Subtract(logits[i], maxVal));
                sumExp = ops.Add(sumExp, expValues[i]);
            }

            T eps = ops.FromDouble(SoftmaxEpsilon);
            if (ops.LessThan(sumExp, eps))
            {
                sumExp = eps;
            }

            var result = new Vector<T>(logits.Length);
            for (int i = 0; i < logits.Length; i++)
            {
                result[i] = ops.Divide(expValues[i], sumExp);
            }

            return result;
        }

        internal static Vector<T> SoftmaxWithTemperature<T>(Vector<T> logits, T temperature, INumericOperations<T> ops)
        {
            if (logits.Length == 0)
            {
                return new Vector<T>(0);
            }

            var scaled = new Vector<T>(logits.Length);
            for (int i = 0; i < logits.Length; i++)
            {
                scaled[i] = ops.Divide(logits[i], temperature);
            }

            return Softmax(scaled, ops);
        }

        internal static Matrix<T> SoftmaxRowsWithTemperature<T>(Matrix<T> logits, T temperature, INumericOperations<T> ops)
        {
            var result = new Matrix<T>(logits.Rows, logits.Columns);

            if (logits.Rows == 0 || logits.Columns == 0)
            {
                return result;
            }

            for (int row = 0; row < logits.Rows; row++)
            {
                T maxVal = ops.Divide(logits[row, 0], temperature);
                for (int col = 1; col < logits.Columns; col++)
                {
                    T scaled = ops.Divide(logits[row, col], temperature);
                    if (ops.GreaterThan(scaled, maxVal))
                    {
                        maxVal = scaled;
                    }
                }

                T sumExp = ops.Zero;
                var expValues = new T[logits.Columns];
                for (int col = 0; col < logits.Columns; col++)
                {
                    T scaled = ops.Divide(logits[row, col], temperature);
                    expValues[col] = ops.Exp(ops.Subtract(scaled, maxVal));
                    sumExp = ops.Add(sumExp, expValues[col]);
                }

                T eps = ops.FromDouble(SoftmaxEpsilon);
                if (ops.LessThan(sumExp, eps))
                {
                    sumExp = eps;
                }

                for (int col = 0; col < logits.Columns; col++)
                {
                    result[row, col] = ops.Divide(expValues[col], sumExp);
                }
            }

            return result;
        }

        internal static Matrix<T> SoftmaxRows<T>(Matrix<T> logits, INumericOperations<T> ops)
        {
            var result = new Matrix<T>(logits.Rows, logits.Columns);

            if (logits.Rows == 0 || logits.Columns == 0)
            {
                return result;
            }

            for (int row = 0; row < logits.Rows; row++)
            {
                T maxVal = logits[row, 0];
                for (int col = 1; col < logits.Columns; col++)
                {
                    if (ops.GreaterThan(logits[row, col], maxVal))
                    {
                        maxVal = logits[row, col];
                    }
                }

                T sumExp = ops.Zero;
                var expValues = new T[logits.Columns];
                for (int col = 0; col < logits.Columns; col++)
                {
                    expValues[col] = ops.Exp(ops.Subtract(logits[row, col], maxVal));
                    sumExp = ops.Add(sumExp, expValues[col]);
                }

                T eps = ops.FromDouble(SoftmaxEpsilon);
                if (ops.LessThan(sumExp, eps))
                {
                    sumExp = eps;
                }

                for (int col = 0; col < logits.Columns; col++)
                {
                    result[row, col] = ops.Divide(expValues[col], sumExp);
                }
            }

            return result;
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
                var rowLogits = new Vector<T>(logits.Columns);
                for (int col = 0; col < logits.Columns; col++)
                {
                    rowLogits[col] = logits[row, col];
                }

                var rowResult = GumbelSoftmax(rowLogits, temperature, ops, random, hard);
                for (int col = 0; col < logits.Columns; col++)
                {
                    result[row, col] = rowResult[col];
                }
            }

            return result;
        }
    }
}

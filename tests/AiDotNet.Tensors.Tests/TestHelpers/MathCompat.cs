// Math compatibility layer for .NET Framework 4.7.1
// Provides Log2, Cbrt, Acosh, Asinh, Atanh that don't exist in older frameworks

using System;
using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.Tests.TestHelpers
{
    /// <summary>
    /// Provides math functions compatible with all target frameworks.
    /// Use these instead of Math.Log2/Math.Cbrt/Math.Acosh etc which don't exist in .NET Framework.
    /// </summary>
    public static class MathCompat
    {
        private const double Log2Constant = 0.6931471805599453; // Math.Log(2)

        #region Log2

        /// <summary>
        /// Computes the base-2 logarithm of a double value.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Log2(double x)
        {
#if NET5_0_OR_GREATER
            return Math.Log2(x);
#else
            return Math.Log(x) / Log2Constant;
#endif
        }

        /// <summary>
        /// Computes the base-2 logarithm of a float value.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Log2(float x)
        {
#if NET5_0_OR_GREATER
            return MathF.Log2(x);
#else
            return (float)(Math.Log(x) / Log2Constant);
#endif
        }

        #endregion

        #region Cbrt

        /// <summary>
        /// Computes the cube root of a double value.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Cbrt(double x)
        {
#if NET5_0_OR_GREATER
            return Math.Cbrt(x);
#else
            if (x >= 0)
            {
                return Math.Pow(x, 1.0 / 3.0);
            }
            else
            {
                return -Math.Pow(-x, 1.0 / 3.0);
            }
#endif
        }

        /// <summary>
        /// Computes the cube root of a float value.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Cbrt(float x)
        {
#if NET5_0_OR_GREATER
            return MathF.Cbrt(x);
#else
            if (x >= 0)
            {
                return (float)Math.Pow(x, 1.0 / 3.0);
            }
            else
            {
                return (float)(-Math.Pow(-x, 1.0 / 3.0));
            }
#endif
        }

        #endregion

        #region Acosh

        /// <summary>
        /// Computes the inverse hyperbolic cosine of a double value.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Acosh(double x)
        {
#if NET5_0_OR_GREATER
            return Math.Acosh(x);
#else
            // acosh(x) = ln(x + sqrt(x^2 - 1))
            return Math.Log(x + Math.Sqrt(x * x - 1));
#endif
        }

        /// <summary>
        /// Computes the inverse hyperbolic cosine of a float value.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Acosh(float x)
        {
#if NET5_0_OR_GREATER
            return MathF.Acosh(x);
#else
            return (float)Math.Log(x + Math.Sqrt(x * x - 1));
#endif
        }

        #endregion

        #region Asinh

        /// <summary>
        /// Computes the inverse hyperbolic sine of a double value.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Asinh(double x)
        {
#if NET5_0_OR_GREATER
            return Math.Asinh(x);
#else
            // asinh(x) = ln(x + sqrt(x^2 + 1))
            return Math.Log(x + Math.Sqrt(x * x + 1));
#endif
        }

        /// <summary>
        /// Computes the inverse hyperbolic sine of a float value.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Asinh(float x)
        {
#if NET5_0_OR_GREATER
            return MathF.Asinh(x);
#else
            return (float)Math.Log(x + Math.Sqrt(x * x + 1));
#endif
        }

        #endregion

        #region Atanh

        /// <summary>
        /// Computes the inverse hyperbolic tangent of a double value.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Atanh(double x)
        {
#if NET5_0_OR_GREATER
            return Math.Atanh(x);
#else
            // atanh(x) = 0.5 * ln((1 + x) / (1 - x))
            return 0.5 * Math.Log((1 + x) / (1 - x));
#endif
        }

        /// <summary>
        /// Computes the inverse hyperbolic tangent of a float value.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Atanh(float x)
        {
#if NET5_0_OR_GREATER
            return MathF.Atanh(x);
#else
            return (float)(0.5 * Math.Log((1 + x) / (1 - x)));
#endif
        }

        #endregion
    }
}

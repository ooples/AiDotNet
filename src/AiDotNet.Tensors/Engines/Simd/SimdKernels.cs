using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics.Arm;

namespace AiDotNet.Tensors.Engines.Simd
{
    /// <summary>
    /// SIMD-optimized kernels for common operations.
    /// Provides hardware-accelerated implementations using AVX2, SSE, and ARM NEON.
    /// </summary>
    public static class SimdKernels
    {
        /// <summary>
        /// SIMD-optimized vector addition for float arrays
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void VectorAdd(float* a, float* b, float* result, int length)
        {
            int i = 0;

            // AVX2 path (8 floats at a time)
            if (Avx2.IsSupported && length >= 8)
            {
                int simdLength = length & ~7; // Round down to multiple of 8
                for (; i < simdLength; i += 8)
                {
                    var va = Avx.LoadVector256(a + i);
                    var vb = Avx.LoadVector256(b + i);
                    var vr = Avx.Add(va, vb);
                    Avx.Store(result + i, vr);
                }
            }
            // SSE path (4 floats at a time)
            else if (Sse.IsSupported && length >= 4)
            {
                int simdLength = length & ~3; // Round down to multiple of 4
                for (; i < simdLength; i += 4)
                {
                    var va = Sse.LoadVector128(a + i);
                    var vb = Sse.LoadVector128(b + i);
                    var vr = Sse.Add(va, vb);
                    Sse.Store(result + i, vr);
                }
            }
            // NEON path (4 floats at a time)
            else if (AdvSimd.IsSupported && length >= 4)
            {
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    var va = AdvSimd.LoadVector128(a + i);
                    var vb = AdvSimd.LoadVector128(b + i);
                    var vr = AdvSimd.Add(va, vb);
                    AdvSimd.Store(result + i, vr);
                }
            }

            // Scalar fallback for remaining elements
            for (; i < length; i++)
            {
                result[i] = a[i] + b[i];
            }
        }

        /// <summary>
        /// SIMD-optimized vector multiplication
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void VectorMultiply(float* a, float* b, float* result, int length)
        {
            int i = 0;

            if (Avx2.IsSupported && length >= 8)
            {
                int simdLength = length & ~7;
                for (; i < simdLength; i += 8)
                {
                    var va = Avx.LoadVector256(a + i);
                    var vb = Avx.LoadVector256(b + i);
                    var vr = Avx.Multiply(va, vb);
                    Avx.Store(result + i, vr);
                }
            }
            else if (Sse.IsSupported && length >= 4)
            {
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    var va = Sse.LoadVector128(a + i);
                    var vb = Sse.LoadVector128(b + i);
                    var vr = Sse.Multiply(va, vb);
                    Sse.Store(result + i, vr);
                }
            }
            else if (AdvSimd.IsSupported && length >= 4)
            {
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    var va = AdvSimd.LoadVector128(a + i);
                    var vb = AdvSimd.LoadVector128(b + i);
                    var vr = AdvSimd.Multiply(va, vb);
                    AdvSimd.Store(result + i, vr);
                }
            }

            for (; i < length; i++)
            {
                result[i] = a[i] * b[i];
            }
        }

        /// <summary>
        /// SIMD-optimized dot product
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe float DotProduct(float* a, float* b, int length)
        {
            float sum = 0.0f;
            int i = 0;

            if (Avx2.IsSupported && length >= 8)
            {
                var vsum = Vector256<float>.Zero;
                int simdLength = length & ~7;

                for (; i < simdLength; i += 8)
                {
                    var va = Avx.LoadVector256(a + i);
                    var vb = Avx.LoadVector256(b + i);
                    vsum = Fma.IsSupported
                        ? Fma.MultiplyAdd(va, vb, vsum)
                        : Avx.Add(vsum, Avx.Multiply(va, vb));
                }

                // Horizontal sum of vector
                var high = Avx.ExtractVector128(vsum, 1);
                var low = Avx.GetLowerHalf(vsum);
                var sum128 = Sse.Add(high, low);

                // Further reduce 4 floats to 1
                var shuf = Sse.Shuffle(sum128, sum128, 0b_11_10_11_10);
                sum128 = Sse.Add(sum128, shuf);
                shuf = Sse.Shuffle(sum128, sum128, 0b_01_01_01_01);
                sum128 = Sse.Add(sum128, shuf);
                sum = Sse.ConvertToSingle(sum128);
            }
            else if (Sse.IsSupported && length >= 4)
            {
                var vsum = Vector128<float>.Zero;
                int simdLength = length & ~3;

                for (; i < simdLength; i += 4)
                {
                    var va = Sse.LoadVector128(a + i);
                    var vb = Sse.LoadVector128(b + i);
                    vsum = Sse.Add(vsum, Sse.Multiply(va, vb));
                }

                // Horizontal sum
                var shuf = Sse.Shuffle(vsum, vsum, 0b_11_10_11_10);
                vsum = Sse.Add(vsum, shuf);
                shuf = Sse.Shuffle(vsum, vsum, 0b_01_01_01_01);
                vsum = Sse.Add(vsum, shuf);
                sum = Sse.ConvertToSingle(vsum);
            }
            else if (AdvSimd.IsSupported && length >= 4)
            {
                var vsum = Vector128<float>.Zero;
                int simdLength = length & ~3;

                for (; i < simdLength; i += 4)
                {
                    var va = AdvSimd.LoadVector128(a + i);
                    var vb = AdvSimd.LoadVector128(b + i);
                    vsum = AdvSimd.Add(vsum, AdvSimd.Multiply(va, vb));
                }

                // Horizontal sum for ARM
                sum = AdvSimd.Arm64.AddAcross(vsum).ToScalar();
            }

            // Scalar remainder
            for (; i < length; i++)
            {
                sum += a[i] * b[i];
            }

            return sum;
        }

        /// <summary>
        /// SIMD-optimized scalar multiply-add (result = a + b * scalar)
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void ScalarMultiplyAdd(float* a, float* b, float scalar, float* result, int length)
        {
            int i = 0;

            if (Avx2.IsSupported && length >= 8)
            {
                var vscalar = Vector256.Create(scalar);
                int simdLength = length & ~7;

                for (; i < simdLength; i += 8)
                {
                    var va = Avx.LoadVector256(a + i);
                    var vb = Avx.LoadVector256(b + i);
                    var vr = Fma.IsSupported
                        ? Fma.MultiplyAdd(vb, vscalar, va)
                        : Avx.Add(va, Avx.Multiply(vb, vscalar));
                    Avx.Store(result + i, vr);
                }
            }
            else if (Sse.IsSupported && length >= 4)
            {
                var vscalar = Vector128.Create(scalar);
                int simdLength = length & ~3;

                for (; i < simdLength; i += 4)
                {
                    var va = Sse.LoadVector128(a + i);
                    var vb = Sse.LoadVector128(b + i);
                    var vr = Sse.Add(va, Sse.Multiply(vb, vscalar));
                    Sse.Store(result + i, vr);
                }
            }
            else if (AdvSimd.IsSupported && length >= 4)
            {
                var vscalar = Vector128.Create(scalar);
                int simdLength = length & ~3;

                for (; i < simdLength; i += 4)
                {
                    var va = AdvSimd.LoadVector128(a + i);
                    var vb = AdvSimd.LoadVector128(b + i);
                    var vr = AdvSimd.Add(va, AdvSimd.Multiply(vb, vscalar));
                    AdvSimd.Store(result + i, vr);
                }
            }

            for (; i < length; i++)
            {
                result[i] = a[i] + b[i] * scalar;
            }
        }

        /// <summary>
        /// SIMD-optimized ReLU activation
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void ReLU(float* input, float* output, int length)
        {
            int i = 0;

            if (Avx2.IsSupported && length >= 8)
            {
                var vzero = Vector256<float>.Zero;
                int simdLength = length & ~7;

                for (; i < simdLength; i += 8)
                {
                    var v = Avx.LoadVector256(input + i);
                    var vr = Avx.Max(v, vzero);
                    Avx.Store(output + i, vr);
                }
            }
            else if (Sse.IsSupported && length >= 4)
            {
                var vzero = Vector128<float>.Zero;
                int simdLength = length & ~3;

                for (; i < simdLength; i += 4)
                {
                    var v = Sse.LoadVector128(input + i);
                    var vr = Sse.Max(v, vzero);
                    Sse.Store(output + i, vr);
                }
            }
            else if (AdvSimd.IsSupported && length >= 4)
            {
                var vzero = Vector128<float>.Zero;
                int simdLength = length & ~3;

                for (; i < simdLength; i += 4)
                {
                    var v = AdvSimd.LoadVector128(input + i);
                    var vr = AdvSimd.Max(v, vzero);
                    AdvSimd.Store(output + i, vr);
                }
            }

            for (; i < length; i++)
            {
                output[i] = Math.Max(0.0f, input[i]);
            }
        }

        /// <summary>
        /// SIMD-optimized element-wise exponential
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void Exp(float* input, float* output, int length)
        {
            // Note: True SIMD exp requires approximation algorithms
            // This is a scalar fallback - can be optimized with SVML or custom approximations
            for (int i = 0; i < length; i++)
            {
                output[i] = MathF.Exp(input[i]);
            }
        }

        /// <summary>
        /// SIMD-optimized sum reduction
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe float Sum(float* data, int length)
        {
            float sum = 0.0f;
            int i = 0;

            if (Avx2.IsSupported && length >= 8)
            {
                var vsum = Vector256<float>.Zero;
                int simdLength = length & ~7;

                for (; i < simdLength; i += 8)
                {
                    var v = Avx.LoadVector256(data + i);
                    vsum = Avx.Add(vsum, v);
                }

                var high = Avx.ExtractVector128(vsum, 1);
                var low = Avx.GetLowerHalf(vsum);
                var sum128 = Sse.Add(high, low);

                var shuf = Sse.Shuffle(sum128, sum128, 0b_11_10_11_10);
                sum128 = Sse.Add(sum128, shuf);
                shuf = Sse.Shuffle(sum128, sum128, 0b_01_01_01_01);
                sum128 = Sse.Add(sum128, shuf);
                sum = Sse.ConvertToSingle(sum128);
            }
            else if (Sse.IsSupported && length >= 4)
            {
                var vsum = Vector128<float>.Zero;
                int simdLength = length & ~3;

                for (; i < simdLength; i += 4)
                {
                    var v = Sse.LoadVector128(data + i);
                    vsum = Sse.Add(vsum, v);
                }

                var shuf = Sse.Shuffle(vsum, vsum, 0b_11_10_11_10);
                vsum = Sse.Add(vsum, shuf);
                shuf = Sse.Shuffle(vsum, vsum, 0b_01_01_01_01);
                vsum = Sse.Add(vsum, shuf);
                sum = Sse.ConvertToSingle(vsum);
            }
            else if (AdvSimd.IsSupported && length >= 4)
            {
                var vsum = Vector128<float>.Zero;
                int simdLength = length & ~3;

                for (; i < simdLength; i += 4)
                {
                    var v = AdvSimd.LoadVector128(data + i);
                    vsum = AdvSimd.Add(vsum, v);
                }

                sum = AdvSimd.Arm64.AddAcross(vsum).ToScalar();
            }

            for (; i < length; i++)
            {
                sum += data[i];
            }

            return sum;
        }
    }
}

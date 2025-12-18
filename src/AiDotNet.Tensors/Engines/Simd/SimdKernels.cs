using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.Simd
{
    /// <summary>
    /// SIMD-optimized kernels for common operations.
    /// Provides hardware-accelerated implementations using AVX/SSE and ARM NEON.
    /// Falls back to scalar operations when intrinsics are unavailable.
    /// </summary>
    public static class SimdKernels
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void VectorAdd(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result)
        {
            if (a.Length != b.Length || a.Length != result.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            int length = result.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 8)
            {
                int simdLength = length & ~7;
                for (; i < simdLength; i += 8)
                {
                    var va = ReadVector256(a, i);
                    var vb = ReadVector256(b, i);
                    WriteVector256(result, i, Avx.Add(va, vb));
                }
            }
            else if (Sse.IsSupported && length >= 4)
            {
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    var va = ReadVector128(a, i);
                    var vb = ReadVector128(b, i);
                    WriteVector128(result, i, Sse.Add(va, vb));
                }
            }
            else if (AdvSimd.IsSupported && length >= 4)
            {
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    var va = ReadVector128(a, i);
                    var vb = ReadVector128(b, i);
                    WriteVector128(result, i, AdvSimd.Add(va, vb));
                }
            }
#endif

            for (; i < length; i++)
            {
                result[i] = a[i] + b[i];
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void VectorMultiply(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result)
        {
            if (a.Length != b.Length || a.Length != result.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            int length = result.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 8)
            {
                int simdLength = length & ~7;
                for (; i < simdLength; i += 8)
                {
                    var va = ReadVector256(a, i);
                    var vb = ReadVector256(b, i);
                    WriteVector256(result, i, Avx.Multiply(va, vb));
                }
            }
            else if (Sse.IsSupported && length >= 4)
            {
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    var va = ReadVector128(a, i);
                    var vb = ReadVector128(b, i);
                    WriteVector128(result, i, Sse.Multiply(va, vb));
                }
            }
            else if (AdvSimd.IsSupported && length >= 4)
            {
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    var va = ReadVector128(a, i);
                    var vb = ReadVector128(b, i);
                    WriteVector128(result, i, AdvSimd.Multiply(va, vb));
                }
            }
#endif

            for (; i < length; i++)
            {
                result[i] = a[i] * b[i];
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float DotProduct(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
        {
            if (a.Length != b.Length)
            {
                throw new ArgumentException("Input spans must have the same length.");
            }

            int length = a.Length;
            int i = 0;
            float sum = 0f;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 8)
            {
                var vsum = Vector256<float>.Zero;
                int simdLength = length & ~7;
                for (; i < simdLength; i += 8)
                {
                    var va = ReadVector256(a, i);
                    var vb = ReadVector256(b, i);
                    vsum = Fma.IsSupported ? Fma.MultiplyAdd(va, vb, vsum) : Avx.Add(vsum, Avx.Multiply(va, vb));
                }

                sum += HorizontalSum(vsum);
            }
            else if (Sse.IsSupported && length >= 4)
            {
                var vsum = Vector128<float>.Zero;
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    var va = ReadVector128(a, i);
                    var vb = ReadVector128(b, i);
                    vsum = Sse.Add(vsum, Sse.Multiply(va, vb));
                }

                sum += HorizontalSum(vsum);
            }
            else if (AdvSimd.IsSupported && length >= 4)
            {
                var vsum = Vector128<float>.Zero;
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    var va = ReadVector128(a, i);
                    var vb = ReadVector128(b, i);
                    vsum = AdvSimd.Add(vsum, AdvSimd.Multiply(va, vb));
                }

                sum += HorizontalSum(vsum);
            }
#endif

            for (; i < length; i++)
            {
                sum += a[i] * b[i];
            }

            return sum;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ScalarMultiplyAdd(ReadOnlySpan<float> a, ReadOnlySpan<float> b, float scalar, Span<float> result)
        {
            if (a.Length != b.Length || a.Length != result.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            int length = result.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 8)
            {
                var vscalar = Vector256.Create(scalar);
                int simdLength = length & ~7;
                for (; i < simdLength; i += 8)
                {
                    var va = ReadVector256(a, i);
                    var vb = ReadVector256(b, i);
                    var vr = Fma.IsSupported ? Fma.MultiplyAdd(vb, vscalar, va) : Avx.Add(va, Avx.Multiply(vb, vscalar));
                    WriteVector256(result, i, vr);
                }
            }
            else if (Sse.IsSupported && length >= 4)
            {
                var vscalar = Vector128.Create(scalar);
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    var va = ReadVector128(a, i);
                    var vb = ReadVector128(b, i);
                    WriteVector128(result, i, Sse.Add(va, Sse.Multiply(vb, vscalar)));
                }
            }
            else if (AdvSimd.IsSupported && length >= 4)
            {
                var vscalar = Vector128.Create(scalar);
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    var va = ReadVector128(a, i);
                    var vb = ReadVector128(b, i);
                    WriteVector128(result, i, AdvSimd.Add(va, AdvSimd.Multiply(vb, vscalar)));
                }
            }
#endif

            for (; i < length; i++)
            {
                result[i] = a[i] + scalar * b[i];
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ReLU(ReadOnlySpan<float> input, Span<float> output)
        {
            if (input.Length != output.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            int length = output.Length;
            int i = 0;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 8)
            {
                var vzero = Vector256<float>.Zero;
                int simdLength = length & ~7;
                for (; i < simdLength; i += 8)
                {
                    WriteVector256(output, i, Avx.Max(ReadVector256(input, i), vzero));
                }
            }
            else if (Sse.IsSupported && length >= 4)
            {
                var vzero = Vector128<float>.Zero;
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    WriteVector128(output, i, Sse.Max(ReadVector128(input, i), vzero));
                }
            }
            else if (AdvSimd.IsSupported && length >= 4)
            {
                var vzero = Vector128<float>.Zero;
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    WriteVector128(output, i, AdvSimd.Max(ReadVector128(input, i), vzero));
                }
            }
#endif

            for (; i < length; i++)
            {
                output[i] = input[i] > 0f ? input[i] : 0f;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Exp(ReadOnlySpan<float> input, Span<float> output)
        {
            if (input.Length != output.Length)
            {
                throw new ArgumentException("Input and output spans must have the same length.");
            }

            for (int i = 0; i < input.Length; i++)
            {
#if NET5_0_OR_GREATER
                output[i] = MathF.Exp(input[i]);
#else
                output[i] = (float)Math.Exp(input[i]);
#endif
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Sum(ReadOnlySpan<float> data)
        {
            int length = data.Length;
            int i = 0;
            float sum = 0f;

#if NET5_0_OR_GREATER
            if (Avx.IsSupported && length >= 8)
            {
                var vsum = Vector256<float>.Zero;
                int simdLength = length & ~7;
                for (; i < simdLength; i += 8)
                {
                    vsum = Avx.Add(vsum, ReadVector256(data, i));
                }

                sum += HorizontalSum(vsum);
            }
            else if (Sse.IsSupported && length >= 4)
            {
                var vsum = Vector128<float>.Zero;
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    vsum = Sse.Add(vsum, ReadVector128(data, i));
                }

                sum += HorizontalSum(vsum);
            }
            else if (AdvSimd.IsSupported && length >= 4)
            {
                var vsum = Vector128<float>.Zero;
                int simdLength = length & ~3;
                for (; i < simdLength; i += 4)
                {
                    vsum = AdvSimd.Add(vsum, ReadVector128(data, i));
                }

                sum += HorizontalSum(vsum);
            }
#endif

            for (; i < length; i++)
            {
                sum += data[i];
            }

            return sum;
        }

#if NET5_0_OR_GREATER
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<float> ReadVector256(ReadOnlySpan<float> data, int offset)
        {
            ref float start = ref MemoryMarshal.GetReference(data);
            ref float element = ref Unsafe.Add(ref start, offset);
            return Unsafe.ReadUnaligned<Vector256<float>>(ref Unsafe.As<float, byte>(ref element));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void WriteVector256(Span<float> data, int offset, Vector256<float> value)
        {
            ref float start = ref MemoryMarshal.GetReference(data);
            ref float element = ref Unsafe.Add(ref start, offset);
            Unsafe.WriteUnaligned(ref Unsafe.As<float, byte>(ref element), value);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector128<float> ReadVector128(ReadOnlySpan<float> data, int offset)
        {
            ref float start = ref MemoryMarshal.GetReference(data);
            ref float element = ref Unsafe.Add(ref start, offset);
            return Unsafe.ReadUnaligned<Vector128<float>>(ref Unsafe.As<float, byte>(ref element));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void WriteVector128(Span<float> data, int offset, Vector128<float> value)
        {
            ref float start = ref MemoryMarshal.GetReference(data);
            ref float element = ref Unsafe.Add(ref start, offset);
            Unsafe.WriteUnaligned(ref Unsafe.As<float, byte>(ref element), value);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float HorizontalSum(Vector256<float> v)
        {
            Span<float> tmp = stackalloc float[8];
            Unsafe.WriteUnaligned(ref Unsafe.As<float, byte>(ref MemoryMarshal.GetReference(tmp)), v);
            float sum = 0f;
            for (int i = 0; i < tmp.Length; i++)
            {
                sum += tmp[i];
            }

            return sum;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float HorizontalSum(Vector128<float> v)
        {
            Span<float> tmp = stackalloc float[4];
            Unsafe.WriteUnaligned(ref Unsafe.As<float, byte>(ref MemoryMarshal.GetReference(tmp)), v);
            return tmp[0] + tmp[1] + tmp[2] + tmp[3];
        }
#endif
    }
}

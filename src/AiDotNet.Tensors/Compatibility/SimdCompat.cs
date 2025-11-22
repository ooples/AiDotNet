// Copyright (c) AiDotNet. All rights reserved.
// SIMD compatibility layer for .NET Framework 4.6.2+ which lacks System.Runtime.Intrinsics
// This polyfill allows the same code to compile across net462, net471, and net8.0 without conditionals

#if !NET5_0_OR_GREATER

using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace System.Runtime.Intrinsics
{
    /// <summary>
    /// Polyfill for Vector128&lt;T&gt; on .NET Framework.
    /// Uses scalar fallback since true SIMD intrinsics aren't available in net462/net471.
    /// </summary>
    public readonly struct Vector128<T> where T : struct
    {
        private readonly T _e0, _e1;

        public static int Count
        {
            get
            {
                // 128 bits / (sizeof(T) * 8 bits per byte)
                if (typeof(T) == typeof(double)) return 2;
                if (typeof(T) == typeof(float)) return 4;
                if (typeof(T) == typeof(long)) return 2;
                if (typeof(T) == typeof(ulong)) return 2;
                if (typeof(T) == typeof(int)) return 4;
                if (typeof(T) == typeof(uint)) return 4;
                if (typeof(T) == typeof(short)) return 8;
                if (typeof(T) == typeof(ushort)) return 8;
                if (typeof(T) == typeof(byte)) return 16;
                if (typeof(T) == typeof(sbyte)) return 16;
                return 1;
            }
        }

        internal Vector128(T e0, T e1)
        {
            _e0 = e0;
            _e1 = e1;
        }

        public void CopyTo(Span<T> destination)
        {
            if (destination.Length < Count)
                throw new ArgumentException("Destination too small");

            // Fill with repeated values (scalar fallback behavior)
            for (int i = 0; i < Count; i++)
            {
                destination[i] = i == 0 ? _e0 : _e1;
            }
        }
    }

    /// <summary>
    /// Polyfill for Vector256&lt;T&gt; on .NET Framework.
    /// </summary>
    public readonly struct Vector256<T> where T : struct
    {
        private readonly T _e0, _e1, _e2, _e3;

        public static int Count
        {
            get
            {
                // 256 bits / (sizeof(T) * 8 bits per byte)
                if (typeof(T) == typeof(double)) return 4;
                if (typeof(T) == typeof(float)) return 8;
                if (typeof(T) == typeof(long)) return 4;
                if (typeof(T) == typeof(ulong)) return 4;
                if (typeof(T) == typeof(int)) return 8;
                if (typeof(T) == typeof(uint)) return 8;
                if (typeof(T) == typeof(short)) return 16;
                if (typeof(T) == typeof(ushort)) return 16;
                if (typeof(T) == typeof(byte)) return 32;
                if (typeof(T) == typeof(sbyte)) return 32;
                return 1;
            }
        }

        internal Vector256(T e0, T e1, T e2, T e3)
        {
            _e0 = e0;
            _e1 = e1;
            _e2 = e2;
            _e3 = e3;
        }

        public void CopyTo(Span<T> destination)
        {
            if (destination.Length < Count)
                throw new ArgumentException("Destination too small");

            for (int i = 0; i < Count; i++)
            {
                destination[i] = i < Count / 4 ? _e0 :
                                i < Count / 2 ? _e1 :
                                i < 3 * Count / 4 ? _e2 : _e3;
            }
        }
    }

    /// <summary>
    /// Polyfill for Vector512&lt;T&gt; on .NET Framework.
    /// </summary>
    public readonly struct Vector512<T> where T : struct
    {
        private readonly T _e0, _e1, _e2, _e3, _e4, _e5, _e6, _e7;

        public static int Count
        {
            get
            {
                // 512 bits / (sizeof(T) * 8 bits per byte)
                if (typeof(T) == typeof(double)) return 8;
                if (typeof(T) == typeof(float)) return 16;
                if (typeof(T) == typeof(long)) return 8;
                if (typeof(T) == typeof(ulong)) return 8;
                if (typeof(T) == typeof(int)) return 16;
                if (typeof(T) == typeof(uint)) return 16;
                if (typeof(T) == typeof(short)) return 32;
                if (typeof(T) == typeof(ushort)) return 32;
                if (typeof(T) == typeof(byte)) return 64;
                if (typeof(T) == typeof(sbyte)) return 64;
                return 1;
            }
        }

        internal Vector512(T e0, T e1, T e2, T e3, T e4, T e5, T e6, T e7)
        {
            _e0 = e0;
            _e1 = e1;
            _e2 = e2;
            _e3 = e3;
            _e4 = e4;
            _e5 = e5;
            _e6 = e6;
            _e7 = e7;
        }

        public void CopyTo(Span<T> destination)
        {
            if (destination.Length < Count)
                throw new ArgumentException("Destination too small");

            for (int i = 0; i < Count; i++)
            {
                destination[i] = i < Count / 8 ? _e0 :
                                i < 2 * Count / 8 ? _e1 :
                                i < 3 * Count / 8 ? _e2 :
                                i < 4 * Count / 8 ? _e3 :
                                i < 5 * Count / 8 ? _e4 :
                                i < 6 * Count / 8 ? _e5 :
                                i < 7 * Count / 8 ? _e6 : _e7;
            }
        }
    }

    /// <summary>
    /// Static helper class for Vector128 operations.
    /// </summary>
    public static class Vector128
    {
        /// <summary>
        /// Indicates whether hardware acceleration is available.
        /// Uses System.Numerics.Vector which IS available in .NET Framework 4.6+.
        /// </summary>
        public static bool IsHardwareAccelerated => Vector.IsHardwareAccelerated;

        public static Vector128<T> Create<T>(T value) where T : struct
        {
            return new Vector128<T>(value, value);
        }

        public static Vector128<T> Create<T>(T e0, T e1) where T : struct
        {
            return new Vector128<T>(e0, e1);
        }

        public static Vector128<double> Create(double e0, double e1)
        {
            return new Vector128<double>(e0, e1);
        }

        public static Vector128<float> Create(float e0, float e1, float e2, float e3)
        {
            // In scalar fallback, we just store first two values
            return new Vector128<float>(e0, e1);
        }

        public static T GetElement<T>(Vector128<T> vector, int index) where T : unmanaged
        {
            Span<T> temp = stackalloc T[Vector128<T>.Count];
            vector.CopyTo(temp);
            return temp[index];
        }
    }

    /// <summary>
    /// Static helper class for Vector256 operations.
    /// </summary>
    public static class Vector256
    {
        public static bool IsHardwareAccelerated => Vector.IsHardwareAccelerated;

        public static Vector256<T> Create<T>(T value) where T : struct
        {
            return new Vector256<T>(value, value, value, value);
        }

        public static Vector256<double> Create(double e0, double e1, double e2, double e3)
        {
            return new Vector256<double>(e0, e1, e2, e3);
        }

        public static Vector256<float> Create(float e0, float e1, float e2, float e3, float e4, float e5, float e6, float e7)
        {
            // Scalar fallback stores first 4 values
            return new Vector256<float>(e0, e1, e2, e3);
        }

        public static T GetElement<T>(Vector256<T> vector, int index) where T : unmanaged
        {
            Span<T> temp = stackalloc T[Vector256<T>.Count];
            vector.CopyTo(temp);
            return temp[index];
        }
    }

    /// <summary>
    /// Static helper class for Vector512 operations.
    /// </summary>
    public static class Vector512
    {
        public static bool IsHardwareAccelerated => Vector.IsHardwareAccelerated;

        public static Vector512<T> Create<T>(T value) where T : struct
        {
            return new Vector512<T>(value, value, value, value, value, value, value, value);
        }

        public static Vector512<double> Create(double e0, double e1, double e2, double e3, double e4, double e5, double e6, double e7)
        {
            return new Vector512<double>(e0, e1, e2, e3, e4, e5, e6, e7);
        }

        public static Vector512<float> Create(float e0, float e1, float e2, float e3, float e4, float e5, float e6, float e7,
                                              float e8, float e9, float e10, float e11, float e12, float e13, float e14, float e15)
        {
            // Scalar fallback stores first 8 values
            return new Vector512<float>(e0, e1, e2, e3, e4, e5, e6, e7);
        }

        public static T GetElement<T>(Vector512<T> vector, int index) where T : unmanaged
        {
            Span<T> temp = stackalloc T[Vector512<T>.Count];
            vector.CopyTo(temp);
            return temp[index];
        }
    }
}

#endif

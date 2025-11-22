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
    /// Properly stores all vector lanes for correct scalar fallback behavior.
    /// </summary>
    public readonly struct Vector128<T>
    {
        private readonly T[] _elements;

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

        internal Vector128(params T[] elements)
        {
            _elements = new T[Count];
            if (elements != null && elements.Length > 0)
            {
                int copyCount = Math.Min(elements.Length, Count);
                Array.Copy(elements, _elements, copyCount);
            }
        }

        public void CopyTo(Span<T> destination)
        {
            if (destination.Length < Count)
                throw new ArgumentException("Destination too small");

            if (_elements is null)
            {
                // Handle default(Vector128<T>) - zero-fill
                destination.Slice(0, Count).Clear();
                return;
            }

            _elements.AsSpan(0, Count).CopyTo(destination);
        }
    }

    /// <summary>
    /// Polyfill for Vector256&lt;T&gt; on .NET Framework.
    /// Properly stores all vector lanes for correct scalar fallback behavior.
    /// </summary>
    public readonly struct Vector256<T>
    {
        private readonly T[] _elements;

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

        internal Vector256(params T[] elements)
        {
            _elements = new T[Count];
            if (elements != null && elements.Length > 0)
            {
                int copyCount = Math.Min(elements.Length, Count);
                Array.Copy(elements, _elements, copyCount);
            }
        }

        public void CopyTo(Span<T> destination)
        {
            if (destination.Length < Count)
                throw new ArgumentException("Destination too small");

            if (_elements is null)
            {
                // Handle default(Vector256<T>) - zero-fill
                destination.Slice(0, Count).Clear();
                return;
            }

            _elements.AsSpan(0, Count).CopyTo(destination);
        }
    }

    /// <summary>
    /// Polyfill for Vector512&lt;T&gt; on .NET Framework.
    /// Properly stores all vector lanes for correct scalar fallback behavior.
    /// </summary>
    public readonly struct Vector512<T>
    {
        private readonly T[] _elements;

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

        internal Vector512(params T[] elements)
        {
            _elements = new T[Count];
            if (elements != null && elements.Length > 0)
            {
                int copyCount = Math.Min(elements.Length, Count);
                Array.Copy(elements, _elements, copyCount);
            }
        }

        public void CopyTo(Span<T> destination)
        {
            if (destination.Length < Count)
                throw new ArgumentException("Destination too small");

            if (_elements is null)
            {
                // Handle default(Vector512<T>) - zero-fill
                destination.Slice(0, Count).Clear();
                return;
            }

            _elements.AsSpan(0, Count).CopyTo(destination);
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

        public static Vector128<T> Create<T>(T value)
        {
            var elements = new T[Vector128<T>.Count];
            for (int i = 0; i < elements.Length; i++)
            {
                elements[i] = value;
            }
            return new Vector128<T>(elements);
        }

        public static Vector128<T> Create<T>(T e0, T e1)
        {
            return new Vector128<T>(new[] { e0, e1 });
        }

        public static Vector128<double> Create(double e0, double e1)
        {
            return new Vector128<double>(new[] { e0, e1 });
        }

        public static Vector128<float> Create(float e0, float e1, float e2, float e3)
        {
            return new Vector128<float>(new[] { e0, e1, e2, e3 });
        }

        public static T GetElement<T>(Vector128<T> vector, int index) where T : unmanaged
        {
            if (index < 0 || index >= Vector128<T>.Count)
                throw new ArgumentOutOfRangeException(nameof(index));

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

        public static Vector256<T> Create<T>(T value)
        {
            var elements = new T[Vector256<T>.Count];
            for (int i = 0; i < elements.Length; i++)
            {
                elements[i] = value;
            }
            return new Vector256<T>(elements);
        }

        public static Vector256<double> Create(double e0, double e1, double e2, double e3)
        {
            return new Vector256<double>(new[] { e0, e1, e2, e3 });
        }

        public static Vector256<float> Create(float e0, float e1, float e2, float e3, float e4, float e5, float e6, float e7)
        {
            return new Vector256<float>(new[] { e0, e1, e2, e3, e4, e5, e6, e7 });
        }

        public static T GetElement<T>(Vector256<T> vector, int index) where T : unmanaged
        {
            if (index < 0 || index >= Vector256<T>.Count)
                throw new ArgumentOutOfRangeException(nameof(index));

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

        public static Vector512<T> Create<T>(T value)
        {
            var elements = new T[Vector512<T>.Count];
            for (int i = 0; i < elements.Length; i++)
            {
                elements[i] = value;
            }
            return new Vector512<T>(elements);
        }

        public static Vector512<double> Create(double e0, double e1, double e2, double e3, double e4, double e5, double e6, double e7)
        {
            return new Vector512<double>(new[] { e0, e1, e2, e3, e4, e5, e6, e7 });
        }

        public static Vector512<float> Create(float e0, float e1, float e2, float e3, float e4, float e5, float e6, float e7,
                                              float e8, float e9, float e10, float e11, float e12, float e13, float e14, float e15)
        {
            return new Vector512<float>(new[] { e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15 });
        }

        public static T GetElement<T>(Vector512<T> vector, int index) where T : unmanaged
        {
            if (index < 0 || index >= Vector512<T>.Count)
                throw new ArgumentOutOfRangeException(nameof(index));

            Span<T> temp = stackalloc T[Vector512<T>.Count];
            vector.CopyTo(temp);
            return temp[index];
        }
    }
}

#endif

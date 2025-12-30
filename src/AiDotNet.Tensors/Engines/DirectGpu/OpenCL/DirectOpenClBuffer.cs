// Copyright (c) AiDotNet. All rights reserved.
// Pure P/Invoke OpenCL buffer - NO ILGPU dependency.
// Works on ALL .NET versions including .NET Framework 4.6.2.

using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    /// <summary>
    /// OpenCL buffer wrapper using pure P/Invoke. No ILGPU dependency.
    /// </summary>
    internal sealed class DirectOpenClBuffer : IDisposable
    {
        private IntPtr _buffer;
        private readonly DirectOpenClContext _context;
        private readonly int _length;
        private bool _disposed;

        public IntPtr Handle => _buffer;
        public int Length => _length;

        /// <summary>
        /// Creates a buffer and uploads data from host.
        /// </summary>
        public DirectOpenClBuffer(DirectOpenClContext context, float[] data)
        {
            _context = context;
            _length = data.Length;

            GCHandle handle = GCHandle.Alloc(data, GCHandleType.Pinned);
            try
            {
                _buffer = OpenClNativeBindings.CreateBuffer(
                    context.Context,
                    OpenClNativeBindings.CL_MEM_READ_WRITE | OpenClNativeBindings.CL_MEM_COPY_HOST_PTR,
                    (UIntPtr)(data.Length * sizeof(float)),
                    handle.AddrOfPinnedObject(),
                    out int err);

                if (err != OpenClNativeBindings.CL_SUCCESS || _buffer == IntPtr.Zero)
                    throw new InvalidOperationException($"Failed to create OpenCL buffer: {err}");
            }
            finally
            {
                handle.Free();
            }
        }

        /// <summary>
        /// Creates an empty buffer of specified size.
        /// </summary>
        public DirectOpenClBuffer(DirectOpenClContext context, int size)
        {
            _context = context;
            _length = size;

            _buffer = OpenClNativeBindings.CreateBuffer(
                context.Context,
                OpenClNativeBindings.CL_MEM_READ_WRITE,
                (UIntPtr)(size * sizeof(float)),
                IntPtr.Zero,
                out int err);

            if (err != OpenClNativeBindings.CL_SUCCESS || _buffer == IntPtr.Zero)
                throw new InvalidOperationException($"Failed to create OpenCL buffer: {err}");
        }

        /// <summary>
        /// Downloads buffer contents to a new array.
        /// </summary>
        public float[] ToArray()
        {
            var result = new float[_length];
            CopyToHost(result);
            return result;
        }

        /// <summary>
        /// Downloads buffer contents to existing array.
        /// </summary>
        public void CopyToHost(float[] destination)
        {
            if (destination.Length < _length)
                throw new ArgumentException("Destination array too small");

            GCHandle handle = GCHandle.Alloc(destination, GCHandleType.Pinned);
            try
            {
                int err = OpenClNativeBindings.EnqueueReadBuffer(
                    _context.CommandQueue,
                    _buffer,
                    1, // blocking
                    UIntPtr.Zero,
                    (UIntPtr)(_length * sizeof(float)),
                    handle.AddrOfPinnedObject(),
                    0,
                    IntPtr.Zero,
                    IntPtr.Zero);

                if (err != OpenClNativeBindings.CL_SUCCESS)
                    throw new InvalidOperationException($"Failed to read OpenCL buffer: {err}");
            }
            finally
            {
                handle.Free();
            }
        }

        /// <summary>
        /// Uploads data to buffer.
        /// </summary>
        public void CopyFromHost(float[] source)
        {
            if (source.Length > _length)
                throw new ArgumentException("Source array too large");

            GCHandle handle = GCHandle.Alloc(source, GCHandleType.Pinned);
            try
            {
                int err = OpenClNativeBindings.EnqueueWriteBuffer(
                    _context.CommandQueue,
                    _buffer,
                    1, // blocking
                    UIntPtr.Zero,
                    (UIntPtr)(source.Length * sizeof(float)),
                    handle.AddrOfPinnedObject(),
                    0,
                    IntPtr.Zero,
                    IntPtr.Zero);

                if (err != OpenClNativeBindings.CL_SUCCESS)
                    throw new InvalidOperationException($"Failed to write OpenCL buffer: {err}");
            }
            finally
            {
                handle.Free();
            }
        }

        public void Dispose()
        {
            if (_disposed) return;

            if (_buffer != IntPtr.Zero)
            {
                OpenClNativeBindings.ReleaseMemObject(_buffer);
                _buffer = IntPtr.Zero;
            }

            _disposed = true;
        }
    }

    /// <summary>
    /// OpenCL byte buffer wrapper using pure P/Invoke.
    /// Used for storing packed sparse indices (1 byte per group of 4 elements).
    /// </summary>
    internal sealed class DirectOpenClByteBuffer : IDisposable
    {
        private IntPtr _buffer;
        private readonly DirectOpenClContext _context;
        private readonly int _length;
        private bool _disposed;

        public IntPtr Handle => _buffer;
        public int Length => _length;

        /// <summary>
        /// Creates a byte buffer and uploads data from host.
        /// </summary>
        public DirectOpenClByteBuffer(DirectOpenClContext context, byte[] data)
        {
            _context = context;
            _length = data.Length;

            GCHandle handle = GCHandle.Alloc(data, GCHandleType.Pinned);
            try
            {
                _buffer = OpenClNativeBindings.CreateBuffer(
                    context.Context,
                    OpenClNativeBindings.CL_MEM_READ_WRITE | OpenClNativeBindings.CL_MEM_COPY_HOST_PTR,
                    (UIntPtr)data.Length,
                    handle.AddrOfPinnedObject(),
                    out int err);

                if (err != OpenClNativeBindings.CL_SUCCESS || _buffer == IntPtr.Zero)
                    throw new InvalidOperationException($"Failed to create OpenCL byte buffer: {err}");
            }
            finally
            {
                handle.Free();
            }
        }

        /// <summary>
        /// Creates an empty byte buffer of specified size.
        /// </summary>
        public DirectOpenClByteBuffer(DirectOpenClContext context, int size)
        {
            _context = context;
            _length = size;

            _buffer = OpenClNativeBindings.CreateBuffer(
                context.Context,
                OpenClNativeBindings.CL_MEM_READ_WRITE,
                (UIntPtr)size,
                IntPtr.Zero,
                out int err);

            if (err != OpenClNativeBindings.CL_SUCCESS || _buffer == IntPtr.Zero)
                throw new InvalidOperationException($"Failed to create OpenCL byte buffer: {err}");
        }

        /// <summary>
        /// Downloads buffer contents to a new array.
        /// </summary>
        public byte[] ToArray()
        {
            var result = new byte[_length];
            CopyToHost(result);
            return result;
        }

        /// <summary>
        /// Downloads buffer contents to existing array.
        /// </summary>
        public void CopyToHost(byte[] destination)
        {
            if (destination.Length < _length)
                throw new ArgumentException("Destination array too small");

            GCHandle handle = GCHandle.Alloc(destination, GCHandleType.Pinned);
            try
            {
                int err = OpenClNativeBindings.EnqueueReadBuffer(
                    _context.CommandQueue,
                    _buffer,
                    1, // blocking
                    UIntPtr.Zero,
                    (UIntPtr)_length,
                    handle.AddrOfPinnedObject(),
                    0,
                    IntPtr.Zero,
                    IntPtr.Zero);

                if (err != OpenClNativeBindings.CL_SUCCESS)
                    throw new InvalidOperationException($"Failed to read OpenCL byte buffer: {err}");
            }
            finally
            {
                handle.Free();
            }
        }

        public void Dispose()
        {
            if (_disposed) return;

            if (_buffer != IntPtr.Zero)
            {
                OpenClNativeBindings.ReleaseMemObject(_buffer);
                _buffer = IntPtr.Zero;
            }

            _disposed = true;
        }
    }

    /// <summary>
    /// OpenCL int buffer wrapper using pure P/Invoke.
    /// Used for atomic counters in work-stealing kernels.
    /// </summary>
    internal sealed class DirectOpenClIntBuffer : IDisposable
    {
        private IntPtr _buffer;
        private readonly DirectOpenClContext _context;
        private readonly int _length;
        private bool _disposed;

        public IntPtr Handle => _buffer;
        public int Length => _length;

        /// <summary>
        /// Creates an int buffer and uploads data from host.
        /// </summary>
        public DirectOpenClIntBuffer(DirectOpenClContext context, int[] data)
        {
            _context = context;
            _length = data.Length;

            GCHandle handle = GCHandle.Alloc(data, GCHandleType.Pinned);
            try
            {
                _buffer = OpenClNativeBindings.CreateBuffer(
                    context.Context,
                    OpenClNativeBindings.CL_MEM_READ_WRITE | OpenClNativeBindings.CL_MEM_COPY_HOST_PTR,
                    (UIntPtr)(data.Length * sizeof(int)),
                    handle.AddrOfPinnedObject(),
                    out int err);

                if (err != OpenClNativeBindings.CL_SUCCESS || _buffer == IntPtr.Zero)
                    throw new InvalidOperationException($"Failed to create OpenCL int buffer: {err}");
            }
            finally
            {
                handle.Free();
            }
        }

        /// <summary>
        /// Creates an empty int buffer of specified size.
        /// </summary>
        public DirectOpenClIntBuffer(DirectOpenClContext context, int size)
        {
            _context = context;
            _length = size;

            _buffer = OpenClNativeBindings.CreateBuffer(
                context.Context,
                OpenClNativeBindings.CL_MEM_READ_WRITE,
                (UIntPtr)(size * sizeof(int)),
                IntPtr.Zero,
                out int err);

            if (err != OpenClNativeBindings.CL_SUCCESS || _buffer == IntPtr.Zero)
                throw new InvalidOperationException($"Failed to create OpenCL int buffer: {err}");
        }

        /// <summary>
        /// Downloads buffer contents to a new array.
        /// </summary>
        public int[] ToArray()
        {
            var result = new int[_length];
            CopyToHost(result);
            return result;
        }

        /// <summary>
        /// Downloads buffer contents to existing array.
        /// </summary>
        public void CopyToHost(int[] destination)
        {
            if (destination.Length < _length)
                throw new ArgumentException("Destination array too small");

            GCHandle handle = GCHandle.Alloc(destination, GCHandleType.Pinned);
            try
            {
                int err = OpenClNativeBindings.EnqueueReadBuffer(
                    _context.CommandQueue,
                    _buffer,
                    1, // blocking
                    UIntPtr.Zero,
                    (UIntPtr)(_length * sizeof(int)),
                    handle.AddrOfPinnedObject(),
                    0,
                    IntPtr.Zero,
                    IntPtr.Zero);

                if (err != OpenClNativeBindings.CL_SUCCESS)
                    throw new InvalidOperationException($"Failed to read OpenCL int buffer: {err}");
            }
            finally
            {
                handle.Free();
            }
        }

        public void Dispose()
        {
            if (_disposed) return;

            if (_buffer != IntPtr.Zero)
            {
                OpenClNativeBindings.ReleaseMemObject(_buffer);
                _buffer = IntPtr.Zero;
            }

            _disposed = true;
        }
    }
}

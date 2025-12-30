// Copyright (c) AiDotNet. All rights reserved.
// Pure P/Invoke OpenCL kernel - NO ILGPU dependency.
// Works on ALL .NET versions including .NET Framework 4.6.2.

using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    /// <summary>
    /// OpenCL kernel wrapper using pure P/Invoke. No ILGPU dependency.
    /// </summary>
    internal sealed class DirectOpenClKernel : IDisposable
    {
        private IntPtr _kernel;
        private readonly DirectOpenClContext _context;
        private bool _disposed;

        public IntPtr Handle => _kernel;

        public DirectOpenClKernel(DirectOpenClContext context, DirectOpenClProgram program, string kernelName)
        {
            _context = context;

            _kernel = OpenClNativeBindings.CreateKernel(program.Handle, kernelName, out int err);
            if (err != OpenClNativeBindings.CL_SUCCESS || _kernel == IntPtr.Zero)
                throw new InvalidOperationException($"Failed to create OpenCL kernel '{kernelName}': {err}");
        }

        #region SetArg Overloads

        public void SetArg(uint index, IntPtr bufferHandle)
        {
            IntPtr ptr = Marshal.AllocHGlobal(IntPtr.Size);
            try
            {
                Marshal.WriteIntPtr(ptr, bufferHandle);
                int err = OpenClNativeBindings.SetKernelArg(_kernel, index, (UIntPtr)IntPtr.Size, ptr);
                if (err != OpenClNativeBindings.CL_SUCCESS)
                    throw new InvalidOperationException($"Failed to set kernel arg {index}: {err}");
            }
            finally
            {
                Marshal.FreeHGlobal(ptr);
            }
        }

        public void SetArg(uint index, int value)
        {
            IntPtr ptr = Marshal.AllocHGlobal(sizeof(int));
            try
            {
                Marshal.WriteInt32(ptr, value);
                int err = OpenClNativeBindings.SetKernelArg(_kernel, index, (UIntPtr)sizeof(int), ptr);
                if (err != OpenClNativeBindings.CL_SUCCESS)
                    throw new InvalidOperationException($"Failed to set kernel arg {index}: {err}");
            }
            finally
            {
                Marshal.FreeHGlobal(ptr);
            }
        }

        public void SetArg(uint index, float value)
        {
            IntPtr ptr = Marshal.AllocHGlobal(sizeof(float));
            try
            {
                Marshal.Copy(new float[] { value }, 0, ptr, 1);
                int err = OpenClNativeBindings.SetKernelArg(_kernel, index, (UIntPtr)sizeof(float), ptr);
                if (err != OpenClNativeBindings.CL_SUCCESS)
                    throw new InvalidOperationException($"Failed to set kernel arg {index}: {err}");
            }
            finally
            {
                Marshal.FreeHGlobal(ptr);
            }
        }

        /// <summary>
        /// Sets a local memory argument (for shared memory allocation).
        /// </summary>
        public void SetLocalArg(uint index, int sizeInBytes)
        {
            int err = OpenClNativeBindings.SetKernelArg(_kernel, index, (UIntPtr)sizeInBytes, IntPtr.Zero);
            if (err != OpenClNativeBindings.CL_SUCCESS)
                throw new InvalidOperationException($"Failed to set local arg {index}: {err}");
        }

        #endregion

        #region Execution

        /// <summary>
        /// Executes kernel with 1D work distribution.
        /// </summary>
        public void Execute1D(int globalSize, int localSize)
        {
            // Round up global size to multiple of local size
            int alignedGlobal = ((globalSize + localSize - 1) / localSize) * localSize;

            var globalSizes = new UIntPtr[] { (UIntPtr)alignedGlobal };
            var localSizes = new UIntPtr[] { (UIntPtr)localSize };

            int err = OpenClNativeBindings.EnqueueNDRangeKernel(
                _context.CommandQueue,
                _kernel,
                1, // work_dim
                null, // global_work_offset
                globalSizes,
                localSizes,
                0,
                IntPtr.Zero,
                IntPtr.Zero);

            if (err != OpenClNativeBindings.CL_SUCCESS)
                throw new InvalidOperationException($"Failed to enqueue kernel: {err}");
        }

        /// <summary>
        /// Executes kernel with 2D work distribution.
        /// </summary>
        public void Execute2D(int globalSizeX, int globalSizeY, int localSizeX, int localSizeY)
        {
            // Round up global sizes to multiples of local sizes
            int alignedGlobalX = ((globalSizeX + localSizeX - 1) / localSizeX) * localSizeX;
            int alignedGlobalY = ((globalSizeY + localSizeY - 1) / localSizeY) * localSizeY;

            var globalSizes = new UIntPtr[] { (UIntPtr)alignedGlobalX, (UIntPtr)alignedGlobalY };
            var localSizes = new UIntPtr[] { (UIntPtr)localSizeX, (UIntPtr)localSizeY };

            int err = OpenClNativeBindings.EnqueueNDRangeKernel(
                _context.CommandQueue,
                _kernel,
                2, // work_dim
                null, // global_work_offset
                globalSizes,
                localSizes,
                0,
                IntPtr.Zero,
                IntPtr.Zero);

            if (err != OpenClNativeBindings.CL_SUCCESS)
                throw new InvalidOperationException($"Failed to enqueue kernel: {err}");
        }

        #endregion

        #region Profiled Execution

        /// <summary>
        /// Executes kernel with 2D work distribution on the profiling queue and returns an event handle.
        /// The caller must release the event after getting profiling info.
        /// </summary>
        /// <returns>Event handle for profiling, or IntPtr.Zero if profiling is not available.</returns>
        public IntPtr Execute2DProfiled(int globalSizeX, int globalSizeY, int localSizeX, int localSizeY)
        {
            if (!_context.IsProfilingEnabled)
            {
                // Fall back to non-profiled execution
                Execute2D(globalSizeX, globalSizeY, localSizeX, localSizeY);
                return IntPtr.Zero;
            }

            // Round up global sizes to multiples of local sizes
            int alignedGlobalX = ((globalSizeX + localSizeX - 1) / localSizeX) * localSizeX;
            int alignedGlobalY = ((globalSizeY + localSizeY - 1) / localSizeY) * localSizeY;

            var globalSizes = new UIntPtr[] { (UIntPtr)alignedGlobalX, (UIntPtr)alignedGlobalY };
            var localSizes = new UIntPtr[] { (UIntPtr)localSizeX, (UIntPtr)localSizeY };

            // Allocate event handle
            IntPtr eventHandle = Marshal.AllocHGlobal(IntPtr.Size);
            try
            {
                int err = OpenClNativeBindings.EnqueueNDRangeKernel(
                    _context.ProfilingCommandQueue,
                    _kernel,
                    2, // work_dim
                    null, // global_work_offset
                    globalSizes,
                    localSizes,
                    0,
                    IntPtr.Zero,
                    eventHandle);

                if (err != OpenClNativeBindings.CL_SUCCESS)
                    throw new InvalidOperationException($"Failed to enqueue kernel: {err}");

                // Read the event pointer from the allocated memory
                IntPtr eventPtr = Marshal.ReadIntPtr(eventHandle);
                return eventPtr;
            }
            finally
            {
                Marshal.FreeHGlobal(eventHandle);
            }
        }

        /// <summary>
        /// Executes kernel with 1D work distribution on the profiling queue and returns an event handle.
        /// </summary>
        public IntPtr Execute1DProfiled(int globalSize, int localSize)
        {
            if (!_context.IsProfilingEnabled)
            {
                Execute1D(globalSize, localSize);
                return IntPtr.Zero;
            }

            int alignedGlobal = ((globalSize + localSize - 1) / localSize) * localSize;

            var globalSizes = new UIntPtr[] { (UIntPtr)alignedGlobal };
            var localSizes = new UIntPtr[] { (UIntPtr)localSize };

            IntPtr eventHandle = Marshal.AllocHGlobal(IntPtr.Size);
            try
            {
                int err = OpenClNativeBindings.EnqueueNDRangeKernel(
                    _context.ProfilingCommandQueue,
                    _kernel,
                    1,
                    null,
                    globalSizes,
                    localSizes,
                    0,
                    IntPtr.Zero,
                    eventHandle);

                if (err != OpenClNativeBindings.CL_SUCCESS)
                    throw new InvalidOperationException($"Failed to enqueue kernel: {err}");

                return Marshal.ReadIntPtr(eventHandle);
            }
            finally
            {
                Marshal.FreeHGlobal(eventHandle);
            }
        }

        #endregion

        public void Dispose()
        {
            if (_disposed) return;

            if (_kernel != IntPtr.Zero)
            {
                OpenClNativeBindings.ReleaseKernel(_kernel);
                _kernel = IntPtr.Zero;
            }

            _disposed = true;
        }
    }
}

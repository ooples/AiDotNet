using System;
using AiDotNet.Interfaces;

namespace AiDotNet.HardwareAcceleration
{
    /// <summary>
    /// Device memory handle for hardware accelerators
    /// </summary>
    /// <typeparam name="T">The type of data stored in device memory</typeparam>
    public class DeviceMemory<T> : IDisposable where T : unmanaged
    {
        private bool _disposed;

        /// <summary>
        /// Gets or sets the pointer to the device memory
        /// </summary>
        public IntPtr Pointer { get; set; }

        /// <summary>
        /// Gets or sets the size of the allocated memory in number of elements
        /// </summary>
        public int Size { get; set; }

        /// <summary>
        /// Gets or sets the device ID
        /// </summary>
        public int DeviceId { get; set; }

        /// <summary>
        /// Gets or sets the associated accelerator
        /// </summary>
        public IAccelerator Accelerator { get; set; } = default!;

        /// <summary>
        /// Initializes a new instance of the DeviceMemory class
        /// </summary>
        public DeviceMemory()
        {
        }

        /// <summary>
        /// Initializes a new instance of the DeviceMemory class
        /// </summary>
        /// <param name="pointer">Pointer to the device memory</param>
        /// <param name="size">Size in number of elements</param>
        /// <param name="accelerator">The associated accelerator</param>
        public DeviceMemory(IntPtr pointer, int size, IAccelerator accelerator)
        {
            Pointer = pointer;
            Size = size;
            Accelerator = accelerator;
        }

        /// <summary>
        /// Releases the device memory
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Releases the device memory
        /// </summary>
        /// <param name="disposing">True if disposing managed resources</param>
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    // Free device memory through the accelerator
                    if (Pointer != IntPtr.Zero && Accelerator != null)
                    {
                        Accelerator.FreeDeviceMemory(Pointer);
                    }
                }
                _disposed = true;
            }
        }
    }
}
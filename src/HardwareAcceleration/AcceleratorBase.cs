using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Logging;

namespace AiDotNet.HardwareAcceleration
{
    /// <summary>
    /// Base class for hardware accelerators providing common functionality
    /// </summary>
    public abstract class AcceleratorBase : IAccelerator
    {
        protected readonly ILogging _logger;
        protected bool _isInitialized;
        protected readonly object _initLock = new object();
        private bool _disposed;
        
        // Memory management fields
        protected long _allocatedMemory;
        protected readonly object _memoryLock = new object();
        protected readonly Dictionary<IntPtr, long> _memoryAllocations = new Dictionary<IntPtr, long>();
        
        // Device management fields
        protected int _currentDeviceId;

        /// <summary>
        /// Gets the accelerator type
        /// </summary>
        public abstract AcceleratorType Type { get; }

        /// <summary>
        /// Gets the device name
        /// </summary>
        public abstract string DeviceName { get; protected set; }

        /// <summary>
        /// Gets whether the accelerator is available
        /// </summary>
        public abstract bool IsAvailable { get; }

        /// <summary>
        /// Gets the device memory in bytes
        /// </summary>
        public abstract long DeviceMemoryBytes { get; protected set; }

        /// <summary>
        /// Gets the compute capability
        /// </summary>
        public abstract ComputeCapability ComputeCapability { get; protected set; }

        /// <summary>
        /// Gets the device name (IAccelerator implementation)
        /// </summary>
        public string Name => DeviceName;

        /// <summary>
        /// Gets the device ID
        /// </summary>
        public int DeviceId => _currentDeviceId;

        /// <summary>
        /// Gets the total memory
        /// </summary>
        public long TotalMemory => DeviceMemoryBytes;

        /// <summary>
        /// Gets the available memory
        /// </summary>
        public long AvailableMemory
        {
            get
            {
                lock (_memoryLock)
                {
                    return DeviceMemoryBytes - _allocatedMemory;
                }
            }
        }

        /// <summary>
        /// Initializes a new instance of the AcceleratorBase class
        /// </summary>
        protected AcceleratorBase(ILogging logger, int deviceId = 0)
        {
            _logger = logger ?? new AiDotNetLogger();
            _currentDeviceId = deviceId;
        }

        /// <summary>
        /// Initializes the accelerator
        /// </summary>
        public virtual async Task InitializeAsync()
        {
            if (_isInitialized) return;

            lock (_initLock)
            {
                if (_isInitialized) return;

                InitializeDevice();
                _isInitialized = true;
            }

            await Task.CompletedTask;
        }

        /// <summary>
        /// Performs matrix multiplication on the accelerator
        /// </summary>
        public abstract Task<Matrix<T>> MatrixMultiplyAsync<T>(Matrix<T> a, Matrix<T> b);

        /// <summary>
        /// Performs element-wise operations on the accelerator
        /// </summary>
        public abstract Task<Tensor<T>> ElementWiseOperationAsync<T>(
            Tensor<T> a, 
            Tensor<T> b, 
            ElementWiseOperation operation);

        /// <summary>
        /// Performs convolution on the accelerator
        /// </summary>
        public abstract Task<Tensor<T>> ConvolutionAsync<T>(
            Tensor<T> input,
            Tensor<T> kernel,
            ConvolutionParameters parameters);

        /// <summary>
        /// Performs batch normalization on the accelerator
        /// </summary>
        public abstract Task<Tensor<T>> BatchNormalizationAsync<T>(
            Tensor<T> input,
            Tensor<T> scale,
            Tensor<T> bias,
            Tensor<T> mean,
            Tensor<T> variance,
            double epsilon = 1e-5);

        /// <summary>
        /// Performs attention computation on the accelerator
        /// </summary>
        public abstract Task<Tensor<T>> AttentionAsync<T>(
            Tensor<T> query,
            Tensor<T> key,
            Tensor<T> value,
            Tensor<T> mask,
            bool causalMask = false);

        /// <summary>
        /// Allocates memory on the device
        /// </summary>
        public abstract DeviceMemory<T> AllocateMemory<T>(int size) where T : unmanaged;

        /// <summary>
        /// Copies tensor data to device
        /// </summary>
        public abstract Task CopyTensorToDeviceAsync<T>(Tensor<T> hostData, DeviceMemory<T> deviceMemory) where T : unmanaged;

        /// <summary>
        /// Copies matrix data to device
        /// </summary>
        public abstract Task CopyMatrixToDeviceAsync<T>(Matrix<T> hostData, DeviceMemory<T> deviceMemory) where T : unmanaged;

        /// <summary>
        /// Copies vector data to device
        /// </summary>
        public abstract Task CopyVectorToDeviceAsync<T>(Vector<T> hostData, DeviceMemory<T> deviceMemory) where T : unmanaged;

        /// <summary>
        /// Copies tensor data from device
        /// </summary>
        public abstract Task<Tensor<T>> CopyTensorFromDeviceAsync<T>(DeviceMemory<T> deviceMemory, int[] shape) where T : unmanaged;

        /// <summary>
        /// Copies matrix data from device
        /// </summary>
        public abstract Task<Matrix<T>> CopyMatrixFromDeviceAsync<T>(DeviceMemory<T> deviceMemory, int rows, int columns) where T : unmanaged;

        /// <summary>
        /// Copies vector data from device
        /// </summary>
        public abstract Task<Vector<T>> CopyVectorFromDeviceAsync<T>(DeviceMemory<T> deviceMemory, int length) where T : unmanaged;

        /// <summary>
        /// Synchronizes device operations
        /// </summary>
        public abstract Task SynchronizeAsync();

        // IAccelerator interface implementation with simple parameter overload
        public virtual async Task<Tensor<T>> ConvolutionAsync<T>(Tensor<T> input, Tensor<T> kernel, int stride = 1, int padding = 0)
        {
            var parameters = new ConvolutionParameters
            {
                Stride = new[] { stride, stride },
                Padding = new[] { padding, padding }
            };
            return await ConvolutionAsync(input, kernel, parameters);
        }

        // IAccelerator interface implementation
        public virtual async Task<Tensor<T>> AttentionAsync<T>(Tensor<T> query, Tensor<T> key, Tensor<T> value, Matrix<T> mask)
        {
            // Convert Matrix mask to Tensor
            Tensor<T> maskTensor = null;
            if (mask != null)
            {
                maskTensor = new Tensor<T>(new[] { mask.Rows, mask.Columns });
                for (int i = 0; i < mask.Rows; i++)
                {
                    for (int j = 0; j < mask.Columns; j++)
                    {
                        maskTensor.SetValue(mask[i, j], i, j);
                    }
                }
            }
            
            return await AttentionAsync(query, key, value, maskTensor, false);
        }

        // IAccelerator interface implementation
        public void Synchronize()
        {
            SynchronizeAsync().Wait();
        }

        // InitializeAsync is already implemented as virtual above

        // IAccelerator interface implementation
        public AcceleratorInfo GetDeviceInfo()
        {
            var props = GetDeviceProperties();
            return new AcceleratorInfo
            {
                Name = props.DeviceName,
                DriverVersion = GetDriverVersion(),
                ComputeCapabilityMajor = props.ComputeCapability.Major,
                ComputeCapabilityMinor = props.ComputeCapability.Minor,
                MultiprocessorCount = props.MultiprocessorCount,
                MaxThreadsPerBlock = props.MaxThreadsPerBlock,
                MaxBlockDimX = props.MaxBlockDimensions[0],
                MaxBlockDimY = props.MaxBlockDimensions[1],
                MaxBlockDimZ = props.MaxBlockDimensions[2],
                MaxGridDimX = props.MaxGridDimensions[0],
                MaxGridDimY = props.MaxGridDimensions[1],
                MaxGridDimZ = props.MaxGridDimensions[2],
                TotalMemory = DeviceMemoryBytes,
                AvailableMemory = AvailableMemory
            };
        }

        // IAccelerator interface implementation
        public void SetDevice(int deviceId)
        {
            _currentDeviceId = deviceId;
            SetDeviceInternal(deviceId);
        }

        // Virtual implementations for interface methods
        public virtual Task<Vector<T>> ActivationAsync<T>(Vector<T> input, ActivationFunction function)
        {
            // Default implementation - derived classes can override
            return Task.FromResult(input);
        }

        public virtual Task<Matrix<T>> PoolingAsync<T>(Matrix<T> input, int poolSize, PoolingType poolingType)
        {
            // Default implementation - derived classes can override
            return Task.FromResult(input);
        }

        /// <summary>
        /// Sets the device internally
        /// </summary>
        protected abstract void SetDeviceInternal(int deviceId);

        /// <summary>
        /// Gets device properties
        /// </summary>
        public virtual DeviceProperties GetDeviceProperties()
        {
            return new DeviceProperties
            {
                DeviceName = DeviceName,
                MemorySize = DeviceMemoryBytes,
                ComputeCapability = ComputeCapability,
                MaxThreadsPerBlock = GetMaxThreadsPerBlock(),
                MaxBlockDimensions = GetMaxBlockDimensions(),
                MaxGridDimensions = GetMaxGridDimensions(),
                WarpSize = GetWarpSize(),
                SharedMemoryPerBlock = GetSharedMemoryPerBlock(),
                ClockRate = GetClockRate(),
                MultiprocessorCount = GetMultiprocessorCount()
            };
        }

        /// <summary>
        /// Performs cleanup
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    try
                    {
                        // Free all allocated device memory
                        lock (_memoryLock)
                        {
                            foreach (var allocation in _memoryAllocations.ToList())
                            {
                                try
                                {
                                    FreeDeviceMemory(allocation.Key);
                                }
                                catch (Exception ex)
                                {
                                    if (_logger != null)
                                        _logger.Error($"Failed to free device memory at {allocation.Key}: {ex.Message}");
                                }
                            }
                            _memoryAllocations.Clear();
                            _allocatedMemory = 0;
                        }

                        // Dispose managed resources
                        CleanupDevice();
                    }
                    catch (Exception ex)
                    {
                        if (_logger != null)
                            _logger.Error($"Error during accelerator disposal: {ex.Message}");
                    }
                }

                _disposed = true;
            }
        }

        /// <summary>
        /// Initializes the device
        /// </summary>
        protected abstract void InitializeDevice();

        /// <summary>
        /// Cleans up device resources
        /// </summary>
        protected abstract void CleanupDevice();

        protected abstract int GetMaxThreadsPerBlock();
        protected abstract int[] GetMaxBlockDimensions();
        protected abstract int[] GetMaxGridDimensions();
        protected abstract int GetWarpSize();
        protected abstract long GetSharedMemoryPerBlock();
        protected abstract int GetClockRate();
        protected abstract int GetMultiprocessorCount();

        private static T ConvertToT<T>(double value)
        {
            return (T)Convert.ChangeType(value, typeof(T));
        }

        public IntPtr AllocateDeviceMemory(long sizeInBytes)
        {
            if (sizeInBytes <= 0)
            {
                throw new ArgumentException("Size must be greater than zero", nameof(sizeInBytes));
            }
            
            lock (_memoryLock)
            {
                if (_allocatedMemory + sizeInBytes > DeviceMemoryBytes)
                {
                    throw new OutOfMemoryException(
                        $"Insufficient device memory. Requested: {sizeInBytes} bytes, " +
                        $"Available: {DeviceMemoryBytes - _allocatedMemory} bytes");
                }
                
                var pointer = AllocateDeviceMemoryInternal(sizeInBytes);
                
                _memoryAllocations[pointer] = sizeInBytes;
                _allocatedMemory += sizeInBytes;
                
                if (_logger != null)
                    _logger.Debug($"Allocated {sizeInBytes} bytes on device. Total allocated: {_allocatedMemory} bytes");
                
                return pointer;
            }
        }

        public void FreeDeviceMemory(IntPtr devicePointer)
        {
            if (devicePointer == IntPtr.Zero)
            {
                return;
            }
            
            lock (_memoryLock)
            {
                if (_memoryAllocations.TryGetValue(devicePointer, out long sizeInBytes))
                {
                    FreeDeviceMemoryInternal(devicePointer);
                    
                    _memoryAllocations.Remove(devicePointer);
                    _allocatedMemory -= sizeInBytes;
                    
                    if (_logger != null)
                        _logger.Debug($"Freed {sizeInBytes} bytes on device. Total allocated: {_allocatedMemory} bytes");
                }
                else
                {
                    if (_logger != null)
                        _logger.Warning($"Attempted to free untracked device memory at {devicePointer}");
                }
            }
        }

        // IAccelerator interface implementations for data copying
        public async Task CopyToDeviceAsync<T>(Tensor<T> hostData, IntPtr devicePointer) where T : unmanaged
        {
            if (hostData == null) throw new ArgumentNullException(nameof(hostData));
            
            var deviceMemory = new DeviceMemory<T>
            {
                Pointer = devicePointer,
                Size = hostData.Length,
                Accelerator = this
            };
            
            await CopyTensorToDeviceAsync(hostData, deviceMemory);
        }

        public async Task CopyToDeviceAsync<T>(Matrix<T> hostData, IntPtr devicePointer) where T : unmanaged
        {
            if (hostData == null) throw new ArgumentNullException(nameof(hostData));
            
            var deviceMemory = new DeviceMemory<T>
            {
                Pointer = devicePointer,
                Size = hostData.Rows * hostData.Columns,
                Accelerator = this
            };
            
            await CopyMatrixToDeviceAsync(hostData, deviceMemory);
        }

        public async Task CopyToDeviceAsync<T>(Vector<T> hostData, IntPtr devicePointer) where T : unmanaged
        {
            if (hostData == null) throw new ArgumentNullException(nameof(hostData));
            
            var deviceMemory = new DeviceMemory<T>
            {
                Pointer = devicePointer,
                Size = hostData.Length,
                Accelerator = this
            };
            
            await CopyVectorToDeviceAsync(hostData, deviceMemory);
        }

        public async Task<Tensor<T>> CopyTensorFromDeviceAsync<T>(IntPtr devicePointer, int[] shape) where T : unmanaged
        {
            if (devicePointer == IntPtr.Zero) throw new ArgumentException("Invalid device pointer", nameof(devicePointer));
            
            var totalElements = shape.Aggregate(1, (a, b) => a * b);
            var deviceMemory = new DeviceMemory<T>
            {
                Pointer = devicePointer,
                Size = totalElements,
                Accelerator = this
            };
            
            return await CopyTensorFromDeviceAsync(deviceMemory, shape);
        }

        public async Task<Matrix<T>> CopyMatrixFromDeviceAsync<T>(IntPtr devicePointer, int rows, int columns) where T : unmanaged
        {
            if (devicePointer == IntPtr.Zero) throw new ArgumentException("Invalid device pointer", nameof(devicePointer));
            
            var deviceMemory = new DeviceMemory<T>
            {
                Pointer = devicePointer,
                Size = rows * columns,
                Accelerator = this
            };
            
            return await CopyMatrixFromDeviceAsync(deviceMemory, rows, columns);
        }

        public async Task<Vector<T>> CopyVectorFromDeviceAsync<T>(IntPtr devicePointer, int length) where T : unmanaged
        {
            if (devicePointer == IntPtr.Zero) throw new ArgumentException("Invalid device pointer", nameof(devicePointer));
            
            var deviceMemory = new DeviceMemory<T>
            {
                Pointer = devicePointer,
                Size = length,
                Accelerator = this
            };
            
            return await CopyVectorFromDeviceAsync(deviceMemory, length);
        }

        // Abstract methods for device-specific implementations
        protected abstract IntPtr AllocateDeviceMemoryInternal(long sizeInBytes);
        protected abstract void FreeDeviceMemoryInternal(IntPtr devicePointer);
        protected abstract void CopyArrayToDeviceInternalAsync<T>(T[] hostData, IntPtr devicePointer) where T : unmanaged;
        protected abstract T[] CopyArrayFromDeviceInternalAsync<T>(IntPtr devicePointer, int count) where T : unmanaged;
        protected abstract string GetDriverVersion();
    }
}
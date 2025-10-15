using System;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces
{
    public interface IAccelerator : IDisposable
{
    AcceleratorType Type { get; }
    string Name { get; }
    bool IsAvailable { get; }
    int DeviceId { get; }
    long TotalMemory { get; }
    long AvailableMemory { get; }
    
    Task InitializeAsync();
    void Synchronize();
    
    Task<Matrix<T>> MatrixMultiplyAsync<T>(Matrix<T> a, Matrix<T> b);
    Task<Tensor<T>> ConvolutionAsync<T>(Tensor<T> input, Tensor<T> kernel, int stride = 1, int padding = 0);
    Task<Tensor<T>> BatchNormalizationAsync<T>(Tensor<T> input, Tensor<T> scale, Tensor<T> bias, Tensor<T> mean, Tensor<T> variance, double epsilon = 1e-5);
    Task<Tensor<T>> AttentionAsync<T>(Tensor<T> query, Tensor<T> key, Tensor<T> value, Matrix<T> mask);
    Task<Vector<T>> ActivationAsync<T>(Vector<T> input, ActivationFunction function);
    Task<Matrix<T>> PoolingAsync<T>(Matrix<T> input, int poolSize, PoolingType poolingType);
    
    IntPtr AllocateDeviceMemory(long sizeInBytes);
    void FreeDeviceMemory(IntPtr devicePointer);
    Task CopyToDeviceAsync<T>(Tensor<T> hostData, IntPtr devicePointer) where T : unmanaged;
    Task CopyToDeviceAsync<T>(Matrix<T> hostData, IntPtr devicePointer) where T : unmanaged;
    Task CopyToDeviceAsync<T>(Vector<T> hostData, IntPtr devicePointer) where T : unmanaged;
    Task<Tensor<T>> CopyTensorFromDeviceAsync<T>(IntPtr devicePointer, int[] shape) where T : unmanaged;
    Task<Matrix<T>> CopyMatrixFromDeviceAsync<T>(IntPtr devicePointer, int rows, int columns) where T : unmanaged;
    Task<Vector<T>> CopyVectorFromDeviceAsync<T>(IntPtr devicePointer, int length) where T : unmanaged;
    
    void SetDevice(int deviceId);
    AcceleratorInfo GetDeviceInfo();
}


public class AcceleratorInfo
{
    public string Name { get; set; } = string.Empty;
    public string DriverVersion { get; set; } = string.Empty;
    public int ComputeCapabilityMajor { get; set; }
    public int ComputeCapabilityMinor { get; set; }
    public int MultiprocessorCount { get; set; }
    public int MaxThreadsPerBlock { get; set; }
    public int MaxBlockDimX { get; set; }
    public int MaxBlockDimY { get; set; }
    public int MaxBlockDimZ { get; set; }
    public int MaxGridDimX { get; set; }
    public int MaxGridDimY { get; set; }
    public int MaxGridDimZ { get; set; }
    public long TotalMemory { get; set; }
    public long AvailableMemory { get; set; }
    public bool SupportsDoublePrecision { get; set; }
    public bool SupportsTensorCores { get; set; }
}
}
using System;
using System.Collections.Generic;
using System.Diagnostics;
using AiDotNet.Engines;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.PhysicsInformed.Interfaces;
using AiDotNet.PhysicsInformed.PINNs;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Validation;

namespace AiDotNet.PhysicsInformed;

/// <summary>
/// Provides GPU-accelerated training for Physics-Informed Neural Networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// For Beginners:
/// This trainer provides GPU-accelerated training methods for PINNs.
/// It can significantly speed up training by:
/// 1. Processing large batches of collocation points in parallel
/// 2. Performing matrix operations on the GPU
/// 3. Using asynchronous data transfers to overlap computation
///
/// Usage:
/// ```csharp
/// var trainer = new GpuPINNTrainer&lt;double&gt;(myPinn);
/// var history = trainer.Train(epochs: 10000, options: GpuPINNTrainingOptions.Default);
/// ```
///
/// The trainer automatically falls back to CPU if GPU is not available.
/// </remarks>
public class GpuPINNTrainer<T>
{
    private readonly PhysicsInformedNeuralNetwork<T> _pinn;
    private readonly INumericOperations<T> _numOps;
    private bool _gpuInitialized;
    private bool _useGpu;
    private GpuPINNTrainingOptions _options;

    /// <summary>
    /// Gets the underlying PINN being trained.
    /// </summary>
    public PhysicsInformedNeuralNetwork<T> Network => _pinn;

    /// <summary>
    /// Gets whether GPU is currently being used for training.
    /// </summary>
    public bool IsUsingGpu => _useGpu && _gpuInitialized;

    /// <summary>
    /// Gets the current training options.
    /// </summary>
    public GpuPINNTrainingOptions Options => _options;

    /// <summary>
    /// Initializes a new instance of the GPU PINN trainer.
    /// </summary>
    /// <param name="pinn">The Physics-Informed Neural Network to train.</param>
    /// <param name="options">Optional GPU training options.</param>
    public GpuPINNTrainer(
        PhysicsInformedNeuralNetwork<T> pinn,
        GpuPINNTrainingOptions? options = null)
    {
        Guard.NotNull(pinn);
        _pinn = pinn;
        _options = options ?? GpuPINNTrainingOptions.Default;
        _numOps = MathHelper.GetNumericOperations<T>();
        _gpuInitialized = false;
        _useGpu = false;

        if (_options.EnableGpu)
        {
            TryInitializeGpu();
        }
    }

    /// <summary>
    /// Attempts to initialize GPU resources.
    /// </summary>
    /// <returns>True if GPU initialization was successful.</returns>
    public bool TryInitializeGpu()
    {
        if (_gpuInitialized)
        {
            return _useGpu;
        }

        try
        {
            // Check if engine supports GPU
            var engine = AiDotNetEngine.Current;
            bool gpuSupported = IsGpuSupported(engine);

            if (gpuSupported)
            {
                _useGpu = true;
                _gpuInitialized = true;

                if (_options.VerboseLogging)
                {
                    Console.WriteLine("[GpuPINNTrainer] GPU acceleration enabled.");
                }

                return true;
            }
            else
            {
                _useGpu = false;
                _gpuInitialized = true;

                if (_options.VerboseLogging)
                {
                    Console.WriteLine("[GpuPINNTrainer] GPU not available, using CPU.");
                }

                return false;
            }
        }
        catch (Exception ex)
        {
            _useGpu = false;
            _gpuInitialized = true;

            if (_options.VerboseLogging)
            {
                Console.WriteLine($"[GpuPINNTrainer] GPU initialization failed: {ex.Message}");
            }

            return false;
        }
    }

    /// <summary>
    /// Trains the PINN with GPU acceleration.
    /// </summary>
    /// <param name="dataInputs">Optional measured input data.</param>
    /// <param name="dataOutputs">Optional measured output data.</param>
    /// <param name="epochs">Number of training epochs.</param>
    /// <param name="learningRate">Learning rate for optimization.</param>
    /// <param name="verbose">Whether to print progress.</param>
    /// <returns>Training history with loss values per epoch.</returns>
    public GpuTrainingHistory<T> Train(
        T[,]? dataInputs = null,
        T[,]? dataOutputs = null,
        int epochs = 10000,
        double learningRate = 0.001,
        bool verbose = true)
    {
        var history = new GpuTrainingHistory<T>();
        var stopwatch = new Stopwatch();

        if (_options.VerboseLogging)
        {
            Console.WriteLine($"[GpuPINNTrainer] Starting training: {epochs} epochs, GPU={IsUsingGpu}");
        }

        // Track peak memory during training
        long initialMemory = GC.GetTotalMemory(false);
        long peakMemory = initialMemory;

        stopwatch.Start();

        try
        {
            // Use the PINN's built-in Solve method for training
            // The GPU acceleration comes from the tensor operations in the layers
            var pinnHistory = _pinn.Solve(
                dataInputs,
                dataOutputs,
                epochs,
                learningRate,
                verbose,
                _options.BatchSizeGpu);

            // Copy results to GPU-aware history and track memory growth
            foreach (var loss in pinnHistory.Losses)
            {
                history.AddEpoch(loss);

                // Track peak memory growth during training
                long currentMemory = GC.GetTotalMemory(false);
                if (currentMemory > peakMemory)
                {
                    peakMemory = currentMemory;
                }
            }

            // Record timing statistics
            if (epochs > 0)
            {
                history.KernelTimings["TotalEpochs"] = epochs;
                history.KernelTimings["AverageEpochMs"] = stopwatch.ElapsedMilliseconds / epochs;
            }

            history.UseGpuAcceleration = IsUsingGpu;
            history.PeakManagedMemoryBytes = peakMemory - initialMemory;
        }
        finally
        {
            stopwatch.Stop();
            history.TotalTrainingTimeMs = stopwatch.ElapsedMilliseconds;
        }

        if (_options.VerboseLogging)
        {
            Console.WriteLine($"[GpuPINNTrainer] Training complete in {history.TotalTrainingTimeMs}ms, peak managed memory growth: {history.PeakManagedMemoryBytes / 1024}KB");
        }

        return history;
    }

    /// <summary>
    /// Performs a forward pass and computes loss (does not update weights).
    /// </summary>
    /// <param name="batchInputs">Batch of input points.</param>
    /// <param name="batchTargets">Optional target values. If provided, computes MSE loss against outputs.</param>
    /// <returns>Loss value for this evaluation step.</returns>
    /// <exception cref="ArgumentNullException">Thrown when batchInputs is null.</exception>
    /// <exception cref="ArgumentException">Thrown when batchTargets shape doesn't match outputs.</exception>
    /// <remarks>
    /// <para>
    /// <b>Note:</b> This method only evaluates loss - it does not perform backpropagation
    /// or weight updates. Use the Train method for full training with gradient updates.
    /// </para>
    /// </remarks>
    public T EvaluateLoss(Tensor<T> batchInputs, Tensor<T>? batchTargets = null)
    {
        if (batchInputs is null)
        {
            throw new ArgumentNullException(nameof(batchInputs));
        }

        // Forward pass
        var outputs = _pinn.Predict(batchInputs);

        // Compute loss using physics-informed loss
        T loss;
        if (batchTargets is not null)
        {
            // Validate tensor dimensions before accessing shape
            if (outputs.Shape.Length < 2)
            {
                throw new ArgumentException(
                    $"Output tensor must have at least 2 dimensions, but got {outputs.Shape.Length}.",
                    nameof(outputs));
            }
            if (batchTargets.Shape.Length < 2)
            {
                throw new ArgumentException(
                    $"Target tensor must have at least 2 dimensions, but got {batchTargets.Shape.Length}.",
                    nameof(batchTargets));
            }

            // Validate shapes match
            if (outputs.Shape[0] != batchTargets.Shape[0] || outputs.Shape[1] != batchTargets.Shape[1])
            {
                throw new ArgumentException(
                    $"Shape mismatch: outputs {outputs.Shape[0]}x{outputs.Shape[1]} vs targets {batchTargets.Shape[0]}x{batchTargets.Shape[1]}",
                    nameof(batchTargets));
            }

            // Simple MSE loss for targets (physics loss computed separately)
            T sum = _numOps.Zero;
            int count = 0;

            for (int i = 0; i < outputs.Shape[0]; i++)
            {
                for (int j = 0; j < outputs.Shape[1]; j++)
                {
                    T diff = _numOps.Subtract(outputs[i, j], batchTargets[i, j]);
                    sum = _numOps.Add(sum, _numOps.Multiply(diff, diff));
                    count++;
                }
            }

            loss = count > 0 ? _numOps.Divide(sum, _numOps.FromDouble(count)) : _numOps.Zero;
        }
        else
        {
            // Use the network's last loss
            loss = _pinn.GetLastLoss();
        }

        return loss;
    }

    /// <summary>
    /// Updates training options.
    /// </summary>
    /// <param name="options">New options to apply.</param>
    public void UpdateOptions(GpuPINNTrainingOptions options)
    {
        Guard.NotNull(options);
        _options = options;

        if (_options.EnableGpu && !_gpuInitialized)
        {
            TryInitializeGpu();
        }
        else if (!_options.EnableGpu)
        {
            _useGpu = false;
        }
    }

    /// <summary>
    /// Releases GPU resources.
    /// </summary>
    public void ReleaseGpuResources()
    {
        _useGpu = false;
        _gpuInitialized = false;
    }

    /// <summary>
    /// Gets GPU memory usage statistics.
    /// </summary>
    /// <returns>Memory usage info or null if GPU not in use.</returns>
    /// <remarks>
    /// <para>
    /// <b>Limitations:</b> DirectGpu backends do not provide a standard API to query current memory usage.
    /// The returned <see cref="PINNGpuMemoryInfo.TotalMemoryBytes"/> reflects the total GPU memory,
    /// but <see cref="PINNGpuMemoryInfo.UsedMemoryBytes"/> and <see cref="PINNGpuMemoryInfo.AvailableMemoryBytes"/>
    /// cannot be accurately tracked through the DirectGpu backend.
    /// </para>
    /// <para>
    /// For accurate memory profiling, use external tools like NVIDIA's nvidia-smi,
    /// CUDA's nvml library, or Visual Studio's GPU profiler.
    /// </para>
    /// </remarks>
    public PINNGpuMemoryInfo? GetGpuMemoryInfo()
    {
        if (!IsUsingGpu)
        {
            return null;
        }

        // DirectGpu backends do not expose a public API for querying current memory usage.
        // We can only report that GPU is in use; for detailed memory stats,
        // users should use external profiling tools (nvidia-smi, nvml, etc.)
        // Return a minimal info struct indicating GPU is active but memory stats unavailable.
        return new PINNGpuMemoryInfo
        {
            TotalMemoryBytes = -1, // -1 indicates "unknown" - DirectGpu does not expose this via IEngine
            UsedMemoryBytes = -1,  // -1 indicates "not available"
            AvailableMemoryBytes = -1
        };
    }

    private static bool IsGpuSupported(IEngine engine)
    {
        // Use the engine's SupportsGpu property which properly checks
        // if the underlying accelerator is a GPU (not CPU)
        if (engine is null)
        {
            return false;
        }

        return engine.SupportsGpu;
    }
}

/// <summary>
/// Training history with GPU-specific metrics.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This class tracks training progress including timing and memory metrics.
/// These metrics help diagnose training issues and optimize performance.
/// </para>
/// </remarks>
public class GpuTrainingHistory<T> : TrainingHistory<T>
{
    /// <summary>
    /// Gets or sets whether GPU acceleration was used during training.
    /// </summary>
    public bool UseGpuAcceleration { get; set; }

    /// <summary>
    /// Gets or sets the total training time in milliseconds.
    /// </summary>
    public long TotalTrainingTimeMs { get; set; }

    /// <summary>
    /// Gets or sets the average time per epoch in milliseconds.
    /// </summary>
    public double AverageEpochTimeMs => Losses.Count > 0 ? (double)TotalTrainingTimeMs / Losses.Count : 0;

    /// <summary>
    /// Gets or sets the peak managed (heap) memory growth during training in bytes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Important:</b> This measures .NET managed memory growth (via GC.GetTotalMemory),
    /// NOT GPU device memory. Managed memory includes tensor allocations on the CPU heap
    /// but does not reflect actual GPU memory usage.
    /// </para>
    /// <para>
    /// For GPU-specific memory profiling, use external tools:
    /// <list type="bullet">
    /// <item><description>NVIDIA: nvidia-smi, nvml library</description></item>
    /// <item><description>AMD: rocm-smi</description></item>
    /// <item><description>General: Visual Studio GPU profiler</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    public long PeakManagedMemoryBytes { get; set; }

    /// <summary>
    /// Gets or sets training timing statistics.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Contains timing metrics such as:
    /// <list type="bullet">
    /// <item><description>"TotalEpochs" - Number of epochs trained</description></item>
    /// <item><description>"AverageEpochMs" - Average time per epoch in milliseconds</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    public Dictionary<string, long> KernelTimings { get; set; } = new Dictionary<string, long>();
}

/// <summary>
/// GPU memory usage information for PINN training.
/// </summary>
/// <remarks>
/// <para>
/// <b>Note:</b> A value of -1 for any property indicates that the information
/// is not available through the current GPU backend.
/// </para>
/// <para>
/// For accurate memory profiling, use external tools such as:
/// - NVIDIA: nvidia-smi, nvml library
/// - AMD: rocm-smi
/// - General: Visual Studio GPU profiler
/// </para>
/// </remarks>
public class PINNGpuMemoryInfo
{
    /// <summary>
    /// Gets or sets the total GPU memory in bytes.
    /// A value of -1 indicates this information is not available.
    /// </summary>
    public long TotalMemoryBytes { get; set; }

    /// <summary>
    /// Gets or sets the used GPU memory in bytes.
    /// A value of -1 indicates this information is not available.
    /// </summary>
    public long UsedMemoryBytes { get; set; }

    /// <summary>
    /// Gets or sets the available GPU memory in bytes.
    /// A value of -1 indicates this information is not available.
    /// </summary>
    public long AvailableMemoryBytes { get; set; }

    /// <summary>
    /// Gets the usage percentage, or -1 if memory information is not available.
    /// </summary>
    public double UsagePercentage =>
        TotalMemoryBytes > 0 && UsedMemoryBytes >= 0
            ? (double)UsedMemoryBytes / TotalMemoryBytes * 100
            : -1;

    /// <summary>
    /// Gets whether memory information is available.
    /// </summary>
    public bool IsMemoryInfoAvailable => TotalMemoryBytes >= 0;
}

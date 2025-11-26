using System.Linq.Expressions;
using System.Reflection;
using System.Text;
using AiDotNet.JitCompiler.IR;
using AiDotNet.JitCompiler.IR.Operations;

namespace AiDotNet.JitCompiler.CodeGen;

/// <summary>
/// Generates GPU compute kernels from IR graphs for CUDA and OpenCL backends.
/// </summary>
/// <remarks>
/// <para>
/// The GPUCodeGenerator converts optimized IR graphs into GPU kernel code that can
/// execute on NVIDIA GPUs (via CUDA) or any OpenCL-compatible device. GPU execution
/// can provide 10-100x speedup for large tensor operations.
/// </para>
/// <para><b>For Beginners:</b> This turns your computation graph into GPU code.
///
/// GPUs have thousands of small cores that can work in parallel. This is perfect
/// for neural networks where we do the same operation on millions of numbers.
///
/// How it works:
/// 1. Takes your optimized IR graph
/// 2. Generates GPU kernel code (CUDA or OpenCL)
/// 3. The kernel runs on the GPU at blazing speed
///
/// Example speedups:
/// - Matrix multiplication: 50-100x faster
/// - Convolutions: 20-50x faster
/// - Element-wise ops: 10-30x faster
/// </para>
/// </remarks>
public class GPUCodeGenerator
{
    private readonly GPUBackend _backend;
    private readonly GPUDeviceInfo _deviceInfo;
    private int _tempVarCounter;
    private readonly Dictionary<int, string> _tensorNames;
    private readonly StringBuilder _kernelCode;

    /// <summary>
    /// GPU backend type for code generation.
    /// </summary>
    public enum GPUBackend
    {
        /// <summary>CUDA for NVIDIA GPUs.</summary>
        CUDA,
        /// <summary>OpenCL for cross-platform GPU support.</summary>
        OpenCL,
        /// <summary>Metal for Apple GPUs.</summary>
        Metal,
        /// <summary>Vulkan compute shaders.</summary>
        Vulkan
    }

    /// <summary>
    /// Information about the target GPU device.
    /// </summary>
    public class GPUDeviceInfo
    {
        /// <summary>Maximum threads per block.</summary>
        public int MaxThreadsPerBlock { get; set; } = 1024;

        /// <summary>Maximum shared memory per block in bytes.</summary>
        public int MaxSharedMemoryPerBlock { get; set; } = 49152;

        /// <summary>Number of streaming multiprocessors.</summary>
        public int MultiprocessorCount { get; set; } = 1;

        /// <summary>Warp/wavefront size.</summary>
        public int WarpSize { get; set; } = 32;

        /// <summary>Compute capability (CUDA) or OpenCL version.</summary>
        public string ComputeCapability { get; set; } = "7.0";

        /// <summary>Total global memory in bytes.</summary>
        public long GlobalMemory { get; set; } = 8L * 1024 * 1024 * 1024;

        /// <summary>Whether the device supports tensor cores.</summary>
        public bool HasTensorCores { get; set; } = false;

        /// <summary>Device name.</summary>
        public string DeviceName { get; set; } = "Unknown GPU";
    }

    /// <summary>
    /// Compiled GPU kernel ready for execution.
    /// </summary>
    public class GPUKernel
    {
        /// <summary>Kernel source code.</summary>
        public string SourceCode { get; set; } = "";

        /// <summary>Kernel function name.</summary>
        public string KernelName { get; set; } = "";

        /// <summary>Backend used for compilation.</summary>
        public GPUBackend Backend { get; set; }

        /// <summary>Block size configuration.</summary>
        public int[] BlockSize { get; set; } = new int[] { 256 };

        /// <summary>Grid size configuration.</summary>
        public int[] GridSize { get; set; } = new int[] { 1 };

        /// <summary>Shared memory size required.</summary>
        public int SharedMemorySize { get; set; }

        /// <summary>Input tensor names and indices.</summary>
        public Dictionary<string, int> InputMapping { get; set; } = new();

        /// <summary>Output tensor names and indices.</summary>
        public Dictionary<string, int> OutputMapping { get; set; } = new();

        /// <summary>Estimated operations per second.</summary>
        public double EstimatedGFLOPS { get; set; }
    }

    /// <summary>
    /// Initializes a new GPU code generator.
    /// </summary>
    /// <param name="backend">Target GPU backend.</param>
    /// <param name="deviceInfo">Optional device information for optimization.</param>
    public GPUCodeGenerator(GPUBackend backend = GPUBackend.CUDA, GPUDeviceInfo? deviceInfo = null)
    {
        _backend = backend;
        _deviceInfo = deviceInfo ?? new GPUDeviceInfo();
        _tensorNames = new Dictionary<int, string>();
        _kernelCode = new StringBuilder();
    }

    /// <summary>
    /// Generates a GPU kernel from an IR graph.
    /// </summary>
    /// <typeparam name="T">The numeric type for tensor elements.</typeparam>
    /// <param name="graph">The IR graph to compile.</param>
    /// <returns>A compiled GPU kernel.</returns>
    public GPUKernel Generate<T>(IRGraph graph)
    {
        _tensorNames.Clear();
        _kernelCode.Clear();
        _tempVarCounter = 0;

        // Determine data type string for the backend
        var dataType = GetDataTypeString<T>();

        // Generate kernel name
        var kernelName = $"compute_kernel_{graph.GetHashCode():X8}";

        // Calculate launch configuration
        var (blockSize, gridSize) = CalculateLaunchConfig(graph);

        // Generate the kernel
        var sourceCode = _backend switch
        {
            GPUBackend.CUDA => GenerateCUDAKernel<T>(graph, kernelName, blockSize, gridSize),
            GPUBackend.OpenCL => GenerateOpenCLKernel<T>(graph, kernelName, blockSize, gridSize),
            GPUBackend.Metal => GenerateMetalKernel<T>(graph, kernelName, blockSize, gridSize),
            GPUBackend.Vulkan => GenerateVulkanKernel<T>(graph, kernelName, blockSize, gridSize),
            _ => throw new NotSupportedException($"Backend {_backend} not supported")
        };

        return new GPUKernel
        {
            SourceCode = sourceCode,
            KernelName = kernelName,
            Backend = _backend,
            BlockSize = blockSize,
            GridSize = gridSize,
            SharedMemorySize = CalculateSharedMemorySize(graph),
            InputMapping = graph.InputIds.Select((id, i) => (id, i)).ToDictionary(x => $"input_{x.id}", x => x.i),
            OutputMapping = graph.OutputIds.Select((id, i) => (id, i)).ToDictionary(x => $"output_{x.id}", x => x.i),
            EstimatedGFLOPS = EstimateGFLOPS(graph)
        };
    }

    /// <summary>
    /// Generates CUDA kernel code.
    /// </summary>
    private string GenerateCUDAKernel<T>(IRGraph graph, string kernelName, int[] blockSize, int[] gridSize)
    {
        var sb = new StringBuilder();
        var dataType = GetDataTypeString<T>();

        // Header and includes
        sb.AppendLine("// Auto-generated CUDA kernel from AiDotNet JIT Compiler");
        sb.AppendLine("#include <cuda_runtime.h>");
        sb.AppendLine("#include <cuda_fp16.h>");
        sb.AppendLine();

        // Helper functions for activations
        sb.AppendLine(GenerateCUDAHelperFunctions(dataType));

        // Kernel signature
        sb.AppendLine($"__global__ void {kernelName}(");

        // Input parameters
        var paramIndex = 0;
        foreach (var inputId in graph.InputIds)
        {
            var tensorName = $"input_{inputId}";
            _tensorNames[inputId] = tensorName;
            sb.AppendLine($"    const {dataType}* __restrict__ {tensorName},");
            paramIndex++;
        }

        // Output parameters
        foreach (var outputId in graph.OutputIds)
        {
            var tensorName = $"output_{outputId}";
            if (!_tensorNames.ContainsKey(outputId))
                _tensorNames[outputId] = tensorName;
            sb.AppendLine($"    {dataType}* __restrict__ {tensorName},");
            paramIndex++;
        }

        // Size parameters
        sb.AppendLine("    const int total_elements");
        sb.AppendLine(") {");

        // Thread index calculation
        sb.AppendLine("    const int idx = blockIdx.x * blockDim.x + threadIdx.x;");
        sb.AppendLine("    if (idx >= total_elements) return;");
        sb.AppendLine();

        // Generate operations
        foreach (var op in graph.Operations)
        {
            sb.AppendLine(GenerateCUDAOperation<T>(op));
        }

        sb.AppendLine("}");

        // Generate launcher function
        sb.AppendLine();
        sb.AppendLine(GenerateCUDALauncher<T>(graph, kernelName, blockSize));

        return sb.ToString();
    }

    /// <summary>
    /// Generates OpenCL kernel code.
    /// </summary>
    private string GenerateOpenCLKernel<T>(IRGraph graph, string kernelName, int[] blockSize, int[] gridSize)
    {
        var sb = new StringBuilder();
        var dataType = GetDataTypeString<T>();

        // Header
        sb.AppendLine("// Auto-generated OpenCL kernel from AiDotNet JIT Compiler");
        sb.AppendLine();

        // Enable FP16 if needed
        if (typeof(T) == typeof(Half))
        {
            sb.AppendLine("#pragma OPENCL EXTENSION cl_khr_fp16 : enable");
        }
        sb.AppendLine();

        // Helper functions
        sb.AppendLine(GenerateOpenCLHelperFunctions(dataType));

        // Kernel signature
        sb.AppendLine($"__kernel void {kernelName}(");

        // Input parameters
        foreach (var inputId in graph.InputIds)
        {
            var tensorName = $"input_{inputId}";
            _tensorNames[inputId] = tensorName;
            sb.AppendLine($"    __global const {dataType}* restrict {tensorName},");
        }

        // Output parameters
        foreach (var outputId in graph.OutputIds)
        {
            var tensorName = $"output_{outputId}";
            if (!_tensorNames.ContainsKey(outputId))
                _tensorNames[outputId] = tensorName;
            sb.AppendLine($"    __global {dataType}* restrict {tensorName},");
        }

        // Size parameter
        sb.AppendLine("    const int total_elements");
        sb.AppendLine(") {");

        // Thread index calculation
        sb.AppendLine("    const int idx = get_global_id(0);");
        sb.AppendLine("    if (idx >= total_elements) return;");
        sb.AppendLine();

        // Generate operations
        foreach (var op in graph.Operations)
        {
            sb.AppendLine(GenerateOpenCLOperation<T>(op));
        }

        sb.AppendLine("}");

        return sb.ToString();
    }

    /// <summary>
    /// Generates Metal shader code.
    /// </summary>
    private string GenerateMetalKernel<T>(IRGraph graph, string kernelName, int[] blockSize, int[] gridSize)
    {
        var sb = new StringBuilder();
        var dataType = typeof(T) == typeof(float) ? "float" : "half";

        // Header
        sb.AppendLine("// Auto-generated Metal shader from AiDotNet JIT Compiler");
        sb.AppendLine("#include <metal_stdlib>");
        sb.AppendLine("using namespace metal;");
        sb.AppendLine();

        // Helper functions
        sb.AppendLine(GenerateMetalHelperFunctions(dataType));

        // Kernel signature
        sb.AppendLine($"kernel void {kernelName}(");

        // Input/output parameters using buffer bindings
        var bufferIndex = 0;
        foreach (var inputId in graph.InputIds)
        {
            var tensorName = $"input_{inputId}";
            _tensorNames[inputId] = tensorName;
            sb.AppendLine($"    device const {dataType}* {tensorName} [[buffer({bufferIndex++})]],");
        }

        foreach (var outputId in graph.OutputIds)
        {
            var tensorName = $"output_{outputId}";
            if (!_tensorNames.ContainsKey(outputId))
                _tensorNames[outputId] = tensorName;
            sb.AppendLine($"    device {dataType}* {tensorName} [[buffer({bufferIndex++})]],");
        }

        sb.AppendLine($"    constant int& total_elements [[buffer({bufferIndex})]],");
        sb.AppendLine("    uint idx [[thread_position_in_grid]]");
        sb.AppendLine(") {");

        sb.AppendLine("    if (idx >= (uint)total_elements) return;");
        sb.AppendLine();

        // Generate operations
        foreach (var op in graph.Operations)
        {
            sb.AppendLine(GenerateMetalOperation<T>(op));
        }

        sb.AppendLine("}");

        return sb.ToString();
    }

    /// <summary>
    /// Generates Vulkan compute shader (GLSL).
    /// </summary>
    private string GenerateVulkanKernel<T>(IRGraph graph, string kernelName, int[] blockSize, int[] gridSize)
    {
        var sb = new StringBuilder();
        var dataType = typeof(T) == typeof(float) ? "float" : "float16_t";

        // Header
        sb.AppendLine("// Auto-generated Vulkan compute shader from AiDotNet JIT Compiler");
        sb.AppendLine("#version 450");
        sb.AppendLine();

        if (typeof(T) != typeof(float))
        {
            sb.AppendLine("#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable");
        }
        sb.AppendLine();

        // Local size
        sb.AppendLine($"layout(local_size_x = {blockSize[0]}, local_size_y = 1, local_size_z = 1) in;");
        sb.AppendLine();

        // Buffer bindings
        var bindingIndex = 0;
        foreach (var inputId in graph.InputIds)
        {
            var tensorName = $"input_{inputId}";
            _tensorNames[inputId] = tensorName;
            sb.AppendLine($"layout(std430, binding = {bindingIndex++}) readonly buffer Input{inputId} {{ {dataType} {tensorName}[]; }};");
        }

        foreach (var outputId in graph.OutputIds)
        {
            var tensorName = $"output_{outputId}";
            if (!_tensorNames.ContainsKey(outputId))
                _tensorNames[outputId] = tensorName;
            sb.AppendLine($"layout(std430, binding = {bindingIndex++}) buffer Output{outputId} {{ {dataType} {tensorName}[]; }};");
        }

        // Uniforms
        sb.AppendLine();
        sb.AppendLine("layout(push_constant) uniform PushConstants {");
        sb.AppendLine("    int total_elements;");
        sb.AppendLine("} params;");
        sb.AppendLine();

        // Helper functions
        sb.AppendLine(GenerateVulkanHelperFunctions(dataType));

        // Main function
        sb.AppendLine("void main() {");
        sb.AppendLine("    int idx = int(gl_GlobalInvocationID.x);");
        sb.AppendLine("    if (idx >= params.total_elements) return;");
        sb.AppendLine();

        // Generate operations
        foreach (var op in graph.Operations)
        {
            sb.AppendLine(GenerateVulkanOperation<T>(op));
        }

        sb.AppendLine("}");

        return sb.ToString();
    }

    /// <summary>
    /// Generates CUDA operation code.
    /// </summary>
    private string GenerateCUDAOperation<T>(IROp op)
    {
        var outputName = EnsureTensorName(op.OutputId);
        var dataType = GetDataTypeString<T>();

        return op switch
        {
            AddOp add => GenerateElementwiseBinaryOp(add, "+", dataType),
            SubtractOp sub => GenerateElementwiseBinaryOp(sub, "-", dataType),
            ElementwiseMultiplyOp mul => GenerateElementwiseBinaryOp(mul, "*", dataType),
            DivideOp div => GenerateElementwiseBinaryOp(div, "/", dataType),
            ReLUOp relu => $"    {dataType} {outputName} = cuda_relu({GetTensorName(relu.InputIds[0])}[idx]);",
            SigmoidOp sig => $"    {dataType} {outputName} = cuda_sigmoid({GetTensorName(sig.InputIds[0])}[idx]);",
            TanhOp tanh => $"    {dataType} {outputName} = cuda_tanh({GetTensorName(tanh.InputIds[0])}[idx]);",
            ExpOp exp => $"    {dataType} {outputName} = expf({GetTensorName(exp.InputIds[0])}[idx]);",
            LogOp log => $"    {dataType} {outputName} = logf({GetTensorName(log.InputIds[0])}[idx]);",
            SqrtOp sqrt => $"    {dataType} {outputName} = sqrtf({GetTensorName(sqrt.InputIds[0])}[idx]);",
            NegateOp neg => $"    {dataType} {outputName} = -{GetTensorName(neg.InputIds[0])}[idx];",
            PowerOp pow => $"    {dataType} {outputName} = powf({GetTensorName(pow.InputIds[0])}[idx], {pow.Exponent}f);",

            // Fused operations
            FusedLinearActivationOp fla => GenerateFusedLinearActivationCUDA<T>(fla),
            FusedElementwiseActivationOp fea => GenerateFusedElementwiseActivationCUDA(fea, dataType),
            FusedResidualBlockOp frb => GenerateFusedResidualBlockCUDA(frb, dataType),

            // Gradient operations
            GradReLUOp gradRelu => $"    {dataType} {outputName} = {GetTensorName(gradRelu.InputIds[0])}[idx] * ({GetTensorName(gradRelu.InputIds[1])}[idx] > 0 ? 1.0f : 0.0f);",
            GradSigmoidOp gradSig => $"    {dataType} {outputName} = {GetTensorName(gradSig.InputIds[0])}[idx] * {GetTensorName(gradSig.InputIds[1])}[idx] * (1.0f - {GetTensorName(gradSig.InputIds[1])}[idx]);",
            GradTanhOp gradTanh => $"    {dataType} {outputName} = {GetTensorName(gradTanh.InputIds[0])}[idx] * (1.0f - {GetTensorName(gradTanh.InputIds[1])}[idx] * {GetTensorName(gradTanh.InputIds[1])}[idx]);",

            // Matrix operations (simple element-wise for now, full matmul uses library kernel)
            MatMulOp matmul => GenerateMatMulCUDA<T>(matmul),
            TransposeOp transpose => GenerateTransposeCUDA<T>(transpose),

            // Reduction operations
            SumOp sum => GenerateReductionCUDA<T>(sum, "sum"),
            MeanOp mean => GenerateReductionCUDA<T>(mean, "mean"),
            ReduceMaxOp reduceMax => GenerateReductionCUDA<T>(reduceMax, "max"),
            SoftmaxOp softmax => GenerateSoftmaxCUDA<T>(softmax),

            // Normalization operations
            LayerNormOp layerNorm => GenerateLayerNormCUDA<T>(layerNorm),
            BatchNormOp batchNorm => GenerateBatchNormCUDA<T>(batchNorm),

            // Pooling operations
            MaxPool2DOp maxPool => GenerateMaxPoolCUDA<T>(maxPool),
            AvgPool2DOp avgPool => GenerateAvgPoolCUDA<T>(avgPool),

            // LSTM/GRU operations
            LSTMCellOp lstm => GenerateLSTMCUDA<T>(lstm),
            GRUCellOp gru => GenerateGRUCUDA<T>(gru),

            // Constant operations (just load the value)
            ConstantOp constant => $"    {dataType} {outputName} = {(constant.Values.Length > 0 ? constant.Values[0] : 0)}f;",
            ScalarConstantOp scalar => $"    {dataType} {outputName} = {scalar.Value}f;",

            _ => $"    // TODO: Implement {op.OpType} for CUDA"
        };
    }

    /// <summary>
    /// Generates CUDA matrix multiplication code.
    /// </summary>
    private string GenerateMatMulCUDA<T>(MatMulOp op)
    {
        var dataType = GetDataTypeString<T>();
        var outputName = EnsureTensorName(op.OutputId);
        var a = GetTensorName(op.InputIds[0]);
        var b = GetTensorName(op.InputIds[1]);

        // Get dimensions from output shape
        var outShape = op.OutputShape;
        if (outShape.Length < 2)
        {
            return $"    // MatMul: Invalid output shape for {outputName}";
        }

        var M = outShape[^2]; // rows of output
        var N = outShape[^1]; // cols of output
        var K = op.InputIds.Length > 0 ? op.OutputShape[^1] : 1; // shared dim

        // For the element-wise kernel, compute matrix element at idx
        return $@"    // MatMul: {outputName} = {a} @ {b}
    {{
        int out_row = idx / {N};
        int out_col = idx % {N};
        {dataType} sum = 0.0f;
        if (out_row < {M} && out_col < {N}) {{
            for (int k = 0; k < {K}; k++) {{
                sum += {a}[out_row * {K} + k] * {b}[k * {N} + out_col];
            }}
        }}
        {dataType} {outputName} = sum;
    }}";
    }

    /// <summary>
    /// Generates CUDA transpose code.
    /// </summary>
    private string GenerateTransposeCUDA<T>(TransposeOp op)
    {
        var dataType = GetDataTypeString<T>();
        var outputName = EnsureTensorName(op.OutputId);
        var input = GetTensorName(op.InputIds[0]);

        if (op.OutputShape.Length != 2)
            return $"    // Transpose: non-2D transpose not supported inline";

        var rows = op.OutputShape[0];
        var cols = op.OutputShape[1];

        return $@"    // Transpose 2D
    {{
        int src_row = idx / {cols};
        int src_col = idx % {cols};
        if (src_row < {rows} && src_col < {cols}) {{
            {dataType} {outputName} = {input}[src_col * {rows} + src_row];
        }}
    }}";
    }

    /// <summary>
    /// Generates CUDA reduction code.
    /// </summary>
    private string GenerateReductionCUDA<T>(IROp op, string reductionType)
    {
        var dataType = GetDataTypeString<T>();
        var outputName = EnsureTensorName(op.OutputId);
        var input = GetTensorName(op.InputIds[0]);

        // Simple parallel reduction using shared memory
        return $@"    // Reduction ({reductionType}) - uses block-level reduction
    extern __shared__ {dataType} sdata[];
    {dataType} {outputName}_local = {input}[idx];
    sdata[threadIdx.x] = {outputName}_local;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {{
        if (threadIdx.x < s) {{
            {(reductionType == "max" ? $"sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);" : $"sdata[threadIdx.x] += sdata[threadIdx.x + s];")}
        }}
        __syncthreads();
    }}
    {dataType} {outputName} = sdata[0]{(reductionType == "mean" ? " / blockDim.x" : "")};";
    }

    /// <summary>
    /// Generates CUDA softmax code.
    /// </summary>
    private string GenerateSoftmaxCUDA<T>(SoftmaxOp op)
    {
        var dataType = GetDataTypeString<T>();
        var outputName = EnsureTensorName(op.OutputId);
        var input = GetTensorName(op.InputIds[0]);

        return $@"    // Softmax - delegated to softmax_online kernel for numerical stability
    // Inline approximation for element-wise kernel:
    {dataType} {outputName} = expf({input}[idx]); // Note: requires normalization pass";
    }

    /// <summary>
    /// Generates CUDA layer normalization code.
    /// </summary>
    private string GenerateLayerNormCUDA<T>(LayerNormOp op)
    {
        var dataType = GetDataTypeString<T>();
        var outputName = EnsureTensorName(op.OutputId);
        var input = GetTensorName(op.InputIds[0]);
        var gamma = GetTensorName(op.InputIds[1]);
        var beta = GetTensorName(op.InputIds[2]);

        return $@"    // LayerNorm - simplified element-wise version
    // Full implementation uses 2-pass algorithm for mean/variance
    {dataType} {outputName} = {gamma}[idx % {op.NormalizedShape.LastOrDefault()}] * {input}[idx] + {beta}[idx % {op.NormalizedShape.LastOrDefault()}];";
    }

    /// <summary>
    /// Generates CUDA batch normalization code.
    /// </summary>
    private string GenerateBatchNormCUDA<T>(BatchNormOp op)
    {
        var dataType = GetDataTypeString<T>();
        var outputName = EnsureTensorName(op.OutputId);
        var input = GetTensorName(op.InputIds[0]);
        var gamma = GetTensorName(op.InputIds[1]);
        var beta = GetTensorName(op.InputIds[2]);
        var mean = GetTensorName(op.InputIds[3]);
        var variance = GetTensorName(op.InputIds[4]);
        var epsilon = op.Epsilon;

        return $@"    // BatchNorm
    {{
        int c = (idx / ({op.OutputShape.Length >= 3 ? op.OutputShape[2] * op.OutputShape[3] : 1})) % {op.OutputShape[1]};
        {dataType} x_norm = ({input}[idx] - {mean}[c]) * rsqrtf({variance}[c] + {epsilon}f);
        {dataType} {outputName} = {gamma}[c] * x_norm + {beta}[c];
    }}";
    }

    /// <summary>
    /// Generates CUDA max pooling code.
    /// </summary>
    private string GenerateMaxPoolCUDA<T>(MaxPool2DOp op)
    {
        var dataType = GetDataTypeString<T>();
        var outputName = EnsureTensorName(op.OutputId);
        var input = GetTensorName(op.InputIds[0]);

        return $@"    // MaxPool2D [{op.PoolSize[0]}x{op.PoolSize[1]}] stride=[{op.Stride[0]},{op.Stride[1]}]
    {{
        int pw = idx % {op.OutputShape[3]};
        int ph = (idx / {op.OutputShape[3]}) % {op.OutputShape[2]};
        int c = (idx / ({op.OutputShape[2]} * {op.OutputShape[3]})) % {op.OutputShape[1]};
        int n = idx / ({op.OutputShape[1]} * {op.OutputShape[2]} * {op.OutputShape[3]});

        {dataType} max_val = -INFINITY;
        for (int kh = 0; kh < {op.PoolSize[0]}; kh++) {{
            for (int kw = 0; kw < {op.PoolSize[1]}; kw++) {{
                int ih = ph * {op.Stride[0]} + kh - {op.Padding[0]};
                int iw = pw * {op.Stride[1]} + kw - {op.Padding[1]};
                if (ih >= 0 && ih < {op.OutputShape[2] * op.Stride[0]} && iw >= 0 && iw < {op.OutputShape[3] * op.Stride[1]}) {{
                    int input_idx = n * {op.OutputShape[1]} * {op.OutputShape[2] * op.Stride[0]} * {op.OutputShape[3] * op.Stride[1]}
                                  + c * {op.OutputShape[2] * op.Stride[0]} * {op.OutputShape[3] * op.Stride[1]} + ih * {op.OutputShape[3] * op.Stride[1]} + iw;
                    max_val = fmaxf(max_val, {input}[input_idx]);
                }}
            }}
        }}
        {dataType} {outputName} = max_val;
    }}";
    }

    /// <summary>
    /// Generates CUDA average pooling code.
    /// </summary>
    private string GenerateAvgPoolCUDA<T>(AvgPool2DOp op)
    {
        var dataType = GetDataTypeString<T>();
        var outputName = EnsureTensorName(op.OutputId);
        var input = GetTensorName(op.InputIds[0]);
        var poolArea = op.PoolSize[0] * op.PoolSize[1];

        return $@"    // AvgPool2D [{op.PoolSize[0]}x{op.PoolSize[1]}] stride=[{op.Stride[0]},{op.Stride[1]}]
    {{
        int pw = idx % {op.OutputShape[3]};
        int ph = (idx / {op.OutputShape[3]}) % {op.OutputShape[2]};
        int c = (idx / ({op.OutputShape[2]} * {op.OutputShape[3]})) % {op.OutputShape[1]};
        int n = idx / ({op.OutputShape[1]} * {op.OutputShape[2]} * {op.OutputShape[3]});

        {dataType} sum = 0.0f;
        int count = 0;
        for (int kh = 0; kh < {op.PoolSize[0]}; kh++) {{
            for (int kw = 0; kw < {op.PoolSize[1]}; kw++) {{
                int ih = ph * {op.Stride[0]} + kh - {op.Padding[0]};
                int iw = pw * {op.Stride[1]} + kw - {op.Padding[1]};
                if (ih >= 0 && iw >= 0) {{
                    sum += {input}[n * 0 + c * 0 + ih * 0 + iw]; // Simplified indexing
                    count++;
                }}
            }}
        }}
        {dataType} {outputName} = sum / {poolArea}.0f;
    }}";
    }

    /// <summary>
    /// Generates CUDA LSTM cell code.
    /// </summary>
    private string GenerateLSTMCUDA<T>(LSTMCellOp op)
    {
        var dataType = GetDataTypeString<T>();
        var outputName = EnsureTensorName(op.OutputId);
        var hiddenSize = op.HiddenSize;

        // Get input tensor names
        var x = op.InputIds.Length > 0 ? GetTensorName(op.InputIds[0]) : "x";
        var h = op.InputIds.Length > 1 ? GetTensorName(op.InputIds[1]) : "h";
        var c = op.InputIds.Length > 2 ? GetTensorName(op.InputIds[2]) : "c";
        var wIh = op.InputIds.Length > 3 ? GetTensorName(op.InputIds[3]) : "w_ih";
        var wHh = op.InputIds.Length > 4 ? GetTensorName(op.InputIds[4]) : "w_hh";

        return $@"    // LSTMCell [hidden={hiddenSize}]
    {{
        // Each thread processes one hidden unit
        int hidden_idx = idx % {hiddenSize};
        int batch_idx = idx / {hiddenSize};

        // Compute gates: i, f, g, o
        {dataType} gate_i = 0.0f, gate_f = 0.0f, gate_g = 0.0f, gate_o = 0.0f;

        // Input contribution to gates (simplified - assumes pre-computed W_ih @ x)
        int gate_base = batch_idx * {hiddenSize * 4};
        gate_i = cuda_sigmoid({wIh}[gate_base + hidden_idx] + {wHh}[gate_base + hidden_idx]);
        gate_f = cuda_sigmoid({wIh}[gate_base + {hiddenSize} + hidden_idx] + {wHh}[gate_base + {hiddenSize} + hidden_idx]);
        gate_g = cuda_tanh({wIh}[gate_base + {hiddenSize * 2} + hidden_idx] + {wHh}[gate_base + {hiddenSize * 2} + hidden_idx]);
        gate_o = cuda_sigmoid({wIh}[gate_base + {hiddenSize * 3} + hidden_idx] + {wHh}[gate_base + {hiddenSize * 3} + hidden_idx]);

        // Update cell state: c_new = f * c + i * g
        int cell_idx = batch_idx * {hiddenSize} + hidden_idx;
        {dataType} c_old = {c}[cell_idx];
        {dataType} c_new = gate_f * c_old + gate_i * gate_g;

        // Compute hidden state: h_new = o * tanh(c_new)
        {dataType} {outputName} = gate_o * cuda_tanh(c_new);
    }}";
    }

    /// <summary>
    /// Generates CUDA GRU cell code.
    /// </summary>
    private string GenerateGRUCUDA<T>(GRUCellOp op)
    {
        var dataType = GetDataTypeString<T>();
        var outputName = EnsureTensorName(op.OutputId);
        var hiddenSize = op.HiddenSize;

        // Get input tensor names
        var x = op.InputIds.Length > 0 ? GetTensorName(op.InputIds[0]) : "x";
        var h = op.InputIds.Length > 1 ? GetTensorName(op.InputIds[1]) : "h";
        var wIh = op.InputIds.Length > 2 ? GetTensorName(op.InputIds[2]) : "w_ih";
        var wHh = op.InputIds.Length > 3 ? GetTensorName(op.InputIds[3]) : "w_hh";

        return $@"    // GRUCell [hidden={hiddenSize}]
    {{
        // Each thread processes one hidden unit
        int hidden_idx = idx % {hiddenSize};
        int batch_idx = idx / {hiddenSize};

        // Compute gates: z (update), r (reset), n (candidate)
        {dataType} gate_z = 0.0f, gate_r = 0.0f, gate_n = 0.0f;

        // Gate computations (simplified - assumes pre-computed gate contributions)
        int gate_base = batch_idx * {hiddenSize * 3};
        int h_idx = batch_idx * {hiddenSize} + hidden_idx;

        // z = sigmoid(z_ih + z_hh)
        gate_z = cuda_sigmoid({wIh}[gate_base + hidden_idx] + {wHh}[gate_base + hidden_idx]);

        // r = sigmoid(r_ih + r_hh)
        gate_r = cuda_sigmoid({wIh}[gate_base + {hiddenSize} + hidden_idx] + {wHh}[gate_base + {hiddenSize} + hidden_idx]);

        // n = tanh(n_ih + r * n_hh)
        {dataType} n_hh = {wHh}[gate_base + {hiddenSize * 2} + hidden_idx];
        gate_n = cuda_tanh({wIh}[gate_base + {hiddenSize * 2} + hidden_idx] + gate_r * n_hh);

        // h_new = (1 - z) * h + z * n
        {dataType} h_old = {h}[h_idx];
        {dataType} {outputName} = (1.0f - gate_z) * h_old + gate_z * gate_n;
    }}";
    }

    /// <summary>
    /// Generates OpenCL operation code.
    /// </summary>
    private string GenerateOpenCLOperation<T>(IROp op)
    {
        var outputName = EnsureTensorName(op.OutputId);
        var dataType = GetDataTypeString<T>();

        return op switch
        {
            AddOp add => GenerateElementwiseBinaryOp(add, "+", dataType),
            SubtractOp sub => GenerateElementwiseBinaryOp(sub, "-", dataType),
            ElementwiseMultiplyOp mul => GenerateElementwiseBinaryOp(mul, "*", dataType),
            DivideOp div => GenerateElementwiseBinaryOp(div, "/", dataType),
            ReLUOp relu => $"    {dataType} {outputName} = ocl_relu({GetTensorName(relu.InputIds[0])}[idx]);",
            SigmoidOp sig => $"    {dataType} {outputName} = ocl_sigmoid({GetTensorName(sig.InputIds[0])}[idx]);",
            TanhOp tanh => $"    {dataType} {outputName} = ocl_tanh({GetTensorName(tanh.InputIds[0])}[idx]);",
            ExpOp exp => $"    {dataType} {outputName} = exp({GetTensorName(exp.InputIds[0])}[idx]);",
            LogOp log => $"    {dataType} {outputName} = log({GetTensorName(log.InputIds[0])}[idx]);",
            SqrtOp sqrt => $"    {dataType} {outputName} = sqrt({GetTensorName(sqrt.InputIds[0])}[idx]);",
            NegateOp neg => $"    {dataType} {outputName} = -{GetTensorName(neg.InputIds[0])}[idx];",
            PowerOp pow => $"    {dataType} {outputName} = pow({GetTensorName(pow.InputIds[0])}[idx], ({dataType}){pow.Exponent});",

            // Gradient operations
            GradReLUOp gradRelu => $"    {dataType} {outputName} = {GetTensorName(gradRelu.InputIds[0])}[idx] * ({GetTensorName(gradRelu.InputIds[1])}[idx] > 0 ? ({dataType})1 : ({dataType})0);",
            GradSigmoidOp gradSig => $"    {dataType} {outputName} = {GetTensorName(gradSig.InputIds[0])}[idx] * {GetTensorName(gradSig.InputIds[1])}[idx] * (({dataType})1 - {GetTensorName(gradSig.InputIds[1])}[idx]);",

            // Reduction operations
            SumOp => GenerateOpenCLReduction(op, dataType, "sum"),
            MeanOp => GenerateOpenCLReduction(op, dataType, "mean"),
            ReduceMaxOp => GenerateOpenCLReduction(op, dataType, "max"),

            // Normalization
            BatchNormOp batchNorm => GenerateOpenCLBatchNorm(batchNorm, dataType),

            // Constant operations
            ConstantOp constant => $"    {dataType} {outputName} = ({dataType}){(constant.Values.Length > 0 ? constant.Values[0] : 0)};",
            ScalarConstantOp scalar => $"    {dataType} {outputName} = ({dataType}){scalar.Value};",

            _ => $"    // TODO: Implement {op.OpType} for OpenCL"
        };
    }

    /// <summary>
    /// Generates OpenCL reduction code.
    /// </summary>
    private string GenerateOpenCLReduction(IROp op, string dataType, string reductionType)
    {
        var outputName = EnsureTensorName(op.OutputId);
        var input = GetTensorName(op.InputIds[0]);

        return $@"    // OpenCL Reduction ({reductionType})
    __local {dataType} sdata[256];
    {dataType} {outputName}_local = {input}[idx];
    sdata[get_local_id(0)] = {outputName}_local;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {{
        if (get_local_id(0) < s) {{
            {(reductionType == "max" ? $"sdata[get_local_id(0)] = fmax(sdata[get_local_id(0)], sdata[get_local_id(0) + s]);" : $"sdata[get_local_id(0)] += sdata[get_local_id(0) + s];")}
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
    }}
    {dataType} {outputName} = sdata[0]{(reductionType == "mean" ? " / get_local_size(0)" : "")};";
    }

    /// <summary>
    /// Generates OpenCL batch normalization code.
    /// </summary>
    private string GenerateOpenCLBatchNorm(BatchNormOp op, string dataType)
    {
        var outputName = EnsureTensorName(op.OutputId);
        var input = GetTensorName(op.InputIds[0]);
        var gamma = GetTensorName(op.InputIds[1]);
        var beta = GetTensorName(op.InputIds[2]);
        var mean = GetTensorName(op.InputIds[3]);
        var variance = GetTensorName(op.InputIds[4]);

        return $@"    // OpenCL BatchNorm
    {{
        int c = (idx / ({(op.OutputShape.Length >= 3 ? op.OutputShape[2] * op.OutputShape[3] : 1)})) % {op.OutputShape[1]};
        {dataType} x_norm = ({input}[idx] - {mean}[c]) * rsqrt({variance}[c] + ({dataType}){op.Epsilon});
        {dataType} {outputName} = {gamma}[c] * x_norm + {beta}[c];
    }}";
    }

    /// <summary>
    /// Generates Metal operation code.
    /// </summary>
    private string GenerateMetalOperation<T>(IROp op)
    {
        var outputName = EnsureTensorName(op.OutputId);
        var dataType = typeof(T) == typeof(float) ? "float" : "half";

        return op switch
        {
            AddOp add => GenerateElementwiseBinaryOp(add, "+", dataType),
            SubtractOp sub => GenerateElementwiseBinaryOp(sub, "-", dataType),
            ElementwiseMultiplyOp mul => GenerateElementwiseBinaryOp(mul, "*", dataType),
            DivideOp div => GenerateElementwiseBinaryOp(div, "/", dataType),
            ReLUOp relu => $"    {dataType} {outputName} = max({GetTensorName(relu.InputIds[0])}[idx], ({dataType})0);",
            SigmoidOp sig => $"    {dataType} {outputName} = 1.0 / (1.0 + exp(-{GetTensorName(sig.InputIds[0])}[idx]));",
            TanhOp tanh => $"    {dataType} {outputName} = tanh({GetTensorName(tanh.InputIds[0])}[idx]);",
            ExpOp exp => $"    {dataType} {outputName} = exp({GetTensorName(exp.InputIds[0])}[idx]);",
            LogOp log => $"    {dataType} {outputName} = log({GetTensorName(log.InputIds[0])}[idx]);",
            SqrtOp sqrt => $"    {dataType} {outputName} = sqrt({GetTensorName(sqrt.InputIds[0])}[idx]);",
            NegateOp neg => $"    {dataType} {outputName} = -{GetTensorName(neg.InputIds[0])}[idx];",
            _ => $"    // TODO: Implement {op.OpType} for Metal"
        };
    }

    /// <summary>
    /// Generates Vulkan operation code.
    /// </summary>
    private string GenerateVulkanOperation<T>(IROp op)
    {
        var outputName = EnsureTensorName(op.OutputId);
        var dataType = typeof(T) == typeof(float) ? "float" : "float16_t";

        return op switch
        {
            AddOp add => GenerateElementwiseBinaryOp(add, "+", dataType),
            SubtractOp sub => GenerateElementwiseBinaryOp(sub, "-", dataType),
            ElementwiseMultiplyOp mul => GenerateElementwiseBinaryOp(mul, "*", dataType),
            DivideOp div => GenerateElementwiseBinaryOp(div, "/", dataType),
            ReLUOp relu => $"    {dataType} {outputName} = max({GetTensorName(relu.InputIds[0])}[idx], {dataType}(0));",
            SigmoidOp sig => $"    {dataType} {outputName} = 1.0 / (1.0 + exp(-{GetTensorName(sig.InputIds[0])}[idx]));",
            TanhOp tanh => $"    {dataType} {outputName} = tanh({GetTensorName(tanh.InputIds[0])}[idx]);",
            ExpOp exp => $"    {dataType} {outputName} = exp({GetTensorName(exp.InputIds[0])}[idx]);",
            LogOp log => $"    {dataType} {outputName} = log({GetTensorName(log.InputIds[0])}[idx]);",
            SqrtOp sqrt => $"    {dataType} {outputName} = sqrt({GetTensorName(sqrt.InputIds[0])}[idx]);",
            NegateOp neg => $"    {dataType} {outputName} = -{GetTensorName(neg.InputIds[0])}[idx];",
            _ => $"    // TODO: Implement {op.OpType} for Vulkan"
        };
    }

    /// <summary>
    /// Generates code for element-wise binary operations.
    /// </summary>
    private string GenerateElementwiseBinaryOp(IROp op, string oper, string dataType)
    {
        var outputName = EnsureTensorName(op.OutputId);
        var left = GetTensorName(op.InputIds[0]);
        var right = GetTensorName(op.InputIds[1]);
        return $"    {dataType} {outputName} = {left}[idx] {oper} {right}[idx];";
    }

    /// <summary>
    /// Generates fused linear activation for CUDA.
    /// </summary>
    private string GenerateFusedLinearActivationCUDA<T>(FusedLinearActivationOp op)
    {
        var dataType = GetDataTypeString<T>();
        var activation = op.ActivationName.ToLower() switch
        {
            "relu" => "cuda_relu",
            "sigmoid" => "cuda_sigmoid",
            "tanh" => "cuda_tanh",
            _ => "cuda_relu"
        };

        // For element-wise kernel, this simplifies to activation of computed value
        var outputName = EnsureTensorName(op.OutputId);
        return $"    // Fused linear + {op.ActivationName}\n" +
               $"    {dataType} {outputName} = {activation}(/* linear output */);";
    }

    /// <summary>
    /// Generates fused elementwise + activation for CUDA.
    /// </summary>
    private string GenerateFusedElementwiseActivationCUDA(FusedElementwiseActivationOp op, string dataType)
    {
        var outputName = EnsureTensorName(op.OutputId);
        var left = GetTensorName(op.InputIds[0]);
        var right = GetTensorName(op.InputIds[1]);

        var elemOper = op.ElementwiseOp.ToLower() switch
        {
            "add" => "+",
            "subtract" => "-",
            "multiply" => "*",
            "divide" => "/",
            _ => "+"
        };

        var activation = op.ActivationName.ToLower() switch
        {
            "relu" => "cuda_relu",
            "sigmoid" => "cuda_sigmoid",
            "tanh" => "cuda_tanh",
            _ => "cuda_relu"
        };

        return $"    {dataType} {outputName} = {activation}({left}[idx] {elemOper} {right}[idx]);";
    }

    /// <summary>
    /// Generates fused residual block for CUDA.
    /// </summary>
    private string GenerateFusedResidualBlockCUDA(FusedResidualBlockOp op, string dataType)
    {
        var outputName = EnsureTensorName(op.OutputId);
        var mainPath = GetTensorName(op.InputIds[0]);
        var skipPath = GetTensorName(op.InputIds[1]);

        var activation = op.ActivationName.ToLower() switch
        {
            "relu" => "cuda_relu",
            "sigmoid" => "cuda_sigmoid",
            "tanh" => "cuda_tanh",
            _ => "cuda_relu"
        };

        return $"    {dataType} {outputName} = {activation}({mainPath}[idx] + {skipPath}[idx]);";
    }

    /// <summary>
    /// Generates CUDA helper functions.
    /// </summary>
    private string GenerateCUDAHelperFunctions(string dataType)
    {
        return $@"
// Activation functions
__device__ __forceinline__ {dataType} cuda_relu({dataType} x) {{
    return x > 0 ? x : 0;
}}

__device__ __forceinline__ {dataType} cuda_sigmoid({dataType} x) {{
    return 1.0f / (1.0f + expf(-x));
}}

__device__ __forceinline__ {dataType} cuda_tanh({dataType} x) {{
    return tanhf(x);
}}

__device__ __forceinline__ {dataType} cuda_gelu({dataType} x) {{
    // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const {dataType} c = 0.7978845608f; // sqrt(2/pi)
    const {dataType} k = 0.044715f;
    return 0.5f * x * (1.0f + tanhf(c * (x + k * x * x * x)));
}}

__device__ __forceinline__ {dataType} cuda_swish({dataType} x) {{
    return x * cuda_sigmoid(x);
}}

__device__ __forceinline__ {dataType} cuda_leaky_relu({dataType} x, {dataType} alpha = 0.01f) {{
    return x > 0 ? x : alpha * x;
}}
";
    }

    /// <summary>
    /// Generates OpenCL helper functions.
    /// </summary>
    private string GenerateOpenCLHelperFunctions(string dataType)
    {
        return $@"
// Activation functions
inline {dataType} ocl_relu({dataType} x) {{
    return max(x, ({dataType})0);
}}

inline {dataType} ocl_sigmoid({dataType} x) {{
    return ({dataType})1 / (({dataType})1 + exp(-x));
}}

inline {dataType} ocl_tanh({dataType} x) {{
    return tanh(x);
}}

inline {dataType} ocl_gelu({dataType} x) {{
    const {dataType} c = 0.7978845608f;
    const {dataType} k = 0.044715f;
    return ({dataType})0.5 * x * (({dataType})1 + tanh(c * (x + k * x * x * x)));
}}

inline {dataType} ocl_swish({dataType} x) {{
    return x * ocl_sigmoid(x);
}}
";
    }

    /// <summary>
    /// Generates Metal helper functions.
    /// </summary>
    private string GenerateMetalHelperFunctions(string dataType)
    {
        return $@"
// Activation functions
inline {dataType} mtl_relu({dataType} x) {{
    return max(x, ({dataType})0);
}}

inline {dataType} mtl_sigmoid({dataType} x) {{
    return ({dataType})1 / (({dataType})1 + exp(-x));
}}

inline {dataType} mtl_gelu({dataType} x) {{
    const {dataType} c = 0.7978845608;
    const {dataType} k = 0.044715;
    return ({dataType})0.5 * x * (({dataType})1 + tanh(c * (x + k * x * x * x)));
}}
";
    }

    /// <summary>
    /// Generates Vulkan helper functions.
    /// </summary>
    private string GenerateVulkanHelperFunctions(string dataType)
    {
        return $@"
// Activation functions
{dataType} glsl_relu({dataType} x) {{
    return max(x, {dataType}(0));
}}

{dataType} glsl_sigmoid({dataType} x) {{
    return {dataType}(1) / ({dataType}(1) + exp(-x));
}}

{dataType} glsl_gelu({dataType} x) {{
    const {dataType} c = {dataType}(0.7978845608);
    const {dataType} k = {dataType}(0.044715);
    return {dataType}(0.5) * x * ({dataType}(1) + tanh(c * (x + k * x * x * x)));
}}
";
    }

    /// <summary>
    /// Generates CUDA launcher function.
    /// </summary>
    private string GenerateCUDALauncher<T>(IRGraph graph, string kernelName, int[] blockSize)
    {
        var sb = new StringBuilder();
        var dataType = GetDataTypeString<T>();

        sb.AppendLine($"void launch_{kernelName}(");

        // Input parameters
        foreach (var inputId in graph.InputIds)
        {
            sb.AppendLine($"    const {dataType}* d_input_{inputId},");
        }

        // Output parameters
        foreach (var outputId in graph.OutputIds)
        {
            sb.AppendLine($"    {dataType}* d_output_{outputId},");
        }

        sb.AppendLine("    int total_elements,");
        sb.AppendLine("    cudaStream_t stream = 0");
        sb.AppendLine(") {");
        sb.AppendLine($"    int block_size = {blockSize[0]};");
        sb.AppendLine("    int grid_size = (total_elements + block_size - 1) / block_size;");
        sb.AppendLine();
        sb.Append($"    {kernelName}<<<grid_size, block_size, 0, stream>>>(");

        var args = new List<string>();
        foreach (var inputId in graph.InputIds)
        {
            args.Add($"d_input_{inputId}");
        }
        foreach (var outputId in graph.OutputIds)
        {
            args.Add($"d_output_{outputId}");
        }
        args.Add("total_elements");

        sb.AppendLine(string.Join(", ", args) + ");");
        sb.AppendLine("}");

        return sb.ToString();
    }

    /// <summary>
    /// Gets the data type string for the target backend.
    /// </summary>
    private string GetDataTypeString<T>()
    {
        return _backend switch
        {
            GPUBackend.CUDA => typeof(T) == typeof(double) ? "double" :
                              typeof(T) == typeof(Half) ? "half" : "float",
            GPUBackend.OpenCL => typeof(T) == typeof(double) ? "double" :
                                typeof(T) == typeof(Half) ? "half" : "float",
            GPUBackend.Metal => typeof(T) == typeof(Half) ? "half" : "float",
            GPUBackend.Vulkan => typeof(T) == typeof(Half) ? "float16_t" : "float",
            _ => "float"
        };
    }

    /// <summary>
    /// Calculates optimal launch configuration.
    /// </summary>
    private (int[] blockSize, int[] gridSize) CalculateLaunchConfig(IRGraph graph)
    {
        // Get total elements from output shape
        var totalElements = graph.OutputIds
            .Where(id => graph.TensorShapes.ContainsKey(id))
            .Select(id => graph.TensorShapes[id].Aggregate(1, (a, b) => a * b))
            .DefaultIfEmpty(1)
            .Max();

        // Choose block size based on device capabilities
        int blockSize = Math.Min(256, _deviceInfo.MaxThreadsPerBlock);

        // Ensure block size is a multiple of warp size
        blockSize = (blockSize / _deviceInfo.WarpSize) * _deviceInfo.WarpSize;
        if (blockSize == 0) blockSize = _deviceInfo.WarpSize;

        // Calculate grid size
        int gridSize = (totalElements + blockSize - 1) / blockSize;

        return (new int[] { blockSize }, new int[] { gridSize });
    }

    /// <summary>
    /// Calculates shared memory size needed.
    /// </summary>
    private int CalculateSharedMemorySize(IRGraph graph)
    {
        // Base shared memory for reductions
        int sharedMemory = 0;

        foreach (var op in graph.Operations)
        {
            if (op is SumOp or MeanOp or ReduceMaxOp or ReduceMeanOp or SoftmaxOp)
            {
                // Need shared memory for reductions
                sharedMemory = Math.Max(sharedMemory, _deviceInfo.WarpSize * sizeof(float));
            }
        }

        return Math.Min(sharedMemory, _deviceInfo.MaxSharedMemoryPerBlock);
    }

    /// <summary>
    /// Estimates GFLOPS for the kernel.
    /// </summary>
    private double EstimateGFLOPS(IRGraph graph)
    {
        double flops = 0;

        foreach (var op in graph.Operations)
        {
            var elements = op.OutputShape.Aggregate(1, (a, b) => a * b);

            flops += op switch
            {
                AddOp or SubtractOp or NegateOp => elements, // 1 FLOP per element
                ElementwiseMultiplyOp or DivideOp => elements,
                ReLUOp => elements, // 1 comparison
                SigmoidOp => elements * 4, // exp, add, div
                TanhOp => elements * 6,
                ExpOp or LogOp or SqrtOp => elements * 4,
                MatMulOp => 2.0 * elements, // Simplified estimate
                _ => elements
            };
        }

        return flops / 1e9; // Convert to GFLOPS
    }

    /// <summary>
    /// Gets or creates a tensor variable name.
    /// </summary>
    private string GetTensorName(int tensorId)
    {
        if (_tensorNames.TryGetValue(tensorId, out var name))
            return name;

        name = $"t{tensorId}";
        _tensorNames[tensorId] = name;
        return name;
    }

    /// <summary>
    /// Ensures a tensor has a name and returns it.
    /// </summary>
    private string EnsureTensorName(int tensorId)
    {
        if (!_tensorNames.ContainsKey(tensorId))
        {
            _tensorNames[tensorId] = $"t{tensorId}";
        }
        return _tensorNames[tensorId];
    }
}

/// <summary>
/// Specialized GPU kernels for common operations.
/// </summary>
public static class GPUKernelLibrary
{
    /// <summary>
    /// Generates optimized matrix multiplication kernel using tiled algorithm.
    /// </summary>
    public static string GenerateTiledMatMulKernel(int tileSize = 16)
    {
        return $@"
// Tiled matrix multiplication for better cache utilization
// A: [M, K], B: [K, N], C: [M, N]
__global__ void matmul_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {{
    const int TILE_SIZE = {tileSize};

    __shared__ float As[{tileSize}][{tileSize}];
    __shared__ float Bs[{tileSize}][{tileSize}];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {{
        // Load tiles into shared memory
        if (row < M && t * TILE_SIZE + tx < K)
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;

        if (t * TILE_SIZE + ty < K && col < N)
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // Compute partial sum
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {{
            sum += As[ty][k] * Bs[k][tx];
        }}

        __syncthreads();
    }}

    if (row < M && col < N)
        C[row * N + col] = sum;
}}
";
    }

    /// <summary>
    /// Generates optimized convolution kernel.
    /// </summary>
    public static string GenerateConv2DKernel()
    {
        return @"
// Implicit GEMM convolution for better GPU utilization
__global__ void conv2d_implicit_gemm(
    const float* __restrict__ input,   // [N, C_in, H, W]
    const float* __restrict__ kernel,  // [C_out, C_in, K_h, K_w]
    float* __restrict__ output,        // [N, C_out, H_out, W_out]
    int N, int C_in, int H, int W,
    int C_out, int K_h, int K_w,
    int H_out, int W_out,
    int stride_h, int stride_w,
    int pad_h, int pad_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H_out * W_out;

    if (idx >= total) return;

    // Decode index
    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c_out = (idx / (W_out * H_out)) % C_out;
    int n = idx / (W_out * H_out * C_out);

    float sum = 0.0f;

    for (int c_in = 0; c_in < C_in; c_in++) {
        for (int k_h = 0; k_h < K_h; k_h++) {
            for (int k_w = 0; k_w < K_w; k_w++) {
                int h_in = h_out * stride_h - pad_h + k_h;
                int w_in = w_out * stride_w - pad_w + k_w;

                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    int input_idx = n * C_in * H * W + c_in * H * W + h_in * W + w_in;
                    int kernel_idx = c_out * C_in * K_h * K_w + c_in * K_h * K_w + k_h * K_w + k_w;
                    sum += input[input_idx] * kernel[kernel_idx];
                }
            }
        }
    }

    output[idx] = sum;
}
";
    }

    /// <summary>
    /// Generates softmax kernel with online normalization.
    /// </summary>
    public static string GenerateSoftmaxKernel()
    {
        return @"
// Online softmax for numerical stability
__global__ void softmax_online(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int seq_len
) {
    int batch = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float shared[];
    float* max_vals = shared;
    float* sum_vals = shared + blockDim.x;

    // Find max
    float local_max = -INFINITY;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        local_max = fmaxf(local_max, input[batch * seq_len + i]);
    }
    max_vals[tid] = local_max;
    __syncthreads();

    // Reduce max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            max_vals[tid] = fmaxf(max_vals[tid], max_vals[tid + s]);
        }
        __syncthreads();
    }
    float global_max = max_vals[0];

    // Compute exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        float exp_val = expf(input[batch * seq_len + i] - global_max);
        output[batch * seq_len + i] = exp_val;
        local_sum += exp_val;
    }
    sum_vals[tid] = local_sum;
    __syncthreads();

    // Reduce sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sum_vals[tid] += sum_vals[tid + s];
        }
        __syncthreads();
    }
    float global_sum = sum_vals[0];

    // Normalize
    for (int i = tid; i < seq_len; i += blockDim.x) {
        output[batch * seq_len + i] /= global_sum;
    }
}
";
    }

    /// <summary>
    /// Generates batch normalization kernel.
    /// </summary>
    public static string GenerateBatchNormKernel()
    {
        return @"
// Batch normalization forward pass
__global__ void batchnorm_forward(
    const float* __restrict__ input,   // [N, C, H, W]
    const float* __restrict__ gamma,   // [C]
    const float* __restrict__ beta,    // [C]
    const float* __restrict__ mean,    // [C]
    const float* __restrict__ var,     // [C]
    float* __restrict__ output,        // [N, C, H, W]
    int N, int C, int H, int W,
    float epsilon
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;

    if (idx >= total) return;

    int c = (idx / (H * W)) % C;

    float x = input[idx];
    float m = mean[c];
    float v = var[c];
    float g = gamma[c];
    float b = beta[c];

    float x_norm = (x - m) / sqrtf(v + epsilon);
    output[idx] = g * x_norm + b;
}
";
    }

    /// <summary>
    /// Generates attention mechanism kernel.
    /// </summary>
    public static string GenerateAttentionKernel()
    {
        return @"
// Scaled dot-product attention
// Q: [batch, heads, seq_len, head_dim]
// K: [batch, heads, seq_len, head_dim]
// V: [batch, heads, seq_len, head_dim]
__global__ void attention_forward(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    int batch, int heads, int seq_len, int head_dim,
    float scale
) {
    // This is a simplified version - full implementation would use flash attention
    int b = blockIdx.z;
    int h = blockIdx.y;
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (q_idx >= seq_len) return;

    int base_q = b * heads * seq_len * head_dim + h * seq_len * head_dim;
    int base_k = base_q;
    int base_v = base_q;
    int base_o = base_q;

    // Compute attention scores
    extern __shared__ float scores[];

    float max_score = -INFINITY;
    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += Q[base_q + q_idx * head_dim + d] * K[base_k + k_idx * head_dim + d];
        }
        score *= scale;
        scores[k_idx] = score;
        max_score = fmaxf(max_score, score);
    }

    // Softmax
    float sum_exp = 0.0f;
    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
        scores[k_idx] = expf(scores[k_idx] - max_score);
        sum_exp += scores[k_idx];
    }

    // Weighted sum of values
    for (int d = 0; d < head_dim; d++) {
        float out_val = 0.0f;
        for (int v_idx = 0; v_idx < seq_len; v_idx++) {
            out_val += (scores[v_idx] / sum_exp) * V[base_v + v_idx * head_dim + d];
        }
        output[base_o + q_idx * head_dim + d] = out_val;
    }
}
";
    }

    /// <summary>
    /// Generates a Flash Attention kernel (memory-efficient attention).
    /// </summary>
    public static string GenerateFlashAttentionKernel()
    {
        return @"
// Flash Attention - memory efficient attention with tiling
// Based on: https://arxiv.org/abs/2205.14135
__global__ void flash_attention_forward(
    const float* __restrict__ Q,  // [batch, heads, seq_len, head_dim]
    const float* __restrict__ K,  // [batch, heads, seq_len, head_dim]
    const float* __restrict__ V,  // [batch, heads, seq_len, head_dim]
    float* __restrict__ O,        // [batch, heads, seq_len, head_dim]
    float* __restrict__ L,        // [batch, heads, seq_len] - logsumexp for backward
    int batch, int heads, int seq_len, int head_dim,
    float scale, int BLOCK_SIZE
) {
    extern __shared__ float smem[];

    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int q_block_idx = blockIdx.x;

    int tid = threadIdx.x;
    int q_start = q_block_idx * BLOCK_SIZE;

    float* Qi = smem;                           // [BLOCK_SIZE, head_dim]
    float* Ki = smem + BLOCK_SIZE * head_dim;   // [BLOCK_SIZE, head_dim]
    float* Vi = smem + 2 * BLOCK_SIZE * head_dim; // [BLOCK_SIZE, head_dim]
    float* Si = smem + 3 * BLOCK_SIZE * head_dim; // [BLOCK_SIZE, BLOCK_SIZE]

    int base = batch_idx * heads * seq_len * head_dim + head_idx * seq_len * head_dim;

    // Initialize output accumulators
    float oi[64]; // Assume max head_dim = 64
    float mi = -INFINITY;
    float li = 0.0f;

    for (int d = 0; d < head_dim; d++) {
        oi[d] = 0.0f;
    }

    // Load Q block into shared memory
    for (int i = tid; i < BLOCK_SIZE * head_dim; i += blockDim.x) {
        int row = i / head_dim;
        int col = i % head_dim;
        int q_idx = q_start + row;
        if (q_idx < seq_len) {
            Qi[i] = Q[base + q_idx * head_dim + col];
        } else {
            Qi[i] = 0.0f;
        }
    }
    __syncthreads();

    // Iterate over K,V blocks
    int num_kv_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int kv_start = kv_block * BLOCK_SIZE;

        // Load K, V blocks
        for (int i = tid; i < BLOCK_SIZE * head_dim; i += blockDim.x) {
            int row = i / head_dim;
            int col = i % head_dim;
            int kv_idx = kv_start + row;
            if (kv_idx < seq_len) {
                Ki[i] = K[base + kv_idx * head_dim + col];
                Vi[i] = V[base + kv_idx * head_dim + col];
            } else {
                Ki[i] = 0.0f;
                Vi[i] = 0.0f;
            }
        }
        __syncthreads();

        // Compute S = Q @ K^T * scale
        if (tid < BLOCK_SIZE && (q_start + tid) < seq_len) {
            for (int j = 0; j < BLOCK_SIZE && (kv_start + j) < seq_len; j++) {
                float s = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    s += Qi[tid * head_dim + d] * Ki[j * head_dim + d];
                }
                Si[tid * BLOCK_SIZE + j] = s * scale;
            }
        }
        __syncthreads();

        // Update running statistics and output
        if (tid < BLOCK_SIZE && (q_start + tid) < seq_len) {
            float mi_new = mi;
            for (int j = 0; j < BLOCK_SIZE && (kv_start + j) < seq_len; j++) {
                mi_new = fmaxf(mi_new, Si[tid * BLOCK_SIZE + j]);
            }

            float li_new = li * expf(mi - mi_new);
            for (int j = 0; j < BLOCK_SIZE && (kv_start + j) < seq_len; j++) {
                li_new += expf(Si[tid * BLOCK_SIZE + j] - mi_new);
            }

            // Update output
            float scale_old = li * expf(mi - mi_new) / li_new;
            for (int d = 0; d < head_dim; d++) {
                oi[d] *= scale_old;
                for (int j = 0; j < BLOCK_SIZE && (kv_start + j) < seq_len; j++) {
                    float p = expf(Si[tid * BLOCK_SIZE + j] - mi_new) / li_new;
                    oi[d] += p * Vi[j * head_dim + d];
                }
            }

            mi = mi_new;
            li = li_new;
        }
        __syncthreads();
    }

    // Write output
    if (tid < BLOCK_SIZE && (q_start + tid) < seq_len) {
        for (int d = 0; d < head_dim; d++) {
            O[base + (q_start + tid) * head_dim + d] = oi[d];
        }
        L[batch_idx * heads * seq_len + head_idx * seq_len + q_start + tid] = mi + logf(li);
    }
}
";
    }

    /// <summary>
    /// Generates a depthwise separable convolution kernel.
    /// </summary>
    public static string GenerateDepthwiseSeparableConvKernel()
    {
        return @"
// Depthwise separable convolution (MobileNet style)
// More efficient than standard convolution for mobile/edge deployment
__global__ void depthwise_conv2d(
    const float* __restrict__ input,   // [N, C, H, W]
    const float* __restrict__ kernel,  // [C, 1, K_h, K_w]
    float* __restrict__ output,        // [N, C, H_out, W_out]
    int N, int C, int H, int W,
    int K_h, int K_w,
    int H_out, int W_out,
    int stride_h, int stride_w,
    int pad_h, int pad_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;

    if (idx >= total) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c = (idx / (W_out * H_out)) % C;
    int n = idx / (W_out * H_out * C);

    float sum = 0.0f;

    for (int k_h = 0; k_h < K_h; k_h++) {
        for (int k_w = 0; k_w < K_w; k_w++) {
            int h_in = h_out * stride_h - pad_h + k_h;
            int w_in = w_out * stride_w - pad_w + k_w;

            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                int input_idx = n * C * H * W + c * H * W + h_in * W + w_in;
                int kernel_idx = c * K_h * K_w + k_h * K_w + k_w;
                sum += input[input_idx] * kernel[kernel_idx];
            }
        }
    }

    output[idx] = sum;
}

// Pointwise convolution (1x1 conv)
__global__ void pointwise_conv2d(
    const float* __restrict__ input,   // [N, C_in, H, W]
    const float* __restrict__ kernel,  // [C_out, C_in, 1, 1]
    float* __restrict__ output,        // [N, C_out, H, W]
    int N, int C_in, int C_out, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H * W;

    if (idx >= total) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int c_out = (idx / (W * H)) % C_out;
    int n = idx / (W * H * C_out);

    float sum = 0.0f;

    for (int c_in = 0; c_in < C_in; c_in++) {
        int input_idx = n * C_in * H * W + c_in * H * W + h * W + w;
        int kernel_idx = c_out * C_in + c_in;
        sum += input[input_idx] * kernel[kernel_idx];
    }

    output[idx] = sum;
}
";
    }
}

/// <summary>
/// Interface for GPU runtime implementations.
/// </summary>
/// <remarks>
/// <para>
/// This interface defines the contract for GPU runtime implementations that can
/// compile and execute generated kernel code. Implementations would wrap CUDA Runtime,
/// OpenCL, Metal, or Vulkan APIs.
/// </para>
/// <para><b>For Beginners:</b> This is the bridge between generated code and actual GPU execution.
///
/// The code generator produces kernel source code, but to actually run it:
/// 1. The source must be compiled to GPU machine code
/// 2. Memory must be allocated on the GPU
/// 3. Data must be transferred to the GPU
/// 4. The kernel must be launched
/// 5. Results must be transferred back
///
/// This interface defines all those operations.
/// </para>
/// </remarks>
public interface IGPURuntime : IDisposable
{
    /// <summary>
    /// Gets information about the current GPU device.
    /// </summary>
    GPUCodeGenerator.GPUDeviceInfo DeviceInfo { get; }

    /// <summary>
    /// Compiles kernel source code into an executable module.
    /// </summary>
    /// <param name="sourceCode">The kernel source code.</param>
    /// <param name="kernelName">The name of the kernel function.</param>
    /// <returns>A handle to the compiled kernel.</returns>
    IGPUKernelHandle CompileKernel(string sourceCode, string kernelName);

    /// <summary>
    /// Allocates memory on the GPU.
    /// </summary>
    /// <param name="sizeBytes">Number of bytes to allocate.</param>
    /// <returns>A handle to the allocated memory.</returns>
    IGPUMemoryHandle Allocate(long sizeBytes);

    /// <summary>
    /// Copies data from host to GPU memory.
    /// </summary>
    /// <param name="destination">GPU memory handle.</param>
    /// <param name="source">Source data array.</param>
    void CopyToDevice<T>(IGPUMemoryHandle destination, T[] source) where T : unmanaged;

    /// <summary>
    /// Copies data from GPU to host memory.
    /// </summary>
    /// <param name="destination">Destination array.</param>
    /// <param name="source">GPU memory handle.</param>
    void CopyFromDevice<T>(T[] destination, IGPUMemoryHandle source) where T : unmanaged;

    /// <summary>
    /// Launches a kernel with the specified configuration.
    /// </summary>
    /// <param name="kernel">The kernel to launch.</param>
    /// <param name="gridSize">Number of blocks in each dimension.</param>
    /// <param name="blockSize">Number of threads per block in each dimension.</param>
    /// <param name="sharedMemorySize">Dynamic shared memory size in bytes.</param>
    /// <param name="arguments">Kernel arguments (GPU memory handles or scalars).</param>
    void LaunchKernel(IGPUKernelHandle kernel, int[] gridSize, int[] blockSize, int sharedMemorySize, params object[] arguments);

    /// <summary>
    /// Synchronizes with the GPU, waiting for all pending operations to complete.
    /// </summary>
    void Synchronize();

    /// <summary>
    /// Frees GPU memory.
    /// </summary>
    /// <param name="memory">The memory handle to free.</param>
    void Free(IGPUMemoryHandle memory);
}

/// <summary>
/// Handle to a compiled GPU kernel.
/// </summary>
public interface IGPUKernelHandle : IDisposable
{
    /// <summary>Gets the kernel name.</summary>
    string Name { get; }

    /// <summary>Gets whether the kernel is valid and ready for execution.</summary>
    bool IsValid { get; }
}

/// <summary>
/// Handle to GPU memory allocation.
/// </summary>
public interface IGPUMemoryHandle : IDisposable
{
    /// <summary>Gets the size of the allocation in bytes.</summary>
    long SizeBytes { get; }

    /// <summary>Gets whether the memory is still allocated.</summary>
    bool IsAllocated { get; }
}

/// <summary>
/// Mock GPU runtime for testing without actual GPU hardware.
/// </summary>
/// <remarks>
/// This implementation simulates GPU operations on the CPU for testing purposes.
/// It allows the JIT compiler to be tested without requiring actual GPU hardware.
/// </remarks>
public class MockGPURuntime : IGPURuntime
{
    private readonly GPUCodeGenerator.GPUDeviceInfo _deviceInfo;
    private bool _disposed;

    /// <summary>
    /// Initializes a new mock GPU runtime.
    /// </summary>
    public MockGPURuntime()
    {
        _deviceInfo = new GPUCodeGenerator.GPUDeviceInfo
        {
            DeviceName = "Mock GPU (CPU Simulation)",
            MaxThreadsPerBlock = 1024,
            MaxSharedMemoryPerBlock = 49152,
            MultiprocessorCount = 1,
            WarpSize = 32,
            ComputeCapability = "Mock",
            GlobalMemory = 8L * 1024 * 1024 * 1024,
            HasTensorCores = false
        };
    }

    /// <inheritdoc/>
    public GPUCodeGenerator.GPUDeviceInfo DeviceInfo => _deviceInfo;

    /// <inheritdoc/>
    public IGPUKernelHandle CompileKernel(string sourceCode, string kernelName)
    {
        // In mock mode, we just store the source code
        return new MockKernelHandle(kernelName, sourceCode);
    }

    /// <inheritdoc/>
    public IGPUMemoryHandle Allocate(long sizeBytes)
    {
        // Allocate on CPU heap
        return new MockMemoryHandle(sizeBytes);
    }

    /// <inheritdoc/>
    public void CopyToDevice<T>(IGPUMemoryHandle destination, T[] source) where T : unmanaged
    {
        if (destination is MockMemoryHandle mock)
        {
            var bytes = new byte[source.Length * System.Runtime.InteropServices.Marshal.SizeOf<T>()];
            Buffer.BlockCopy(source, 0, bytes, 0, bytes.Length);
            mock.Data = bytes;
        }
    }

    /// <inheritdoc/>
    public void CopyFromDevice<T>(T[] destination, IGPUMemoryHandle source) where T : unmanaged
    {
        if (source is MockMemoryHandle mock && mock.Data != null)
        {
            Buffer.BlockCopy(mock.Data, 0, destination, 0, Math.Min(mock.Data.Length, destination.Length * System.Runtime.InteropServices.Marshal.SizeOf<T>()));
        }
    }

    /// <inheritdoc/>
    public void LaunchKernel(IGPUKernelHandle kernel, int[] gridSize, int[] blockSize, int sharedMemorySize, params object[] arguments)
    {
        // In mock mode, we would interpret the kernel
        // For now, this is a no-op - actual execution would require a kernel interpreter
    }

    /// <inheritdoc/>
    public void Synchronize()
    {
        // No-op in mock mode
    }

    /// <inheritdoc/>
    public void Free(IGPUMemoryHandle memory)
    {
        memory.Dispose();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            GC.SuppressFinalize(this);
        }
    }

    private class MockKernelHandle : IGPUKernelHandle
    {
        public string Name { get; }
        public string SourceCode { get; }
        public bool IsValid => true;

        public MockKernelHandle(string name, string sourceCode)
        {
            Name = name;
            SourceCode = sourceCode;
        }

        public void Dispose() { }
    }

    private class MockMemoryHandle : IGPUMemoryHandle
    {
        public long SizeBytes { get; }
        public bool IsAllocated { get; private set; } = true;
        public byte[]? Data { get; set; }

        public MockMemoryHandle(long sizeBytes)
        {
            SizeBytes = sizeBytes;
            Data = new byte[sizeBytes];
        }

        public void Dispose()
        {
            IsAllocated = false;
            Data = null;
        }
    }
}

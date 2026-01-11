using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a time embedding layer that encodes timesteps using sinusoidal embeddings for diffusion models.
/// </summary>
/// <remarks>
/// <para>
/// The time embedding layer converts scalar timesteps into high-dimensional embeddings using sinusoidal
/// functions, similar to positional encodings in transformers. This embedding is then projected through
/// a small MLP to produce the final time conditioning vector used in diffusion U-Net blocks.
/// </para>
/// <para><b>For Beginners:</b> In diffusion models, the network needs to know "what time step are we at?"
///
/// - At early timesteps (t near 0), images are clean and noise is minimal
/// - At late timesteps (t near T), images are mostly noise
/// - The network needs this information to know how much denoising to apply
///
/// This layer encodes the timestep number into a rich vector representation that:
/// 1. Uses sine and cosine functions at different frequencies (sinusoidal encoding)
/// 2. Passes through a small neural network (MLP) to learn task-specific representations
/// 3. Gets injected into every ResNet block of the U-Net
///
/// The sinusoidal encoding is inspired by transformer positional encodings:
/// - Low frequencies capture coarse time information
/// - High frequencies capture fine-grained time details
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class TimeEmbeddingLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The dimension of the sinusoidal embedding before MLP projection.
    /// </summary>
    private readonly int _embeddingDim;

    /// <summary>
    /// The dimension of the output after MLP projection.
    /// </summary>
    private readonly int _outputDim;

    /// <summary>
    /// First linear layer weights: [embeddingDim, outputDim]
    /// </summary>
    private Tensor<T> _linear1Weights;

    /// <summary>
    /// First linear layer biases: [outputDim]
    /// </summary>
    private Tensor<T> _linear1Bias;

    /// <summary>
    /// Second linear layer weights: [outputDim, outputDim]
    /// </summary>
    private Tensor<T> _linear2Weights;

    /// <summary>
    /// Second linear layer biases: [outputDim]
    /// </summary>
    private Tensor<T> _linear2Bias;

    /// <summary>
    /// Cached sinusoidal embedding from last forward pass.
    /// </summary>
    private Tensor<T>? _lastSinusoidalEmbed;

    /// <summary>
    /// Cached intermediate output after first linear + activation.
    /// </summary>
    private Tensor<T>? _lastHidden;

    /// <summary>
    /// Cached input timesteps from last forward pass.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Gradient for first linear layer weights.
    /// </summary>
    private Tensor<T>? _linear1WeightsGradient;

    /// <summary>
    /// Gradient for first linear layer biases.
    /// </summary>
    private Tensor<T>? _linear1BiasGradient;

    /// <summary>
    /// Gradient for second linear layer weights.
    /// </summary>
    private Tensor<T>? _linear2WeightsGradient;

    /// <summary>
    /// Gradient for second linear layer biases.
    /// </summary>
    private Tensor<T>? _linear2BiasGradient;

    // GPU cached tensors for backward pass
    private IGpuTensor<T>? _gpuTimesteps;
    private IGpuTensor<T>? _gpuSinusoidalEmbed;
    private IGpuTensor<T>? _gpuHidden;
    private IGpuTensor<T>? _gpuPreActivation;
    private int[]? _gpuInputShape;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    protected override bool SupportsGpuExecution => true;

    private Tensor<T>? _frequencies;

    /// <inheritdoc/>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length == 0) throw new ArgumentException("TimeEmbeddingLayer requires an input tensor.");
        var input = inputs[0];

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires DirectGpuTensorEngine.");

        if (_frequencies == null)
        {
            int halfDim = _embeddingDim / 2;
            double logMax = Math.Log(10000.0);
            _frequencies = new Tensor<T>([halfDim, 1]);
            for (int i = 0; i < halfDim; i++)
            {
                double freq = Math.Exp(-logMax * i / halfDim);
                _frequencies[i, 0] = NumOps.FromDouble(freq);
            }
            RegisterTrainableParameter(_frequencies, PersistentTensorRole.Constant);
        }

        int batch = input.Shape[0];
        IGpuTensor<T> timesteps = input.Shape.Length == 1
            ? gpuEngine.ReshapeGpu(input, [batch, 1])
            : input;

        // Compute sinusoidal embedding args: timesteps @ frequencies^T
        var freqsT = _frequencies.Transpose();
        var args = gpuEngine.FusedLinearGpu(timesteps, freqsT, null, FusedActivationType.None);

        var sinPart = gpuEngine.SinGpu(args);
        var cosPart = gpuEngine.CosGpu(args);

        // Concatenate sin and cos components along feature axis
        var embedding = gpuEngine.ConcatGpu<T>([sinPart, cosPart], 1);

        // MLP projection: Linear1 -> SiLU (Swish) -> Linear2
        // For backward pass, we need the pre-activation values, so compute linear1 without activation first
        var preActivation = gpuEngine.FusedLinearGpu(embedding, _linear1Weights, _linear1Bias, FusedActivationType.None);
        var hidden = gpuEngine.ActivationGpu<T>(preActivation, FusedActivationType.Swish);
        var output = gpuEngine.FusedLinearGpu(hidden, _linear2Weights, _linear2Bias, FusedActivationType.None);

        if (IsTrainingMode)
        {
            _lastInput = input.ToTensor();
            _lastSinusoidalEmbed = embedding.ToTensor();
            _lastHidden = hidden.ToTensor();

            // Cache GPU tensors for backward pass
            _gpuInputShape = input.Shape;
            _gpuTimesteps = timesteps;
            _gpuSinusoidalEmbed = embedding;
            _gpuHidden = hidden;
            _gpuPreActivation = preActivation;
        }

        return output;
    }

    /// <summary>
    /// Performs the GPU-resident backward pass of the time embedding layer.
    /// </summary>
    /// <param name="outputGradient">The GPU tensor containing the gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the timestep input (typically zeros since sinusoidal embedding is fixed).</returns>
    public override IGpuTensor<T> BackwardGpu(IGpuTensor<T> outputGradient)
    {
        if (_gpuSinusoidalEmbed == null || _gpuHidden == null || _gpuPreActivation == null || _gpuInputShape == null)
            throw new InvalidOperationException("ForwardGpu must be called in training mode before BackwardGpu.");

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("BackwardGpu requires DirectGpuTensorEngine.");

        int batch = outputGradient.Shape[0];

        // Step 1: Backprop through Linear2 (output = hidden @ W2 + b2)
        // dL/dHidden = dL/dOutput @ W2^T
        var linear2WeightsGpu = gpuEngine.UploadToGpu<T>(_linear2Weights, GpuTensorRole.Weight);
        var linear2WeightsT = gpuEngine.TransposeGpu<T>(linear2WeightsGpu);
        var hiddenGradient = gpuEngine.MatMulGpuTensors<T>(outputGradient, linear2WeightsT);

        // dL/dW2 = hidden^T @ dL/dOutput
        var hiddenT = gpuEngine.TransposeGpu<T>(_gpuHidden);
        var w2Grad = gpuEngine.MatMulGpuTensors<T>(hiddenT, outputGradient);
        _linear2WeightsGradient = w2Grad.ToTensor();

        // dL/db2 = sum(dL/dOutput, axis=0)
        _linear2BiasGradient = gpuEngine.SumAxisGpu<T>(outputGradient, 0).ToTensor();

        // Step 2: Backprop through Swish activation
        // SwishBackwardGpu takes (gradOutput, output) where output is the Swish output (hidden)
        var preActivationGradient = gpuEngine.SwishBackwardGpu<T>(hiddenGradient, _gpuHidden);

        // Step 3: Backprop through Linear1 (preActivation = embedding @ W1 + b1)
        // dL/dEmbedding = dL/dPreActivation @ W1^T (not needed since sinusoidal embedding is fixed)
        // dL/dW1 = embedding^T @ dL/dPreActivation
        var embeddingT = gpuEngine.TransposeGpu<T>(_gpuSinusoidalEmbed);
        var w1Grad = gpuEngine.MatMulGpuTensors<T>(embeddingT, preActivationGradient);
        _linear1WeightsGradient = w1Grad.ToTensor();

        // dL/db1 = sum(dL/dPreActivation, axis=0)
        _linear1BiasGradient = gpuEngine.SumAxisGpu<T>(preActivationGradient, 0).ToTensor();

        // Step 4: Return zeros for input gradient (timesteps are not learnable)
        // The sinusoidal embedding is fixed, so gradient w.r.t. timesteps is typically not needed
        var backend = gpuEngine.GetBackend();
        if (backend == null)
            throw new InvalidOperationException("GPU backend unavailable.");

        int inputSize = 1;
        foreach (var dim in _gpuInputShape)
            inputSize *= dim;

        var zeroBuffer = backend.AllocateBuffer(inputSize);
        backend.Fill(zeroBuffer, 0.0f, inputSize);

        return new GpuTensor<T>(backend, zeroBuffer, _gpuInputShape, GpuTensorRole.Gradient, ownsBuffer: true);
    }

    /// <summary>
    /// The maximum timestep value for scaling embeddings.
    /// </summary>
    private readonly T _maxTimestep;

    /// <summary>
    /// Initializes a new instance of the <see cref="TimeEmbeddingLayer{T}"/> class.
    /// </summary>
    /// <param name="embeddingDim">The dimension of the sinusoidal embedding (typically model_dim / 4).</param>
    /// <param name="outputDim">The dimension of the output embedding (typically model_dim * 4).</param>
    /// <param name="maxTimestep">Maximum timestep value for normalization. Default: 1000.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Create a time embedding layer with specified dimensions.
    ///
    /// Common configurations:
    /// - embeddingDim = 64, outputDim = 256 for small models
    /// - embeddingDim = 128, outputDim = 512 for medium models
    /// - embeddingDim = 320, outputDim = 1280 for Stable Diffusion scale
    ///
    /// The layer consists of:
    /// 1. Sinusoidal encoding: timestep -> [embeddingDim] features
    /// 2. Linear1 + SiLU: [embeddingDim] -> [outputDim]
    /// 3. Linear2: [outputDim] -> [outputDim]
    /// </para>
    /// </remarks>
    public TimeEmbeddingLayer(int embeddingDim, int outputDim, int maxTimestep = 1000)
        : base([1], [outputDim])
    {
        _embeddingDim = embeddingDim;
        _outputDim = outputDim;
        _maxTimestep = NumOps.FromDouble(maxTimestep);

        // Initialize weights using Xavier initialization
        // Use RandomHelper, NEVER new Random() directly
        var random = RandomHelper.CreateSeededRandom(42);
        double scale1 = Math.Sqrt(2.0 / (embeddingDim + outputDim));
        double scale2 = Math.Sqrt(2.0 / (outputDim + outputDim));

        _linear1Weights = new Tensor<T>([embeddingDim, outputDim]);
        _linear1Bias = Tensor<T>.CreateDefault([outputDim], NumOps.Zero);
        _linear2Weights = new Tensor<T>([outputDim, outputDim]);
        _linear2Bias = Tensor<T>.CreateDefault([outputDim], NumOps.Zero);

        // Initialize weights with Xavier/Glorot
        for (int i = 0; i < embeddingDim; i++)
        {
            for (int j = 0; j < outputDim; j++)
            {
                _linear1Weights[i, j] = NumOps.FromDouble((random.NextDouble() * 2 - 1) * scale1);
            }
        }

        for (int i = 0; i < outputDim; i++)
        {
            for (int j = 0; j < outputDim; j++)
            {
                _linear2Weights[i, j] = NumOps.FromDouble((random.NextDouble() * 2 - 1) * scale2);
            }
        }

        // Register trainable parameters for GPU memory optimization
        RegisterTrainableParameter(_linear1Weights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_linear1Bias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_linear2Weights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_linear2Bias, PersistentTensorRole.Biases);
    }

    /// <summary>
    /// Computes sinusoidal embedding for the given timesteps.
    /// </summary>
    /// <param name="timesteps">Tensor of timesteps with shape [batch] or [batch, 1].</param>
    /// <returns>Sinusoidal embedding tensor with shape [batch, embeddingDim].</returns>
    private Tensor<T> ComputeSinusoidalEmbedding(Tensor<T> timesteps)
    {
        int batch = timesteps.Shape[0];
        int halfDim = _embeddingDim / 2;

        // Compute frequency scaling factors: exp(-log(10000) * i / halfDim)
        // These create logarithmically spaced frequencies from 1 to 1/10000
        double logMax = Math.Log(10000.0);

        var embedding = new Tensor<T>([batch, _embeddingDim]);

        for (int b = 0; b < batch; b++)
        {
            double t = NumOps.ToDouble(timesteps[b]);

            for (int i = 0; i < halfDim; i++)
            {
                double freq = Math.Exp(-logMax * i / halfDim);
                double angle = t * freq;

                // Sin component
                embedding[b, i] = NumOps.FromDouble(Math.Sin(angle));
                // Cos component
                embedding[b, i + halfDim] = NumOps.FromDouble(Math.Cos(angle));
            }
        }

        return embedding;
    }

    /// <summary>
    /// Applies the SiLU (Swish) activation function: x * sigmoid(x).
    /// </summary>
    private T SiLU(T x)
    {
        double xd = NumOps.ToDouble(x);
        double sigmoid = 1.0 / (1.0 + Math.Exp(-xd));
        return NumOps.FromDouble(xd * sigmoid);
    }

    /// <summary>
    /// Computes the derivative of SiLU activation.
    /// </summary>
    private T SiLUDerivative(T x)
    {
        double xd = NumOps.ToDouble(x);
        double sigmoid = 1.0 / (1.0 + Math.Exp(-xd));
        // d/dx [x * sigmoid(x)] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        return NumOps.FromDouble(sigmoid + xd * sigmoid * (1 - sigmoid));
    }

    /// <summary>
    /// Performs the forward pass of the time embedding layer.
    /// </summary>
    /// <param name="input">Input tensor containing timesteps. Shape: [batch] or [batch, 1].</param>
    /// <returns>Time embedding tensor with shape [batch, outputDim].</returns>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int batch = input.Shape[0];

        // Flatten if needed
        var timesteps = input.Rank == 1 ? input : input.Reshape([batch]);

        // Step 1: Compute sinusoidal embedding
        var sinEmbed = ComputeSinusoidalEmbedding(timesteps);
        _lastSinusoidalEmbed = sinEmbed;

        // Step 2: First linear layer + SiLU activation
        var hidden = new Tensor<T>([batch, _outputDim]);
        for (int b = 0; b < batch; b++)
        {
            for (int j = 0; j < _outputDim; j++)
            {
                var sum = _linear1Bias[j];
                for (int i = 0; i < _embeddingDim; i++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(sinEmbed[b, i], _linear1Weights[i, j]));
                }
                hidden[b, j] = SiLU(sum);
            }
        }
        _lastHidden = hidden;

        // Step 3: Second linear layer
        var output = new Tensor<T>([batch, _outputDim]);
        for (int b = 0; b < batch; b++)
        {
            for (int j = 0; j < _outputDim; j++)
            {
                var sum = _linear2Bias[j];
                for (int i = 0; i < _outputDim; i++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(hidden[b, i], _linear2Weights[i, j]));
                }
                output[b, j] = sum;
            }
        }

        return output;
    }

    /// <summary>
    /// Performs the backward pass of the time embedding layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input (timesteps).</returns>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastSinusoidalEmbed == null || _lastHidden == null || _lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batch = outputGradient.Shape[0];

        // Initialize gradients
        _linear2WeightsGradient = Tensor<T>.CreateDefault([_outputDim, _outputDim], NumOps.Zero);
        _linear2BiasGradient = Tensor<T>.CreateDefault([_outputDim], NumOps.Zero);
        _linear1WeightsGradient = Tensor<T>.CreateDefault([_embeddingDim, _outputDim], NumOps.Zero);
        _linear1BiasGradient = Tensor<T>.CreateDefault([_outputDim], NumOps.Zero);

        // Backprop through second linear layer
        var hiddenGradient = new Tensor<T>([batch, _outputDim]);
        for (int b = 0; b < batch; b++)
        {
            for (int j = 0; j < _outputDim; j++)
            {
                var grad = outputGradient[b, j];
                _linear2BiasGradient[j] = NumOps.Add(_linear2BiasGradient[j], grad);

                for (int i = 0; i < _outputDim; i++)
                {
                    _linear2WeightsGradient[i, j] = NumOps.Add(_linear2WeightsGradient[i, j],
                        NumOps.Multiply(grad, _lastHidden[b, i]));
                    hiddenGradient[b, i] = NumOps.Add(hiddenGradient[b, i],
                        NumOps.Multiply(grad, _linear2Weights[i, j]));
                }
            }
        }

        // Backprop through SiLU activation
        var preActivationGrad = new Tensor<T>([batch, _outputDim]);
        for (int b = 0; b < batch; b++)
        {
            for (int j = 0; j < _outputDim; j++)
            {
                // Recompute pre-activation value
                var sum = _linear1Bias[j];
                for (int i = 0; i < _embeddingDim; i++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(_lastSinusoidalEmbed[b, i], _linear1Weights[i, j]));
                }
                preActivationGrad[b, j] = NumOps.Multiply(hiddenGradient[b, j], SiLUDerivative(sum));
            }
        }

        // Backprop through first linear layer
        for (int b = 0; b < batch; b++)
        {
            for (int j = 0; j < _outputDim; j++)
            {
                var grad = preActivationGrad[b, j];
                _linear1BiasGradient[j] = NumOps.Add(_linear1BiasGradient[j], grad);

                for (int i = 0; i < _embeddingDim; i++)
                {
                    _linear1WeightsGradient[i, j] = NumOps.Add(_linear1WeightsGradient[i, j],
                        NumOps.Multiply(grad, _lastSinusoidalEmbed[b, i]));
                }
            }
        }

        // Return gradient w.r.t. timesteps (typically not used but required by interface)
        return _lastInput.Rank == 1
            ? Tensor<T>.CreateDefault([batch], NumOps.Zero)
            : Tensor<T>.CreateDefault([batch, 1], NumOps.Zero);
    }

    /// <summary>
    /// Updates the parameters of the layer using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    public override void UpdateParameters(T learningRate)
    {
        if (_linear1WeightsGradient == null || _linear1BiasGradient == null ||
            _linear2WeightsGradient == null || _linear2BiasGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        _linear1Weights = Engine.TensorSubtract(_linear1Weights, Engine.TensorMultiplyScalar(_linear1WeightsGradient, learningRate));
        _linear1Bias = Engine.TensorSubtract(_linear1Bias, Engine.TensorMultiplyScalar(_linear1BiasGradient, learningRate));
        _linear2Weights = Engine.TensorSubtract(_linear2Weights, Engine.TensorMultiplyScalar(_linear2WeightsGradient, learningRate));
        _linear2Bias = Engine.TensorSubtract(_linear2Bias, Engine.TensorMultiplyScalar(_linear2BiasGradient, learningRate));

        // Notify GPU that tensor data has changed
        Engine.InvalidatePersistentTensor(_linear1Weights);
        Engine.InvalidatePersistentTensor(_linear1Bias);
        Engine.InvalidatePersistentTensor(_linear2Weights);
        Engine.InvalidatePersistentTensor(_linear2Bias);
    }

    /// <summary>
    /// Gets all trainable parameters of the layer as a single vector.
    /// </summary>
    public override Vector<T> GetParameters()
    {
        var parts = new Vector<T>[]
        {
            _linear1Weights.ToVector(),
            _linear1Bias.ToVector(),
            _linear2Weights.ToVector(),
            _linear2Bias.ToVector()
        };
        return Vector<T>.Concatenate(parts);
    }

    /// <summary>
    /// Sets all trainable parameters of the layer from a vector.
    /// </summary>
    public override void SetParameters(Vector<T> parameters)
    {
        int offset = 0;
        int size1 = _embeddingDim * _outputDim;
        int size2 = _outputDim;
        int size3 = _outputDim * _outputDim;
        int size4 = _outputDim;

        _linear1Weights = Tensor<T>.FromVector(parameters.Slice(offset, size1), [_embeddingDim, _outputDim]);
        offset += size1;
        _linear1Bias = Tensor<T>.FromVector(parameters.Slice(offset, size2), [_outputDim]);
        offset += size2;
        _linear2Weights = Tensor<T>.FromVector(parameters.Slice(offset, size3), [_outputDim, _outputDim]);
        offset += size3;
        _linear2Bias = Tensor<T>.FromVector(parameters.Slice(offset, size4), [_outputDim]);

        // Notify GPU that tensor data has changed
        Engine.InvalidatePersistentTensor(_linear1Weights);
        Engine.InvalidatePersistentTensor(_linear1Bias);
        Engine.InvalidatePersistentTensor(_linear2Weights);
        Engine.InvalidatePersistentTensor(_linear2Bias);
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    public override void ResetState()
    {
        _lastSinusoidalEmbed = null;
        _lastHidden = null;
        _lastInput = null;
        _linear1WeightsGradient = null;
        _linear1BiasGradient = null;
        _linear2WeightsGradient = null;
        _linear2BiasGradient = null;

        // Clear GPU cached tensors
        _gpuTimesteps = null;
        _gpuSinusoidalEmbed = null;
        _gpuHidden = null;
        _gpuPreActivation = null;
        _gpuInputShape = null;
    }

    /// <summary>
    /// Gets whether this layer supports JIT compilation.
    /// </summary>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Exports the layer as a computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List of input nodes (expects one node containing timesteps).</param>
    /// <returns>A computation node representing the time embedding output.</returns>
    /// <remarks>
    /// <para>
    /// This method builds a computation graph for the time embedding:
    /// 1. Sinusoidal embedding of timesteps
    /// 2. First linear layer (matrix multiply + bias)
    /// 3. SiLU/Swish activation
    /// 4. Second linear layer (matrix multiply + bias)
    /// </para>
    /// </remarks>
    public override Autodiff.ComputationNode<T> ExportComputationGraph(List<Autodiff.ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null || inputNodes.Count < 1)
            throw new ArgumentException("TimeEmbeddingLayer requires exactly one input node (timesteps).", nameof(inputNodes));

        var timesteps = inputNodes[0];

        // Step 1: Compute sinusoidal time embedding
        var sinEmbed = Autodiff.TensorOperations<T>.SinusoidalTimeEmbedding(timesteps, _embeddingDim);

        // Step 2: First linear layer - MatMul(sinEmbed, weights1) + bias1
        var weights1Node = Autodiff.TensorOperations<T>.Variable(_linear1Weights, "time_linear1_weights", requiresGradient: true);
        var bias1Node = Autodiff.TensorOperations<T>.Variable(_linear1Bias, "time_linear1_bias", requiresGradient: true);

        var linear1Out = Autodiff.TensorOperations<T>.MatrixMultiply(sinEmbed, weights1Node);
        var linear1WithBias = Autodiff.TensorOperations<T>.Add(linear1Out, bias1Node);

        // Step 3: SiLU/Swish activation
        var hidden = Autodiff.TensorOperations<T>.Swish(linear1WithBias);

        // Step 4: Second linear layer - MatMul(hidden, weights2) + bias2
        var weights2Node = Autodiff.TensorOperations<T>.Variable(_linear2Weights, "time_linear2_weights", requiresGradient: true);
        var bias2Node = Autodiff.TensorOperations<T>.Variable(_linear2Bias, "time_linear2_bias", requiresGradient: true);

        var linear2Out = Autodiff.TensorOperations<T>.MatrixMultiply(hidden, weights2Node);
        var output = Autodiff.TensorOperations<T>.Add(linear2Out, bias2Node);

        return output;
    }
}

using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a readout layer that performs the final mapping from features to output in a neural network.
/// </summary>
/// <remarks>
/// <para>
/// The ReadoutLayer is typically used as the final layer in a neural network to transform features 
/// extracted by previous layers into the desired output format. It applies a linear transformation 
/// (weights and bias) followed by an activation function. This layer is similar to a dense or fully 
/// connected layer but is specifically designed for outputting the final results.
/// </para>
/// <para><b>For Beginners:</b> This layer serves as the final "decision maker" in a neural network.
/// 
/// Think of the ReadoutLayer as a panel of judges in a competition:
/// - Each judge (output neuron) receives information from all contestants (input features)
/// - Each judge has their own preferences (weights) for different skills
/// - Judges combine all this information to make their final scores (outputs)
/// - An activation function then shapes these scores into the desired format
/// 
/// For example, in an image classification network:
/// - Previous layers extract features like edges, shapes, and patterns
/// - The ReadoutLayer takes all these features and combines them into class scores
/// - If there are 10 possible classes, the ReadoutLayer might have 10 outputs
/// - Each output represents the network's confidence that the image belongs to that class
/// 
/// This layer learns which features are most important for each output category during training.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.Graph)]
[LayerTask(LayerTask.GraphProcessing)]
[LayerTask(LayerTask.Projection)]
[LayerProperty(IsTrainable = true, ChangesShape = true)]
public partial class ReadoutLayer<T> : LayerBase<T>
{
    /// <summary>
    /// Tensor storing the weight parameters for connections between inputs and outputs.
    /// </summary>
    /// <remarks>
    /// This tensor has shape [outputSize, inputSize], where each row represents the weights
    /// for one output neuron. These weights determine how strongly each input feature influences
    /// each output neuron and are the primary trainable parameters of the layer.
    /// </remarks>
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _weights;

    /// <summary>
    /// Tensor storing the bias parameters for each output neuron.
    /// </summary>
    /// <remarks>
    /// This tensor has shape [outputSize], where each element is a constant value added to the
    /// weighted sum for the corresponding output neuron. Biases allow the network to shift the
    /// activation function, giving it more flexibility to fit the data.
    /// </remarks>
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _bias;

    /// <summary>
    /// Tensor storing the gradients of the loss with respect to the weight parameters.
    /// </summary>
    /// <remarks>
    /// This tensor has the same shape as the _weights tensor and stores the accumulated
    /// gradients for all weight parameters during the backward pass. These gradients are used
    /// to update the weights during the parameter update step.
    /// </remarks>
    private Tensor<T> _weightGradients;

    /// <summary>
    /// Tensor storing the gradients of the loss with respect to the bias parameters.
    /// </summary>
    /// <remarks>
    /// This tensor has the same shape as the _bias tensor and stores the accumulated
    /// gradients for all bias parameters during the backward pass. These gradients are used
    /// to update the biases during the parameter update step.
    /// </remarks>
    private Tensor<T> _biasGradients;

    /// <summary>
    /// Stores the input tensor from the most recent forward pass for use in backpropagation.
    /// </summary>
    /// <remarks>
    /// This cached input is needed during the backward pass to compute gradients. It holds the
    /// input tensor that was processed in the most recent forward pass. The tensor is null
    /// before the first forward pass or after a reset.
    /// </remarks>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stores the output tensor (post-activation) from the most recent forward pass for use in backpropagation.
    /// </summary>
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// Stores the pre-activation output tensor from the most recent forward pass.
    /// </summary>
    private Tensor<T>? _lastPreActivation;

    private int[] _originalInputShape = [];

    // GPU cached tensors for backward pass
    private Tensor<T>? _gpuInput;
    private Tensor<T>? _gpuPreActivation;
    private Tensor<T>? _gpuOutput;
    private int _gpuBatchDim;
    private int _gpuInputSize;
    private FusedActivationType _gpuActivationType;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// Always <c>true</c> for ReadoutLayer, indicating that the layer can be trained through backpropagation.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates that the ReadoutLayer has trainable parameters (weights and biases) that
    /// can be optimized during the training process using backpropagation. The gradients of these parameters
    /// are calculated during the backward pass and used to update the parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you if the layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer has values (weights and biases) that can be adjusted during training
    /// - It will improve its performance as it sees more data
    /// - It participates in the learning process of the neural network
    /// 
    /// When you train a neural network containing this layer, the weights and biases will 
    /// automatically adjust to better recognize patterns specific to your data.
    /// </para>
    /// </remarks>
    public override int ParameterCount => GetParameters().Length;
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="ReadoutLayer{T}"/> class with a scalar activation function.
    /// </summary>
    /// <param name="inputSize">The size of the input to the layer.</param>
    /// <param name="outputSize">The size of the output from the layer.</param>
    /// <param name="scalarActivation">The activation function to apply to individual elements of the output.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new ReadoutLayer with the specified dimensions and a scalar activation function.
    /// The weights are initialized with small random values, and the biases are initialized to zero. A scalar
    /// activation function is applied element-wise to each output neuron independently.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new readout layer for your neural network using a simple activation function.
    /// 
    /// When you create this layer, you specify:
    /// - inputSize: How many features come into the layer
    /// - outputSize: How many outputs you want from the layer
    /// - scalarActivation: How to transform each output (e.g., sigmoid, ReLU, tanh)
    /// 
    /// A scalar activation means each output is calculated independently. For example, in a 10-class
    /// classification problem, you might use inputSize=512 (512 features from previous layers),
    /// outputSize=10 (one for each class), and a softmax activation to get class probabilities.
    /// 
    /// The layer starts with small random weights and zero biases that will be refined during training.
    /// </para>
    /// </remarks>
    public ReadoutLayer(int inputSize, int outputSize, IActivationFunction<T> scalarActivation,
        IInitializationStrategy<T>? initializationStrategy = null)
        : base([inputSize], [outputSize], scalarActivation)
    {
        InitializationStrategy = initializationStrategy ?? InitializationStrategies<T>.Eager;

        _weights = new Tensor<T>([outputSize, inputSize]);
        _bias = new Tensor<T>([outputSize]);
        _weightGradients = new Tensor<T>([outputSize, inputSize]);
        _biasGradients = new Tensor<T>([outputSize]);

        InitializeParameters(inputSize, outputSize);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="ReadoutLayer{T}"/> class with a vector activation function.
    /// </summary>
    /// <param name="inputSize">The size of the input to the layer.</param>
    /// <param name="outputSize">The size of the output from the layer.</param>
    /// <param name="vectorActivation">The activation function to apply to the entire output vector.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new ReadoutLayer with the specified dimensions and a vector activation function.
    /// The weights are initialized with small random values, and the biases are initialized to zero. A vector
    /// activation function is applied to the entire output vector at once, which allows for interactions between
    /// different output neurons.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new readout layer for your neural network using an advanced activation function.
    /// 
    /// When you create this layer, you specify:
    /// - inputSize: How many features come into the layer
    /// - outputSize: How many outputs you want from the layer
    /// - vectorActivation: How to transform the entire output as a group
    /// 
    /// A vector activation means all outputs are calculated together, which can capture relationships between outputs.
    /// For example, softmax is a vector activation that ensures all outputs sum to 1, making them behave like probabilities.
    /// 
    /// This is particularly useful for:
    /// - Multi-class classification (using softmax activation)
    /// - Problems where outputs should be interdependent
    /// - Cases where you need to enforce specific constraints across all outputs
    /// 
    /// The layer starts with small random weights and zero biases that will be refined during training.
    /// </para>
    /// </remarks>
    public ReadoutLayer(int inputSize, int outputSize, IVectorActivationFunction<T> vectorActivation,
        IInitializationStrategy<T>? initializationStrategy = null)
        : base([inputSize], [outputSize], vectorActivation)
    {
        InitializationStrategy = initializationStrategy ?? InitializationStrategies<T>.Eager;

        _weights = new Tensor<T>([outputSize, inputSize]);
        _bias = new Tensor<T>([outputSize]);
        _weightGradients = new Tensor<T>([outputSize, inputSize]);
        _biasGradients = new Tensor<T>([outputSize]);

        InitializeParameters(inputSize, outputSize);
    }

    /// <summary>
    /// Performs the forward pass of the readout layer.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after readout processing.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the readout layer. It converts the input tensor to a vector,
    /// applies a linear transformation (weights and bias), and then applies the activation function. The input
    /// is cached for use during the backward pass. The method handles both scalar and vector activation functions.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes your data through the readout layer.
    /// 
    /// During the forward pass:
    /// 1. Your input data is flattened into a simple list of numbers
    /// 2. Each output neuron calculates a weighted sum of all inputs plus its bias
    /// 3. The activation function transforms these sums into the final outputs
    /// 
    /// The formula for each output is basically:
    /// output = activation(weights × inputs + bias)
    /// 
    /// This is similar to how a teacher might grade an exam:
    /// - Different questions have different weights (more important questions get more points)
    /// - There might be a curve applied to the final scores (activation function)
    /// 
    /// The layer saves the input for later use during training.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        _originalInputShape = input._shape;

        int inputSize = input.Shape[^1];
        if (inputSize != InputShape[0])
        {
            throw new ArgumentException(
                $"Input size {inputSize} does not match expected {InputShape[0]}.");
        }

        Tensor<T> flattenedInput;
        if (input.Rank == 1)
        {
            flattenedInput = Engine.Reshape(input, new[] { 1, inputSize });
        }
        else if (input.Rank == 2)
        {
            flattenedInput = input;
        }
        else
        {
            int batchDim = 1;
            for (int i = 0; i < input.Rank - 1; i++)
            {
                batchDim *= input.Shape[i];
            }
            flattenedInput = Engine.Reshape(input, new[] { batchDim, inputSize });
        }

        // Forward: output = input @ weights.T + bias
        var weightsTransposed = Engine.TensorTranspose(_weights);
        var matmul = Engine.TensorMatMul(flattenedInput, weightsTransposed);
        var withBias = Engine.TensorBroadcastAdd(matmul, _bias);

        _lastPreActivation = withBias;

        Tensor<T> activated = UsingVectorActivation
            ? (VectorActivation ?? throw new InvalidOperationException("VectorActivation has not been initialized.")).Activate(withBias)
            : (ScalarActivation ?? throw new InvalidOperationException("ScalarActivation has not been initialized.")).Activate(withBias);

        _lastOutput = activated;

        if (input.Rank == 1)
        {
            return Engine.Reshape(activated, [OutputShape[0]]);
        }

        if (input.Rank > 2)
        {
            var outputShape = new int[input.Rank];
            for (int i = 0; i < input.Rank - 1; i++)
            {
                outputShape[i] = _originalInputShape[i];
            }
            outputShape[^1] = OutputShape[0];
            return Engine.Reshape(activated, outputShape);
        }

        return activated;
    }

    /// <summary>
    /// Performs the forward pass on GPU using FusedLinearGpu.
    /// Supports both scalar and vector (softmax) activations.
    /// </summary>
    /// <param name="inputs">The GPU input tensors.</param>
    /// <returns>The GPU output tensor.</returns>
    public override Tensor<T> ForwardGpu(params Tensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires a DirectGpuTensorEngine.");

        var input = inputs[0];
        _originalInputShape = input._shape;

        int inputSize = input.Shape[^1];
        if (inputSize != InputShape[0])
        {
            throw new ArgumentException(
                $"Input size {inputSize} does not match expected {InputShape[0]}.");
        }

        // Compute batch dimension
        int batchDim = 1;
        for (int d = 0; d < input.Shape.Length - 1; d++)
            batchDim *= input.Shape[d];

        // Reshape input to 2D [batch, inputSize] for matrix multiply
        Tensor<T> input2D = input;
        if (input.Shape.Length == 1)
        {
            input2D = input.Reshape([1, inputSize]);
            batchDim = 1;
        }
        else if (input.Shape.Length > 2)
        {
            input2D = input.Reshape([batchDim, inputSize]);
        }

        // Transpose weights for FusedLinearGpu
        // ReadoutLayer _weights: [outputSize, inputSize] -> [inputSize, outputSize]
        var weightsT = Engine.TensorTranspose(_weights);
        int outputSize = OutputShape[0];

        Tensor<T> result;

        if (UsingVectorActivation)
        {
            // For vector activations (like softmax), apply linear then activation separately
            var preActivation = gpuEngine.FusedLinearGpu(input2D, weightsT, _bias, FusedActivationType.None);

            // Check if it's softmax and apply on GPU
            if (VectorActivation is SoftmaxActivation<T>)
            {
                result = gpuEngine.ActivationGpu(preActivation, FusedActivationType.Softmax);
            }
            else
            {
                // For other vector activations, fall back to CPU
                var cpuPreActivation = preActivation;
                var vecAct = VectorActivation ?? throw new InvalidOperationException("VectorActivation has not been initialized.");
                var cpuActivated = vecAct.Activate(cpuPreActivation);
                result = gpuEngine.UploadToGpu(cpuActivated, GpuTensorRole.Activation);
            }
        }
        else
        {
            // For scalar activations, use fused linear with activation
            var fusedActivation = GetFusedActivationType();
            result = gpuEngine.FusedLinearGpu(input2D, weightsT, _bias, fusedActivation);
        }

        // Cache state for backward pass only during training
        if (IsTrainingMode)
        {
            // Cache GPU tensors for BackwardGpu
            _gpuInput = input2D;
            var preActResult = gpuEngine.FusedLinearGpu(input2D, weightsT, _bias, FusedActivationType.None);
            _gpuPreActivation = preActResult;
            _gpuOutput = result;
            _gpuBatchDim = batchDim;
            _gpuInputSize = inputSize;
            _gpuActivationType = UsingVectorActivation && VectorActivation is SoftmaxActivation<T>
                ? FusedActivationType.Softmax
                : (UsingVectorActivation ? FusedActivationType.None : GetFusedActivationType());

            // Also cache CPU tensors for CPU backward compatibility
            _lastInput = input;
            _lastPreActivation = preActResult;
            _lastOutput = result;
        }

        // Reshape output to match expected shape
        if (input.Shape.Length == 1)
        {
            return result.Reshape([outputSize]);
        }
        else if (input.Shape.Length > 2)
        {
            var outputShape = new int[input.Shape.Length];
            for (int d = 0; d < input.Shape.Length - 1; d++)
                outputShape[d] = _originalInputShape[d];
            outputShape[^1] = outputSize;
            return result.Reshape(outputShape);
        }

        return result;
    }

    /// <summary>
    /// Updates the parameters of the readout layer using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method updates the weights and biases of the readout layer based on the gradients
    /// calculated during the backward pass. The learning rate controls the size of the parameter
    /// updates. This method should be called after the backward pass to apply the calculated updates.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the layer's internal values during training.
    /// 
    /// When updating parameters:
    /// 1. The weight values are adjusted based on their gradients
    /// 2. The bias values are adjusted based on their gradients
    /// 3. The learning rate controls how big each update step is
    /// 
    /// This is like making small adjustments based on feedback:
    /// - Weights that contributed to errors are reduced
    /// - Weights that would have helped are increased
    /// - The learning rate determines how quickly the model adapts
    /// 
    /// Smaller learning rates mean slower but more stable learning, while larger learning rates
    /// mean faster but potentially unstable learning.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // Update weights using Engine operations: w = w - lr * gradient
        var scaledWeightGradients = Engine.TensorMultiplyScalar(_weightGradients, learningRate);
        _weights = Engine.TensorSubtract(_weights, scaledWeightGradients);

        // Update biases using Engine operations: b = b - lr * gradient
        var scaledBiasGradients = Engine.TensorMultiplyScalar(_biasGradients, learningRate);
        _bias = Engine.TensorSubtract(_bias, scaledBiasGradients);
    }

    /// <summary>
    /// Gets all trainable parameters of the readout layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters (weights and biases).</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters (weights and biases) of the readout layer as a
    /// single vector. The weights are stored first, followed by the biases. This is useful for optimization
    /// algorithms that operate on all parameters at once, or for saving and loading model weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the learnable values from the readout layer.
    /// 
    /// The parameters:
    /// - Are the weights and biases that the readout layer learns during training
    /// - Control how the layer processes information
    /// - Are returned as a single list (vector)
    /// 
    /// This is useful for:
    /// - Saving the model to disk
    /// - Loading parameters from a previously trained model
    /// - Advanced optimization techniques that need access to all parameters
    /// 
    /// The weights are stored first in the vector, followed by all the bias values.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Use Vector<T>.Concatenate for efficient parameter collection
        var flatWeights = new Vector<T>(_weights.ToArray());
        var flatBias = new Vector<T>(_bias.ToArray());
        return Vector<T>.Concatenate(flatWeights, flatBias);
    }

    /// <summary>
    /// Sets the trainable parameters of the readout layer.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters (weights and biases) to set.</param>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This method sets the trainable parameters (weights and biases) of the readout layer from a single vector.
    /// The vector should contain the weight values first, followed by the bias values. This is useful for loading
    /// saved model weights or for implementing optimization algorithms that operate on all parameters at once.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the weights and biases in the readout layer.
    /// 
    /// When setting parameters:
    /// - The input must be a vector with the correct total length
    /// - The first part of the vector is used for the weights
    /// - The second part of the vector is used for the biases
    /// 
    /// This is useful for:
    /// - Loading a previously saved model
    /// - Transferring parameters from another model
    /// - Testing different parameter values
    /// 
    /// An error is thrown if the input vector doesn't have the expected number of parameters.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameterGradients()
    {
        var flatWeightGrads = new Vector<T>(_weightGradients.ToArray());
        var flatBiasGrads = new Vector<T>(_biasGradients.ToArray());
        return Vector<T>.Concatenate(flatWeightGrads, flatBiasGrads);
    }

    public override void ClearGradients()
    {
        _weightGradients = new Tensor<T>(_weights._shape);
        _biasGradients = new Tensor<T>(_bias._shape);
    }

    public override void SetParameters(Vector<T> parameters)
    {
        int outputSize = _weights.Shape[0];
        int inputSize = _weights.Shape[1];
        int weightCount = outputSize * inputSize;
        int totalParams = weightCount + _bias.Shape[0];

        if (parameters.Length != totalParams)
        {
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");
        }

        // Extract weight parameters and reshape using Tensor<T>.FromVector
        var weightParams = new Vector<T>(weightCount);
        for (int i = 0; i < weightCount; i++)
        {
            weightParams[i] = parameters[i];
        }
        _weights = Tensor<T>.FromVector(weightParams).Reshape([outputSize, inputSize]);

        // Extract bias parameters
        var biasParams = new Vector<T>(_bias.Shape[0]);
        for (int i = 0; i < _bias.Shape[0]; i++)
        {
            biasParams[i] = parameters[weightCount + i];
        }
        _bias = Tensor<T>.FromVector(biasParams);
    }

    /// <summary>
    /// Resets the internal state of the readout layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the readout layer, including the cached input from the
    /// forward pass and the gradients from the backward pass. This is useful when starting to process
    /// a new sequence or batch of data.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - Stored input from previous calculations is cleared
    /// - Calculated gradients are reset to zero
    /// - The layer forgets any information from previous batches
    /// 
    /// This is important for:
    /// - Processing a new, unrelated batch of data
    /// - Preventing information from one batch affecting another
    /// - Starting a new training episode
    /// 
    /// The weights and biases (the learned parameters) are not reset,
    /// only the temporary state information.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached CPU values from forward pass
        _lastInput = null;
        _lastOutput = null;
        _lastPreActivation = null;
        _originalInputShape = [];

        // Clear cached GPU tensors
        _gpuInput = null;
        _gpuPreActivation = null;
        _gpuOutput = null;

        // Reset gradients using Tensor<T>
        _weightGradients = new Tensor<T>([_weights.Shape[0], _weights.Shape[1]]);
        _weightGradients.Fill(NumOps.Zero);
        _biasGradients = new Tensor<T>([_bias.Shape[0]]);
        _biasGradients.Fill(NumOps.Zero);
    }

    /// <summary>
    /// Initializes the weights and biases of the readout layer with small random values and zeros.
    /// </summary>
    /// <param name="inputSize">The size of the input to the layer.</param>
    /// <param name="outputSize">The size of the output from the layer.</param>
    /// <remarks>
    /// This private method initializes the weights with small random values centered around zero
    /// and initializes the biases to zero. This provides a good starting point for training
    /// the readout layer, avoiding issues like vanishing or exploding gradients.
    /// </remarks>
    private void InitializeParameters(int inputSize, int outputSize)
    {
        InitializeLayerWeights(_weights, outputSize, inputSize);
        InitializeLayerBiases(_bias);

        RegisterTrainableParameter(_weights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_bias, PersistentTensorRole.Biases);
    }
}

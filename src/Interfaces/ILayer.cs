using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for neural network layers in the AiDotNet framework.
/// </summary>
/// <typeparam name="T">The numeric type used for computations (typically float or double).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> A neural network is made up of layers, similar to how a sandwich has layers.
/// Each layer processes data in some way and passes it to the next layer.
/// 
/// This interface defines what all layers must be able to do, regardless of their specific type.
/// Think of it as a checklist of abilities that every layer must have to work within our neural network.
/// </remarks>
public interface ILayer<T> : IJitCompilable<T>, IDiagnosticsProvider, IWeightLoadable<T>
{
    /// <summary>
    /// Gets the shape (dimensions) of the input data expected by this layer.
    /// </summary>
    /// <returns>An array of integers representing the input dimensions.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This tells us what size and shape of data this layer expects to receive.
    /// For example, if processing images, this might be [3, 28, 28] for 28×28 pixel images with 3 color channels.
    /// </remarks>
    int[] GetInputShape();

    /// <summary>
    /// Gets the shape (dimensions) of the output data produced by this layer.
    /// </summary>
    /// <returns>An array of integers representing the output dimensions.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This tells us what size and shape of data this layer will produce.
    /// The output shape often differs from the input shape because the layer may transform the data.
    /// For example, a pooling layer might reduce the dimensions from [3, 28, 28] to [3, 14, 14].
    /// </remarks>
    int[] GetOutputShape();

    /// <summary>
    /// Gets the weight tensor for layers that have trainable weights.
    /// </summary>
    /// <returns>The weight tensor, or null if the layer has no weights.</returns>
    Tensor<T>? GetWeights();

    /// <summary>
    /// Gets the bias tensor for layers that have trainable biases.
    /// </summary>
    /// <returns>The bias tensor, or null if the layer has no biases.</returns>
    Tensor<T>? GetBiases();


    /// <summary>
    /// Processes input data through the layer during the forward pass.
    /// </summary>
    /// <param name="input">The input tensor to be processed.</param>
    /// <returns>The output tensor after processing.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This is where the actual work happens when data flows through the network.
    /// Each layer takes some input, applies its specific operation, and produces an output.
    /// 
    /// For example:
    /// - A convolutional layer detects patterns in images
    /// - A pooling layer reduces the size of data
    /// - A dense layer combines all inputs with weights to produce outputs
    /// 
    /// This is called the "forward pass" because data is moving forward through the network.
    /// </remarks>
    Tensor<T> Forward(Tensor<T> input);

    /// <summary>
    /// Processes input data through the layer with automatic mixed-precision handling.
    /// </summary>
    /// <param name="input">The input tensor to be processed.</param>
    /// <returns>The output tensor after processing with appropriate precision.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method wraps the standard Forward pass with automatic
    /// precision handling for mixed-precision training.
    ///
    /// When mixed-precision is active:
    /// - Layers check if they need full precision (FP32) based on their type
    /// - Normalization layers (BatchNorm, LayerNorm) typically stay in FP32
    /// - Other layers may use reduced precision (FP16) for faster computation
    ///
    /// This enables faster training on modern GPUs while maintaining numerical stability.
    ///
    /// <b>Note:</b> This is a new interface member added for mixed-precision support.
    /// Implementers should call <see cref="Forward"/> by default if no special precision
    /// handling is needed.
    /// </remarks>
    Tensor<T> ForwardWithPrecisionCheck(Tensor<T> input);

    /// <summary>
    /// Gets the name of this layer for mixed-precision policy lookup.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This name is used to determine whether the layer should
    /// use full precision (FP32) or reduced precision (FP16/FP8) during mixed-precision training.
    ///
    /// <b>Note:</b> This is a new interface member added for mixed-precision support.
    /// Implementers should return a unique name that can be matched against patterns
    /// in the precision policy. A good default is <c>GetType().Name + "_" + instanceId</c>
    /// to distinguish multiple instances of the same layer type.
    /// </remarks>
    string LayerName { get; }

    /// <summary>
    /// Performs a GPU-resident forward pass, keeping the result on the GPU.
    /// </summary>
    /// <param name="inputs">GPU-resident input tensor(s). Most layers use only the first input,
    /// but some layers like cross-attention require multiple inputs.</param>
    /// <returns>GPU-resident output tensor. Caller is responsible for disposal.</returns>
    /// <exception cref="InvalidOperationException">
    /// Thrown if <see cref="CanExecuteOnGpu"/> is false or no GPU backend is available.
    /// </exception>
    /// <remarks>
    /// <b>For Beginners:</b> This is like the regular Forward method, but optimized for GPU execution.
    ///
    /// Instead of copying data back and forth between CPU and GPU after each layer,
    /// this method keeps everything on the GPU until you're done processing. This can be
    /// 10-50x faster for deep networks with many layers!
    ///
    /// Use this method when:
    /// - You have a GPU available and want maximum performance
    /// - You're running inference (predictions) on batches of data
    /// - You have multiple layers that all support GPU execution
    ///
    /// The result stays on the GPU until you call ToTensor() to download it to CPU memory.
    /// </remarks>
    IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs);

    /// <summary>
    /// Gets whether this layer can execute on GPU.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Not all layers support GPU acceleration yet. This property tells you
    /// if this particular layer can run on the GPU.
    ///
    /// If this returns true, you can use <see cref="ForwardGpu"/> for faster execution.
    /// If this returns false, the layer will fall back to CPU execution.
    ///
    /// Common layers like Dense, Convolutional, and Attention layers typically support GPU.
    /// Specialized layers may only support CPU execution.
    /// </remarks>
    bool CanExecuteOnGpu { get; }

    /// <summary>
    /// Calculates gradients during the backward pass of backpropagation.
    /// </summary>
    /// <param name="outputGradient">The gradient flowing back from the next layer.</param>
    /// <returns>The gradient to pass to the previous layer.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> During training, neural networks learn by adjusting their parameters.
    /// To know how to adjust them, we need to calculate how much each parameter affects the error.
    /// 
    /// This method handles the "backward pass" where error information flows backward through the network.
    /// It takes the gradient (direction of error) from the next layer and calculates:
    /// 1. How to update this layer's parameters
    /// 2. What gradient to pass to the previous layer
    /// 
    /// Think of it like tracing back through a series of decisions to figure out which ones led to a mistake.
    /// </remarks>
    Tensor<T> Backward(Tensor<T> outputGradient);

    /// <summary>
    /// Updates the layer's parameters using the specified learning rate.
    /// </summary>
    /// <param name="learningRate">The learning rate that controls how much parameters change.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This method adjusts the layer's internal values (weights and biases) during training.
    /// 
    /// The learning rate is like the size of the steps you take when adjusting parameters:
    /// - A large learning rate means big changes (faster learning but might overshoot)
    /// - A small learning rate means small changes (more precise but slower learning)
    /// 
    /// This method uses the gradients calculated during the backward pass to update parameters.
    /// </remarks>
    void UpdateParameters(T learningRate);

    /// <summary>
    /// Updates the layer's parameters using the provided parameter values.
    /// </summary>
    /// <param name="parameters">The new parameter values to apply.</param>
    /// <remarks>
    /// <b>For Beginners:</b> While the other UpdateParameters method calculates new values based on gradients,
    /// this method lets you directly specify what the new parameter values should be.
    /// 
    /// This is useful for advanced optimization techniques where parameters are calculated externally,
    /// or when loading pre-trained parameters.
    /// </remarks>
    void UpdateParameters(Vector<T> parameters);

    /// <summary>
    /// Gets the total number of trainable parameters in this layer.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This tells you how many adjustable values this layer has.
    /// 
    /// For example:
    /// - A dense layer with 10 inputs and 5 outputs has 10×5=50 weights plus 5 biases = 55 parameters
    /// - A convolutional layer with 16 3×3 filters has 16×3×3=144 weights plus 16 biases = 160 parameters
    /// - A pooling layer has 0 parameters (it just selects values, no weights to adjust)
    /// 
    /// More parameters means the layer can learn more complex patterns, but also requires more data to train effectively.
    /// </remarks>
    int ParameterCount { get; }

    /// <summary>
    /// Saves the layer's configuration and parameters to a binary stream.
    /// </summary>
    /// <param name="writer">The binary writer to write the data to.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This method saves all the information about the layer so it can be loaded later.
    /// 
    /// It's like saving your progress in a video game - all the important information about the layer
    /// (its type, shape, and learned parameters) gets written to a file so you can continue using
    /// the same trained layer later without having to train it again.
    /// </remarks>
    void Serialize(BinaryWriter writer);

    /// <summary>
    /// Loads the layer's configuration and parameters from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader to read the data from.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This method loads previously saved information about the layer.
    /// 
    /// It's the counterpart to Serialize - if Serialize is like saving your game progress,
    /// Deserialize is like loading that saved game. It restores all the layer's settings and
    /// learned parameters from a file.
    /// </remarks>
    void Deserialize(BinaryReader reader);

    /// <summary>
    /// Gets the activation functions used by this layer.
    /// </summary>
    /// <returns>A collection of activation functions used by this layer.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Activation functions determine how a neuron responds to its input.
    /// 
    /// They introduce non-linearity, which is crucial for neural networks to learn complex patterns.
    /// Common activation functions include:
    /// - ReLU: Returns 0 for negative inputs, or the input value for positive inputs
    /// - Sigmoid: Squashes values between 0 and 1 (useful for probabilities)
    /// - Tanh: Squashes values between -1 and 1
    /// 
    /// This method tells you which activation functions this layer uses.
    /// </remarks>
    IEnumerable<ActivationFunction> GetActivationTypes();

    /// <summary>
    /// Gets all trainable parameters of the layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method returns all the values that can be adjusted during training
    /// (weights and biases) as a single list.
    /// 
    /// This is useful for:
    /// 1. Saving and loading models
    /// 2. Advanced optimization techniques
    /// 3. Analyzing or visualizing the learned parameters
    /// 
    /// Some layers (like pooling layers) might return an empty vector because they have no trainable parameters.
    /// </remarks>
    Vector<T> GetParameters();

    /// <summary>
    /// Indicates whether this layer supports training operations.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This property tells you whether this layer can be trained (adjusted) during learning.
    /// 
    /// Most layers (like convolutional or dense layers) return true because they have weights that need to be learned.
    /// Some layers (like input layers or certain normalization layers) might return false because they
    /// don't have parameters that need training or they should remain fixed during training.
    /// </remarks>
    bool SupportsTraining { get; }

    /// <summary>
    /// Sets the layer to training or evaluation mode.
    /// </summary>
    /// <param name="isTraining">True to set the layer to training mode, false for evaluation mode.</param>
    /// <remarks>
    /// <b>For Beginners:</b> Some layers behave differently during training versus when making predictions.
    /// 
    /// For example:
    /// - Dropout layers randomly disable neurons during training but use all neurons during evaluation
    /// - Batch normalization layers collect statistics during training but use fixed statistics during evaluation
    /// 
    /// This method switches the layer between these two modes.
    /// </remarks>
    void SetTrainingMode(bool isTraining);

    /// <summary>
    /// Gets the gradients of all trainable parameters.
    /// </summary>
    /// <returns>A vector containing the gradients for all trainable parameters.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> During training, we calculate how much each parameter affects the error (the gradient).
    /// 
    /// This method returns those gradients, which show:
    /// 1. Which direction to adjust each parameter (increase or decrease)
    /// 2. How strongly each parameter affects the error
    /// 
    /// These gradients are used to update the parameters during training.
    /// </remarks>
    Vector<T> GetParameterGradients();

    /// <summary>
    /// Clears all accumulated gradients in the layer.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> During training, gradients are calculated and accumulated.
    /// 
    /// This method resets those accumulated gradients to zero, which is typically done:
    /// 1. At the start of each training batch
    /// 2. After parameters have been updated
    /// 3. When switching to a new training example
    /// 
    /// It's like wiping a whiteboard clean before starting a new calculation.
    /// </remarks>
    void ClearGradients();

    /// <summary>
    /// Sets all trainable parameters of the layer to the specified values.
    /// </summary>
    /// <param name="parameters">The new parameter values to set.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This method lets you directly set all the weights and biases of the layer.
    /// 
    /// This is useful when:
    /// 1. Loading a pre-trained model
    /// 2. Initializing parameters in a specific way
    /// 3. Testing how different parameter values affect performance
    /// 
    /// It's like replacing all the settings in the layer at once.
    /// </remarks>
    void SetParameters(Vector<T> parameters);

    /// <summary>
    /// Resets the internal state of the layer to its initial condition.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Some layers (like recurrent layers) keep track of information from previous inputs.
    /// This is called the layer's "state" - it's like the layer's memory of what it has seen before.
    /// 
    /// This method clears that memory, making the layer forget all previous inputs. It's useful when:
    /// 1. Starting to process a new sequence of data
    /// 2. You want predictions to be independent of previous inputs
    /// 3. Testing the layer with different inputs and want to start fresh each time
    /// 
    /// For example, if you're processing sentences one at a time, you might want to reset
    /// the state between different sentences so information from one sentence doesn't
    /// affect how the next sentence is processed.
    /// </remarks>
    void ResetState();

    #region GPU Training Support

    /// <summary>
    /// Gets whether this layer supports GPU-resident training (forward, backward, and parameter updates on GPU).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This tells you if the layer can be trained entirely on the GPU.
    /// GPU training is much faster because:
    /// - Data stays on GPU between forward and backward passes
    /// - No expensive CPU-GPU transfers during each training step
    /// - GPU kernels handle all computation in parallel
    /// 
    /// If this returns true, you can use the GPU training methods (BackwardGpu, UpdateParametersGpu).
    /// If false, training will fall back to the CPU implementation.
    /// </remarks>
    bool SupportsGpuTraining { get; }

    /// <summary>
    /// Updates the layer's parameters on GPU using the specified optimizer configuration.
    /// </summary>
    /// <param name="config">The GPU optimizer configuration specifying the update algorithm and hyperparameters.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This method updates the layer's weights and biases entirely on the GPU.
    /// 
    /// The optimizer config determines how the update is done:
    /// - SGD: Simple gradient descent with optional momentum
    /// - Adam: Adaptive learning rates with moment estimates
    /// - AdamW: Adam with decoupled weight decay
    /// - And more...
    /// 
    /// Using this method keeps all training computation on the GPU, which is much faster
    /// than downloading gradients to CPU, computing updates, and uploading back.
    /// 
    /// Call BackwardGpu first to compute gradients, then call this to update weights.
    /// </remarks>
    void UpdateParametersGpu(IGpuOptimizerConfig config);

    /// <summary>
    /// Uploads the layer's weights and biases to GPU memory for GPU-resident training.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Call this once before starting GPU training.
    /// 
    /// It copies the layer's learned values (weights and biases) from CPU memory to GPU memory.
    /// After this, training can happen entirely on the GPU without CPU involvement.
    /// 
    /// Also allocates GPU buffers for:
    /// - Gradients (computed during BackwardGpu)
    /// - Optimizer state (momentum, Adam moments, etc.)
    /// </remarks>
    void UploadWeightsToGpu();

    /// <summary>
    /// Downloads the layer's weights and biases from GPU memory back to CPU.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Call this after GPU training to sync the learned values back to CPU.
    /// 
    /// During GPU training, only the GPU copies of weights are updated.
    /// Call this to:
    /// - Save the model to disk
    /// - Switch to CPU inference
    /// - Inspect the trained weights
    /// 
    /// This is relatively expensive, so only call it when necessary (not every batch).
    /// </remarks>
    void DownloadWeightsFromGpu();

    /// <summary>
    /// Resets the GPU gradient accumulators to zero.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Call this at the start of each training batch.
    /// 
    /// Each batch computes new gradients. Before processing a new batch:
    /// 1. Call this to clear old gradients
    /// 2. Run ForwardGpu on the batch
    /// 3. Run BackwardGpu to compute new gradients
    /// 4. Call UpdateParametersGpu to apply the updates
    /// 
    /// If you forget to zero gradients, they accumulate and training goes wrong!
    /// </remarks>
    void ZeroGradientsGpu();

    #endregion
}

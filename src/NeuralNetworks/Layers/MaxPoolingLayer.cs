namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements a max pooling layer for neural networks, which reduces the spatial dimensions
/// of the input by taking the maximum value in each pooling window.
/// </summary>
/// <typeparam name="T">The numeric type used for computations (typically float or double).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> A max pooling layer helps reduce the size of data flowing through a neural network
/// while keeping the most important information. It works by dividing the input into small windows
/// (determined by the pool size) and keeping only the largest value from each window.
/// 
/// Think of it like summarizing a detailed picture: instead of describing every pixel,
/// you just point out the most noticeable feature in each area of the image.
/// 
/// This helps the network:
/// 1. Focus on the most important features
/// 2. Reduce computation needs
/// 3. Make the model more robust to small changes in input position
/// </remarks>
public class MaxPoolingLayer<T> : LayerBase<T>
{
    /// <summary>
    /// Gets the size of the pooling window.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This determines how large of an area we look at when selecting the maximum value.
    /// For example, a pool size of 2 means we look at 2×2 squares of the input.
    /// </remarks>
    public int PoolSize { get; private set; }

    /// <summary>
    /// Gets the step size when moving the pooling window across the input.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This controls how much we move our window each time.
    /// For example, a stride of 2 means we move the window 2 pixels at a time,
    /// which reduces the output size to half of the input size (assuming pool size is also 2).
    /// </remarks>
    public int Strides { get; private set; }

    /// <summary>
    /// Indicates whether this layer supports training operations.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This property tells the neural network system whether this layer
    /// can be trained (adjusted) during the learning process. Max pooling layers don't have
    /// parameters to train, but they do support the training process by allowing gradients
    /// to flow backward through them.
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Stores the indices of the maximum values found during the forward pass.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This keeps track of which input value was the maximum in each pooling window.
    /// We need this information during the backward pass to know where to send the gradients.
    /// </remarks>
    private Tensor<int> _maxIndices = default!;

    /// <summary>
    /// Creates a new max pooling layer with the specified parameters.
    /// </summary>
    /// <param name="inputShape">The shape of the input data (channels, height, width).</param>
    /// <param name="poolSize">The size of the pooling window.</param>
    /// <param name="strides">The step size when moving the pooling window.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This constructor sets up the max pooling layer with your chosen settings.
    /// It calculates what the output shape will be based on your input shape, pool size, and strides.
    /// </remarks>
    public MaxPoolingLayer(int[] inputShape, int poolSize, int strides) 
        : base(inputShape, CalculateOutputShape(inputShape, poolSize, strides))
    {
        PoolSize = poolSize;
        Strides = strides;
        _maxIndices = new Tensor<int>(OutputShape);
    }

    /// <summary>
    /// Calculates the output shape based on the input shape and pooling parameters.
    /// </summary>
    /// <param name="inputShape">The shape of the input data.</param>
    /// <param name="poolSize">The size of the pooling window.</param>
    /// <param name="strides">The step size when moving the pooling window.</param>
    /// <returns>The calculated output shape.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method figures out how big the output will be after max pooling.
    /// The formula used is a standard way to calculate how many complete windows fit into the input,
    /// taking into account the stride (step size).
    /// </remarks>
    private static int[] CalculateOutputShape(int[] inputShape, int poolSize, int strides)
    {
        int outputHeight = (inputShape[1] - poolSize) / strides + 1;
        int outputWidth = (inputShape[2] - poolSize) / strides + 1;

        return [inputShape[0], outputHeight, outputWidth];
    }

    /// <summary>
    /// Performs the forward pass of the max pooling operation.
    /// </summary>
    /// <param name="input">The input tensor to apply max pooling to.</param>
    /// <returns>The output tensor after max pooling.</returns>
    /// <exception cref="ArgumentException">Thrown when the input tensor doesn't have 3 dimensions.</exception>
    /// <remarks>
    /// <b>For Beginners:</b> This is where the actual max pooling happens. For each small window in the input:
    /// 1. We look at all values in that window
    /// 2. We find the largest value
    /// 3. We put that value in the output
    /// 4. We remember where that maximum value was located (for the backward pass)
    /// 
    /// The method processes the input channel by channel, sliding the pooling window across
    /// the height and width dimensions.
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        if (input.Shape.Length != 3)
            throw new ArgumentException("Input tensor must have 3 dimensions (channels, height, width)");

        int channels = input.Shape[0];
        int inputHeight = input.Shape[1];
        int inputWidth = input.Shape[2];
        int outputHeight = OutputShape[1];
        int outputWidth = OutputShape[2];

        var output = new Tensor<T>(OutputShape);
        _maxIndices = new Tensor<int>(OutputShape);

        for (int c = 0; c < channels; c++)
        {
            for (int h = 0; h < outputHeight; h++)
            {
                for (int w = 0; w < outputWidth; w++)
                {
                    T maxVal = NumOps.Zero;
                    int maxIdx = -1;

                    for (int ph = 0; ph < PoolSize; ph++)
                    {
                        for (int pw = 0; pw < PoolSize; pw++)
                        {
                            int ih = h * Strides + ph;
                            int iw = w * Strides + pw;

                            if (ih < inputHeight && iw < inputWidth)
                            {
                                T val = input[c, ih, iw];
                                if (maxIdx == -1 || NumOps.GreaterThan(maxVal, NumOps.Zero))
                                {
                                    maxVal = val;
                                    maxIdx = ph * PoolSize + pw;
                                }
                            }
                        }
                    }

                    output[c, h, w] = maxVal;
                    _maxIndices[c, h, w] = maxIdx;
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Performs the backward pass of the max pooling operation.
    /// </summary>
    /// <param name="outputGradient">The gradient flowing back from the next layer.</param>
    /// <returns>The gradient to pass to the previous layer.</returns>
    /// <exception cref="ArgumentException">Thrown when the output gradient tensor doesn't have 3 dimensions.</exception>
    /// <remarks>
    /// <b>For Beginners:</b> During training, neural networks need to adjust their parameters based on
    /// how much error they made. This adjustment flows backward through the network.
    /// 
    /// In max pooling, only the maximum value from each window contributed to the output.
    /// So during the backward pass, the gradient only flows back to that maximum value's position.
    /// All other positions receive zero gradient because they didn't contribute to the output.
    /// 
    /// Think of it like giving credit only to the team member who contributed the most to a project.
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (outputGradient.Shape.Length != 3)
            throw new ArgumentException("Output gradient tensor must have 3 dimensions (channels, height, width)");

        int channels = InputShape[0];
        int inputHeight = InputShape[1];
        int inputWidth = InputShape[2];

        var inputGradient = new Tensor<T>(InputShape);

        for (int c = 0; c < channels; c++)
        {
            for (int h = 0; h < outputGradient.Shape[1]; h++)
            {
                for (int w = 0; w < outputGradient.Shape[2]; w++)
                {
                    int maxIdx = _maxIndices[c, h, w];
                    int ph = maxIdx / PoolSize;
                    int pw = maxIdx % PoolSize;

                    int ih = h * Strides + ph;
                    int iw = w * Strides + pw;

                    if (ih < inputHeight && iw < inputWidth)
                    {
                        inputGradient[c, ih, iw] = outputGradient[c, h, w];
                    }
                }
            }
        }

        return inputGradient;
    }

    /// <summary>
    /// Saves the layer's configuration to a binary stream.
    /// </summary>
    /// <param name="writer">The binary writer to write the data to.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This method saves the layer's settings (pool size and stride)
    /// so that you can reload the exact same layer later. It's like saving your game
    /// progress so you can continue from where you left off.
    /// </remarks>
    public override void Serialize(BinaryWriter writer)
    {
        base.Serialize(writer);
        writer.Write(PoolSize);
        writer.Write(Strides);
    }

    /// <summary>
    /// Loads the layer's configuration from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader to read the data from.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This method loads previously saved settings for the layer.
    /// It's the counterpart to Serialize - if Serialize is like saving your game,
    /// Deserialize is like loading that saved game.
    /// </remarks>
    public override void Deserialize(BinaryReader reader)
    {
        base.Deserialize(reader);
        PoolSize = reader.ReadInt32();
        Strides = reader.ReadInt32();
    }

    /// <summary>
    /// Returns the activation functions used by this layer.
    /// </summary>
    /// <returns>An empty collection since max pooling layers don't use activation functions.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Activation functions are mathematical operations that determine
    /// the output of a neural network node. They introduce non-linearity, which helps
    /// neural networks learn complex patterns.
    /// 
    /// However, max pooling layers don't use activation functions - they simply
    /// select the maximum value from each window. That's why this method returns an empty collection.
    /// </remarks>
    public override IEnumerable<ActivationFunction> GetActivationTypes()
    {
        // Max pooling doesn't have an activation function
        return [];
    }

    /// <summary>
    /// Updates the layer's parameters during training.
    /// </summary>
    /// <param name="learningRate">The learning rate that controls how much parameters change.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This method is part of the neural network training process.
    /// 
    /// During training, most layers need to update their internal values (parameters) to learn
    /// from data. However, max pooling layers don't have any trainable parameters - they just
    /// pass through the maximum values from each window.
    /// 
    /// Think of it like a simple rule that doesn't need to be adjusted: "Always pick the largest number."
    /// Since this rule never changes, there's nothing to update in this method.
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // Max pooling layer doesn't have trainable parameters
    }

    /// <summary>
    /// Gets all trainable parameters of the layer.
    /// </summary>
    /// <returns>An empty vector since max pooling layers have no trainable parameters.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method returns all the values that can be adjusted during training.
    /// 
    /// Many neural network layers have weights and biases that get updated as the network learns.
    /// However, max pooling layers simply select the maximum value from each window - there are
    /// no weights or biases to adjust.
    /// 
    /// This is why the method returns an empty vector (essentially a list with no elements).
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // MaxPoolingLayer has no trainable parameters
        return Vector<T>.Empty();
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This method clears any information the layer has stored from previous
    /// calculations.
    /// 
    /// During the forward pass, the max pooling layer remembers which positions had the maximum
    /// values (stored in _maxIndices). This is needed for the backward pass.
    /// 
    /// Resetting the state clears this memory, which is useful when:
    /// 1. Starting a new training session
    /// 2. Processing a new batch of data
    /// 3. Switching from training to evaluation mode
    /// 
    /// It's like wiping a whiteboard clean before starting a new calculation.
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward pass
        _maxIndices = new Tensor<int>(OutputShape);
    }
}
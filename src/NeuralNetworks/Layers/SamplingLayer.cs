namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements a sampling (pooling) layer for neural networks that reduces the spatial dimensions of input data.
/// </summary>
/// <remarks>
/// <para>
/// For Beginners: A sampling layer (also called pooling layer) is like a summarizer for images. 
/// It takes a group of pixels and combines them into a single value, making the image smaller 
/// while keeping the important information. This helps the neural network focus on what matters 
/// and reduces computation.
/// </para>
/// <para>
/// Think of it like looking at a high-resolution photo: if you squint your eyes (pooling), 
/// you still see the main objects but with less detail. This makes processing faster and helps 
/// the network focus on patterns rather than exact pixel values.
/// </para>
/// <para>
/// Common types of pooling:
/// - Max pooling: Takes the brightest/strongest value from each group (like finding the most important feature)
/// - Average pooling: Takes the average of all values (like getting the general impression)
/// - L2 Norm pooling: A special mathematical way to combine values that emphasizes larger values
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used in the layer (e.g., float, double).</typeparam>
public class SamplingLayer<T> : LayerBase<T>
{
    /// <summary>
    /// Gets the size of the pooling window.
    /// </summary>
    /// <remarks>
    /// For Beginners: This is how many pixels the layer looks at together in each step.
    /// For example, a pool size of 2 means it looks at 2×2 squares of pixels at once.
    /// </remarks>
    public int PoolSize { get; private set; }

    /// <summary>
    /// Gets the step size when moving the pooling window across the input.
    /// </summary>
    /// <remarks>
    /// For Beginners: This is how many pixels the layer moves each time it takes a sample.
    /// If strides = 2, it moves 2 pixels to the right or down after each operation.
    /// </remarks>
    public int Strides { get; private set; }

    /// <summary>
    /// Gets the type of sampling/pooling operation performed by this layer.
    /// </summary>
    /// <remarks>
    /// For Beginners: This determines how the layer combines multiple values into one:
    /// - Max: Takes the largest value (like finding the brightest pixel)
    /// - Average: Takes the average of all values (like finding the general brightness)
    /// - L2Norm: A special mathematical way that emphasizes larger values
    /// </remarks>
    public SamplingType SamplingType { get; private set; }

    /// <summary>
    /// Indicates whether this layer supports training (updating through backpropagation).
    /// </summary>
    /// <remarks>
    /// For Beginners: This tells if the layer can learn from its mistakes. Sampling layers
    /// don't have parameters to learn, but they do participate in the learning process
    /// by passing information backward.
    /// </remarks>
    public override bool SupportsTraining => true;

    // Private fields for internal operations
    private Func<Tensor<T>, Tensor<T>>? _forwardStrategy;
    private Func<Tensor<T>, Tensor<T>, Tensor<T>>? _backwardStrategy;
    private Tensor<T>? _lastInput;
    private Tensor<T>? _maxIndices;

    /// <summary>
    /// Creates a new sampling layer with the specified parameters.
    /// </summary>
    /// <param name="inputShape">The shape of the input data [channels, height, width].</param>
    /// <param name="poolSize">The size of the pooling window.</param>
    /// <param name="strides">The step size when moving the pooling window.</param>
    /// <param name="samplingType">The type of sampling operation to perform.</param>
    /// <remarks>
    /// For Beginners: This sets up the sampling layer with your chosen settings.
    /// - inputShape: The dimensions of your input data (like image size)
    /// - poolSize: How many pixels to look at together
    /// - strides: How far to move after each operation
    /// - samplingType: Which method to use for combining values (Max, Average, or L2Norm)
    /// </remarks>
    public SamplingLayer(int[] inputShape, int poolSize, int strides, SamplingType samplingType) 
        : base(inputShape, CalculateOutputShape(inputShape, poolSize, strides))
    {
        PoolSize = poolSize;
        Strides = strides;
        SamplingType = samplingType;

        SetStrategies();
    }

    /// <summary>
    /// Calculates the output shape based on the input shape and pooling parameters.
    /// </summary>
    /// <param name="inputShape">The shape of the input data.</param>
    /// <param name="poolSize">The size of the pooling window.</param>
    /// <param name="strides">The step size when moving the pooling window.</param>
    /// <returns>The calculated output shape.</returns>
    /// <remarks>
    /// For Beginners: This calculates how big the output will be after pooling.
    /// The formula is: (input_size - pool_size) / strides + 1
    /// 
    /// For example, if you have a 10×10 image with pool_size=2 and strides=2,
    /// the output will be 5×5 ((10-2)/2+1 = 5).
    /// </remarks>
    private static int[] CalculateOutputShape(int[] inputShape, int poolSize, int strides)
    {
        int outputHeight = (inputShape[1] - poolSize) / strides + 1;
        int outputWidth = (inputShape[2] - poolSize) / strides + 1;

        return [inputShape[0], outputHeight, outputWidth];
    }

    /// <summary>
    /// Sets up the appropriate forward and backward computation strategies based on the sampling type.
    /// </summary>
    /// <remarks>
    /// For Beginners: This method chooses the right algorithm based on whether you selected
    /// Max, Average, or L2Norm pooling. Each type needs different calculations for both the
    /// forward pass (processing data) and backward pass (learning).
    /// </remarks>
    private void SetStrategies()
    {
        switch (SamplingType)
        {
            case SamplingType.Max:
                _forwardStrategy = MaxPoolForward;
                _backwardStrategy = MaxPoolBackward;
                break;
            case SamplingType.Average:
                _forwardStrategy = AveragePoolForward;
                _backwardStrategy = AveragePoolBackward;
                break;
            case SamplingType.L2Norm:
                _forwardStrategy = L2NormPoolForward;
                _backwardStrategy = L2NormPoolBackward;
                break;
            default:
                throw new NotImplementedException($"Sampling type {SamplingType} not implemented.");
        }
    }

    /// <summary>
    /// Performs the forward pass of the sampling layer, processing the input data.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after sampling/pooling.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the forward strategy is not set.</exception>
    /// <remarks>
    /// For Beginners: This is where the actual pooling happens. The layer takes your input
    /// (like an image) and applies the pooling operation you selected (Max, Average, or L2Norm)
    /// to make a smaller version that keeps the important features.
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        if (_forwardStrategy == null)
        {
            throw new InvalidOperationException("Forward strategy is not set.");
        }

        _lastInput = input;
        return _forwardStrategy(input);
    }

    /// <summary>
    /// Performs the backward pass of the sampling layer, propagating gradients back to the previous layer.
    /// </summary>
    /// <param name="outputGradient">The gradient from the next layer.</param>
    /// <returns>The gradient to pass to the previous layer.</returns>
    /// <exception cref="InvalidOperationException">Thrown when backward strategy is not set or Forward hasn't been called.</exception>
    /// <remarks>
    /// For Beginners: This is part of how neural networks learn. When the network makes a mistake,
    /// it needs to figure out how each layer contributed to that mistake. This method helps pass
    /// that information backward through the sampling layer to the previous layers.
    /// 
    /// Think of it like tracing back through your steps to find where you went wrong.
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_backwardStrategy == null)
        {
            throw new InvalidOperationException("Backward strategy is not set.");
        }

        if (_lastInput == null)
        {
            throw new InvalidOperationException("Backward called before Forward.");
        }

        return _backwardStrategy(_lastInput, outputGradient);
    }

    /// <summary>
    /// Implements the forward pass for max pooling.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>The output tensor after max pooling.</returns>
    /// <remarks>
    /// For Beginners: Max pooling looks at each group of pixels and keeps only the strongest/brightest one.
    /// For example, if you have four pixels with values [2,5,1,3], max pooling would keep only the 5.
    /// 
    /// This helps the network focus on the most prominent features, like edges or specific patterns.
    /// </remarks>
    private Tensor<T> MaxPoolForward(Tensor<T> input)
    {
        var output = new Tensor<T>(OutputShape);
        var maxIndices = new Tensor<T>(OutputShape);

        for (int b = 0; b < OutputShape[0]; b++)
        {
            for (int h = 0; h < OutputShape[1]; h++)
            {
                for (int w = 0; w < OutputShape[2]; w++)
                {
                    T maxVal = NumOps.MinValue;
                    int maxIndex = -1;
                    for (int ph = 0; ph < PoolSize; ph++)
                    {
                        for (int pw = 0; pw < PoolSize; pw++)
                        {
                            int ih = h * Strides + ph;
                            int iw = w * Strides + pw;
                            if (ih < InputShape[1] && iw < InputShape[2])
                            {
                                T val = input[b, ih, iw];
                                if (maxIndex == -1 || NumOps.GreaterThan(val, maxVal))
                                {
                                    maxVal = val;
                                    maxIndex = ph * PoolSize + pw;
                                }
                            }
                        }
                    }

                    output[b, h, w] = maxVal;
                    maxIndices[b, h, w] = NumOps.FromDouble(maxIndex);
                }
            }
        }

        _maxIndices = maxIndices;
        return output;
    }

    /// <summary>
    /// Performs the backward pass for max pooling, propagating gradients back to the previous layer.
    /// </summary>
    /// <param name="input">The original input tensor from the forward pass.</param>
    /// <param name="outputGradient">The gradient from the next layer.</param>
    /// <returns>The gradient to pass to the previous layer.</returns>
    /// <exception cref="InvalidOperationException">Thrown when called before MaxPoolForward.</exception>
    /// <remarks>
    /// For Beginners: During max pooling, only the maximum value from each window was kept.
    /// When we go backward, we need to send the gradient only to those maximum values that
    /// contributed to the output, while all other values receive zero gradient.
    /// 
    /// Think of it like giving credit only to the "winners" from each pooling window.
    /// </remarks>
    private Tensor<T> MaxPoolBackward(Tensor<T> input, Tensor<T> outputGradient)
    {
        if (_maxIndices == null)
        {
            throw new InvalidOperationException("MaxPoolBackward called before MaxPoolForward.");
        }

        var inputGradient = new Tensor<T>(InputShape);

        for (int b = 0; b < OutputShape[0]; b++)
        {
            for (int h = 0; h < OutputShape[1]; h++)
            {
                for (int w = 0; w < OutputShape[2]; w++)
                {
                    var maxIndex = _maxIndices[b, h, w];
                    var ph = NumOps.Divide(maxIndex, NumOps.FromDouble(PoolSize));
                    var pw = MathHelper.Modulo(maxIndex, NumOps.FromDouble(PoolSize));
                    var ih = NumOps.Add(NumOps.Multiply(NumOps.FromDouble(h), NumOps.FromDouble(Strides)), ph);
                    var iw = NumOps.Add(NumOps.Multiply(NumOps.FromDouble(w), NumOps.FromDouble(Strides)), pw);
            
                    var ihInt = (int)Convert.ToDouble(ih);
                    var iwInt = (int)Convert.ToDouble(iw);
            
                    if (ihInt < InputShape[1] && iwInt < InputShape[2])
                    {
                        inputGradient[b, ihInt, iwInt] = NumOps.Add(inputGradient[b, ihInt, iwInt], outputGradient[b, h, w]);
                    }
                }
            }
        }

        return inputGradient;
    }

    /// <summary>
    /// Implements the forward pass for average pooling.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>The output tensor after average pooling.</returns>
    /// <remarks>
    /// For Beginners: Average pooling takes the average value of each window of pixels.
    /// For example, if you have four pixels with values [2,5,1,3], average pooling would
    /// produce (2+5+1+3)/4 = 2.75.
    /// 
    /// This helps the network capture the general intensity or pattern in each region,
    /// rather than just the strongest feature.
    /// </remarks>
    private Tensor<T> AveragePoolForward(Tensor<T> input)
    {
        var output = new Tensor<T>(OutputShape);

        for (int b = 0; b < OutputShape[0]; b++)
        {
            for (int h = 0; h < OutputShape[1]; h++)
            {
                for (int w = 0; w < OutputShape[2]; w++)
                {
                    T sum = NumOps.Zero;
                    int count = 0;
                    for (int ph = 0; ph < PoolSize; ph++)
                    {
                        for (int pw = 0; pw < PoolSize; pw++)
                        {
                            int ih = h * Strides + ph;
                            int iw = w * Strides + pw;
                            if (ih < InputShape[1] && iw < InputShape[2])
                            {
                                sum = NumOps.Add(sum, input[b, ih, iw]);
                                count++;
                            }
                        }
                    }

                    output[b, h, w] = NumOps.Divide(sum, NumOps.FromDouble(count));
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Performs the backward pass for average pooling, propagating gradients back to the previous layer.
    /// </summary>
    /// <param name="input">The original input tensor from the forward pass.</param>
    /// <param name="outputGradient">The gradient from the next layer.</param>
    /// <returns>The gradient to pass to the previous layer.</returns>
    /// <remarks>
    /// For Beginners: In average pooling, each input value contributed equally to the output.
    /// When going backward, we distribute the gradient equally to all input positions that
    /// contributed to each output value.
    /// 
    /// Think of it like sharing credit equally among all participants in each pooling window.
    /// </remarks>
    private Tensor<T> AveragePoolBackward(Tensor<T> input, Tensor<T> outputGradient)
    {
        var inputGradient = new Tensor<T>(InputShape);

        for (int b = 0; b < OutputShape[0]; b++)
        {
            for (int h = 0; h < OutputShape[1]; h++)
            {
                for (int w = 0; w < OutputShape[2]; w++)
                {
                    T gradientValue = NumOps.Divide(outputGradient[b, h, w], NumOps.FromDouble(PoolSize * PoolSize));
                    for (int ph = 0; ph < PoolSize; ph++)
                    {
                        for (int pw = 0; pw < PoolSize; pw++)
                        {
                            int ih = h * Strides + ph;
                            int iw = w * Strides + pw;
                            if (ih < InputShape[1] && iw < InputShape[2])
                            {
                                inputGradient[b, ih, iw] = NumOps.Add(inputGradient[b, ih, iw], gradientValue);
                            }
                        }
                    }
                }
            }
        }

        return inputGradient;
    }

    /// <summary>
    /// Implements the forward pass for L2 norm pooling.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>The output tensor after L2 norm pooling.</returns>
    /// <remarks>
    /// For Beginners: L2 norm pooling (also called root-mean-square pooling) is like taking
    /// the "energy" of each window of pixels. It works by:
    /// 1. Squaring each value (multiplying it by itself)
    /// 2. Adding up all the squared values
    /// 3. Taking the square root of the sum
    /// 
    /// For example, if you have four pixels with values [2,5,1,3], L2 norm pooling would produce
    /// sqrt(2²+5²+1²+3²) = sqrt(4+25+1+9) = sqrt(39) ≈ 6.24
    /// 
    /// This type of pooling is useful when you want to capture the overall magnitude or energy
    /// in a region, giving more weight to larger values than average pooling would.
    /// </remarks>
    private Tensor<T> L2NormPoolForward(Tensor<T> input)
    {
        var output = new Tensor<T>(OutputShape);

        for (int b = 0; b < OutputShape[0]; b++)
        {
            for (int h = 0; h < OutputShape[1]; h++)
            {
                for (int w = 0; w < OutputShape[2]; w++)
                {
                    T sumSquares = NumOps.Zero;
                    for (int ph = 0; ph < PoolSize; ph++)
                    {
                        for (int pw = 0; pw < PoolSize; pw++)
                        {
                            int ih = h * Strides + ph;
                            int iw = w * Strides + pw;
                            if (ih < InputShape[1] && iw < InputShape[2])
                            {
                                T val = input[b, ih, iw];
                                sumSquares = NumOps.Add(sumSquares, NumOps.Multiply(val, val));
                            }
                        }
                    }

                    output[b, h, w] = NumOps.Sqrt(sumSquares);
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Performs the backward pass for L2 norm pooling, propagating gradients back to the previous layer.
    /// </summary>
    /// <param name="input">The original input tensor from the forward pass.</param>
    /// <param name="outputGradient">The gradient from the next layer.</param>
    /// <returns>The gradient to pass to the previous layer.</returns>
    /// <exception cref="InvalidOperationException">Thrown when called before Forward pass.</exception>
    /// <remarks>
    /// For Beginners: In L2 norm pooling, each input value contributed proportionally to its
    /// magnitude. When going backward, we distribute the gradient to each input position based
    /// on how much that position contributed to the L2 norm.
    /// 
    /// Larger input values receive more of the gradient, while smaller values receive less.
    /// This is different from max pooling (where only the maximum gets gradient) and average
    /// pooling (where all inputs get equal gradient).
    /// </remarks>
    private Tensor<T> L2NormPoolBackward(Tensor<T> input, Tensor<T> outputGradient)
    {
        if (_lastInput == null)
        {
            throw new InvalidOperationException("L2NormPoolBackward called before Forward pass. _lastInput is null.");
        }

        var inputGradient = new Tensor<T>(InputShape);

        for (int b = 0; b < OutputShape[0]; b++)
        {
            for (int h = 0; h < OutputShape[1]; h++)
            {
                for (int w = 0; w < OutputShape[2]; w++)
                {
                    T l2Norm = _lastInput[b, h, w];
                    for (int ph = 0; ph < PoolSize; ph++)
                    {
                        for (int pw = 0; pw < PoolSize; pw++)
                        {
                            int ih = h * Strides + ph;
                            int iw = w * Strides + pw;
                            if (ih < InputShape[1] && iw < InputShape[2])
                            {
                                T val = input[b, ih, iw];
                                T gradientValue = NumOps.Multiply(outputGradient[b, h, w], 
                                    NumOps.Divide(val, l2Norm));
                                inputGradient[b, ih, iw] = NumOps.Add(inputGradient[b, ih, iw], gradientValue);
                            }
                        }
                    }
                }
            }
        }

        return inputGradient;
    }

    /// <summary>
    /// Updates the layer's parameters during training.
    /// </summary>
    /// <param name="learningRate">The learning rate that controls how much to adjust parameters.</param>
    /// <remarks>
    /// For Beginners: This method is part of the neural network training process.
    /// Normally, layers with trainable parameters (like weights) would update them here.
    /// However, sampling/pooling layers don't have any trainable parameters, so this method
    /// doesn't need to do anything.
    /// 
    /// Think of it like some parts of a machine that need regular adjustments (trainable layers)
    /// while other parts (like this sampling layer) are fixed and don't need adjustments.
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // Sampling layers typically don't have trainable parameters
    }

    /// <summary>
    /// Saves the layer's configuration to a binary stream.
    /// </summary>
    /// <param name="writer">The binary writer to write the data to.</param>
    /// <remarks>
    /// For Beginners: This method saves the layer's settings (like pool size and stride)
    /// so that you can reload the exact same layer later. It's like saving your game
    /// progress so you can continue from where you left off.
    /// 
    /// The data is written in binary format, which is compact and efficient for storage.
    /// </remarks>
    public override void Serialize(BinaryWriter writer)
    {
        base.Serialize(writer);

        writer.Write(PoolSize);
        writer.Write(Strides);
        writer.Write((int)SamplingType);
    }

    /// <summary>
    /// Loads the layer's configuration from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader to read the data from.</param>
    /// <remarks>
    /// For Beginners: This method loads previously saved settings for the layer.
    /// It's the counterpart to Serialize - if Serialize is like saving your game,
    /// Deserialize is like loading that saved game.
    /// 
    /// After loading the settings, it calls SetStrategies() to make sure the layer
    /// is ready to use the right algorithms for the specified sampling type.
    /// </remarks>
    public override void Deserialize(BinaryReader reader)
    {
        base.Deserialize(reader);

        PoolSize = reader.ReadInt32();
        Strides = reader.ReadInt32();
        SamplingType = (SamplingType)reader.ReadInt32();

        SetStrategies();
    }

    /// <summary>
    /// Returns the activation functions used by this layer.
    /// </summary>
    /// <returns>An empty collection since sampling layers don't use activation functions.</returns>
    /// <remarks>
    /// For Beginners: Activation functions are mathematical operations that determine
    /// the output of a neural network node. They introduce non-linearity, which helps
    /// neural networks learn complex patterns.
    /// 
    /// However, sampling/pooling layers don't use activation functions - they simply
    /// perform operations like taking the maximum or average of values. That's why
    /// this method returns an empty collection.
    /// </remarks>
    public override IEnumerable<ActivationFunction> GetActivationTypes()
    {
        // Sampling layers don't have activation functions
        return [];
    }

    /// <summary>
    /// Returns all trainable parameters of this layer.
    /// </summary>
    /// <returns>An empty vector since sampling layers have no trainable parameters.</returns>
    /// <remarks>
    /// For Beginners: Trainable parameters are values that the neural network adjusts
    /// during learning to improve its performance. Common examples are weights and biases
    /// in dense or convolutional layers.
    /// 
    /// Sampling/pooling layers don't have any trainable parameters - they just apply a fixed
    /// operation (max, average, etc.) to reduce the size of the data. That's why this method
    /// returns an empty vector.
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Sampling layers have no trainable parameters, so return an empty vector
        return Vector<T>.Empty();
    }

    /// <summary>
    /// Resets the internal state of the sampling layer.
    /// </summary>
    /// <remarks>
    /// This method clears any cached data from previous forward passes, 
    /// essentially "wiping the slate clean" for the layer.
    /// 
    /// For Beginners: Neural network layers sometimes need to remember information
    /// from previous calculations (like what the input was or which values were
    /// the maximum). This method helps "forget" that information when we want to
    /// start fresh, such as when processing a new batch of data or when reusing
    /// the network for a different task.
    /// 
    /// Think of it like erasing your scratch work after solving one math problem
    /// so you can use the same paper for a new problem.
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward pass
        _lastInput = null;
        _maxIndices = null;
    }
}
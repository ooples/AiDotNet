using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.PhysicsInformed.NeuralOperators;

namespace AiDotNet.Helpers;

/// <summary>
/// Provides helper methods for creating various neural network layer configurations.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// This class contains factory methods that create pre-configured sets of neural network layers
/// for common architectures like standard feed-forward networks, CNNs, ResNets, and more.
/// </remarks>
public static class LayerHelper<T>
{
    /// <summary>
    /// Provides operations for the numeric type T.
    /// </summary>
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Creates a standard feed-forward neural network with configurable hidden layers.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="hiddenLayerCount">Number of hidden layers (default: 1).</param>
    /// <param name="hiddenLayerSize">Number of neurons in each hidden layer (default: 64).</param>
    /// <param name="outputSize">Number of output neurons (default: 1).</param>
    /// <returns>A collection of layers forming a feed-forward neural network.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A feed-forward neural network is the simplest type of neural network where
    /// information flows in one direction from input to output. Think of it as an assembly line
    /// where each layer processes the data and passes it to the next layer.
    /// </para>
    /// <para>
    /// This method creates:
    /// - An input layer that takes your data
    /// - One or more hidden layers that learn patterns in your data
    /// - An output layer that produces the final prediction
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultLayers(
        NeuralNetworkArchitecture<T> architecture,
        int hiddenLayerCount = 1,
        int hiddenLayerSize = 64,
        int outputSize = 1)
    {
        ValidateLayerParameters(hiddenLayerCount, hiddenLayerSize, outputSize);

        int inputSize = architecture.CalculatedInputSize;

        // Input layer
        yield return new DenseLayer<T>(inputSize, hiddenLayerSize, new ReLUActivation<T>() as IActivationFunction<T>);

        // Hidden layers
        for (int i = 0; i < hiddenLayerCount - 1; i++)
        {
            yield return new DenseLayer<T>(hiddenLayerSize, hiddenLayerSize, new ReLUActivation<T>() as IActivationFunction<T>);
        }

        // Output layer (assuming classification task with softmax)
        yield return new DenseLayer<T>(hiddenLayerSize, outputSize, new SoftmaxActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Creates a Convolutional Neural Network (CNN) with configurable layers.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="convLayerCount">Number of convolutional layers (default: 2).</param>
    /// <param name="filterCount">Number of filters in each convolutional layer (default: 32).</param>
    /// <param name="kernelSize">Size of the convolutional kernel (default: 3).</param>
    /// <param name="denseLayerCount">Number of dense layers after convolutional layers (default: 1).</param>
    /// <param name="denseLayerSize">Number of neurons in each dense layer (default: 64).</param>
    /// <param name="outputSize">Number of output neurons (default: 1).</param>
    /// <returns>A collection of layers forming a CNN.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Convolutional Neural Network (CNN) is specialized for processing grid-like data,
    /// such as images. Instead of connecting every input to every neuron (which would be inefficient for images),
    /// CNNs use filters that scan across the image to detect features like edges, textures, and shapes.
    /// </para>
    /// <para>
    /// Key components in this CNN:
    /// - Convolutional layers: Detect features in the input using filters
    /// - Pooling layers: Reduce the size of the data while keeping important information
    /// - Flatten layer: Converts the multi-dimensional data to a flat vector
    /// - Dense layers: Process the extracted features to make predictions
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultCNNLayers(
        NeuralNetworkArchitecture<T> architecture,
        int convLayerCount = 2,
        int filterCount = 32,
        int kernelSize = 3,
        int denseLayerCount = 1,
        int denseLayerSize = 64,
        int outputSize = 1)
    {
        ValidateLayerParameters(convLayerCount, filterCount, kernelSize);
        ValidateLayerParameters(denseLayerCount, denseLayerSize, outputSize);

        var inputShape = architecture.GetInputShape();

        // Convolutional layers
        for (int i = 0; i < convLayerCount; i++)
        {
            yield return new ConvolutionalLayer<T>(
                inputDepth: i == 0 ? inputShape[0] : filterCount,
                inputHeight: inputShape[1],
                inputWidth: inputShape[2],
                outputDepth: filterCount,
                kernelSize: kernelSize,
                stride: 1,
                padding: 1,
                activation: new ReLUActivation<T>()
            );
            yield return new MaxPoolingLayer<T>(
                inputShape: [filterCount, inputShape[1], inputShape[2]],
                poolSize: 2,
                strides: 2
            );

            // Update input shape for next layer
            inputShape = [filterCount, inputShape[1] / 2, inputShape[2] / 2];
        }

        // Flatten layer
        yield return new FlattenLayer<T>(inputShape: inputShape);

        // Calculate the output size of the convolutional layers
        int convOutputSize = filterCount * inputShape[1] * inputShape[2];

        // Dense layers
        for (int i = 0; i < denseLayerCount; i++)
        {
            yield return new DenseLayer<T>(
                inputSize: i == 0 ? convOutputSize : denseLayerSize,
                outputSize: denseLayerSize,
                activationFunction: new ReLUActivation<T>()
            );
        }

        // Output layer
        yield return new DenseLayer<T>(denseLayerSize, outputSize, new SoftmaxActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Creates default layers for an occupancy detection neural network with temporal data.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration that defines input and output shapes.</param>
    /// <param name="historyWindowSize">The number of time steps to consider in the temporal data (how many past observations to include).</param>
    /// <returns>A collection of layers forming a temporal occupancy detection network.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method builds a neural network specifically designed to detect occupancy 
    /// (whether a space is occupied by people) using data that changes over time. It uses special layer types 
    /// like LSTM (Long Short-Term Memory) that can "remember" patterns in sequential data, and attention 
    /// mechanisms that help the network focus on the most important time steps in the data sequence.
    /// </para>
    /// <para>
    /// Temporal data refers to data collected over time, where the sequence and patterns across time 
    /// points are important for making predictions. For example, sensor readings collected every minute
    /// over several hours would be temporal data.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultOccupancyTemporalLayers(
        NeuralNetworkArchitecture<T> architecture,
        int historyWindowSize)
    {
        ValidateLayerParameters(1, 32, architecture.OutputSize);

        var inputShape = architecture.GetInputShape();
        int inputFeatures = inputShape[2];  // Assuming shape is [batch, time, features]

        // LSTM layers to process temporal data
        yield return new LSTMLayer<T>(
            inputSize: inputFeatures,
            hiddenSize: 64,
            inputShape: [historyWindowSize, inputFeatures],
            activation: new TanhActivation<T>() as IActivationFunction<T>,
            recurrentActivation: new SigmoidActivation<T>()
        );
        yield return new LSTMLayer<T>(
            inputSize: 64,
            hiddenSize: 32,
            inputShape: [historyWindowSize, 64],
            activation: new TanhActivation<T>() as IActivationFunction<T>,
            recurrentActivation: new SigmoidActivation<T>()
        );

        // Add a TimeDistributed layer to process each time step
        yield return new TimeDistributedLayer<T>(
            innerLayer: new DenseLayer<T>(32, 16, new ReLUActivation<T>() as IActivationFunction<T>),
            inputShape: [historyWindowSize, 32],
            activationFunction: null
        );

        // Add multi-head attention mechanism to focus on relevant time steps
        yield return new MultiHeadAttentionLayer<T>(
            sequenceLength: historyWindowSize,
            embeddingDimension: 16,
            headCount: 4,
            activationFunction: new ReLUActivation<T>()
        );

        // Flatten the output
        yield return new FlattenLayer<T>([historyWindowSize, 16]);

        // Flatten the output of LSTM layers
        yield return new FlattenLayer<T>([historyWindowSize, 32]);

        // Dense layers for further processing
        yield return new DenseLayer<T>(historyWindowSize * 32, 64, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new BatchNormalizationLayer<T>(64);
        yield return new DropoutLayer<T>(0.3f);

        yield return new DenseLayer<T>(64, 32, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new BatchNormalizationLayer<T>(32);
        yield return new DropoutLayer<T>(0.2f);

        // Output layer
        yield return new DenseLayer<T>(32, architecture.OutputSize, new SigmoidActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Creates default layers for a Deep Boltzmann Machine (DBM).
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration that defines input and output shapes.</param>
    /// <returns>A collection of layers forming a Deep Boltzmann Machine.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Deep Boltzmann Machine is a type of neural network that learns to recognize patterns 
    /// in data without supervision. It's made up of multiple layers of "hidden units" that learn to represent 
    /// features of the input data. DBMs are particularly good at learning complex patterns and can be used for 
    /// tasks like feature learning, dimensionality reduction, and generating new data similar to the training set.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultDeepBoltzmannMachineLayers(
        NeuralNetworkArchitecture<T> architecture)
    {
        ValidateLayerParameters(1, 32, architecture.OutputSize);

        var inputShape = architecture.GetInputShape();
        int inputSize = inputShape[0];

        // Define the sizes of each layer in the DBM
        int[] layerSizes = [inputSize, 500, 500, 2000, architecture.OutputSize];

        // Create layers
        for (int i = 0; i < layerSizes.Length - 1; i++)
        {
            yield return new RBMLayer<T>(
                visibleUnits: layerSizes[i],
                hiddenUnits: layerSizes[i + 1],
                new SigmoidActivation<T>() as IActivationFunction<T>
            );

            // Add a BatchNormalization layer after each RBM layer except the last one
            if (i < layerSizes.Length - 2)
            {
                yield return new BatchNormalizationLayer<T>(layerSizes[i + 1]);
            }
        }

        // Output layer
        yield return new DenseLayer<T>(layerSizes[layerSizes.Length - 2], layerSizes[layerSizes.Length - 1], new SigmoidActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Creates default layers for an occupancy detection neural network without temporal data.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration that defines input and output shapes.</param>
    /// <returns>A collection of layers forming a non-temporal occupancy detection network.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method builds a simpler neural network for detecting occupancy 
    /// (whether a space is occupied by people) using data from a single point in time, rather than 
    /// a sequence of time points. It uses standard Dense layers (also called fully connected layers) 
    /// to process the input features.
    /// </para>
    /// <para>
    /// Non-temporal data means the model makes predictions based only on current data points
    /// without considering how values have changed over time. For example, using the current 
    /// temperature, humidity, and CO2 levels to predict occupancy without looking at historical values.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultOccupancyLayers(
        NeuralNetworkArchitecture<T> architecture)
    {
        ValidateLayerParameters(1, 32, architecture.OutputSize);

        var inputShape = architecture.GetInputShape();
        int inputFeatures = inputShape[0];

        // Dense layers for processing input features
        yield return new DenseLayer<T>(inputFeatures, 64, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new BatchNormalizationLayer<T>(64);
        yield return new DropoutLayer<T>(0.3f);

        yield return new DenseLayer<T>(64, 32, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new BatchNormalizationLayer<T>(32);
        yield return new DropoutLayer<T>(0.2f);

        yield return new DenseLayer<T>(32, 16, new ReLUActivation<T>() as IActivationFunction<T>);

        // Output layer
        yield return new DenseLayer<T>(16, architecture.OutputSize, new SigmoidActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Validates the parameters used for creating neural network layers.
    /// </summary>
    /// <param name="layerCount">The number of layers in the network.</param>
    /// <param name="layerSize">The size (number of neurons) in each layer.</param>
    /// <param name="outputSize">The size of the output layer.</param>
    /// <exception cref="ArgumentException">Thrown when any parameter is less than 1.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This helper method makes sure that the neural network configuration
    /// makes sense before trying to build it. It checks that we have at least one layer,
    /// that each layer has at least one neuron, and that the output has at least one value.
    /// This validation prevents errors that might occur from invalid configurations.
    /// </para>
    /// </remarks>
    private static void ValidateLayerParameters(int layerCount, int layerSize, int outputSize)
    {
        if (layerCount < 1)
            throw new ArgumentException($"Layer count must be at least 1.", nameof(layerCount));
        if (layerSize < 1)
            throw new ArgumentException($"Layer size must be at least 1.", nameof(layerSize));
        if (outputSize < 1)
            throw new ArgumentException("Output size must be at least 1.", nameof(outputSize));
    }

    /// <summary>
    /// Creates a Residual Neural Network (ResNet) with configurable blocks.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="blockCount">Number of residual blocks (default: 3).</param>
    /// <param name="blockSize">Number of convolutional layers in each block (default: 2).</param>
    /// <returns>A collection of layers forming a ResNet.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Residual Network (ResNet) is designed to solve the "vanishing gradient problem" 
    /// that occurs when training very deep networks. It does this by adding "skip connections" that 
    /// allow information to bypass some layers.
    /// </para>
    /// <para>
    /// Think of it like this: In a traditional network, each layer must learn everything from scratch.
    /// In a ResNet, each layer only needs to learn the "difference" (or residual) between its input and 
    /// the desired output, which is often easier to learn.
    /// </para>
    /// <para>
    /// Key components:
    /// - Initial convolutional layer: Processes the raw input
    /// - Residual blocks: Groups of layers with skip connections
    /// - Global pooling: Reduces the spatial dimensions to a single value per feature map
    /// - Final dense layer: Makes the prediction based on the extracted features
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultResNetLayers(NeuralNetworkArchitecture<T> architecture, int blockCount = 3, int blockSize = 2)
    {
        ValidateLayerParameters(blockCount, blockSize, architecture.OutputSize);

        var inputShape = architecture.GetInputShape();

        // Initial convolutional layer
        yield return new ConvolutionalLayer<T>(
            inputDepth: inputShape[0],
            inputHeight: inputShape[1],
            inputWidth: inputShape[2],
            outputDepth: 64,
            kernelSize: 7,
            stride: 2,
            padding: 3,
            activation: new ReLUActivation<T>()
        );

        yield return new MaxPoolingLayer<T>(
            inputShape: [64, inputShape[1] / 2, inputShape[2] / 2],
            poolSize: 3,
            strides: 2
        );

        // Residual blocks
        int currentDepth = 64;
        int currentHeight = inputShape[1] / 4;
        int currentWidth = inputShape[2] / 4;

        for (int i = 0; i < blockCount; i++)
        {
            int outputDepth = currentDepth * 2;
            for (int j = 0; j < blockSize; j++)
            {
                foreach (var layer in CreateResidualBlock(currentDepth, outputDepth, currentHeight, currentWidth, j == 0))
                {
                    yield return layer;
                }
                currentDepth = outputDepth;
            }
            if (i < blockCount - 1)
            {
                yield return new MaxPoolingLayer<T>(
                    inputShape: [currentDepth, currentHeight, currentWidth],
                    poolSize: 2,
                    strides: 2
                );
                currentHeight /= 2;
                currentWidth /= 2;
            }
        }

        // Global average pooling
        yield return new GlobalPoolingLayer<T>(
            inputShape: [currentDepth, currentHeight, currentWidth],
            poolingType: PoolingType.Average,
            activationFunction: new IdentityActivation<T>()
        );

        // Final dense layer
        yield return new DenseLayer<T>(currentDepth, architecture.OutputSize, new SoftmaxActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Creates a residual block for ResNet-style architectures.
    /// </summary>
    /// <param name="inputDepth">The number of input channels.</param>
    /// <param name="outputDepth">The number of output channels.</param>
    /// <param name="height">The height of the input feature map.</param>
    /// <param name="width">The width of the input feature map.</param>
    /// <param name="isFirstInBlock">Whether this is the first residual block in a series.</param>
    /// <returns>A collection of layers that form a residual block.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A residual block is a special structure in neural networks that allows information to 
    /// "skip" over some layers. This helps solve the "vanishing gradient problem" in deep networks, making 
    /// them easier to train. Think of it like a highway bypass that lets some traffic go directly from 
    /// point A to point C without going through point B.
    /// </para>
    /// </remarks>
    private static IEnumerable<ILayer<T>> CreateResidualBlock(int inputDepth, int outputDepth, int height, int width, bool isFirstInBlock)
    {
        // Create the skip connection with the appropriate inner layer
        ILayer<T>? innerLayer = null;
        if (isFirstInBlock && inputDepth != outputDepth)
        {
            innerLayer = new ConvolutionalLayer<T>(
                inputDepth: inputDepth,
                outputDepth: outputDepth,
                kernelSize: 1,
                inputHeight: height,
                inputWidth: width,
                stride: 1,
                padding: 0,
                activation: new IdentityActivation<T>()
            );
        }

        yield return new ResidualLayer<T>(
             inputShape: [outputDepth, height, width],
             innerLayer: innerLayer,
             activation: new IdentityActivation<T>()
         );

        yield return new ConvolutionalLayer<T>(
            inputDepth: inputDepth,
            outputDepth: outputDepth,
            kernelSize: 3,
            inputHeight: height,
            inputWidth: width,
            stride: 1,
            padding: 1,
            activation: new ReLUActivation<T>()
        );

        yield return new ConvolutionalLayer<T>(
            inputDepth: outputDepth,
            outputDepth: outputDepth,
            kernelSize: 3,
            inputHeight: height,
            inputWidth: width,
            stride: 1,
            padding: 1,
            activation: new ReLUActivation<T>()
        );

        // Use IdentityActivation for the AddLayer
        yield return new AddLayer<T>([[outputDepth, height, width]], new IdentityActivation<T>() as IActivationFunction<T>);

        // Keep ReLU activation after addition
        yield return new ActivationLayer<T>([outputDepth, height, width], new ReLUActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Creates a default set of attention-based layers for transformer-style architectures.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <returns>A collection of layers forming an attention-based neural network.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Attention mechanisms allow neural networks to focus on specific parts of the input 
    /// that are most relevant for a given task. Similar to how humans pay attention to specific details 
    /// in a conversation, these layers help the network "pay attention" to important parts of the data.
    /// Transformers use this mechanism to process sequences (like text) very effectively.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultAttentionLayers(NeuralNetworkArchitecture<T> architecture)
    {
        var inputShape = architecture.GetInputShape();
        int embeddingSize = 128;
        int headCount = 8;
        int sequenceLength = inputShape[0];

        yield return new InputLayer<T>(inputShape[0]);

        yield return new EmbeddingLayer<T>(inputShape[0], embeddingSize);

        yield return new PositionalEncodingLayer<T>(sequenceLength, embeddingSize);

        // Multiple transformer blocks
        for (int i = 0; i < 3; i++)
        {
            yield return new MultiHeadAttentionLayer<T>(sequenceLength, embeddingSize, headCount, new GELUActivation<T>() as IActivationFunction<T>);

            yield return new LayerNormalizationLayer<T>(embeddingSize);

            yield return new DenseLayer<T>(embeddingSize, embeddingSize * 4, new ReLUActivation<T>() as IActivationFunction<T>);
            yield return new DenseLayer<T>(embeddingSize * 4, embeddingSize, new ReLUActivation<T>() as IActivationFunction<T>);

            yield return new LayerNormalizationLayer<T>(embeddingSize);
        }

        yield return new DenseLayer<T>(embeddingSize, architecture.OutputSize, new SoftmaxActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Creates a default autoencoder neural network architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <returns>A collection of layers forming an autoencoder neural network.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> An autoencoder is a type of neural network that learns to compress data into a 
    /// smaller representation and then reconstruct it back to the original form. Think of it like 
    /// learning to create a thumbnail of an image and then expanding it back to full size. The network 
    /// has two main parts: an encoder that compresses the data and a decoder that reconstructs it.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultAutoEncoderLayers(NeuralNetworkArchitecture<T> architecture)
    {
        var inputShape = architecture.GetInputShape();
        int inputSize = inputShape[0];
        int[] layerSizes = architecture.GetLayerSizes();

        if (layerSizes.Length < 3)
        {
            throw new InvalidOperationException("The autoencoder must have at least an input, encoded, and output layer.");
        }

        int middleIndex = layerSizes.Length / 2;

        // Encoder layers
        for (int i = 0; i < middleIndex; i++)
        {
            int outputSize = layerSizes[i + 1];
            yield return new DenseLayer<T>(inputSize, outputSize, new ReLUActivation<T>() as IActivationFunction<T>);

            if (i < middleIndex - 1)
            {
                yield return new ActivationLayer<T>([outputSize], new ReLUActivation<T>() as IActivationFunction<T>);
            }
            else
            {
                // Use linear activation for the encoded layer
                yield return new ActivationLayer<T>([outputSize], new IdentityActivation<T>() as IActivationFunction<T>);
            }

            inputSize = outputSize;
        }

        // Decoder layers
        for (int i = middleIndex; i < layerSizes.Length - 1; i++)
        {
            int outputSize = layerSizes[i + 1];
            yield return new DenseLayer<T>(inputSize, outputSize, new ReLUActivation<T>() as IActivationFunction<T>);

            if (i < layerSizes.Length - 2)
            {
                yield return new ActivationLayer<T>([outputSize], new ReLUActivation<T>() as IActivationFunction<T>);
            }
            else
            {
                // Use sigmoid activation for the output layer to constrain values between 0 and 1
                yield return new ActivationLayer<T>([outputSize], new SigmoidActivation<T>() as IActivationFunction<T>);
            }

            inputSize = outputSize;
        }
    }

    /// <summary>
    /// Creates a default capsule network architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <returns>A collection of layers forming a capsule network.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A capsule network is an advanced type of neural network that tries to better 
    /// understand spatial relationships in data. Unlike traditional networks that just detect features, 
    /// capsule networks also track the position, orientation, and size of features. Think of it like 
    /// the difference between recognizing a face by just its parts (eyes, nose, mouth) versus understanding 
    /// how those parts relate to each other in 3D space.
    /// </para>
    /// <para>
    /// The network consists of special "capsule" layers that group neurons together to represent entities 
    /// and their properties, allowing the network to better understand complex structures in data.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultCapsuleNetworkLayers(NeuralNetworkArchitecture<T> architecture)
    {
        if (architecture.CalculatedInputSize == 0)
        {
            throw new InvalidOperationException("The Capsule Network must have a valid input size.");
        }

        int inputDepth = architecture.InputDepth;
        int inputHeight = architecture.InputHeight;
        int inputWidth = architecture.InputWidth;

        // Add initial convolutional layer
        yield return new ConvolutionalLayer<T>(
            inputDepth: inputDepth,
            outputDepth: 256,
            kernelSize: 9,
            inputHeight: inputHeight,
            inputWidth: inputWidth,
            stride: 1,
            padding: 0,
            activation: new ReLUActivation<T>()
        );

        // Add PrimaryCapsules layer
        yield return new PrimaryCapsuleLayer<T>(
            inputChannels: 256,
            capsuleChannels: 32,
            capsuleDimension: 8,
            kernelSize: 9,
            stride: 2,
            scalarActivation: new SquashActivation<T>()
        );

        // Add DigitCapsules layer (final capsule layer)
        int numClasses = architecture.OutputSize;
        yield return new DigitCapsuleLayer<T>(
            inputCapsules: 32 * 6 * 6,
            inputCapsuleDimension: 8,
            numClasses: numClasses,
            outputCapsuleDimension: 16,
            routingIterations: 3
        );

        // Add Reconstruction layer (optional, for regularization)
        yield return new ReconstructionLayer<T>(
            inputDimension: numClasses * 16,  // numClasses * capsuleDimension
            hidden1Dimension: 512,
            hidden2Dimension: 1024,
            outputDimension: inputHeight * inputWidth * inputDepth,
            hiddenActivation: new ReLUActivation<T>(),
            outputActivation: new SigmoidActivation<T>()
        );
    }

    /// <summary>
    /// Creates a default Deep Belief Network (DBN) with pre-configured layers.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <returns>A collection of layers forming a Deep Belief Network.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Deep Belief Network is a type of neural network that learns to recognize patterns 
    /// in data by building multiple layers that each specialize in finding specific features. It works by 
    /// training each layer one at a time (called "pre-training"), which helps the network learn more 
    /// effectively, especially when you don't have a lot of labeled training data.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultDeepBeliefNetworkLayers(NeuralNetworkArchitecture<T> architecture)
    {
        // Default layer sizes for DBN (can be adjusted as needed)
        int[] layerSizes = [architecture.GetInputShape()[0], 500, 500, 2000, architecture.OutputSize];

        IActivationFunction<T> sigmoidActivation = new SigmoidActivation<T>();
        IActivationFunction<T> softmaxActivation = new SoftmaxActivation<T>();

        // Initialize layers
        for (int i = 0; i < layerSizes.Length - 1; i++)
        {
            int visibleUnits = layerSizes[i];
            int hiddenUnits = layerSizes[i + 1];

            // Create and add RBM layer
            yield return new RBMLayer<T>(
                visibleUnits: visibleUnits,
                hiddenUnits: hiddenUnits,
                scalarActivation: sigmoidActivation
            );

            // Add activation layer for each RBM
            yield return new ActivationLayer<T>([hiddenUnits], sigmoidActivation);
        }

        // Add the final output layer
        int outputSize = layerSizes[layerSizes.Length - 1];
        yield return new DenseLayer<T>(outputSize, outputSize, softmaxActivation);
        yield return new ActivationLayer<T>([outputSize], softmaxActivation);
    }

    /// <summary>
    /// Creates a default Deep Q-Network (DQN) with pre-configured layers for reinforcement learning.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <returns>A collection of layers forming a Deep Q-Network.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Deep Q-Network is a type of neural network used in reinforcement learning, 
    /// which is how computers learn to make decisions by trying different actions and receiving rewards. 
    /// Think of it like teaching a dog new tricks with treats. The network learns which actions 
    /// (like moving left or right in a game) will lead to the highest rewards over time.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultDeepQNetworkLayers(NeuralNetworkArchitecture<T> architecture)
    {
        int inputSize = architecture.CalculatedInputSize;
        int actionSpace = architecture.OutputSize;
        int hiddenLayerCount = 2; // Default to 2 hidden layers
        int defaultHiddenSize = 64; // Default size for hidden layers

        // Input layer to first hidden layer
        yield return new DenseLayer<T>(inputSize, defaultHiddenSize, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new ActivationLayer<T>([defaultHiddenSize], new ReLUActivation<T>() as IActivationFunction<T>);

        // Hidden layers
        for (int i = 1; i < hiddenLayerCount; i++)
        {
            yield return new DenseLayer<T>(defaultHiddenSize, defaultHiddenSize, new ReLUActivation<T>() as IActivationFunction<T>);
            yield return new ActivationLayer<T>([defaultHiddenSize], new ReLUActivation<T>() as IActivationFunction<T>);
        }

        // Output layer (Q-values for each action)
        yield return new DenseLayer<T>(defaultHiddenSize, actionSpace, new IdentityActivation<T>() as IActivationFunction<T>);
        // No activation for the output layer as Q-values can be any real number
    }

    /// <summary>
    /// Creates a default Differentiable Neural Computer (DNC) with pre-configured layers.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="controllerSize">The size of the controller network.</param>
    /// <param name="memoryWordSize">The size of each memory word.</param>
    /// <param name="readHeads">The number of read heads.</param>
    /// <param name="interfaceSize">The size of the interface between controller and memory.</param>
    /// <returns>A collection of layers forming a Differentiable Neural Computer.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Differentiable Neural Computer (DNC) is like a neural network with a built-in 
    /// memory system. Traditional neural networks process information and then forget it, but a DNC 
    /// can store information in its "memory" and retrieve it later when needed. This makes DNCs good 
    /// at tasks that require remembering information over time, like answering questions about a story 
    /// or navigating through complex environments.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultDNCLayers(NeuralNetworkArchitecture<T> architecture, int controllerSize, int memoryWordSize, int readHeads, int interfaceSize)
    {
        var inputShape = architecture.GetInputShape();

        if (inputShape == null || inputShape.Length == 0)
        {
            throw new InvalidOperationException("Input shape must be specified for DNC.");
        }

        int inputSize = inputShape[0];
        int outputSize = architecture.OutputSize > 0
            ? architecture.OutputSize
            : throw new InvalidOperationException("Output size must be specified and greater than 0 for DNC.");

        // Controller (Feed-forward network)
        yield return new DenseLayer<T>(inputSize, controllerSize, new ReLUActivation<T>() as IActivationFunction<T>);

        // Memory interface
        yield return new DenseLayer<T>(controllerSize, interfaceSize, new IdentityActivation<T>() as IActivationFunction<T>);

        // Output layer
        yield return new DenseLayer<T>(controllerSize + readHeads * memoryWordSize, outputSize, new IdentityActivation<T>() as IActivationFunction<T>);
        yield return new ActivationLayer<T>([outputSize], new SoftmaxActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Creates a default Echo State Network (ESN) with pre-configured layers.
    /// </summary>
    /// <param name="inputSize">The size of the input layer.</param>
    /// <param name="outputSize">The size of the output layer.</param>
    /// <param name="reservoirSize">The size of the reservoir (hidden layer).</param>
    /// <param name="spectralRadius">Controls the stability of the reservoir dynamics (default: 0.9).</param>
    /// <param name="sparsity">The connection sparsity in the reservoir (default: 0.1).</param>
    /// <returns>A collection of layers forming an Echo State Network.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> An Echo State Network is a special type of recurrent neural network where most 
    /// of the connections between neurons are fixed (not trained). Only the connections from the hidden 
    /// layer to the output are trained. Think of it like having a pool of water (the reservoir) that 
    /// you disturb with input signals, and then you learn to read the ripple patterns to predict outputs. 
    /// This makes ESNs very fast to train compared to other recurrent networks.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultESNLayers(int inputSize, int outputSize, int reservoirSize, double spectralRadius = 0.9, double sparsity = 0.1)
    {
        // Input to Reservoir connections (fixed random weights)
        yield return new DenseLayer<T>(inputSize, reservoirSize, new IdentityActivation<T>() as IActivationFunction<T>);

        // Reservoir (recurrent connections, fixed random weights)
        yield return new ReservoirLayer<T>(reservoirSize, reservoirSize, spectralRadius: spectralRadius, connectionProbability: sparsity);

        // Reservoir activation
        yield return new ActivationLayer<T>([reservoirSize], new TanhActivation<T>() as IVectorActivationFunction<T>);

        // Output layer (Reservoir to output, trainable)
        yield return new DenseLayer<T>(reservoirSize, outputSize, new IdentityActivation<T>() as IActivationFunction<T>);

        // Output activation (optional, depends on the problem)
        yield return new ActivationLayer<T>([outputSize], new IdentityActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Creates a default Variational Autoencoder (VAE) with pre-configured layers.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="latentSize">The size of the latent space dimension.</param>
    /// <returns>A collection of layers forming a Variational Autoencoder.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Variational Autoencoder (VAE) is a type of neural network that learns to 
    /// compress data into a smaller representation (encoding) and then reconstruct it back (decoding). 
    /// What makes VAEs special is that they create a "fuzzy" compressed representation rather than 
    /// an exact one, which helps the network learn meaningful patterns in your data. This makes VAEs 
    /// excellent for generating new data similar to your training examples.
    /// </para>
    /// <para>
    /// The latent space is the compressed representation where your data exists in a simplified form.
    /// Think of it as a "creative space" where the network understands the essential features of your data.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultVAELayers(NeuralNetworkArchitecture<T> architecture, int latentSize)
    {
        var inputShape = architecture.GetInputShape();
        if (inputShape == null || inputShape.Length == 0)
        {
            throw new InvalidOperationException("Input shape must be specified for VAE.");
        }

        int inputDepth = inputShape[0];
        int inputHeight = inputShape.Length > 1 ? inputShape[1] : 1;
        int inputWidth = inputShape.Length > 2 ? inputShape[2] : 1;

        // Encoder layers
        yield return new DenseLayer<T>(inputDepth * inputHeight * inputWidth, (inputDepth * inputHeight * inputWidth) / 2, new LeakyReLUActivation<T>() as IActivationFunction<T>);

        // Pooling layer to reduce dimensions
        yield return new PoolingLayer<T>(
            inputDepth: inputDepth,
            inputHeight: inputHeight,
            inputWidth: inputWidth,
            poolSize: 2,
            stride: 2,
            type: PoolingType.Average
        );

        // Calculate new dimensions after pooling
        int pooledDepth = inputDepth;
        int pooledHeight = (inputHeight - 2) / 2 + 1;
        int pooledWidth = (inputWidth - 2) / 2 + 1;
        int pooledSize = pooledDepth * pooledHeight * pooledWidth;

        yield return new DenseLayer<T>(pooledSize, pooledSize / 2, new LeakyReLUActivation<T>() as IActivationFunction<T>);

        // Latent space layers
        int encoderOutputSize = latentSize * 2; // For mean and log variance
        yield return new DenseLayer<T>(pooledSize / 2, encoderOutputSize, new IdentityActivation<T>() as IActivationFunction<T>);

        // Mean and LogVariance layers
        yield return new MeanLayer<T>([encoderOutputSize], axis: 0);
        yield return new LogVarianceLayer<T>([encoderOutputSize], axis: 0);

        // Decoder layers
        yield return new DenseLayer<T>(latentSize, pooledSize / 2, new LeakyReLUActivation<T>() as IActivationFunction<T>);
        yield return new DenseLayer<T>(pooledSize / 2, pooledSize, new LeakyReLUActivation<T>() as IActivationFunction<T>);

        // Add an Upsampling layer to match the pooling in the encoder
        yield return new UpsamplingLayer<T>([pooledDepth, pooledHeight, pooledWidth], 2);

        yield return new DenseLayer<T>(inputDepth * inputHeight * inputWidth, (inputDepth * inputHeight * inputWidth) / 2, new LeakyReLUActivation<T>() as IActivationFunction<T>);

        // Output layer
        yield return new DenseLayer<T>((inputDepth * inputHeight * inputWidth) / 2, inputDepth * inputHeight * inputWidth, new SigmoidActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Creates a default Transformer neural network with pre-configured encoder and decoder layers.
    /// </summary>
    /// <param name="architecture">The transformer architecture configuration.</param>
    /// <returns>A collection of layers forming a Transformer neural network.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Transformer is a powerful type of neural network especially good at processing 
    /// sequences like text or time series data. Unlike older networks, Transformers can look at all parts 
    /// of the input at once (using "attention") rather than processing it step by step. This makes them 
    /// excellent for tasks like translation, text generation, and understanding language.
    /// </para>
    /// <para>
    /// Key concepts:
    /// - Attention: Allows the model to focus on relevant parts of the input regardless of position
    /// - Multi-head attention: Lets the model focus on different aspects of the input simultaneously
    /// - Encoder: Processes the input sequence
    /// - Decoder: Generates the output sequence
    /// - Positional encoding: Helps the model understand the order of elements in a sequence
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultTransformerLayers(
        TransformerArchitecture<T> architecture)
    {
        int modelDimension = architecture.ModelDimension;
        int feedForwardDimension = architecture.FeedForwardDimension;
        int numEncoderLayers = architecture.NumEncoderLayers;
        int numDecoderLayers = architecture.NumDecoderLayers;
        int numHeads = architecture.NumHeads;
        int maxSequenceLength = architecture.MaxSequenceLength;
        double dropoutRate = architecture.DropoutRate;
        int vocabularySize = architecture.VocabularySize;
        bool usePositionalEncoding = architecture.UsePositionalEncoding;
        int outputSize = architecture.OutputSize;
        NeuralNetworkTaskType taskType = architecture.TaskType;
        double temperature = architecture.Temperature;

        // Add embedding layer for text input
        if (vocabularySize > 0)
        {
            yield return new EmbeddingLayer<T>(vocabularySize, modelDimension);
        }

        // Add positional encoding if specified
        if (usePositionalEncoding)
        {
            yield return new PositionalEncodingLayer<T>(maxSequenceLength, modelDimension);
        }

        // Add dropout layer after embedding
        if (dropoutRate > 0)
        {
            yield return new DropoutLayer<T>(dropoutRate);
        }

        // Add encoder layers
        for (int i = 0; i < numEncoderLayers; i++)
        {
            // Self-attention block
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: maxSequenceLength,
                embeddingDimension: modelDimension,
                headCount: numHeads,
                activationFunction: new IdentityActivation<T>());

            // Add normalization
            yield return new LayerNormalizationLayer<T>(modelDimension);

            // Add dropout if specified
            if (dropoutRate > 0)
            {
                yield return new DropoutLayer<T>(dropoutRate);
            }

            // Feed-forward network
            yield return new DenseLayer<T>(modelDimension, feedForwardDimension, new ReLUActivation<T>() as IActivationFunction<T>);
            yield return new DenseLayer<T>(feedForwardDimension, modelDimension, new IdentityActivation<T>() as IActivationFunction<T>);

            // Add normalization
            yield return new LayerNormalizationLayer<T>(modelDimension);

            // Add dropout if specified
            if (dropoutRate > 0)
            {
                yield return new DropoutLayer<T>(dropoutRate);
            }
        }

        // Add decoder layers if needed
        if (numDecoderLayers > 0)
        {
            for (int i = 0; i < numDecoderLayers; i++)
            {
                // Self-attention block
                yield return new MultiHeadAttentionLayer<T>(
                    sequenceLength: maxSequenceLength,
                    embeddingDimension: modelDimension,
                    headCount: numHeads,
                    activationFunction: new IdentityActivation<T>());

                // Add normalization
                yield return new LayerNormalizationLayer<T>(modelDimension);

                // Add dropout if specified
                if (dropoutRate > 0)
                {
                    yield return new DropoutLayer<T>(dropoutRate);
                }

                // Cross-attention block
                yield return new MultiHeadAttentionLayer<T>(
                    sequenceLength: maxSequenceLength,
                    embeddingDimension: modelDimension,
                    headCount: numHeads,
                    activationFunction: new IdentityActivation<T>());

                // Add normalization
                yield return new LayerNormalizationLayer<T>(modelDimension);

                // Add dropout if specified
                if (dropoutRate > 0)
                {
                    yield return new DropoutLayer<T>(dropoutRate);
                }

                // Feed-forward network
                yield return new DenseLayer<T>(modelDimension, feedForwardDimension, new ReLUActivation<T>() as IActivationFunction<T>);
                yield return new DenseLayer<T>(feedForwardDimension, modelDimension, new IdentityActivation<T>() as IActivationFunction<T>);

                // Add normalization
                yield return new LayerNormalizationLayer<T>(modelDimension);

                // Add dropout if specified
                if (dropoutRate > 0)
                {
                    yield return new DropoutLayer<T>(dropoutRate);
                }
            }
        }

        // Add the final projection layer
        yield return new DenseLayer<T>(modelDimension, outputSize, new IdentityActivation<T>() as IActivationFunction<T>);

        // Add the final activation layer based on task type
        switch (taskType)
        {
            case NeuralNetworkTaskType.BinaryClassification:
            case NeuralNetworkTaskType.MultiClassClassification:
            case NeuralNetworkTaskType.MultiLabelClassification:
            case NeuralNetworkTaskType.SequenceClassification:
            case NeuralNetworkTaskType.ImageClassification:
                yield return new ActivationLayer<T>([outputSize], new SoftmaxActivation<T>() as IVectorActivationFunction<T>);
                break;

            case NeuralNetworkTaskType.Regression:
                yield return new ActivationLayer<T>([outputSize], new IdentityActivation<T>() as IActivationFunction<T>);
                break;

            case NeuralNetworkTaskType.TextGeneration:
                if (temperature != 1.0)
                {
                    yield return new LambdaLayer<T>(
                        [outputSize],
                        [outputSize],
                        input => input.Scale(NumOps.FromDouble(1.0 / temperature)),
                        (input, gradient) => gradient.Scale(NumOps.FromDouble(temperature)),
                        new IdentityActivation<T>() as IActivationFunction<T>);
                }

                yield return new ActivationLayer<T>([outputSize], new SoftmaxActivation<T>() as IVectorActivationFunction<T>);
                break;

            case NeuralNetworkTaskType.Translation:
                yield return new ActivationLayer<T>([outputSize], new SoftmaxActivation<T>() as IVectorActivationFunction<T>);
                break;

            default:
                yield return new ActivationLayer<T>([outputSize], new SoftmaxActivation<T>() as IVectorActivationFunction<T>);
                break;
        }
    }

    /// <summary>
    /// Creates default layers for a Spiking Neural Network (SNN).
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="neuronType">The type of spiking neuron to use.</param>
    /// <param name="tau">The membrane time constant that controls how quickly neurons respond to inputs.</param>
    /// <param name="refractoryPeriod">The period after firing during which a neuron cannot fire again.</param>
    /// <param name="useLayerNormalization">Whether to use layer normalization to stabilize training.</param>
    /// <param name="useOutputConversion">Whether to convert spike outputs to continuous values.</param>
    /// <returns>A collection of layers forming a Spiking Neural Network.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Spiking Neural Networks (SNNs) are a type of neural network that more closely 
    /// mimics how real neurons in the brain work. Unlike traditional neural networks that use continuous 
    /// values, SNNs use "spikes" (binary on/off signals) to communicate between neurons. This makes them 
    /// more biologically realistic and potentially more energy-efficient for certain tasks.
    /// </para>
    /// <para>
    /// The tau parameter controls how quickly a neuron "forgets" previous inputs - larger values make 
    /// the neuron remember inputs for longer. The refractory period is like a "rest time" after a neuron 
    /// fires, during which it cannot fire again, similar to how real neurons behave.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultSpikingLayers(
            NeuralNetworkArchitecture<T> architecture,
            SpikingNeuronType neuronType = SpikingNeuronType.LeakyIntegrateAndFire,
            double tau = 10.0,
            double refractoryPeriod = 2.0,
            bool useLayerNormalization = false,
            bool useOutputConversion = true)
    {
        // Get input and output dimensions
        var inputShape = architecture.GetInputShape();

        if (inputShape == null || inputShape.Length == 0)
        {
            throw new InvalidOperationException("Input shape must be specified for Spiking Neural Network.");
        }

        // Determine layer sizes based on architecture
        int inputSize = architecture.CalculatedInputSize;
        int outputSize = architecture.OutputSize > 0
            ? architecture.OutputSize
            : throw new InvalidOperationException("Output size must be specified and greater than 0 for Spiking Neural Network.");

        // Default layer configuration if no custom layers are provided
        List<int> layerSizes;
        if (architecture.Layers != null && architecture.Layers.Count > 0)
        {
            // If custom layers are provided, we'll use their input/output shapes
            layerSizes = new List<int> { inputSize };
            foreach (var layer in architecture.Layers)
            {
                layerSizes.Add(layer.GetOutputShape().Aggregate(1, (a, b) => a * b));
            }
        }
        else
        {
            // Default architecture with two hidden layers
            layerSizes = new List<int> { inputSize, 128, 64, outputSize };
        }

        // Create layers
        for (int i = 0; i < layerSizes.Count - 1; i++)
        {
            int currentSize = layerSizes[i];
            int nextSize = layerSizes[i + 1];

            // Add spiking layer
            yield return new SpikingLayer<T>(
                inputSize: currentSize,
                outputSize: nextSize,
                neuronType: neuronType,
                tau: tau,
                refractoryPeriod: refractoryPeriod
            );

            // Add normalization layer to stabilize spiking activity
            if (useLayerNormalization)
            {
                yield return new LayerNormalizationLayer<T>(nextSize);
            }
        }

        // Add output layer - typically a dense layer to convert spikes to continuous values
        if (useOutputConversion)
        {
            yield return new DenseLayer<T>(
                layerSizes[layerSizes.Count - 1],
                outputSize,
                new IdentityActivation<T>() as IActivationFunction<T>
            );

            // Add appropriate activation based on task type
            if (architecture.TaskType == NeuralNetworkTaskType.BinaryClassification)
            {
                yield return new ActivationLayer<T>([outputSize], new SigmoidActivation<T>() as IActivationFunction<T>);
            }
            else if (architecture.TaskType == NeuralNetworkTaskType.MultiClassClassification)
            {
                yield return new ActivationLayer<T>([outputSize], new SoftmaxActivation<T>() as IActivationFunction<T>);
            }
            else
            {
                // For regression or other tasks, use linear activation
                yield return new ActivationLayer<T>([outputSize], new IdentityActivation<T>() as IActivationFunction<T>);
            }
        }
    }

    /// <summary>
    /// Creates default layers for an Extreme Learning Machine (ELM) neural network.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="hiddenLayerSize">The size of the hidden layer.</param>
    /// <returns>A collection of layers forming an Extreme Learning Machine.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> An Extreme Learning Machine (ELM) is a simplified neural network where only the 
    /// output layer weights are trained. The hidden layer weights are randomly initialized and never updated. 
    /// This makes ELMs very fast to train compared to traditional neural networks, while still providing 
    /// good performance for many tasks. Think of it as a "shortcut" approach to neural network training.
    /// </para>
    /// <para>
    /// ELMs work by projecting the input data into a higher-dimensional space using random weights, 
    /// then finding the best output weights to solve the problem. They're particularly useful when you 
    /// need a quick solution and don't have time for extensive training.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultELMLayers(NeuralNetworkArchitecture<T> architecture, int hiddenLayerSize)
    {
        // Determine layer sizes based on architecture
        int inputSize = architecture.CalculatedInputSize;
        int outputSize = architecture.OutputSize > 0
            ? architecture.OutputSize
            : throw new InvalidOperationException("Output size must be specified and greater than 0 for Extreme Learning Machines.");

        // Random projection layer (input to hidden)
        yield return new DenseLayer<T>(inputSize, hiddenLayerSize, new IdentityActivation<T>() as IActivationFunction<T>);

        // Activation for hidden layer
        yield return new ActivationLayer<T>([hiddenLayerSize], new SigmoidActivation<T>() as IActivationFunction<T>);

        // Output layer (hidden to output)
        yield return new DenseLayer<T>(hiddenLayerSize, outputSize, new IdentityActivation<T>() as IActivationFunction<T>);

        // Output activation (optional, depends on the problem)
        yield return new ActivationLayer<T>([outputSize], new IdentityActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Creates default layers for a Graph Neural Network (GNN).
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <returns>A collection of layers forming a Graph Neural Network.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Graph Neural Networks (GNNs) are specialized neural networks designed to work with 
    /// graph-structured data, where information is represented as nodes (points) connected by edges (lines). 
    /// Examples include social networks, molecular structures, or road networks.
    /// </para>
    /// <para>
    /// Unlike standard neural networks that process individual data points independently, GNNs can 
    /// understand relationships between data points. They work by passing information between connected 
    /// nodes, allowing each node to "learn" from its neighbors. This makes GNNs powerful for tasks where 
    /// relationships between entities matter, such as recommending friends on social media, predicting 
    /// protein interactions, or analyzing traffic patterns.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultGNNLayers(NeuralNetworkArchitecture<T> architecture)
    {
        // Check if we have the minimum required network dimensions
        if (architecture.CalculatedInputSize <= 0 || architecture.OutputSize <= 0)
        {
            throw new InvalidOperationException("The network must have valid input and output dimensions.");
        }

        // Define network structure with sensible defaults
        int inputSize = architecture.CalculatedInputSize;
        int outputSize = architecture.OutputSize;

        // Define default GNN architecture - typically 2-3 graph convolutional layers
        // with decreasing sizes is a good starting point for many graph problems
        int firstHiddenSize = 64;
        int secondHiddenSize = 32;

        // Create the input layer - first graph convolution
        yield return new GraphConvolutionalLayer<T>(
            inputFeatures: inputSize,
            outputFeatures: firstHiddenSize,
            activationFunction: new ReLUActivation<T>()
        );

        // Add dropout for regularization (common in GNNs)
        yield return new DropoutLayer<T>(0.2);

        // Create second graph convolution layer
        yield return new GraphConvolutionalLayer<T>(
            inputFeatures: firstHiddenSize,
            outputFeatures: secondHiddenSize,
            activationFunction: new ReLUActivation<T>()
        );

        // Add dropout for regularization
        yield return new DropoutLayer<T>(0.2);

        // Create the output layer
        yield return new GraphConvolutionalLayer<T>(
            inputFeatures: secondHiddenSize,
            outputFeatures: outputSize,
            activationFunction: new IdentityActivation<T>()
        );

        // Add final activation based on task type
        if (architecture.TaskType == NeuralNetworkTaskType.BinaryClassification)
        {
            yield return new ActivationLayer<T>([outputSize], new SigmoidActivation<T>() as IVectorActivationFunction<T>);
        }
        else if (architecture.TaskType == NeuralNetworkTaskType.MultiClassClassification)
        {
            yield return new ActivationLayer<T>([outputSize], new SoftmaxActivation<T>() as IVectorActivationFunction<T>);
        }
        else
        {
            // For regression tasks
            yield return new ActivationLayer<T>([outputSize], new IdentityActivation<T>() as IVectorActivationFunction<T>);
        }
    }

    /// <summary>
    /// Creates default layers for a GraphSAGE (Graph Sample and Aggregate) Network.
    /// </summary>
    /// <param name="architecture">The neural network architecture specification.</param>
    /// <param name="aggregatorType">The type of aggregation function (default: Mean).</param>
    /// <param name="numLayers">Number of GraphSAGE layers (default: 2).</param>
    /// <param name="normalize">Whether to apply L2 normalization (default: true).</param>
    /// <returns>A collection of layers configured for GraphSAGE processing.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> GraphSAGE learns to aggregate neighbor information for inductive learning.
    /// It can generalize to new, unseen nodes by learning aggregation functions.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultGraphSAGELayers(
        NeuralNetworkArchitecture<T> architecture,
        SAGEAggregatorType aggregatorType = SAGEAggregatorType.Mean,
        int numLayers = 2,
        bool normalize = true)
    {
        if (architecture.CalculatedInputSize <= 0 || architecture.OutputSize <= 0)
        {
            throw new InvalidOperationException("The network must have valid input and output dimensions.");
        }

        int inputSize = architecture.CalculatedInputSize;
        int outputSize = architecture.OutputSize;
        int hiddenSize = 64;

        int currentInputDim = inputSize;

        // Add GraphSAGE layers
        for (int i = 0; i < numLayers; i++)
        {
            bool isLastLayer = (i == numLayers - 1);
            int outputDim = isLastLayer ? outputSize : hiddenSize;

            yield return new GraphSAGELayer<T>(
                inputFeatures: currentInputDim,
                outputFeatures: outputDim,
                aggregatorType: aggregatorType,
                normalize: normalize && !isLastLayer,
                activationFunction: isLastLayer ? null : new ReLUActivation<T>());

            currentInputDim = outputDim;
        }

        // Add final activation based on task type
        if (architecture.TaskType == NeuralNetworkTaskType.BinaryClassification)
        {
            yield return new ActivationLayer<T>([outputSize], new SigmoidActivation<T>() as IVectorActivationFunction<T>);
        }
        else if (architecture.TaskType == NeuralNetworkTaskType.MultiClassClassification)
        {
            yield return new ActivationLayer<T>([outputSize], new SoftmaxActivation<T>() as IVectorActivationFunction<T>);
        }
    }

    /// <summary>
    /// Creates default layers for a Graph Attention Network (GAT).
    /// </summary>
    /// <param name="architecture">The neural network architecture specification.</param>
    /// <param name="numHeads">Number of attention heads per layer (default: 8).</param>
    /// <param name="numLayers">Number of GAT layers (default: 2).</param>
    /// <param name="dropoutRate">Dropout rate for attention coefficients (default: 0.6).</param>
    /// <returns>A collection of layers configured for GAT processing.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> GAT uses attention mechanisms to learn which neighbors are most important
    /// for each node, allowing dynamic weighting of neighbor contributions.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultGraphAttentionLayers(
        NeuralNetworkArchitecture<T> architecture,
        int numHeads = 8,
        int numLayers = 2,
        double dropoutRate = 0.6)
    {
        if (architecture.CalculatedInputSize <= 0 || architecture.OutputSize <= 0)
        {
            throw new InvalidOperationException("The network must have valid input and output dimensions.");
        }

        int inputSize = architecture.CalculatedInputSize;
        int outputSize = architecture.OutputSize;
        int hiddenSize = 64;

        int currentInputDim = inputSize;

        // Add GAT layers
        for (int i = 0; i < numLayers; i++)
        {
            bool isLastLayer = (i == numLayers - 1);
            int outputDim = isLastLayer ? outputSize : hiddenSize;
            int heads = isLastLayer ? 1 : numHeads;

            yield return new GraphAttentionLayer<T>(
                inputFeatures: currentInputDim,
                outputFeatures: outputDim,
                numHeads: heads,
                dropoutRate: dropoutRate,
                activationFunction: isLastLayer ? null : new LeakyReLUActivation<T>(0.2));

            currentInputDim = outputDim;
        }

        // Add final activation based on task type
        if (architecture.TaskType == NeuralNetworkTaskType.BinaryClassification)
        {
            yield return new ActivationLayer<T>([outputSize], new SigmoidActivation<T>() as IVectorActivationFunction<T>);
        }
        else if (architecture.TaskType == NeuralNetworkTaskType.MultiClassClassification)
        {
            yield return new ActivationLayer<T>([outputSize], new SoftmaxActivation<T>() as IVectorActivationFunction<T>);
        }
    }

    /// <summary>
    /// Creates default layers for a Graph Isomorphism Network (GIN).
    /// </summary>
    /// <param name="architecture">The neural network architecture specification.</param>
    /// <param name="mlpHiddenDim">Hidden dimension for MLP within GIN layers (default: 64).</param>
    /// <param name="numLayers">Number of GIN layers (default: 5).</param>
    /// <param name="learnEpsilon">Whether to learn epsilon parameter (default: true).</param>
    /// <param name="initialEpsilon">Initial value for epsilon (default: 0.0).</param>
    /// <returns>A collection of layers configured for GIN processing.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> GIN is provably as powerful as the Weisfeiler-Lehman graph isomorphism test,
    /// making it optimal for distinguishing graph structures.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultGraphIsomorphismLayers(
        NeuralNetworkArchitecture<T> architecture,
        int mlpHiddenDim = 64,
        int numLayers = 5,
        bool learnEpsilon = true,
        double initialEpsilon = 0.0)
    {
        if (architecture.CalculatedInputSize <= 0 || architecture.OutputSize <= 0)
        {
            throw new InvalidOperationException("The network must have valid input and output dimensions.");
        }

        int inputSize = architecture.CalculatedInputSize;
        int outputSize = architecture.OutputSize;
        int hiddenSize = 64;

        int currentInputDim = inputSize;

        // Add GIN layers
        for (int i = 0; i < numLayers; i++)
        {
            bool isLastLayer = (i == numLayers - 1);
            int outputDim = isLastLayer ? outputSize : hiddenSize;

            yield return new GraphIsomorphismLayer<T>(
                inputFeatures: currentInputDim,
                outputFeatures: outputDim,
                mlpHiddenDim: mlpHiddenDim,
                learnEpsilon: learnEpsilon,
                epsilon: initialEpsilon,
                activationFunction: isLastLayer ? null : new ReLUActivation<T>());

            currentInputDim = outputDim;
        }

        // Add final activation based on task type
        if (architecture.TaskType == NeuralNetworkTaskType.BinaryClassification)
        {
            yield return new ActivationLayer<T>([outputSize], new SigmoidActivation<T>() as IVectorActivationFunction<T>);
        }
        else if (architecture.TaskType == NeuralNetworkTaskType.MultiClassClassification)
        {
            yield return new ActivationLayer<T>([outputSize], new SoftmaxActivation<T>() as IVectorActivationFunction<T>);
        }
    }

    /// <summary>
    /// Creates a default Gated Recurrent Unit (GRU) neural network layer configuration.
    /// </summary>
    /// <param name="architecture">The neural network architecture specification.</param>
    /// <returns>A collection of layers configured for GRU-based processing.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A GRU (Gated Recurrent Unit) is a type of recurrent neural network that's 
    /// especially good at learning patterns in sequences of data, like text or time series. 
    /// It's similar to LSTM but with a simpler structure, making it faster to train while 
    /// still capturing long-term dependencies in data.
    /// </para>
    /// <para>
    /// This method automatically configures appropriate GRU layers based on your task type,
    /// with sensible defaults for hidden layer sizes and activation functions.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown when the architecture has invalid input or output dimensions.</exception>
    public static IEnumerable<ILayer<T>> CreateDefaultGRULayers(NeuralNetworkArchitecture<T> architecture)
    {
        // Check if we have the minimum required network dimensions
        if (architecture.CalculatedInputSize <= 0 || architecture.OutputSize <= 0)
        {
            throw new InvalidOperationException("The network must have valid input and output dimensions.");
        }

        // Define network structure with sensible defaults
        int inputSize = architecture.CalculatedInputSize;
        int outputSize = architecture.OutputSize;

        // Define default GRU architecture
        // For sequence modeling, a common approach is to use 1-2 GRU layers followed by a dense output layer
        int hiddenSize = Math.Max(64, inputSize); // Reasonable hidden size for most sequence tasks

        // Determine if we need bidirectional GRU based on task type
        bool useBidirectional = architecture.TaskType == NeuralNetworkTaskType.SequenceClassification ||
                               architecture.TaskType == NeuralNetworkTaskType.SequenceToSequence;

        // Determine if we should return sequences based on task type
        bool returnSequences = architecture.TaskType == NeuralNetworkTaskType.SequenceToSequence ||
                              (architecture.Complexity == NetworkComplexity.Deep &&
                               architecture.TaskType == NeuralNetworkTaskType.SequenceClassification);

        // Create the GRU layer with recommended activations
        if (useBidirectional && architecture.Complexity != NetworkComplexity.Simple)
        {
            // Bidirectional GRU for better sequence understanding
            yield return new BidirectionalLayer<T>(
                new GRULayer<T>(
                    inputSize,
                    hiddenSize / 2, // Half size for each direction
                    returnSequences: returnSequences,
                    new TanhActivation<T>() as IActivationFunction<T>,  // Scalar activation for candidate hidden state
                    new SigmoidActivation<T>()  // Scalar activation for gates
                ), activationFunction: new IdentityActivation<T>()
            );
        }
        else
        {
            // Standard GRU
            yield return new GRULayer<T>(
                inputSize,
                hiddenSize,
                returnSequences: returnSequences,
                new TanhActivation<T>() as IActivationFunction<T>,  // Scalar activation for candidate hidden state
                new SigmoidActivation<T>()  // Scalar activation for gates
            );
        }

        // Add dropout for regularization (common in RNNs to prevent overfitting)
        yield return new DropoutLayer<T>(0.2);

        // For deeper networks, add another GRU layer if needed
        if (architecture.Complexity == NetworkComplexity.Deep)
        {
            int secondHiddenSize = hiddenSize / 2; // Typically decreasing size
            bool finalReturnSequences = architecture.TaskType == NeuralNetworkTaskType.SequenceToSequence;

            yield return new GRULayer<T>(
                hiddenSize,
                secondHiddenSize,
                returnSequences: finalReturnSequences,
                new TanhActivation<T>(),
                new SigmoidActivation<T>() as IActivationFunction<T>
            );

            yield return new DropoutLayer<T>(0.2);

            // Update hidden size for the output layer
            hiddenSize = secondHiddenSize;
        }

        // For sequence-to-sequence tasks, we might need a time-distributed dense layer
        if (architecture.TaskType == NeuralNetworkTaskType.SequenceToSequence)
        {
            // Choose appropriate activation based on task subtype
            IActivationFunction<T> timeDistributedActivation;

            // Determine the appropriate activation function based on the specific task
            if (architecture.TaskType == NeuralNetworkTaskType.BinaryClassification)
            {
                timeDistributedActivation = new SigmoidActivation<T>();
            }
            else if (architecture.TaskType == NeuralNetworkTaskType.MultiClassClassification)
            {
                timeDistributedActivation = new SoftmaxActivation<T>();
            }
            else
            {
                // For regression or other sequence tasks, use linear activation
                timeDistributedActivation = new IdentityActivation<T>();
            }

            yield return new TimeDistributedLayer<T>(
                new DenseLayer<T>(
                    hiddenSize,
                    outputSize,
                    new IdentityActivation<T>() as IActivationFunction<T>
                ), timeDistributedActivation
            );
        }
        else
        {
            // Standard dense output layer for other tasks
            yield return new DenseLayer<T>(
                hiddenSize,
                outputSize,
                new IdentityActivation<T>() as IActivationFunction<T>
            );
        }

        // Add final activation based on task type
        if (architecture.TaskType == NeuralNetworkTaskType.BinaryClassification)
        {
            yield return new ActivationLayer<T>([outputSize], new SigmoidActivation<T>() as IVectorActivationFunction<T>);
        }
        else if (architecture.TaskType == NeuralNetworkTaskType.MultiClassClassification)
        {
            yield return new ActivationLayer<T>([outputSize], new SoftmaxActivation<T>() as IVectorActivationFunction<T>);
        }
        else if (architecture.TaskType == NeuralNetworkTaskType.SequenceToSequence)
        {
            // For sequence-to-sequence, apply activation to each time step
            yield return new TimeDistributedLayer<T>(
                new ActivationLayer<T>([outputSize], new SoftmaxActivation<T>() as IVectorActivationFunction<T>), new ReLUActivation<T>() as IActivationFunction<T>
            );
        }
        else if (architecture.TaskType == NeuralNetworkTaskType.SequenceClassification)
        {
            yield return new ActivationLayer<T>([outputSize], new SoftmaxActivation<T>() as IVectorActivationFunction<T>);
        }
        else
        {
            // For regression tasks
            yield return new ActivationLayer<T>([outputSize], new IdentityActivation<T>() as IVectorActivationFunction<T>);
        }
    }

    /// <summary>
    /// Creates a default Hierarchical Temporal Memory (HTM) neural network layer configuration.
    /// </summary>
    /// <param name="architecture">The neural network architecture specification.</param>
    /// <param name="columnCount">The number of columns in the HTM network.</param>
    /// <param name="cellsPerColumn">The number of cells per column.</param>
    /// <param name="sparsityThreshold">The sparsity threshold for the spatial pooler.</param>
    /// <returns>A collection of layers configured for HTM processing.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Hierarchical Temporal Memory (HTM) is a machine learning technology that 
    /// mimics certain structural and algorithmic properties of the neocortex (the part of the brain 
    /// responsible for higher-order thinking). HTM is particularly good at learning patterns in 
    /// sequential data and making predictions.
    /// </para>
    /// <para>
    /// Key HTM concepts:
    /// - Columns: Vertical arrangements of cells that work together
    /// - Cells: The basic processing units (like neurons)
    /// - Sparsity: Only a small percentage of cells are active at any time, which helps with learning
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown when the architecture has invalid input or output dimensions.</exception>
    public static IEnumerable<ILayer<T>> CreateDefaultHTMLayers(NeuralNetworkArchitecture<T> architecture, int columnCount, int cellsPerColumn, double sparsityThreshold)
    {
        var inputShape = architecture.GetInputShape();

        if (inputShape == null || inputShape.Length == 0)
        {
            throw new InvalidOperationException("Input shape must be specified for HTM network.");
        }

        int inputSize = inputShape[0];
        int outputSize = architecture.OutputSize > 0
            ? architecture.OutputSize
            : throw new InvalidOperationException("Output size must be specified and greater than 0 for HTM network.");

        // Spatial Pooler Layer
        yield return new SpatialPoolerLayer<T>(inputSize, columnCount, sparsityThreshold);

        // Temporal Memory Layer
        yield return new TemporalMemoryLayer<T>(columnCount, cellsPerColumn);

        // Output Layer
        yield return new DenseLayer<T>(columnCount * cellsPerColumn, outputSize, new IdentityActivation<T>() as IActivationFunction<T>);
        yield return new ActivationLayer<T>([outputSize], new SoftmaxActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Creates a default Memory Network layer configuration.
    /// </summary>
    /// <param name="architecture">The neural network architecture specification.</param>
    /// <param name="memorySize">The size of the memory component (number of memory slots).</param>
    /// <param name="embeddingSize">The dimension of the embedding vectors.</param>
    /// <returns>A collection of layers configured for a Memory Network.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Memory Network is a type of neural network that has an explicit memory component.
    /// Think of it like a notebook that the network can write to and read from while processing information.
    /// This makes it particularly good at tasks that require remembering context from earlier in a sequence,
    /// such as answering questions about a story or maintaining a conversation.
    /// </para>
    /// <para>
    /// The memory size parameter controls how many "pages" are in the notebook, while the embedding size
    /// determines how detailed each "note" can be.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown when the architecture has invalid input or output dimensions.</exception>
    public static IEnumerable<ILayer<T>> CreateDefaultMemoryNetworkLayers(
            NeuralNetworkArchitecture<T> architecture,
            int memorySize,
            int embeddingSize)
    {
        var inputShape = architecture.GetInputShape();

        if (inputShape == null || inputShape.Length == 0)
        {
            throw new InvalidOperationException("Input shape must be specified for Memory Network.");
        }

        int inputSize = inputShape[0];
        int outputSize = architecture.OutputSize > 0
            ? architecture.OutputSize
            : throw new InvalidOperationException("Output size must be specified and greater than 0 for Memory Network.");

        // Calculate hidden layer size based on architecture complexity
        int hiddenSize;
        switch (architecture.Complexity)
        {
            case NetworkComplexity.Simple:
                hiddenSize = Math.Max(32, inputSize / 2);
                break;
            case NetworkComplexity.Medium:
                hiddenSize = Math.Max(64, inputSize);
                break;
            case NetworkComplexity.Deep:
                hiddenSize = Math.Max(128, inputSize * 2);
                break;
            default:
                hiddenSize = Math.Max(inputSize, outputSize);
                break;
        }

        // Input Embedding Layer
        yield return new EmbeddingLayer<T>(inputSize, embeddingSize);

        // Memory Read Layer
        yield return new MemoryReadLayer<T>(
            inputDimension: embeddingSize,
            memoryDimension: memorySize,
            outputDimension: embeddingSize,
            activationFunction: new ReLUActivation<T>() as IActivationFunction<T>
        );

        // Dense Layer for processing combined input and memory
        yield return new DenseLayer<T>(
            inputSize: embeddingSize * 2,
            outputSize: hiddenSize,
            activationFunction: new ReLUActivation<T>()
        );

        // Memory Write Layer
        yield return new MemoryWriteLayer<T>(
            inputDimension: hiddenSize,
            memoryDimension: memorySize,
            activationFunction: new TanhActivation<T>() as IActivationFunction<T>
        );

        // Add the final Dense Layer
        yield return new DenseLayer<T>(
            inputSize: hiddenSize,
            outputSize: outputSize,
            activationFunction: new IdentityActivation<T>()
        );

        // Add the final Activation Layer (typically Softmax for classification tasks)
        yield return new ActivationLayer<T>([outputSize], new SoftmaxActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Creates a default Recurrent Neural Network (RNN) layer configuration.
    /// </summary>
    /// <param name="architecture">The neural network architecture specification.</param>
    /// <returns>A collection of layers configured for RNN-based processing.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Recurrent Neural Network (RNN) is designed to work with sequential data
    /// by maintaining a form of "memory" of previous inputs. Unlike standard neural networks,
    /// RNNs can use their internal state to process sequences of inputs, making them ideal for
    /// tasks like text analysis, speech recognition, or time series prediction.
    /// </para>
    /// <para>
    /// This method automatically configures appropriate RNN layers with sensible defaults,
    /// including hidden layer sizes and activation functions.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultRNNLayers(NeuralNetworkArchitecture<T> architecture)
    {
        // Get input and output dimensions from the architecture
        var inputShape = architecture.GetInputShape();
        int inputSize = inputShape[0];
        int outputSize = architecture.OutputSize;

        // Default hidden layer size
        int hiddenSize = Math.Max(64, Math.Max(inputSize, outputSize));

        // Default number of recurrent layers
        int recurrentLayerCount = 2;

        // Input layer
        yield return new InputLayer<T>(inputSize);

        // First RNN Layer
        yield return new RecurrentLayer<T>(
            inputSize: inputSize,
            hiddenSize: hiddenSize,
            activationFunction: new TanhActivation<T>()
        );

        yield return new ActivationLayer<T>([hiddenSize], new TanhActivation<T>() as IActivationFunction<T>);

        // Additional RNN layers if needed
        for (int i = 1; i < recurrentLayerCount; i++)
        {
            yield return new RecurrentLayer<T>(
                inputSize: hiddenSize,
                hiddenSize: hiddenSize,
                activationFunction: new TanhActivation<T>()
            );

            yield return new ActivationLayer<T>([hiddenSize], new TanhActivation<T>() as IActivationFunction<T>);
        }

        // Add the final Dense Layer to map to output size
        yield return new DenseLayer<T>(
            inputSize: hiddenSize,
            outputSize: outputSize,
            activationFunction: null
        );

        // Add the final Activation Layer (typically Softmax for classification tasks)
        yield return new ActivationLayer<T>([outputSize], new SoftmaxActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Creates a default Radial Basis Function (RBF) neural network layer configuration.
    /// </summary>
    /// <param name="architecture">The neural network architecture specification.</param>
    /// <param name="hiddenSize">The size of the hidden layer. If set to 0 or negative, a default size will be calculated.</param>
    /// <param name="rbfFunction">The radial basis function to use. If null, a default Gaussian RBF will be used.</param>
    /// <returns>A collection of layers configured for RBF network processing.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Radial Basis Function (RBF) Network is a special type of neural network that uses
    /// "distance" to make predictions. Instead of gradually learning patterns through weights like standard
    /// neural networks, RBF networks measure how similar or different an input is from known examples.
    /// </para>
    /// <para>
    /// Think of it like this: if you want to identify a fruit, you might compare how similar it looks to
    /// fruits you already know. An RBF network works in a similar way - it has "reference points" and
    /// measures how close new data is to these points.
    /// </para>
    /// <para>
    /// RBF networks are particularly good at function approximation, pattern recognition, and time series prediction.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultRBFNetworkLayers(
        NeuralNetworkArchitecture<T> architecture,
        int hiddenSize = 0,
        IRadialBasisFunction<T>? rbfFunction = null)
    {
        var inputShape = architecture.GetInputShape();
        int inputSize = inputShape[0];
        int outputSize = architecture.OutputSize;

        // If hiddenSize is not specified, use a reasonable default
        if (hiddenSize <= 0)
        {
            hiddenSize = Math.Max(10, (inputSize + outputSize) / 2);
        }

        // Use default Gaussian RBF if not provided
        IRadialBasisFunction<T> rbf = rbfFunction ?? new GaussianRBF<T>();

        // Input layer (just a placeholder, doesn't do any computation)
        yield return new InputLayer<T>(inputSize);

        // RBF Layer
        yield return new RBFLayer<T>(inputSize, hiddenSize, rbf);

        // Output Layer (Dense)
        yield return new DenseLayer<T>(hiddenSize, outputSize, new IdentityActivation<T>() as IActivationFunction<T>);

        // Add the final Activation Layer based on task type
        if (architecture.TaskType == NeuralNetworkTaskType.BinaryClassification)
        {
            yield return new ActivationLayer<T>([outputSize], new SigmoidActivation<T>() as IVectorActivationFunction<T>);
        }
        else if (architecture.TaskType == NeuralNetworkTaskType.MultiClassClassification)
        {
            yield return new ActivationLayer<T>([outputSize], new SoftmaxActivation<T>() as IVectorActivationFunction<T>);
        }
        else
        {
            // For regression tasks
            yield return new ActivationLayer<T>([outputSize], new IdentityActivation<T>() as IVectorActivationFunction<T>);
        }
    }

    /// <summary>
    /// Creates a default configuration of layers for a Quantum Neural Network.
    /// </summary>
    /// <param name="architecture">The neural network architecture specification.</param>
    /// <param name="numQubits">The number of qubits to use in quantum layers (default: 4).</param>
    /// <returns>A collection of layers configured for a Quantum Neural Network.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Quantum Neural Network combines quantum computing concepts with neural networks.
    /// Think of qubits as special units that can exist in multiple states at once (unlike regular bits
    /// that are either 0 or 1). This gives quantum networks potential advantages for certain problems.
    /// The numQubits parameter controls how many of these special quantum units are used in each
    /// quantum layer.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when architecture is null.</exception>
    /// <exception cref="ArgumentException">Thrown when numQubits is not positive.</exception>
    public static IEnumerable<ILayer<T>> CreateDefaultQuantumNetworkLayers(
            NeuralNetworkArchitecture<T> architecture,
            int numQubits = 4)
    {
        if (architecture == null)
            throw new ArgumentNullException(nameof(architecture));

        if (numQubits <= 0)
            throw new ArgumentException("Number of qubits must be positive", nameof(numQubits));

        var inputShape = architecture.GetInputShape();
        int inputSize = inputShape[0];
        int outputSize = architecture.OutputSize;

        // Define recommended default sizes for quantum networks
        int hiddenSize = Math.Max(32, Math.Max(inputSize, outputSize));

        // Input layer
        yield return new InputLayer<T>(inputSize);

        // First quantum layer with measurement
        yield return new QuantumLayer<T>(inputSize, hiddenSize, numQubits);
        yield return new MeasurementLayer<T>(hiddenSize);

        // Add a dense layer after measurement
        yield return new DenseLayer<T>(
            inputSize: hiddenSize,
            outputSize: hiddenSize,
            activationFunction: new ReLUActivation<T>()
        );

        // Second quantum layer with measurement
        yield return new QuantumLayer<T>(hiddenSize, hiddenSize, numQubits);
        yield return new MeasurementLayer<T>(hiddenSize);

        // Final dense layer to map to output size
        yield return new DenseLayer<T>(
            inputSize: hiddenSize,
            outputSize: outputSize,
            activationFunction: null
        );

        // Final activation based on task type
        if (architecture.TaskType == NeuralNetworkTaskType.MultiClassClassification)
        {
            yield return new ActivationLayer<T>([outputSize], new SoftmaxActivation<T>() as IVectorActivationFunction<T>);
        }
        else if (architecture.TaskType == NeuralNetworkTaskType.BinaryClassification)
        {
            yield return new ActivationLayer<T>([outputSize], new SigmoidActivation<T>() as IVectorActivationFunction<T>);
        }
        else
        {
            yield return new ActivationLayer<T>([outputSize], new IdentityActivation<T>() as IVectorActivationFunction<T>);
        }
    }

    /// <summary>
    /// Creates a default configuration of layers for a Neural Turing Machine (NTM).
    /// </summary>
    /// <param name="architecture">The neural network architecture specification.</param>
    /// <param name="memorySize">The number of memory locations (default: 128).</param>
    /// <param name="memoryVectorSize">The size of each memory vector (default: 20).</param>
    /// <param name="controllerSize">The size of the controller network (default: 100).</param>
    /// <returns>A collection of layers configured for a Neural Turing Machine.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Neural Turing Machine (NTM) is a type of neural network that has an external 
    /// memory component, similar to how computers have RAM. The network learns to read from and write to 
    /// this memory, which helps it solve tasks that require remembering information over long periods.
    /// </para>
    /// <para>
    /// - memorySize: How many "slots" are in the memory (like pages in a notebook)
    /// - memoryVectorSize: How much information each memory slot can hold
    /// - controllerSize: How complex the "brain" of the network is that decides what to read/write
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when architecture is null.</exception>
    /// <exception cref="ArgumentException">Thrown when memory parameters are not positive.</exception>
    public static IEnumerable<ILayer<T>> CreateDefaultNTMLayers(
        NeuralNetworkArchitecture<T> architecture,
        int memorySize = 128,
        int memoryVectorSize = 20,
        int controllerSize = 100)
    {
        if (architecture == null)
            throw new ArgumentNullException(nameof(architecture));

        if (memorySize <= 0)
            throw new ArgumentException("Memory size must be positive", nameof(memorySize));

        if (memoryVectorSize <= 0)
            throw new ArgumentException("Memory vector size must be positive", nameof(memoryVectorSize));

        if (controllerSize <= 0)
            throw new ArgumentException("Controller size must be positive", nameof(controllerSize));

        var inputShape = architecture.GetInputShape();
        int inputSize = inputShape[0];
        int outputSize = architecture.OutputSize;

        // Input layer
        yield return new InputLayer<T>(inputSize);

        // Controller (Feed-forward network)
        yield return new DenseLayer<T>(inputSize, controllerSize, new TanhActivation<T>() as IActivationFunction<T>);

        // Read heads - typically use content-based addressing with cosine similarity
        yield return new MemoryReadLayer<T>(controllerSize, memoryVectorSize, memoryVectorSize,
            new SigmoidActivation<T>() as IActivationFunction<T>);

        // Write heads - typically use gated mechanism with sigmoid for gates
        yield return new MemoryWriteLayer<T>(
            controllerSize,
            memoryVectorSize,
            new TanhActivation<T>() as IActivationFunction<T>
        );

        // Output layer - linear projection before final task-specific activation
        yield return new DenseLayer<T>(controllerSize + memoryVectorSize, outputSize,
            new IdentityActivation<T>() as IActivationFunction<T>);

        // Final activation based on task type
        if (architecture.TaskType == NeuralNetworkTaskType.MultiClassClassification)
        {
            yield return new ActivationLayer<T>(new[] { outputSize }, new SoftmaxActivation<T>() as IVectorActivationFunction<T>);
        }
        else if (architecture.TaskType == NeuralNetworkTaskType.BinaryClassification)
        {
            yield return new ActivationLayer<T>(new[] { outputSize }, new SigmoidActivation<T>() as IVectorActivationFunction<T>);
        }
        else
        {
            yield return new ActivationLayer<T>(new[] { outputSize }, new IdentityActivation<T>() as IVectorActivationFunction<T>);
        }
    }

    /// <summary>
    /// Creates a default configuration of layers for a standard neural network.
    /// </summary>
    /// <param name="architecture">The neural network architecture specification.</param>
    /// <returns>A collection of layers configured for a standard neural network.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method creates the basic building blocks (layers) of a neural network.
    /// Think of layers as a series of connected processing units that transform your input data
    /// step by step until it produces the desired output. The complexity parameter in the architecture
    /// determines how many layers and neurons your network will have - Simple networks have fewer layers
    /// while Deep networks have more layers for handling more complex problems.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when architecture is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when input size or output size is not positive.</exception>
    public static IEnumerable<ILayer<T>> CreateDefaultNeuralNetworkLayers(NeuralNetworkArchitecture<T> architecture)
    {
        // Validate input
        if (architecture == null)
        {
            throw new ArgumentNullException(nameof(architecture));
        }

        // Get input shape and output size
        int inputSize = architecture.GetInputShape()[0];
        int outputSize = architecture.OutputSize;

        if (inputSize <= 0)
        {
            throw new InvalidOperationException("Input size must be greater than zero.");
        }

        if (outputSize <= 0)
        {
            throw new InvalidOperationException("Output size must be greater than zero.");
        }

        // Determine hidden layer sizes based on network complexity
        List<int> hiddenLayerSizes = new List<int>();

        switch (architecture.Complexity)
        {
            case NetworkComplexity.Simple:
                // One hidden layer with size between input and output
                hiddenLayerSizes.Add((inputSize + outputSize) / 2);
                break;

            case NetworkComplexity.Medium:
                // Two hidden layers
                hiddenLayerSizes.Add(inputSize * 2);
                hiddenLayerSizes.Add(inputSize);
                break;

            case NetworkComplexity.Deep:
                // Three hidden layers
                hiddenLayerSizes.Add(inputSize * 2);
                hiddenLayerSizes.Add(inputSize * 2);
                hiddenLayerSizes.Add(inputSize);
                break;

            default:
                // Default to one hidden layer
                hiddenLayerSizes.Add(inputSize);
                break;
        }

        IActivationFunction<T>? outputActivation = architecture.TaskType switch
        {
            NeuralNetworkTaskType.BinaryClassification => new SigmoidActivation<T>(),
            NeuralNetworkTaskType.MultiClassClassification => new SoftmaxActivation<T>(),
            NeuralNetworkTaskType.SequenceClassification => new SoftmaxActivation<T>(),
            NeuralNetworkTaskType.MultiLabelClassification => new SigmoidActivation<T>(),
            _ => null // Regression and other task types default to linear outputs
        };

        // Create input layer to first hidden layer
        int firstHiddenLayerSize = hiddenLayerSizes.Count > 0 ? hiddenLayerSizes[0] : outputSize;
        yield return new DenseLayer<T>(inputSize, firstHiddenLayerSize, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new ActivationLayer<T>([firstHiddenLayerSize], new ReLUActivation<T>() as IActivationFunction<T>);

        // Create hidden layers
        for (int i = 0; i < hiddenLayerSizes.Count - 1; i++)
        {
            int currentLayerSize = hiddenLayerSizes[i];
            int nextLayerSize = hiddenLayerSizes[i + 1];

            yield return new DenseLayer<T>(currentLayerSize, nextLayerSize, new ReLUActivation<T>() as IActivationFunction<T>);
            yield return new ActivationLayer<T>([nextLayerSize], new ReLUActivation<T>() as IActivationFunction<T>);
        }

        // Create last hidden layer to output layer
        if (hiddenLayerSizes.Count > 0)
        {
            int lastHiddenLayerSize = hiddenLayerSizes[hiddenLayerSizes.Count - 1];
            yield return new DenseLayer<T>(lastHiddenLayerSize, outputSize, (IActivationFunction<T>)new IdentityActivation<T>());
        }
        else
        {
            // If no hidden layers, connect input directly to output
            yield return new DenseLayer<T>(inputSize, outputSize, (IActivationFunction<T>)new IdentityActivation<T>());
        }

        if (outputActivation != null)
        {
            yield return new ActivationLayer<T>(new[] { outputSize }, outputActivation);
        }
    }

    /// <summary>
    /// Creates a default configuration of layers for a Bayesian neural network (Bayes-by-Backprop style).
    /// </summary>
    /// <remarks>
    /// This mirrors the library's default dense+activation patterns, but uses Bayesian dense layers so the network can
    /// express epistemic uncertainty through weight distributions.
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultBayesianNeuralNetworkLayers(NeuralNetworkArchitecture<T> architecture)
    {
        if (architecture == null)
        {
            throw new ArgumentNullException(nameof(architecture));
        }

        int inputSize = architecture.GetInputShape()[0];
        int outputSize = architecture.OutputSize;

        if (inputSize <= 0)
        {
            throw new InvalidOperationException("Input size must be greater than zero.");
        }

        if (outputSize <= 0)
        {
            throw new InvalidOperationException("Output size must be greater than zero.");
        }

        List<int> hiddenLayerSizes = new List<int>();
        switch (architecture.Complexity)
        {
            case NetworkComplexity.Simple:
                hiddenLayerSizes.Add((inputSize + outputSize) / 2);
                break;
            case NetworkComplexity.Medium:
                hiddenLayerSizes.Add(inputSize * 2);
                hiddenLayerSizes.Add(inputSize);
                break;
            case NetworkComplexity.Deep:
                hiddenLayerSizes.Add(inputSize * 2);
                hiddenLayerSizes.Add(inputSize * 2);
                hiddenLayerSizes.Add(inputSize);
                break;
            default:
                hiddenLayerSizes.Add(inputSize);
                break;
        }

        int firstHiddenLayerSize = hiddenLayerSizes.Count > 0 ? hiddenLayerSizes[0] : outputSize;
        yield return new AiDotNet.UncertaintyQuantification.Layers.BayesianDenseLayer<T>(inputSize, firstHiddenLayerSize, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new ActivationLayer<T>([firstHiddenLayerSize], new ReLUActivation<T>() as IActivationFunction<T>);

        for (int i = 0; i < hiddenLayerSizes.Count - 1; i++)
        {
            int currentLayerSize = hiddenLayerSizes[i];
            int nextLayerSize = hiddenLayerSizes[i + 1];

            yield return new AiDotNet.UncertaintyQuantification.Layers.BayesianDenseLayer<T>(currentLayerSize, nextLayerSize, new ReLUActivation<T>() as IActivationFunction<T>);
            yield return new ActivationLayer<T>([nextLayerSize], new ReLUActivation<T>() as IActivationFunction<T>);
        }

        if (hiddenLayerSizes.Count > 0)
        {
            int lastHiddenLayerSize = hiddenLayerSizes[hiddenLayerSizes.Count - 1];
            yield return new AiDotNet.UncertaintyQuantification.Layers.BayesianDenseLayer<T>(lastHiddenLayerSize, outputSize, new SoftmaxActivation<T>() as IActivationFunction<T>);
        }
        else
        {
            yield return new AiDotNet.UncertaintyQuantification.Layers.BayesianDenseLayer<T>(inputSize, outputSize, new SoftmaxActivation<T>() as IActivationFunction<T>);
        }

        yield return new ActivationLayer<T>(new[] { outputSize }, new SoftmaxActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Creates a default configuration of layers for a Liquid State Machine (LSM) neural network.
    /// </summary>
    /// <param name="architecture">The neural network architecture specification.</param>
    /// <param name="reservoirSize">The size of the reservoir (number of neurons in the reservoir layer). Default is 100.</param>
    /// <param name="connectionProbability">The probability of connection between neurons in the reservoir. Default is 0.1 (10%).</param>
    /// <param name="spectralRadius">Controls the stability of the reservoir dynamics. Default is 0.9.</param>
    /// <param name="inputScaling">Scaling factor for input connections. Default is 1.0.</param>
    /// <param name="leakingRate">Controls how quickly the reservoir responds to new inputs. Default is 1.0.</param>
    /// <returns>A collection of layers configured for a Liquid State Machine.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Liquid State Machine is a special type of neural network inspired by how 
    /// the brain processes information. The key component is the "reservoir" - imagine it as a pool 
    /// of randomly connected neurons that create complex patterns when input is fed into them.
    /// 
    /// - The reservoirSize is how many neurons are in this pool
    /// - The connectionProbability determines how densely connected these neurons are
    /// - The spectralRadius affects how stable the patterns in the reservoir are
    /// - The inputScaling controls how strongly the input affects the reservoir
    /// - The leakingRate determines how quickly the reservoir responds to new information
    /// 
    /// LSMs are particularly good at processing time-dependent data like speech or video.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when architecture is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when input shape is not specified or input/output size is not positive.</exception>
    public static IEnumerable<ILayer<T>> CreateDefaultLSMLayers(
        NeuralNetworkArchitecture<T> architecture,
        int reservoirSize = 100,
        double connectionProbability = 0.1,
        double spectralRadius = 0.9,
        double inputScaling = 1.0,
        double leakingRate = 1.0)
    {
        // Validate input
        if (architecture == null)
        {
            throw new ArgumentNullException(nameof(architecture));
        }

        // Get input shape and output size
        int[] inputShape = architecture.GetInputShape();
        if (inputShape == null || inputShape.Length == 0)
        {
            throw new InvalidOperationException("Input shape must be specified for Liquid State Machine.");
        }

        int inputSize = inputShape[0];
        int outputSize = architecture.OutputSize;

        if (inputSize <= 0)
        {
            throw new InvalidOperationException("Input size must be greater than zero.");
        }

        if (outputSize <= 0)
        {
            throw new InvalidOperationException("Output size must be greater than zero.");
        }

        // Adjust reservoir size based on complexity if not explicitly provided
        if (reservoirSize <= 0)
        {
            switch (architecture.Complexity)
            {
                case NetworkComplexity.Simple:
                    reservoirSize = Math.Max(50, inputSize * 2);
                    break;
                case NetworkComplexity.Medium:
                    reservoirSize = Math.Max(100, inputSize * 4);
                    break;
                case NetworkComplexity.Deep:
                    reservoirSize = Math.Max(200, inputSize * 8);
                    break;
                default:
                    reservoirSize = 100;
                    break;
            }
        }

        // Input layer
        yield return new DenseLayer<T>(inputSize, reservoirSize, new TanhActivation<T>() as IActivationFunction<T>);

        // Reservoir layer (liquid)
        yield return new ReservoirLayer<T>(
            inputSize,
            reservoirSize,
            connectionProbability,
            spectralRadius,
            inputScaling,
            leakingRate);

        // Output layer
        yield return new DenseLayer<T>(reservoirSize, outputSize, new IdentityActivation<T>() as IActivationFunction<T>);

        // Add the final Activation Layer based on task type
        if (architecture.TaskType == NeuralNetworkTaskType.MultiClassClassification)
        {
            yield return new ActivationLayer<T>([outputSize], new SoftmaxActivation<T>() as IVectorActivationFunction<T>);
        }
        else if (architecture.TaskType == NeuralNetworkTaskType.BinaryClassification)
        {
            yield return new ActivationLayer<T>([outputSize], new SigmoidActivation<T>() as IActivationFunction<T>);
        }
        else // Regression
        {
            yield return new ActivationLayer<T>([outputSize], new IdentityActivation<T>() as IActivationFunction<T>);
        }

        // Add the final Activation Layer (typically Softmax for classification tasks)
        yield return new ActivationLayer<T>([outputSize], new SoftmaxActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Creates a default configuration of layers for a Long Short-Term Memory (LSTM) neural network.
    /// </summary>
    /// <param name="architecture">The neural network architecture specification.</param>
    /// <returns>A collection of layers configured for an LSTM neural network.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> LSTM (Long Short-Term Memory) networks are a special kind of neural network
    /// designed to remember information for long periods of time. Think of them like a person with
    /// a good memory who can recall things from the past to make decisions in the present.
    /// </para>
    /// <para>
    /// LSTMs are particularly useful for:
    /// - Text prediction (like autocomplete on your phone)
    /// - Speech recognition
    /// - Time series forecasting (like stock prices or weather)
    /// - Any task where the order of data matters
    /// </para>
    /// <para>
    /// Key terms explained:
    /// - Hidden Size: How much information the network can remember at once (bigger = more memory)
    /// - Layers: How many processing steps the data goes through (more layers = more complex patterns)
    /// - Activation Function: How neurons decide whether to fire (like Tanh or Sigmoid)
    /// - Recurrent Activation: Special activation function used for the memory gates
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when architecture is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when input shape is not specified or input/output size is not positive.</exception>
    public static IEnumerable<ILayer<T>> CreateDefaultLSTMNetworkLayers(NeuralNetworkArchitecture<T> architecture)
    {
        // Validate input
        if (architecture == null)
        {
            throw new ArgumentNullException(nameof(architecture));
        }

        // Get input shape and output size
        var inputShape = architecture.GetInputShape();

        if (inputShape == null || inputShape.Length == 0)
        {
            throw new InvalidOperationException("Input shape must be specified for LSTM network.");
        }

        int inputSize = inputShape[0];
        int outputSize = architecture.OutputSize;

        if (inputSize <= 0)
        {
            throw new InvalidOperationException("Input size must be greater than zero.");
        }

        if (outputSize <= 0)
        {
            throw new InvalidOperationException("Output size must be greater than zero.");
        }

        // Calculate hidden layer sizes based on network complexity
        int _hiddenSize;  // Size of hidden state in LSTM cells
        int _numLayers;   // Number of stacked LSTM layers

        switch (architecture.Complexity)
        {
            case NetworkComplexity.Simple:
                _hiddenSize = Math.Max(32, inputSize);
                _numLayers = 1;
                break;
            case NetworkComplexity.Medium:
                _hiddenSize = Math.Max(64, inputSize * 2);
                _numLayers = 2;
                break;
            case NetworkComplexity.Deep:
                _hiddenSize = Math.Max(128, inputSize * 3);
                _numLayers = 3;
                break;
            default:
                _hiddenSize = Math.Max(64, inputSize);
                _numLayers = 2;
                break;
        }

        // Input layer - receives the raw input data
        yield return new InputLayer<T>(inputSize);

        // LSTM layers - process sequential information with memory capabilities
        int _currentInputSize = inputSize;

        for (int i = 0; i < _numLayers; i++)
        {
            // For deeper networks, gradually decrease the hidden size
            int _layerHiddenSize = i == _numLayers - 1 ?
                Math.Max(outputSize, _hiddenSize / 2) :
                _hiddenSize;

            // Add LSTM Layer
            yield return new LSTMLayer<T>(
                inputSize: _currentInputSize,
                hiddenSize: _layerHiddenSize,
                inputShape: [_currentInputSize],
                activation: new TanhActivation<T>(),
                recurrentActivation: new SigmoidActivation<T>() as IActivationFunction<T>
            );

            // Add Activation Layer after LSTM
            yield return new ActivationLayer<T>([_layerHiddenSize], new TanhActivation<T>() as IActivationFunction<T>);

            _currentInputSize = _layerHiddenSize;
        }

        // Add the final Dense Layer - transforms LSTM output to desired output size
        yield return new DenseLayer<T>(
            inputSize: _currentInputSize,
            outputSize: outputSize,
            activationFunction: new IdentityActivation<T>()
        );

        // Add the final Activation Layer based on task type
        if (architecture.TaskType == NeuralNetworkTaskType.MultiClassClassification)
        {
            // For multi-class classification (choosing one class from many)
            yield return new ActivationLayer<T>([outputSize], new SoftmaxActivation<T>() as IVectorActivationFunction<T>);
        }
        else if (architecture.TaskType == NeuralNetworkTaskType.BinaryClassification)
        {
            // For binary classification (yes/no decisions)
            yield return new ActivationLayer<T>([outputSize], new SigmoidActivation<T>() as IActivationFunction<T>);
        }
        else // Regression or default
        {
            // For regression (predicting continuous values)
            yield return new ActivationLayer<T>([outputSize], new IdentityActivation<T>() as IActivationFunction<T>);
        }
    }

    /// <summary>
    /// Creates default layers for a feed-forward neural network.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration that defines input and output shapes.</param>
    /// <param name="hiddenLayerCount">Number of hidden layers (default: 2).</param>
    /// <param name="hiddenLayerSize">Number of neurons in each hidden layer (default: 64).</param>
    /// <returns>A collection of layers forming a feed-forward neural network.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method builds a basic feed-forward neural network. Think of it as a series of 
    /// connected layers where information flows from the input, through "hidden" processing layers, to the output.
    /// </para>
    /// <para>
    /// Key components:
    /// - Input layer: Receives the raw data
    /// - Hidden layers: Process and transform the data, learning patterns
    /// - Output layer: Produces the final prediction or classification
    /// 
    /// The network automatically adjusts for different types of tasks (like classification or regression) 
    /// by choosing appropriate activation functions for the output layer.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultFeedForwardLayers(
        NeuralNetworkArchitecture<T> architecture,
        int hiddenLayerCount = 2,
        int hiddenLayerSize = 64)
    {
        ValidateLayerParameters(hiddenLayerCount, hiddenLayerSize, architecture.OutputSize);

        var inputShape = architecture.GetInputShape();
        int inputSize = inputShape.Aggregate((a, b) => a * b);  // Flatten multi-dimensional input

        // Input layer (flattening if necessary)
        if (inputShape.Length > 1)
        {
            yield return new FlattenLayer<T>(inputShape);
        }

        // First hidden layer
        yield return new DenseLayer<T>(inputSize, hiddenLayerSize, new ReLUActivation<T>() as IActivationFunction<T>);

        // Additional hidden layers
        for (int i = 1; i < hiddenLayerCount; i++)
        {
            yield return new DenseLayer<T>(hiddenLayerSize, hiddenLayerSize, new ReLUActivation<T>() as IActivationFunction<T>);
        }

        // Output layer
        var outputActivation = NeuralNetworkHelper<T>.GetDefaultActivationFunction(architecture.TaskType);

        yield return new DenseLayer<T>(hiddenLayerSize, architecture.OutputSize, outputActivation);
    }

    /// <summary>
    /// Creates default layers for a Node Classification model.
    /// </summary>
    /// <param name="architecture">The neural network architecture specification.</param>
    /// <param name="hiddenDim">Hidden dimension size (default: 64).</param>
    /// <param name="numLayers">Number of GNN layers (default: 2).</param>
    /// <param name="dropoutRate">Dropout rate for regularization (default: 0.5).</param>
    /// <returns>A collection of layers configured for node classification.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Node classification predicts labels for individual nodes in a graph.
    /// This architecture uses GCN layers with dropout for semi-supervised learning on graphs.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultNodeClassificationLayers(
        NeuralNetworkArchitecture<T> architecture,
        int hiddenDim = 64,
        int numLayers = 2,
        double dropoutRate = 0.5)
    {
        if (architecture.CalculatedInputSize <= 0 || architecture.OutputSize <= 0)
        {
            throw new InvalidOperationException("The network must have valid input and output dimensions.");
        }

        int inputFeatures = architecture.CalculatedInputSize;
        int numClasses = architecture.OutputSize;
        var reluActivation = new ReLUActivation<T>();

        // First GCN layer: input_features -> hidden_dim
        yield return new GraphConvolutionalLayer<T>(inputFeatures, hiddenDim, (IActivationFunction<T>?)null);
        yield return new ActivationLayer<T>([hiddenDim], (IActivationFunction<T>)reluActivation);
        if (dropoutRate > 0)
        {
            yield return new DropoutLayer<T>(dropoutRate);
        }

        // Additional intermediate layers
        for (int i = 1; i < numLayers - 1; i++)
        {
            yield return new GraphConvolutionalLayer<T>(hiddenDim, hiddenDim, (IActivationFunction<T>?)null);
            yield return new ActivationLayer<T>([hiddenDim], (IActivationFunction<T>)reluActivation);
            if (dropoutRate > 0)
            {
                yield return new DropoutLayer<T>(dropoutRate);
            }
        }

        // Final GCN layer: hidden_dim -> num_classes
        yield return new GraphConvolutionalLayer<T>(hiddenDim, numClasses, (IActivationFunction<T>?)null);
    }

    /// <summary>
    /// Creates default layers for a Link Prediction model encoder.
    /// </summary>
    /// <param name="architecture">The neural network architecture specification.</param>
    /// <param name="hiddenDim">Hidden dimension size (default: 64).</param>
    /// <param name="embeddingDim">Node embedding dimension (default: 32).</param>
    /// <param name="numLayers">Number of GNN layers (default: 2).</param>
    /// <param name="dropoutRate">Dropout rate for regularization (default: 0.5).</param>
    /// <returns>A collection of layers configured for link prediction.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Link prediction predicts whether edges should exist between nodes.
    /// This encoder learns node embeddings that can be combined to score potential edges.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultLinkPredictionLayers(
        NeuralNetworkArchitecture<T> architecture,
        int hiddenDim = 64,
        int embeddingDim = 32,
        int numLayers = 2,
        double dropoutRate = 0.5)
    {
        if (architecture.CalculatedInputSize <= 0)
        {
            throw new InvalidOperationException("The network must have valid input dimensions.");
        }

        int inputFeatures = architecture.CalculatedInputSize;
        var reluActivation = new ReLUActivation<T>();

        // First GCN layer: input_features -> hidden_dim
        yield return new GraphConvolutionalLayer<T>(inputFeatures, hiddenDim, (IActivationFunction<T>?)null);
        yield return new ActivationLayer<T>([hiddenDim], (IActivationFunction<T>)reluActivation);
        if (dropoutRate > 0)
        {
            yield return new DropoutLayer<T>(dropoutRate);
        }

        // Additional intermediate layers
        for (int i = 1; i < numLayers - 1; i++)
        {
            yield return new GraphConvolutionalLayer<T>(hiddenDim, hiddenDim, (IActivationFunction<T>?)null);
            yield return new ActivationLayer<T>([hiddenDim], (IActivationFunction<T>)reluActivation);
            if (dropoutRate > 0)
            {
                yield return new DropoutLayer<T>(dropoutRate);
            }
        }

        // Final layer: hidden_dim -> embedding_dim
        yield return new GraphConvolutionalLayer<T>(hiddenDim, embeddingDim, (IActivationFunction<T>?)null);
    }

    /// <summary>
    /// Creates default layers for a Graph Classification model.
    /// </summary>
    /// <param name="architecture">The neural network architecture specification.</param>
    /// <param name="hiddenDim">Hidden dimension size (default: 64).</param>
    /// <param name="embeddingDim">Graph embedding dimension (default: 128).</param>
    /// <param name="numGnnLayers">Number of GNN layers (default: 3).</param>
    /// <param name="dropoutRate">Dropout rate for regularization (default: 0.5).</param>
    /// <returns>A collection of layers configured for graph classification.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Graph classification predicts labels for entire graphs.
    /// This architecture uses multiple GCN layers followed by pooling and classification.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultGraphClassificationLayers(
        NeuralNetworkArchitecture<T> architecture,
        int hiddenDim = 64,
        int embeddingDim = 128,
        int numGnnLayers = 3,
        double dropoutRate = 0.5)
    {
        if (architecture.CalculatedInputSize <= 0)
        {
            throw new InvalidOperationException("The network must have valid input dimensions.");
        }

        int inputFeatures = architecture.CalculatedInputSize;
        var reluActivation = new ReLUActivation<T>();

        // First GCN layer: input_features -> hidden_dim
        yield return new GraphConvolutionalLayer<T>(inputFeatures, hiddenDim, (IActivationFunction<T>?)null);
        yield return new ActivationLayer<T>([hiddenDim], (IActivationFunction<T>)reluActivation);
        if (dropoutRate > 0)
        {
            yield return new DropoutLayer<T>(dropoutRate);
        }

        // Additional GNN layers: hidden_dim -> hidden_dim
        for (int i = 1; i < numGnnLayers - 1; i++)
        {
            yield return new GraphConvolutionalLayer<T>(hiddenDim, hiddenDim, (IActivationFunction<T>?)null);
            yield return new ActivationLayer<T>([hiddenDim], (IActivationFunction<T>)reluActivation);
            if (dropoutRate > 0)
            {
                yield return new DropoutLayer<T>(dropoutRate);
            }
        }

        // Final GNN layer: hidden_dim -> embedding_dim
        yield return new GraphConvolutionalLayer<T>(hiddenDim, embeddingDim, (IActivationFunction<T>?)null);
    }

    /// <summary>
    /// Creates default layers for a Graph Generation model (VGAE encoder).
    /// </summary>
    /// <param name="architecture">The neural network architecture specification.</param>
    /// <param name="hiddenDim">Hidden dimension size (default: 32).</param>
    /// <param name="numEncoderLayers">Number of encoder GNN layers (default: 2).</param>
    /// <returns>A collection of layers configured for graph generation encoder.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Graph generation models learn to create new graph structures.
    /// This encoder uses GCN layers to map node features to a latent space.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultGraphGenerationLayers(
        NeuralNetworkArchitecture<T> architecture,
        int hiddenDim = 32,
        int numEncoderLayers = 2)
    {
        if (architecture.CalculatedInputSize <= 0)
        {
            throw new InvalidOperationException("The network must have valid input dimensions.");
        }

        int inputFeatures = architecture.CalculatedInputSize;
        int currentInputDim = inputFeatures;

        // Add GCN encoder layers
        for (int i = 0; i < numEncoderLayers; i++)
        {
            yield return new GraphConvolutionalLayer<T>(
                inputFeatures: currentInputDim,
                outputFeatures: hiddenDim,
                activationFunction: new ReLUActivation<T>());
            currentInputDim = hiddenDim;
        }
    }

    /// <summary>
    /// Creates default layers for a Hamiltonian Neural Network.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration that defines input and output shapes.</param>
    /// <param name="hiddenLayerCount">Number of hidden layers (default: 3).</param>
    /// <param name="hiddenLayerSize">Number of neurons in each hidden layer (default: 64).</param>
    /// <returns>A collection of layers forming a Hamiltonian neural network.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Hamiltonian Neural Networks (HNNs) learn the energy function (Hamiltonian)
    /// of a physical system. The network takes a state vector [q, p] (positions and momenta) as input
    /// and outputs a scalar energy value.
    /// </para>
    /// <para>
    /// Key design choices:
    /// - Uses Tanh activation in hidden layers for smooth, bounded outputs that help with gradient computation
    /// - Output layer has linear activation since the Hamiltonian can be any real number
    /// - Architecture is designed for computing gradients (∂H/∂q, ∂H/∂p) to derive dynamics
    ///
    /// The network structure enables Hamilton's equations:
    /// - dq/dt = ∂H/∂p (velocity from momentum gradient)
    /// - dp/dt = -∂H/∂q (force from position gradient)
    ///
    /// This guarantees energy conservation by construction.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultHamiltonianLayers(
        NeuralNetworkArchitecture<T> architecture,
        int hiddenLayerCount = 3,
        int hiddenLayerSize = 64)
    {
        ValidateLayerParameters(hiddenLayerCount, hiddenLayerSize, architecture.OutputSize);

        if (architecture.OutputSize != 1)
        {
            throw new ArgumentException(
                "Hamiltonian networks require a scalar output (OutputSize = 1).",
                nameof(architecture));
        }

        var inputShape = architecture.GetInputShape();
        int inputSize = inputShape.Aggregate((a, b) => a * b);

        // Hamiltonian networks use Tanh for smooth gradients
        var hiddenActivation = new TanhActivation<T>() as IActivationFunction<T>;

        // First hidden layer
        yield return new DenseLayer<T>(inputSize, hiddenLayerSize, hiddenActivation);

        // Additional hidden layers
        for (int i = 1; i < hiddenLayerCount; i++)
        {
            yield return new DenseLayer<T>(hiddenLayerSize, hiddenLayerSize, hiddenActivation);
        }

        // Output layer - linear activation for unbounded energy output
        yield return new DenseLayer<T>(hiddenLayerSize, 1, new IdentityActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Creates default layers for a Lagrangian Neural Network.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration that defines input and output shapes.</param>
    /// <param name="hiddenLayerCount">Number of hidden layers (default: 3).</param>
    /// <param name="hiddenLayerSize">Number of neurons in each hidden layer (default: 64).</param>
    /// <returns>A collection of layers forming a Lagrangian neural network.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Lagrangian Neural Networks (LNNs) learn the Lagrangian function L(q, q̇)
    /// of a physical system. The Lagrangian is typically L = T - V (kinetic minus potential energy).
    /// </para>
    /// <para>
    /// Key design choices:
    /// - Uses Tanh activation in hidden layers for smooth derivatives needed in Euler-Lagrange equations
    /// - Output is scalar (the Lagrangian value)
    /// - Structure supports computing second derivatives for equations of motion
    ///
    /// The Euler-Lagrange equation: d/dt(∂L/∂q̇) = ∂L/∂q
    /// This gives the equations of motion while automatically respecting conservation laws.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultLagrangianLayers(
        NeuralNetworkArchitecture<T> architecture,
        int hiddenLayerCount = 3,
        int hiddenLayerSize = 64)
    {
        ValidateLayerParameters(hiddenLayerCount, hiddenLayerSize, architecture.OutputSize);

        if (architecture.OutputSize != 1)
        {
            throw new ArgumentException(
                "Lagrangian networks require a scalar output (OutputSize = 1).",
                nameof(architecture));
        }

        var inputShape = architecture.GetInputShape();
        int inputSize = inputShape.Aggregate((a, b) => a * b);

        // Lagrangian networks use Tanh for smooth second derivatives
        var hiddenActivation = new TanhActivation<T>() as IActivationFunction<T>;

        // First hidden layer
        yield return new DenseLayer<T>(inputSize, hiddenLayerSize, hiddenActivation);

        // Additional hidden layers
        for (int i = 1; i < hiddenLayerCount; i++)
        {
            yield return new DenseLayer<T>(hiddenLayerSize, hiddenLayerSize, hiddenActivation);
        }

        // Output layer - linear activation for unbounded Lagrangian output
        yield return new DenseLayer<T>(hiddenLayerSize, 1, new IdentityActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Creates default layers for a Universal Differential Equation (UDE) network.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="hiddenLayerCount">Number of hidden layers (default: 2).</param>
    /// <param name="hiddenLayerSize">Number of neurons in each hidden layer (default: 32).</param>
    /// <returns>A collection of layers forming a UDE neural network component.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Universal Differential Equations combine known physics with neural networks.
    /// The neural network learns the unknown parts of the dynamics while known physics equations
    /// are added explicitly. This is perfect for scientific applications where you know some
    /// of the physics but not all of it.
    /// </para>
    /// <para>
    /// The network takes [state, time] as input and outputs the learned correction to the dynamics.
    /// Uses Tanh activation for smooth derivatives needed in ODE integration.
    /// Output uses linear (identity) activation since corrections can be positive or negative.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultUniversalDELayers(
        NeuralNetworkArchitecture<T> architecture,
        int hiddenLayerCount = 2,
        int hiddenLayerSize = 32)
    {
        ValidateLayerParameters(hiddenLayerCount, hiddenLayerSize, architecture.OutputSize);

        var inputShape = architecture.GetInputShape();
        int inputSize = inputShape.Aggregate((a, b) => a * b);
        int outputSize = architecture.OutputSize;

        // UDE networks use Tanh for smooth derivatives in ODE integration
        var hiddenActivation = new TanhActivation<T>() as IActivationFunction<T>;

        // First hidden layer
        yield return new DenseLayer<T>(inputSize, hiddenLayerSize, hiddenActivation);

        // Additional hidden layers
        for (int i = 1; i < hiddenLayerCount; i++)
        {
            yield return new DenseLayer<T>(hiddenLayerSize, hiddenLayerSize, hiddenActivation);
        }

        // Output layer - linear activation for learned dynamics corrections
        yield return new DenseLayer<T>(hiddenLayerSize, outputSize, new IdentityActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Creates default layers for a Deep Operator Network (DeepONet).
    /// </summary>
    /// <param name="branchInputSize">Size of the branch network input (function samples).</param>
    /// <param name="trunkInputSize">Size of the trunk network input (query locations).</param>
    /// <param name="outputSize">
    /// Number of output components (default: 1 for scalar operators).
    /// For multi-output operators, each output component uses <paramref name="hiddenLayerSize"/> basis functions,
    /// so the final layer outputs <c>hiddenLayerSize * outputSize</c> values that are reshaped and summed.
    /// </param>
    /// <param name="hiddenLayerCount">Number of hidden layers in each sub-network (default: 3).</param>
    /// <param name="hiddenLayerSize">
    /// Number of neurons in each hidden layer, and the number of basis functions per output component (default: 64).
    /// </param>
    /// <returns>A tuple of (branchLayers, trunkLayers) for the DeepONet architecture.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> DeepONet learns operators - functions that take functions as input.
    /// For example, an operator might take a temperature distribution as input and output
    /// the resulting heat flow. The branch network encodes the input function, while the
    /// trunk network handles where you want to evaluate the output.
    /// </para>
    /// <para>
    /// <b>Architecture:</b> Branch encodes input function, Trunk encodes query location.
    /// Output = sum(Branch * Trunk) + bias, allowing learning of complex operators.
    /// </para>
    /// <para>
    /// <b>Multi-output handling:</b> For operators with multiple output components (e.g., velocity
    /// with x,y,z components), set <paramref name="outputSize"/> to the number of components.
    /// Each component gets its own set of basis functions. The branch and trunk networks
    /// output <c>hiddenLayerSize * outputSize</c> values, which are grouped as
    /// [component1_basis1..p, component2_basis1..p, ...] where p = <paramref name="hiddenLayerSize"/>.
    /// </para>
    /// </remarks>
    public static (IEnumerable<ILayer<T>> BranchLayers, IEnumerable<ILayer<T>> TrunkLayers) CreateDefaultDeepOperatorNetworkLayers(
        int branchInputSize,
        int trunkInputSize,
        int outputSize = 1,
        int hiddenLayerCount = 3,
        int hiddenLayerSize = 64)
    {
        if (hiddenLayerCount < 1)
            throw new ArgumentException("Must have at least 1 hidden layer.", nameof(hiddenLayerCount));
        if (hiddenLayerSize < 1)
            throw new ArgumentException("Hidden layer size must be positive.", nameof(hiddenLayerSize));
        if (outputSize < 1)
            throw new ArgumentException("Output size must be positive.", nameof(outputSize));

        return (
            CreateDeepONetBranchLayers(branchInputSize, hiddenLayerCount, hiddenLayerSize, outputSize),
            CreateDeepONetTrunkLayers(trunkInputSize, hiddenLayerCount, hiddenLayerSize, outputSize)
        );
    }

    /// <summary>
    /// Creates branch network layers for DeepONet using yield pattern.
    /// </summary>
    private static IEnumerable<ILayer<T>> CreateDeepONetBranchLayers(
        int branchInputSize,
        int hiddenLayerCount,
        int hiddenLayerSize,
        int outputSize)
    {
        var hiddenActivation = new TanhActivation<T>() as IActivationFunction<T>;
        int finalOutputDim = hiddenLayerSize * outputSize;

        // First hidden layer
        yield return new DenseLayer<T>(branchInputSize, hiddenLayerSize, hiddenActivation);

        // Additional hidden layers
        for (int i = 1; i < hiddenLayerCount; i++)
        {
            yield return new DenseLayer<T>(hiddenLayerSize, hiddenLayerSize, hiddenActivation);
        }

        // Branch output dimension: p * outputSize for multi-output support
        yield return new DenseLayer<T>(hiddenLayerSize, finalOutputDim, hiddenActivation);
    }

    /// <summary>
    /// Creates trunk network layers for DeepONet using yield pattern.
    /// </summary>
    private static IEnumerable<ILayer<T>> CreateDeepONetTrunkLayers(
        int trunkInputSize,
        int hiddenLayerCount,
        int hiddenLayerSize,
        int outputSize)
    {
        var hiddenActivation = new TanhActivation<T>() as IActivationFunction<T>;
        int finalOutputDim = hiddenLayerSize * outputSize;

        // First hidden layer
        yield return new DenseLayer<T>(trunkInputSize, hiddenLayerSize, hiddenActivation);

        // Additional hidden layers
        for (int i = 1; i < hiddenLayerCount; i++)
        {
            yield return new DenseLayer<T>(hiddenLayerSize, hiddenLayerSize, hiddenActivation);
        }

        // Trunk output dimension: p * outputSize to match branch for element-wise product
        yield return new DenseLayer<T>(hiddenLayerSize, finalOutputDim, hiddenActivation);
    }

    /// <summary>
    /// Creates default layers for a Fourier Neural Operator (FNO).
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="spatialDimensions">
    /// Dimensions of the spatial domain (e.g., [64, 64] for 2D grid, [32] for 1D).
    /// This determines the FFT size for spectral operations.
    /// </param>
    /// <param name="numFourierLayers">Number of Fourier layers (default: 4).</param>
    /// <param name="hiddenChannels">Number of hidden channels/width (default: 64).</param>
    /// <param name="numModes">
    /// Number of Fourier modes to retain (default: 12). Lower = smoother, higher = more detail.
    /// Should be less than or equal to smallest spatial dimension.
    /// </param>
    /// <returns>A collection of layers forming a Fourier Neural Operator.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Fourier Neural Operators learn mappings between function spaces
    /// by operating in frequency domain. They're especially powerful for PDEs because
    /// many physical phenomena have simple representations in frequency space.
    /// </para>
    /// <para>
    /// <b>Architecture:</b>
    /// <list type="number">
    /// <item><description>Lifting layer: Projects input to higher-dimensional channel space</description></item>
    /// <item><description>Fourier layers: Apply spectral convolution (FFT → learnable weights → IFFT) + local linear transform</description></item>
    /// <item><description>Projection layers: Map back to output dimension</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// <b>Key FNO Properties:</b>
    /// <list type="bullet">
    /// <item><description>Resolution-invariant: Train at one resolution, evaluate at another</description></item>
    /// <item><description>Global receptive field through spectral operations</description></item>
    /// <item><description>Efficient for smooth functions (low-frequency dominated)</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// <b>Note:</b> For full FNO functionality with training, use the <see cref="FourierNeuralOperator{T}"/>
    /// class directly, which provides a complete neural operator implementation.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when spatialDimensions is null.</exception>
    /// <exception cref="ArgumentException">Thrown when spatialDimensions is empty.</exception>
    public static IEnumerable<ILayer<T>> CreateDefaultFourierNeuralOperatorLayers(
        NeuralNetworkArchitecture<T> architecture,
        int[] spatialDimensions,
        int numFourierLayers = 4,
        int hiddenChannels = 64,
        int numModes = 12)
    {
        if (spatialDimensions is null)
        {
            throw new ArgumentNullException(nameof(spatialDimensions));
        }

        if (spatialDimensions.Length == 0)
        {
            throw new ArgumentException("Spatial dimensions cannot be empty.", nameof(spatialDimensions));
        }

        var inputShape = architecture.GetInputShape();
        int inputSize = inputShape.Aggregate((a, b) => a * b);
        int outputSize = architecture.OutputSize;

        var hiddenActivation = new GELUActivation<T>() as IActivationFunction<T>;

        // Lifting layer: project input to higher dimension
        yield return new DenseLayer<T>(inputSize, hiddenChannels, hiddenActivation);

        // Fourier layers with spectral convolution (FFT-based)
        for (int i = 0; i < numFourierLayers; i++)
        {
            yield return new FourierLayer<T>(hiddenChannels, numModes, spatialDimensions, hiddenActivation);
        }

        // Projection layers: project back to output dimension
        yield return new DenseLayer<T>(hiddenChannels, hiddenChannels, hiddenActivation);
        yield return new DenseLayer<T>(hiddenChannels, outputSize, new IdentityActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Creates default layers for a Variational Physics-Informed Neural Network (VPINN).
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="hiddenLayerCount">Number of hidden layers (default: 4).</param>
    /// <param name="hiddenLayerSize">Number of neurons in each hidden layer (default: 50).</param>
    /// <returns>A collection of layers forming a VPINN.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Variational PINNs solve PDEs using the weak (variational) form
    /// instead of the strong form. This is similar to Finite Element Methods but using
    /// neural networks. Often more stable for complex PDEs than standard PINNs.
    /// </para>
    /// <para>
    /// Uses Tanh activation throughout for smooth derivatives needed in variational formulation.
    /// Linear output layer since PDE solutions can take any real value.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultVariationalPINNLayers(
        NeuralNetworkArchitecture<T> architecture,
        int hiddenLayerCount = 4,
        int hiddenLayerSize = 50)
    {
        ValidateLayerParameters(hiddenLayerCount, hiddenLayerSize, architecture.OutputSize);

        var inputShape = architecture.GetInputShape();
        int inputSize = inputShape.Aggregate((a, b) => a * b);
        int outputSize = architecture.OutputSize;

        var hiddenActivation = new TanhActivation<T>() as IActivationFunction<T>;

        // First hidden layer
        yield return new DenseLayer<T>(inputSize, hiddenLayerSize, hiddenActivation);

        // Additional hidden layers
        for (int i = 1; i < hiddenLayerCount; i++)
        {
            yield return new DenseLayer<T>(hiddenLayerSize, hiddenLayerSize, hiddenActivation);
        }

        // Output layer - linear for PDE solution values
        yield return new DenseLayer<T>(hiddenLayerSize, outputSize, new IdentityActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Creates default layers for the Deep Ritz Method network.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="hiddenLayerCount">Number of hidden layers (default: 4).</param>
    /// <param name="hiddenLayerSize">Number of neurons in each hidden layer (default: 50).</param>
    /// <returns>A collection of layers forming a Deep Ritz network.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Deep Ritz Method solves PDEs by minimizing an energy functional
    /// instead of directly enforcing the PDE. This is based on the Ritz method from
    /// calculus of variations. The network learns the function that minimizes the energy.
    /// </para>
    /// <para>
    /// Similar architecture to VPINN but used with energy-based loss functions.
    /// Tanh activation provides smooth second derivatives needed for energy computations.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultDeepRitzLayers(
        NeuralNetworkArchitecture<T> architecture,
        int hiddenLayerCount = 4,
        int hiddenLayerSize = 50)
    {
        ValidateLayerParameters(hiddenLayerCount, hiddenLayerSize, architecture.OutputSize);

        var inputShape = architecture.GetInputShape();
        int inputSize = inputShape.Aggregate((a, b) => a * b);
        int outputSize = architecture.OutputSize;

        var hiddenActivation = new TanhActivation<T>() as IActivationFunction<T>;

        // First hidden layer
        yield return new DenseLayer<T>(inputSize, hiddenLayerSize, hiddenActivation);

        // Additional hidden layers
        for (int i = 1; i < hiddenLayerCount; i++)
        {
            yield return new DenseLayer<T>(hiddenLayerSize, hiddenLayerSize, hiddenActivation);
        }

        // Output layer - linear for energy/solution values
        yield return new DenseLayer<T>(hiddenLayerSize, outputSize, new IdentityActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Creates default layers for a Physics-Informed Neural Network (PINN).
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="hiddenLayerCount">Number of hidden layers (default: 4).</param>
    /// <param name="hiddenLayerSize">Number of neurons in each hidden layer (default: 32).</param>
    /// <returns>A collection of layers forming a PINN.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Physics-Informed Neural Networks (PINNs) solve PDEs by training
    /// a neural network to minimize the PDE residual at collocation points. The network
    /// learns the solution function u(x,t) while respecting the physics (PDE, boundary
    /// conditions, and initial conditions).
    /// </para>
    /// <para>
    /// Uses Tanh activation for smooth derivatives (important for computing PDE residuals).
    /// Multiple hidden layers capture complex solution behavior.
    /// Linear output layer since PDE solutions can take any real value.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultPINNLayers(
        NeuralNetworkArchitecture<T> architecture,
        int hiddenLayerCount = 4,
        int hiddenLayerSize = 32)
    {
        ValidateLayerParameters(hiddenLayerCount, hiddenLayerSize, architecture.OutputSize);

        var inputShape = architecture.GetInputShape();
        int inputSize = inputShape.Aggregate((a, b) => a * b);
        int outputSize = architecture.OutputSize;

        var hiddenActivation = new TanhActivation<T>() as IActivationFunction<T>;

        // First hidden layer
        yield return new DenseLayer<T>(inputSize, hiddenLayerSize, hiddenActivation);

        // Additional hidden layers - deeper networks for complex PDE solutions
        for (int i = 1; i < hiddenLayerCount; i++)
        {
            yield return new DenseLayer<T>(hiddenLayerSize, hiddenLayerSize, hiddenActivation);
        }

        // Output layer - linear for PDE solution values
        yield return new DenseLayer<T>(hiddenLayerSize, outputSize, new IdentityActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Creates default layers for a Voxel-based 3D Convolutional Neural Network.
    /// </summary>
    /// <param name="architecture">The neural network architecture specification.</param>
    /// <param name="voxelResolution">The resolution of the voxel grid (e.g., 32 for 32x32x32). Default is 32.</param>
    /// <param name="numConvBlocks">The number of convolutional blocks (each block has Conv3D + MaxPool3D). Default is 3.</param>
    /// <param name="baseFilters">The number of filters in the first convolutional layer. Doubles with each block. Default is 32.</param>
    /// <returns>A collection of layers configured for voxel-based 3D classification.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Voxel CNN is like a 3D version of a regular image classifier.
    /// Instead of looking at a 2D image, it examines a 3D grid of "blocks" (voxels) to understand
    /// 3D shapes. This is like how Minecraft represents the world - each block is either filled
    /// or empty, and the pattern of blocks creates recognizable objects.
    /// </para>
    /// <para>
    /// The architecture follows a standard pattern:
    /// - Multiple Conv3D + MaxPool3D blocks to extract hierarchical 3D features
    /// - Each block doubles the number of filters while halving the spatial resolution
    /// - Global average pooling to aggregate spatial information
    /// - Dense output layer for classification
    /// </para>
    /// <para>
    /// Applications include:
    /// - Recognizing 3D objects from voxelized point clouds (e.g., ModelNet40)
    /// - Medical image analysis (CT, MRI volumetric scans)
    /// - Spatial occupancy prediction from depth sensors
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown when the architecture has invalid input or output dimensions.</exception>
    public static IEnumerable<ILayer<T>> CreateDefaultVoxelCNNLayers(
        NeuralNetworkArchitecture<T> architecture,
        int voxelResolution = 32,
        int numConvBlocks = 3,
        int baseFilters = 32)
    {
        if (architecture.OutputSize <= 0)
        {
            throw new InvalidOperationException("Output size must be specified and greater than 0 for VoxelCNN.");
        }

        if (voxelResolution <= 0)
        {
            throw new ArgumentException("Voxel resolution must be positive.", nameof(voxelResolution));
        }

        int numClasses = architecture.OutputSize;
        int currentResolution = voxelResolution;
        int currentFilters = baseFilters;
        int inputChannels = 1; // Typically single-channel occupancy grid

        // Create Conv3D + MaxPool3D blocks
        for (int block = 0; block < numConvBlocks; block++)
        {
            int outputFilters = currentFilters * (1 << block); // Double filters each block
            int inChannels = (block == 0) ? inputChannels : (currentFilters * (1 << (block - 1)));

            // Conv3D layer with padding to maintain resolution before pooling
            yield return new Conv3DLayer<T>(
                inputChannels: inChannels,
                outputChannels: outputFilters,
                kernelSize: 3,
                inputDepth: currentResolution,
                inputHeight: currentResolution,
                inputWidth: currentResolution,
                stride: 1,
                padding: 1,
                activation: new ReLUActivation<T>());

            // MaxPool3D layer to downsample by factor of 2
            if (currentResolution >= 2)
            {
                yield return new MaxPool3DLayer<T>(
                    inputShape: [outputFilters, currentResolution, currentResolution, currentResolution],
                    poolSize: 2,
                    stride: 2);
                currentResolution /= 2;
            }
        }

        // Final number of filters after all blocks
        int finalFilters = currentFilters * (1 << (numConvBlocks - 1));

        // Global average pooling to aggregate spatial information
        yield return new GlobalPoolingLayer<T>(
            inputShape: [finalFilters, currentResolution, currentResolution, currentResolution],
            poolingType: PoolingType.Average,
            activationFunction: (IActivationFunction<T>?)null);

        // Dense output layer for classification
        IActivationFunction<T> outputActivation = architecture.TaskType == NeuralNetworkTaskType.MultiClassClassification
            ? new SoftmaxActivation<T>()
            : architecture.TaskType == NeuralNetworkTaskType.BinaryClassification
                ? new SigmoidActivation<T>()
                : new IdentityActivation<T>();

        yield return new DenseLayer<T>(
            inputSize: finalFilters,
            outputSize: numClasses,
            activationFunction: outputActivation);
    }

    /// <summary>
    /// Creates default layers for a 3D U-Net architecture for volumetric segmentation.
    /// </summary>
    /// <param name="architecture">The neural network architecture specification.</param>
    /// <param name="voxelResolution">The resolution of the voxel grid (e.g., 32 for 32x32x32). Default is 32.</param>
    /// <param name="numEncoderBlocks">The number of encoder blocks. Default is 4.</param>
    /// <param name="baseFilters">The number of filters in the first convolutional layer. Doubles with each block. Default is 32.</param>
    /// <returns>A collection of layers configured for 3D volumetric segmentation.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A 3D U-Net is like a specialized 3D image processor that can identify
    /// different parts of a 3D volume (like organs in a CT scan or objects in a point cloud).
    /// </para>
    /// <para>
    /// The U-shape architecture:
    /// - Encoder: Progressively downsamples to capture context (like zooming out)
    /// - Bottleneck: Smallest representation capturing global features
    /// - Decoder: Progressively upsamples to restore resolution (like zooming in)
    /// - Skip connections: Link encoder to decoder to preserve fine details
    /// </para>
    /// <para>
    /// Applications include:
    /// - 3D semantic segmentation of point clouds
    /// - Medical image segmentation (organs, tumors in CT/MRI)
    /// - Part segmentation of 3D shapes
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown when the architecture has invalid dimensions.</exception>
    public static IEnumerable<ILayer<T>> CreateDefaultUNet3DLayers(
        NeuralNetworkArchitecture<T> architecture,
        int voxelResolution = 32,
        int numEncoderBlocks = 4,
        int baseFilters = 32)
    {
        if (architecture.OutputSize <= 0)
        {
            throw new InvalidOperationException("Output size (number of segmentation classes) must be greater than 0.");
        }

        if (voxelResolution <= 0)
        {
            throw new ArgumentException("Voxel resolution must be positive.", nameof(voxelResolution));
        }

        // Verify resolution is sufficient for the number of encoder blocks
        int minResolution = 1 << numEncoderBlocks; // 2^numEncoderBlocks
        if (voxelResolution < minResolution)
        {
            throw new ArgumentOutOfRangeException(nameof(voxelResolution),
                $"VoxelResolution must be at least {minResolution} for {numEncoderBlocks} encoder blocks.");
        }

        int numClasses = architecture.OutputSize;
        int currentResolution = voxelResolution;
        int inputChannels = 1; // Typically single-channel occupancy grid

        // Track encoder output filter counts for skip connections
        var encoderFilters = new int[numEncoderBlocks];

        // ============== ENCODER PATH ==============
        // Each encoder block: Conv3D -> Conv3D -> MaxPool3D
        for (int block = 0; block < numEncoderBlocks; block++)
        {
            int outputFilters = baseFilters * (1 << block); // Double filters each block
            int inChannels = block == 0 ? inputChannels : encoderFilters[block - 1];
            encoderFilters[block] = outputFilters;

            // First Conv3D in block
            yield return new Conv3DLayer<T>(
                inputChannels: inChannels,
                outputChannels: outputFilters,
                kernelSize: 3,
                inputDepth: currentResolution,
                inputHeight: currentResolution,
                inputWidth: currentResolution,
                stride: 1,
                padding: 1,
                activation: new ReLUActivation<T>());

            // Second Conv3D in block
            yield return new Conv3DLayer<T>(
                inputChannels: outputFilters,
                outputChannels: outputFilters,
                kernelSize: 3,
                inputDepth: currentResolution,
                inputHeight: currentResolution,
                inputWidth: currentResolution,
                stride: 1,
                padding: 1,
                activation: new ReLUActivation<T>());

            // MaxPool3D to downsample (except last encoder block)
            if (block < numEncoderBlocks - 1)
            {
                yield return new MaxPool3DLayer<T>(
                    inputShape: [outputFilters, currentResolution, currentResolution, currentResolution],
                    poolSize: 2,
                    stride: 2);
                currentResolution /= 2;
            }
        }

        // ============== BOTTLENECK ==============
        // Additional convolutions at the bottleneck
        int bottleneckFilters = baseFilters * (1 << (numEncoderBlocks - 1)) * 2;
        yield return new Conv3DLayer<T>(
            inputChannels: encoderFilters[numEncoderBlocks - 1],
            outputChannels: bottleneckFilters,
            kernelSize: 3,
            inputDepth: currentResolution,
            inputHeight: currentResolution,
            inputWidth: currentResolution,
            stride: 1,
            padding: 1,
            activation: new ReLUActivation<T>());

        // ============== DECODER PATH ==============
        // Each decoder block: Upsample3D -> Conv3D -> Conv3D
        // Note: Skip connections need to be handled by the network model
        for (int block = numEncoderBlocks - 2; block >= 0; block--)
        {
            int outputFilters = encoderFilters[block];
            int inChannels = block == numEncoderBlocks - 2 ? bottleneckFilters : encoderFilters[block + 1] * 2;

            // Upsample3D to increase resolution
            yield return new Upsample3DLayer<T>(
                inputShape: [inChannels, currentResolution, currentResolution, currentResolution],
                scaleFactor: 2);
            currentResolution *= 2;

            // First Conv3D after upsample (would concatenate with skip in full U-Net)
            // For simplicity, we assume channels are doubled from skip connection
            yield return new Conv3DLayer<T>(
                inputChannels: inChannels, // In full U-Net: inChannels + encoderFilters[block] from skip
                outputChannels: outputFilters,
                kernelSize: 3,
                inputDepth: currentResolution,
                inputHeight: currentResolution,
                inputWidth: currentResolution,
                stride: 1,
                padding: 1,
                activation: new ReLUActivation<T>());

            // Second Conv3D in decoder block
            yield return new Conv3DLayer<T>(
                inputChannels: outputFilters,
                outputChannels: outputFilters,
                kernelSize: 3,
                inputDepth: currentResolution,
                inputHeight: currentResolution,
                inputWidth: currentResolution,
                stride: 1,
                padding: 1,
                activation: new ReLUActivation<T>());
        }

        // ============== OUTPUT LAYER ==============
        // 1x1x1 convolution to produce per-voxel class predictions
        yield return new Conv3DLayer<T>(
            inputChannels: baseFilters,
            outputChannels: numClasses,
            kernelSize: 1,
            inputDepth: currentResolution,
            inputHeight: currentResolution,
            inputWidth: currentResolution,
            stride: 1,
            padding: 0,
            activation: numClasses > 1 ? new SoftmaxActivation<T>() : new SigmoidActivation<T>());
    }

    /// <summary>
    /// Creates default layers for a MeshCNN architecture for mesh classification/segmentation.
    /// </summary>
    /// <param name="architecture">The neural network architecture specification.</param>
    /// <param name="inputFeatures">Number of input features per edge. Default is 5.</param>
    /// <param name="convChannels">Channel sizes for each edge convolution block.</param>
    /// <param name="poolTargets">Target edge counts after each pooling operation.</param>
    /// <param name="fcSizes">Sizes of fully connected layers before output.</param>
    /// <param name="numNeighbors">Number of neighboring edges per edge. Default is 4.</param>
    /// <param name="useBatchNorm">Whether to use batch normalization. Default is true.</param>
    /// <param name="dropoutRate">Dropout rate for regularization. Default is 0.5.</param>
    /// <param name="useGlobalAveragePooling">Whether to use global average pooling. Default is false (max pooling).</param>
    /// <returns>A collection of layers configured for mesh processing.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> MeshCNN processes 3D mesh data by learning from edge features.
    /// </para>
    /// <para>
    /// The architecture consists of:
    /// - Edge convolution blocks: Learn patterns from edge neighborhoods
    /// - Mesh pooling: Simplify the mesh by removing less important edges
    /// - Global pooling: Aggregate all edge features into a fixed-size vector
    /// - Fully connected layers: Map aggregated features to class predictions
    /// </para>
    /// <para>
    /// Applications include:
    /// - 3D shape classification from mesh data
    /// - Mesh segmentation (labeling different parts)
    /// - Learning from CAD models and 3D scans
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown when the architecture has invalid output size.</exception>
    public static IEnumerable<ILayer<T>> CreateDefaultMeshCNNLayers(
        NeuralNetworkArchitecture<T> architecture,
        int inputFeatures = 5,
        int[]? convChannels = null,
        int[]? poolTargets = null,
        int[]? fcSizes = null,
        int numNeighbors = 4,
        bool useBatchNorm = true,
        double dropoutRate = 0.5,
        bool useGlobalAveragePooling = false)
    {
        if (architecture.OutputSize <= 0)
        {
            throw new InvalidOperationException("Output size must be specified and greater than 0 for MeshCNN.");
        }

        convChannels ??= [64, 128, 256, 256];
        poolTargets ??= [1800, 1350, 600];
        fcSizes ??= [100];

        if (inputFeatures <= 0)
        {
            throw new ArgumentException("Input features must be positive.", nameof(inputFeatures));
        }

        int numClasses = architecture.OutputSize;
        int currentChannels = inputFeatures;

        // Edge convolution blocks with optional pooling
        for (int block = 0; block < convChannels.Length; block++)
        {
            int outChannels = convChannels[block];

            // MeshEdgeConv layer
            yield return new MeshEdgeConvLayer<T>(
                inputChannels: currentChannels,
                outputChannels: outChannels,
                numNeighbors: numNeighbors,
                activation: new ReLUActivation<T>());

            currentChannels = outChannels;

            // MeshPool layer (if we have a target for this block)
            if (block < poolTargets.Length)
            {
                yield return new MeshPoolLayer<T>(
                    inputChannels: currentChannels,
                    targetEdges: poolTargets[block],
                    numNeighbors: numNeighbors);
            }
        }

        // Global pooling to aggregate edge features
        // Note: MeshCNN typically uses a simple max/avg over all edges
        yield return new GlobalPoolingLayer<T>(
            inputShape: [currentChannels],
            poolingType: useGlobalAveragePooling ? PoolingType.Average : PoolingType.Max,
            activationFunction: (IActivationFunction<T>?)null);

        // Fully connected layers
        int fcInput = currentChannels;
        foreach (var fcSize in fcSizes)
        {
            yield return new DenseLayer<T>(
                inputSize: fcInput,
                outputSize: fcSize,
                activationFunction: new ReLUActivation<T>());

            if (dropoutRate > 0)
            {
                yield return new DropoutLayer<T>(dropoutRate);
            }

            fcInput = fcSize;
        }

        // Output layer
        IActivationFunction<T> outputActivation = architecture.TaskType == NeuralNetworkTaskType.MultiClassClassification
            ? new SoftmaxActivation<T>()
            : architecture.TaskType == NeuralNetworkTaskType.BinaryClassification
                ? new SigmoidActivation<T>()
                : new IdentityActivation<T>();

        yield return new DenseLayer<T>(
            inputSize: fcInput,
            outputSize: numClasses,
            activationFunction: outputActivation);
    }

    /// <summary>
    /// Creates the default layer sequence for a SpiralNet mesh neural network.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="inputFeatures">Number of input features per vertex (default: 3 for coordinates).</param>
    /// <param name="spiralLength">Length of spiral sequences for convolutions.</param>
    /// <param name="convChannels">Channel sizes for each spiral convolution block.</param>
    /// <param name="poolRatios">Pooling ratios for mesh simplification at each level.</param>
    /// <param name="fcSizes">Sizes of fully connected layers before output.</param>
    /// <param name="useBatchNorm">Whether to use batch normalization after convolutions.</param>
    /// <param name="dropoutRate">Dropout rate for fully connected layers.</param>
    /// <param name="useGlobalAveragePooling">Whether to use global average (true) or max (false) pooling.</param>
    /// <returns>An enumerable of layers forming the SpiralNet architecture.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method builds the default layer stack for SpiralNet++.</para>
    /// <para>
    /// Architecture pattern:
    /// - Multiple spiral convolution blocks (SpiralConv + optional BatchNorm)
    /// - Global pooling to aggregate vertex features
    /// - Fully connected layers for classification
    /// 
    /// Applications:
    /// - 3D face recognition and reconstruction
    /// - Human body shape analysis
    /// - Medical mesh analysis
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown when the architecture has invalid output size.</exception>
    public static IEnumerable<ILayer<T>> CreateDefaultSpiralNetLayers(
        NeuralNetworkArchitecture<T> architecture,
        int inputFeatures = 3,
        int spiralLength = 9,
        int[]? convChannels = null,
        double[]? poolRatios = null,
        int[]? fcSizes = null,
        bool useBatchNorm = true,
        double dropoutRate = 0.5,
        bool useGlobalAveragePooling = true)
    {
        if (architecture.OutputSize <= 0)
        {
            throw new InvalidOperationException("Output size must be specified and greater than 0 for SpiralNet.");
        }

        convChannels ??= [32, 64, 128, 256];
        poolRatios ??= [0.5, 0.5];
        fcSizes ??= [256, 128];

        if (inputFeatures <= 0)
        {
            throw new ArgumentException("Input features must be positive.", nameof(inputFeatures));
        }

        if (spiralLength <= 0)
        {
            throw new ArgumentException("Spiral length must be positive.", nameof(spiralLength));
        }

        int numClasses = architecture.OutputSize;
        int currentChannels = inputFeatures;

        // Spiral convolution blocks
        for (int block = 0; block < convChannels.Length; block++)
        {
            int outChannels = convChannels[block];

            // SpiralConv layer
            yield return new SpiralConvLayer<T>(
                inputChannels: currentChannels,
                outputChannels: outChannels,
                spiralLength: spiralLength,
                activation: new ReLUActivation<T>());

            currentChannels = outChannels;

            // Optional batch normalization
            if (useBatchNorm)
            {
                yield return new BatchNormalizationLayer<T>(currentChannels);
            }
        }

        // Global pooling to aggregate vertex features
        yield return new GlobalPoolingLayer<T>(
            inputShape: [currentChannels],
            poolingType: useGlobalAveragePooling ? PoolingType.Average : PoolingType.Max,
            activationFunction: (IActivationFunction<T>?)null);

        // Fully connected layers
        int fcInput = currentChannels;
        foreach (var fcSize in fcSizes)
        {
            yield return new DenseLayer<T>(
                inputSize: fcInput,
                outputSize: fcSize,
                activationFunction: new ReLUActivation<T>());

            if (dropoutRate > 0)
            {
                yield return new DropoutLayer<T>(dropoutRate);
            }

            fcInput = fcSize;
        }

        // Output layer
        IActivationFunction<T> outputActivation = architecture.TaskType == NeuralNetworkTaskType.MultiClassClassification
            ? new SoftmaxActivation<T>()
            : architecture.TaskType == NeuralNetworkTaskType.BinaryClassification
                ? new SigmoidActivation<T>()
                : new IdentityActivation<T>();

        yield return new DenseLayer<T>(
            inputSize: fcInput,
            outputSize: numClasses,
            activationFunction: outputActivation);
    }

    /// <summary>
    /// Creates default layers for CLIP-style multimodal networks.
    /// </summary>
    /// <param name="architecture">The neural network architecture specification.</param>
    /// <param name="projectionDim">The projection dimension for embeddings (default: 512).</param>
    /// <returns>A collection of projection layers for CLIP fine-tuning.</returns>
    /// <remarks>
    /// <para>
    /// CLIP uses pre-trained ONNX encoders for most of its work,
    /// but these layers provide optional projection heads for fine-tuning or feature extraction.
    /// </para>
    /// <para><b>For Beginners:</b> CLIP has two main parts: an image encoder and a text encoder.
    /// These pre-trained encoders are loaded from ONNX files. The projection layers here are
    /// optional additions that can:
    /// - Adapt the embeddings for specific tasks
    /// - Allow fine-tuning on new domains
    /// - Match embedding dimensions between different model variants
    ///
    /// If you're just using CLIP for inference (getting embeddings), you typically don't
    /// need these layers. They're useful when you want to adapt CLIP for a specific task.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultClipLayers(
        NeuralNetworkArchitecture<T> architecture,
        int projectionDim = 512)
    {
        // CLIP typically uses 768 (ViT-L) or 512 (ViT-B) as embedding dimensions
        int imageEmbeddingDim = architecture.ImageEmbeddingDim > 0 ? architecture.ImageEmbeddingDim : 768;
        int textEmbeddingDim = architecture.TextEmbeddingDim > 0 ? architecture.TextEmbeddingDim : 512;

        // Image projection head (optional, for fine-tuning)
        // Projects image embeddings to the shared projection space
        yield return new DenseLayer<T>(
            inputSize: imageEmbeddingDim,
            outputSize: projectionDim,
            activationFunction: null); // Linear projection (no activation)

        // Text projection head (optional, for fine-tuning)
        // Projects text embeddings to the shared projection space
        yield return new DenseLayer<T>(
            inputSize: textEmbeddingDim,
            outputSize: projectionDim,
            activationFunction: null); // Linear projection (no activation)
    }
}

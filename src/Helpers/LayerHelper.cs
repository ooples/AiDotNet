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
                activationFunction: new ReLUActivation<T>()
            );
            yield return new MaxPoolingLayer<T>(
                inputShape: [filterCount, inputShape[1], inputShape[2]],
                poolSize: 2,
                stride: 2
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
    /// Creates layers for a VGG network based on the specified configuration.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="configuration">The VGG-specific configuration.</param>
    /// <returns>A collection of layers forming a VGG network.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> VGG networks are deep convolutional neural networks known for their
    /// simplicity and effectiveness. They use stacks of 3x3 convolutions followed by max pooling
    /// to progressively extract higher-level features from images.
    /// </para>
    /// <para>
    /// The VGG architecture consists of:
    /// <list type="bullet">
    /// <item>5 convolutional blocks with increasing number of filters (64 -> 128 -> 256 -> 512 -> 512)</item>
    /// <item>Max pooling after each block to reduce spatial dimensions by half</item>
    /// <item>Optional batch normalization after each convolution (in _BN variants)</item>
    /// <item>3 fully connected layers (4096 -> 4096 -> numClasses)</item>
    /// <item>Dropout regularization in the fully connected layers</item>
    /// </list>
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultVGGLayers(
        NeuralNetworkArchitecture<T> architecture,
        Configuration.VGGConfiguration configuration)
    {
        if (configuration == null)
        {
            throw new ArgumentNullException(nameof(configuration));
        }

        var inputShape = architecture.GetInputShape();
        int currentChannels = inputShape[0];
        int currentHeight = inputShape[1];
        int currentWidth = inputShape[2];

        var blockConfig = configuration.BlockConfiguration;

        // Process each VGG block
        for (int blockIdx = 0; blockIdx < blockConfig.Length; blockIdx++)
        {
            var block = blockConfig[blockIdx];

            // Add convolutional layers in this block
            for (int convIdx = 0; convIdx < block.Length; convIdx++)
            {
                int outputChannels = block[convIdx];

                // Convolutional layer with 3x3 kernel
                yield return new ConvolutionalLayer<T>(
                    inputDepth: currentChannels,
                    inputHeight: currentHeight,
                    inputWidth: currentWidth,
                    outputDepth: outputChannels,
                    kernelSize: 3,
                    stride: 1,
                    padding: 1,  // Same padding to preserve spatial dimensions
                    activationFunction: new ReLUActivation<T>()
                );

                // Optional batch normalization (per-channel normalization)
                // BatchNormalizationLayer only needs the number of channels - spatial dimensions
                // are handled dynamically in the forward pass via Engine.BatchNorm
                if (configuration.UseBatchNormalization)
                {
                    yield return new BatchNormalizationLayer<T>(outputChannels);
                }

                currentChannels = outputChannels;
            }

            // Max pooling after each block (2x2, stride 2)
            yield return new MaxPoolingLayer<T>(
                inputShape: [currentChannels, currentHeight, currentWidth],
                poolSize: 2,
                stride: 2
            );

            currentHeight /= 2;
            currentWidth /= 2;
        }

        // Flatten before fully connected layers
        int flattenedSize = currentChannels * currentHeight * currentWidth;
        yield return new FlattenLayer<T>(inputShape: [currentChannels, currentHeight, currentWidth]);

        // Classifier (fully connected layers) - only if included
        if (configuration.IncludeClassifier)
        {
            // FC1: flattenedSize -> 4096
            yield return new DenseLayer<T>(
                inputSize: flattenedSize,
                outputSize: 4096,
                activationFunction: new ReLUActivation<T>()
            );
            yield return new DropoutLayer<T>((float)configuration.DropoutRate);

            // FC2: 4096 -> 4096
            yield return new DenseLayer<T>(
                inputSize: 4096,
                outputSize: 4096,
                activationFunction: new ReLUActivation<T>()
            );
            yield return new DropoutLayer<T>((float)configuration.DropoutRate);

            // FC3 (Output): 4096 -> numClasses
            yield return new DenseLayer<T>(
                inputSize: 4096,
                outputSize: configuration.NumClasses,
                activationFunction: new SoftmaxActivation<T>() as IActivationFunction<T>
            );
        }
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

        // Handle different input dimensionalities
        if (inputShape.Length == 1)
        {
            // 1D input: Use Dense layers with residual connections (MLP-style ResNet)
            foreach (var layer in CreateDefaultResNet1DLayers(architecture, inputShape[0], blockCount, blockSize))
            {
                yield return layer;
            }
        }
        else if (inputShape.Length == 2)
        {
            // 2D input: Treat as single-channel image [1, height, width]
            foreach (var layer in CreateDefaultResNet2DLayers(architecture, inputShape, blockCount, blockSize))
            {
                yield return layer;
            }
        }
        else
        {
            // 3D input: Standard CNN-based ResNet
            foreach (var layer in CreateDefaultResNet3DLayers(architecture, inputShape, blockCount, blockSize))
            {
                yield return layer;
            }
        }
    }

    /// <summary>
    /// Creates ResNet layers for 1D (flat vector) input using Dense layers with residual connections.
    /// </summary>
    private static IEnumerable<ILayer<T>> CreateDefaultResNet1DLayers(NeuralNetworkArchitecture<T> architecture, int inputSize, int blockCount, int blockSize)
    {
        int hiddenSize = Math.Max(64, inputSize);
        int currentSize = inputSize;

        // Initial projection layer
        yield return new DenseLayer<T>(currentSize, hiddenSize, new ReLUActivation<T>() as IActivationFunction<T>);
        currentSize = hiddenSize;

        // Residual blocks using Dense layers
        for (int i = 0; i < blockCount; i++)
        {
            for (int j = 0; j < blockSize; j++)
            {
                // Each "residual block" is a Dense layer with skip connection via ResidualLayer
                yield return new ResidualLayer<T>(
                    inputShape: [currentSize],
                    innerLayer: new DenseLayer<T>(currentSize, currentSize, new ReLUActivation<T>() as IActivationFunction<T>),
                    activationFunction: new ReLUActivation<T>()
                );
            }

            // Optionally expand dimensions between blocks (except last)
            if (i < blockCount - 1)
            {
                int newSize = Math.Min(currentSize * 2, 512);
                yield return new DenseLayer<T>(currentSize, newSize, new ReLUActivation<T>() as IActivationFunction<T>);
                currentSize = newSize;
            }
        }

        // Final output layer
        yield return new DenseLayer<T>(currentSize, architecture.OutputSize, new SoftmaxActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Creates ResNet layers for 2D input by treating it as a single-channel image.
    /// </summary>
    private static IEnumerable<ILayer<T>> CreateDefaultResNet2DLayers(NeuralNetworkArchitecture<T> architecture, int[] inputShape, int blockCount, int blockSize)
    {
        // For 2D input [height, width], treat as single-channel image [1, height, width]
        int inputDepth = 1;
        int inputHeight = inputShape[0];
        int inputWidth = inputShape[1];

        foreach (var layer in CreateResNetConvLayers(architecture, inputDepth, inputHeight, inputWidth, blockCount, blockSize))
        {
            yield return layer;
        }
    }

    /// <summary>
    /// Creates ResNet layers for 3D input (standard CNN-based ResNet).
    /// </summary>
    private static IEnumerable<ILayer<T>> CreateDefaultResNet3DLayers(NeuralNetworkArchitecture<T> architecture, int[] inputShape, int blockCount, int blockSize)
    {
        int inputDepth = inputShape[0];
        int inputHeight = inputShape[1];
        int inputWidth = inputShape[2];

        foreach (var layer in CreateResNetConvLayers(architecture, inputDepth, inputHeight, inputWidth, blockCount, blockSize))
        {
            yield return layer;
        }
    }

    /// <summary>
    /// Creates convolutional ResNet layers for 2D/3D image-like input.
    /// </summary>
    private static IEnumerable<ILayer<T>> CreateResNetConvLayers(NeuralNetworkArchitecture<T> architecture, int inputDepth, int inputHeight, int inputWidth, int blockCount, int blockSize)
    {
        int ConvolutionOutputSize(int inputSize, int kernelSize, int stride, int padding)
        {
            if (inputSize + 2 * padding < kernelSize)
                throw new ArgumentException("Input dimensions with padding must be at least kernel size.");

            return (inputSize - kernelSize + 2 * padding) / stride + 1;
        }

        int PoolingOutputSize(int inputSize, int poolSize, int stride)
        {
            return (inputSize - poolSize) / stride + 1;
        }

        const int initialKernelSize = 7;
        const int initialStride = 2;
        const int initialPadding = 3;

        int convOutputHeight = ConvolutionOutputSize(inputHeight, initialKernelSize, initialStride, initialPadding);
        int convOutputWidth = ConvolutionOutputSize(inputWidth, initialKernelSize, initialStride, initialPadding);

        const int initialPoolSize = 3;
        const int initialPoolStride = 2;

        int pooledHeight = PoolingOutputSize(convOutputHeight, initialPoolSize, initialPoolStride);
        int pooledWidth = PoolingOutputSize(convOutputWidth, initialPoolSize, initialPoolStride);
        // Initial convolutional layer
        yield return new ConvolutionalLayer<T>(
            inputDepth: inputDepth,
            inputHeight: inputHeight,
            inputWidth: inputWidth,
            outputDepth: 64,
            kernelSize: initialKernelSize,
            stride: initialStride,
            padding: initialPadding,
            activationFunction: new ReLUActivation<T>()
        );

        yield return new MaxPoolingLayer<T>(
            inputShape: [64, convOutputHeight, convOutputWidth],
            poolSize: initialPoolSize,
            stride: initialPoolStride
        );

        // Residual blocks
        int currentDepth = 64;
        int currentHeight = pooledHeight;
        int currentWidth = pooledWidth;

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
                int nextHeight = PoolingOutputSize(currentHeight, 2, 2);
                int nextWidth = PoolingOutputSize(currentWidth, 2, 2);

                yield return new MaxPoolingLayer<T>(
                    inputShape: [currentDepth, currentHeight, currentWidth],
                    poolSize: 2,
                    stride: 2
                );
                currentHeight = nextHeight;
                currentWidth = nextWidth;
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
                activationFunction: new IdentityActivation<T>()
            );
        }

        yield return new ResidualLayer<T>(
             inputShape: [outputDepth, height, width],
             innerLayer: innerLayer,
             activationFunction: new IdentityActivation<T>()
         );

        yield return new ConvolutionalLayer<T>(
            inputDepth: inputDepth,
            outputDepth: outputDepth,
            kernelSize: 3,
            inputHeight: height,
            inputWidth: width,
            stride: 1,
            padding: 1,
            activationFunction: new ReLUActivation<T>()
        );

        yield return new ConvolutionalLayer<T>(
            inputDepth: outputDepth,
            outputDepth: outputDepth,
            kernelSize: 3,
            inputHeight: height,
            inputWidth: width,
            stride: 1,
            padding: 1,
            activationFunction: new ReLUActivation<T>()
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
        int inputSize = inputShape.Length > 0 ? inputShape.Aggregate(1, (a, b) => a * b) : architecture.CalculatedInputSize;
        int[] layerSizes = architecture.GetLayerSizes();

        // If no layers specified, create a default symmetric autoencoder architecture
        // Structure: input -> hidden1 -> bottleneck -> hidden2 -> output
        if (layerSizes.Length < 3)
        {
            int outputSize = architecture.OutputSize > 0 ? architecture.OutputSize : inputSize;
            int hidden1 = Math.Max(inputSize / 2, 8);
            int bottleneck = Math.Max(inputSize / 4, 4);
            int hidden2 = hidden1;
            layerSizes = [inputSize, hidden1, bottleneck, hidden2, outputSize];
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
            activationFunction: new ReLUActivation<T>()
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

        // Controller input includes read vectors concatenated: inputSize + readHeads * memoryWordSize
        int controllerInputSize = inputSize + readHeads * memoryWordSize;

        // Controller (Feed-forward network) - first layer takes the combined input
        yield return new DenseLayer<T>(controllerInputSize, controllerSize, new ReLUActivation<T>() as IActivationFunction<T>);

        // Controller output layer - produces BOTH direct output (controllerSize) AND interface signals
        // The DNC's CombineControllerOutputWithReadVectors expects:
        // controllerOutput.Shape[1] = controllerDirectOutputSize + interfaceSize
        int controllerOutputSize = controllerSize + interfaceSize;
        yield return new DenseLayer<T>(controllerSize, controllerOutputSize, new IdentityActivation<T>() as IActivationFunction<T>);
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
        int flatInputSize = inputDepth * inputHeight * inputWidth;

        // For 1D input or very small spatial dimensions, use dense-only architecture
        // Pooling doesn't make sense for 1x1 spatial dimensions
        bool use1DArchitecture = inputHeight <= 1 && inputWidth <= 1;

        if (use1DArchitecture)
        {
            // 1D VAE: All dense layers, no pooling
            int hidden1 = Math.Max(flatInputSize / 2, latentSize * 4);
            int hidden2 = Math.Max(hidden1 / 2, latentSize * 2);
            int encoderOutputSize = latentSize * 2;

            // Encoder layers
            yield return new DenseLayer<T>(flatInputSize, hidden1, new LeakyReLUActivation<T>() as IActivationFunction<T>);
            yield return new DenseLayer<T>(hidden1, hidden2, new LeakyReLUActivation<T>() as IActivationFunction<T>);
            yield return new DenseLayer<T>(hidden2, encoderOutputSize, new IdentityActivation<T>() as IActivationFunction<T>);

            // Mean and LogVariance layers
            yield return new MeanLayer<T>([encoderOutputSize], axis: 0);
            yield return new LogVarianceLayer<T>([encoderOutputSize], axis: 0);

            // Decoder layers (mirror of encoder)
            yield return new DenseLayer<T>(latentSize, hidden2, new LeakyReLUActivation<T>() as IActivationFunction<T>);
            yield return new DenseLayer<T>(hidden2, hidden1, new LeakyReLUActivation<T>() as IActivationFunction<T>);

            // Output layer
            yield return new DenseLayer<T>(hidden1, flatInputSize, new SigmoidActivation<T>() as IActivationFunction<T>);
        }
        else
        {
            // 2D/3D VAE: With pooling and upsampling
            // Encoder layers
            yield return new DenseLayer<T>(flatInputSize, flatInputSize / 2, new LeakyReLUActivation<T>() as IActivationFunction<T>);

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
            int encoderOutputSize = latentSize * 2;
            yield return new DenseLayer<T>(pooledSize / 2, encoderOutputSize, new IdentityActivation<T>() as IActivationFunction<T>);

            // Mean and LogVariance layers
            yield return new MeanLayer<T>([encoderOutputSize], axis: 0);
            yield return new LogVarianceLayer<T>([encoderOutputSize], axis: 0);

            // Decoder layers
            yield return new DenseLayer<T>(latentSize, pooledSize / 2, new LeakyReLUActivation<T>() as IActivationFunction<T>);
            yield return new DenseLayer<T>(pooledSize / 2, pooledSize, new LeakyReLUActivation<T>() as IActivationFunction<T>);

            // Add an Upsampling layer to match the pooling in the encoder
            yield return new UpsamplingLayer<T>([pooledDepth, pooledHeight, pooledWidth], 2);

            yield return new DenseLayer<T>(flatInputSize, flatInputSize / 2, new LeakyReLUActivation<T>() as IActivationFunction<T>);

            // Output layer
            yield return new DenseLayer<T>(flatInputSize / 2, flatInputSize, new SigmoidActivation<T>() as IActivationFunction<T>);
        }
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
        else
        {
            // For continuous inputs (no embedding), add input projection layer if needed
            // This projects the input from inputSize to modelDimension
            int inputSize = architecture.InputSize;
            if (inputSize > 0 && inputSize != modelDimension)
            {
                yield return new DenseLayer<T>(inputSize, modelDimension, new IdentityActivation<T>() as IActivationFunction<T>);
            }
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

        // For classification tasks, add global pooling to reduce 3D [batch, seq, dim] to 2D [batch, dim]
        // This is required because transformer encoder outputs are 3D, but classification heads expect 2D
        if (taskType == NeuralNetworkTaskType.BinaryClassification ||
            taskType == NeuralNetworkTaskType.MultiClassClassification ||
            taskType == NeuralNetworkTaskType.MultiLabelClassification ||
            taskType == NeuralNetworkTaskType.SequenceClassification ||
            taskType == NeuralNetworkTaskType.ImageClassification)
        {
            // Global average pooling over sequence dimension
            // Input: [batch, seq, dim] -> Output: [batch, dim]
            yield return new GlobalPoolingLayer<T>([maxSequenceLength, modelDimension], PoolingType.Average, (IActivationFunction<T>?)null);
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

        // Get input shape to determine feature dimension
        var inputShape = architecture.GetInputShape();
        if (inputShape == null || inputShape.Length == 0)
        {
            throw new InvalidOperationException("Input shape must be specified for GRU network.");
        }

        // For sequence input [seqLen, features], inputSize is the feature dimension (last dim)
        // For 1D input [features], inputSize is the only dimension
        // This matches the LSTM pattern for consistency
        int inputSize = inputShape[inputShape.Length - 1];
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
        // Note: memoryDimension must match the memory vector dimension (embeddingSize),
        // since memory in MemoryNetwork has shape [memorySize, embeddingSize]
        yield return new MemoryReadLayer<T>(
            inputDimension: embeddingSize,
            memoryDimension: embeddingSize,
            outputDimension: embeddingSize,
            activationFunction: new ReLUActivation<T>() as IActivationFunction<T>
        );

        // Dense Layer for processing memory read output
        // Note: MemoryReadLayer outputs embeddingSize features
        yield return new DenseLayer<T>(
            inputSize: embeddingSize,
            outputSize: hiddenSize,
            activationFunction: new ReLUActivation<T>()
        );

        // Memory Write Layer
        // Note: memoryDimension must match the memory vector dimension (embeddingSize)
        yield return new MemoryWriteLayer<T>(
            inputDimension: hiddenSize,
            memoryDimension: embeddingSize,
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
        // For 2D input [seqLen, features], the input size is the feature dimension
        var inputShape = architecture.GetInputShape();
        int inputSize = inputShape.Length >= 2 ? inputShape[1] : inputShape[0];
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

        // Extract the last timestep from the sequence for classification tasks
        // RNN layers output [seqLen, hiddenSize], but Dense layer expects [hiddenSize]
        yield return new SequenceLastLayer<T>(hiddenSize);

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

        // Input layer - projects input to reservoir size
        yield return new DenseLayer<T>(inputSize, reservoirSize, new TanhActivation<T>() as IActivationFunction<T>);

        // Reservoir layer (liquid) - receives output from DenseLayer which is of size reservoirSize
        yield return new ReservoirLayer<T>(
            reservoirSize,  // Input to reservoir is the output of the DenseLayer
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

        // For sequence input [seqLen, features], inputSize is the feature dimension (last dim)
        // For 1D input [features], inputSize is the only dimension
        int inputSize = inputShape[inputShape.Length - 1];
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
                activationFunction: new ReLUActivation<T>());

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
                activationFunction: new ReLUActivation<T>());

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
                activationFunction: new ReLUActivation<T>());

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
            activationFunction: new ReLUActivation<T>());

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
                activationFunction: new ReLUActivation<T>());

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
                activationFunction: new ReLUActivation<T>());
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
            activationFunction: numClasses > 1 ? new SoftmaxActivation<T>() : new SigmoidActivation<T>());
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
                activationFunction: new ReLUActivation<T>());

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
                activationFunction: new ReLUActivation<T>());

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
    /// Creates default layers for a DenseNet network based on the specified configuration.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="configuration">The DenseNet-specific configuration.</param>
    /// <returns>A collection of layers forming a DenseNet network.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> DenseNet (Densely Connected Convolutional Network) connects each layer
    /// to every other layer in a feed-forward fashion. This creates strong gradient flow and
    /// feature reuse, enabling very deep networks with fewer parameters.
    /// </para>
    /// <para>
    /// The DenseNet architecture consists of:
    /// <list type="bullet">
    /// <item>Stem: Initial 7x7 conv with stride 2, followed by 3x3 max pooling</item>
    /// <item>Dense Blocks: Multiple dense blocks with transition layers between them</item>
    /// <item>Transition Layers: 1x1 conv for channel reduction followed by 2x2 avg pooling</item>
    /// <item>Classification Head: Global average pooling followed by a dense layer</item>
    /// </list>
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultDenseNetLayers(
        NeuralNetworkArchitecture<T> architecture,
        Configuration.DenseNetConfiguration configuration)
    {
        if (configuration == null)
        {
            throw new ArgumentNullException(nameof(configuration));
        }

        int currentHeight = configuration.InputHeight;
        int currentWidth = configuration.InputWidth;
        var blockLayers = configuration.GetBlockLayers();

        // Stem: 7x7 conv, stride 2, padding 3
        int stemChannels = 64;
        yield return new ConvolutionalLayer<T>(
            inputDepth: configuration.InputChannels,
            outputDepth: stemChannels,
            kernelSize: 7,
            inputHeight: currentHeight,
            inputWidth: currentWidth,
            stride: 2,
            padding: 3,
            activationFunction: new IdentityActivation<T>());

        currentHeight = (currentHeight + 2 * 3 - 7) / 2 + 1;
        currentWidth = (currentWidth + 2 * 3 - 7) / 2 + 1;

        yield return new BatchNormalizationLayer<T>(stemChannels);
        yield return new ActivationLayer<T>([stemChannels, currentHeight, currentWidth],
            activationFunction: new ReLUActivation<T>());

        // MaxPool 3x3, stride 2, padding 1
        yield return new MaxPoolingLayer<T>(
            inputShape: [stemChannels, currentHeight, currentWidth],
            poolSize: 3,
            stride: 2);

        currentHeight = (currentHeight + 2 * 1 - 3) / 2 + 1;
        currentWidth = (currentWidth + 2 * 1 - 3) / 2 + 1;

        int currentChannels = stemChannels;

        // Dense blocks and transitions
        for (int i = 0; i < blockLayers.Length; i++)
        {
            int numLayersInBlock = blockLayers[i];

            // Add Dense Block
            var denseBlock = new DenseBlock<T>(
                inputChannels: currentChannels,
                numLayers: numLayersInBlock,
                growthRate: configuration.GrowthRate,
                inputHeight: currentHeight,
                inputWidth: currentWidth);

            yield return denseBlock;
            currentChannels = denseBlock.OutputChannels;

            // Add Transition (except after the last block)
            if (i < blockLayers.Length - 1)
            {
                var transition = new TransitionLayer<T>(
                    inputChannels: currentChannels,
                    inputHeight: currentHeight,
                    inputWidth: currentWidth,
                    compressionFactor: configuration.CompressionFactor);

                yield return transition;
                currentChannels = transition.OutputChannels;
                currentHeight /= 2;
                currentWidth /= 2;
            }
        }

        // Final BN and ReLU
        yield return new BatchNormalizationLayer<T>(currentChannels);
        yield return new ActivationLayer<T>([currentChannels, currentHeight, currentWidth],
            activationFunction: new ReLUActivation<T>());

        // Global average pooling
        yield return new AdaptiveAveragePoolingLayer<T>(currentChannels, currentHeight, currentWidth, 1, 1);

        // Flatten
        yield return new FlattenLayer<T>([currentChannels, 1, 1]);

        // Classification head
        yield return new DenseLayer<T>(currentChannels, configuration.NumClasses,
            activationFunction: new IdentityActivation<T>());
    }

    /// <summary>
    /// Creates default layers for an EfficientNet network based on the specified configuration.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="configuration">The EfficientNet-specific configuration.</param>
    /// <returns>A collection of layers forming an EfficientNet network.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> EfficientNet uses compound scaling to balance network depth, width,
    /// and resolution. Each variant (B0-B7) represents a different scale factor, achieving
    /// excellent accuracy with fewer parameters than previous architectures.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultEfficientNetLayers(
        NeuralNetworkArchitecture<T> architecture,
        Configuration.EfficientNetConfiguration configuration)
    {
        if (configuration == null)
        {
            throw new ArgumentNullException(nameof(configuration));
        }

        var widthCoeff = configuration.GetWidthMultiplier();
        var depthCoeff = configuration.GetDepthMultiplier();
        var resolution = configuration.GetInputHeight();

        int currentHeight = resolution;
        int currentWidth = resolution;

        // Stem: 3x3 conv, stride 2
        int stemChannels = MakeScaledChannels(32, widthCoeff);
        yield return new ConvolutionalLayer<T>(
            inputDepth: configuration.InputChannels,
            outputDepth: stemChannels,
            kernelSize: 3,
            inputHeight: currentHeight,
            inputWidth: currentWidth,
            stride: 2,
            padding: 1,
            activationFunction: new IdentityActivation<T>());

        currentHeight = (currentHeight + 2 * 1 - 3) / 2 + 1;
        currentWidth = (currentWidth + 2 * 1 - 3) / 2 + 1;

        yield return new BatchNormalizationLayer<T>(stemChannels);
        yield return new ActivationLayer<T>([stemChannels, currentHeight, currentWidth],
            activationFunction: new SwishActivation<T>());

        int currentChannels = stemChannels;

        // EfficientNet-B0 block configuration:
        // (expansion, output_channels, num_layers, stride, kernel_size)
        var blockConfigs = new (int expansion, int outChannels, int numLayers, int stride, int kernelSize)[]
        {
            (1, 16, 1, 1, 3),
            (6, 24, 2, 2, 3),
            (6, 40, 2, 2, 5),
            (6, 80, 3, 2, 3),
            (6, 112, 3, 1, 5),
            (6, 192, 4, 2, 5),
            (6, 320, 1, 1, 3)
        };

        // Add MBConv blocks with SE and Swish activation
        foreach (var (expansion, outChannels, numLayers, stride, kernelSize) in blockConfigs)
        {
            int scaledOutChannels = MakeScaledChannels(outChannels, widthCoeff);
            int scaledNumLayers = MakeScaledDepth(numLayers, depthCoeff);

            // First block in each stage may have stride > 1
            yield return new InvertedResidualBlock<T>(
                inChannels: currentChannels,
                outChannels: scaledOutChannels,
                inputHeight: currentHeight,
                inputWidth: currentWidth,
                expansionRatio: expansion,
                stride: stride,
                useSE: true,
                seRatio: 4,
                activationFunction: new SwishActivation<T>());

            // Update dimensions after first block
            currentHeight = (currentHeight + stride - 1) / stride;
            currentWidth = (currentWidth + stride - 1) / stride;
            currentChannels = scaledOutChannels;

            // Remaining blocks in the stage (stride=1)
            for (int i = 1; i < scaledNumLayers; i++)
            {
                yield return new InvertedResidualBlock<T>(
                    inChannels: currentChannels,
                    outChannels: currentChannels,
                    inputHeight: currentHeight,
                    inputWidth: currentWidth,
                    expansionRatio: expansion,
                    stride: 1,
                    useSE: true,
                    seRatio: 4,
                    activationFunction: new SwishActivation<T>());
            }
        }

        // Head: 1x1 conv
        int headChannels = MakeScaledChannels(1280, widthCoeff);
        yield return new ConvolutionalLayer<T>(
            inputDepth: currentChannels,
            outputDepth: headChannels,
            kernelSize: 1,
            inputHeight: currentHeight,
            inputWidth: currentWidth,
            stride: 1,
            padding: 0,
            activationFunction: new IdentityActivation<T>());

        yield return new BatchNormalizationLayer<T>(headChannels);
        yield return new ActivationLayer<T>([headChannels, currentHeight, currentWidth],
            activationFunction: new SwishActivation<T>());

        // Global average pooling
        yield return new AdaptiveAveragePoolingLayer<T>(headChannels, currentHeight, currentWidth, 1, 1);

        // Flatten
        yield return new FlattenLayer<T>([headChannels, 1, 1]);

        // Classification head
        yield return new DenseLayer<T>(headChannels, configuration.NumClasses,
            activationFunction: new IdentityActivation<T>());
    }

    /// <summary>
    /// Creates default layers for a MobileNetV2 network based on the specified configuration.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="configuration">The MobileNetV2-specific configuration.</param>
    /// <returns>A collection of layers forming a MobileNetV2 network.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> MobileNetV2 is designed for efficient mobile inference, using
    /// inverted residual blocks with linear bottlenecks to achieve high accuracy with
    /// low computational cost.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultMobileNetV2Layers(
        NeuralNetworkArchitecture<T> architecture,
        Configuration.MobileNetV2Configuration configuration)
    {
        if (configuration == null)
        {
            throw new ArgumentNullException(nameof(configuration));
        }

        int currentHeight = configuration.InputHeight;
        int currentWidth = configuration.InputWidth;
        var alpha = configuration.Alpha;

        // Initial convolution: 3x3, stride 2
        int firstConvChannels = MakeScaledChannels(32, alpha);
        yield return new ConvolutionalLayer<T>(
            inputDepth: configuration.InputChannels,
            outputDepth: firstConvChannels,
            kernelSize: 3,
            inputHeight: currentHeight,
            inputWidth: currentWidth,
            stride: 2,
            padding: 1,
            activationFunction: new IdentityActivation<T>());

        currentHeight = (currentHeight + 2 * 1 - 3) / 2 + 1;
        currentWidth = (currentWidth + 2 * 1 - 3) / 2 + 1;

        yield return new BatchNormalizationLayer<T>(firstConvChannels);
        yield return new ActivationLayer<T>([firstConvChannels, currentHeight, currentWidth],
            activationFunction: new ReLU6Activation<T>());

        int currentChannels = firstConvChannels;

        // MobileNetV2 inverted residual block configuration:
        // (expansion, output_channels, num_blocks, stride)
        var blockConfigs = new (int expansion, int outChannels, int numBlocks, int stride)[]
        {
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1)
        };

        // Add inverted residual blocks
        foreach (var (expansion, outChannels, numBlocks, stride) in blockConfigs)
        {
            int scaledOutChannels = MakeScaledChannels(outChannels, alpha);

            // First block in each stage may have stride > 1
            yield return new InvertedResidualBlock<T>(
                inChannels: currentChannels,
                outChannels: scaledOutChannels,
                inputHeight: currentHeight,
                inputWidth: currentWidth,
                expansionRatio: expansion,
                stride: stride,
                useSE: false,
                activationFunction: new ReLU6Activation<T>());

            // Update dimensions after first block
            currentHeight = (currentHeight + stride - 1) / stride;
            currentWidth = (currentWidth + stride - 1) / stride;
            currentChannels = scaledOutChannels;

            // Remaining blocks in the stage (stride=1)
            for (int i = 1; i < numBlocks; i++)
            {
                yield return new InvertedResidualBlock<T>(
                    inChannels: currentChannels,
                    outChannels: currentChannels,
                    inputHeight: currentHeight,
                    inputWidth: currentWidth,
                    expansionRatio: expansion,
                    stride: 1,
                    useSE: false,
                    activationFunction: new ReLU6Activation<T>());
            }
        }

        // Final 1x1 convolution
        // Per MobileNetV2 spec: base is 1280, scaled by alpha for alpha > 1.0
        int finalConvChannels = configuration.WidthMultiplier switch
        {
            Enums.MobileNetV2WidthMultiplier.Alpha140 => 1792,  // 1280 * 1.4 = 1792
            Enums.MobileNetV2WidthMultiplier.Alpha130 => 1664,  // 1280 * 1.3 = 1664
            Enums.MobileNetV2WidthMultiplier.Alpha125 => 1600,  // 1280 * 1.25 = 1600
            _ => 1280  // For alpha <= 1.0, keep at 1280 for better accuracy
        };
        yield return new ConvolutionalLayer<T>(
            inputDepth: currentChannels,
            outputDepth: finalConvChannels,
            kernelSize: 1,
            inputHeight: currentHeight,
            inputWidth: currentWidth,
            stride: 1,
            padding: 0,
            activationFunction: new IdentityActivation<T>());

        yield return new BatchNormalizationLayer<T>(finalConvChannels);
        yield return new ActivationLayer<T>([finalConvChannels, currentHeight, currentWidth],
            activationFunction: new ReLU6Activation<T>());

        // Global average pooling
        yield return new AdaptiveAveragePoolingLayer<T>(finalConvChannels, currentHeight, currentWidth, 1, 1);

        // Flatten
        yield return new FlattenLayer<T>([finalConvChannels, 1, 1]);

        // Classification head
        yield return new DenseLayer<T>(finalConvChannels, configuration.NumClasses,
            activationFunction: new IdentityActivation<T>());
    }

    /// <summary>
    /// Creates default layers for a MobileNetV3 network based on the specified configuration.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="configuration">The MobileNetV3-specific configuration.</param>
    /// <returns>A collection of layers forming a MobileNetV3 network.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> MobileNetV3 builds on MobileNetV2 with additional optimizations
    /// including squeeze-and-excitation blocks and hard-swish activation for improved
    /// accuracy and efficiency.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultMobileNetV3Layers(
        NeuralNetworkArchitecture<T> architecture,
        Configuration.MobileNetV3Configuration configuration)
    {
        if (configuration == null)
        {
            throw new ArgumentNullException(nameof(configuration));
        }

        int currentHeight = configuration.InputHeight;
        int currentWidth = configuration.InputWidth;
        var alpha = configuration.Alpha;
        bool isLarge = configuration.Variant == Enums.MobileNetV3Variant.Large;

        // Initial convolution: 3x3, stride 2
        int firstConvChannels = MakeScaledChannels(16, alpha);
        yield return new ConvolutionalLayer<T>(
            inputDepth: configuration.InputChannels,
            outputDepth: firstConvChannels,
            kernelSize: 3,
            inputHeight: currentHeight,
            inputWidth: currentWidth,
            stride: 2,
            padding: 1,
            activationFunction: new IdentityActivation<T>());

        currentHeight = (currentHeight + 2 * 1 - 3) / 2 + 1;
        currentWidth = (currentWidth + 2 * 1 - 3) / 2 + 1;

        yield return new BatchNormalizationLayer<T>(firstConvChannels);
        yield return new ActivationLayer<T>([firstConvChannels, currentHeight, currentWidth],
            activationFunction: new HardSwishActivation<T>());

        int currentChannels = firstConvChannels;

        // Get block configurations based on variant
        var blockConfigs = isLarge
            ? GetMobileNetV3LargeBlocks(alpha)
            : GetMobileNetV3SmallBlocks(alpha);

        // Add inverted residual blocks
        foreach (var block in blockConfigs)
        {
            yield return new InvertedResidualBlock<T>(
                inChannels: currentChannels,
                outChannels: block.outChannels,
                inputHeight: currentHeight,
                inputWidth: currentWidth,
                expansionRatio: block.expansion,
                stride: block.stride,
                useSE: block.useSE,
                seRatio: 4,
                activationFunction: block.useHardSwish ? new HardSwishActivation<T>() : new ReLUActivation<T>());

            // Update dimensions after block
            currentHeight = (currentHeight + block.stride - 1) / block.stride;
            currentWidth = (currentWidth + block.stride - 1) / block.stride;
            currentChannels = block.outChannels;
        }

        // Final convolution layers
        int penultimateChannels = isLarge ? MakeScaledChannels(960, alpha) : MakeScaledChannels(576, alpha);
        yield return new ConvolutionalLayer<T>(
            inputDepth: currentChannels,
            outputDepth: penultimateChannels,
            kernelSize: 1,
            inputHeight: currentHeight,
            inputWidth: currentWidth,
            stride: 1,
            padding: 0,
            activationFunction: new IdentityActivation<T>());

        yield return new BatchNormalizationLayer<T>(penultimateChannels);
        yield return new ActivationLayer<T>([penultimateChannels, currentHeight, currentWidth],
            activationFunction: new HardSwishActivation<T>());

        // Global average pooling
        yield return new AdaptiveAveragePoolingLayer<T>(penultimateChannels, currentHeight, currentWidth, 1, 1);

        // Final classification layers
        int finalChannels = isLarge ? 1280 : 1024;
        yield return new ConvolutionalLayer<T>(
            inputDepth: penultimateChannels,
            outputDepth: finalChannels,
            kernelSize: 1,
            inputHeight: 1,
            inputWidth: 1,
            stride: 1,
            padding: 0,
            activationFunction: new IdentityActivation<T>());

        yield return new ActivationLayer<T>([finalChannels, 1, 1],
            activationFunction: new HardSwishActivation<T>());

        // Flatten
        yield return new FlattenLayer<T>([finalChannels, 1, 1]);

        // Classification head
        yield return new DenseLayer<T>(finalChannels, configuration.NumClasses,
            activationFunction: new IdentityActivation<T>());
    }

    /// <summary>
    /// Gets MobileNetV3-Large block configurations.
    /// </summary>
    private static IEnumerable<(int outChannels, int expansion, int stride, bool useSE, bool useHardSwish)> GetMobileNetV3LargeBlocks(double alpha)
    {
        // MobileNetV3-Large inverted residual block configuration
        return new[]
        {
            (MakeScaledChannels(16, alpha), 1, 1, false, false),
            (MakeScaledChannels(24, alpha), 4, 2, false, false),
            (MakeScaledChannels(24, alpha), 3, 1, false, false),
            (MakeScaledChannels(40, alpha), 3, 2, true, false),
            (MakeScaledChannels(40, alpha), 3, 1, true, false),
            (MakeScaledChannels(40, alpha), 3, 1, true, false),
            (MakeScaledChannels(80, alpha), 6, 2, false, true),
            (MakeScaledChannels(80, alpha), 2, 1, false, true),
            (MakeScaledChannels(80, alpha), 2, 1, false, true),
            (MakeScaledChannels(80, alpha), 2, 1, false, true),
            (MakeScaledChannels(112, alpha), 6, 1, true, true),
            (MakeScaledChannels(112, alpha), 6, 1, true, true),
            (MakeScaledChannels(160, alpha), 6, 2, true, true),
            (MakeScaledChannels(160, alpha), 6, 1, true, true),
            (MakeScaledChannels(160, alpha), 6, 1, true, true)
        };
    }

    /// <summary>
    /// Gets MobileNetV3-Small block configurations.
    /// </summary>
    private static IEnumerable<(int outChannels, int expansion, int stride, bool useSE, bool useHardSwish)> GetMobileNetV3SmallBlocks(double alpha)
    {
        // MobileNetV3-Small inverted residual block configuration
        return new[]
        {
            (MakeScaledChannels(16, alpha), 1, 2, true, false),
            (MakeScaledChannels(24, alpha), 4, 2, false, false),
            (MakeScaledChannels(24, alpha), 11, 1, false, false),
            (MakeScaledChannels(40, alpha), 4, 2, true, true),
            (MakeScaledChannels(40, alpha), 6, 1, true, true),
            (MakeScaledChannels(40, alpha), 6, 1, true, true),
            (MakeScaledChannels(48, alpha), 3, 1, true, true),
            (MakeScaledChannels(48, alpha), 3, 1, true, true),
            (MakeScaledChannels(96, alpha), 6, 2, true, true),
            (MakeScaledChannels(96, alpha), 6, 1, true, true),
            (MakeScaledChannels(96, alpha), 6, 1, true, true)
        };
    }

    /// <summary>
    /// Scales channel count by the width coefficient for EfficientNet/MobileNet architectures.
    /// </summary>
    private static int MakeScaledChannels(int channels, double widthCoefficient)
    {
        int scaled = (int)Math.Round(channels * widthCoefficient);
        return Math.Max(8, (scaled + 4) / 8 * 8);
    }

    /// <summary>
    /// Scales layer repeat count by the depth coefficient for EfficientNet.
    /// </summary>
    private static int MakeScaledDepth(int numLayers, double depthCoefficient)
    {
        return (int)Math.Ceiling(numLayers * depthCoefficient);
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

    /// <summary>
    /// Creates default layers for Whisper-style speech recognition models.
    /// </summary>
    /// <param name="numMels">Number of mel spectrogram bins (default: 80).</param>
    /// <param name="modelDimension">Hidden dimension of the model (default: 512).</param>
    /// <param name="numEncoderLayers">Number of encoder layers (default: 6).</param>
    /// <param name="numDecoderLayers">Number of decoder layers (default: 6).</param>
    /// <param name="numHeads">Number of attention heads (default: 8).</param>
    /// <param name="feedForwardDim">Feed-forward dimension (default: 2048).</param>
    /// <param name="vocabularySize">Output vocabulary size (default: 51865).</param>
    /// <param name="maxSequenceLength">Maximum sequence length (default: 1500).</param>
    /// <param name="dropoutRate">Dropout rate (default: 0.1).</param>
    /// <returns>A collection of layers forming a Whisper-style ASR model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Whisper is an encoder-decoder transformer for speech recognition.
    ///
    /// The architecture consists of:
    /// 1. Audio encoder: Converts mel spectrograms to hidden representations
    ///    - Convolutional layers to process spectrogram
    ///    - Transformer encoder layers with self-attention
    /// 2. Text decoder: Generates text tokens autoregressively
    ///    - Embedding layer for text tokens
    ///    - Transformer decoder layers with self-attention
    ///    - Output projection to vocabulary
    ///
    /// <b>IMPORTANT LIMITATION:</b> This method creates a flat sequential layer list which does NOT
    /// support true encoder-decoder cross-attention. The "cross-attention" layers in the decoder
    /// are actually additional self-attention layers because the flat architecture cannot route
    /// encoder outputs to the decoder. For a proper Whisper implementation with cross-attention,
    /// use the ONNX-based WhisperModel with pretrained weights, or implement a custom forward pass
    /// that explicitly passes encoder outputs to decoder cross-attention layers.
    ///
    /// This creates a trainable model structure from scratch. For inference with pre-trained weights,
    /// use the ONNX-based WhisperModel.CreateAsync() method instead.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultWhisperLayers(
        int numMels = 80,
        int modelDimension = 512,
        int numEncoderLayers = 6,
        int numDecoderLayers = 6,
        int numHeads = 8,
        int feedForwardDim = 2048,
        int vocabularySize = 51865,
        int maxSequenceLength = 1500,
        double dropoutRate = 0.1)
    {
        IActivationFunction<T> geluActivation = new GELUActivation<T>();
        IActivationFunction<T> identityActivation = new IdentityActivation<T>();

        // === AUDIO ENCODER ===

        // Initial projection from mel spectrogram to model dimension
        // Using Dense layer to project numMels features to modelDimension
        yield return new DenseLayer<T>(numMels, modelDimension, geluActivation);

        // Second projection for feature extraction
        yield return new DenseLayer<T>(modelDimension, modelDimension, geluActivation);

        // Positional encoding for encoder
        yield return new PositionalEncodingLayer<T>(maxSequenceLength, modelDimension);

        // Encoder dropout
        if (dropoutRate > 0)
        {
            yield return new DropoutLayer<T>(dropoutRate);
        }

        // Encoder transformer layers
        for (int i = 0; i < numEncoderLayers; i++)
        {
            // Self-attention
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: maxSequenceLength,
                embeddingDimension: modelDimension,
                headCount: numHeads,
                activationFunction: identityActivation);

            // Layer normalization
            yield return new LayerNormalizationLayer<T>(modelDimension);

            // Dropout
            if (dropoutRate > 0)
            {
                yield return new DropoutLayer<T>(dropoutRate);
            }

            // Feed-forward network
            yield return new DenseLayer<T>(modelDimension, feedForwardDim, geluActivation);
            yield return new DenseLayer<T>(feedForwardDim, modelDimension, identityActivation);

            // Layer normalization
            yield return new LayerNormalizationLayer<T>(modelDimension);

            // Dropout
            if (dropoutRate > 0)
            {
                yield return new DropoutLayer<T>(dropoutRate);
            }
        }

        // === TEXT DECODER ===

        // Token embedding layer
        yield return new EmbeddingLayer<T>(vocabularySize, modelDimension);

        // Positional encoding for decoder
        yield return new PositionalEncodingLayer<T>(maxSequenceLength, modelDimension);

        // Decoder dropout
        if (dropoutRate > 0)
        {
            yield return new DropoutLayer<T>(dropoutRate);
        }

        // Decoder transformer layers
        for (int i = 0; i < numDecoderLayers; i++)
        {
            // Self-attention layer for decoder
            // NOTE: Causal masking for autoregressive decoding should be applied during
            // the forward pass, not in the layer configuration. The MultiHeadAttentionLayer
            // does not automatically apply causal masking - this must be handled by the
            // model's forward implementation.
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: maxSequenceLength,
                embeddingDimension: modelDimension,
                headCount: numHeads,
                activationFunction: identityActivation);

            // Layer normalization
            yield return new LayerNormalizationLayer<T>(modelDimension);

            // Dropout
            if (dropoutRate > 0)
            {
                yield return new DropoutLayer<T>(dropoutRate);
            }

            // NOTE: This is a placeholder for cross-attention but functions as self-attention
            // in the current flat sequential architecture. True cross-attention would require
            // encoder output to be passed as key/value, which the flat layer list cannot support.
            // For production use with proper cross-attention, use ONNX models or implement
            // a custom forward pass.
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: maxSequenceLength,
                embeddingDimension: modelDimension,
                headCount: numHeads,
                activationFunction: identityActivation);

            // Layer normalization
            yield return new LayerNormalizationLayer<T>(modelDimension);

            // Dropout
            if (dropoutRate > 0)
            {
                yield return new DropoutLayer<T>(dropoutRate);
            }

            // Feed-forward network
            yield return new DenseLayer<T>(modelDimension, feedForwardDim, geluActivation);
            yield return new DenseLayer<T>(feedForwardDim, modelDimension, identityActivation);

            // Layer normalization
            yield return new LayerNormalizationLayer<T>(modelDimension);

            // Dropout
            if (dropoutRate > 0)
            {
                yield return new DropoutLayer<T>(dropoutRate);
            }
        }

        // Final layer normalization
        yield return new LayerNormalizationLayer<T>(modelDimension);

        // Output projection to vocabulary
        yield return new DenseLayer<T>(modelDimension, vocabularySize, identityActivation);
    }

    #region Language Identification Layers

    /// <summary>
    /// Creates default ECAPA-TDNN layers for spoken language identification.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="numMels">Number of mel filterbank channels (default: 80).</param>
    /// <param name="tdnnChannels">Number of TDNN channels (default: 1024).</param>
    /// <param name="embeddingDimension">Embedding dimension (default: 192).</param>
    /// <param name="numLanguages">Number of languages to classify (default: 20).</param>
    /// <param name="dilations">Dilation factors for TDNN layers (default: [1, 2, 3, 4, 1]).</param>
    /// <returns>A collection of layers forming an ECAPA-TDNN language identifier.</returns>
    /// <remarks>
    /// <para>
    /// ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation TDNN)
    /// is a state-of-the-art architecture for speaker and language recognition using:
    /// - SE-Res2Net blocks with channel attention
    /// - Multi-layer feature aggregation (MFA)
    /// - Attentive statistics pooling
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultECAPATDNNLanguageIdentifierLayers(
        NeuralNetworkArchitecture<T> architecture,
        int numMels = 80,
        int tdnnChannels = 1024,
        int embeddingDimension = 192,
        int numLanguages = 20,
        int[]? dilations = null)
    {
        dilations ??= [1, 2, 3, 4, 1];
        IActivationFunction<T> reluActivation = new ReLUActivation<T>();
        IActivationFunction<T> sigmoidActivation = new SigmoidActivation<T>();

        int inputDim = numMels * 3; // MFCC + delta + delta-delta

        // Initial TDNN layer
        yield return new DenseLayer<T>(inputDim, tdnnChannels, reluActivation);
        yield return new BatchNormalizationLayer<T>(tdnnChannels);

        // SE-Res2Net blocks for each dilation
        foreach (int dilation in dilations)
        {
            // 1x1 reduction
            yield return new DenseLayer<T>(tdnnChannels, tdnnChannels / 4, reluActivation);
            yield return new BatchNormalizationLayer<T>(tdnnChannels / 4);

            // Dilated conv (simulated)
            yield return new DenseLayer<T>(tdnnChannels / 4, tdnnChannels / 4, reluActivation);
            yield return new BatchNormalizationLayer<T>(tdnnChannels / 4);

            // 1x1 expansion
            yield return new DenseLayer<T>(tdnnChannels / 4, tdnnChannels, reluActivation);
            yield return new BatchNormalizationLayer<T>(tdnnChannels);

            // Squeeze-Excitation block
            int seReduction = 8;
            yield return new DenseLayer<T>(tdnnChannels, tdnnChannels / seReduction, reluActivation);
            yield return new DenseLayer<T>(tdnnChannels / seReduction, tdnnChannels, sigmoidActivation);
        }

        // Attentive Statistics Pooling projection
        int mfaOutputDim = tdnnChannels * dilations.Length;
        yield return new DenseLayer<T>(mfaOutputDim, embeddingDimension * 2);

        // Final batch normalization
        yield return new BatchNormalizationLayer<T>(embeddingDimension);

        // Classification layer
        yield return new DenseLayer<T>(embeddingDimension, numLanguages);
    }

    /// <summary>
    /// Creates default Wav2Vec2 layers for spoken language identification.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="hiddenSize">Hidden size of transformer (default: 768).</param>
    /// <param name="numLayers">Number of transformer layers (default: 12).</param>
    /// <param name="numAttentionHeads">Number of attention heads (default: 12).</param>
    /// <param name="intermediateSize">Feed-forward intermediate size (default: 3072).</param>
    /// <param name="numLanguages">Number of languages to classify (default: 20).</param>
    /// <param name="dropoutRate">Dropout rate (default: 0.1).</param>
    /// <returns>A collection of layers forming a Wav2Vec2 language identifier.</returns>
    /// <remarks>
    /// <para>
    /// Wav2Vec2-LID uses Meta's self-supervised speech representation model:
    /// - 7-layer CNN feature encoder processing raw waveform
    /// - Transformer encoder for contextual representations
    /// - Classification head for language prediction
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultWav2Vec2LanguageIdentifierLayers(
        NeuralNetworkArchitecture<T> architecture,
        int hiddenSize = 768,
        int numLayers = 12,
        int numAttentionHeads = 12,
        int intermediateSize = 3072,
        int numLanguages = 20,
        double dropoutRate = 0.1)
    {
        IActivationFunction<T> geluActivation = new GELUActivation<T>();
        IActivationFunction<T> tanhActivation = new TanhActivation<T>();

        // Feature encoder: 7 temporal convolution layers
        int[] kernelSizes = [10, 3, 3, 3, 3, 2, 2];
        int[] channels = [512, 512, 512, 512, 512, 512, 512];

        int inputDim = 1; // Raw waveform
        for (int i = 0; i < kernelSizes.Length; i++)
        {
            int outputDim = channels[i];
            yield return new DenseLayer<T>(inputDim * kernelSizes[i], outputDim, geluActivation);
            yield return new LayerNormalizationLayer<T>(outputDim);
            inputDim = outputDim;
        }

        // Feature projection
        yield return new DenseLayer<T>(channels[^1], hiddenSize, geluActivation);
        if (dropoutRate > 0)
        {
            yield return new DropoutLayer<T>(dropoutRate);
        }

        // Transformer encoder layers
        for (int i = 0; i < numLayers; i++)
        {
            // Self-attention (simplified as dense)
            yield return new DenseLayer<T>(hiddenSize, hiddenSize);
            yield return new LayerNormalizationLayer<T>(hiddenSize);

            // Feed-forward
            yield return new DenseLayer<T>(hiddenSize, intermediateSize, geluActivation);
            yield return new DenseLayer<T>(intermediateSize, hiddenSize);
            yield return new LayerNormalizationLayer<T>(hiddenSize);

            if (dropoutRate > 0)
            {
                yield return new DropoutLayer<T>(dropoutRate);
            }
        }

        // Classification head
        yield return new DenseLayer<T>(hiddenSize, hiddenSize, tanhActivation);
        yield return new DenseLayer<T>(hiddenSize, numLanguages);
    }

    /// <summary>
    /// Creates default VoxLingua107 layers for 107-language identification.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="numMels">Number of mel filterbank channels (default: 80).</param>
    /// <param name="tdnnChannels">Number of TDNN channels (default: 1024).</param>
    /// <param name="embeddingDimension">Embedding dimension (default: 256).</param>
    /// <param name="dilations">Dilation factors for TDNN layers (default: [1, 2, 3, 4, 1]).</param>
    /// <returns>A collection of layers forming a VoxLingua107 language identifier.</returns>
    /// <remarks>
    /// <para>
    /// VoxLingua107 uses ECAPA-TDNN architecture trained on 107 languages from
    /// the VoxLingua107 dataset (YouTube speech samples).
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultVoxLingua107Layers(
        NeuralNetworkArchitecture<T> architecture,
        int numMels = 80,
        int tdnnChannels = 1024,
        int embeddingDimension = 256,
        int[]? dilations = null)
    {
        // VoxLingua107 uses ECAPA-TDNN with 107 output classes
        return CreateDefaultECAPATDNNLanguageIdentifierLayers(
            architecture,
            numMels: numMels,
            tdnnChannels: tdnnChannels,
            embeddingDimension: embeddingDimension,
            numLanguages: 107,
            dilations: dilations);
    }

    #endregion

    #region Audio Generation Layers

    /// <summary>
    /// Creates default AudioGen layers for text-to-audio generation.
    /// </summary>
    /// <param name="textHiddenDim">Text encoder hidden dimension (default: 768 for T5-base).</param>
    /// <param name="lmHiddenDim">Language model hidden dimension (default: 1536).</param>
    /// <param name="numLmLayers">Number of language model transformer layers (default: 24).</param>
    /// <param name="numHeads">Number of attention heads (default: 16).</param>
    /// <param name="numCodebooks">Number of EnCodec codebooks (default: 4).</param>
    /// <param name="codebookSize">Size of each codebook vocabulary (default: 1024).</param>
    /// <param name="maxTextLength">Maximum text sequence length (default: 256).</param>
    /// <param name="maxAudioTokens">Maximum audio tokens (~50 tokens/sec) (default: 1500 for 30s).</param>
    /// <param name="dropoutRate">Dropout rate (default: 0.1).</param>
    /// <returns>A collection of layers forming an AudioGen model.</returns>
    /// <remarks>
    /// <para>
    /// AudioGen is a text-to-audio generation model that uses a transformer language model
    /// operating over EnCodec audio codes. Unlike MusicGen, it focuses on general audio
    /// and environmental sounds rather than music.
    /// </para>
    /// <list type="bullet">
    /// <item><description>T5-based text encoder for conditioning</description></item>
    /// <item><description>Transformer decoder generating audio codes autoregressively</description></item>
    /// <item><description>EnCodec neural audio codec for audio reconstruction</description></item>
    /// </list>
    /// <para>
    /// Reference: "AudioGen: Textually Guided Audio Generation" by Kreuk et al., 2022
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultAudioGenLayers(
        int textHiddenDim = 768,
        int lmHiddenDim = 1536,
        int numLmLayers = 24,
        int numHeads = 16,
        int numCodebooks = 4,
        int codebookSize = 1024,
        int maxTextLength = 256,
        int maxAudioTokens = 1500,
        double dropoutRate = 0.1)
    {
        IActivationFunction<T> geluActivation = new GELUActivation<T>();
        IActivationFunction<T> identityActivation = new IdentityActivation<T>();

        // === TEXT ENCODER (T5-style) ===

        // Token embedding: T5 vocabulary to hidden dimension
        yield return new EmbeddingLayer<T>(32128, textHiddenDim);

        // Positional encoding for text
        yield return new PositionalEncodingLayer<T>(maxTextLength, textHiddenDim);

        // Encoder dropout
        if (dropoutRate > 0)
        {
            yield return new DropoutLayer<T>(dropoutRate);
        }

        // Text encoder transformer layers (6 layers, T5-base style)
        for (int i = 0; i < 6; i++)
        {
            // Self-attention
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: maxTextLength,
                embeddingDimension: textHiddenDim,
                headCount: numHeads);

            // Layer norm
            yield return new LayerNormalizationLayer<T>(textHiddenDim);

            // Feedforward
            yield return new DenseLayer<T>(textHiddenDim, textHiddenDim * 4, geluActivation);
            yield return new DenseLayer<T>(textHiddenDim * 4, textHiddenDim, identityActivation);

            // Layer norm
            yield return new LayerNormalizationLayer<T>(textHiddenDim);

            if (dropoutRate > 0)
            {
                yield return new DropoutLayer<T>(dropoutRate);
            }
        }

        // Project text to language model dimension
        yield return new DenseLayer<T>(textHiddenDim, lmHiddenDim, identityActivation);

        // === AUDIO CODE EMBEDDING ===

        // Embedding for audio codes from all codebooks
        yield return new EmbeddingLayer<T>(codebookSize * numCodebooks, lmHiddenDim);

        // Positional encoding for audio sequence
        yield return new PositionalEncodingLayer<T>(maxAudioTokens, lmHiddenDim);

        if (dropoutRate > 0)
        {
            yield return new DropoutLayer<T>(dropoutRate);
        }

        // === LANGUAGE MODEL DECODER ===

        // Transformer decoder layers
        for (int i = 0; i < numLmLayers; i++)
        {
            yield return new TransformerDecoderLayer<T>(
                embeddingSize: lmHiddenDim,
                numHeads: numHeads,
                feedForwardDim: lmHiddenDim * 4,
                sequenceLength: maxAudioTokens,
                ffnActivation: geluActivation);

            if (dropoutRate > 0 && i < numLmLayers - 1)
            {
                yield return new DropoutLayer<T>(dropoutRate);
            }
        }

        // Final layer norm
        yield return new LayerNormalizationLayer<T>(lmHiddenDim);

        // === OUTPUT PROJECTION ===

        // Project to codebook logits
        yield return new DenseLayer<T>(lmHiddenDim, codebookSize * numCodebooks, identityActivation);
    }

    /// <summary>
    /// Creates default MusicGen layers for text-to-music generation.
    /// </summary>
    /// <param name="textHiddenDim">Text encoder hidden dimension (default: 768 for T5-base).</param>
    /// <param name="lmHiddenDim">Language model hidden dimension (default: 1536).</param>
    /// <param name="numLmLayers">Number of language model transformer layers (default: 24).</param>
    /// <param name="numHeads">Number of attention heads (default: 16).</param>
    /// <param name="numCodebooks">Number of EnCodec codebooks (default: 4).</param>
    /// <param name="codebookSize">Size of each codebook vocabulary (default: 2048).</param>
    /// <param name="maxTextLength">Maximum text sequence length (default: 256).</param>
    /// <param name="maxAudioTokens">Maximum audio tokens (~50 tokens/sec) (default: 1500 for 30s).</param>
    /// <param name="dropoutRate">Dropout rate (default: 0.1).</param>
    /// <returns>A collection of layers forming a MusicGen model.</returns>
    /// <remarks>
    /// <para>
    /// MusicGen is Meta's text-to-music generation model that uses a single-stage
    /// transformer language model operating over EnCodec audio codes. Key features:
    /// </para>
    /// <list type="bullet">
    /// <item><description>Delay pattern for codebook interleaving (reduces sequence length)</description></item>
    /// <item><description>T5-based text encoder for conditioning</description></item>
    /// <item><description>Transformer decoder generating audio codes autoregressively</description></item>
    /// <item><description>EnCodec neural audio codec for high-quality audio reconstruction</description></item>
    /// </list>
    /// <para>
    /// Reference: "Simple and Controllable Music Generation" by Copet et al., 2023
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultMusicGenLayers(
        int textHiddenDim = 768,
        int lmHiddenDim = 1536,
        int numLmLayers = 24,
        int numHeads = 16,
        int numCodebooks = 4,
        int codebookSize = 2048,
        int maxTextLength = 256,
        int maxAudioTokens = 1500,
        double dropoutRate = 0.1)
    {
        IActivationFunction<T> geluActivation = new GELUActivation<T>();
        IActivationFunction<T> identityActivation = new IdentityActivation<T>();

        // === TEXT ENCODER (T5-style) ===

        // Token embedding: T5 vocabulary to hidden dimension
        yield return new EmbeddingLayer<T>(32128, textHiddenDim);

        // Positional encoding for text
        yield return new PositionalEncodingLayer<T>(maxTextLength, textHiddenDim);

        // Encoder dropout
        if (dropoutRate > 0)
        {
            yield return new DropoutLayer<T>(dropoutRate);
        }

        // Text encoder transformer layers (6 layers, T5-base style)
        for (int i = 0; i < 6; i++)
        {
            // Self-attention
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: maxTextLength,
                embeddingDimension: textHiddenDim,
                headCount: 12,
                activationFunction: identityActivation);

            yield return new LayerNormalizationLayer<T>(textHiddenDim);

            if (dropoutRate > 0)
            {
                yield return new DropoutLayer<T>(dropoutRate);
            }

            // Feed-forward network
            yield return new DenseLayer<T>(textHiddenDim, textHiddenDim * 4, geluActivation);
            yield return new DenseLayer<T>(textHiddenDim * 4, textHiddenDim, identityActivation);
            yield return new LayerNormalizationLayer<T>(textHiddenDim);

            if (dropoutRate > 0)
            {
                yield return new DropoutLayer<T>(dropoutRate);
            }
        }

        // Project text encoder output to LM dimension
        yield return new DenseLayer<T>(textHiddenDim, lmHiddenDim, identityActivation);

        // === AUDIO CODE EMBEDDING ===

        // Combined codebook embedding (all codebooks share embedding space)
        yield return new EmbeddingLayer<T>(codebookSize * numCodebooks + 1, lmHiddenDim); // +1 for start token

        // Positional encoding for audio sequence
        yield return new PositionalEncodingLayer<T>(maxAudioTokens, lmHiddenDim);

        if (dropoutRate > 0)
        {
            yield return new DropoutLayer<T>(dropoutRate);
        }

        // === TRANSFORMER LANGUAGE MODEL ===

        // Decoder layers with cross-attention to text encoder
        for (int i = 0; i < numLmLayers; i++)
        {
            // Self-attention (causal/masked for autoregressive generation)
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: maxAudioTokens,
                embeddingDimension: lmHiddenDim,
                headCount: numHeads,
                activationFunction: identityActivation);

            yield return new LayerNormalizationLayer<T>(lmHiddenDim);

            if (dropoutRate > 0)
            {
                yield return new DropoutLayer<T>(dropoutRate);
            }

            // Cross-attention to text encoder output
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: maxAudioTokens,
                embeddingDimension: lmHiddenDim,
                headCount: numHeads,
                activationFunction: identityActivation);

            yield return new LayerNormalizationLayer<T>(lmHiddenDim);

            if (dropoutRate > 0)
            {
                yield return new DropoutLayer<T>(dropoutRate);
            }

            // Feed-forward network
            yield return new DenseLayer<T>(lmHiddenDim, lmHiddenDim * 4, geluActivation);
            yield return new DenseLayer<T>(lmHiddenDim * 4, lmHiddenDim, identityActivation);
            yield return new LayerNormalizationLayer<T>(lmHiddenDim);

            if (dropoutRate > 0)
            {
                yield return new DropoutLayer<T>(dropoutRate);
            }
        }

        // Final layer normalization
        yield return new LayerNormalizationLayer<T>(lmHiddenDim);

        // === OUTPUT PROJECTION ===

        // Project to codebook logits (one set per codebook for delay pattern)
        for (int cb = 0; cb < numCodebooks; cb++)
        {
            yield return new DenseLayer<T>(lmHiddenDim, codebookSize, identityActivation);
        }
    }

    /// <summary>
    /// Creates default AudioLDM layers for text-to-audio generation using latent diffusion.
    /// </summary>
    /// <param name="textHiddenDim">Text encoder hidden dimension (default: 768 for CLAP).</param>
    /// <param name="latentDim">Latent space dimension (default: 8).</param>
    /// <param name="unetChannels">U-Net base channels (default: 256).</param>
    /// <param name="numResBlocks">Number of residual blocks per level (default: 2).</param>
    /// <param name="attentionResolutions">Resolutions at which to apply attention (default: [4, 2, 1]).</param>
    /// <param name="numHeads">Number of attention heads (default: 8).</param>
    /// <param name="numMels">Number of mel spectrogram channels (default: 64).</param>
    /// <param name="maxTextLength">Maximum text sequence length (default: 77).</param>
    /// <param name="dropoutRate">Dropout rate (default: 0.1).</param>
    /// <returns>A collection of layers forming an AudioLDM model.</returns>
    /// <remarks>
    /// <para>
    /// AudioLDM uses latent diffusion for text-to-audio generation:
    /// </para>
    /// <list type="bullet">
    /// <item><description>CLAP text encoder for conditioning</description></item>
    /// <item><description>VAE to encode/decode mel spectrograms to latent space</description></item>
    /// <item><description>U-Net for denoising in latent space</description></item>
    /// <item><description>HiFi-GAN vocoder for waveform generation</description></item>
    /// </list>
    /// <para>
    /// Reference: "AudioLDM: Text-to-Audio Generation with Latent Diffusion Models" by Liu et al., 2023
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultAudioLDMLayers(
        int textHiddenDim = 768,
        int latentDim = 8,
        int unetChannels = 256,
        int numResBlocks = 2,
        int[]? attentionResolutions = null,
        int numHeads = 8,
        int numMels = 64,
        int maxTextLength = 77,
        double dropoutRate = 0.1)
    {
        attentionResolutions ??= [4, 2, 1];
        IActivationFunction<T> siluActivation = new SwishActivation<T>();
        IActivationFunction<T> identityActivation = new IdentityActivation<T>();

        // === TEXT ENCODER (CLAP-style) ===

        // Token embedding
        yield return new EmbeddingLayer<T>(49408, textHiddenDim); // CLIP vocabulary

        // Positional encoding
        yield return new PositionalEncodingLayer<T>(maxTextLength, textHiddenDim);

        // Transformer encoder layers
        for (int i = 0; i < 12; i++)
        {
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: maxTextLength,
                embeddingDimension: textHiddenDim,
                headCount: 12,
                activationFunction: identityActivation);

            yield return new LayerNormalizationLayer<T>(textHiddenDim);

            yield return new DenseLayer<T>(textHiddenDim, textHiddenDim * 4, siluActivation);
            yield return new DenseLayer<T>(textHiddenDim * 4, textHiddenDim, identityActivation);
            yield return new LayerNormalizationLayer<T>(textHiddenDim);
        }

        // === VAE ENCODER ===

        // Initial convolution from mel spectrogram
        yield return new DenseLayer<T>(numMels, unetChannels, siluActivation);

        // Down-sampling path
        int[] channelMults = [1, 2, 4, 4];
        int currentChannels = unetChannels;

        foreach (int mult in channelMults)
        {
            int outChannels = unetChannels * mult;

            for (int r = 0; r < numResBlocks; r++)
            {
                yield return new DenseLayer<T>(currentChannels, outChannels, siluActivation);
                yield return new LayerNormalizationLayer<T>(outChannels);
                currentChannels = outChannels;
            }

            // Downsample (except last level)
            if (mult != channelMults[^1])
            {
                yield return new DenseLayer<T>(currentChannels, currentChannels, siluActivation);
            }
        }

        // Latent projection
        yield return new DenseLayer<T>(currentChannels, latentDim * 2, identityActivation); // mean + log_var

        // === U-NET DENOISER ===

        // Time embedding
        yield return new DenseLayer<T>(latentDim, unetChannels * 4, siluActivation);
        yield return new DenseLayer<T>(unetChannels * 4, unetChannels * 4, siluActivation);

        // U-Net encoder path
        currentChannels = latentDim;
        yield return new DenseLayer<T>(currentChannels, unetChannels, siluActivation);
        currentChannels = unetChannels;

        foreach (int mult in channelMults)
        {
            int outChannels = unetChannels * mult;

            for (int r = 0; r < numResBlocks; r++)
            {
                yield return new DenseLayer<T>(currentChannels, outChannels, siluActivation);
                yield return new LayerNormalizationLayer<T>(outChannels);

                // Cross-attention at specified resolutions
                if (attentionResolutions.Contains(mult))
                {
                    yield return new MultiHeadAttentionLayer<T>(
                        sequenceLength: maxTextLength,
                        embeddingDimension: outChannels,
                        headCount: numHeads,
                        activationFunction: identityActivation);
                    yield return new LayerNormalizationLayer<T>(outChannels);
                }

                currentChannels = outChannels;
            }
        }

        // Middle block
        yield return new DenseLayer<T>(currentChannels, currentChannels, siluActivation);
        yield return new MultiHeadAttentionLayer<T>(
            sequenceLength: maxTextLength,
            embeddingDimension: currentChannels,
            headCount: numHeads,
            activationFunction: identityActivation);
        yield return new DenseLayer<T>(currentChannels, currentChannels, siluActivation);

        // U-Net decoder path (symmetric to encoder)
        for (int i = channelMults.Length - 1; i >= 0; i--)
        {
            int mult = channelMults[i];
            int outChannels = unetChannels * mult;

            for (int r = 0; r < numResBlocks + 1; r++)
            {
                yield return new DenseLayer<T>(currentChannels, outChannels, siluActivation);
                yield return new LayerNormalizationLayer<T>(outChannels);

                if (attentionResolutions.Contains(mult))
                {
                    yield return new MultiHeadAttentionLayer<T>(
                        sequenceLength: maxTextLength,
                        embeddingDimension: outChannels,
                        headCount: numHeads,
                        activationFunction: identityActivation);
                    yield return new LayerNormalizationLayer<T>(outChannels);
                }

                currentChannels = outChannels;
            }

            // Upsample (except first level)
            if (i > 0)
            {
                yield return new DenseLayer<T>(currentChannels, currentChannels, siluActivation);
            }
        }

        // Output projection to latent
        yield return new LayerNormalizationLayer<T>(currentChannels);
        yield return new DenseLayer<T>(currentChannels, latentDim, identityActivation);

        // === VAE DECODER ===

        // Latent to channels
        yield return new DenseLayer<T>(latentDim, unetChannels * channelMults[^1], siluActivation);
        currentChannels = unetChannels * channelMults[^1];

        // Up-sampling path
        for (int i = channelMults.Length - 1; i >= 0; i--)
        {
            int mult = channelMults[i];
            int outChannels = unetChannels * mult;

            for (int r = 0; r < numResBlocks + 1; r++)
            {
                yield return new DenseLayer<T>(currentChannels, outChannels, siluActivation);
                yield return new LayerNormalizationLayer<T>(outChannels);
                currentChannels = outChannels;
            }

            // Upsample (except first level)
            if (i > 0)
            {
                yield return new DenseLayer<T>(currentChannels, currentChannels, siluActivation);
            }
        }

        // Output projection to mel spectrogram
        yield return new LayerNormalizationLayer<T>(currentChannels);
        yield return new DenseLayer<T>(currentChannels, numMels, identityActivation);
    }

    /// <summary>
    /// Creates default Stable Audio layers for text-to-audio generation.
    /// </summary>
    /// <param name="textHiddenDim">Text encoder hidden dimension (default: 768).</param>
    /// <param name="latentDim">Latent space dimension (default: 64).</param>
    /// <param name="ditHiddenDim">DiT hidden dimension (default: 1024).</param>
    /// <param name="numDitBlocks">Number of DiT transformer blocks (default: 24).</param>
    /// <param name="numHeads">Number of attention heads (default: 16).</param>
    /// <param name="maxTextLength">Maximum text sequence length (default: 512).</param>
    /// <param name="maxAudioLength">Maximum audio latent sequence length (default: 2048).</param>
    /// <param name="dropoutRate">Dropout rate (default: 0.1).</param>
    /// <returns>A collection of layers forming a Stable Audio model.</returns>
    /// <remarks>
    /// <para>
    /// Stable Audio by Stability AI uses a Diffusion Transformer (DiT) architecture:
    /// </para>
    /// <list type="bullet">
    /// <item><description>T5-based text encoder for conditioning</description></item>
    /// <item><description>Variational autoencoder for audio latent compression</description></item>
    /// <item><description>DiT (Diffusion Transformer) for denoising in latent space</description></item>
    /// <item><description>Supports variable-length audio generation with timing conditioning</description></item>
    /// </list>
    /// <para>
    /// Reference: "Stable Audio: Fast Timing-Conditioned Latent Audio Diffusion" by Evans et al., 2024
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultStableAudioLayers(
        int textHiddenDim = 768,
        int latentDim = 64,
        int ditHiddenDim = 1024,
        int numDitBlocks = 24,
        int numHeads = 16,
        int maxTextLength = 512,
        int maxAudioLength = 2048,
        double dropoutRate = 0.1)
    {
        IActivationFunction<T> geluActivation = new GELUActivation<T>();
        IActivationFunction<T> siluActivation = new SwishActivation<T>();
        IActivationFunction<T> identityActivation = new IdentityActivation<T>();

        // === T5 TEXT ENCODER ===

        // Token embedding
        yield return new EmbeddingLayer<T>(32128, textHiddenDim);

        // Positional encoding
        yield return new PositionalEncodingLayer<T>(maxTextLength, textHiddenDim);

        if (dropoutRate > 0)
        {
            yield return new DropoutLayer<T>(dropoutRate);
        }

        // T5 encoder layers
        for (int i = 0; i < 12; i++)
        {
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: maxTextLength,
                embeddingDimension: textHiddenDim,
                headCount: 12,
                activationFunction: identityActivation);

            yield return new LayerNormalizationLayer<T>(textHiddenDim);

            yield return new DenseLayer<T>(textHiddenDim, textHiddenDim * 4, geluActivation);
            yield return new DenseLayer<T>(textHiddenDim * 4, textHiddenDim, identityActivation);
            yield return new LayerNormalizationLayer<T>(textHiddenDim);

            if (dropoutRate > 0)
            {
                yield return new DropoutLayer<T>(dropoutRate);
            }
        }

        // Project to DiT dimension
        yield return new DenseLayer<T>(textHiddenDim, ditHiddenDim, identityActivation);

        // === TIMING CONDITIONING ===

        // Start/end time embedding (seconds conditioning)
        yield return new DenseLayer<T>(2, ditHiddenDim, siluActivation);
        yield return new DenseLayer<T>(ditHiddenDim, ditHiddenDim, siluActivation);

        // === VAE ENCODER ===

        // Audio waveform to latent space
        yield return new DenseLayer<T>(1, 128, siluActivation);
        yield return new DenseLayer<T>(128, 256, siluActivation);
        yield return new DenseLayer<T>(256, 512, siluActivation);
        yield return new DenseLayer<T>(512, latentDim * 2, identityActivation); // mean + log_var

        // === DiT (DIFFUSION TRANSFORMER) ===

        // Latent projection
        yield return new DenseLayer<T>(latentDim, ditHiddenDim, identityActivation);

        // Positional encoding for audio latents
        yield return new PositionalEncodingLayer<T>(maxAudioLength, ditHiddenDim);

        // Timestep embedding (sinusoidal + MLP)
        yield return new DenseLayer<T>(ditHiddenDim, ditHiddenDim * 4, siluActivation);
        yield return new DenseLayer<T>(ditHiddenDim * 4, ditHiddenDim, identityActivation);

        // DiT blocks (transformer with AdaLN conditioning)
        for (int i = 0; i < numDitBlocks; i++)
        {
            // Self-attention
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: maxAudioLength,
                embeddingDimension: ditHiddenDim,
                headCount: numHeads,
                activationFunction: identityActivation);

            yield return new LayerNormalizationLayer<T>(ditHiddenDim);

            if (dropoutRate > 0)
            {
                yield return new DropoutLayer<T>(dropoutRate);
            }

            // Cross-attention to text encoder output
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: maxAudioLength,
                embeddingDimension: ditHiddenDim,
                headCount: numHeads,
                activationFunction: identityActivation);

            yield return new LayerNormalizationLayer<T>(ditHiddenDim);

            if (dropoutRate > 0)
            {
                yield return new DropoutLayer<T>(dropoutRate);
            }

            // Feed-forward network with GELU
            yield return new DenseLayer<T>(ditHiddenDim, ditHiddenDim * 4, geluActivation);
            yield return new DenseLayer<T>(ditHiddenDim * 4, ditHiddenDim, identityActivation);
            yield return new LayerNormalizationLayer<T>(ditHiddenDim);

            if (dropoutRate > 0)
            {
                yield return new DropoutLayer<T>(dropoutRate);
            }
        }

        // Final layer normalization
        yield return new LayerNormalizationLayer<T>(ditHiddenDim);

        // Output projection to latent space
        yield return new DenseLayer<T>(ditHiddenDim, latentDim, identityActivation);

        // === VAE DECODER ===

        // Latent to waveform
        yield return new DenseLayer<T>(latentDim, 512, siluActivation);
        yield return new DenseLayer<T>(512, 256, siluActivation);
        yield return new DenseLayer<T>(256, 128, siluActivation);
        yield return new DenseLayer<T>(128, 1, identityActivation); // mono audio output
    }

    /// <summary>
    /// Creates default Whisper layers for automatic speech recognition.
    /// </summary>
    /// <param name="modelDim">Model hidden dimension (default: 512 for Base).</param>
    /// <param name="numEncoderLayers">Number of encoder transformer layers (default: 6 for Base).</param>
    /// <param name="numDecoderLayers">Number of decoder transformer layers (default: 6 for Base).</param>
    /// <param name="numHeads">Number of attention heads (default: 8 for Base).</param>
    /// <param name="ffDim">Feed-forward hidden dimension (default: 2048 for Base).</param>
    /// <param name="numMels">Number of mel spectrogram bins (default: 80).</param>
    /// <param name="maxFrames">Maximum mel spectrogram frames (default: 3000 for 30s audio).</param>
    /// <param name="maxTokens">Maximum output token sequence length (default: 448).</param>
    /// <param name="vocabSize">Whisper vocabulary size (default: 51865).</param>
    /// <param name="dropoutRate">Dropout rate (default: 0.0 for inference-optimized).</param>
    /// <returns>A collection of layers forming a Whisper encoder-decoder architecture.</returns>
    /// <remarks>
    /// <para>
    /// Whisper is OpenAI's state-of-the-art automatic speech recognition model with:
    /// </para>
    /// <list type="bullet">
    /// <item><description>Mel spectrogram audio preprocessing (80 bins, 16kHz)</description></item>
    /// <item><description>Convolutional stem for initial audio feature extraction</description></item>
    /// <item><description>Transformer encoder for audio representation learning</description></item>
    /// <item><description>Transformer decoder with cross-attention for text generation</description></item>
    /// <item><description>Support for 99+ languages and translation to English</description></item>
    /// </list>
    /// <para>
    /// Reference: "Robust Speech Recognition via Large-Scale Weak Supervision" by Radford et al., 2022
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultWhisperLayers(
        int modelDim = 512,
        int numEncoderLayers = 6,
        int numDecoderLayers = 6,
        int numHeads = 8,
        int ffDim = 2048,
        int numMels = 80,
        int maxFrames = 3000,
        int maxTokens = 448,
        int vocabSize = 51865,
        double dropoutRate = 0.0)
    {
        IActivationFunction<T> geluActivation = new GELUActivation<T>();
        IActivationFunction<T> identityActivation = new IdentityActivation<T>();

        // === AUDIO ENCODER ===

        // Initial projection from mel spectrogram to model dimension
        // (Simulating convolutional stem with dense layers for framework compatibility)
        yield return new DenseLayer<T>(numMels, modelDim, geluActivation);
        yield return new DenseLayer<T>(modelDim, modelDim, geluActivation);

        // Positional encoding for encoder
        yield return new PositionalEncodingLayer<T>(maxFrames, modelDim);

        if (dropoutRate > 0)
        {
            yield return new DropoutLayer<T>(dropoutRate);
        }

        // Encoder transformer layers
        for (int i = 0; i < numEncoderLayers; i++)
        {
            // Self-attention
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: maxFrames,
                embeddingDimension: modelDim,
                headCount: numHeads);

            // Layer normalization (pre-LN architecture)
            yield return new LayerNormalizationLayer<T>(modelDim);

            // Feed-forward network
            yield return new DenseLayer<T>(modelDim, ffDim, geluActivation);
            yield return new DenseLayer<T>(ffDim, modelDim, identityActivation);

            // Layer normalization
            yield return new LayerNormalizationLayer<T>(modelDim);

            if (dropoutRate > 0)
            {
                yield return new DropoutLayer<T>(dropoutRate);
            }
        }

        // Final encoder layer normalization
        yield return new LayerNormalizationLayer<T>(modelDim);

        // === TEXT DECODER ===

        // Token embedding (Whisper vocabulary)
        yield return new EmbeddingLayer<T>(vocabSize, modelDim);

        // Positional encoding for decoder
        yield return new PositionalEncodingLayer<T>(maxTokens, modelDim);

        if (dropoutRate > 0)
        {
            yield return new DropoutLayer<T>(dropoutRate);
        }

        // Decoder transformer layers with cross-attention
        for (int i = 0; i < numDecoderLayers; i++)
        {
            yield return new TransformerDecoderLayer<T>(
                embeddingSize: modelDim,
                numHeads: numHeads,
                feedForwardDim: ffDim,
                sequenceLength: maxTokens,
                ffnActivation: geluActivation);

            if (dropoutRate > 0 && i < numDecoderLayers - 1)
            {
                yield return new DropoutLayer<T>(dropoutRate);
            }
        }

        // Final decoder layer normalization
        yield return new LayerNormalizationLayer<T>(modelDim);

        // Output projection to vocabulary logits
        yield return new DenseLayer<T>(modelDim, vocabSize, identityActivation);
    }

    /// <summary>
    /// Creates default TTS (Text-to-Speech) layers for speech synthesis.
    /// </summary>
    /// <param name="textHiddenDim">Text encoder hidden dimension (default: 256).</param>
    /// <param name="audioHiddenDim">Audio decoder hidden dimension (default: 512).</param>
    /// <param name="numEncoderLayers">Number of encoder transformer layers (default: 6).</param>
    /// <param name="numDecoderLayers">Number of decoder transformer layers (default: 6).</param>
    /// <param name="numHeads">Number of attention heads (default: 8).</param>
    /// <param name="numMels">Number of mel spectrogram bins (default: 80).</param>
    /// <param name="maxTextLength">Maximum input text length (default: 512).</param>
    /// <param name="maxMelFrames">Maximum mel spectrogram frames (default: 1000).</param>
    /// <param name="vocabSize">Phoneme/character vocabulary size (default: 148).</param>
    /// <param name="dropoutRate">Dropout rate (default: 0.1).</param>
    /// <returns>A collection of layers forming a TTS encoder-decoder architecture.</returns>
    /// <remarks>
    /// <para>
    /// TTS architecture with:
    /// </para>
    /// <list type="bullet">
    /// <item><description>Character/phoneme embedding with positional encoding</description></item>
    /// <item><description>Transformer encoder for text representation</description></item>
    /// <item><description>Transformer decoder with cross-attention for mel generation</description></item>
    /// <item><description>Post-net convolutional refinement (simulated with dense layers)</description></item>
    /// </list>
    /// <para>
    /// Reference: "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions" (Tacotron 2)
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultTtsLayers(
        int textHiddenDim = 256,
        int audioHiddenDim = 512,
        int numEncoderLayers = 6,
        int numDecoderLayers = 6,
        int numHeads = 8,
        int numMels = 80,
        int maxTextLength = 512,
        int maxMelFrames = 1000,
        int vocabSize = 148,
        double dropoutRate = 0.1)
    {
        IActivationFunction<T> reluActivation = new ReLUActivation<T>();
        IActivationFunction<T> tanhActivation = new TanhActivation<T>();
        IActivationFunction<T> identityActivation = new IdentityActivation<T>();

        // === TEXT ENCODER ===

        // Character/phoneme embedding
        yield return new EmbeddingLayer<T>(vocabSize, textHiddenDim);

        // Positional encoding for text
        yield return new PositionalEncodingLayer<T>(maxTextLength, textHiddenDim);

        if (dropoutRate > 0)
        {
            yield return new DropoutLayer<T>(dropoutRate);
        }

        // Text encoder transformer layers
        for (int i = 0; i < numEncoderLayers; i++)
        {
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: maxTextLength,
                embeddingDimension: textHiddenDim,
                headCount: numHeads);

            yield return new LayerNormalizationLayer<T>(textHiddenDim);

            yield return new DenseLayer<T>(textHiddenDim, textHiddenDim * 4, reluActivation);
            yield return new DenseLayer<T>(textHiddenDim * 4, textHiddenDim, identityActivation);

            yield return new LayerNormalizationLayer<T>(textHiddenDim);

            if (dropoutRate > 0)
            {
                yield return new DropoutLayer<T>(dropoutRate);
            }
        }

        // Project to decoder dimension
        yield return new DenseLayer<T>(textHiddenDim, audioHiddenDim, identityActivation);

        // === MEL DECODER ===

        // Pre-net for mel input (autoregressive conditioning)
        yield return new DenseLayer<T>(numMels, audioHiddenDim, reluActivation);
        yield return new DenseLayer<T>(audioHiddenDim, audioHiddenDim, reluActivation);

        if (dropoutRate > 0)
        {
            yield return new DropoutLayer<T>(dropoutRate);
        }

        // Decoder transformer layers
        for (int i = 0; i < numDecoderLayers; i++)
        {
            yield return new TransformerDecoderLayer<T>(
                embeddingSize: audioHiddenDim,
                numHeads: numHeads,
                feedForwardDim: audioHiddenDim * 4,
                sequenceLength: maxMelFrames,
                ffnActivation: reluActivation);

            if (dropoutRate > 0 && i < numDecoderLayers - 1)
            {
                yield return new DropoutLayer<T>(dropoutRate);
            }
        }

        // Mel projection
        yield return new DenseLayer<T>(audioHiddenDim, numMels, identityActivation);

        // === POST-NET (5 convolutional layers simulated with dense) ===
        yield return new DenseLayer<T>(numMels, 512, tanhActivation);
        yield return new DenseLayer<T>(512, 512, tanhActivation);
        yield return new DenseLayer<T>(512, 512, tanhActivation);
        yield return new DenseLayer<T>(512, 512, tanhActivation);
        yield return new DenseLayer<T>(512, numMels, identityActivation);

        // Stop token prediction
        yield return new DenseLayer<T>(audioHiddenDim, 1, (IActivationFunction<T>)new SigmoidActivation<T>());
    }

    /// <summary>
    /// Creates default speaker embedding layers for speaker verification and identification.
    /// </summary>
    /// <param name="numMels">Number of mel spectrogram bins (default: 80).</param>
    /// <param name="hiddenDim">Hidden layer dimension (default: 512).</param>
    /// <param name="embeddingDim">Output embedding dimension (default: 256).</param>
    /// <param name="numLayers">Number of LSTM-like layers (default: 3).</param>
    /// <param name="maxFrames">Maximum input frames (default: 500).</param>
    /// <param name="dropoutRate">Dropout rate (default: 0.1).</param>
    /// <returns>A collection of layers for speaker embedding extraction.</returns>
    /// <remarks>
    /// <para>
    /// ECAPA-TDNN inspired architecture for speaker embedding with:
    /// </para>
    /// <list type="bullet">
    /// <item><description>Frame-level feature extraction with attention</description></item>
    /// <item><description>Temporal context aggregation</description></item>
    /// <item><description>Attentive statistics pooling</description></item>
    /// <item><description>Speaker embedding projection</description></item>
    /// </list>
    /// <para>
    /// Reference: "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN"
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultSpeakerEmbeddingLayers(
        int numMels = 80,
        int hiddenDim = 512,
        int embeddingDim = 256,
        int numLayers = 3,
        int maxFrames = 500,
        double dropoutRate = 0.1)
    {
        IActivationFunction<T> reluActivation = new ReLUActivation<T>();
        IActivationFunction<T> identityActivation = new IdentityActivation<T>();

        // Initial feature projection
        yield return new DenseLayer<T>(numMels, hiddenDim, reluActivation);
        yield return new LayerNormalizationLayer<T>(hiddenDim);

        if (dropoutRate > 0)
        {
            yield return new DropoutLayer<T>(dropoutRate);
        }

        // Frame-level processing with attention (simulating TDNN with attention)
        for (int i = 0; i < numLayers; i++)
        {
            // Self-attention for temporal modeling
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: maxFrames,
                embeddingDimension: hiddenDim,
                headCount: 8);

            yield return new LayerNormalizationLayer<T>(hiddenDim);

            // Feed-forward with residual-like structure
            yield return new DenseLayer<T>(hiddenDim, hiddenDim * 2, reluActivation);
            yield return new DenseLayer<T>(hiddenDim * 2, hiddenDim, identityActivation);

            yield return new LayerNormalizationLayer<T>(hiddenDim);

            if (dropoutRate > 0)
            {
                yield return new DropoutLayer<T>(dropoutRate);
            }
        }

        // Attentive statistics pooling (simplified)
        yield return new DenseLayer<T>(hiddenDim, hiddenDim, (IActivationFunction<T>)new TanhActivation<T>());
        yield return new DenseLayer<T>(hiddenDim, hiddenDim, identityActivation);

        // Final embedding projection
        yield return new DenseLayer<T>(hiddenDim, embeddingDim, identityActivation);

        // L2 normalization is handled in the model code
    }

    /// <summary>
    /// Creates default genre classification layers.
    /// </summary>
    /// <param name="numMels">Number of mel spectrogram bins (default: 128).</param>
    /// <param name="hiddenDim">Hidden layer dimension (default: 256).</param>
    /// <param name="numClasses">Number of genre classes (default: 10).</param>
    /// <param name="maxFrames">Maximum input frames (default: 1000).</param>
    /// <param name="numAttentionLayers">Number of attention layers (default: 4).</param>
    /// <param name="dropoutRate">Dropout rate (default: 0.3).</param>
    /// <returns>A collection of layers for genre classification.</returns>
    /// <remarks>
    /// <para>
    /// Audio classification architecture with:
    /// </para>
    /// <list type="bullet">
    /// <item><description>Mel spectrogram feature extraction</description></item>
    /// <item><description>Transformer encoder for temporal modeling</description></item>
    /// <item><description>Global average pooling</description></item>
    /// <item><description>Classification head with softmax output</description></item>
    /// </list>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultGenreClassifierLayers(
        int numMels = 128,
        int hiddenDim = 256,
        int numClasses = 10,
        int maxFrames = 1000,
        int numAttentionLayers = 4,
        double dropoutRate = 0.3)
    {
        IActivationFunction<T> reluActivation = new ReLUActivation<T>();
        IActivationFunction<T> identityActivation = new IdentityActivation<T>();

        // Feature projection
        yield return new DenseLayer<T>(numMels, hiddenDim, reluActivation);
        yield return new LayerNormalizationLayer<T>(hiddenDim);

        if (dropoutRate > 0)
        {
            yield return new DropoutLayer<T>(dropoutRate);
        }

        // Positional encoding
        yield return new PositionalEncodingLayer<T>(maxFrames, hiddenDim);

        // Transformer encoder layers
        for (int i = 0; i < numAttentionLayers; i++)
        {
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: maxFrames,
                embeddingDimension: hiddenDim,
                headCount: 8);

            yield return new LayerNormalizationLayer<T>(hiddenDim);

            yield return new DenseLayer<T>(hiddenDim, hiddenDim * 4, reluActivation);
            yield return new DenseLayer<T>(hiddenDim * 4, hiddenDim, identityActivation);

            yield return new LayerNormalizationLayer<T>(hiddenDim);

            if (dropoutRate > 0)
            {
                yield return new DropoutLayer<T>(dropoutRate);
            }
        }

        // Classification head
        yield return new DenseLayer<T>(hiddenDim, hiddenDim, reluActivation);

        if (dropoutRate > 0)
        {
            yield return new DropoutLayer<T>(dropoutRate);
        }

        yield return new DenseLayer<T>(hiddenDim, numClasses, identityActivation);
        // Softmax is applied in the model's prediction logic
    }

    /// <summary>
    /// Creates default music source separation layers (U-Net style).
    /// </summary>
    /// <param name="numMels">Number of spectrogram frequency bins (default: 513 for STFT with 1024 window).</param>
    /// <param name="baseChannels">Base channel count for U-Net (default: 32).</param>
    /// <param name="numSources">Number of output sources (default: 4 for vocals, drums, bass, other).</param>
    /// <param name="maxFrames">Maximum time frames (default: 512).</param>
    /// <param name="dropoutRate">Dropout rate (default: 0.1).</param>
    /// <returns>A collection of layers for music source separation.</returns>
    /// <remarks>
    /// <para>
    /// U-Net inspired architecture for source separation with:
    /// </para>
    /// <list type="bullet">
    /// <item><description>Encoder path with downsampling</description></item>
    /// <item><description>Bottleneck with attention</description></item>
    /// <item><description>Decoder path with upsampling and skip connections</description></item>
    /// <item><description>Multi-source mask prediction</description></item>
    /// </list>
    /// <para>
    /// Reference: "Open-Unmix - A Reference Implementation for Music Source Separation"
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultSourceSeparationLayers(
        int numMels = 513,
        int baseChannels = 32,
        int numSources = 4,
        int maxFrames = 512,
        double dropoutRate = 0.1)
    {
        IActivationFunction<T> reluActivation = new ReLUActivation<T>();
        IActivationFunction<T> sigmoidActivation = new SigmoidActivation<T>();
        IActivationFunction<T> identityActivation = new IdentityActivation<T>();

        // === ENCODER PATH ===

        // Initial feature extraction
        yield return new DenseLayer<T>(numMels, baseChannels * 4, reluActivation);
        yield return new LayerNormalizationLayer<T>(baseChannels * 4);

        // Encoder level 1
        yield return new DenseLayer<T>(baseChannels * 4, baseChannels * 8, reluActivation);
        yield return new LayerNormalizationLayer<T>(baseChannels * 8);

        if (dropoutRate > 0)
        {
            yield return new DropoutLayer<T>(dropoutRate);
        }

        // Encoder level 2
        yield return new DenseLayer<T>(baseChannels * 8, baseChannels * 16, reluActivation);
        yield return new LayerNormalizationLayer<T>(baseChannels * 16);

        // === BOTTLENECK ===

        // Attention for global context
        yield return new MultiHeadAttentionLayer<T>(
            sequenceLength: maxFrames,
            embeddingDimension: baseChannels * 16,
            headCount: 8);

        yield return new LayerNormalizationLayer<T>(baseChannels * 16);

        // LSTM-like temporal modeling (using attention + dense)
        yield return new DenseLayer<T>(baseChannels * 16, baseChannels * 16, reluActivation);
        yield return new DenseLayer<T>(baseChannels * 16, baseChannels * 16, identityActivation);

        yield return new LayerNormalizationLayer<T>(baseChannels * 16);

        // === DECODER PATH ===

        // Decoder level 2
        yield return new DenseLayer<T>(baseChannels * 16, baseChannels * 8, reluActivation);
        yield return new LayerNormalizationLayer<T>(baseChannels * 8);

        if (dropoutRate > 0)
        {
            yield return new DropoutLayer<T>(dropoutRate);
        }

        // Decoder level 1
        yield return new DenseLayer<T>(baseChannels * 8, baseChannels * 4, reluActivation);
        yield return new LayerNormalizationLayer<T>(baseChannels * 4);

        // === OUTPUT LAYER ===

        // Project to output masks for all sources
        yield return new DenseLayer<T>(baseChannels * 4, numMels * numSources, sigmoidActivation);
    }

    #endregion

    #region Video AI Layers

    /// <summary>
    /// Creates layers for a video super-resolution model (Real-ESRGAN/BasicVSR++ style).
    /// </summary>
    /// <param name="inputChannels">Number of input channels (default: 3 for RGB).</param>
    /// <param name="inputHeight">Input video height.</param>
    /// <param name="inputWidth">Input video width.</param>
    /// <param name="numFeatures">Number of feature channels (default: 64).</param>
    /// <param name="numResBlocks">Number of residual blocks (default: 16).</param>
    /// <param name="scaleFactor">Upscaling factor (default: 2).</param>
    /// <param name="useTemporalConsistency">Whether to add temporal aggregation layer (default: true).</param>
    /// <returns>A collection of layers for video super-resolution.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Super-resolution models increase video resolution. This architecture
    /// uses residual blocks (skip connections) to preserve details while learning to add new ones.
    /// The upsampling at the end increases the spatial size by the scale factor.
    ///
    /// Architecture overview:
    /// 1. Initial convolution to extract features
    /// 2. Multiple residual blocks for deep feature learning
    /// 3. Temporal aggregation for video consistency (optional)
    /// 4. Pixel shuffle upsampling for resolution increase
    /// 5. Final convolution for output reconstruction
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultVideoSuperResolutionLayers(
        int inputChannels = 3,
        int inputHeight = 128,
        int inputWidth = 128,
        int numFeatures = 64,
        int numResBlocks = 16,
        int scaleFactor = 2,
        bool useTemporalConsistency = true)
    {
        // Validate scaleFactor is a positive power of two (Real-ESRGAN only supports 2x/4x)
        if (scaleFactor <= 0 || (scaleFactor & (scaleFactor - 1)) != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(scaleFactor),
                $"scaleFactor must be a positive power of two (e.g., 2, 4, 8). Got: {scaleFactor}");
        }

        // Track current spatial dimensions
        int currentHeight = inputHeight;
        int currentWidth = inputWidth;
        int currentChannels = inputChannels;

        // Initial feature extraction (no activation - will be followed by residual blocks)
        yield return new ConvolutionalLayer<T>(
            inputDepth: currentChannels,
            inputHeight: currentHeight,
            inputWidth: currentWidth,
            outputDepth: numFeatures,
            kernelSize: 3,
            stride: 1,
            padding: 1);
        currentChannels = numFeatures;

        // Residual blocks for deep feature extraction
        for (int i = 0; i < numResBlocks; i++)
        {
            // Each residual block: Conv -> ReLU -> Conv + Skip
            yield return new ConvolutionalLayer<T>(
                inputDepth: currentChannels,
                inputHeight: currentHeight,
                inputWidth: currentWidth,
                outputDepth: numFeatures,
                kernelSize: 3,
                stride: 1,
                padding: 1,
                activationFunction: new ReLUActivation<T>() as IActivationFunction<T>);

            yield return new ConvolutionalLayer<T>(
                inputDepth: numFeatures,
                inputHeight: currentHeight,
                inputWidth: currentWidth,
                outputDepth: numFeatures,
                kernelSize: 3,
                stride: 1,
                padding: 1);

            // Note: Skip connection would be handled in the model's forward pass
        }

        // Temporal aggregation layer for video consistency
        if (useTemporalConsistency)
        {
            yield return new ConvolutionalLayer<T>(
                inputDepth: currentChannels,
                inputHeight: currentHeight,
                inputWidth: currentWidth,
                outputDepth: numFeatures,
                kernelSize: 3,
                stride: 1,
                padding: 1);
        }

        // Upsampling layers using pixel shuffle
        int currentScale = 1;
        while (currentScale < scaleFactor)
        {
            // Each pixel shuffle doubles the resolution
            // Conv to expand channels for pixel shuffle (with ReLU activation)
            yield return new ConvolutionalLayer<T>(
                inputDepth: currentChannels,
                inputHeight: currentHeight,
                inputWidth: currentWidth,
                outputDepth: numFeatures * 4,  // 4x channels for 2x spatial
                kernelSize: 3,
                stride: 1,
                padding: 1,
                activationFunction: new ReLUActivation<T>() as IActivationFunction<T>);

            // Pixel shuffle: [C*4, H, W] -> [C, H*2, W*2]
            yield return new PixelShuffleLayer<T>(
                inputShape: [numFeatures * 4, currentHeight, currentWidth],
                upscaleFactor: 2);

            currentHeight *= 2;
            currentWidth *= 2;
            currentChannels = numFeatures;

            currentScale *= 2;
        }

        // Final reconstruction convolution (no activation - output should be in original range)
        yield return new ConvolutionalLayer<T>(
            inputDepth: currentChannels,
            inputHeight: currentHeight,
            inputWidth: currentWidth,
            outputDepth: inputChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1);
    }

    /// <summary>
    /// Creates a simple super-resolution architecture for testing and lightweight use.
    /// </summary>
    /// <param name="inputChannels">Number of input channels (default: 3 for RGB).</param>
    /// <param name="inputHeight">Input video height.</param>
    /// <param name="inputWidth">Input video width.</param>
    /// <param name="scaleFactor">Upscaling factor (default: 2).</param>
    /// <returns>A collection of layers for simple super-resolution.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is a smaller, faster model that trades quality for speed.
    /// Good for real-time applications or when GPU memory is limited.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateSimpleVideoSuperResolutionLayers(
        int inputChannels = 3,
        int inputHeight = 128,
        int inputWidth = 128,
        int scaleFactor = 2)
    {
        // Validate scaleFactor is a positive power of two
        if (scaleFactor <= 0 || (scaleFactor & (scaleFactor - 1)) != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(scaleFactor),
                $"scaleFactor must be a positive power of two (e.g., 2, 4, 8). Got: {scaleFactor}");
        }

        int numFeatures = 32;  // Smaller feature dimension
        int currentHeight = inputHeight;
        int currentWidth = inputWidth;

        // Initial feature extraction
        yield return new ConvolutionalLayer<T>(inputChannels, currentHeight, currentWidth, numFeatures, 5, 1, 2,
            new ReLUActivation<T>() as IActivationFunction<T>);

        // A few residual blocks
        for (int i = 0; i < 4; i++)
        {
            yield return new ConvolutionalLayer<T>(numFeatures, currentHeight, currentWidth, numFeatures, 3, 1, 1,
                new ReLUActivation<T>() as IActivationFunction<T>);
            yield return new ConvolutionalLayer<T>(numFeatures, currentHeight, currentWidth, numFeatures, 3, 1, 1);
        }

        // Upsampling
        int scale = scaleFactor;
        while (scale > 1)
        {
            // Conv with ReLU before pixel shuffle
            yield return new ConvolutionalLayer<T>(numFeatures, currentHeight, currentWidth, numFeatures * 4, 3, 1, 1,
                new ReLUActivation<T>() as IActivationFunction<T>);

            yield return new PixelShuffleLayer<T>(
                inputShape: [numFeatures * 4, currentHeight, currentWidth],
                upscaleFactor: 2);

            currentHeight *= 2;
            currentWidth *= 2;

            scale /= 2;
        }

        // Output (no activation)
        yield return new ConvolutionalLayer<T>(numFeatures, currentHeight, currentWidth, inputChannels, 3, 1, 1);
    }

    /// <summary>
    /// Creates layers for an optical flow estimation model (RAFT-style).
    /// </summary>
    /// <param name="inputChannels">Number of input channels (default: 3 for RGB).</param>
    /// <param name="inputHeight">Input frame height.</param>
    /// <param name="inputWidth">Input frame width.</param>
    /// <param name="hiddenDim">Hidden dimension for flow estimation (default: 192).</param>
    /// <returns>A collection of layers for optical flow estimation.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Optical flow tells you how each pixel moves between two frames.
    /// This is useful for motion analysis, video editing, and as input to other models.
    /// The output is a 2-channel tensor showing horizontal and vertical motion.
    ///
    /// Architecture:
    /// 1. Feature encoder extracts features from both frames
    /// 2. Correlation volume computes matching scores
    /// 3. Iterative refinement improves the flow estimate
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultOpticalFlowLayers(
        int inputChannels = 3,
        int inputHeight = 128,
        int inputWidth = 128,
        int hiddenDim = 192)
    {
        int h = inputHeight;
        int w = inputWidth;

        // Feature encoder (shared for both frames)
        yield return new ConvolutionalLayer<T>(inputChannels, h, w, 64, 7, 2, 3, new ReLUActivation<T>() as IActivationFunction<T>);
        h /= 2; w /= 2;

        yield return new ConvolutionalLayer<T>(64, h, w, 128, 3, 2, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        h /= 2; w /= 2;

        yield return new ConvolutionalLayer<T>(128, h, w, 256, 3, 2, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        h /= 2; w /= 2;

        // Context encoder (reset dimensions)
        h = inputHeight; w = inputWidth;
        yield return new ConvolutionalLayer<T>(inputChannels, h, w, 64, 7, 2, 3, new ReLUActivation<T>() as IActivationFunction<T>);
        h /= 2; w /= 2;

        yield return new ConvolutionalLayer<T>(64, h, w, 128, 3, 2, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        h /= 2; w /= 2;

        yield return new ConvolutionalLayer<T>(128, h, w, hiddenDim, 3, 2, 1);
        h /= 2; w /= 2;

        // Flow head (produces 2-channel flow output)
        yield return new ConvolutionalLayer<T>(hiddenDim, h, w, 128, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(128, h, w, 64, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(64, h, w, 2, 3, 1, 1);
    }

    /// <summary>
    /// Creates layers for a frame interpolation model (FILM/RIFE-style).
    /// </summary>
    /// <param name="inputChannels">Number of input channels (default: 3 for RGB).</param>
    /// <param name="inputHeight">Input frame height.</param>
    /// <param name="inputWidth">Input frame width.</param>
    /// <param name="numFeatures">Number of feature channels (default: 64).</param>
    /// <returns>A collection of layers for frame interpolation.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Frame interpolation creates new frames between existing ones
    /// to make video smoother (e.g., 30fps to 60fps). The model learns to "imagine"
    /// what the in-between frames should look like based on the surrounding frames.
    ///
    /// Architecture:
    /// 1. Feature pyramid extracts multi-scale features
    /// 2. Flow estimation predicts motion
    /// 3. Synthesis network generates interpolated frames
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultFrameInterpolationLayers(
        int inputChannels = 3,
        int inputHeight = 128,
        int inputWidth = 128,
        int numFeatures = 64)
    {
        int h = inputHeight;
        int w = inputWidth;

        // Feature pyramid network (two frames concatenated = inputChannels * 2)
        // Level 1
        yield return new ConvolutionalLayer<T>(inputChannels * 2, h, w, 32, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(32, h, w, 32, 3, 2, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        h /= 2; w /= 2;

        // Level 2
        yield return new ConvolutionalLayer<T>(32, h, w, 64, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(64, h, w, 64, 3, 2, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        h /= 2; w /= 2;

        // Level 3
        yield return new ConvolutionalLayer<T>(64, h, w, 96, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(96, h, w, 96, 3, 2, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        h /= 2; w /= 2;

        // Flow estimation head (outputs at downsampled resolution: h/8 x w/8)
        yield return new ConvolutionalLayer<T>(96, h, w, 64, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(64, h, w, 4, 3, 1, 1);  // 4 = 2 flows * 2 directions

        // NOTE: The synthesis network below expects input at the ORIGINAL resolution.
        // Higher-level models (FILM, FLAVR, etc.) should:
        // 1. Run the feature pyramid layers (indices 0-7) to get downsampled flow
        // 2. Upsample the flow to original resolution
        // 3. Concatenate original frames with upsampled flow
        // 4. Run the synthesis network layers (indices 8+) on that concatenation
        // The layer shapes below are defined for the concatenated input at original resolution.

        // Synthesis network (expects original resolution: inputHeight x inputWidth)
        // Input: [frames_concat (C*2), upsampled_flow (4)] = C*2 + 4 channels
        int synthH = inputHeight;
        int synthW = inputWidth;
        yield return new ConvolutionalLayer<T>(inputChannels * 2 + 4, synthH, synthW, numFeatures, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);

        for (int i = 0; i < 4; i++)
        {
            yield return new ConvolutionalLayer<T>(numFeatures, synthH, synthW, numFeatures, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        }

        yield return new ConvolutionalLayer<T>(numFeatures, synthH, synthW, inputChannels, 3, 1, 1);
    }

    /// <summary>
    /// Creates layers for a video stabilization model (StabNet-style).
    /// </summary>
    /// <param name="inputChannels">Number of input channels (default: 3 for RGB).</param>
    /// <param name="inputHeight">Input frame height.</param>
    /// <param name="inputWidth">Input frame width.</param>
    /// <returns>A collection of layers for video stabilization.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Video stabilization removes camera shake. The model predicts
    /// how to warp each frame to align with a smooth camera path. This is similar to
    /// what smartphone cameras do in real-time.
    ///
    /// Architecture:
    /// 1. Feature encoder processes input frames
    /// 2. Motion estimator predicts camera motion
    /// 3. Smoother learns the smooth target path
    /// 4. Warper transforms frames to match smooth path
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultVideoStabilizationLayers(
        int inputChannels = 3,
        int inputHeight = 128,
        int inputWidth = 128)
    {
        int h = inputHeight;
        int w = inputWidth;

        // Feature encoder
        yield return new ConvolutionalLayer<T>(inputChannels, h, w, 64, 7, 2, 3, new ReLUActivation<T>() as IActivationFunction<T>);
        h /= 2; w /= 2;

        yield return new ConvolutionalLayer<T>(64, h, w, 128, 3, 2, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        h /= 2; w /= 2;

        yield return new ConvolutionalLayer<T>(128, h, w, 256, 3, 2, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        h /= 2; w /= 2;

        // Motion estimation layers
        yield return new ConvolutionalLayer<T>(256, h, w, 256, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(256, h, w, 128, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);

        // Global average pooling to get fixed-size feature vector
        yield return new GlobalPoolingLayer<T>(
            inputShape: [128, h, w],
            poolingType: PoolingType.Average);

        // Output: 6 parameters for affine transformation
        yield return new DenseLayer<T>(128, 64, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new DenseLayer<T>(64, 6, new IdentityActivation<T>() as IActivationFunction<T>);  // 6 affine params
    }

    /// <summary>
    /// Creates layers for an InternVideo2-style video understanding model.
    /// </summary>
    /// <param name="inputChannels">Number of input channels (default: 3 for RGB).</param>
    /// <param name="inputHeight">Input frame height.</param>
    /// <param name="inputWidth">Input frame width.</param>
    /// <param name="embedDim">Embedding dimension (default: 768).</param>
    /// <param name="numEncoderLayers">Number of transformer encoder layers (default: 12).</param>
    /// <param name="patchSize">Patch size for video tokenization (default: 14).</param>
    /// <returns>A collection of layers for video understanding.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> InternVideo2 understands video content by encoding frames
    /// into embeddings that capture both spatial (what's in each frame) and temporal
    /// (how things change over time) information. It can be used for:
    /// - Video classification (identifying what's happening)
    /// - Video-text retrieval (finding videos matching descriptions)
    /// - Video question answering
    ///
    /// Architecture (based on the paper):
    /// 1. Patch embedding converts video frames into tokens
    /// 2. Spatial attention processes within-frame relationships
    /// 3. Temporal attention processes across-frame relationships
    /// 4. FFN layers add non-linearity and expressiveness
    /// 5. Projection maps to a shared video-text embedding space
    /// </para>
    /// <para>
    /// <b>Reference:</b> "InternVideo2: Scaling Video Foundation Models for Multimodal Video Understanding"
    /// https://arxiv.org/abs/2403.15377
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultInternVideo2Layers(
        int inputChannels = 3,
        int inputHeight = 224,
        int inputWidth = 224,
        int embedDim = 768,
        int numEncoderLayers = 12,
        int patchSize = 14)
    {
        int patchH = inputHeight / patchSize;
        int patchW = inputWidth / patchSize;

        // Patch embedding: converts image to sequence of patch embeddings
        yield return new ConvolutionalLayer<T>(inputChannels, inputHeight, inputWidth, embedDim, patchSize, patchSize, 0, new GELUActivation<T>() as IActivationFunction<T>);

        // Encoder layers with spatial and temporal attention
        for (int i = 0; i < numEncoderLayers; i++)
        {
            // Spatial self-attention (approximated as 1x1 conv for efficiency)
            yield return new ConvolutionalLayer<T>(embedDim, patchH, patchW, embedDim, 1, 1, 0, new GELUActivation<T>() as IActivationFunction<T>);

            // Temporal attention (every other layer for efficiency)
            if (i % 2 == 1)
            {
                yield return new ConvolutionalLayer<T>(embedDim, patchH, patchW, embedDim, 1, 1, 0, new GELUActivation<T>() as IActivationFunction<T>);
            }

            // FFN with expansion factor of 4
            yield return new ConvolutionalLayer<T>(embedDim, patchH, patchW, embedDim * 4, 1, 1, 0, new GELUActivation<T>() as IActivationFunction<T>);
            yield return new ConvolutionalLayer<T>(embedDim * 4, patchH, patchW, embedDim, 1, 1, 0);
        }

        // Global average pooling for CLS-like token
        yield return new GlobalPoolingLayer<T>(
            inputShape: [embedDim, patchH, patchW],
            poolingType: PoolingType.Average);

        // Projection to shared embedding space (512 is common for CLIP-like models)
        yield return new DenseLayer<T>(embedDim, 512, new IdentityActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Creates layers for a VRT (Video Restoration Transformer) model.
    /// </summary>
    /// <param name="inputChannels">Number of input channels (default: 3 for RGB).</param>
    /// <param name="inputHeight">Input frame height.</param>
    /// <param name="inputWidth">Input frame width.</param>
    /// <param name="embedDim">Embedding dimension (default: 120).</param>
    /// <param name="numFrames">Number of temporal frames (default: 6).</param>
    /// <param name="numBlocks">Number of transformer blocks (default: 8).</param>
    /// <param name="scaleFactor">Upscaling factor for super-resolution. Supported values: 1, 2, or 4 (default: 4).</param>
    /// <exception cref="ArgumentException">Thrown when scaleFactor is not 1, 2, or 4.</exception>
    /// <returns>A collection of layers for video restoration.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> VRT (Video Restoration Transformer) is a powerful model for:
    /// - Video super-resolution (increasing video resolution)
    /// - Video deblurring (removing motion blur)
    /// - Video denoising (removing noise from videos)
    ///
    /// It uses attention mechanisms to leverage both spatial and temporal information
    /// from multiple video frames to produce high-quality restored frames.
    ///
    /// Architecture (based on the paper):
    /// 1. Shallow feature extraction from input frames
    /// 2. Temporal mutual self-attention (TMSA) blocks
    /// 3. Deep feature extraction with parallel warping
    /// 4. Reconstruction module for output
    /// </para>
    /// <para>
    /// <b>Reference:</b> "VRT: A Video Restoration Transformer"
    /// https://arxiv.org/abs/2201.12288
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultVRTLayers(
        int inputChannels = 3,
        int inputHeight = 64,
        int inputWidth = 64,
        int embedDim = 120,
        int numFrames = 6,
        int numBlocks = 8,
        int scaleFactor = 4)
    {
        // Validate scaleFactor - only 1, 2, or 4 are supported due to pixel shuffle implementation
        if (scaleFactor != 1 && scaleFactor != 2 && scaleFactor != 4)
            throw new ArgumentException($"scaleFactor must be 1, 2, or 4. Got: {scaleFactor}", nameof(scaleFactor));

        int h = inputHeight;
        int w = inputWidth;

        // Shallow feature extraction
        yield return new ConvolutionalLayer<T>(inputChannels, h, w, embedDim, 3, 1, 1, new LeakyReLUActivation<T>(0.1) as IActivationFunction<T>);

        // Multi-scale feature extraction with encoder structure
        int currentDim = embedDim;
        for (int i = 0; i < 3; i++)
        {
            // Temporal mutual self-attention approximated with conv blocks
            for (int j = 0; j < numBlocks / 4; j++)
            {
                yield return new ConvolutionalLayer<T>(currentDim, h, w, currentDim, 3, 1, 1, new LeakyReLUActivation<T>(0.1) as IActivationFunction<T>);
                yield return new ConvolutionalLayer<T>(currentDim, h, w, currentDim, 3, 1, 1, new LeakyReLUActivation<T>(0.1) as IActivationFunction<T>);
            }

            if (i < 2)
            {
                // Downsample
                currentDim *= 2;
                yield return new ConvolutionalLayer<T>(currentDim / 2, h, w, currentDim, 4, 2, 1, new LeakyReLUActivation<T>(0.1) as IActivationFunction<T>);
                h /= 2; w /= 2;
            }
        }

        // Bottleneck with deep features
        for (int i = 0; i < numBlocks / 2; i++)
        {
            yield return new ConvolutionalLayer<T>(currentDim, h, w, currentDim, 3, 1, 1, new LeakyReLUActivation<T>(0.1) as IActivationFunction<T>);
        }

        // Decoder with upsampling for super-resolution
        for (int i = 0; i < 2; i++)
        {
            int prevDim = currentDim;
            currentDim /= 2;
            h *= 2; w *= 2;
            yield return new ConvolutionalLayer<T>(prevDim, h / 2, w / 2, currentDim * 4, 3, 1, 1, new LeakyReLUActivation<T>(0.1) as IActivationFunction<T>);
            yield return new PixelShuffleLayer<T>([currentDim * 4, h / 2, w / 2], 2);
        }

        // Upscaling for super-resolution (pixel shuffle for efficient upsampling)
        if (scaleFactor >= 2)
        {
            yield return new ConvolutionalLayer<T>(currentDim, h, w, currentDim * 4, 3, 1, 1, new LeakyReLUActivation<T>(0.1) as IActivationFunction<T>);
            yield return new PixelShuffleLayer<T>([currentDim * 4, h, w], 2);
            h *= 2; w *= 2;
        }
        if (scaleFactor >= 4)
        {
            yield return new ConvolutionalLayer<T>(currentDim, h, w, currentDim * 4, 3, 1, 1, new LeakyReLUActivation<T>(0.1) as IActivationFunction<T>);
            yield return new PixelShuffleLayer<T>([currentDim * 4, h, w], 2);
            h *= 2; w *= 2;
        }

        // Final reconstruction
        yield return new ConvolutionalLayer<T>(currentDim, h, w, currentDim, 3, 1, 1, new LeakyReLUActivation<T>(0.1) as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(currentDim, h, w, inputChannels, 3, 1, 1);
    }

    /// <summary>
    /// Creates layers for a CogVideo text-to-video generation model.
    /// </summary>
    /// <param name="inputChannels">Number of input channels for latent (default: 4).</param>
    /// <param name="inputHeight">Input latent height (default: 32).</param>
    /// <param name="inputWidth">Input latent width (default: 32).</param>
    /// <param name="embedDim">Embedding dimension (default: 1024).</param>
    /// <param name="numLayers">Number of transformer layers (default: 24).</param>
    /// <param name="numFrames">Number of video frames to generate (default: 16).</param>
    /// <returns>A collection of layers for video generation.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> CogVideo generates videos from text descriptions.
    /// It works in the latent space (compressed representation) and uses a
    /// diffusion-based approach to iteratively refine noise into coherent video.
    ///
    /// Architecture (based on the CogVideoX paper):
    /// 1. Text encoder processes the input prompt
    /// 2. Latent space diffusion model generates video frames
    /// 3. VAE decoder converts latent to pixel space
    ///
    /// This creates the denoising U-Net backbone that refines latent codes.
    /// </para>
    /// <para>
    /// <b>Reference:</b> "CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer"
    /// https://arxiv.org/abs/2408.06072
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultCogVideoLayers(
        int inputChannels = 4,
        int inputHeight = 32,
        int inputWidth = 32,
        int embedDim = 1024,
        int numLayers = 24,
        int numFrames = 16)
    {
        int h = inputHeight;
        int w = inputWidth;

        // Input projection for latent + timestep conditioning
        yield return new ConvolutionalLayer<T>(inputChannels, h, w, embedDim, 3, 1, 1, new SiLUActivation<T>() as IActivationFunction<T>);

        // Encoder (downsampling path)
        int currentDim = embedDim;
        int[] channelMults = { 1, 2, 4, 4 };

        foreach (var mult in channelMults)
        {
            int outDim = embedDim * mult;

            // Two residual-style conv blocks per level
            yield return new ConvolutionalLayer<T>(currentDim, h, w, outDim, 3, 1, 1, new SiLUActivation<T>() as IActivationFunction<T>);
            yield return new ConvolutionalLayer<T>(outDim, h, w, outDim, 3, 1, 1, new SiLUActivation<T>() as IActivationFunction<T>);

            currentDim = outDim;

            // Downsample (except last level)
            if (mult != channelMults[^1])
            {
                yield return new ConvolutionalLayer<T>(currentDim, h, w, currentDim, 4, 2, 1, new SiLUActivation<T>() as IActivationFunction<T>);
                h /= 2; w /= 2;
            }
        }

        // Middle block with transformer layers
        for (int i = 0; i < Math.Min(numLayers / 4, 6); i++)
        {
            // Spatial attention approximation
            yield return new ConvolutionalLayer<T>(currentDim, h, w, currentDim, 1, 1, 0, new SiLUActivation<T>() as IActivationFunction<T>);
            // Temporal attention approximation
            yield return new ConvolutionalLayer<T>(currentDim, h, w, currentDim, 1, 1, 0, new SiLUActivation<T>() as IActivationFunction<T>);
            // FFN
            yield return new ConvolutionalLayer<T>(currentDim, h, w, currentDim * 4, 1, 1, 0, new SiLUActivation<T>() as IActivationFunction<T>);
            yield return new ConvolutionalLayer<T>(currentDim * 4, h, w, currentDim, 1, 1, 0);
        }

        // Decoder (upsampling path)
        for (int i = channelMults.Length - 1; i >= 0; i--)
        {
            int outDim = i > 0 ? embedDim * channelMults[i - 1] : embedDim;

            // Two residual-style conv blocks per level
            yield return new ConvolutionalLayer<T>(currentDim, h, w, currentDim, 3, 1, 1, new SiLUActivation<T>() as IActivationFunction<T>);
            yield return new ConvolutionalLayer<T>(currentDim, h, w, outDim, 3, 1, 1, new SiLUActivation<T>() as IActivationFunction<T>);

            currentDim = outDim;

            // Upsample (except first level)
            if (i > 0)
            {
                yield return new ConvolutionalLayer<T>(currentDim, h, w, currentDim * 4, 3, 1, 1, new SiLUActivation<T>() as IActivationFunction<T>);
                yield return new PixelShuffleLayer<T>([currentDim * 4, h, w], 2);
                h *= 2; w *= 2;
            }
        }

        // Output projection back to latent channels
        yield return new ConvolutionalLayer<T>(currentDim, h, w, currentDim, 3, 1, 1, new SiLUActivation<T>() as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(currentDim, h, w, inputChannels, 3, 1, 1);
    }

    /// <summary>
    /// Creates layers for an AnimateDiff motion module that adds temporal coherence.
    /// </summary>
    /// <param name="inputChannels">Number of input feature channels (default: 320).</param>
    /// <param name="inputHeight">Input feature height (default: 64).</param>
    /// <param name="inputWidth">Input feature width (default: 64).</param>
    /// <param name="numLayers">Number of motion transformer layers (default: 8).</param>
    /// <param name="numFrames">Number of video frames (default: 16).</param>
    /// <returns>A collection of layers for motion modeling.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> AnimateDiff is a motion module that plugs into existing
    /// image generation models (like Stable Diffusion) to create animated videos.
    /// It learns temporal dynamics from video data.
    ///
    /// Architecture (based on the paper):
    /// 1. Input features come from the base image model
    /// 2. Temporal attention layers model motion across frames
    /// 3. Cross-attention with motion context enables coherent animation
    /// 4. Output features blend back into the base model
    ///
    /// The motion module is designed to be inserted at multiple points in the U-Net.
    /// </para>
    /// <para>
    /// <b>Reference:</b> "AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models"
    /// https://arxiv.org/abs/2307.04725
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultAnimateDiffLayers(
        int inputChannels = 320,
        int inputHeight = 64,
        int inputWidth = 64,
        int numLayers = 8,
        int numFrames = 16)
    {
        int h = inputHeight;
        int w = inputWidth;

        // Input normalization and projection
        yield return new ConvolutionalLayer<T>(inputChannels, h, w, inputChannels, 1, 1, 0, new SiLUActivation<T>() as IActivationFunction<T>);

        // Motion transformer layers with temporal attention
        for (int i = 0; i < numLayers; i++)
        {
            // Temporal self-attention (approximated with 1x1 conv for efficiency)
            yield return new ConvolutionalLayer<T>(inputChannels, h, w, inputChannels, 1, 1, 0, new SiLUActivation<T>() as IActivationFunction<T>);

            // Position-wise FFN with expansion
            yield return new ConvolutionalLayer<T>(inputChannels, h, w, inputChannels * 4, 1, 1, 0, new GELUActivation<T>() as IActivationFunction<T>);
            yield return new ConvolutionalLayer<T>(inputChannels * 4, h, w, inputChannels, 1, 1, 0);
        }

        // Multi-scale temporal processing for different motion granularities
        int[] channelMults = { 1, 2, 4 };
        int currentChannels = inputChannels;

        foreach (var mult in channelMults)
        {
            int outChannels = inputChannels * mult;

            // Downsample
            yield return new ConvolutionalLayer<T>(currentChannels, h, w, outChannels, 4, 2, 1, new SiLUActivation<T>() as IActivationFunction<T>);
            h /= 2; w /= 2;
            currentChannels = outChannels;

            // Temporal attention at this scale
            yield return new ConvolutionalLayer<T>(currentChannels, h, w, currentChannels, 1, 1, 0, new SiLUActivation<T>() as IActivationFunction<T>);
            yield return new ConvolutionalLayer<T>(currentChannels, h, w, currentChannels, 3, 1, 1, new SiLUActivation<T>() as IActivationFunction<T>);
        }

        // Upsample back to original resolution
        for (int i = channelMults.Length - 1; i >= 0; i--)
        {
            int outChannels = i > 0 ? inputChannels * channelMults[i - 1] : inputChannels;

            // Upsample using pixel shuffle
            yield return new ConvolutionalLayer<T>(currentChannels, h, w, outChannels * 4, 3, 1, 1, new SiLUActivation<T>() as IActivationFunction<T>);
            yield return new PixelShuffleLayer<T>([outChannels * 4, h, w], 2);
            h *= 2; w *= 2;
            currentChannels = outChannels;

            // Temporal processing at this scale
            yield return new ConvolutionalLayer<T>(currentChannels, h, w, currentChannels, 3, 1, 1, new SiLUActivation<T>() as IActivationFunction<T>);
        }

        // Output projection with residual
        yield return new ConvolutionalLayer<T>(currentChannels, h, w, inputChannels, 1, 1, 0);
    }

    /// <summary>
    /// Creates layers for a Cutie video object segmentation model.
    /// </summary>
    /// <param name="inputChannels">Number of input channels (default: 3 for RGB).</param>
    /// <param name="inputHeight">Input frame height (default: 480).</param>
    /// <param name="inputWidth">Input frame width (default: 854).</param>
    /// <param name="numFeatures">Feature dimension (default: 256).</param>
    /// <returns>A collection of layers for video object segmentation.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Cutie is designed for semi-supervised video object segmentation (VOS).
    /// Given a mask for an object in the first frame, it tracks and segments that object
    /// throughout the entire video with high accuracy.
    ///
    /// Architecture:
    /// 1. Image encoder (ResNet-like backbone) extracts features
    /// 2. Object encoder processes mask with features
    /// 3. Memory attention matches current frame to stored memories
    /// 4. Mask decoder produces segmentation output
    /// </para>
    /// <para>
    /// <b>Reference:</b> "Putting the Object Back into Video Object Segmentation"
    /// https://arxiv.org/abs/2310.12982
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultCutieLayers(
        int inputChannels = 3,
        int inputHeight = 480,
        int inputWidth = 854,
        int numFeatures = 256)
    {
        // Helper to compute convolution output size: (input + 2*padding - kernel) / stride + 1
        static int ConvOutSize(int input, int kernel, int stride, int padding) =>
            (input + 2 * padding - kernel) / stride + 1;

        int h = inputHeight;
        int w = inputWidth;

        // Image encoder (ResNet-like backbone)
        yield return new ConvolutionalLayer<T>(inputChannels, h, w, 64, 7, 2, 3, new ReLUActivation<T>() as IActivationFunction<T>);
        h = ConvOutSize(h, 7, 2, 3); w = ConvOutSize(w, 7, 2, 3);
        yield return new ConvolutionalLayer<T>(64, h, w, 64, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(64, h, w, 128, 3, 2, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        h = ConvOutSize(h, 3, 2, 1); w = ConvOutSize(w, 3, 2, 1);
        yield return new ConvolutionalLayer<T>(128, h, w, 256, 3, 2, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        h = ConvOutSize(h, 3, 2, 1); w = ConvOutSize(w, 3, 2, 1);
        yield return new ConvolutionalLayer<T>(256, h, w, numFeatures, 3, 2, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        h = ConvOutSize(h, 3, 2, 1); w = ConvOutSize(w, 3, 2, 1);

        // Object encoder (processes mask with image features)
        // Note: This takes numFeatures + 1 channels (features + mask)
        yield return new ConvolutionalLayer<T>(numFeatures + 1, h, w, numFeatures, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, numFeatures, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);

        // Query/Key/Value projections for memory attention
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, numFeatures, 1, 1, 0);
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, numFeatures, 1, 1, 0);
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, numFeatures, 1, 1, 0);

        // Memory attention layers
        for (int i = 0; i < 4; i++)
        {
            yield return new ConvolutionalLayer<T>(numFeatures, h, w, numFeatures, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        }

        // Mask decoder with upsampling
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, 128, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        h *= 2; w *= 2;
        yield return new ConvolutionalLayer<T>(128, h, w, 64, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        h *= 2; w *= 2;
        yield return new ConvolutionalLayer<T>(64, h, w, 32, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        h *= 2; w *= 2;
        yield return new ConvolutionalLayer<T>(32, h, w, 16, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        h *= 2; w *= 2;

        // Final mask head (outputs 1 channel for binary segmentation)
        yield return new ConvolutionalLayer<T>(16, h, w, 1, 3, 1, 1, new SigmoidActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Creates layers for an XMem long-term video object segmentation model.
    /// </summary>
    /// <param name="inputChannels">Number of input channels (default: 3 for RGB).</param>
    /// <param name="inputHeight">Input frame height (default: 480).</param>
    /// <param name="inputWidth">Input frame width (default: 854).</param>
    /// <param name="numFeatures">Feature dimension (default: 256).</param>
    /// <returns>A collection of layers for long-term video object segmentation.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> XMem is designed for tracking objects in very long videos
    /// using a three-tier memory system inspired by human memory:
    /// - Sensory memory: Very recent frames (high detail, fast to forget)
    /// - Working memory: Important recent frames (moderate detail)
    /// - Long-term memory: Key historical frames (compressed, permanent)
    /// </para>
    /// <para>
    /// <b>Reference:</b> "XMem: Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model"
    /// https://arxiv.org/abs/2207.07115
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultXMemLayers(
        int inputChannels = 3,
        int inputHeight = 480,
        int inputWidth = 854,
        int numFeatures = 256)
    {
        // Helper to compute convolution output size: (input + 2*padding - kernel) / stride + 1
        static int ConvOutSize(int input, int kernel, int stride, int padding) =>
            (input + 2 * padding - kernel) / stride + 1;

        int h = inputHeight;
        int w = inputWidth;

        // Encoder
        yield return new ConvolutionalLayer<T>(inputChannels, h, w, 64, 7, 2, 3, new ReLUActivation<T>() as IActivationFunction<T>);
        h = ConvOutSize(h, 7, 2, 3); w = ConvOutSize(w, 7, 2, 3);
        yield return new ConvolutionalLayer<T>(64, h, w, 128, 3, 2, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        h = ConvOutSize(h, 3, 2, 1); w = ConvOutSize(w, 3, 2, 1);
        yield return new ConvolutionalLayer<T>(128, h, w, 256, 3, 2, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        h = ConvOutSize(h, 3, 2, 1); w = ConvOutSize(w, 3, 2, 1);
        yield return new ConvolutionalLayer<T>(256, h, w, numFeatures, 3, 2, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        h = ConvOutSize(h, 3, 2, 1); w = ConvOutSize(w, 3, 2, 1);

        // Sensory memory network (high resolution, short-term)
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, numFeatures, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, numFeatures, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);

        // Working memory network (medium resolution)
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, numFeatures / 2, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(numFeatures / 2, h, w, numFeatures / 2, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);

        // Long-term memory network (compressed)
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, numFeatures / 4, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(numFeatures / 4, h, w, numFeatures / 4, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);

        // Memory fusion (combines sensory + working + long-term)
        int totalFusionChannels = numFeatures + numFeatures / 2 + numFeatures / 4;
        yield return new ConvolutionalLayer<T>(totalFusionChannels, h, w, numFeatures, 1, 1, 0, new ReLUActivation<T>() as IActivationFunction<T>);

        // Decoder with upsampling
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, 128, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        h *= 2; w *= 2;
        yield return new ConvolutionalLayer<T>(128, h, w, 64, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        h *= 2; w *= 2;
        yield return new ConvolutionalLayer<T>(64, h, w, 32, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        h *= 2; w *= 2;
        yield return new ConvolutionalLayer<T>(32, h, w, 16, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        h *= 2; w *= 2;

        // Final mask head
        yield return new ConvolutionalLayer<T>(16, h, w, 1, 3, 1, 1, new SigmoidActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Creates default layers for SAM2 (Segment Anything Model 2) video object segmentation.
    /// </summary>
    /// <param name="inputChannels">Number of input channels (default: 3 for RGB).</param>
    /// <param name="inputHeight">Input height (default: 1024).</param>
    /// <param name="inputWidth">Input width (default: 1024).</param>
    /// <param name="numFeatures">Number of feature channels (default: 256).</param>
    /// <returns>An enumerable of layers configured for SAM2.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> SAM2 is a powerful model that can segment any object in video.
    /// You can interact with it by clicking on objects or drawing boxes, and it will
    /// track and segment those objects across all video frames.
    /// </para>
    /// <para>
    /// <b>Architecture:</b>
    /// - Hierarchical image encoder (ViT-like)
    /// - Prompt encoders (points, boxes, masks)
    /// - Memory attention for temporal consistency
    /// - Mask decoder with multiple mask candidates
    /// </para>
    /// <para>
    /// <b>Reference:</b> "SAM 2: Segment Anything in Images and Videos"
    /// https://arxiv.org/abs/2408.00714
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultSAM2Layers(
        int inputChannels = 3,
        int inputHeight = 1024,
        int inputWidth = 1024,
        int numFeatures = 256)
    {
        int h = inputHeight;
        int w = inputWidth;

        // Stage 1: Initial patch embedding (4x downsample)
        yield return new ConvolutionalLayer<T>(inputChannels, h, w, 64, 4, 4, 0, new ReLUActivation<T>() as IActivationFunction<T>);
        h /= 4; w /= 4;
        yield return new ConvolutionalLayer<T>(64, h, w, 64, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(64, h, w, 64, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);

        // Stage 2: 2x downsample
        yield return new ConvolutionalLayer<T>(64, h, w, 128, 3, 2, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        h /= 2; w /= 2;
        yield return new ConvolutionalLayer<T>(128, h, w, 128, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(128, h, w, 128, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);

        // Stage 3: 2x downsample
        yield return new ConvolutionalLayer<T>(128, h, w, 256, 3, 2, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        h /= 2; w /= 2;
        yield return new ConvolutionalLayer<T>(256, h, w, 256, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(256, h, w, 256, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);

        // Stage 4: 2x downsample (final encoder stage)
        yield return new ConvolutionalLayer<T>(256, h, w, numFeatures * 2, 3, 2, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        h /= 2; w /= 2;
        yield return new ConvolutionalLayer<T>(numFeatures * 2, h, w, numFeatures * 2, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);

        // Neck (feature pyramid fusion)
        yield return new ConvolutionalLayer<T>(numFeatures * 2, h, w, numFeatures, 1, 1, 0, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, numFeatures, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);

        // Prompt encoders (point, box, mask)
        yield return new ConvolutionalLayer<T>(2, 1, 1, numFeatures, 1, 1, 0, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(4, 1, 1, numFeatures, 1, 1, 0, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(1, h * 4, w * 4, numFeatures / 4, 4, 4, 0, new ReLUActivation<T>() as IActivationFunction<T>);

        // Memory attention layers
        yield return new ConvolutionalLayer<T>(numFeatures * 2, h, w, numFeatures, 1, 1, 0, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, numFeatures, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(numFeatures * 2, h, w, numFeatures, 1, 1, 0, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, numFeatures, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);

        // Memory projection
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, numFeatures, 1, 1, 0, new ReLUActivation<T>() as IActivationFunction<T>);

        // Mask decoder
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, numFeatures, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, numFeatures, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);

        // Output heads: mask candidates, IoU scores, occlusion
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, 4, 1, 1, 0, new SigmoidActivation<T>() as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(numFeatures, 1, 1, 4, 1, 1, 0, new SigmoidActivation<T>() as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(numFeatures, 1, 1, 1, 1, 1, 0, new SigmoidActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Creates default layers for VideoMAE (Video Masked Autoencoder) action recognition model.
    /// </summary>
    /// <param name="inputChannels">Number of input channels (default: 3 for RGB).</param>
    /// <param name="inputHeight">Input height (default: 224).</param>
    /// <param name="inputWidth">Input width (default: 224).</param>
    /// <param name="numFeatures">Number of feature channels (default: 768).</param>
    /// <param name="numClasses">Number of action classes (default: 400 for Kinetics).</param>
    /// <param name="tubeletSize">Temporal size of each tube (default: 2).</param>
    /// <returns>An enumerable of layers configured for VideoMAE.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> VideoMAE is a self-supervised learning model that learns video
    /// representations by masking and reconstructing video patches. It's used for action
    /// recognition and video understanding tasks.
    /// </para>
    /// <para>
    /// <b>Architecture:</b>
    /// - 3D patch embedding (spatiotemporal)
    /// - Transformer encoder blocks
    /// - Classification head for action recognition
    /// - Decoder for masked reconstruction during pretraining
    /// </para>
    /// <para>
    /// <b>Reference:</b> "VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training"
    /// https://arxiv.org/abs/2203.12602
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultVideoMAELayers(
        int inputChannels = 3,
        int inputHeight = 224,
        int inputWidth = 224,
        int numFeatures = 768,
        int numClasses = 400,
        int tubeletSize = 2)
    {
        int patchSize = 16;
        int featH = inputHeight / patchSize;
        int featW = inputWidth / patchSize;

        // 3D patch embedding (spatiotemporal)
        yield return new ConvolutionalLayer<T>(inputChannels * tubeletSize, inputHeight, inputWidth, numFeatures, patchSize, patchSize, 0, new ReLUActivation<T>() as IActivationFunction<T>);

        // Transformer encoder blocks (12 blocks)
        for (int i = 0; i < 12; i++)
        {
            yield return new ConvolutionalLayer<T>(numFeatures, featH, featW, numFeatures, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        }

        // Classification head
        yield return new ConvolutionalLayer<T>(numFeatures, featH, featW, numFeatures, 1, 1, 0, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(numFeatures, 1, 1, numClasses, 1, 1, 0, new SoftmaxActivation<T>() as IActivationFunction<T>);

        // Decoder blocks for reconstruction (4 blocks)
        for (int i = 0; i < 4; i++)
        {
            yield return new ConvolutionalLayer<T>(numFeatures, featH, featW, numFeatures, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        }

        // Reconstruction head
        yield return new ConvolutionalLayer<T>(numFeatures, featH, featW, inputChannels * tubeletSize * patchSize * patchSize, 1, 1, 0);
    }

    /// <summary>
    /// Creates default layers for Depth Anything V2 monocular depth estimation model.
    /// </summary>
    /// <param name="inputChannels">Number of input channels (default: 3 for RGB).</param>
    /// <param name="inputHeight">Input height (default: 480).</param>
    /// <param name="inputWidth">Input width (default: 640).</param>
    /// <param name="numFeatures">Number of feature channels (default: 768 for Base).</param>
    /// <param name="numEncoderBlocks">Number of encoder transformer blocks (default: 12).</param>
    /// <returns>An enumerable of layers configured for Depth Anything V2.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Depth Anything V2 estimates depth maps from single images.
    /// Given an RGB image, it predicts the relative distance of each pixel from the camera.
    /// </para>
    /// <para>
    /// <b>Architecture:</b>
    /// - ViT-based encoder with DINOv2 initialization
    /// - Multi-scale decoder for dense prediction
    /// - Depth prediction head
    /// </para>
    /// <para>
    /// <b>Reference:</b> "Depth Anything V2"
    /// https://arxiv.org/abs/2406.09414
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultDepthAnythingV2Layers(
        int inputChannels = 3,
        int inputHeight = 480,
        int inputWidth = 640,
        int numFeatures = 768,
        int numEncoderBlocks = 12)
    {
        int patchSize = 16;
        int featH = inputHeight / patchSize;
        int featW = inputWidth / patchSize;

        // Patch embedding
        yield return new ConvolutionalLayer<T>(inputChannels, inputHeight, inputWidth, numFeatures, patchSize, patchSize, 0, new ReLUActivation<T>() as IActivationFunction<T>);

        // Encoder transformer blocks
        for (int i = 0; i < numEncoderBlocks; i++)
        {
            yield return new ConvolutionalLayer<T>(numFeatures, featH, featW, numFeatures, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        }

        // Decoder blocks with progressive upsampling
        int h = featH;
        int w = featW;
        int currentFeatures = numFeatures;

        // Stage 1
        yield return new ConvolutionalLayer<T>(currentFeatures, h, w, numFeatures / 2, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        currentFeatures = numFeatures / 2;

        // Stage 2
        h *= 2; w *= 2;
        yield return new ConvolutionalLayer<T>(currentFeatures, h, w, numFeatures / 4, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        currentFeatures = numFeatures / 4;

        // Stage 3
        h *= 2; w *= 2;
        yield return new ConvolutionalLayer<T>(currentFeatures, h, w, numFeatures / 8, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        currentFeatures = numFeatures / 8;

        // Stage 4
        h *= 2; w *= 2;
        yield return new ConvolutionalLayer<T>(currentFeatures, h, w, 64, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);

        // Depth head
        h *= 2; w *= 2;
        yield return new ConvolutionalLayer<T>(64, h, w, 1, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Creates default layers for TimeSformer video classification.
    /// </summary>
    public static IEnumerable<ILayer<T>> CreateDefaultTimeSformerLayers(
        int inputChannels = 3,
        int inputHeight = 224,
        int inputWidth = 224,
        int embedDim = 768,
        int numLayers = 12,
        int patchSize = 16,
        int numClasses = 400)
    {
        int numPatches = (inputHeight / patchSize) * (inputWidth / patchSize);

        // Patch embedding
        yield return new ConvolutionalLayer<T>(inputChannels, inputHeight, inputWidth, embedDim, patchSize, patchSize, 0);

        // Transformer encoder blocks (divided space-time attention)
        for (int i = 0; i < numLayers; i++)
        {
            // Temporal attention
            yield return new ConvolutionalLayer<T>(embedDim, 1, numPatches, embedDim, 1, 1, 0, new GELUActivation<T>() as IActivationFunction<T>);
            // Spatial attention
            yield return new ConvolutionalLayer<T>(embedDim, 1, numPatches, embedDim, 1, 1, 0, new GELUActivation<T>() as IActivationFunction<T>);
            // MLP
            yield return new ConvolutionalLayer<T>(embedDim, 1, numPatches, embedDim * 4, 1, 1, 0, new GELUActivation<T>() as IActivationFunction<T>);
            yield return new ConvolutionalLayer<T>(embedDim * 4, 1, numPatches, embedDim, 1, 1, 0);
        }

        // Classification head
        yield return new DenseLayer<T>(embedDim, numClasses, new SoftmaxActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Creates default layers for SlowFast video recognition.
    /// </summary>
    public static IEnumerable<ILayer<T>> CreateDefaultSlowFastLayers(
        int inputChannels = 3,
        int inputHeight = 224,
        int inputWidth = 224,
        int numClasses = 400,
        int slowChannels = 64,
        int fastChannels = 8,
        int alpha = 8)
    {
        int h = inputHeight;
        int w = inputWidth;

        // Slow pathway - processes fewer frames at higher channel capacity
        yield return new ConvolutionalLayer<T>(inputChannels, h, w, slowChannels, 7, 2, 3, new ReLUActivation<T>() as IActivationFunction<T>);
        h /= 2; w /= 2;
        yield return new ConvolutionalLayer<T>(slowChannels, h, w, slowChannels * 2, 3, 2, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        h /= 2; w /= 2;
        yield return new ConvolutionalLayer<T>(slowChannels * 2, h, w, slowChannels * 4, 3, 2, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        h /= 2; w /= 2;
        yield return new ConvolutionalLayer<T>(slowChannels * 4, h, w, slowChannels * 8, 3, 2, 1, new ReLUActivation<T>() as IActivationFunction<T>);

        // Fast pathway - processes more frames at lower channel capacity
        int fh = inputHeight;
        int fw = inputWidth;
        yield return new ConvolutionalLayer<T>(inputChannels, fh, fw, fastChannels, 7, 2, 3, new ReLUActivation<T>() as IActivationFunction<T>);
        fh /= 2; fw /= 2;
        yield return new ConvolutionalLayer<T>(fastChannels, fh, fw, fastChannels * 2, 3, 2, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        fh /= 2; fw /= 2;
        yield return new ConvolutionalLayer<T>(fastChannels * 2, fh, fw, fastChannels * 4, 3, 2, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        fh /= 2; fw /= 2;
        yield return new ConvolutionalLayer<T>(fastChannels * 4, fh, fw, fastChannels * 8, 3, 2, 1, new ReLUActivation<T>() as IActivationFunction<T>);

        // Lateral connections and fusion
        int fusedChannels = slowChannels * 8 + fastChannels * 8;

        // Global average pooling + classification
        yield return new DenseLayer<T>(fusedChannels, numClasses, new SoftmaxActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Creates default layers for MiDaS depth estimation.
    /// </summary>
    public static IEnumerable<ILayer<T>> CreateDefaultMiDaSLayers(
        int inputChannels = 3,
        int inputHeight = 384,
        int inputWidth = 384,
        int embedDim = 768,
        int numEncoderLayers = 12)
    {
        int h = inputHeight;
        int w = inputWidth;
        int patchSize = 16;
        int numPatches = (h / patchSize) * (w / patchSize);

        // Patch embedding (ViT-style)
        yield return new ConvolutionalLayer<T>(inputChannels, h, w, embedDim, patchSize, patchSize, 0);

        // Transformer encoder blocks
        for (int i = 0; i < numEncoderLayers; i++)
        {
            yield return new ConvolutionalLayer<T>(embedDim, 1, numPatches, embedDim, 1, 1, 0, new GELUActivation<T>() as IActivationFunction<T>);
            yield return new ConvolutionalLayer<T>(embedDim, 1, numPatches, embedDim * 4, 1, 1, 0, new GELUActivation<T>() as IActivationFunction<T>);
            yield return new ConvolutionalLayer<T>(embedDim * 4, 1, numPatches, embedDim, 1, 1, 0);
        }

        // Decoder with reassemble and fusion
        h = inputHeight / 16; w = inputWidth / 16;
        yield return new ConvolutionalLayer<T>(embedDim, h, w, 256, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        h *= 2; w *= 2;
        yield return new ConvolutionalLayer<T>(256, h, w, 128, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        h *= 2; w *= 2;
        yield return new ConvolutionalLayer<T>(128, h, w, 64, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        h *= 2; w *= 2;
        yield return new ConvolutionalLayer<T>(64, h, w, 32, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        h *= 2; w *= 2;

        // Depth head (relative depth output)
        yield return new ConvolutionalLayer<T>(32, h, w, 1, 1, 1, 0);
    }

    /// <summary>
    /// Creates default layers for EDVR video restoration.
    /// </summary>
    public static IEnumerable<ILayer<T>> CreateDefaultEDVRLayers(
        int inputChannels = 3,
        int inputHeight = 256,
        int inputWidth = 256,
        int numFeatures = 64,
        int numFrames = 5,
        int numGroups = 8,
        int numBlocks = 5)
    {
        int h = inputHeight;
        int w = inputWidth;

        // Feature extraction
        yield return new ConvolutionalLayer<T>(inputChannels * numFrames, h, w, numFeatures, 3, 1, 1, new LeakyReLUActivation<T>() as IActivationFunction<T>);

        // PCD (Pyramid Cascading and Deformable) alignment
        for (int level = 0; level < 3; level++)
        {
            int scale = (int)Math.Pow(2, level);
            int scaledH = h / scale;
            int scaledW = w / scale;
            yield return new ConvolutionalLayer<T>(numFeatures, scaledH, scaledW, numFeatures, 3, 1, 1, new LeakyReLUActivation<T>() as IActivationFunction<T>);
        }

        // TSA (Temporal and Spatial Attention) fusion
        yield return new ConvolutionalLayer<T>(numFeatures * numFrames, h, w, numFeatures, 1, 1, 0, new LeakyReLUActivation<T>() as IActivationFunction<T>);

        // Reconstruction with residual blocks
        for (int i = 0; i < numBlocks; i++)
        {
            yield return new ConvolutionalLayer<T>(numFeatures, h, w, numFeatures, 3, 1, 1, new LeakyReLUActivation<T>() as IActivationFunction<T>);
            yield return new ConvolutionalLayer<T>(numFeatures, h, w, numFeatures, 3, 1, 1);
        }

        // Upsampling
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, numFeatures * 4, 3, 1, 1, new LeakyReLUActivation<T>() as IActivationFunction<T>);
        h *= 2; w *= 2;
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, numFeatures * 4, 3, 1, 1, new LeakyReLUActivation<T>() as IActivationFunction<T>);
        h *= 2; w *= 2;

        // Output
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, inputChannels, 3, 1, 1);
    }

    /// <summary>
    /// Creates default layers for FLAVR frame interpolation.
    /// </summary>
    public static IEnumerable<ILayer<T>> CreateDefaultFLAVRLayers(
        int inputChannels = 3,
        int inputHeight = 256,
        int inputWidth = 256,
        int numFeatures = 64,
        int numInputFrames = 4)
    {
        int h = inputHeight;
        int w = inputWidth;

        // Encoder (3D convolutions for spatiotemporal features)
        yield return new ConvolutionalLayer<T>(inputChannels * numInputFrames, h, w, numFeatures, 7, 1, 3, new LeakyReLUActivation<T>() as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, numFeatures * 2, 3, 2, 1, new LeakyReLUActivation<T>() as IActivationFunction<T>);
        h /= 2; w /= 2;
        yield return new ConvolutionalLayer<T>(numFeatures * 2, h, w, numFeatures * 4, 3, 2, 1, new LeakyReLUActivation<T>() as IActivationFunction<T>);
        h /= 2; w /= 2;
        yield return new ConvolutionalLayer<T>(numFeatures * 4, h, w, numFeatures * 8, 3, 2, 1, new LeakyReLUActivation<T>() as IActivationFunction<T>);
        h /= 2; w /= 2;

        // Bottleneck
        yield return new ConvolutionalLayer<T>(numFeatures * 8, h, w, numFeatures * 8, 3, 1, 1, new LeakyReLUActivation<T>() as IActivationFunction<T>);

        // Decoder (flow-agnostic reconstruction)
        yield return new ConvolutionalLayer<T>(numFeatures * 8, h, w, numFeatures * 4, 3, 1, 1, new LeakyReLUActivation<T>() as IActivationFunction<T>);
        h *= 2; w *= 2;
        yield return new ConvolutionalLayer<T>(numFeatures * 4, h, w, numFeatures * 2, 3, 1, 1, new LeakyReLUActivation<T>() as IActivationFunction<T>);
        h *= 2; w *= 2;
        yield return new ConvolutionalLayer<T>(numFeatures * 2, h, w, numFeatures, 3, 1, 1, new LeakyReLUActivation<T>() as IActivationFunction<T>);
        h *= 2; w *= 2;

        // Output (single interpolated frame)
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, inputChannels, 3, 1, 1);
    }

    /// <summary>
    /// Creates default layers for FlowFormer optical flow estimation.
    /// </summary>
    public static IEnumerable<ILayer<T>> CreateDefaultFlowFormerLayers(
        int inputChannels = 3,
        int inputHeight = 448,
        int inputWidth = 1024,
        int embedDim = 256,
        int numLayers = 6)
    {
        int h = inputHeight;
        int w = inputWidth;

        // CNN feature encoder (shared for both frames)
        yield return new ConvolutionalLayer<T>(inputChannels * 2, h, w, 64, 7, 2, 3, new ReLUActivation<T>() as IActivationFunction<T>);
        h /= 2; w /= 2;
        yield return new ConvolutionalLayer<T>(64, h, w, 128, 3, 2, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        h /= 2; w /= 2;
        yield return new ConvolutionalLayer<T>(128, h, w, embedDim, 3, 2, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        h /= 2; w /= 2;

        // Cost volume encoder
        yield return new ConvolutionalLayer<T>(embedDim, h, w, embedDim, 3, 1, 1, new GELUActivation<T>() as IActivationFunction<T>);

        // Transformer blocks for cost aggregation
        for (int i = 0; i < numLayers; i++)
        {
            yield return new ConvolutionalLayer<T>(embedDim, h, w, embedDim, 1, 1, 0, new GELUActivation<T>() as IActivationFunction<T>);
            yield return new ConvolutionalLayer<T>(embedDim, h, w, embedDim * 4, 1, 1, 0, new GELUActivation<T>() as IActivationFunction<T>);
            yield return new ConvolutionalLayer<T>(embedDim * 4, h, w, embedDim, 1, 1, 0);
        }

        // Flow decoder
        yield return new ConvolutionalLayer<T>(embedDim, h, w, 128, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(128, h, w, 64, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);

        // Flow head (2 channels: horizontal and vertical flow)
        yield return new ConvolutionalLayer<T>(64, h, w, 2, 3, 1, 1);
    }

    /// <summary>
    /// Creates default layers for ByteTrack multi-object tracking.
    /// </summary>
    public static IEnumerable<ILayer<T>> CreateDefaultByteTrackLayers(
        int inputChannels = 3,
        int inputHeight = 800,
        int inputWidth = 1440,
        int numFeatures = 256,
        int numClasses = 1)
    {
        int h = inputHeight;
        int w = inputWidth;

        // Backbone (CSPDarknet-style)
        yield return new ConvolutionalLayer<T>(inputChannels, h, w, 32, 3, 2, 1, new SiLUActivation<T>() as IActivationFunction<T>);
        h /= 2; w /= 2;
        yield return new ConvolutionalLayer<T>(32, h, w, 64, 3, 2, 1, new SiLUActivation<T>() as IActivationFunction<T>);
        h /= 2; w /= 2;
        yield return new ConvolutionalLayer<T>(64, h, w, 128, 3, 2, 1, new SiLUActivation<T>() as IActivationFunction<T>);
        h /= 2; w /= 2;
        yield return new ConvolutionalLayer<T>(128, h, w, numFeatures, 3, 2, 1, new SiLUActivation<T>() as IActivationFunction<T>);
        h /= 2; w /= 2;
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, numFeatures * 2, 3, 2, 1, new SiLUActivation<T>() as IActivationFunction<T>);
        h /= 2; w /= 2;

        // FPN neck for multi-scale features
        yield return new ConvolutionalLayer<T>(numFeatures * 2, h, w, numFeatures, 1, 1, 0, new SiLUActivation<T>() as IActivationFunction<T>);

        // Detection head (outputs: x, y, w, h, objectness, class)
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, numFeatures, 3, 1, 1, new SiLUActivation<T>() as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, 5 + numClasses, 1, 1, 0); // bbox + obj + classes
    }

    /// <summary>
    /// Creates default layers for DIFRINT video stabilization.
    /// </summary>
    public static IEnumerable<ILayer<T>> CreateDefaultDIFRINTLayers(
        int inputChannels = 3,
        int inputHeight = 480,
        int inputWidth = 640,
        int numFeatures = 64,
        int numIterations = 3)
    {
        int h = inputHeight;
        int w = inputWidth;

        // Motion estimation encoder
        yield return new ConvolutionalLayer<T>(inputChannels * 2, h, w, numFeatures, 7, 2, 3, new LeakyReLUActivation<T>() as IActivationFunction<T>);
        h /= 2; w /= 2;
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, numFeatures * 2, 3, 2, 1, new LeakyReLUActivation<T>() as IActivationFunction<T>);
        h /= 2; w /= 2;
        yield return new ConvolutionalLayer<T>(numFeatures * 2, h, w, numFeatures * 4, 3, 2, 1, new LeakyReLUActivation<T>() as IActivationFunction<T>);
        h /= 2; w /= 2;

        // Iterative refinement blocks
        for (int i = 0; i < numIterations; i++)
        {
            yield return new ConvolutionalLayer<T>(numFeatures * 4, h, w, numFeatures * 4, 3, 1, 1, new LeakyReLUActivation<T>() as IActivationFunction<T>);
            yield return new ConvolutionalLayer<T>(numFeatures * 4, h, w, numFeatures * 4, 3, 1, 1);
        }

        // Decoder for stabilized frame
        yield return new ConvolutionalLayer<T>(numFeatures * 4, h, w, numFeatures * 2, 3, 1, 1, new LeakyReLUActivation<T>() as IActivationFunction<T>);
        h *= 2; w *= 2;
        yield return new ConvolutionalLayer<T>(numFeatures * 2, h, w, numFeatures, 3, 1, 1, new LeakyReLUActivation<T>() as IActivationFunction<T>);
        h *= 2; w *= 2;
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, numFeatures, 3, 1, 1, new LeakyReLUActivation<T>() as IActivationFunction<T>);
        h *= 2; w *= 2;

        // Output frame
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, inputChannels, 3, 1, 1);
    }

    /// <summary>
    /// Creates default layers for RVM (Robust Video Matting).
    /// </summary>
    public static IEnumerable<ILayer<T>> CreateDefaultRVMLayers(
        int inputChannels = 3,
        int inputHeight = 512,
        int inputWidth = 512,
        int numFeatures = 32)
    {
        int h = inputHeight;
        int w = inputWidth;

        // Encoder (MobileNetV3-style)
        yield return new ConvolutionalLayer<T>(inputChannels, h, w, numFeatures, 3, 2, 1, new ReLU6Activation<T>() as IActivationFunction<T>);
        h /= 2; w /= 2;
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, numFeatures * 2, 3, 2, 1, new ReLU6Activation<T>() as IActivationFunction<T>);
        h /= 2; w /= 2;
        yield return new ConvolutionalLayer<T>(numFeatures * 2, h, w, numFeatures * 4, 3, 2, 1, new ReLU6Activation<T>() as IActivationFunction<T>);
        h /= 2; w /= 2;
        yield return new ConvolutionalLayer<T>(numFeatures * 4, h, w, numFeatures * 8, 3, 2, 1, new ReLU6Activation<T>() as IActivationFunction<T>);
        h /= 2; w /= 2;

        // Recurrent module (GRU-style for temporal consistency)
        yield return new ConvolutionalLayer<T>(numFeatures * 8, h, w, numFeatures * 8, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(numFeatures * 8, h, w, numFeatures * 8, 3, 1, 1, new SigmoidActivation<T>() as IActivationFunction<T>);

        // Decoder
        yield return new ConvolutionalLayer<T>(numFeatures * 8, h, w, numFeatures * 4, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        h *= 2; w *= 2;
        yield return new ConvolutionalLayer<T>(numFeatures * 4, h, w, numFeatures * 2, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        h *= 2; w *= 2;
        yield return new ConvolutionalLayer<T>(numFeatures * 2, h, w, numFeatures, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        h *= 2; w *= 2;
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, numFeatures, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        h *= 2; w *= 2;

        // Output heads: alpha matte (1 channel) + foreground (3 channels)
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, 4, 1, 1, 0); // RGBA output
    }

    /// <summary>
    /// Creates default layers for FastDVDNet video denoising.
    /// </summary>
    public static IEnumerable<ILayer<T>> CreateDefaultFastDVDNetLayers(
        int inputChannels = 3,
        int inputHeight = 480,
        int inputWidth = 854,
        int numFeatures = 32,
        int numInputFrames = 5)
    {
        int h = inputHeight;
        int w = inputWidth;

        // Stage 1: Multi-frame denoising blocks (process frames in pairs)
        int stage1Input = inputChannels * 3 + 1; // 3 frames + noise map
        yield return new ConvolutionalLayer<T>(stage1Input, h, w, numFeatures, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, numFeatures, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, numFeatures, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, numFeatures, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, inputChannels, 3, 1, 1);

        // Stage 2: Temporal fusion (combine stage 1 outputs)
        int stage2Input = inputChannels * 3 + 1; // 3 denoised frames + noise map
        yield return new ConvolutionalLayer<T>(stage2Input, h, w, numFeatures, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, numFeatures, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, numFeatures, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, numFeatures, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, inputChannels, 3, 1, 1);
    }


    #endregion
}

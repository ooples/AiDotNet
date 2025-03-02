using System.Runtime.InteropServices;

namespace AiDotNet.Helpers;

public static class LayerHelper<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

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

    private static void ValidateLayerParameters(int layerCount, int layerSize, int outputSize)
    {
        if (layerCount < 1)
            throw new ArgumentException($"Layer count must be at least 1.", nameof(layerCount));
        if (layerSize < 1)
            throw new ArgumentException($"Layer size must be at least 1.", nameof(layerSize));
        if (outputSize < 1)
            throw new ArgumentException("Output size must be at least 1.", nameof(outputSize));
    }

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
            activationFunction: new LinearActivation<T>()
        );

        // Final dense layer
        yield return new DenseLayer<T>(currentDepth, architecture.OutputSize, new SoftmaxActivation<T>() as IActivationFunction<T>);
    }

    private static IEnumerable<ILayer<T>> CreateResidualBlock(int inputDepth, int outputDepth, int height, int width, bool isFirstInBlock)
    {
        // Create the skip connection with the appropriate inner layer
        ILayer<T>? skipInnerLayer = null;
        if (isFirstInBlock && inputDepth != outputDepth)
        {
            skipInnerLayer = new ConvolutionalLayer<T>(
                inputDepth: inputDepth,
                outputDepth: outputDepth,
                kernelSize: 1,
                inputHeight: height,
                inputWidth: width,
                stride: 1,
                padding: 0,
                activation: new LinearActivation<T>()
            );
        }

        yield return new SkipConnectionLayer<T>(skipInnerLayer, new LinearActivation<T>() as IActivationFunction<T>);

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

        // Use LinearActivation for the AddLayer
        yield return new AddLayer<T>([[outputDepth, height, width]], new LinearActivation<T>() as IActivationFunction<T>);

        // Keep ReLU activation after addition
        yield return new ActivationLayer<T>([outputDepth, height, width], new ReLUActivation<T>() as IActivationFunction<T>);
    }

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
                yield return new ActivationLayer<T>([outputSize], new LinearActivation<T>() as IActivationFunction<T>);
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

    public static IEnumerable<ILayer<T>> CreateDefaultDeepBeliefNetworkLayers(NeuralNetworkArchitecture<T> architecture)
    {
        var rbmLayers = new List<RestrictedBoltzmannMachine<T>>();

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
            var rbm = new RestrictedBoltzmannMachine<T>(architecture, visibleUnits, hiddenUnits, sigmoidActivation);
            rbmLayers.Add(rbm);

            // Add dense layer and activation layer for each RBM
            yield return new DenseLayer<T>(visibleUnits, hiddenUnits, sigmoidActivation);
            yield return new ActivationLayer<T>([hiddenUnits], sigmoidActivation);
        }

        // Add the final output layer
        int outputSize = layerSizes[layerSizes.Length - 1];
        yield return new DenseLayer<T>(outputSize, outputSize, softmaxActivation);
        yield return new ActivationLayer<T>([outputSize], softmaxActivation);

        architecture.RbmLayers = rbmLayers;
    }

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
        yield return new DenseLayer<T>(defaultHiddenSize, actionSpace, new LinearActivation<T>() as IActivationFunction<T>);
        // No activation for the output layer as Q-values can be any real number
    }

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
        yield return new DenseLayer<T>(controllerSize, interfaceSize, new LinearActivation<T>() as IActivationFunction<T>);

        // Output layer
        yield return new DenseLayer<T>(controllerSize + readHeads * memoryWordSize, outputSize, new LinearActivation<T>() as IActivationFunction<T>);
        yield return new ActivationLayer<T>([outputSize], new SoftmaxActivation<T>() as IActivationFunction<T>);
    }

    public static IEnumerable<ILayer<T>> CreateDefaultESNLayers(int inputSize, int outputSize, int reservoirSize, double spectralRadius = 0.9, double sparsity = 0.1)
    {
        // Input to Reservoir connections (fixed random weights)
        yield return new DenseLayer<T>(inputSize, reservoirSize, new LinearActivation<T>() as IActivationFunction<T>);

        // Reservoir (recurrent connections, fixed random weights)
        yield return new ReservoirLayer<T>(reservoirSize, reservoirSize, spectralRadius: spectralRadius, connectionProbability: sparsity);

        // Reservoir activation
        yield return new ActivationLayer<T>([reservoirSize], new TanhActivation<T>() as IVectorActivationFunction<T>);

        // Output layer (Reservoir to output, trainable)
        yield return new DenseLayer<T>(reservoirSize, outputSize, new LinearActivation<T>() as IActivationFunction<T>);

        // Output activation (optional, depends on the problem)
        yield return new ActivationLayer<T>([outputSize], new LinearActivation<T>() as IActivationFunction<T>);
    }

    public static IEnumerable<ILayer<T>> CreateDefaultVAELayers(NeuralNetworkArchitecture<T> architecture, int latentSize)
    {
        var inputShape = architecture.GetInputShape();

        if (inputShape == null || inputShape.Length == 0)
        {
            throw new InvalidOperationException("Input shape must be specified for VAE.");
        }

        int inputSize = inputShape[0];

        // Encoder layers
        int[] encoderSizes = { inputSize / 2, inputSize / 4 };

        for (int i = 0; i < encoderSizes.Length; i++)
        {
            int outputSize = encoderSizes[i];
            yield return new DenseLayer<T>(inputSize, outputSize, new LeakyReLUActivation<T>() as IActivationFunction<T>);
            inputSize = outputSize;
        }

        // Latent space layers
        int encoderOutputSize = latentSize * 2; // For mean and log variance
        yield return new DenseLayer<T>(inputSize, encoderOutputSize, new LinearActivation<T>() as IActivationFunction<T>);

        // Mean and LogVariance layers
        yield return new MeanLayer<T>([encoderOutputSize], axis: 0);
        yield return new LogVarianceLayer<T>([encoderOutputSize], axis: 0);

        // Add a SamplingLayer before the latent space
        yield return new SamplingLayer<T>([1, inputSize, 1], 2, 2, SamplingType.Average);

        // Decoder layers
        int[] decoderSizes = [.. encoderSizes.Reverse()];
        inputSize = latentSize;

        for (int i = 0; i < decoderSizes.Length; i++)
        {
            int outputSize = decoderSizes[i];
            yield return new DenseLayer<T>(inputSize, outputSize, new LeakyReLUActivation<T>() as IActivationFunction<T>);
            inputSize = outputSize;
        }

        // Output layer
        yield return new DenseLayer<T>(inputSize, inputShape[0], new SigmoidActivation<T>() as IActivationFunction<T>);
    }

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
                activationFunction: new LinearActivation<T>());
            
            // Add normalization
            yield return new LayerNormalizationLayer<T>(modelDimension);
        
            // Add dropout if specified
            if (dropoutRate > 0)
            {
                yield return new DropoutLayer<T>(dropoutRate);
            }
        
            // Feed-forward network
            yield return new DenseLayer<T>(modelDimension, feedForwardDimension, new ReLUActivation<T>() as IActivationFunction<T>);
            yield return new DenseLayer<T>(feedForwardDimension, modelDimension, new LinearActivation<T>() as IActivationFunction<T>);
        
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
                    activationFunction: new LinearActivation<T>());
                
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
                    activationFunction: new LinearActivation<T>());
                
                // Add normalization
                yield return new LayerNormalizationLayer<T>(modelDimension);
            
                // Add dropout if specified
                if (dropoutRate > 0)
                {
                    yield return new DropoutLayer<T>(dropoutRate);
                }
            
                // Feed-forward network
                yield return new DenseLayer<T>(modelDimension, feedForwardDimension, new ReLUActivation<T>() as IActivationFunction<T>);
                yield return new DenseLayer<T>(feedForwardDimension, modelDimension, new LinearActivation<T>() as IActivationFunction<T>);
            
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
        yield return new DenseLayer<T>(modelDimension, outputSize, new LinearActivation<T>() as IActivationFunction<T>);
    
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
                yield return new ActivationLayer<T>([outputSize], new LinearActivation<T>() as IActivationFunction<T>);
                break;
            
            case NeuralNetworkTaskType.TextGeneration:
                if (temperature != 1.0)
                {
                    yield return new LambdaLayer<T>(
                        [outputSize], 
                        [outputSize],
                        input => input.Scale(NumOps.FromDouble(1.0 / temperature)),
                        (input, gradient) => gradient.Scale(NumOps.FromDouble(temperature)),
                        new LinearActivation<T>() as IActivationFunction<T>);
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
                new LinearActivation<T>() as IActivationFunction<T>
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
                yield return new ActivationLayer<T>([outputSize], new LinearActivation<T>() as IActivationFunction<T>);
            }
        }
    }

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
        yield return new ActivationLayer<T>([outputSize], new LinearActivation<T>() as IActivationFunction<T>);
    }

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
            activationFunction: new LinearActivation<T>()
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
            yield return new ActivationLayer<T>([outputSize], new LinearActivation<T>() as IVectorActivationFunction<T>);
        }
    }

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
                timeDistributedActivation = new LinearActivation<T>();
            }
            
            yield return new TimeDistributedLayer<T>(
                new DenseLayer<T>(
                    hiddenSize, 
                    outputSize, 
                    new LinearActivation<T>() as IActivationFunction<T>
                ), timeDistributedActivation
            );
        }
        else
        {
            // Standard dense output layer for other tasks
            yield return new DenseLayer<T>(
                hiddenSize, 
                outputSize, 
                new LinearActivation<T>() as IActivationFunction<T>
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
            yield return new ActivationLayer<T>([outputSize], new LinearActivation<T>() as IVectorActivationFunction<T>);
        }
    }

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
        yield return new DenseLayer<T>(columnCount * cellsPerColumn, outputSize, new LinearActivation<T>() as IActivationFunction<T>);
        yield return new ActivationLayer<T>([outputSize], new SoftmaxActivation<T>() as IActivationFunction<T>);
    }

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
            activationFunction: new LinearActivation<T>()
        );

        // Add the final Activation Layer (typically Softmax for classification tasks)
        yield return new ActivationLayer<T>([outputSize], new SoftmaxActivation<T>() as IActivationFunction<T>);
    }

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
            yield return new ActivationLayer<T>(new[] { outputSize }, new SigmoidActivation<T>() as IVectorActivationFunction<T>);
        }
        else if (architecture.TaskType == NeuralNetworkTaskType.MultiClassClassification)
        {
            yield return new ActivationLayer<T>(new[] { outputSize }, new SoftmaxActivation<T>() as IVectorActivationFunction<T>);
        }
        else
        {
            // For regression tasks
            yield return new ActivationLayer<T>(new[] { outputSize }, new LinearActivation<T>() as IVectorActivationFunction<T>);
        }
    }

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
            yield return new ActivationLayer<T>([outputSize], new LinearActivation<T>() as IVectorActivationFunction<T>);
        }
    }

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
            new LinearActivation<T>() as IActivationFunction<T>);
    
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
            yield return new ActivationLayer<T>(new[] { outputSize }, new LinearActivation<T>() as IVectorActivationFunction<T>);
        }
    }

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

        // Create input layer to first hidden layer
        int firstHiddenLayerSize = hiddenLayerSizes.Count > 0 ? hiddenLayerSizes[0] : outputSize;
        yield return new DenseLayer<T>(inputSize, firstHiddenLayerSize, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new ActivationLayer<T>(new[] { firstHiddenLayerSize }, new ReLUActivation<T>() as IActivationFunction<T>);

        // Create hidden layers
        for (int i = 0; i < hiddenLayerSizes.Count - 1; i++)
        {
            int currentLayerSize = hiddenLayerSizes[i];
            int nextLayerSize = hiddenLayerSizes[i + 1];
        
            yield return new DenseLayer<T>(currentLayerSize, nextLayerSize, new ReLUActivation<T>() as IActivationFunction<T>);
            yield return new ActivationLayer<T>(new[] { nextLayerSize }, new ReLUActivation<T>() as IActivationFunction<T>);
        }

        // Create last hidden layer to output layer
        if (hiddenLayerSizes.Count > 0)
        {
            int lastHiddenLayerSize = hiddenLayerSizes[hiddenLayerSizes.Count - 1];
            yield return new DenseLayer<T>(lastHiddenLayerSize, outputSize, new SoftmaxActivation<T>() as IActivationFunction<T>);
        }
        else
        {
            // If no hidden layers, connect input directly to output
            yield return new DenseLayer<T>(inputSize, outputSize, new SoftmaxActivation<T>() as IActivationFunction<T>);
        }
    
        // Final activation layer for output
        yield return new ActivationLayer<T>(new[] { outputSize }, new SoftmaxActivation<T>() as IActivationFunction<T>);
    }

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
        yield return new DenseLayer<T>(reservoirSize, outputSize, new LinearActivation<T>() as IActivationFunction<T>);

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
            yield return new ActivationLayer<T>([outputSize], new LinearActivation<T>() as IActivationFunction<T>);
        }

        // Add the final Activation Layer (typically Softmax for classification tasks)
        yield return new ActivationLayer<T>([outputSize], new SoftmaxActivation<T>() as IActivationFunction<T>);
    }

    public static IEnumerable<ILayer<T>> CreateDefaultLSTMNetworkLayers(NeuralNetworkArchitecture<T> architecture)
    {
        var inputShape = architecture.GetInputShape();
        int inputSize = inputShape[0];
        int outputSize = architecture.OutputSize;
    
        // Calculate hidden layer sizes based on network complexity
        int hiddenSize;
        int numLayers;
    
        switch (architecture.Complexity)
        {
            case NetworkComplexity.Simple:
                hiddenSize = Math.Max(32, inputSize);
                numLayers = 1;
                break;
            case NetworkComplexity.Medium:
                hiddenSize = Math.Max(64, inputSize * 2);
                numLayers = 2;
                break;
            case NetworkComplexity.Deep:
                hiddenSize = Math.Max(128, inputSize * 3);
                numLayers = 3;
                break;
            default:
                hiddenSize = Math.Max(64, inputSize);
                numLayers = 2;
                break;
        }

        // Input layer
        yield return new InputLayer<T>(inputSize);
    
        // LSTM layers
        int currentInputSize = inputSize;
    
        for (int i = 0; i < numLayers; i++)
        {
            // For deeper networks, gradually decrease the hidden size
            int layerHiddenSize = i == numLayers - 1 ? 
                Math.Max(outputSize, hiddenSize / 2) : 
                hiddenSize;
            
            // Add LSTM Layer
            yield return new LSTMLayer<T>(
                inputSize: currentInputSize,
                hiddenSize: layerHiddenSize,
                activation: new TanhActivation<T>() as IActivationFunction<T>,
                recurrentActivation: new SigmoidActivation<T>()
            );
        
            // Add Activation Layer after LSTM
            yield return new ActivationLayer<T>([layerHiddenSize], new TanhActivation<T>() as IActivationFunction<T>);
        
            currentInputSize = layerHiddenSize;
        }

        // Add the final Dense Layer
        yield return new DenseLayer<T>(
            inputSize: currentInputSize, 
            outputSize: outputSize,
            activationFunction: new LinearActivation<T>()
        );

        // Add the final Activation Layer based on task type
        if (architecture.TaskType == NeuralNetworkTaskType.MultiClassClassification)
        {
            yield return new ActivationLayer<T>([outputSize], new SoftmaxActivation<T>() as IVectorActivationFunction<T>);
        }
        else if (architecture.TaskType == NeuralNetworkTaskType.BinaryClassification)
        {
            yield return new ActivationLayer<T>([outputSize], new SigmoidActivation<T>() as IActivationFunction<T>);
        }
        else // Regression or default
        {
            yield return new ActivationLayer<T>([outputSize], new LinearActivation<T>() as IActivationFunction<T>);
        }
    }
}
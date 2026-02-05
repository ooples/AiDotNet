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
    /// Creates the default layer configuration for a Deep Portfolio Management model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="numAssets">The number of assets in the portfolio.</param>
    /// <returns>A collection of layers forming the portfolio optimization network.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Deep Portfolio models use neural networks to directly
    /// output optimal asset weights. This method uses a Softmax activation at the
    /// end to ensure weights sum to 100%, mimicking a fully invested portfolio.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultDeepPortfolioLayers(
        NeuralNetworkArchitecture<T> architecture,
        int numAssets)
    {
        // 2-layer MLP with Softmax output for weight allocation
        yield return new DenseLayer<T>(architecture.CalculatedInputSize, 64, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new DenseLayer<T>(64, numAssets, new SoftmaxActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Creates the default layer configuration for a Neural Value-at-Risk (VaR) model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="numFeatures">The number of input features (market factors).</param>
    /// <returns>A collection of layers forming the Neural VaR network.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Neural VaR models use deep learning to map market
    /// conditions to the potential loss of a portfolio. It learns complex,
    /// non-linear relationships that traditional statistical VaR might miss.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultNeuralVaRLayers(
        NeuralNetworkArchitecture<T> architecture,
        int numFeatures)
    {
        // Deep MLP for risk mapping
        return CreateDefaultLayers(architecture, 3, 128, 1);
    }

    /// <summary>
    /// Creates the default layer configuration for a Market Making agent.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="stateSize">The size of the market state vector.</param>
    /// <param name="actionSize">The size of the action vector (typically bid/ask spreads).</param>
    /// <param name="hiddenSize">The size of hidden layers.</param>
    /// <returns>A collection of layers forming the market making policy network.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Market making agents need to process market depth,
    /// order flow, and their own inventory to set optimal bid and ask prices.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultMarketMakingLayers(
        NeuralNetworkArchitecture<T> architecture,
        int stateSize,
        int actionSize,
        int hiddenSize = 64)
    {
        // Simple MLP for market making policy
        return CreateDefaultLayers(architecture, 2, hiddenSize, actionSize);
    }

    /// <summary>
    /// Creates the default layer configuration for InvestLM.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="maxSequenceLength">Maximum sequence length.</param>
    /// <param name="vocabularySize">Size of the vocabulary.</param>
    /// <param name="hiddenSize">Hidden layer size.</param>
    /// <param name="numAttentionHeads">Number of attention heads.</param>
    /// <param name="numHiddenLayers">Number of transformer layers.</param>
    /// <param name="dropoutProbability">Dropout probability.</param>
    /// <returns>A collection of layers forming InvestLM.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> InvestLM uses a decoder-only transformer architecture
    /// fine-tuned on high-quality investment research and financial analysis.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultInvestLMLayers(
        NeuralNetworkArchitecture<T> architecture,
        int maxSequenceLength = 1024,
        int vocabularySize = 32000,
        int hiddenSize = 768,
        int numAttentionHeads = 12,
        int numHiddenLayers = 12,
        double dropoutProbability = 0.1)
    {
        // Follows the GPT/LLaMA-style decoder-only pattern
        return CreateDefaultFinGPTLayers(architecture, maxSequenceLength, vocabularySize, 
            hiddenSize, numAttentionHeads, numHiddenLayers, dropoutProbability);
    }

    /// <summary>
    /// Creates the default layer configuration for FinMA.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="maxSequenceLength">Maximum sequence length.</param>
    /// <param name="vocabularySize">Size of the vocabulary.</param>
    /// <param name="numAgents">Number of specialized agents.</param>
    /// <param name="hiddenSize">Hidden layer size.</param>
    /// <param name="numAttentionHeads">Number of attention heads.</param>
    /// <param name="numHiddenLayers">Number of transformer layers.</param>
    /// <param name="dropoutProbability">Dropout probability.</param>
    /// <returns>A collection of layers forming FinMA.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> FinMA (Financial Multi-Agent) uses a set of specialized
    /// transformer models that collaborate. This method initializes the base
    /// transformer architecture used by the agents.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultFinMALayers(
        NeuralNetworkArchitecture<T> architecture,
        int maxSequenceLength = 512,
        int vocabularySize = 32000,
        int numAgents = 4,
        int hiddenSize = 768,
        int numAttentionHeads = 12,
        int numHiddenLayers = 12,
        double dropoutProbability = 0.1)
    {
        // Follows a LLaMA-style decoder-only pattern for each agent
        // In this implementation, we return the layers for a single representative agent
        return CreateDefaultFinGPTLayers(architecture, maxSequenceLength, vocabularySize, 
            hiddenSize, numAttentionHeads, numHiddenLayers, dropoutProbability);
    }

    /// <summary>
    /// Creates the default layer configuration for BloombergGPT.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="maxSequenceLength">Maximum sequence length.</param>
    /// <param name="vocabularySize">Size of the vocabulary.</param>
    /// <param name="hiddenSize">Hidden layer size.</param>
    /// <param name="numAttentionHeads">Number of attention heads.</param>
    /// <param name="numHiddenLayers">Number of transformer layers.</param>
    /// <param name="dropoutProbability">Dropout probability.</param>
    /// <returns>A collection of layers forming BloombergGPT.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> BloombergGPT uses a massive decoder-only transformer
    /// architecture (like GPT-3) trained on Bloomberg's extensive financial data
    /// archives plus general web data.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultBloombergGPTLayers(
        NeuralNetworkArchitecture<T> architecture,
        int maxSequenceLength = 2048,
        int vocabularySize = 131072,
        int hiddenSize = 768,
        int numAttentionHeads = 12,
        int numHiddenLayers = 12,
        double dropoutProbability = 0.1)
    {
        // Follows standard GPT pattern but with ALiBi or Rotary positional embeddings usually
        // For simplicity in the library, we'll use learned positional embeddings
        return CreateDefaultFinGPTLayers(architecture, maxSequenceLength, vocabularySize, 
            hiddenSize, numAttentionHeads, numHiddenLayers, dropoutProbability);
    }

    /// <summary>
    /// Creates the default layer configuration for FinGPT.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="maxSequenceLength">Maximum sequence length.</param>
    /// <param name="vocabularySize">Size of the vocabulary.</param>
    /// <param name="hiddenSize">Hidden layer size.</param>
    /// <param name="numAttentionHeads">Number of attention heads.</param>
    /// <param name="numHiddenLayers">Number of transformer layers.</param>
    /// <param name="dropoutProbability">Dropout probability.</param>
    /// <returns>A collection of layers forming FinGPT.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> FinGPT uses a GPT (Generative Pre-trained Transformer)
    /// architecture optimized for financial text generation and market analysis.
    /// Unlike BERT (which is an encoder), GPT is a decoder-only model designed to
    /// predict the next word in a sequence.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultFinGPTLayers(
        NeuralNetworkArchitecture<T> architecture,
        int maxSequenceLength = 1024,
        int vocabularySize = 50257,
        int hiddenSize = 768,
        int numAttentionHeads = 12,
        int numHiddenLayers = 12,
        double dropoutProbability = 0.1)
    {
        // 1. Embedding Layers
        yield return new EmbeddingLayer<T>(vocabularySize, hiddenSize);
        yield return new PositionalEncodingLayer<T>(maxSequenceLength, hiddenSize);
        
        yield return new DropoutLayer<T>(dropoutProbability);

        // 2. Transformer Decoder Blocks
        for (int i = 0; i < numHiddenLayers; i++)
        {
            // Masked self-attention (standard MultiHeadAttention handles masking via causal mask)
            yield return new MultiHeadAttentionLayer<T>(maxSequenceLength, hiddenSize, numAttentionHeads, (IActivationFunction<T>?)null);
            yield return new LayerNormalizationLayer<T>(hiddenSize);

            // Feed-forward
            yield return new DenseLayer<T>(hiddenSize, hiddenSize * 4, new GELUActivation<T>() as IActivationFunction<T>);
            yield return new DenseLayer<T>(hiddenSize * 4, hiddenSize);
            yield return new LayerNormalizationLayer<T>(hiddenSize);
            yield return new DropoutLayer<T>(dropoutProbability);
        }

        // 3. Final Norm
        yield return new LayerNormalizationLayer<T>(hiddenSize);

        // 4. Output Head (predict next token from vocabulary)
        yield return new DenseLayer<T>(hiddenSize, vocabularySize);
    }

    /// <summary>
    /// Creates the default layer configuration for FinBERT-tone.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="maxSequenceLength">Maximum sequence length.</param>
    /// <param name="vocabularySize">Size of the vocabulary.</param>
    /// <param name="numClasses">Number of sentiment classes.</param>
    /// <param name="hiddenSize">Hidden layer size.</param>
    /// <param name="numAttentionHeads">Number of attention heads.</param>
    /// <param name="numHiddenLayers">Number of transformer layers.</param>
    /// <param name="dropoutProbability">Dropout probability.</param>
    /// <returns>A collection of layers forming FinBERT-tone.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> FinBERT-tone uses the BERT architecture with a specific
    /// classification head designed to detect the tone or sentiment of financial text.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultFinBERTToneLayers(
        NeuralNetworkArchitecture<T> architecture,
        int maxSequenceLength = 512,
        int vocabularySize = 30522,
        int numClasses = 3,
        int hiddenSize = 768,
        int numAttentionHeads = 12,
        int numHiddenLayers = 12,
        double dropoutProbability = 0.1)
    {
        // Word embeddings
        yield return new EmbeddingLayer<T>(vocabularySize, hiddenSize);
        // Positional embeddings
        yield return new PositionalEncodingLayer<T>(maxSequenceLength, hiddenSize);
        // Type embeddings
        yield return new EmbeddingLayer<T>(2, hiddenSize);
        
        yield return new LayerNormalizationLayer<T>(hiddenSize);
        yield return new DropoutLayer<T>(dropoutProbability);

        for (int i = 0; i < numHiddenLayers; i++)
        {
            yield return new MultiHeadAttentionLayer<T>(maxSequenceLength, hiddenSize, numAttentionHeads, (IActivationFunction<T>?)null);
            yield return new LayerNormalizationLayer<T>(hiddenSize);

            yield return new DenseLayer<T>(hiddenSize, hiddenSize * 4, new GELUActivation<T>() as IActivationFunction<T>);
            yield return new DenseLayer<T>(hiddenSize * 4, hiddenSize);
            yield return new LayerNormalizationLayer<T>(hiddenSize);
            yield return new DropoutLayer<T>(dropoutProbability);
        }

        // Pooler
        yield return new DenseLayer<T>(hiddenSize, hiddenSize, new TanhActivation<T>() as IActivationFunction<T>);

        // Classification Head for Tone/Sentiment
        yield return new DenseLayer<T>(hiddenSize, numClasses, new SoftmaxActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Creates the default layer configuration for FinancialBERT.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="maxSequenceLength">Maximum sequence length.</param>
    /// <param name="vocabularySize">Size of the vocabulary.</param>
    /// <param name="hiddenSize">Hidden layer size.</param>
    /// <param name="numAttentionHeads">Number of attention heads.</param>
    /// <param name="numHiddenLayers">Number of transformer layers.</param>
    /// <param name="dropoutProbability">Dropout probability.</param>
    /// <returns>A collection of layers forming FinancialBERT.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> FinancialBERT uses the standard BERT architecture but is
    /// domain-adapted through training on a large corpus of financial text.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultFinancialBERTLayers(
        NeuralNetworkArchitecture<T> architecture,
        int maxSequenceLength = 512,
        int vocabularySize = 30522,
        int hiddenSize = 768,
        int numAttentionHeads = 12,
        int numHiddenLayers = 12,
        double dropoutProbability = 0.1)
    {
        // Re-use standard BERT pattern
        return CreateDefaultSECBERTLayers(architecture, maxSequenceLength, vocabularySize, 
            hiddenSize, numAttentionHeads, numHiddenLayers, dropoutProbability);
    }

    /// <summary>
    /// Creates the default layer configuration for SEC-BERT.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="maxSequenceLength">Maximum sequence length.</param>
    /// <param name="vocabularySize">Size of the vocabulary.</param>
    /// <param name="hiddenSize">Hidden layer size.</param>
    /// <param name="numAttentionHeads">Number of attention heads.</param>
    /// <param name="numHiddenLayers">Number of transformer layers.</param>
    /// <param name="dropoutProbability">Dropout probability.</param>
    /// <returns>A collection of layers forming SEC-BERT.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> SEC-BERT is based on the BERT (Bidirectional Encoder
    /// Representations from Transformers) architecture. It works by:
    /// </para>
    /// <para>
    /// <list type="number">
    /// <item><b>Embeddings:</b> Converting words into numerical vectors that
    /// capture their financial meaning from SEC filings.</item>
    /// <item><b>Transformer Layers:</b> Using attention mechanisms to understand
    /// the relationships between words in a sentence, regardless of their position.</item>
    /// <item><b>Task Head:</b> Converting the final representations into predictions
    /// for specific financial tasks like sentiment analysis or risk classification.</item>
    /// </list>
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultSECBERTLayers(
        NeuralNetworkArchitecture<T> architecture,
        int maxSequenceLength = 512,
        int vocabularySize = 30522,
        int hiddenSize = 768,
        int numAttentionHeads = 12,
        int numHiddenLayers = 12,
        double dropoutProbability = 0.1)
    {
        // 1. Embedding Layers
        // Word embeddings
        yield return new EmbeddingLayer<T>(vocabularySize, hiddenSize);
        // Positional embeddings
        yield return new PositionalEncodingLayer<T>(maxSequenceLength, hiddenSize);
        // Type embeddings
        yield return new EmbeddingLayer<T>(2, hiddenSize);
        
        // Add & Norm for embeddings
        yield return new LayerNormalizationLayer<T>(hiddenSize);
        yield return new DropoutLayer<T>(dropoutProbability);

        // 2. Transformer Blocks
        for (int i = 0; i < numHiddenLayers; i++)
        {
            // Self-attention
            yield return new MultiHeadAttentionLayer<T>(maxSequenceLength, hiddenSize, numAttentionHeads, (IActivationFunction<T>?)null);
            yield return new LayerNormalizationLayer<T>(hiddenSize);

            // Feed-forward
            yield return new DenseLayer<T>(hiddenSize, hiddenSize * 4, new GELUActivation<T>() as IActivationFunction<T>);
            yield return new DenseLayer<T>(hiddenSize * 4, hiddenSize);
            yield return new LayerNormalizationLayer<T>(hiddenSize);
            yield return new DropoutLayer<T>(dropoutProbability);
        }

        // 3. Pooler
        yield return new DenseLayer<T>(hiddenSize, hiddenSize, new TanhActivation<T>() as IActivationFunction<T>);

        // 4. Task Head (default to 1 for regression or 2 for binary classification)
        yield return new DenseLayer<T>(hiddenSize, 1);
    }

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

        // Input layer defines the expected input shape for the network.
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

        // QuantumLayer outputs 2^numQubits probability values, not the configured outputSize
        int quantumDim = 1 << numQubits; // 2^numQubits
        int hiddenSize = Math.Max(32, Math.Max(inputSize, outputSize));

        // Input layer
        yield return new InputLayer<T>(inputSize);

        // First quantum layer with measurement
        // QuantumLayer outputs quantumDim values regardless of its outputSize parameter
        yield return new QuantumLayer<T>(inputSize, quantumDim, numQubits);
        yield return new MeasurementLayer<T>(quantumDim);

        // Add a dense layer after measurement to project to hiddenSize
        yield return new DenseLayer<T>(
            inputSize: quantumDim,
            outputSize: hiddenSize,
            activationFunction: new ReLUActivation<T>()
        );

        // Second quantum layer with measurement
        yield return new QuantumLayer<T>(hiddenSize, quantumDim, numQubits);
        yield return new MeasurementLayer<T>(quantumDim);

        // Final dense layer to map to output size
        yield return new DenseLayer<T>(
            inputSize: quantumDim,
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
    /// This creates a trainable model structure from scratch. The decoder layers expect encoder
    /// outputs to be provided during the forward pass (as implemented in <see cref="AiDotNet.Audio.Whisper.WhisperModel{T}"/>).
    /// For inference with pre-trained weights, use the ONNX-based WhisperModel.CreateAsync() method instead.
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
        foreach (var layer in CreateDefaultWhisperLayers(
            modelDim: modelDimension,
            numEncoderLayers: numEncoderLayers,
            numDecoderLayers: numDecoderLayers,
            numHeads: numHeads,
            ffDim: feedForwardDim,
            numMels: numMels,
            maxFrames: maxSequenceLength,
            maxTokens: maxSequenceLength,
            vocabSize: vocabularySize,
            dropoutRate: dropoutRate))
        {
            yield return layer;
        }
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

        // Decoder (upsampling path) - symmetric with encoder
        for (int i = channelMults.Length - 1; i >= 0; i--)
        {
            int outDim = i > 0 ? embedDim * channelMults[i - 1] : embedDim;

            // Two residual-style conv blocks per level
            yield return new ConvolutionalLayer<T>(currentDim, h, w, currentDim, 3, 1, 1, new SiLUActivation<T>() as IActivationFunction<T>);
            yield return new ConvolutionalLayer<T>(currentDim, h, w, outDim, 3, 1, 1, new SiLUActivation<T>() as IActivationFunction<T>);

            currentDim = outDim;

            // Upsample only at levels that downsampled in encoder (mult != last element)
            // This ensures symmetric encoder/decoder geometry for proper U-Net denoising
            if (i > 0 && channelMults[i - 1] != channelMults[^1])
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
        yield return new UpsamplingLayer<T>([128, h, w], 2);
        h *= 2; w *= 2;
        yield return new ConvolutionalLayer<T>(128, h, w, 64, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new UpsamplingLayer<T>([64, h, w], 2);
        h *= 2; w *= 2;
        yield return new ConvolutionalLayer<T>(64, h, w, 32, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new UpsamplingLayer<T>([32, h, w], 2);
        h *= 2; w *= 2;
        yield return new ConvolutionalLayer<T>(32, h, w, 16, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new UpsamplingLayer<T>([16, h, w], 2);
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
        yield return new UpsamplingLayer<T>([128, h, w], 2);
        h *= 2; w *= 2;
        yield return new ConvolutionalLayer<T>(128, h, w, 64, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new UpsamplingLayer<T>([64, h, w], 2);
        h *= 2; w *= 2;
        yield return new ConvolutionalLayer<T>(64, h, w, 32, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new UpsamplingLayer<T>([32, h, w], 2);
        h *= 2; w *= 2;
        yield return new ConvolutionalLayer<T>(32, h, w, 16, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new UpsamplingLayer<T>([16, h, w], 2);
        h *= 2; w *= 2;

        // Final mask head
        yield return new ConvolutionalLayer<T>(16, h, w, 1, 3, 1, 1, new SigmoidActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Creates the image encoder layers for SAM2 (Segment Anything Model 2).
    /// </summary>
    /// <param name="inputChannels">Number of input channels (default: 3 for RGB).</param>
    /// <param name="inputHeight">Input height (default: 1024).</param>
    /// <param name="inputWidth">Input width (default: 1024).</param>
    /// <param name="numFeatures">Number of output feature channels (default: 256).</param>
    /// <returns>Image encoder layers that downsample input to feature maps.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates the image encoder part of SAM2, which processes
    /// input images into feature maps. The output has shape [numFeatures, H/16, W/16].
    /// </para>
    /// <para>
    /// <b>Note:</b> SAM2 is a multi-branch architecture. Use separate factory methods:
    /// - CreateSAM2ImageEncoderLayers: Image feature extraction (this method)
    /// - CreateSAM2PromptEncoderLayers: Point/box/mask prompt encoding
    /// - CreateSAM2MemoryLayers: Temporal memory attention
    /// - CreateSAM2MaskDecoderLayers: Mask prediction head
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateSAM2ImageEncoderLayers(
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
    }

    /// <summary>
    /// Creates the prompt encoder layers for SAM2 (point, box, and mask prompts).
    /// </summary>
    /// <param name="numFeatures">Number of output feature channels (default: 256).</param>
    /// <param name="maskHeight">Height of mask prompt input (default: 256).</param>
    /// <param name="maskWidth">Width of mask prompt input (default: 256).</param>
    /// <returns>Prompt encoder layers for different prompt types.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> SAM2 accepts different types of prompts to tell it what to segment:
    /// - Points: Click on the object (x, y coordinates)
    /// - Boxes: Draw a bounding box (x1, y1, x2, y2)
    /// - Masks: Provide an initial mask estimate
    /// </para>
    /// <para>
    /// <b>Usage:</b> These layers are applied to prompt inputs separately, then combined
    /// with image features in the mask decoder. They are NOT chained sequentially with
    /// the image encoder.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateSAM2PromptEncoderLayers(
        int numFeatures = 256,
        int maskHeight = 256,
        int maskWidth = 256)
    {
        // Point prompt encoder: input [2] (x, y) -> [numFeatures]
        yield return new DenseLayer<T>(2, numFeatures, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new DenseLayer<T>(numFeatures, numFeatures, new ReLUActivation<T>() as IActivationFunction<T>);

        // Box prompt encoder: input [4] (x1, y1, x2, y2) -> [numFeatures]
        yield return new DenseLayer<T>(4, numFeatures, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new DenseLayer<T>(numFeatures, numFeatures, new ReLUActivation<T>() as IActivationFunction<T>);

        // Mask prompt encoder: input [1, maskHeight, maskWidth] -> [numFeatures/4, maskHeight/4, maskWidth/4]
        yield return new ConvolutionalLayer<T>(1, maskHeight, maskWidth, numFeatures / 4, 4, 4, 0, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(numFeatures / 4, maskHeight / 4, maskWidth / 4, numFeatures / 4, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Creates the memory attention layers for SAM2 temporal consistency.
    /// </summary>
    /// <param name="numFeatures">Number of feature channels (default: 256).</param>
    /// <param name="featureHeight">Height of feature maps (default: 64).</param>
    /// <param name="featureWidth">Width of feature maps (default: 64).</param>
    /// <returns>Memory attention layers for video object tracking.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Memory layers help SAM2 track objects across video frames
    /// by maintaining a memory of past segmentations and matching them to new frames.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateSAM2MemoryLayers(
        int numFeatures = 256,
        int featureHeight = 64,
        int featureWidth = 64)
    {
        int h = featureHeight;
        int w = featureWidth;

        // Memory attention: fuses current features with memory
        yield return new ConvolutionalLayer<T>(numFeatures * 2, h, w, numFeatures, 1, 1, 0, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, numFeatures, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(numFeatures * 2, h, w, numFeatures, 1, 1, 0, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, numFeatures, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);

        // Memory projection
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, numFeatures, 1, 1, 0, new ReLUActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Creates the shared mask decoder refinement layers for SAM2.
    /// </summary>
    /// <param name="numFeatures">Number of feature channels (default: 256).</param>
    /// <param name="featureHeight">Height of feature maps (default: 64).</param>
    /// <param name="featureWidth">Width of feature maps (default: 64).</param>
    /// <returns>Shared refinement layers that process fused features.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> These layers refine the combined image and prompt features
    /// before branching into separate prediction heads. Output shape: [numFeatures, h, w]
    /// </para>
    /// <para>
    /// <b>Usage:</b> Apply these layers first, then branch to the three separate heads:
    /// - CreateSAM2MaskHead: Produces mask candidates
    /// - CreateSAM2IoUHead: Predicts mask quality scores
    /// - CreateSAM2OcclusionHead: Predicts occlusion
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateSAM2MaskDecoderLayers(
        int numFeatures = 256,
        int featureHeight = 64,
        int featureWidth = 64)
    {
        int h = featureHeight;
        int w = featureWidth;

        // Mask decoder refinement - shared feature processing
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, numFeatures, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, numFeatures, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Creates the mask prediction head for SAM2.
    /// </summary>
    /// <param name="numFeatures">Number of input feature channels (default: 256).</param>
    /// <param name="featureHeight">Height of feature maps (default: 64).</param>
    /// <param name="featureWidth">Width of feature maps (default: 64).</param>
    /// <param name="numMaskCandidates">Number of mask candidates to output (default: 4).</param>
    /// <returns>Mask prediction layers. Output shape: [numMaskCandidates, h, w]</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This head produces multiple candidate segmentation masks.
    /// Each candidate is a probability map indicating object presence at each pixel.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateSAM2MaskHead(
        int numFeatures = 256,
        int featureHeight = 64,
        int featureWidth = 64,
        int numMaskCandidates = 4)
    {
        // Input: [numFeatures, h, w] from CreateSAM2MaskDecoderLayers
        // Output: [numMaskCandidates, h, w] mask probability maps
        yield return new ConvolutionalLayer<T>(numFeatures, featureHeight, featureWidth, numMaskCandidates, 1, 1, 0, new SigmoidActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Creates the IoU (Intersection over Union) prediction head for SAM2.
    /// </summary>
    /// <param name="numFeatures">Number of input feature channels (default: 256).</param>
    /// <param name="featureHeight">Height of feature maps (default: 64).</param>
    /// <param name="featureWidth">Width of feature maps (default: 64).</param>
    /// <param name="numMaskCandidates">Number of mask candidates (default: 4).</param>
    /// <returns>IoU prediction layers. Output shape: [numMaskCandidates]</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This head predicts the quality (IoU score) for each mask candidate.
    /// Higher scores indicate better masks. Used to select the best mask from candidates.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateSAM2IoUHead(
        int numFeatures = 256,
        int featureHeight = 64,
        int featureWidth = 64,
        int numMaskCandidates = 4)
    {
        // Input: [numFeatures, h, w] from CreateSAM2MaskDecoderLayers
        // Global pool then predict IoU for each mask candidate
        yield return new GlobalPoolingLayer<T>([numFeatures, featureHeight, featureWidth], PoolingType.Average);
        yield return new DenseLayer<T>(numFeatures, numMaskCandidates, new SigmoidActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Creates the occlusion prediction head for SAM2.
    /// </summary>
    /// <param name="numFeatures">Number of input feature channels (default: 256).</param>
    /// <param name="featureHeight">Height of feature maps (default: 64).</param>
    /// <param name="featureWidth">Width of feature maps (default: 64).</param>
    /// <returns>Occlusion prediction layers. Output shape: [1]</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This head predicts whether the tracked object is occluded
    /// (hidden by other objects). A high score indicates the object may be temporarily invisible.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateSAM2OcclusionHead(
        int numFeatures = 256,
        int featureHeight = 64,
        int featureWidth = 64)
    {
        // Input: [numFeatures, h, w] from CreateSAM2MaskDecoderLayers
        // Global pool then predict occlusion probability
        yield return new GlobalPoolingLayer<T>([numFeatures, featureHeight, featureWidth], PoolingType.Average);
        yield return new DenseLayer<T>(numFeatures, 1, new SigmoidActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Creates all SAM2 layers for backward compatibility.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Warning:</b> This method returns layers from multiple branches that cannot be
    /// chained sequentially. Use the individual factory methods (CreateSAM2ImageEncoderLayers,
    /// CreateSAM2PromptEncoderLayers, CreateSAM2MemoryLayers, CreateSAM2MaskDecoderLayers)
    /// for proper multi-branch handling.
    /// </para>
    /// </remarks>
    [Obsolete("Use individual SAM2 factory methods (CreateSAM2ImageEncoderLayers, etc.) for proper multi-branch architecture.")]
    public static IEnumerable<ILayer<T>> CreateDefaultSAM2Layers(
        int inputChannels = 3,
        int inputHeight = 1024,
        int inputWidth = 1024,
        int numFeatures = 256)
    {
        int featureHeight = inputHeight / 16;
        int featureWidth = inputWidth / 16;

        // Return only the image encoder as a chainable sequence
        // Other branches must be created separately and wired in the SAM2 model
        foreach (var layer in CreateSAM2ImageEncoderLayers(inputChannels, inputHeight, inputWidth, numFeatures))
            yield return layer;
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

        // Classification head with global average pooling
        // First reduce features, then pool to 1x1, then classify
        yield return new ConvolutionalLayer<T>(numFeatures, featH, featW, numFeatures, 1, 1, 0, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new GlobalPoolingLayer<T>([numFeatures, featH, featW], PoolingType.Average);
        // After global pooling, shape is [numFeatures, 1, 1] - use DenseLayer for final classification
        yield return new DenseLayer<T>(numFeatures, numClasses, new SoftmaxActivation<T>() as IActivationFunction<T>);

        // Decoder blocks for reconstruction (4 blocks) - these operate at featH x featW resolution
        // Note: In a real VideoMAE implementation, the decoder would receive encoder features
        // and upsample back to original resolution. This factory provides encoder + classifier;
        // full reconstruction with unpatching is handled by the VideoMAE model class.
        for (int i = 0; i < 4; i++)
        {
            yield return new ConvolutionalLayer<T>(numFeatures, featH, featW, numFeatures, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        }

        // Reconstruction head - outputs patch-sized predictions at feature resolution
        // The VideoMAE model handles reassembly back to video tubelet space
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

        // Stage 1 - no upsampling yet
        yield return new ConvolutionalLayer<T>(currentFeatures, h, w, numFeatures / 2, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        currentFeatures = numFeatures / 2;

        // Stage 2 - 2x upsample
        yield return new UpsamplingLayer<T>([currentFeatures, h, w], 2);
        h *= 2; w *= 2;
        yield return new ConvolutionalLayer<T>(currentFeatures, h, w, numFeatures / 4, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        currentFeatures = numFeatures / 4;

        // Stage 3 - 2x upsample
        yield return new UpsamplingLayer<T>([currentFeatures, h, w], 2);
        h *= 2; w *= 2;
        yield return new ConvolutionalLayer<T>(currentFeatures, h, w, numFeatures / 8, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        currentFeatures = numFeatures / 8;

        // Stage 4 - 2x upsample
        yield return new UpsamplingLayer<T>([currentFeatures, h, w], 2);
        h *= 2; w *= 2;
        yield return new ConvolutionalLayer<T>(currentFeatures, h, w, 64, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);

        // Depth head - 2x upsample to original resolution
        yield return new UpsamplingLayer<T>([64, h, w], 2);
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
    /// Creates the slow pathway layers for SlowFast video recognition.
    /// </summary>
    /// <param name="inputChannels">Number of input channels (default: 3 for RGB).</param>
    /// <param name="inputHeight">Input height (default: 224).</param>
    /// <param name="inputWidth">Input width (default: 224).</param>
    /// <param name="slowChannels">Base channel count for slow pathway (default: 64).</param>
    /// <returns>Slow pathway layers that process fewer frames at higher capacity.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The slow pathway processes video at a low frame rate (e.g., 4 fps)
    /// but with high channel capacity. It captures spatial semantics and appearance features.
    /// Output shape: [slowChannels * 8, H/16, W/16]
    /// </para>
    /// <para>
    /// <b>Note:</b> SlowFast is a dual-pathway architecture. Use separate factory methods:
    /// - CreateSlowFastSlowPathwayLayers: Low frame rate, high capacity (this method)
    /// - CreateSlowFastFastPathwayLayers: High frame rate, low capacity
    /// - CreateSlowFastFusionLayers: Combines pathways for classification
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateSlowFastSlowPathwayLayers(
        int inputChannels = 3,
        int inputHeight = 224,
        int inputWidth = 224,
        int slowChannels = 64)
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
    }

    /// <summary>
    /// Creates the fast pathway layers for SlowFast video recognition.
    /// </summary>
    /// <param name="inputChannels">Number of input channels (default: 3 for RGB).</param>
    /// <param name="inputHeight">Input height (default: 224).</param>
    /// <param name="inputWidth">Input width (default: 224).</param>
    /// <param name="fastChannels">Base channel count for fast pathway (default: 8).</param>
    /// <returns>Fast pathway layers that process more frames at lower capacity.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The fast pathway processes video at a high frame rate (e.g., 32 fps)
    /// but with lower channel capacity (1/8 of slow pathway). It captures motion and temporal dynamics.
    /// Output shape: [fastChannels * 8, H/16, W/16]
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateSlowFastFastPathwayLayers(
        int inputChannels = 3,
        int inputHeight = 224,
        int inputWidth = 224,
        int fastChannels = 8)
    {
        int h = inputHeight;
        int w = inputWidth;

        // Fast pathway - processes more frames at lower channel capacity
        yield return new ConvolutionalLayer<T>(inputChannels, h, w, fastChannels, 7, 2, 3, new ReLUActivation<T>() as IActivationFunction<T>);
        h /= 2; w /= 2;
        yield return new ConvolutionalLayer<T>(fastChannels, h, w, fastChannels * 2, 3, 2, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        h /= 2; w /= 2;
        yield return new ConvolutionalLayer<T>(fastChannels * 2, h, w, fastChannels * 4, 3, 2, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        h /= 2; w /= 2;
        yield return new ConvolutionalLayer<T>(fastChannels * 4, h, w, fastChannels * 8, 3, 2, 1, new ReLUActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Creates the fusion and classification layers for SlowFast.
    /// </summary>
    /// <param name="slowChannels">Base channel count for slow pathway (default: 64).</param>
    /// <param name="fastChannels">Base channel count for fast pathway (default: 8).</param>
    /// <param name="featureHeight">Height of feature maps after pathways (default: 14).</param>
    /// <param name="featureWidth">Width of feature maps after pathways (default: 14).</param>
    /// <param name="numClasses">Number of action classes (default: 400 for Kinetics).</param>
    /// <returns>Fusion layers that combine pathways and classify actions.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This fuses the slow and fast pathway features (after concatenation)
    /// and produces the final action classification. The SlowFast model should:
    /// 1. Run slow pathway on subsampled frames
    /// 2. Run fast pathway on all frames
    /// 3. Concatenate outputs along channel dimension
    /// 4. Apply these fusion layers
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateSlowFastFusionLayers(
        int slowChannels = 64,
        int fastChannels = 8,
        int featureHeight = 14,
        int featureWidth = 14,
        int numClasses = 400)
    {
        int fusedChannels = slowChannels * 8 + fastChannels * 8;
        int h = featureHeight;
        int w = featureWidth;

        // 1x1 conv to fuse concatenated features
        yield return new ConvolutionalLayer<T>(fusedChannels, h, w, fusedChannels, 1, 1, 0, new ReLUActivation<T>() as IActivationFunction<T>);

        // Global average pooling
        yield return new GlobalPoolingLayer<T>([fusedChannels, h, w], PoolingType.Average);

        // Classification head
        yield return new DenseLayer<T>(fusedChannels, numClasses, new SoftmaxActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Creates all SlowFast layers for backward compatibility (returns only slow pathway).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Warning:</b> SlowFast is a dual-pathway architecture that cannot be represented
    /// as a single sequential layer list. Use the individual factory methods:
    /// - CreateSlowFastSlowPathwayLayers
    /// - CreateSlowFastFastPathwayLayers
    /// - CreateSlowFastFusionLayers
    /// </para>
    /// </remarks>
    [Obsolete("Use individual SlowFast factory methods for proper dual-pathway architecture.")]
    public static IEnumerable<ILayer<T>> CreateDefaultSlowFastLayers(
        int inputChannels = 3,
        int inputHeight = 224,
        int inputWidth = 224,
        int numClasses = 400,
        int slowChannels = 64,
        int fastChannels = 8,
        int alpha = 8)
    {
        // Return only the slow pathway as a chainable sequence
        // Fast pathway and fusion must be handled separately in the SlowFast model
        foreach (var layer in CreateSlowFastSlowPathwayLayers(inputChannels, inputHeight, inputWidth, slowChannels))
            yield return layer;
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

        // Upsampling with PixelShuffle (sub-pixel convolution)
        // Conv produces numFeatures*4 channels, PixelShuffle rearranges to numFeatures channels at 2x resolution
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, numFeatures * 4, 3, 1, 1, new LeakyReLUActivation<T>() as IActivationFunction<T>);
        yield return new PixelShuffleLayer<T>([numFeatures * 4, h, w], 2);
        h *= 2; w *= 2;
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, numFeatures * 4, 3, 1, 1, new LeakyReLUActivation<T>() as IActivationFunction<T>);
        yield return new PixelShuffleLayer<T>([numFeatures * 4, h, w], 2);
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

        // Decoder (flow-agnostic reconstruction) with upsampling
        yield return new ConvolutionalLayer<T>(numFeatures * 8, h, w, numFeatures * 4, 3, 1, 1, new LeakyReLUActivation<T>() as IActivationFunction<T>);
        yield return new UpsamplingLayer<T>([numFeatures * 4, h, w], 2);
        h *= 2; w *= 2;
        yield return new ConvolutionalLayer<T>(numFeatures * 4, h, w, numFeatures * 2, 3, 1, 1, new LeakyReLUActivation<T>() as IActivationFunction<T>);
        yield return new UpsamplingLayer<T>([numFeatures * 2, h, w], 2);
        h *= 2; w *= 2;
        yield return new ConvolutionalLayer<T>(numFeatures * 2, h, w, numFeatures, 3, 1, 1, new LeakyReLUActivation<T>() as IActivationFunction<T>);
        yield return new UpsamplingLayer<T>([numFeatures, h, w], 2);
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

        // Decoder for stabilized frame with upsampling
        yield return new ConvolutionalLayer<T>(numFeatures * 4, h, w, numFeatures * 2, 3, 1, 1, new LeakyReLUActivation<T>() as IActivationFunction<T>);
        yield return new UpsamplingLayer<T>([numFeatures * 2, h, w], 2);
        h *= 2; w *= 2;
        yield return new ConvolutionalLayer<T>(numFeatures * 2, h, w, numFeatures, 3, 1, 1, new LeakyReLUActivation<T>() as IActivationFunction<T>);
        yield return new UpsamplingLayer<T>([numFeatures, h, w], 2);
        h *= 2; w *= 2;
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, numFeatures, 3, 1, 1, new LeakyReLUActivation<T>() as IActivationFunction<T>);
        yield return new UpsamplingLayer<T>([numFeatures, h, w], 2);
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

        // Decoder with upsampling
        yield return new ConvolutionalLayer<T>(numFeatures * 8, h, w, numFeatures * 4, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new UpsamplingLayer<T>([numFeatures * 4, h, w], 2);
        h *= 2; w *= 2;
        yield return new ConvolutionalLayer<T>(numFeatures * 4, h, w, numFeatures * 2, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new UpsamplingLayer<T>([numFeatures * 2, h, w], 2);
        h *= 2; w *= 2;
        yield return new ConvolutionalLayer<T>(numFeatures * 2, h, w, numFeatures, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new UpsamplingLayer<T>([numFeatures, h, w], 2);
        h *= 2; w *= 2;
        yield return new ConvolutionalLayer<T>(numFeatures, h, w, numFeatures, 3, 1, 1, new ReLUActivation<T>() as IActivationFunction<T>);
        yield return new UpsamplingLayer<T>([numFeatures, h, w], 2);
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

    #region Document AI Layers

    /// <summary>
    /// Creates default LayoutLMv3 layers for document understanding.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="hiddenDim">Hidden dimension size (default: 768 from paper).</param>
    /// <param name="numLayers">Number of transformer layers (default: 12 from paper).</param>
    /// <param name="numHeads">Number of attention heads (default: 12 from paper).</param>
    /// <param name="vocabSize">Vocabulary size (default: 50265 for RoBERTa tokenizer).</param>
    /// <param name="imageSize">Input image size (default: 224).</param>
    /// <param name="patchSize">Vision patch size (default: 16).</param>
    /// <param name="numClasses">Number of output classes (default: 17 for layout detection).</param>
    /// <returns>A collection of layers forming a LayoutLMv3 architecture.</returns>
    /// <remarks>
    /// <para>
    /// LayoutLMv3 uses unified multimodal pre-training with:
    /// - Text embedding layer (RoBERTa-style)
    /// - Image patch embedding (ViT-style)
    /// - Transformer encoder with spatial-aware self-attention
    /// - Classification head for layout detection or other tasks
    /// </para>
    /// <para>
    /// Reference: "LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking" (ICCV 2022)
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultLayoutLMv3Layers(
        NeuralNetworkArchitecture<T> architecture,
        int hiddenDim = 768,
        int numLayers = 12,
        int numHeads = 12,
        int vocabSize = 50265,
        int imageSize = 224,
        int patchSize = 16,
        int numClasses = 17)
    {
        IActivationFunction<T> geluActivation = new GELUActivation<T>();

        // Text embedding layer (converts token IDs to embeddings)
        yield return new EmbeddingLayer<T>(vocabSize, hiddenDim);

        // Image patch embedding layer (ViT-style)
        int numPatches = (imageSize / patchSize) * (imageSize / patchSize);
        yield return new PatchEmbeddingLayer<T>(imageSize, imageSize, 3, patchSize, hiddenDim);

        // Transformer encoder layers (unified for text and image)
        for (int i = 0; i < numLayers; i++)
        {
            yield return new TransformerEncoderLayer<T>(
                hiddenDim,
                numHeads,
                hiddenDim * 4); // FFN intermediate size = 4x hidden dim (standard)
        }

        // Classification head
        yield return new DenseLayer<T>(hiddenDim, hiddenDim, geluActivation);
        yield return new DenseLayer<T>(hiddenDim, numClasses);
    }

    /// <summary>
    /// Creates default Donut layers for OCR-free document understanding.
    /// </summary>
    /// <param name="imageHeight">Input image height (default: 1920 for donut-base).</param>
    /// <param name="imageWidth">Input image width (default: 2560 for donut-base).</param>
    /// <param name="inputChannels">Number of input channels (default: 3 for RGB).</param>
    /// <param name="embedDim">Initial embedding dimension (default: 128 for Swin-B).</param>
    /// <param name="depths">Depths of each Swin stage (default: {2,2,14,2} for donut-base).</param>
    /// <param name="numHeads">Attention heads per stage (default: {4,8,16,32}).</param>
    /// <param name="windowSize">Window size for attention (default: 10 for donut-base).</param>
    /// <param name="patchSize">Initial patch size (default: 4).</param>
    /// <param name="mlpRatio">MLP expansion ratio (default: 4).</param>
    /// <param name="decoderHiddenDim">Decoder hidden dimension (default: 1024).</param>
    /// <param name="numDecoderLayers">Number of decoder layers (default: 4).</param>
    /// <param name="decoderHeads">Number of decoder attention heads (default: 16).</param>
    /// <param name="vocabSize">Vocabulary size (default: 57522).</param>
    /// <param name="maxGenerationLength">Maximum output sequence length (default: 768).</param>
    /// <returns>A tuple of (EncoderLayers, DecoderLayers) forming a Donut architecture.</returns>
    /// <remarks>
    /// <para>
    /// Donut (Document Understanding Transformer) is an OCR-free end-to-end model:
    /// - Swin Transformer-B encoder with hierarchical stages for image features
    /// - BART-style decoder for text generation
    /// - Direct pixel-to-text conversion without explicit OCR
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This creates a model that can "read" documents directly from pixels
    /// without needing a separate OCR step. The encoder extracts visual features at multiple scales
    /// using the Swin Transformer architecture, while the decoder generates text autoregressively.
    /// </para>
    /// <para>
    /// <b>Default Configuration (donut-base):</b>
    /// - Input: 2560×1920 RGB images
    /// - Encoder: Swin-B with depths {2,2,14,2}, 128 initial dim, window size 10
    /// - Decoder: 4-layer BART-style with 1024 hidden dim
    /// </para>
    /// <para>
    /// Reference: "OCR-free Document Understanding Transformer" (ECCV 2022)
    /// </para>
    /// </remarks>
    public static (IEnumerable<ILayer<T>> EncoderLayers, IEnumerable<ILayer<T>> DecoderLayers) CreateDefaultDonutLayers(
        int imageHeight = 1920,
        int imageWidth = 2560,
        int inputChannels = 3,
        int embedDim = 128,
        int[]? depths = null,
        int[]? numHeads = null,
        int windowSize = 10,
        int patchSize = 4,
        int mlpRatio = 4,
        int decoderHiddenDim = 1024,
        int numDecoderLayers = 4,
        int decoderHeads = 16,
        int vocabSize = 57522,
        int maxGenerationLength = 768)
    {
        // Default depths for Swin-B (donut-base): {2, 2, 14, 2}
        depths ??= [2, 2, 14, 2];

        // Default heads per stage: {4, 8, 16, 32} (doubling each stage)
        numHeads ??= [4, 8, 16, 32];

        if (depths.Length != 4)
            throw new ArgumentException("Swin Transformer requires exactly 4 stages.", nameof(depths));
        if (numHeads.Length != 4)
            throw new ArgumentException("Must specify attention heads for all 4 stages.", nameof(numHeads));

        return (
            CreateDonutEncoderLayers(imageHeight, imageWidth, inputChannels, embedDim, depths, numHeads, windowSize, patchSize, mlpRatio),
            CreateDonutDecoderLayers(embedDim * 8, decoderHiddenDim, numDecoderLayers, decoderHeads, vocabSize, maxGenerationLength)
        );
    }

    /// <summary>
    /// Creates Donut encoder layers (Swin Transformer-B) using yield pattern.
    /// </summary>
    private static IEnumerable<ILayer<T>> CreateDonutEncoderLayers(
        int imageHeight,
        int imageWidth,
        int inputChannels,
        int embedDim,
        int[] depths,
        int[] numHeads,
        int windowSize,
        int patchSize,
        int mlpRatio)
    {
        // Stage 0: Patch embedding
        yield return new SwinPatchEmbeddingLayer<T>(
            imageHeight,
            imageWidth,
            inputChannels,
            patchSize,
            embedDim);

        int currentDim = embedDim;

        // Stages 1-4: Swin Transformer blocks with patch merging between stages
        for (int stage = 0; stage < 4; stage++)
        {
            int stageDepth = depths[stage];
            int stageHeads = numHeads[stage];

            // Swin blocks for this stage (alternating W-MSA and SW-MSA)
            for (int block = 0; block < stageDepth; block++)
            {
                // Alternate between regular and shifted windows
                int shiftSize = (block % 2 == 1) ? windowSize / 2 : 0;

                yield return new SwinTransformerBlockLayer<T>(
                    currentDim,
                    stageHeads,
                    windowSize,
                    shiftSize,
                    mlpRatio);
            }

            // Patch merging between stages (except after last stage)
            if (stage < 3)
            {
                yield return new SwinPatchMergingLayer<T>(currentDim);
                currentDim *= 2; // Channels double after each merge
            }
        }
    }

    /// <summary>
    /// Creates Donut decoder layers (BART-style) using yield pattern.
    /// </summary>
    private static IEnumerable<ILayer<T>> CreateDonutDecoderLayers(
        int encoderOutputDim,
        int decoderHiddenDim,
        int numDecoderLayers,
        int decoderHeads,
        int vocabSize,
        int maxGenerationLength)
    {
        // Text embedding layer
        yield return new EmbeddingLayer<T>(vocabSize, decoderHiddenDim);

        // Optional: projection from encoder output to decoder hidden dim if they differ
        // The cross-attention in TransformerDecoderLayer handles this internally

        // BART-style decoder layers with cross-attention
        IActivationFunction<T>? nullActivation = null;
        for (int i = 0; i < numDecoderLayers; i++)
        {
            yield return new TransformerDecoderLayer<T>(
                decoderHiddenDim,
                decoderHeads,
                decoderHiddenDim * 4,
                maxGenerationLength,
                nullActivation);
        }

        // Output projection to vocabulary
        yield return new DenseLayer<T>(decoderHiddenDim, vocabSize);
    }

    #endregion

    #region DBNet Layers

    /// <summary>
    /// Creates default layers for DBNet text detection model.
    /// </summary>
    /// <param name="imageSize">Input image size (default: 640).</param>
    /// <param name="backboneChannels">Backbone output channels (default: 256).</param>
    /// <param name="innerChannels">FPN inner channels (default: 256).</param>
    /// <returns>Enumerable of layers for DBNet.</returns>
    /// <remarks>
    /// <para>
    /// DBNet uses a ResNet backbone with FPN for multi-scale features,
    /// followed by probability and threshold prediction heads.
    /// </para>
    /// <para>
    /// Reference: "Real-time Scene Text Detection with Differentiable Binarization" (AAAI 2020)
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultDBNetLayers(
        int imageSize = 640,
        int backboneChannels = 256,
        int innerChannels = 256)
    {
        // ResNet-18 style backbone (simplified)
        yield return new ConvolutionalLayer<T>(3, imageSize, imageSize, 64, 7, 2, 3);
        yield return new BatchNormalizationLayer<T>(64);
        yield return new MaxPoolingLayer<T>([64, imageSize / 2, imageSize / 2], 3, 2);

        // ResNet blocks (simplified to conv layers for demonstration)
        yield return new ConvolutionalLayer<T>(64, imageSize / 4, imageSize / 4, 64, 3, 1, 1);
        yield return new BatchNormalizationLayer<T>(64);
        yield return new ConvolutionalLayer<T>(64, imageSize / 4, imageSize / 4, 128, 3, 2, 1);
        yield return new BatchNormalizationLayer<T>(128);
        yield return new ConvolutionalLayer<T>(128, imageSize / 8, imageSize / 8, 256, 3, 2, 1);
        yield return new BatchNormalizationLayer<T>(256);
        yield return new ConvolutionalLayer<T>(256, imageSize / 16, imageSize / 16, backboneChannels, 3, 2, 1);
        yield return new BatchNormalizationLayer<T>(backboneChannels);

        // FPN neck - lateral connections
        yield return new ConvolutionalLayer<T>(backboneChannels, imageSize / 32, imageSize / 32, innerChannels, 1, 1, 0);

        // Probability map head (outputs text probability at each pixel)
        yield return new ConvolutionalLayer<T>(innerChannels, imageSize / 32, imageSize / 32, innerChannels / 4, 3, 1, 1);
        yield return new BatchNormalizationLayer<T>(innerChannels / 4);

        // Threshold map head (outputs adaptive threshold at each pixel)
        yield return new ConvolutionalLayer<T>(innerChannels / 4, imageSize / 32, imageSize / 32, 1, 1, 1, 0);
    }

    #endregion

    #region TrOCR Layers

    /// <summary>
    /// Creates default layers for TrOCR text recognition model.
    /// </summary>
    /// <param name="imageSize">Input image size (default: 384).</param>
    /// <param name="patchSize">ViT patch size (default: 16).</param>
    /// <param name="encoderHiddenDim">Encoder hidden dimension (default: 768).</param>
    /// <param name="decoderHiddenDim">Decoder hidden dimension (default: 768).</param>
    /// <param name="numEncoderLayers">Number of encoder layers (default: 12).</param>
    /// <param name="numDecoderLayers">Number of decoder layers (default: 6).</param>
    /// <param name="numEncoderHeads">Number of encoder heads (default: 12).</param>
    /// <param name="numDecoderHeads">Number of decoder heads (default: 12).</param>
    /// <param name="vocabSize">Vocabulary size (default: 50265).</param>
    /// <param name="maxSequenceLength">Maximum sequence length (default: 128).</param>
    /// <returns>Tuple of encoder and decoder layers.</returns>
    /// <remarks>
    /// <para>
    /// TrOCR uses a Vision Transformer (ViT) encoder and a Transformer decoder.
    /// </para>
    /// <para>
    /// Reference: "TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models" (AAAI 2022)
    /// </para>
    /// </remarks>
    public static (IEnumerable<ILayer<T>> EncoderLayers, IEnumerable<ILayer<T>> DecoderLayers) CreateDefaultTrOCRLayers(
        int imageSize = 384,
        int patchSize = 16,
        int encoderHiddenDim = 768,
        int decoderHiddenDim = 768,
        int numEncoderLayers = 12,
        int numDecoderLayers = 6,
        int numEncoderHeads = 12,
        int numDecoderHeads = 12,
        int vocabSize = 50265,
        int maxSequenceLength = 128)
    {
        return (
            CreateTrOCREncoderLayers(imageSize, patchSize, encoderHiddenDim, numEncoderLayers, numEncoderHeads),
            CreateTrOCRDecoderLayers(decoderHiddenDim, numDecoderLayers, numDecoderHeads, vocabSize, maxSequenceLength)
        );
    }

    private static IEnumerable<ILayer<T>> CreateTrOCREncoderLayers(
        int imageSize,
        int patchSize,
        int hiddenDim,
        int numLayers,
        int numHeads)
    {
        // Patch embedding via convolution (converts image to sequence of patches)
        int numPatches = (imageSize / patchSize) * (imageSize / patchSize);
        yield return new ConvolutionalLayer<T>(3, imageSize, imageSize, hiddenDim, patchSize, patchSize, 0);

        // Layer normalization
        yield return new LayerNormalizationLayer<T>(hiddenDim);

        // ViT encoder blocks
        for (int i = 0; i < numLayers; i++)
        {
            yield return new TransformerEncoderLayer<T>(hiddenDim, numHeads, hiddenDim * 4);
        }

        // Final layer norm
        yield return new LayerNormalizationLayer<T>(hiddenDim);
    }

    private static IEnumerable<ILayer<T>> CreateTrOCRDecoderLayers(
        int hiddenDim,
        int numLayers,
        int numHeads,
        int vocabSize,
        int maxSequenceLength)
    {
        // Text embedding
        yield return new EmbeddingLayer<T>(vocabSize, hiddenDim);

        // Transformer decoder layers
        IActivationFunction<T>? nullActivation = null;
        for (int i = 0; i < numLayers; i++)
        {
            yield return new TransformerDecoderLayer<T>(
                hiddenDim,
                numHeads,
                hiddenDim * 4,
                maxSequenceLength,
                nullActivation);
        }

        // Final layer norm
        yield return new LayerNormalizationLayer<T>(hiddenDim);

        // Output projection to vocabulary
        yield return new DenseLayer<T>(hiddenDim, vocabSize);
    }

    #endregion

    #region TableTransformer Layers

    /// <summary>
    /// Creates default layers for TableTransformer model.
    /// </summary>
    /// <param name="imageSize">Input image size (default: 800).</param>
    /// <param name="hiddenDim">Transformer hidden dimension (default: 256).</param>
    /// <param name="numEncoderLayers">Number of encoder layers (default: 6).</param>
    /// <param name="numDecoderLayers">Number of decoder layers (default: 6).</param>
    /// <param name="numHeads">Number of attention heads (default: 8).</param>
    /// <param name="numQueries">Number of object queries (default: 100).</param>
    /// <param name="numStructureClasses">Number of structure classes (default: 7).</param>
    /// <returns>Enumerable of layers for TableTransformer.</returns>
    /// <remarks>
    /// <para>
    /// TableTransformer uses a DETR-style architecture with ResNet backbone.
    /// </para>
    /// <para>
    /// Reference: "PubTables-1M: Towards Comprehensive Table Extraction" (CVPR 2022)
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultTableTransformerLayers(
        int imageSize = 800,
        int hiddenDim = 256,
        int numEncoderLayers = 6,
        int numDecoderLayers = 6,
        int numHeads = 8,
        int numQueries = 100,
        int numStructureClasses = 7)
    {
        // ResNet-18 backbone (simplified)
        yield return new ConvolutionalLayer<T>(3, imageSize, imageSize, 64, 7, 2, 3);
        yield return new BatchNormalizationLayer<T>(64);
        yield return new MaxPoolingLayer<T>([64, imageSize / 2, imageSize / 2], 3, 2);

        // Downsample to feature map size
        yield return new ConvolutionalLayer<T>(64, imageSize / 4, imageSize / 4, 128, 3, 2, 1);
        yield return new BatchNormalizationLayer<T>(128);
        yield return new ConvolutionalLayer<T>(128, imageSize / 8, imageSize / 8, 256, 3, 2, 1);
        yield return new BatchNormalizationLayer<T>(256);
        yield return new ConvolutionalLayer<T>(256, imageSize / 16, imageSize / 16, hiddenDim, 3, 2, 1);
        yield return new BatchNormalizationLayer<T>(hiddenDim);

        // Flatten spatial dimensions for transformer input
        int featureMapSize = imageSize / 32;
        int seqLen = featureMapSize * featureMapSize;

        // Transformer encoder
        for (int i = 0; i < numEncoderLayers; i++)
        {
            yield return new TransformerEncoderLayer<T>(hiddenDim, numHeads, hiddenDim * 4);
        }

        // Transformer decoder with object queries
        IActivationFunction<T>? nullActivation = null;
        for (int i = 0; i < numDecoderLayers; i++)
        {
            yield return new TransformerDecoderLayer<T>(
                hiddenDim,
                numHeads,
                hiddenDim * 4,
                numQueries,
                nullActivation);
        }

        // Classification head (4 bbox + num_classes)
        yield return new DenseLayer<T>(hiddenDim, 4 + numStructureClasses);
    }

    #endregion

    #region DocBank Layers

    /// <summary>
    /// Creates default layers for DocBank page segmentation model.
    /// </summary>
    /// <param name="imageSize">Input image size (default: 1024).</param>
    /// <param name="backboneChannels">Backbone output channels (default: 256).</param>
    /// <param name="numClasses">Number of segmentation classes (default: 13).</param>
    /// <param name="hiddenDim">Hidden dimension for segmentation head (default: 256).</param>
    /// <returns>Enumerable of layers for DocBank.</returns>
    /// <remarks>
    /// <para>
    /// DocBank uses a ResNet backbone with FPN for semantic segmentation.
    /// </para>
    /// <para>
    /// Reference: "DocBank: A Benchmark Dataset for Document Layout Analysis" (COLING 2020)
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultDocBankLayers(
        int imageSize = 1024,
        int backboneChannels = 256,
        int numClasses = 13,
        int hiddenDim = 256)
    {
        // ResNet-101 style backbone (simplified to ResNet-50-like)
        yield return new ConvolutionalLayer<T>(3, imageSize, imageSize, 64, 7, 2, 3);
        yield return new BatchNormalizationLayer<T>(64);
        yield return new MaxPoolingLayer<T>([64, imageSize / 2, imageSize / 2], 3, 2);

        // Residual blocks (simplified as conv layers)
        yield return new ConvolutionalLayer<T>(64, imageSize / 4, imageSize / 4, 256, 3, 1, 1);
        yield return new BatchNormalizationLayer<T>(256);
        yield return new ConvolutionalLayer<T>(256, imageSize / 4, imageSize / 4, 512, 3, 2, 1);
        yield return new BatchNormalizationLayer<T>(512);
        yield return new ConvolutionalLayer<T>(512, imageSize / 8, imageSize / 8, 1024, 3, 2, 1);
        yield return new BatchNormalizationLayer<T>(1024);
        yield return new ConvolutionalLayer<T>(1024, imageSize / 16, imageSize / 16, backboneChannels, 3, 2, 1);
        yield return new BatchNormalizationLayer<T>(backboneChannels);

        // FPN lateral connections
        yield return new ConvolutionalLayer<T>(backboneChannels, imageSize / 32, imageSize / 32, hiddenDim, 1, 1, 0);

        // Segmentation head
        yield return new ConvolutionalLayer<T>(hiddenDim, imageSize / 32, imageSize / 32, hiddenDim, 3, 1, 1);
        yield return new BatchNormalizationLayer<T>(hiddenDim);

        // Output layer (class predictions per pixel)
        yield return new ConvolutionalLayer<T>(hiddenDim, imageSize / 32, imageSize / 32, numClasses, 1, 1, 0);
    }

    /// <summary>
    /// Creates default LayoutLM (v1) layers for document understanding with layout-aware pre-training.
    /// </summary>
    /// <param name="hiddenDim">Hidden dimension (default: 768 for BERT-base).</param>
    /// <param name="numLayers">Number of transformer layers (default: 12).</param>
    /// <param name="numHeads">Number of attention heads (default: 12).</param>
    /// <param name="vocabSize">Vocabulary size (default: 30522 for BERT).</param>
    /// <param name="maxSequenceLength">Maximum sequence length (default: 512).</param>
    /// <param name="numClasses">Number of output classes (default: 7 for FUNSD).</param>
    /// <returns>A collection of layers forming a LayoutLM model.</returns>
    /// <remarks>
    /// <para>
    /// LayoutLM v1 combines BERT text embeddings with 2D position embeddings to jointly
    /// model text and layout. Unlike v2/v3, it does NOT use visual features.
    /// </para>
    /// <para>
    /// Reference: "LayoutLM: Pre-training of Text and Layout for Document Image Understanding" (KDD 2020)
    /// https://arxiv.org/abs/1912.13318
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultLayoutLMLayers(
        int hiddenDim = 768,
        int numLayers = 12,
        int numHeads = 12,
        int vocabSize = 30522,
        int maxSequenceLength = 512,
        int numClasses = 7)
    {
        IActivationFunction<T> geluActivation = new GELUActivation<T>();
        IActivationFunction<T> identityActivation = new IdentityActivation<T>();
        int intermediateSize = hiddenDim * 4;

        // Word embeddings projection
        yield return new EmbeddingLayer<T>(vocabSize, hiddenDim);

        // Position embeddings
        yield return new PositionalEncodingLayer<T>(maxSequenceLength, hiddenDim);

        // LayerNorm after embeddings
        yield return new LayerNormalizationLayer<T>(hiddenDim);
        yield return new DropoutLayer<T>(0.1);

        // Transformer encoder layers (BERT-style)
        for (int i = 0; i < numLayers; i++)
        {
            // Self-attention
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: maxSequenceLength,
                embeddingDimension: hiddenDim,
                headCount: numHeads,
                activationFunction: identityActivation);

            yield return new LayerNormalizationLayer<T>(hiddenDim);

            // Feed-forward network
            yield return new DenseLayer<T>(hiddenDim, intermediateSize, geluActivation);
            yield return new DenseLayer<T>(intermediateSize, hiddenDim, identityActivation);
            yield return new LayerNormalizationLayer<T>(hiddenDim);

            if (i < numLayers - 1)
            {
                yield return new DropoutLayer<T>(0.1);
            }
        }

        // Classification head for token classification (NER-style)
        yield return new DropoutLayer<T>(0.1);
        yield return new DenseLayer<T>(hiddenDim, numClasses, identityActivation);
    }

    /// <summary>
    /// Creates default LayoutLMv2 layers for document understanding with visual features.
    /// </summary>
    /// <param name="hiddenDim">Hidden dimension (default: 768 for BERT-base).</param>
    /// <param name="numLayers">Number of transformer layers (default: 12).</param>
    /// <param name="numHeads">Number of attention heads (default: 12).</param>
    /// <param name="vocabSize">Vocabulary size (default: 30522 for BERT).</param>
    /// <param name="imageSize">Input image size (default: 224).</param>
    /// <param name="visualBackboneChannels">Visual backbone output channels (default: 256).</param>
    /// <param name="numClasses">Number of output classes (default: 7).</param>
    /// <returns>A collection of layers forming a LayoutLMv2 model.</returns>
    /// <remarks>
    /// <para>
    /// LayoutLMv2 extends LayoutLM by adding visual features from a ResNeXt-FPN backbone,
    /// enabling the model to understand documents through text, layout, AND image features.
    /// </para>
    /// <para>
    /// Key components:
    /// - Visual backbone (ResNeXt-101 with FPN)
    /// - Text encoder (BERT-base)
    /// - Spatial-aware self-attention mechanism
    /// </para>
    /// <para>
    /// Reference: "LayoutLMv2: Multi-modal Pre-training for Visually-rich Document Understanding" (ACL 2021)
    /// https://arxiv.org/abs/2012.14740
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultLayoutLMv2Layers(
        int hiddenDim = 768,
        int numLayers = 12,
        int numHeads = 12,
        int vocabSize = 30522,
        int imageSize = 224,
        int visualBackboneChannels = 256,
        int numClasses = 7)
    {
        IActivationFunction<T> geluActivation = new GELUActivation<T>();
        IActivationFunction<T> reluActivation = new ReLUActivation<T>();
        IActivationFunction<T> identityActivation = new IdentityActivation<T>();
        int intermediateSize = hiddenDim * 4;
        int maxSequenceLength = 512;

        // === VISUAL BACKBONE (ResNeXt-FPN style) ===

        // Initial convolution
        yield return new ConvolutionalLayer<T>(3, imageSize, imageSize, 64, 7, 2, 3);
        yield return new BatchNormalizationLayer<T>(64);

        // Max pooling
        yield return new MaxPoolingLayer<T>([64, imageSize / 2, imageSize / 2], 3, 2);

        // ResNeXt-style stages (simplified)
        yield return new ConvolutionalLayer<T>(64, imageSize / 4, imageSize / 4, 256, 3, 1, 1);
        yield return new BatchNormalizationLayer<T>(256);
        yield return new ConvolutionalLayer<T>(256, imageSize / 4, imageSize / 4, 512, 3, 2, 1);
        yield return new BatchNormalizationLayer<T>(512);
        yield return new ConvolutionalLayer<T>(512, imageSize / 8, imageSize / 8, 1024, 3, 2, 1);
        yield return new BatchNormalizationLayer<T>(1024);
        yield return new ConvolutionalLayer<T>(1024, imageSize / 16, imageSize / 16, visualBackboneChannels, 1, 1, 0);
        yield return new BatchNormalizationLayer<T>(visualBackboneChannels);

        // Project visual features to hidden dimension
        yield return new DenseLayer<T>(visualBackboneChannels, hiddenDim, reluActivation);

        // === TEXT EMBEDDINGS ===

        // Word embeddings
        yield return new EmbeddingLayer<T>(vocabSize, hiddenDim);

        // Position embeddings
        yield return new PositionalEncodingLayer<T>(maxSequenceLength, hiddenDim);

        // LayerNorm after embeddings
        yield return new LayerNormalizationLayer<T>(hiddenDim);
        yield return new DropoutLayer<T>(0.1);

        // === MULTI-MODAL TRANSFORMER ENCODER ===

        for (int i = 0; i < numLayers; i++)
        {
            // Self-attention (spatial-aware)
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: maxSequenceLength,
                embeddingDimension: hiddenDim,
                headCount: numHeads,
                activationFunction: identityActivation);

            yield return new LayerNormalizationLayer<T>(hiddenDim);

            // Feed-forward network
            yield return new DenseLayer<T>(hiddenDim, intermediateSize, geluActivation);
            yield return new DenseLayer<T>(intermediateSize, hiddenDim, identityActivation);
            yield return new LayerNormalizationLayer<T>(hiddenDim);

            if (i < numLayers - 1)
            {
                yield return new DropoutLayer<T>(0.1);
            }
        }

        // === OUTPUT HEAD ===

        // Classification head
        yield return new DropoutLayer<T>(0.1);
        yield return new DenseLayer<T>(hiddenDim, numClasses, identityActivation);
    }

    /// <summary>
    /// Creates default DocFormer layers for document understanding with shared spatial encodings.
    /// </summary>
    /// <param name="hiddenDim">Hidden dimension (default: 768).</param>
    /// <param name="numLayers">Number of transformer layers (default: 12).</param>
    /// <param name="numHeads">Number of attention heads (default: 12).</param>
    /// <param name="vocabSize">Vocabulary size (default: 30522).</param>
    /// <param name="imageSize">Input image size (default: 224).</param>
    /// <param name="spatialDim">Spatial embedding dimension (default: 128).</param>
    /// <param name="numClasses">Number of output classes (default: 16).</param>
    /// <returns>A collection of layers forming a DocFormer model.</returns>
    /// <remarks>
    /// <para>
    /// DocFormer uses shared spatial encodings across text, visual, and layout modalities.
    /// </para>
    /// <para>
    /// Reference: "DocFormer: End-to-End Transformer for Document Understanding" (ICCV 2021)
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultDocFormerLayers(
        int hiddenDim = 768,
        int numLayers = 12,
        int numHeads = 12,
        int vocabSize = 30522,
        int imageSize = 224,
        int spatialDim = 128,
        int numClasses = 16)
    {
        IActivationFunction<T> geluActivation = new GELUActivation<T>();
        IActivationFunction<T> reluActivation = new ReLUActivation<T>();
        IActivationFunction<T> identityActivation = new IdentityActivation<T>();
        int intermediateSize = hiddenDim * 4;
        int maxSequenceLength = 512;

        // === VISUAL ENCODER (ResNet-50 style) ===

        yield return new ConvolutionalLayer<T>(3, imageSize, imageSize, 64, 7, 2, 3);
        yield return new BatchNormalizationLayer<T>(64);
        yield return new MaxPoolingLayer<T>([64, imageSize / 2, imageSize / 2], 3, 2);

        yield return new ConvolutionalLayer<T>(64, imageSize / 4, imageSize / 4, 256, 3, 1, 1);
        yield return new BatchNormalizationLayer<T>(256);
        yield return new ConvolutionalLayer<T>(256, imageSize / 4, imageSize / 4, 512, 3, 2, 1);
        yield return new BatchNormalizationLayer<T>(512);

        // Project visual to hidden dim
        yield return new DenseLayer<T>(512, hiddenDim, reluActivation);

        // === TEXT EMBEDDINGS ===

        yield return new EmbeddingLayer<T>(vocabSize, hiddenDim);
        yield return new PositionalEncodingLayer<T>(maxSequenceLength, hiddenDim);

        // === SPATIAL ENCODINGS (shared) ===

        yield return new DenseLayer<T>(spatialDim * 2, hiddenDim, geluActivation);

        // === MULTI-MODAL TRANSFORMER ===

        yield return new LayerNormalizationLayer<T>(hiddenDim);
        yield return new DropoutLayer<T>(0.1);

        for (int i = 0; i < numLayers; i++)
        {
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: maxSequenceLength,
                embeddingDimension: hiddenDim,
                headCount: numHeads,
                activationFunction: identityActivation);

            yield return new LayerNormalizationLayer<T>(hiddenDim);
            yield return new DenseLayer<T>(hiddenDim, intermediateSize, geluActivation);
            yield return new DenseLayer<T>(intermediateSize, hiddenDim, identityActivation);
            yield return new LayerNormalizationLayer<T>(hiddenDim);

            if (i < numLayers - 1)
            {
                yield return new DropoutLayer<T>(0.1);
            }
        }

        // === OUTPUT HEAD ===

        yield return new DropoutLayer<T>(0.1);
        yield return new DenseLayer<T>(hiddenDim, numClasses, identityActivation);
    }

    /// <summary>
    /// Creates default Pix2Struct layers for screenshot parsing.
    /// </summary>
    /// <param name="hiddenDim">Hidden dimension (default: 1024).</param>
    /// <param name="numEncoderLayers">Number of encoder layers (default: 18).</param>
    /// <param name="numDecoderLayers">Number of decoder layers (default: 18).</param>
    /// <param name="numHeads">Number of attention heads (default: 16).</param>
    /// <param name="vocabSize">Vocabulary size (default: 50000).</param>
    /// <param name="patchSize">Patch size (default: 16).</param>
    /// <param name="maxPatches">Maximum patches (default: 4096).</param>
    /// <param name="maxSequenceLength">Maximum sequence length (default: 1024).</param>
    /// <returns>Tuple of encoder and decoder layers.</returns>
    /// <remarks>
    /// <para>
    /// Reference: "Pix2Struct: Screenshot Parsing as Pretraining" (ICML 2023)
    /// </para>
    /// </remarks>
    public static (IEnumerable<ILayer<T>> EncoderLayers, IEnumerable<ILayer<T>> DecoderLayers) CreateDefaultPix2StructLayers(
        int hiddenDim = 1024,
        int numEncoderLayers = 18,
        int numDecoderLayers = 18,
        int numHeads = 16,
        int vocabSize = 50000,
        int patchSize = 16,
        int maxPatches = 4096,
        int maxSequenceLength = 1024)
    {
        return (
            CreatePix2StructEncoderLayers(hiddenDim, numEncoderLayers, numHeads, patchSize, maxPatches),
            CreatePix2StructDecoderLayers(hiddenDim, numDecoderLayers, numHeads, vocabSize, maxSequenceLength)
        );
    }

    private static IEnumerable<ILayer<T>> CreatePix2StructEncoderLayers(
        int hiddenDim, int numLayers, int numHeads, int patchSize, int maxPatches)
    {
        IActivationFunction<T> geluActivation = new GELUActivation<T>();
        IActivationFunction<T> identityActivation = new IdentityActivation<T>();

        // Patch embedding
        yield return new DenseLayer<T>(patchSize * patchSize * 3, hiddenDim, identityActivation);
        yield return new PositionalEncodingLayer<T>(maxPatches, hiddenDim);
        yield return new LayerNormalizationLayer<T>(hiddenDim);

        // Transformer encoder
        for (int i = 0; i < numLayers; i++)
        {
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: maxPatches,
                embeddingDimension: hiddenDim,
                headCount: numHeads,
                activationFunction: identityActivation);
            yield return new LayerNormalizationLayer<T>(hiddenDim);
            yield return new DenseLayer<T>(hiddenDim, hiddenDim * 4, geluActivation);
            yield return new DenseLayer<T>(hiddenDim * 4, hiddenDim, identityActivation);
            yield return new LayerNormalizationLayer<T>(hiddenDim);
        }
    }

    private static IEnumerable<ILayer<T>> CreatePix2StructDecoderLayers(
        int hiddenDim, int numLayers, int numHeads, int vocabSize, int maxSequenceLength)
    {
        IActivationFunction<T> geluActivation = new GELUActivation<T>();
        IActivationFunction<T> identityActivation = new IdentityActivation<T>();

        yield return new EmbeddingLayer<T>(vocabSize, hiddenDim);
        yield return new PositionalEncodingLayer<T>(maxSequenceLength, hiddenDim);

        for (int i = 0; i < numLayers; i++)
        {
            yield return new TransformerDecoderLayer<T>(
                embeddingSize: hiddenDim,
                numHeads: numHeads,
                feedForwardDim: hiddenDim * 4,
                sequenceLength: maxSequenceLength,
                ffnActivation: geluActivation);
        }

        yield return new LayerNormalizationLayer<T>(hiddenDim);
        yield return new DenseLayer<T>(hiddenDim, vocabSize, identityActivation);
    }

    /// <summary>
    /// Creates default Nougat layers for academic document understanding.
    /// </summary>
    /// <param name="hiddenDim">Hidden dimension (default: 1024).</param>
    /// <param name="numEncoderLayers">Number of encoder layers (default: 12).</param>
    /// <param name="numDecoderLayers">Number of decoder layers (default: 10).</param>
    /// <param name="numHeads">Number of attention heads (default: 16).</param>
    /// <param name="vocabSize">Vocabulary size (default: 50000).</param>
    /// <param name="imageSize">Input image size (default: 896).</param>
    /// <param name="patchSize">Patch size (default: 16).</param>
    /// <param name="maxSequenceLength">Maximum sequence length (default: 4096).</param>
    /// <returns>Tuple of encoder and decoder layers.</returns>
    /// <remarks>
    /// <para>
    /// Reference: "Nougat: Neural Optical Understanding for Academic Documents" (arXiv 2023)
    /// </para>
    /// </remarks>
    public static (IEnumerable<ILayer<T>> EncoderLayers, IEnumerable<ILayer<T>> DecoderLayers) CreateDefaultNougatLayers(
        int hiddenDim = 1024,
        int numEncoderLayers = 12,
        int numDecoderLayers = 10,
        int numHeads = 16,
        int vocabSize = 50000,
        int imageSize = 896,
        int patchSize = 16,
        int maxSequenceLength = 4096)
    {
        return (
            CreateNougatEncoderLayers(hiddenDim, numEncoderLayers, numHeads, imageSize, patchSize),
            CreateNougatDecoderLayers(hiddenDim, numDecoderLayers, numHeads, vocabSize, maxSequenceLength)
        );
    }

    private static IEnumerable<ILayer<T>> CreateNougatEncoderLayers(
        int hiddenDim, int numLayers, int numHeads, int imageSize, int patchSize)
    {
        IActivationFunction<T> geluActivation = new GELUActivation<T>();
        IActivationFunction<T> identityActivation = new IdentityActivation<T>();
        int numPatches = (imageSize / patchSize) * (imageSize / patchSize);

        // Swin-style patch embedding
        yield return new ConvolutionalLayer<T>(3, imageSize, imageSize, hiddenDim, patchSize, patchSize, 0);
        yield return new LayerNormalizationLayer<T>(hiddenDim);

        // Transformer encoder
        for (int i = 0; i < numLayers; i++)
        {
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: numPatches,
                embeddingDimension: hiddenDim,
                headCount: numHeads,
                activationFunction: identityActivation);
            yield return new LayerNormalizationLayer<T>(hiddenDim);
            yield return new DenseLayer<T>(hiddenDim, hiddenDim * 4, geluActivation);
            yield return new DenseLayer<T>(hiddenDim * 4, hiddenDim, identityActivation);
            yield return new LayerNormalizationLayer<T>(hiddenDim);
        }
    }

    private static IEnumerable<ILayer<T>> CreateNougatDecoderLayers(
        int hiddenDim, int numLayers, int numHeads, int vocabSize, int maxSequenceLength)
    {
        IActivationFunction<T> geluActivation = new GELUActivation<T>();
        IActivationFunction<T> identityActivation = new IdentityActivation<T>();

        yield return new EmbeddingLayer<T>(vocabSize, hiddenDim);
        yield return new PositionalEncodingLayer<T>(maxSequenceLength, hiddenDim);

        for (int i = 0; i < numLayers; i++)
        {
            yield return new TransformerDecoderLayer<T>(
                embeddingSize: hiddenDim,
                numHeads: numHeads,
                feedForwardDim: hiddenDim * 4,
                sequenceLength: maxSequenceLength,
                ffnActivation: geluActivation);
        }

        yield return new LayerNormalizationLayer<T>(hiddenDim);
        yield return new DenseLayer<T>(hiddenDim, vocabSize, identityActivation);
    }

    /// <summary>
    /// Creates default UDOP layers for unified document processing.
    /// </summary>
    /// <param name="hiddenDim">Hidden dimension (default: 1024).</param>
    /// <param name="numEncoderLayers">Number of encoder layers (default: 12).</param>
    /// <param name="numDecoderLayers">Number of decoder layers (default: 12).</param>
    /// <param name="numHeads">Number of attention heads (default: 16).</param>
    /// <param name="vocabSize">Vocabulary size (default: 50000).</param>
    /// <param name="imageSize">Input image size (default: 224).</param>
    /// <param name="maxSequenceLength">Maximum sequence length (default: 2048).</param>
    /// <returns>Tuple of encoder and decoder layers.</returns>
    /// <remarks>
    /// <para>
    /// Reference: "UDOP: Unifying Vision, Text, and Layout" (CVPR 2023)
    /// </para>
    /// </remarks>
    public static (IEnumerable<ILayer<T>> EncoderLayers, IEnumerable<ILayer<T>> DecoderLayers) CreateDefaultUDOPLayers(
        int hiddenDim = 1024,
        int numEncoderLayers = 12,
        int numDecoderLayers = 12,
        int numHeads = 16,
        int vocabSize = 50000,
        int imageSize = 224,
        int maxSequenceLength = 2048)
    {
        return (
            CreateUDOPEncoderLayers(hiddenDim, numEncoderLayers, numHeads, vocabSize, imageSize, maxSequenceLength),
            CreateUDOPDecoderLayers(hiddenDim, numDecoderLayers, numHeads, vocabSize, maxSequenceLength)
        );
    }

    private static IEnumerable<ILayer<T>> CreateUDOPEncoderLayers(
        int hiddenDim, int numLayers, int numHeads, int vocabSize, int imageSize, int maxSequenceLength)
    {
        IActivationFunction<T> geluActivation = new GELUActivation<T>();
        IActivationFunction<T> identityActivation = new IdentityActivation<T>();

        // Visual encoder (ViT-style)
        int patchSize = 16;
        int numPatches = (imageSize / patchSize) * (imageSize / patchSize);
        yield return new ConvolutionalLayer<T>(3, imageSize, imageSize, hiddenDim, patchSize, patchSize, 0);
        yield return new PositionalEncodingLayer<T>(numPatches, hiddenDim);

        // Text embeddings
        yield return new EmbeddingLayer<T>(vocabSize, hiddenDim);
        yield return new PositionalEncodingLayer<T>(maxSequenceLength, hiddenDim);

        // Unified encoder
        yield return new LayerNormalizationLayer<T>(hiddenDim);

        for (int i = 0; i < numLayers; i++)
        {
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: maxSequenceLength + numPatches,
                embeddingDimension: hiddenDim,
                headCount: numHeads,
                activationFunction: identityActivation);
            yield return new LayerNormalizationLayer<T>(hiddenDim);
            yield return new DenseLayer<T>(hiddenDim, hiddenDim * 4, geluActivation);
            yield return new DenseLayer<T>(hiddenDim * 4, hiddenDim, identityActivation);
            yield return new LayerNormalizationLayer<T>(hiddenDim);
        }
    }

    private static IEnumerable<ILayer<T>> CreateUDOPDecoderLayers(
        int hiddenDim, int numLayers, int numHeads, int vocabSize, int maxSequenceLength)
    {
        IActivationFunction<T> geluActivation = new GELUActivation<T>();
        IActivationFunction<T> identityActivation = new IdentityActivation<T>();

        yield return new EmbeddingLayer<T>(vocabSize, hiddenDim);
        yield return new PositionalEncodingLayer<T>(maxSequenceLength, hiddenDim);

        for (int i = 0; i < numLayers; i++)
        {
            yield return new TransformerDecoderLayer<T>(
                embeddingSize: hiddenDim,
                numHeads: numHeads,
                feedForwardDim: hiddenDim * 4,
                sequenceLength: maxSequenceLength,
                ffnActivation: geluActivation);
        }

        yield return new LayerNormalizationLayer<T>(hiddenDim);
        yield return new DenseLayer<T>(hiddenDim, vocabSize, identityActivation);
    }

    /// <summary>
    /// Creates default PICK layers for key information extraction.
    /// </summary>
    /// <param name="hiddenDim">Hidden dimension (default: 256).</param>
    /// <param name="numGcnLayers">Number of GCN layers (default: 2).</param>
    /// <param name="numHeads">Number of attention heads (default: 8).</param>
    /// <param name="vocabSize">Vocabulary size (default: 30522).</param>
    /// <param name="numEntityTypes">Number of entity types (default: 14).</param>
    /// <param name="maxSequenceLength">Maximum sequence length (default: 512).</param>
    /// <returns>A collection of layers forming a PICK model.</returns>
    /// <remarks>
    /// <para>
    /// Reference: "PICK: Processing Key Information Extraction" (ICPR 2020)
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultPICKLayers(
        int hiddenDim = 256,
        int numGcnLayers = 2,
        int numHeads = 8,
        int vocabSize = 30522,
        int numEntityTypes = 14,
        int maxSequenceLength = 512)
    {
        IActivationFunction<T> reluActivation = new ReLUActivation<T>();
        IActivationFunction<T> identityActivation = new IdentityActivation<T>();

        // Text encoder (BERT-style)
        yield return new EmbeddingLayer<T>(vocabSize, hiddenDim);
        yield return new PositionalEncodingLayer<T>(maxSequenceLength, hiddenDim);
        yield return new LayerNormalizationLayer<T>(hiddenDim);

        // Transformer layers for text encoding
        for (int i = 0; i < 4; i++)
        {
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: maxSequenceLength,
                embeddingDimension: hiddenDim,
                headCount: numHeads,
                activationFunction: identityActivation);
            yield return new LayerNormalizationLayer<T>(hiddenDim);
            yield return new DenseLayer<T>(hiddenDim, hiddenDim * 4, reluActivation);
            yield return new DenseLayer<T>(hiddenDim * 4, hiddenDim, identityActivation);
            yield return new LayerNormalizationLayer<T>(hiddenDim);
        }

        // Graph Convolutional Network layers (simplified as dense)
        for (int i = 0; i < numGcnLayers; i++)
        {
            yield return new DenseLayer<T>(hiddenDim, hiddenDim, reluActivation);
            yield return new LayerNormalizationLayer<T>(hiddenDim);
            yield return new DropoutLayer<T>(0.1);
        }

        // BiLSTM simulation (using dense layers)
        yield return new DenseLayer<T>(hiddenDim, hiddenDim * 2, reluActivation);
        yield return new DenseLayer<T>(hiddenDim * 2, hiddenDim, identityActivation);

        // Output layer for NER
        yield return new DropoutLayer<T>(0.1);
        yield return new DenseLayer<T>(hiddenDim, numEntityTypes, identityActivation);
    }

    /// <summary>
    /// Creates default CRAFT layers for character-level text detection.
    /// </summary>
    /// <param name="imageSize">Input image size (default: 768).</param>
    /// <param name="backboneChannels">Backbone output channels (default: 512).</param>
    /// <param name="upscaleChannels">Upscale network channels (default: 256).</param>
    /// <returns>A collection of layers forming a CRAFT model.</returns>
    /// <remarks>
    /// <para>
    /// Reference: "Character Region Awareness for Text Detection" (CVPR 2019)
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultCRAFTLayers(
        int imageSize = 768,
        int backboneChannels = 512,
        int upscaleChannels = 256)
    {
        IActivationFunction<T> reluActivation = new ReLUActivation<T>();

        // VGG16-BN style backbone
        int[] vggChannels = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512];
        int currentSize = imageSize;
        int inputChannels = 3;

        for (int i = 0; i < vggChannels.Length; i++)
        {
            yield return new ConvolutionalLayer<T>(inputChannels, currentSize, currentSize, vggChannels[i], 3, 1, 1);
            yield return new BatchNormalizationLayer<T>(vggChannels[i]);

            inputChannels = vggChannels[i];

            // Pooling after certain layers
            if (i == 1 || i == 3 || i == 6 || i == 9 || i == 12)
            {
                yield return new MaxPoolingLayer<T>([inputChannels, currentSize, currentSize], 2, 2);
                currentSize /= 2;
            }
        }

        // U-Net style upsampling
        yield return new ConvolutionalLayer<T>(backboneChannels, currentSize, currentSize, upscaleChannels, 1, 1, 0);

        // Upscale layers
        for (int i = 0; i < 4; i++)
        {
            yield return new ConvolutionalLayer<T>(upscaleChannels, currentSize, currentSize, upscaleChannels, 3, 1, 1);
            yield return new BatchNormalizationLayer<T>(upscaleChannels);
            currentSize *= 2;
        }

        // Output: 2 channels (character region + affinity)
        yield return new ConvolutionalLayer<T>(upscaleChannels, currentSize, currentSize, 32, 3, 1, 1);
        yield return new ConvolutionalLayer<T>(32, currentSize, currentSize, 32, 3, 1, 1);
        yield return new ConvolutionalLayer<T>(32, currentSize, currentSize, 16, 3, 1, 1);
        yield return new ConvolutionalLayer<T>(16, currentSize, currentSize, 16, 1, 1, 0);
        yield return new ConvolutionalLayer<T>(16, currentSize, currentSize, 2, 1, 1, 0);
    }

    /// <summary>
    /// Creates default CRNN layers for sequence text recognition.
    /// </summary>
    /// <param name="imageWidth">Input image width (default: 128).</param>
    /// <param name="imageHeight">Input image height (default: 32).</param>
    /// <param name="cnnChannels">CNN output channels (default: 512).</param>
    /// <param name="rnnHiddenSize">RNN hidden size (default: 256).</param>
    /// <param name="rnnLayers">Number of RNN layers (default: 2).</param>
    /// <param name="charsetSize">Character set size (default: 95).</param>
    /// <returns>A collection of layers forming a CRNN model.</returns>
    /// <remarks>
    /// <para>
    /// Reference: "An End-to-End Trainable Neural Network for Image-based Sequence Recognition" (TPAMI 2017)
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultCRNNLayers(
        int imageWidth = 128,
        int imageHeight = 32,
        int cnnChannels = 512,
        int rnnHiddenSize = 256,
        int rnnLayers = 2,
        int charsetSize = 95)
    {
        IActivationFunction<T> reluActivation = new ReLUActivation<T>();
        IActivationFunction<T> identityActivation = new IdentityActivation<T>();

        int currentHeight = imageHeight;
        int currentWidth = imageWidth;

        // CNN feature extractor (VGG-style)
        int[] channels = [64, 128, 256, 256, 512, 512, 512];
        int inputChannels = 1; // Grayscale

        for (int i = 0; i < channels.Length; i++)
        {
            yield return new ConvolutionalLayer<T>(inputChannels, currentHeight, currentWidth, channels[i], 3, 1, 1);
            yield return new BatchNormalizationLayer<T>(channels[i]);

            inputChannels = channels[i];

            // Pool with (2,2) for first 3 layers, (2,1) for rest
            if (i < 3)
            {
                yield return new MaxPoolingLayer<T>([inputChannels, currentHeight, currentWidth], 2, 2);
                currentHeight /= 2;
                currentWidth /= 2;
            }
            else if (i < 5)
            {
                yield return new MaxPoolingLayer<T>([inputChannels, currentHeight, currentWidth], 2, 1);
                currentHeight /= 2;
            }
        }

        // Map-to-Sequence: reshape CNN output for RNN
        int seqLen = currentWidth;
        int featureDim = cnnChannels * currentHeight;

        // BiLSTM layers (simulated with dense layers)
        yield return new DenseLayer<T>(featureDim, rnnHiddenSize * 2, reluActivation);

        for (int i = 1; i < rnnLayers; i++)
        {
            yield return new DenseLayer<T>(rnnHiddenSize * 2, rnnHiddenSize * 2, reluActivation);
        }

        // Output layer (including CTC blank)
        yield return new DenseLayer<T>(rnnHiddenSize * 2, charsetSize, identityActivation);
    }

    /// <summary>
    /// Creates default LayoutXLM layers for multilingual document understanding.
    /// </summary>
    /// <param name="hiddenDim">Hidden dimension (default: 768).</param>
    /// <param name="numLayers">Number of transformer layers (default: 12).</param>
    /// <param name="numHeads">Number of attention heads (default: 12).</param>
    /// <param name="vocabSize">Vocabulary size (default: 250002 for XLM-RoBERTa).</param>
    /// <param name="imageSize">Input image size (default: 224).</param>
    /// <param name="visualBackboneChannels">Visual backbone channels (default: 256).</param>
    /// <param name="numClasses">Number of output classes (default: 7).</param>
    /// <returns>A collection of layers forming a LayoutXLM model.</returns>
    public static IEnumerable<ILayer<T>> CreateDefaultLayoutXLMLayers(
        int hiddenDim = 768,
        int numLayers = 12,
        int numHeads = 12,
        int vocabSize = 250002,
        int imageSize = 224,
        int visualBackboneChannels = 256,
        int numClasses = 7)
    {
        IActivationFunction<T> geluActivation = new GELUActivation<T>();
        IActivationFunction<T> identityActivation = new IdentityActivation<T>();
        int intermediateSize = hiddenDim * 4;
        int maxSequenceLength = 512;

        // Visual backbone (ResNet-style)
        yield return new ConvolutionalLayer<T>(3, imageSize, imageSize, 64, 7, 2, 3);
        yield return new BatchNormalizationLayer<T>(64);
        yield return new MaxPoolingLayer<T>([64, imageSize / 2, imageSize / 2], 3, 2);
        yield return new ConvolutionalLayer<T>(64, imageSize / 4, imageSize / 4, visualBackboneChannels, 3, 1, 1);

        // Visual projection to hidden dim
        yield return new DenseLayer<T>(visualBackboneChannels * (imageSize / 4) * (imageSize / 4) / 256, hiddenDim, identityActivation);

        // XLM-RoBERTa embeddings
        yield return new EmbeddingLayer<T>(vocabSize, hiddenDim);
        yield return new PositionalEncodingLayer<T>(maxSequenceLength, hiddenDim);
        yield return new LayerNormalizationLayer<T>(hiddenDim);
        yield return new DropoutLayer<T>(0.1);

        // Transformer encoder layers
        for (int i = 0; i < numLayers; i++)
        {
            yield return new MultiHeadAttentionLayer<T>(maxSequenceLength, hiddenDim, numHeads, identityActivation);
            yield return new LayerNormalizationLayer<T>(hiddenDim);
            yield return new DenseLayer<T>(hiddenDim, intermediateSize, geluActivation);
            yield return new DenseLayer<T>(intermediateSize, hiddenDim, identityActivation);
            yield return new LayerNormalizationLayer<T>(hiddenDim);
            if (i < numLayers - 1) yield return new DropoutLayer<T>(0.1);
        }

        yield return new DropoutLayer<T>(0.1);
        yield return new DenseLayer<T>(hiddenDim, numClasses, identityActivation);
    }

    /// <summary>
    /// Creates default DiT (Document Image Transformer) layers.
    /// </summary>
    /// <param name="hiddenDim">Hidden dimension (default: 768).</param>
    /// <param name="numLayers">Number of transformer layers (default: 12).</param>
    /// <param name="numHeads">Number of attention heads (default: 12).</param>
    /// <param name="patchSize">Patch size for ViT (default: 16).</param>
    /// <param name="imageSize">Input image size (default: 224).</param>
    /// <param name="numClasses">Number of output classes (default: 16).</param>
    /// <returns>A collection of layers forming a DiT model.</returns>
    public static IEnumerable<ILayer<T>> CreateDefaultDiTLayers(
        int hiddenDim = 768,
        int numLayers = 12,
        int numHeads = 12,
        int patchSize = 16,
        int imageSize = 224,
        int numClasses = 16)
    {
        IActivationFunction<T> geluActivation = new GELUActivation<T>();
        IActivationFunction<T> identityActivation = new IdentityActivation<T>();
        int intermediateSize = hiddenDim * 4;
        int numPatches = (imageSize / patchSize) * (imageSize / patchSize);

        // Patch embedding (linear projection of flattened patches)
        yield return new ConvolutionalLayer<T>(3, imageSize, imageSize, hiddenDim, patchSize, patchSize, 0);
        yield return new FlattenLayer<T>([hiddenDim, imageSize / patchSize, imageSize / patchSize]);

        // Position embeddings
        yield return new PositionalEncodingLayer<T>(numPatches + 1, hiddenDim);
        yield return new LayerNormalizationLayer<T>(hiddenDim);
        yield return new DropoutLayer<T>(0.1);

        // ViT transformer encoder
        for (int i = 0; i < numLayers; i++)
        {
            yield return new MultiHeadAttentionLayer<T>(numPatches + 1, hiddenDim, numHeads, identityActivation);
            yield return new LayerNormalizationLayer<T>(hiddenDim);
            yield return new DenseLayer<T>(hiddenDim, intermediateSize, geluActivation);
            yield return new DenseLayer<T>(intermediateSize, hiddenDim, identityActivation);
            yield return new LayerNormalizationLayer<T>(hiddenDim);
        }

        // Classification head (from CLS token)
        yield return new LayerNormalizationLayer<T>(hiddenDim);
        yield return new DenseLayer<T>(hiddenDim, numClasses, identityActivation);
    }

    /// <summary>
    /// Creates default LiLT (Language-Independent Layout Transformer) layers.
    /// </summary>
    /// <param name="hiddenDim">Hidden dimension (default: 768).</param>
    /// <param name="numLayers">Number of transformer layers (default: 12).</param>
    /// <param name="numHeads">Number of attention heads (default: 12).</param>
    /// <param name="layoutDim">Layout embedding dimension (default: 768).</param>
    /// <param name="vocabSize">Vocabulary size (default: 30522).</param>
    /// <param name="numClasses">Number of output classes (default: 7).</param>
    /// <returns>A collection of layers forming a LiLT model.</returns>
    public static IEnumerable<ILayer<T>> CreateDefaultLiLTLayers(
        int hiddenDim = 768,
        int numLayers = 12,
        int numHeads = 12,
        int layoutDim = 768,
        int vocabSize = 30522,
        int numClasses = 7)
    {
        IActivationFunction<T> geluActivation = new GELUActivation<T>();
        IActivationFunction<T> identityActivation = new IdentityActivation<T>();
        int intermediateSize = hiddenDim * 4;
        int maxSequenceLength = 512;

        // Text embeddings stream
        yield return new EmbeddingLayer<T>(vocabSize, hiddenDim);
        yield return new PositionalEncodingLayer<T>(maxSequenceLength, hiddenDim);

        // Layout embeddings stream (2D position encoding)
        yield return new DenseLayer<T>(4, layoutDim, identityActivation); // x, y, w, h
        yield return new LayerNormalizationLayer<T>(layoutDim);

        // Dual-stream transformer with BiACM
        for (int i = 0; i < numLayers; i++)
        {
            // Text stream attention
            yield return new MultiHeadAttentionLayer<T>(maxSequenceLength, hiddenDim, numHeads, identityActivation);
            yield return new LayerNormalizationLayer<T>(hiddenDim);

            // Layout stream attention
            yield return new MultiHeadAttentionLayer<T>(maxSequenceLength, layoutDim, numHeads, identityActivation);
            yield return new LayerNormalizationLayer<T>(layoutDim);

            // Feed-forward
            yield return new DenseLayer<T>(hiddenDim, intermediateSize, geluActivation);
            yield return new DenseLayer<T>(intermediateSize, hiddenDim, identityActivation);
            yield return new LayerNormalizationLayer<T>(hiddenDim);
        }

        yield return new DropoutLayer<T>(0.1);
        yield return new DenseLayer<T>(hiddenDim, numClasses, identityActivation);
    }

    /// <summary>
    /// Creates default Dessurt (self-supervised document transformer) layers.
    /// </summary>
    /// <param name="encoderDim">Encoder dimension (default: 768).</param>
    /// <param name="decoderDim">Decoder dimension (default: 768).</param>
    /// <param name="encoderLayers">Number of encoder layers (default: 12).</param>
    /// <param name="decoderLayers">Number of decoder layers (default: 6).</param>
    /// <param name="numHeads">Number of attention heads (default: 12).</param>
    /// <param name="vocabSize">Vocabulary size (default: 50265).</param>
    /// <returns>Encoder and decoder layers for a Dessurt model.</returns>
    public static (IEnumerable<ILayer<T>> EncoderLayers, IEnumerable<ILayer<T>> DecoderLayers) CreateDefaultDessurtLayers(
        int encoderDim = 768,
        int decoderDim = 768,
        int encoderLayers = 12,
        int decoderLayers = 6,
        int numHeads = 12,
        int vocabSize = 50265)
    {
        return (CreateDessurtEncoderLayers(encoderDim, encoderLayers, numHeads),
                CreateDessurtDecoderLayers(decoderDim, decoderLayers, numHeads, vocabSize));
    }

    private static IEnumerable<ILayer<T>> CreateDessurtEncoderLayers(int hiddenDim, int numLayers, int numHeads)
    {
        IActivationFunction<T> geluActivation = new GELUActivation<T>();
        IActivationFunction<T> identityActivation = new IdentityActivation<T>();
        int intermediateSize = hiddenDim * 4;

        // Patch embedding
        yield return new ConvolutionalLayer<T>(3, 224, 224, hiddenDim, 16, 16, 0);
        yield return new LayerNormalizationLayer<T>(hiddenDim);

        for (int i = 0; i < numLayers; i++)
        {
            yield return new MultiHeadAttentionLayer<T>(196, hiddenDim, numHeads, identityActivation);
            yield return new LayerNormalizationLayer<T>(hiddenDim);
            yield return new DenseLayer<T>(hiddenDim, intermediateSize, geluActivation);
            yield return new DenseLayer<T>(intermediateSize, hiddenDim, identityActivation);
            yield return new LayerNormalizationLayer<T>(hiddenDim);
        }
    }

    private static IEnumerable<ILayer<T>> CreateDessurtDecoderLayers(int hiddenDim, int numLayers, int numHeads, int vocabSize)
    {
        IActivationFunction<T> geluActivation = new GELUActivation<T>();
        IActivationFunction<T> identityActivation = new IdentityActivation<T>();
        int intermediateSize = hiddenDim * 4;
        int maxSequenceLength = 512;

        yield return new EmbeddingLayer<T>(vocabSize, hiddenDim);
        yield return new PositionalEncodingLayer<T>(maxSequenceLength, hiddenDim);

        for (int i = 0; i < numLayers; i++)
        {
            yield return new MultiHeadAttentionLayer<T>(maxSequenceLength, hiddenDim, numHeads, identityActivation);
            yield return new LayerNormalizationLayer<T>(hiddenDim);
            yield return new MultiHeadAttentionLayer<T>(maxSequenceLength, hiddenDim, numHeads, identityActivation);
            yield return new LayerNormalizationLayer<T>(hiddenDim);
            yield return new DenseLayer<T>(hiddenDim, intermediateSize, geluActivation);
            yield return new DenseLayer<T>(intermediateSize, hiddenDim, identityActivation);
            yield return new LayerNormalizationLayer<T>(hiddenDim);
        }

        yield return new DenseLayer<T>(hiddenDim, vocabSize, identityActivation);
    }

    /// <summary>
    /// Creates default MATCHA (chart understanding) layers.
    /// </summary>
    /// <param name="encoderDim">Encoder dimension (default: 1536).</param>
    /// <param name="decoderDim">Decoder dimension (default: 1536).</param>
    /// <param name="encoderLayers">Number of encoder layers (default: 18).</param>
    /// <param name="decoderLayers">Number of decoder layers (default: 18).</param>
    /// <param name="numHeads">Number of attention heads (default: 24).</param>
    /// <param name="vocabSize">Vocabulary size (default: 50265).</param>
    /// <param name="maxPatchesPerImage">Maximum patches per image (default: 4096).</param>
    /// <returns>Encoder and decoder layers for a MATCHA model.</returns>
    public static (IEnumerable<ILayer<T>> EncoderLayers, IEnumerable<ILayer<T>> DecoderLayers) CreateDefaultMATCHALayers(
        int encoderDim = 1536,
        int decoderDim = 1536,
        int encoderLayers = 18,
        int decoderLayers = 18,
        int numHeads = 24,
        int vocabSize = 50265,
        int maxPatchesPerImage = 4096)
    {
        return (CreateMATCHAEncoderLayers(encoderDim, encoderLayers, numHeads, maxPatchesPerImage),
                CreateMATCHADecoderLayers(decoderDim, decoderLayers, numHeads, vocabSize));
    }

    private static IEnumerable<ILayer<T>> CreateMATCHAEncoderLayers(int hiddenDim, int numLayers, int numHeads, int maxPatches)
    {
        IActivationFunction<T> geluActivation = new GELUActivation<T>();
        IActivationFunction<T> identityActivation = new IdentityActivation<T>();
        int intermediateSize = hiddenDim * 4;

        yield return new ConvolutionalLayer<T>(3, 64, 64, hiddenDim, 16, 16, 0);
        yield return new LayerNormalizationLayer<T>(hiddenDim);
        yield return new PositionalEncodingLayer<T>(maxPatches, hiddenDim);

        for (int i = 0; i < numLayers; i++)
        {
            yield return new MultiHeadAttentionLayer<T>(Math.Min(maxPatches, 256), hiddenDim, numHeads, identityActivation);
            yield return new LayerNormalizationLayer<T>(hiddenDim);
            yield return new DenseLayer<T>(hiddenDim, intermediateSize, geluActivation);
            yield return new DenseLayer<T>(intermediateSize, hiddenDim, identityActivation);
            yield return new LayerNormalizationLayer<T>(hiddenDim);
        }
    }

    private static IEnumerable<ILayer<T>> CreateMATCHADecoderLayers(int hiddenDim, int numLayers, int numHeads, int vocabSize)
    {
        IActivationFunction<T> geluActivation = new GELUActivation<T>();
        IActivationFunction<T> identityActivation = new IdentityActivation<T>();
        int intermediateSize = hiddenDim * 4;
        int maxSequenceLength = 512;

        yield return new EmbeddingLayer<T>(vocabSize, hiddenDim);
        yield return new PositionalEncodingLayer<T>(maxSequenceLength, hiddenDim);

        for (int i = 0; i < numLayers; i++)
        {
            yield return new MultiHeadAttentionLayer<T>(maxSequenceLength, hiddenDim, numHeads, identityActivation);
            yield return new LayerNormalizationLayer<T>(hiddenDim);
            yield return new MultiHeadAttentionLayer<T>(maxSequenceLength, hiddenDim, numHeads, identityActivation);
            yield return new LayerNormalizationLayer<T>(hiddenDim);
            yield return new DenseLayer<T>(hiddenDim, intermediateSize, geluActivation);
            yield return new DenseLayer<T>(intermediateSize, hiddenDim, identityActivation);
            yield return new LayerNormalizationLayer<T>(hiddenDim);
        }

        yield return new DenseLayer<T>(hiddenDim, vocabSize, identityActivation);
    }

    /// <summary>
    /// Creates default DocOwl (mPLUG-DocOwl) layers for document understanding.
    /// </summary>
    /// <param name="visionDim">Vision encoder dimension (default: 1024).</param>
    /// <param name="textDim">Text encoder dimension (default: 4096).</param>
    /// <param name="visionLayers">Number of vision layers (default: 24).</param>
    /// <param name="textLayers">Number of text layers (default: 32).</param>
    /// <param name="numHeads">Number of attention heads (default: 16).</param>
    /// <param name="vocabSize">Vocabulary size (default: 32000).</param>
    /// <returns>A collection of layers forming a DocOwl model.</returns>
    public static IEnumerable<ILayer<T>> CreateDefaultDocOwlLayers(
        int visionDim = 1024,
        int textDim = 4096,
        int visionLayers = 24,
        int textLayers = 32,
        int numHeads = 16,
        int vocabSize = 32000)
    {
        IActivationFunction<T> siluActivation = new SiLUActivation<T>();
        IActivationFunction<T> identityActivation = new IdentityActivation<T>();

        // Vision encoder (ViT-L style)
        yield return new ConvolutionalLayer<T>(3, 224, 224, visionDim, 14, 14, 0);
        yield return new LayerNormalizationLayer<T>(visionDim);

        for (int i = 0; i < Math.Min(visionLayers, 6); i++)
        {
            yield return new MultiHeadAttentionLayer<T>(256, visionDim, numHeads, identityActivation);
            yield return new LayerNormalizationLayer<T>(visionDim);
            yield return new DenseLayer<T>(visionDim, visionDim * 4, siluActivation);
            yield return new DenseLayer<T>(visionDim * 4, visionDim, identityActivation);
            yield return new LayerNormalizationLayer<T>(visionDim);
        }

        // Visual abstractor projection
        yield return new DenseLayer<T>(visionDim, textDim, identityActivation);

        // LLM decoder layers
        yield return new EmbeddingLayer<T>(vocabSize, textDim);

        for (int i = 0; i < Math.Min(textLayers, 6); i++)
        {
            yield return new MultiHeadAttentionLayer<T>(512, textDim, numHeads, identityActivation);
            yield return new LayerNormalizationLayer<T>(textDim);
            yield return new DenseLayer<T>(textDim, textDim * 4, siluActivation);
            yield return new DenseLayer<T>(textDim * 4, textDim, identityActivation);
            yield return new LayerNormalizationLayer<T>(textDim);
        }

        yield return new DenseLayer<T>(textDim, vocabSize, identityActivation);
    }

    /// <summary>
    /// Creates default InfographicVQA layers for infographic understanding.
    /// </summary>
    /// <param name="imageSize">Input image size (default: 1024).</param>
    /// <param name="visionDim">Vision encoder dimension (default: 768).</param>
    /// <param name="textDim">Text encoder dimension (default: 768).</param>
    /// <param name="fusionDim">Fusion dimension (default: 768).</param>
    /// <param name="visionLayers">Number of vision layers (default: 12).</param>
    /// <param name="fusionLayers">Number of fusion layers (default: 6).</param>
    /// <param name="numHeads">Number of attention heads (default: 12).</param>
    /// <param name="vocabSize">Vocabulary size (default: 30522).</param>
    /// <returns>A collection of layers forming an InfographicVQA model.</returns>
    public static IEnumerable<ILayer<T>> CreateDefaultInfographicVQALayers(
        int imageSize = 1024,
        int visionDim = 768,
        int textDim = 768,
        int fusionDim = 768,
        int visionLayers = 12,
        int fusionLayers = 6,
        int numHeads = 12,
        int vocabSize = 30522)
    {
        IActivationFunction<T> geluActivation = new GELUActivation<T>();
        IActivationFunction<T> identityActivation = new IdentityActivation<T>();

        // Multi-scale vision encoder
        yield return new ConvolutionalLayer<T>(3, imageSize / 16, imageSize / 16, visionDim, 16, 16, 0);
        yield return new LayerNormalizationLayer<T>(visionDim);

        for (int i = 0; i < Math.Min(visionLayers, 6); i++)
        {
            yield return new MultiHeadAttentionLayer<T>(64, visionDim, numHeads, identityActivation);
            yield return new LayerNormalizationLayer<T>(visionDim);
            yield return new DenseLayer<T>(visionDim, visionDim * 4, geluActivation);
            yield return new DenseLayer<T>(visionDim * 4, visionDim, identityActivation);
            yield return new LayerNormalizationLayer<T>(visionDim);
        }

        // Text encoder
        yield return new EmbeddingLayer<T>(vocabSize, textDim);
        yield return new PositionalEncodingLayer<T>(512, textDim);
        yield return new LayerNormalizationLayer<T>(textDim);

        // Fusion layers
        for (int i = 0; i < fusionLayers; i++)
        {
            yield return new MultiHeadAttentionLayer<T>(512, fusionDim, numHeads, identityActivation);
            yield return new LayerNormalizationLayer<T>(fusionDim);
            yield return new DenseLayer<T>(fusionDim, fusionDim * 4, geluActivation);
            yield return new DenseLayer<T>(fusionDim * 4, fusionDim, identityActivation);
            yield return new LayerNormalizationLayer<T>(fusionDim);
        }

        yield return new DenseLayer<T>(fusionDim, vocabSize, identityActivation);
    }

    /// <summary>
    /// Creates default DocGCN (Document Graph Convolutional Network) layers.
    /// </summary>
    /// <param name="inputDim">Input feature dimension (default: 768).</param>
    /// <param name="hiddenDim">Hidden dimension (default: 256).</param>
    /// <param name="numGCNLayers">Number of GCN layers (default: 3).</param>
    /// <param name="numClasses">Number of output classes (default: 7).</param>
    /// <returns>A collection of layers forming a DocGCN model.</returns>
    public static IEnumerable<ILayer<T>> CreateDefaultDocGCNLayers(
        int inputDim = 768,
        int hiddenDim = 256,
        int numGCNLayers = 3,
        int numClasses = 7)
    {
        IActivationFunction<T> reluActivation = new ReLUActivation<T>();
        IActivationFunction<T> identityActivation = new IdentityActivation<T>();

        // Initial projection
        yield return new DenseLayer<T>(inputDim, hiddenDim, reluActivation);
        yield return new DropoutLayer<T>(0.5);

        // GCN layers (using dense as approximation)
        for (int i = 0; i < numGCNLayers; i++)
        {
            yield return new DenseLayer<T>(hiddenDim, hiddenDim, reluActivation);
            yield return new DropoutLayer<T>(0.5);
        }

        // Classification head
        yield return new DenseLayer<T>(hiddenDim, numClasses, identityActivation);
    }

    /// <summary>
    /// Creates default LayoutGraph layers for graph-based layout analysis.
    /// </summary>
    /// <param name="inputDim">Input feature dimension (default: 768).</param>
    /// <param name="hiddenDim">Hidden dimension (default: 256).</param>
    /// <param name="numGraphLayers">Number of graph layers (default: 4).</param>
    /// <param name="numClasses">Number of output classes (default: 7).</param>
    /// <returns>A collection of layers forming a LayoutGraph model.</returns>
    public static IEnumerable<ILayer<T>> CreateDefaultLayoutGraphLayers(
        int inputDim = 768,
        int hiddenDim = 256,
        int numGraphLayers = 4,
        int numClasses = 7)
    {
        IActivationFunction<T> reluActivation = new ReLUActivation<T>();
        IActivationFunction<T> identityActivation = new IdentityActivation<T>();

        // Node feature encoder
        yield return new DenseLayer<T>(inputDim, hiddenDim, reluActivation);
        yield return new LayerNormalizationLayer<T>(hiddenDim);

        // Graph layers with hierarchical structure
        for (int i = 0; i < numGraphLayers; i++)
        {
            yield return new DenseLayer<T>(hiddenDim, hiddenDim, reluActivation);
            yield return new LayerNormalizationLayer<T>(hiddenDim);
            yield return new DropoutLayer<T>(0.3);
        }

        // Reading order and classification heads
        yield return new DenseLayer<T>(hiddenDim, numClasses, identityActivation);
    }

    /// <summary>
    /// Creates default TRIE (Text Reading and Information Extraction) layers.
    /// </summary>
    /// <param name="imageSize">Input image size (default: 512).</param>
    /// <param name="visualDim">Visual encoder dimension (default: 256).</param>
    /// <param name="textDim">Text encoder dimension (default: 256).</param>
    /// <param name="graphDim">Graph dimension (default: 256).</param>
    /// <param name="numEntityTypes">Number of entity types (default: 10).</param>
    /// <param name="maxEntities">Maximum entities (default: 100).</param>
    /// <returns>A collection of layers forming a TRIE model.</returns>
    public static IEnumerable<ILayer<T>> CreateDefaultTRIELayers(
        int imageSize = 512,
        int visualDim = 256,
        int textDim = 256,
        int graphDim = 256,
        int numEntityTypes = 10,
        int maxEntities = 100)
    {
        IActivationFunction<T> reluActivation = new ReLUActivation<T>();
        IActivationFunction<T> identityActivation = new IdentityActivation<T>();

        // Visual encoder (ResNet-style backbone)
        yield return new ConvolutionalLayer<T>(3, imageSize, imageSize, 64, 7, 2, 3);
        yield return new BatchNormalizationLayer<T>(64);
        yield return new MaxPoolingLayer<T>([64, imageSize / 2, imageSize / 2], 3, 2);
        yield return new ConvolutionalLayer<T>(64, imageSize / 4, imageSize / 4, visualDim, 3, 1, 1);

        // Text encoder
        yield return new DenseLayer<T>(textDim, textDim, reluActivation);
        yield return new DenseLayer<T>(textDim, textDim, reluActivation);

        // Graph reasoning module
        yield return new DenseLayer<T>(visualDim + textDim, graphDim, reluActivation);
        yield return new DenseLayer<T>(graphDim, graphDim, reluActivation);

        // Multi-task extraction heads
        yield return new DenseLayer<T>(graphDim, numEntityTypes, identityActivation);
    }

    /// <summary>
    /// Creates default SVTR (Scene Text Visual Transformer Recognizer) layers.
    /// </summary>
    /// <param name="imageWidth">Input image width (default: 256).</param>
    /// <param name="imageHeight">Input image height (default: 64).</param>
    /// <param name="hiddenDim">Hidden dimension (default: 192).</param>
    /// <param name="numLayers">Number of transformer layers (default: 8).</param>
    /// <param name="numHeads">Number of attention heads (default: 6).</param>
    /// <param name="charsetSize">Character set size (default: 95).</param>
    /// <returns>A collection of layers forming an SVTR model.</returns>
    public static IEnumerable<ILayer<T>> CreateDefaultSVTRLayers(
        int imageWidth = 256,
        int imageHeight = 64,
        int hiddenDim = 192,
        int numLayers = 8,
        int numHeads = 6,
        int charsetSize = 95)
    {
        IActivationFunction<T> geluActivation = new GELUActivation<T>();
        IActivationFunction<T> identityActivation = new IdentityActivation<T>();
        int intermediateSize = hiddenDim * 4;
        int seqLen = imageWidth / 4;

        // Patch embedding
        yield return new ConvolutionalLayer<T>(3, imageHeight, imageWidth, 64, 3, 1, 1);
        yield return new BatchNormalizationLayer<T>(64);
        yield return new ConvolutionalLayer<T>(64, imageHeight, imageWidth, hiddenDim, 4, 4, 0);
        yield return new LayerNormalizationLayer<T>(hiddenDim);

        // Transformer layers
        for (int i = 0; i < numLayers; i++)
        {
            yield return new MultiHeadAttentionLayer<T>(seqLen, hiddenDim, numHeads, identityActivation);
            yield return new LayerNormalizationLayer<T>(hiddenDim);
            yield return new DenseLayer<T>(hiddenDim, intermediateSize, geluActivation);
            yield return new DenseLayer<T>(intermediateSize, hiddenDim, identityActivation);
            yield return new LayerNormalizationLayer<T>(hiddenDim);
        }

        // CTC output
        yield return new DenseLayer<T>(hiddenDim, charsetSize, identityActivation);
    }

    /// <summary>
    /// Creates default ABINet (Autonomous, Bidirectional, Iterative) layers.
    /// </summary>
    /// <param name="imageWidth">Input image width (default: 128).</param>
    /// <param name="imageHeight">Input image height (default: 32).</param>
    /// <param name="visionDim">Vision encoder dimension (default: 512).</param>
    /// <param name="languageDim">Language model dimension (default: 512).</param>
    /// <param name="numIterations">Number of refinement iterations (default: 3).</param>
    /// <param name="charsetSize">Character set size (default: 95).</param>
    /// <returns>A collection of layers forming an ABINet model.</returns>
    public static IEnumerable<ILayer<T>> CreateDefaultABINetLayers(
        int imageWidth = 128,
        int imageHeight = 32,
        int visionDim = 512,
        int languageDim = 512,
        int numIterations = 3,
        int charsetSize = 95)
    {
        IActivationFunction<T> reluActivation = new ReLUActivation<T>();
        IActivationFunction<T> identityActivation = new IdentityActivation<T>();
        int seqLen = imageWidth / 4;

        // Vision encoder (ResNet-style)
        yield return new ConvolutionalLayer<T>(3, imageHeight, imageWidth, 64, 3, 1, 1);
        yield return new BatchNormalizationLayer<T>(64);
        yield return new MaxPoolingLayer<T>([64, imageHeight, imageWidth], 2, 2);
        yield return new ConvolutionalLayer<T>(64, imageHeight / 2, imageWidth / 2, 128, 3, 1, 1);
        yield return new BatchNormalizationLayer<T>(128);
        yield return new MaxPoolingLayer<T>([128, imageHeight / 2, imageWidth / 2], 2, 2);
        yield return new ConvolutionalLayer<T>(128, imageHeight / 4, imageWidth / 4, visionDim, 3, 1, 1);

        // Transformer for vision
        yield return new MultiHeadAttentionLayer<T>(seqLen, visionDim, 8, identityActivation);
        yield return new LayerNormalizationLayer<T>(visionDim);

        // Language model
        yield return new EmbeddingLayer<T>(charsetSize, languageDim);
        yield return new MultiHeadAttentionLayer<T>(seqLen, languageDim, 8, identityActivation);
        yield return new LayerNormalizationLayer<T>(languageDim);

        // Fusion with iterative refinement
        for (int i = 0; i < numIterations; i++)
        {
            yield return new DenseLayer<T>(visionDim + languageDim, visionDim, reluActivation);
            yield return new LayerNormalizationLayer<T>(visionDim);
        }

        yield return new DenseLayer<T>(visionDim, charsetSize, identityActivation);
    }

    /// <summary>
    /// Creates default EAST (Efficient and Accurate Scene Text Detector) layers.
    /// </summary>
    /// <param name="imageSize">Input image size (default: 512).</param>
    /// <param name="backboneChannels">Backbone output channels (default: 512).</param>
    /// <param name="featureChannels">Feature map channels (default: 128).</param>
    /// <param name="geometryType">Geometry output type: RBOX or QUAD (default: RBOX).</param>
    /// <returns>A collection of layers forming an EAST model.</returns>
    public static IEnumerable<ILayer<T>> CreateDefaultEASTLayers(
        int imageSize = 512,
        int backboneChannels = 512,
        int featureChannels = 128,
        string geometryType = "RBOX")
    {
        IActivationFunction<T> reluActivation = new ReLUActivation<T>();
        IActivationFunction<T> sigmoidActivation = new SigmoidActivation<T>();
        IActivationFunction<T> identityActivation = new IdentityActivation<T>();

        // Feature extraction backbone (VGG/PVANet style)
        int currentSize = imageSize;
        int[] channels = [64, 128, 256, backboneChannels];

        int inputChannels = 3;
        foreach (int outChannels in channels)
        {
            yield return new ConvolutionalLayer<T>(inputChannels, currentSize, currentSize, outChannels, 3, 1, 1);
            yield return new BatchNormalizationLayer<T>(outChannels);
            yield return new MaxPoolingLayer<T>([outChannels, currentSize, currentSize], 2, 2);
            currentSize /= 2;
            inputChannels = outChannels;
        }

        // Feature merging (U-Net style upsampling)
        yield return new ConvolutionalLayer<T>(backboneChannels, currentSize, currentSize, featureChannels, 1, 1, 0);
        yield return new BatchNormalizationLayer<T>(featureChannels);
        yield return new ConvolutionalLayer<T>(featureChannels, currentSize, currentSize, featureChannels, 3, 1, 1);

        // Output heads
        int geometryChannels = geometryType == "QUAD" ? 8 : 5;
        yield return new ConvolutionalLayer<T>(featureChannels, currentSize, currentSize, 1, 1, 1, 0); // Score map
        yield return new ConvolutionalLayer<T>(featureChannels, currentSize, currentSize, geometryChannels, 1, 1, 0); // Geometry
    }

    /// <summary>
    /// Creates default PSENet (Progressive Scale Expansion Network) layers.
    /// </summary>
    /// <param name="imageSize">Input image size (default: 640).</param>
    /// <param name="backboneChannels">Backbone channels (default: 256).</param>
    /// <param name="featureChannels">Feature channels (default: 256).</param>
    /// <param name="numKernels">Number of scale kernels (default: 7).</param>
    /// <returns>A collection of layers forming a PSENet model.</returns>
    public static IEnumerable<ILayer<T>> CreateDefaultPSENetLayers(
        int imageSize = 640,
        int backboneChannels = 256,
        int featureChannels = 256,
        int numKernels = 7)
    {
        IActivationFunction<T> reluActivation = new ReLUActivation<T>();
        IActivationFunction<T> sigmoidActivation = new SigmoidActivation<T>();

        // ResNet backbone
        int currentSize = imageSize;
        yield return new ConvolutionalLayer<T>(3, currentSize, currentSize, 64, 7, 2, 3);
        yield return new BatchNormalizationLayer<T>(64);
        currentSize /= 2;

        yield return new MaxPoolingLayer<T>([64, currentSize, currentSize], 3, 2);
        currentSize /= 2;

        int[] resnetChannels = [64, 128, backboneChannels, backboneChannels];
        int inputChannels = 64;
        foreach (int outChannels in resnetChannels)
        {
            yield return new ConvolutionalLayer<T>(inputChannels, currentSize, currentSize, outChannels, 3, 1, 1);
            yield return new BatchNormalizationLayer<T>(outChannels);
            inputChannels = outChannels;
            if (outChannels != resnetChannels[^1])
            {
                yield return new MaxPoolingLayer<T>([outChannels, currentSize, currentSize], 2, 2);
                currentSize /= 2;
            }
        }

        // FPN-style feature fusion
        yield return new ConvolutionalLayer<T>(backboneChannels, currentSize, currentSize, featureChannels, 1, 1, 0);
        yield return new ConvolutionalLayer<T>(featureChannels, currentSize, currentSize, featureChannels, 3, 1, 1);

        // Multi-scale kernel output
        yield return new ConvolutionalLayer<T>(featureChannels, currentSize, currentSize, numKernels, 1, 1, 0);
    }

    /// <summary>
    /// Creates default layers for a Siamese neural network using a Transformer-based encoder.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="vocabSize">The size of the vocabulary (default: 30522).</param>
    /// <param name="embeddingDimension">The dimension of the embedding vectors (default: 768).</param>
    /// <param name="maxSequenceLength">The maximum length of input sequences (default: 512).</param>
    /// <returns>A collection of layers forming a Siamese encoder.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Siamese Network uses two identical "twin" networks to process different inputs.
    /// This method sets up the structure for one of those twins, typically using a Transformer encoder
    /// to turn text into a coordinate (embedding) that can be compared to others.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultSiameseLayers(
        NeuralNetworkArchitecture<T> architecture,
        int vocabSize = 30522,
        int embeddingDimension = 768,
        int maxSequenceLength = 512)
    {
        if (vocabSize <= 0) throw new ArgumentOutOfRangeException(nameof(vocabSize));
        if (embeddingDimension <= 0) throw new ArgumentOutOfRangeException(nameof(embeddingDimension));
        if (maxSequenceLength <= 0) throw new ArgumentOutOfRangeException(nameof(maxSequenceLength));
        const int numHeads = 12;
        if (embeddingDimension % numHeads != 0)
            throw new ArgumentException($"embeddingDimension must be divisible by {numHeads}.", nameof(embeddingDimension));

        yield return new EmbeddingLayer<T>(vocabSize, embeddingDimension);
        yield return new PositionalEncodingLayer<T>(maxSequenceLength, embeddingDimension);
        yield return new TransformerEncoderLayer<T>(embeddingDimension, numHeads, 3072);
    }

    /// <summary>
    /// Creates default layers for a Word2Vec model (Skip-Gram or CBOW).
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="vocabSize">The size of the vocabulary.</param>
    /// <param name="embeddingDimension">The dimension of the embedding vectors.</param>
    /// <returns>A collection of layers forming a Word2Vec model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Word2Vec learns to represent words as vectors of numbers (embeddings)
    /// such that words with similar meanings are close to each other.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultWord2VecLayers(
        NeuralNetworkArchitecture<T> architecture,
        int vocabSize,
        int embeddingDimension)
    {
        if (architecture is null) throw new ArgumentNullException(nameof(architecture));
        if (vocabSize < 1) throw new ArgumentOutOfRangeException(nameof(vocabSize));
        if (embeddingDimension < 1) throw new ArgumentOutOfRangeException(nameof(embeddingDimension));
        if (architecture.OutputSize > 0 && architecture.OutputSize != vocabSize)
            throw new ArgumentException("architecture.OutputSize must match vocabSize for Word2Vec softmax output.", nameof(architecture));

        // 1. Target word embeddings (U matrix)
        yield return new EmbeddingLayer<T>(vocabSize, embeddingDimension);

        // 2. Context word projection (V matrix)
        // Maps embedding space back to vocabulary for prediction
        yield return new DenseLayer<T>(embeddingDimension, vocabSize, (IActivationFunction<T>?)null);

        // 3. Output activation
        yield return new ActivationLayer<T>([vocabSize], new SoftmaxActivation<T>() as IVectorActivationFunction<T>);
    }

    /// <summary>
    /// Creates default layers for a GloVe (Global Vectors) model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="vocabSize">The size of the vocabulary.</param>
    /// <param name="embeddingDimension">The dimension of the embedding vectors.</param>
    /// <returns>A collection of layers forming a GloVe model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> GloVe creates word embeddings by learning from the co-occurrence
    /// statistics of words. It uses two sets of embeddings and two sets of biases.
    /// </para>
    /// <para>
    /// <b>Note:</b> The layers returned by this method are <b>not</b> intended to be used as a sequential
    /// feed-forward stack. They represent the four components (W, W_tilde, b, b_tilde) required for
    /// the GloVe model's custom forward pass.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultGloVeLayers(
        NeuralNetworkArchitecture<T> architecture,
        int vocabSize,
        int embeddingDimension)
    {
        if (architecture is null) throw new ArgumentNullException(nameof(architecture));
        if (vocabSize < 1) throw new ArgumentOutOfRangeException(nameof(vocabSize));
        if (embeddingDimension < 1) throw new ArgumentOutOfRangeException(nameof(embeddingDimension));

        // GloVe training is typically dot(W_i, W_tilde_j) + b_i + b_tilde_j = log(X_ij)
        // To represent this sequentially for standard backprop:
        // Input is a pair of indices (i, j).
        // This is tricky for a strictly sequential ILayer stack.
        // However, for inference/embedding lookup, we just need W and W_tilde.
        yield return new EmbeddingLayer<T>(vocabSize, embeddingDimension); // W
        yield return new EmbeddingLayer<T>(vocabSize, embeddingDimension); // W_tilde
        yield return new EmbeddingLayer<T>(vocabSize, 1); // b
        yield return new EmbeddingLayer<T>(vocabSize, 1); // b_tilde
    }

    /// <summary>
    /// Creates default layers for a FastText model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="vocabSize">The size of the vocabulary.</param>
    /// <param name="bucketSize">The number of buckets for n-gram hashing.</param>
    /// <param name="embeddingDimension">The dimension of the embedding vectors.</param>
    /// <returns>A collection of layers forming a FastText model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> FastText improves on Word2Vec by considering sub-word information
    /// (character n-grams). It represents words as the sum of their n-gram embeddings.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultFastTextLayers(
        NeuralNetworkArchitecture<T> architecture,
        int vocabSize,
        int bucketSize,
        int embeddingDimension)
    {
        if (architecture is null) throw new ArgumentNullException(nameof(architecture));
        if (vocabSize < 1) throw new ArgumentOutOfRangeException(nameof(vocabSize));
        if (bucketSize < 1) throw new ArgumentOutOfRangeException(nameof(bucketSize));
        if (embeddingDimension < 1) throw new ArgumentOutOfRangeException(nameof(embeddingDimension));
        if (architecture.OutputSize > 0 && architecture.OutputSize != vocabSize)
            throw new ArgumentException("architecture.OutputSize must match vocabSize for FastText softmax output.", nameof(architecture));

        // FastText architecture:
        // 1. Word Embeddings
        yield return new EmbeddingLayer<T>(vocabSize, embeddingDimension);

        // 2. N-gram Embeddings
        yield return new EmbeddingLayer<T>(bucketSize, embeddingDimension);

        // 3. Context word projection (similar to Word2Vec)
        yield return new DenseLayer<T>(embeddingDimension, vocabSize, (IActivationFunction<T>?)null);

        // 4. Output activation
        yield return new ActivationLayer<T>([vocabSize], new SoftmaxActivation<T>() as IVectorActivationFunction<T>);
    }

    /// <summary>
    /// Creates default layers for a BLIP-2 neural network.
    /// </summary>
    public static IEnumerable<ILayer<T>> CreateDefaultBlip2Layers(
        int imageSize = 224,
        int channels = 3,
        int patchSize = 14,
        int vocabularySize = 30522,
        int embeddingDimension = 256,
        int qformerHiddenDim = 768,
        int visionHiddenDim = 1408,
        int lmHiddenDim = 2560,
        int numQformerLayers = 12,
        int numHeads = 12,
        int numLmDecoderLayers = 6,
        int maxSequenceLength = 32)
    {
        // 1. Vision encoder: Patch embedding
        yield return new PatchEmbeddingLayer<T>(
            imageSize, imageSize, channels, patchSize, visionHiddenDim);

        // 2. Q-Former layers (self-attention, cross-attention, feed-forward)
        int feedForwardDim = qformerHiddenDim * 4;
        for (int i = 0; i < numQformerLayers; i++)
        {
            // Self-attention for queries
            yield return new TransformerEncoderLayer<T>(qformerHiddenDim, numHeads, feedForwardDim);

            // Cross-attention from queries to vision features
            yield return new TransformerEncoderLayer<T>(qformerHiddenDim, numHeads, feedForwardDim);

            // Feed-forward
            yield return new DenseLayer<T>(qformerHiddenDim, qformerHiddenDim, (IActivationFunction<T>?)null);
        }

        // 3. Text embedding for Q-Former
        yield return new EmbeddingLayer<T>(vocabularySize, qformerHiddenDim);

        // 4. Projection heads
        yield return new DenseLayer<T>(qformerHiddenDim, 2, (IActivationFunction<T>?)null); // ITM
        yield return new DenseLayer<T>(qformerHiddenDim, embeddingDimension, (IActivationFunction<T>?)null); // ITC
        yield return new DenseLayer<T>(qformerHiddenDim, lmHiddenDim, (IActivationFunction<T>?)null); // LM Projection

        // 5. LM Decoder layers
        int lmFeedForwardDim = lmHiddenDim * 4;
        int lmNumHeads = Math.Max(8, lmHiddenDim / 64);
        var geluActivation = new GELUActivation<T>();
        for (int i = 0; i < numLmDecoderLayers; i++)
        {
            yield return new TransformerDecoderLayer<T>(
                embeddingSize: lmHiddenDim,
                numHeads: lmNumHeads,
                feedForwardDim: lmFeedForwardDim,
                sequenceLength: maxSequenceLength,
                ffnActivation: geluActivation);
        }

        // 6. LM Head
        yield return new DenseLayer<T>(lmHiddenDim, vocabularySize, (IActivationFunction<T>?)null);
    }

    /// <summary>
    /// Creates default layers for a SimCSE (Simple Contrastive Learning of Sentence Embeddings) model.
    /// </summary>
    public static IEnumerable<ILayer<T>> CreateDefaultSimCSELayers(
        NeuralNetworkArchitecture<T> architecture,
        int vocabSize = 30522,
        int embeddingDimension = 768,
        int maxSequenceLength = 512,
        int numLayers = 12,
        int numHeads = 12,
        int feedForwardDim = 3072)
    {
        // 1. Embedding Layer
        yield return new EmbeddingLayer<T>(vocabSize, embeddingDimension);

        // 2. Positional Encoding
        yield return new PositionalEncodingLayer<T>(maxSequenceLength, embeddingDimension);

        // 3. Transformer Encoder Layers (SimCSE uses standard BERT/RoBERTa stacks)
        for (int i = 0; i < numLayers; i++)
        {
            yield return new TransformerEncoderLayer<T>(embeddingDimension, numHeads, feedForwardDim);
        }

        // 4. MLM Head (Optional, but often used in SimCSE training)
        yield return new DenseLayer<T>(embeddingDimension, vocabSize, (IActivationFunction<T>?)null);
    }

    /// <summary>
    /// Creates default layers for a ColBERT (Contextualized Late Interaction over BERT) model.
    /// </summary>
    public static IEnumerable<ILayer<T>> CreateDefaultColBERTLayers(
        NeuralNetworkArchitecture<T> architecture,
        int vocabSize = 30522,
        int embeddingDimension = 128, // ColBERT typically projects down to smaller dimension
        int maxSequenceLength = 512,
        int numLayers = 12,
        int numHeads = 12,
        int feedForwardDim = 3072)
    {
        // 1. Base Encoder (e.g. BERT)
        yield return new EmbeddingLayer<T>(vocabSize, 768);
        yield return new PositionalEncodingLayer<T>(maxSequenceLength, 768);
        for (int i = 0; i < numLayers; i++)
        {
            yield return new TransformerEncoderLayer<T>(768, numHeads, feedForwardDim);
        }

        // 2. Projection Layer (maps to token-level late interaction embeddings)
        yield return new DenseLayer<T>(768, embeddingDimension, (IActivationFunction<T>?)null);
    }

    /// <summary>
    /// Creates default layers for a SPLADE (Sparse Lexical and Expansion Model) embedding model.
    /// </summary>
    public static IEnumerable<ILayer<T>> CreateDefaultSPLADELayers(
        NeuralNetworkArchitecture<T> architecture,
        int vocabSize = 30522,
        int embeddingDimension = 768,
        int maxSequenceLength = 512,
        int numLayers = 12,
        int numHeads = 12,
        int feedForwardDim = 3072)
    {
        // 1. Base Encoder
        yield return new EmbeddingLayer<T>(vocabSize, embeddingDimension);
        yield return new PositionalEncodingLayer<T>(maxSequenceLength, embeddingDimension);
        for (int i = 0; i < numLayers; i++)
        {
            yield return new TransformerEncoderLayer<T>(embeddingDimension, numHeads, feedForwardDim);
        }

        // 2. SPLADE Head (maps token representations back to vocabulary log-space)
        yield return new DenseLayer<T>(embeddingDimension, vocabSize, new ReLUActivation<T>() as IActivationFunction<T>);
    }

    /// <summary>
    /// Creates default layers for a Matryoshka Representation Learning (MRL) model.
    /// </summary>
    public static IEnumerable<ILayer<T>> CreateDefaultMRLLayers(
        NeuralNetworkArchitecture<T> architecture,
        int vocabSize = 30522,
        int maxEmbeddingDimension = 1536,
        int maxSequenceLength = 512,
        int numLayers = 12,
        int numHeads = 12,
        int feedForwardDim = 3072)
    {
        // 1. Base Encoder
        yield return new EmbeddingLayer<T>(vocabSize, 768);
        yield return new PositionalEncodingLayer<T>(maxSequenceLength, 768);
        for (int i = 0; i < numLayers; i++)
        {
            yield return new TransformerEncoderLayer<T>(768, numHeads, feedForwardDim);
        }

        // 2. Final projection to max Matryoshka dimension
        yield return new DenseLayer<T>(768, maxEmbeddingDimension, (IActivationFunction<T>?)null);
    }

    /// <summary>
    /// Creates default layers for an Instructor/E5 (Instruction-Tuned) embedding model.
    /// </summary>
    public static IEnumerable<ILayer<T>> CreateDefaultInstructorLayers(
        NeuralNetworkArchitecture<T> architecture,
        int vocabSize = 30522,
        int embeddingDimension = 768,
        int maxSequenceLength = 512,
        int numLayers = 12,
        int numHeads = 12,
        int feedForwardDim = 3072)
    {
        // 1. Base Encoder
        yield return new EmbeddingLayer<T>(vocabSize, embeddingDimension);
        yield return new PositionalEncodingLayer<T>(maxSequenceLength, embeddingDimension);
        for (int i = 0; i < numLayers; i++)
        {
            yield return new TransformerEncoderLayer<T>(embeddingDimension, numHeads, feedForwardDim);
        }

        // 2. Instruction Pooling/Projection
        yield return new DenseLayer<T>(embeddingDimension, embeddingDimension, (IActivationFunction<T>?)null);
    }

    /// <summary>
    /// Creates default layers for a BGE (BAAI General Embedding) model.
    /// </summary>
    public static IEnumerable<ILayer<T>> CreateDefaultBGELayers(
        NeuralNetworkArchitecture<T> architecture,
        int vocabSize = 30522,
        int embeddingDimension = 768,
        int maxSequenceLength = 512,
        int numLayers = 12,
        int numHeads = 12,
        int feedForwardDim = 3072)
    {
        // BGE is typically a BERT-style encoder with specialized multi-stage training
        yield return new EmbeddingLayer<T>(vocabSize, embeddingDimension);
        yield return new PositionalEncodingLayer<T>(maxSequenceLength, embeddingDimension);
        for (int i = 0; i < numLayers; i++)
        {
            yield return new TransformerEncoderLayer<T>(embeddingDimension, numHeads, feedForwardDim);
        }

        // BGE often uses an additional normalization/projection head for retrieval
        yield return new LayerNormalizationLayer<T>(embeddingDimension);
    }

    /// <summary>
    /// Creates default layers for an SGPT (Sentence GPT) decoder-only embedding model.
    /// </summary>
    public static IEnumerable<ILayer<T>> CreateDefaultSGPTLayers(
        NeuralNetworkArchitecture<T> architecture,
        int vocabSize = 50257, // GPT-2 vocab size
        int embeddingDimension = 768,
        int maxSequenceLength = 1024,
        int numLayers = 12,
        int numHeads = 12,
        int feedForwardDim = 3072)
    {
        // SGPT uses a decoder-only architecture (e.g. GPT-2, GPT-Neo)
        yield return new EmbeddingLayer<T>(vocabSize, embeddingDimension);
        yield return new PositionalEncodingLayer<T>(maxSequenceLength, embeddingDimension);

        for (int i = 0; i < numLayers; i++)
        {
            // Note: In a real decoder, we would use TransformerDecoderLayer with causal masking
            // For embedding extraction, a simple TransformerEncoderLayer is often used as a proxy
            // if we aren't doing autoregressive generation.
            yield return new TransformerEncoderLayer<T>(embeddingDimension, numHeads, feedForwardDim);
        }

        // SGPT uses the last token's representation
        yield return new DenseLayer<T>(embeddingDimension, embeddingDimension, (IActivationFunction<T>?)null);
    }

    #endregion

    #region Finance Model Layers

    /// <summary>
    /// Creates the default layers for a PatchTST (Patch Time Series Transformer) network.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="sequenceLength">Input sequence length.</param>
    /// <param name="predictionHorizon">Number of future time steps to predict.</param>
    /// <param name="numFeatures">Number of input features (channels).</param>
    /// <param name="patchSize">Size of each patch (segment of time series).</param>
    /// <param name="stride">Stride between consecutive patches.</param>
    /// <param name="numLayers">Number of transformer encoder layers.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="modelDimension">Model dimension (embedding size).</param>
    /// <param name="feedForwardDimension">Feedforward network dimension.</param>
    /// <returns>A collection of layers forming a PatchTST network.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> PatchTST (Patch Time Series Transformer) is a modern architecture
    /// for time series forecasting that works by dividing the input into patches (segments)
    /// and processing them with a transformer encoder.
    /// </para>
    /// <para>
    /// Think of it like reading a book by looking at groups of words (patches) instead of
    /// individual letters. This makes it faster and often more accurate for time series forecasting.
    /// </para>
    /// <para>
    /// The architecture includes:
    /// <list type="bullet">
    /// <item>Patch embedding: Converts each patch into a dense vector representation</item>
    /// <item>Transformer encoder layers: Process the patches using self-attention</item>
    /// <item>Layer normalization: Stabilizes training</item>
    /// <item>Output projection: Maps the encoder output to the prediction horizon</item>
    /// </list>
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultPatchTSTLayers(
        NeuralNetworkArchitecture<T> architecture,
        int sequenceLength = 96,
        int predictionHorizon = 24,
        int numFeatures = 7,
        int patchSize = 16,
        int stride = 8,
        int numLayers = 3,
        int numHeads = 4,
        int modelDimension = 128,
        int feedForwardDimension = 256)
    {
        // Validate parameters
        if (sequenceLength < patchSize)
            throw new ArgumentException("Sequence length must be at least patch size.", nameof(sequenceLength));
        if (patchSize < 1)
            throw new ArgumentOutOfRangeException(nameof(patchSize), "Patch size must be at least 1.");
        if (stride < 1 || stride > patchSize)
            throw new ArgumentOutOfRangeException(nameof(stride), "Stride must be between 1 and patch size.");
        if (modelDimension % numHeads != 0)
            throw new ArgumentException("Model dimension must be divisible by number of heads.", nameof(modelDimension));

        // Calculate number of patches
        int numPatches = (sequenceLength - patchSize) / stride + 1;

        // Patch embedding: maps each patch from patchSize to modelDimension
        yield return new DenseLayer<T>(
            inputSize: patchSize,
            outputSize: modelDimension,
            activationFunction: new ReLUActivation<T>());

        // Transformer encoder layers
        for (int i = 0; i < numLayers; i++)
        {
            yield return new TransformerEncoderLayer<T>(
                embeddingSize: modelDimension,
                numHeads: numHeads,
                feedForwardDim: feedForwardDimension);
        }

        // Final layer normalization
        yield return new LayerNormalizationLayer<T>(modelDimension);

        // Output projection: maps from (numPatches * modelDimension) to predictionHorizon
        yield return new DenseLayer<T>(
            inputSize: numPatches * modelDimension,
            outputSize: predictionHorizon,
            activationFunction: null);
    }

    /// <summary>
    /// Creates the default layer configuration for an iTransformer (Inverted Transformer) model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="sequenceLength">The input sequence length (default: 96).</param>
    /// <param name="predictionHorizon">The number of future time steps to predict (default: 96).</param>
    /// <param name="numFeatures">The number of input features/variables (default: 7).</param>
    /// <param name="numLayers">The number of transformer encoder layers (default: 2).</param>
    /// <param name="numHeads">The number of attention heads (default: 8).</param>
    /// <param name="modelDimension">The model embedding dimension (default: 512).</param>
    /// <param name="feedForwardDimension">The feedforward network dimension (default: 512).</param>
    /// <returns>A collection of layers forming an iTransformer network.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> iTransformer "inverts" the traditional transformer approach for time series.
    /// Instead of treating each time step as a token (like PatchTST), iTransformer treats each
    /// variable/feature as a token. This allows the model to learn relationships between variables
    /// (like how price relates to volume) using the attention mechanism.
    /// </para>
    /// <para>
    /// The architecture consists of:
    /// <list type="bullet">
    /// <item>Variate embedding: Embeds the entire time series of each variable into a token</item>
    /// <item>Transformer encoders: Learn cross-variable dependencies via attention</item>
    /// <item>Output projection: Maps learned representations to predictions</item>
    /// </list>
    /// </para>
    /// <para>
    /// <b>Reference:</b> Liu et al., "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting",
    /// ICLR 2024.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultITransformerLayers(
        NeuralNetworkArchitecture<T> architecture,
        int sequenceLength = 96,
        int predictionHorizon = 96,
        int numFeatures = 7,
        int numLayers = 2,
        int numHeads = 8,
        int modelDimension = 512,
        int feedForwardDimension = 512)
    {
        // Validate parameters
        if (sequenceLength < 1)
            throw new ArgumentOutOfRangeException(nameof(sequenceLength), "Sequence length must be at least 1.");
        if (predictionHorizon < 1)
            throw new ArgumentOutOfRangeException(nameof(predictionHorizon), "Prediction horizon must be at least 1.");
        if (numFeatures < 1)
            throw new ArgumentOutOfRangeException(nameof(numFeatures), "Number of features must be at least 1.");
        if (numLayers < 1)
            throw new ArgumentOutOfRangeException(nameof(numLayers), "Number of layers must be at least 1.");
        if (modelDimension % numHeads != 0)
            throw new ArgumentException("Model dimension must be divisible by number of heads.", nameof(modelDimension));

        // Variate embedding: maps each variable's time series to a token embedding
        // Input: [batch, seq_len] per variable -> Output: [batch, model_dim] per variable
        yield return new DenseLayer<T>(
            inputSize: sequenceLength,
            outputSize: modelDimension,
            activationFunction: new ReLUActivation<T>());

        // Transformer encoder layers with attention across variables
        for (int i = 0; i < numLayers; i++)
        {
            yield return new TransformerEncoderLayer<T>(
                embeddingSize: modelDimension,
                numHeads: numHeads,
                feedForwardDim: feedForwardDimension);
        }

        // Final layer normalization
        yield return new LayerNormalizationLayer<T>(modelDimension);

        // Output projection: maps from model dimension to prediction horizon
        // Each variable token produces its own forecast
        yield return new DenseLayer<T>(
            inputSize: modelDimension,
            outputSize: predictionHorizon,
            activationFunction: null);
    }

    /// <summary>
    /// Creates the default layer configuration for a FEDformer (Frequency Enhanced Decomposed Transformer) model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="sequenceLength">The input sequence length (default: 96).</param>
    /// <param name="predictionHorizon">The number of future time steps to predict (default: 96).</param>
    /// <param name="numFeatures">The number of input features/variables (default: 7).</param>
    /// <param name="numEncoderLayers">The number of encoder layers (default: 2).</param>
    /// <param name="numDecoderLayers">The number of decoder layers (default: 1).</param>
    /// <param name="numHeads">The number of attention heads (default: 8).</param>
    /// <param name="modelDimension">The model embedding dimension (default: 512).</param>
    /// <param name="feedForwardDimension">The feedforward network dimension (default: 2048).</param>
    /// <returns>A collection of layers forming a FEDformer network.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> FEDformer uses frequency-domain attention instead of standard attention,
    /// which makes it much faster (linear complexity vs quadratic). It also decomposes the time series
    /// into trend and seasonal components for better interpretability.
    /// </para>
    /// <para>
    /// The architecture consists of:
    /// <list type="bullet">
    /// <item>Input embedding: Projects input features to model dimension</item>
    /// <item>Encoder: Processes input with frequency-enhanced attention</item>
    /// <item>Decoder: Generates predictions using cross-attention with encoder</item>
    /// <item>Output projection: Maps to final predictions</item>
    /// </list>
    /// </para>
    /// <para>
    /// <b>Reference:</b> Zhou et al., "FEDformer: Frequency Enhanced Decomposed Transformer",
    /// ICML 2022.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultFEDformerLayers(
        NeuralNetworkArchitecture<T> architecture,
        int sequenceLength = 96,
        int predictionHorizon = 96,
        int numFeatures = 7,
        int numEncoderLayers = 2,
        int numDecoderLayers = 1,
        int numHeads = 8,
        int modelDimension = 512,
        int feedForwardDimension = 2048)
    {
        // Validate parameters
        if (sequenceLength < 1)
            throw new ArgumentOutOfRangeException(nameof(sequenceLength), "Sequence length must be at least 1.");
        if (predictionHorizon < 1)
            throw new ArgumentOutOfRangeException(nameof(predictionHorizon), "Prediction horizon must be at least 1.");
        if (numEncoderLayers < 1)
            throw new ArgumentOutOfRangeException(nameof(numEncoderLayers), "Number of encoder layers must be at least 1.");
        if (modelDimension % numHeads != 0)
            throw new ArgumentException("Model dimension must be divisible by number of heads.", nameof(modelDimension));

        // Input embedding
        yield return new DenseLayer<T>(
            inputSize: numFeatures,
            outputSize: modelDimension,
            activationFunction: new ReLUActivation<T>());

        // Encoder layers (using standard transformer encoder as approximation)
        for (int i = 0; i < numEncoderLayers; i++)
        {
            yield return new TransformerEncoderLayer<T>(
                embeddingSize: modelDimension,
                numHeads: numHeads,
                feedForwardDim: feedForwardDimension);
        }

        // Decoder layers
        for (int i = 0; i < numDecoderLayers; i++)
        {
            yield return new TransformerEncoderLayer<T>(
                embeddingSize: modelDimension,
                numHeads: numHeads,
                feedForwardDim: feedForwardDimension);
        }

        // Final layer normalization
        yield return new LayerNormalizationLayer<T>(modelDimension);

        // Output projection
        yield return new DenseLayer<T>(
            inputSize: modelDimension,
            outputSize: numFeatures,
            activationFunction: null);
    }

    /// <summary>
    /// Creates the default layer configuration for an Autoformer model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="sequenceLength">The input sequence length (default: 96).</param>
    /// <param name="predictionHorizon">The number of future time steps to predict (default: 96).</param>
    /// <param name="numFeatures">The number of input features/variables (default: 7).</param>
    /// <param name="numEncoderLayers">The number of encoder layers (default: 2).</param>
    /// <param name="numDecoderLayers">The number of decoder layers (default: 1).</param>
    /// <param name="numHeads">The number of attention heads (default: 8).</param>
    /// <param name="modelDimension">The model embedding dimension (default: 512).</param>
    /// <param name="feedForwardDimension">The feedforward network dimension (default: 2048).</param>
    /// <returns>A collection of layers forming an Autoformer network.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Autoformer replaces standard attention with "Auto-Correlation" which
    /// discovers period-based dependencies by measuring similarity between different time lags.
    /// It's like finding repeating patterns in music by checking if the beat matches at different delays.
    /// </para>
    /// <para>
    /// Key innovations:
    /// <list type="bullet">
    /// <item>Auto-Correlation: Finds periodic patterns efficiently using FFT</item>
    /// <item>Series Decomposition: Separates trend from seasonal patterns progressively</item>
    /// <item>O(L log L) complexity: Much faster than standard attention</item>
    /// </list>
    /// </para>
    /// <para>
    /// <b>Reference:</b> Wu et al., "Autoformer: Decomposition Transformers with Auto-Correlation",
    /// NeurIPS 2021.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultAutoformerLayers(
        NeuralNetworkArchitecture<T> architecture,
        int sequenceLength = 96,
        int predictionHorizon = 96,
        int numFeatures = 7,
        int numEncoderLayers = 2,
        int numDecoderLayers = 1,
        int numHeads = 8,
        int modelDimension = 512,
        int feedForwardDimension = 2048)
    {
        // Validate parameters
        if (sequenceLength < 1)
            throw new ArgumentOutOfRangeException(nameof(sequenceLength), "Sequence length must be at least 1.");
        if (predictionHorizon < 1)
            throw new ArgumentOutOfRangeException(nameof(predictionHorizon), "Prediction horizon must be at least 1.");
        if (modelDimension % numHeads != 0)
            throw new ArgumentException("Model dimension must be divisible by number of heads.", nameof(modelDimension));

        // Input embedding
        yield return new DenseLayer<T>(
            inputSize: numFeatures,
            outputSize: modelDimension,
            activationFunction: new ReLUActivation<T>());

        // Encoder layers with auto-correlation (using transformer encoder as base)
        for (int i = 0; i < numEncoderLayers; i++)
        {
            yield return new TransformerEncoderLayer<T>(
                embeddingSize: modelDimension,
                numHeads: numHeads,
                feedForwardDim: feedForwardDimension);
        }

        // Decoder layers
        for (int i = 0; i < numDecoderLayers; i++)
        {
            yield return new TransformerEncoderLayer<T>(
                embeddingSize: modelDimension,
                numHeads: numHeads,
                feedForwardDim: feedForwardDimension);
        }

        // Final layer normalization
        yield return new LayerNormalizationLayer<T>(modelDimension);

        // Output projection
        yield return new DenseLayer<T>(
            inputSize: modelDimension,
            outputSize: numFeatures,
            activationFunction: null);
    }

    /// <summary>
    /// Creates the default layer configuration for an Informer model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="sequenceLength">The input sequence length (default: 96).</param>
    /// <param name="predictionHorizon">The number of future time steps to predict (default: 96).</param>
    /// <param name="numFeatures">The number of input features/variables (default: 7).</param>
    /// <param name="numEncoderLayers">The number of encoder layers (default: 2).</param>
    /// <param name="numDecoderLayers">The number of decoder layers (default: 1).</param>
    /// <param name="numHeads">The number of attention heads (default: 8).</param>
    /// <param name="modelDimension">The model embedding dimension (default: 512).</param>
    /// <param name="feedForwardDimension">The feedforward network dimension (default: 2048).</param>
    /// <returns>A collection of layers forming an Informer network.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Informer was one of the first transformers designed specifically for
    /// long sequence time series forecasting. It introduces "ProbSparse" attention which only
    /// computes attention for the most important query-key pairs, making it much faster.
    /// </para>
    /// <para>
    /// Key innovations:
    /// <list type="bullet">
    /// <item>ProbSparse Attention: Only attends to top-k important positions</item>
    /// <item>Self-attention Distilling: Progressively reduces sequence length</item>
    /// <item>Generative Decoder: Predicts all future values at once</item>
    /// </list>
    /// </para>
    /// <para>
    /// <b>Reference:</b> Zhou et al., "Informer: Beyond Efficient Transformer for Long Sequence
    /// Time-Series Forecasting", AAAI 2021.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultInformerLayers(
        NeuralNetworkArchitecture<T> architecture,
        int sequenceLength = 96,
        int predictionHorizon = 96,
        int numFeatures = 7,
        int numEncoderLayers = 2,
        int numDecoderLayers = 1,
        int numHeads = 8,
        int modelDimension = 512,
        int feedForwardDimension = 2048)
    {
        // Validate parameters
        if (sequenceLength < 1)
            throw new ArgumentOutOfRangeException(nameof(sequenceLength), "Sequence length must be at least 1.");
        if (predictionHorizon < 1)
            throw new ArgumentOutOfRangeException(nameof(predictionHorizon), "Prediction horizon must be at least 1.");
        if (modelDimension % numHeads != 0)
            throw new ArgumentException("Model dimension must be divisible by number of heads.", nameof(modelDimension));

        // Input embedding
        yield return new DenseLayer<T>(
            inputSize: numFeatures,
            outputSize: modelDimension,
            activationFunction: new ReLUActivation<T>());

        // Encoder layers with ProbSparse attention (using transformer encoder as approximation)
        for (int i = 0; i < numEncoderLayers; i++)
        {
            yield return new TransformerEncoderLayer<T>(
                embeddingSize: modelDimension,
                numHeads: numHeads,
                feedForwardDim: feedForwardDimension);
        }

        // Decoder layers
        for (int i = 0; i < numDecoderLayers; i++)
        {
            yield return new TransformerEncoderLayer<T>(
                embeddingSize: modelDimension,
                numHeads: numHeads,
                feedForwardDim: feedForwardDimension);
        }

        // Final layer normalization
        yield return new LayerNormalizationLayer<T>(modelDimension);

        // Output projection
        yield return new DenseLayer<T>(
            inputSize: modelDimension,
            outputSize: numFeatures,
            activationFunction: null);
    }

    /// <summary>
    /// Creates the default layers for a Temporal Fusion Transformer (TFT) architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="sequenceLength">Input sequence length (lookback window).</param>
    /// <param name="predictionHorizon">Number of future steps to predict.</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="hiddenSize">Hidden state size for LSTM and attention.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="numLayers">Number of GRN layers.</param>
    /// <param name="dropout">Dropout rate for regularization.</param>
    /// <returns>An enumerable of layers configured for TFT.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> TFT uses a combination of LSTM, attention, and gating mechanisms:
    /// </para>
    /// <para>
    /// <list type="bullet">
    /// <item><b>Variable Selection:</b> Learns which features are most important</item>
    /// <item><b>LSTM:</b> Captures local temporal patterns</item>
    /// <item><b>GRN:</b> Gated Residual Networks for flexible processing</item>
    /// <item><b>Attention:</b> Focuses on important time periods</item>
    /// </list>
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultTFTLayers(
        NeuralNetworkArchitecture<T> architecture,
        int sequenceLength = 24,
        int predictionHorizon = 6,
        int numFeatures = 7,
        int hiddenSize = 128,
        int numHeads = 4,
        int numLayers = 2,
        double dropout = 0.1)
    {
        // Validate parameters
        if (sequenceLength < 1)
            throw new ArgumentOutOfRangeException(nameof(sequenceLength), "Sequence length must be at least 1.");
        if (predictionHorizon < 1)
            throw new ArgumentOutOfRangeException(nameof(predictionHorizon), "Prediction horizon must be at least 1.");
        if (hiddenSize % numHeads != 0)
            throw new ArgumentException("Hidden size must be divisible by number of heads.", nameof(hiddenSize));

        // Variable selection networks (TFT uses 3: static, encoder, decoder)
        // Static variable selection: processes time-invariant features
        yield return new DenseLayer<T>(
            inputSize: numFeatures,
            outputSize: hiddenSize,
            activationFunction: new ReLUActivation<T>());

        // Encoder variable selection: processes historical time-varying features
        yield return new DenseLayer<T>(
            inputSize: numFeatures,
            outputSize: hiddenSize,
            activationFunction: new ReLUActivation<T>());

        // Decoder variable selection: processes known future features
        yield return new DenseLayer<T>(
            inputSize: numFeatures,
            outputSize: hiddenSize,
            activationFunction: new ReLUActivation<T>());

        // LSTM Encoder
        // Input shape: [batch, sequence, features] where features = hiddenSize after variable selection
        yield return new LSTMLayer<T>(
            inputSize: hiddenSize,
            hiddenSize: hiddenSize,
            inputShape: new[] { 1, sequenceLength, hiddenSize },
            activation: (IActivationFunction<T>?)null,
            recurrentActivation: null,
            engine: null);

        // LSTM Decoder
        // Input shape: [batch, sequence, features] where features = hiddenSize from encoder
        yield return new LSTMLayer<T>(
            inputSize: hiddenSize,
            hiddenSize: hiddenSize,
            inputShape: new[] { 1, predictionHorizon, hiddenSize },
            activation: (IActivationFunction<T>?)null,
            recurrentActivation: null,
            engine: null);

        // Gated Residual Network layers
        for (int i = 0; i < numLayers; i++)
        {
            yield return new DenseLayer<T>(
                inputSize: hiddenSize,
                outputSize: hiddenSize,
                activationFunction: new ELUActivation<T>());
        }

        // Interpretable multi-head attention
        yield return new TransformerEncoderLayer<T>(
            embeddingSize: hiddenSize,
            numHeads: numHeads,
            feedForwardDim: hiddenSize * 4);

        // Final layer normalization
        yield return new LayerNormalizationLayer<T>(hiddenSize);

        // Output projection: maps to prediction horizon
        yield return new DenseLayer<T>(
            inputSize: hiddenSize,
            outputSize: predictionHorizon * numFeatures,
            activationFunction: null);
    }

    /// <summary>
    /// Creates default layers for the Crossformer (Cross-Dimension Transformer) architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="sequenceLength">The input sequence length. Default: 96.</param>
    /// <param name="predictionHorizon">The number of future time steps to predict. Default: 24.</param>
    /// <param name="numFeatures">The number of input features/variables. Default: 7.</param>
    /// <param name="segmentLength">The length of each time segment. Default: 12.</param>
    /// <param name="modelDimension">The model dimension (embedding size). Default: 128.</param>
    /// <param name="numHeads">The number of attention heads. Default: 4.</param>
    /// <param name="numLayers">The number of transformer layers. Default: 3.</param>
    /// <param name="dropout">The dropout rate. Default: 0.1.</param>
    /// <returns>An enumerable of layers comprising the Crossformer architecture.</returns>
    /// <remarks>
    /// <para>
    /// Crossformer uses a two-stage attention mechanism (TSA) that captures both temporal
    /// dependencies and cross-variable relationships through:
    /// </para>
    /// <list type="bullet">
    /// <item><b>Cross-Time Attention:</b> Captures temporal patterns within each variable</item>
    /// <item><b>Cross-Dimension Attention:</b> Captures relationships across variables</item>
    /// <item><b>Hierarchical Structure:</b> Processes at multiple time scales</item>
    /// </list>
    /// <para>
    /// <b>For Beginners:</b> Crossformer is designed for multivariate time series where both
    /// the relationship between time steps AND the relationship between different variables
    /// matter. For example, in stock prediction, Crossformer can learn that price going up
    /// AND volume going down together have a specific meaning.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultCrossformerLayers(
        NeuralNetworkArchitecture<T> architecture,
        int sequenceLength = 96,
        int predictionHorizon = 24,
        int numFeatures = 7,
        int segmentLength = 12,
        int modelDimension = 128,
        int numHeads = 4,
        int numLayers = 3,
        double dropout = 0.1)
    {
        // Validate parameters
        if (sequenceLength < segmentLength)
            throw new ArgumentOutOfRangeException(nameof(sequenceLength), "Sequence length must be at least segment length.");
        if (predictionHorizon < 1)
            throw new ArgumentOutOfRangeException(nameof(predictionHorizon), "Prediction horizon must be at least 1.");
        if (modelDimension % numHeads != 0)
            throw new ArgumentException("Model dimension must be divisible by number of heads.", nameof(modelDimension));

        int numSegments = (sequenceLength + segmentLength - 1) / segmentLength;

        // Dimension-Segment Embedding (DSE)
        // Embeds each segment for each dimension/variable
        yield return new DenseLayer<T>(
            inputSize: segmentLength * numFeatures,
            outputSize: modelDimension,
            activationFunction: new GELUActivation<T>());

        // Two-Stage Attention (TSA) Layers
        for (int i = 0; i < numLayers; i++)
        {
            // Cross-Time Attention
            yield return new TransformerEncoderLayer<T>(
                embeddingSize: modelDimension,
                numHeads: numHeads,
                feedForwardDim: modelDimension * 4);

            // Cross-Dimension Attention
            yield return new TransformerEncoderLayer<T>(
                embeddingSize: modelDimension,
                numHeads: numHeads,
                feedForwardDim: modelDimension * 4);

            // Dropout between stages
            yield return new DropoutLayer<T>(dropout);
        }

        // Final layer normalization
        yield return new LayerNormalizationLayer<T>(modelDimension);

        // Output projection to prediction horizon
        yield return new DenseLayer<T>(
            inputSize: modelDimension,
            outputSize: predictionHorizon * numFeatures,
            activationFunction: null);
    }

    /// <summary>
    /// Creates default layers for the TimesNet (Temporal 2D-Variation Modeling) architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="sequenceLength">The input sequence length. Default: 96.</param>
    /// <param name="predictionHorizon">The number of future time steps to predict. Default: 24.</param>
    /// <param name="numFeatures">The number of input features/variables. Default: 7.</param>
    /// <param name="modelDimension">The model dimension (embedding size). Default: 64.</param>
    /// <param name="feedForwardDim">The feedforward network dimension. Default: 128.</param>
    /// <param name="numLayers">The number of TimesBlock layers. Default: 2.</param>
    /// <param name="topK">Number of dominant periods to discover. Default: 5.</param>
    /// <param name="convKernelSize">The 2D convolution kernel size. Default: 3.</param>
    /// <param name="dropout">The dropout rate. Default: 0.1.</param>
    /// <returns>An enumerable of layers comprising the TimesNet architecture.</returns>
    /// <remarks>
    /// <para>
    /// TimesNet transforms 1D time series into 2D tensors based on discovered periods,
    /// then applies 2D convolutions to capture both intra-period and inter-period variations.
    /// </para>
    /// <list type="bullet">
    /// <item><b>Period Discovery:</b> Uses FFT to find dominant periods</item>
    /// <item><b>2D Transformation:</b> Reshapes 1D series into 2D based on periods</item>
    /// <item><b>Inception Block:</b> Multi-scale 2D convolutions</item>
    /// <item><b>Aggregation:</b> Combines features from different periods</item>
    /// </list>
    /// <para>
    /// <b>For Beginners:</b> TimesNet is like looking at time series data from a bird's eye view.
    /// Instead of just seeing one long sequence, it rearranges the data to show patterns
    /// that repeat at different time scales (daily, weekly, etc.), making it easier to spot trends.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultTimesNetLayers(
        NeuralNetworkArchitecture<T> architecture,
        int sequenceLength = 96,
        int predictionHorizon = 24,
        int numFeatures = 7,
        int modelDimension = 64,
        int feedForwardDim = 128,
        int numLayers = 2,
        int topK = 5,
        int convKernelSize = 3,
        double dropout = 0.1)
    {
        // Validate parameters
        if (sequenceLength < topK)
            throw new ArgumentOutOfRangeException(nameof(sequenceLength), "Sequence length must be at least TopK.");
        if (predictionHorizon < 1)
            throw new ArgumentOutOfRangeException(nameof(predictionHorizon), "Prediction horizon must be at least 1.");

        // Embedding layer: project input features to model dimension
        yield return new DenseLayer<T>(
            inputSize: numFeatures,
            outputSize: modelDimension,
            activationFunction: new GELUActivation<T>());

        // TimesBlock layers (simplified as conv + feedforward)
        for (int i = 0; i < numLayers; i++)
        {
            // Inception-style convolution (using 1D conv to approximate 2D behavior)
            yield return new ConvolutionalLayer<T>(
                inputDepth: modelDimension,
                inputHeight: 1,
                inputWidth: sequenceLength,
                outputDepth: modelDimension,
                kernelSize: convKernelSize,
                stride: 1,
                padding: convKernelSize / 2);

            // Feedforward network
            yield return new DenseLayer<T>(
                inputSize: modelDimension,
                outputSize: feedForwardDim,
                activationFunction: new GELUActivation<T>());

            yield return new DenseLayer<T>(
                inputSize: feedForwardDim,
                outputSize: modelDimension,
                activationFunction: null);

            // Dropout
            yield return new DropoutLayer<T>(dropout);

            // Layer normalization
            yield return new LayerNormalizationLayer<T>(modelDimension);
        }

        // Final layer normalization
        yield return new LayerNormalizationLayer<T>(modelDimension);

        // Output projection: maps to prediction horizon
        yield return new DenseLayer<T>(
            inputSize: modelDimension,
            outputSize: predictionHorizon * numFeatures,
            activationFunction: null);
    }

    /// <summary>
    /// Creates default layers for the ETSformer (Exponential Smoothing Transformer) architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="sequenceLength">The input sequence length.</param>
    /// <param name="predictionHorizon">The number of time steps to predict.</param>
    /// <param name="numFeatures">The number of input features.</param>
    /// <param name="modelDimension">The model embedding dimension.</param>
    /// <param name="feedForwardDim">The feedforward network dimension.</param>
    /// <param name="numEncoderLayers">Number of encoder layers.</param>
    /// <param name="numDecoderLayers">Number of decoder layers.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="dropout">The dropout rate.</param>
    /// <returns>A collection of layers forming the ETSformer architecture.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> ETSformer combines exponential smoothing with transformers.
    /// It's like having a forecasting expert that can both see patterns in data
    /// (like a transformer) and understand how recent vs. old data matters (exponential smoothing).
    /// This makes it very interpretable - you can see the trend, seasonality, and level components.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultETSformerLayers(
        NeuralNetworkArchitecture<T> architecture,
        int sequenceLength = 96,
        int predictionHorizon = 24,
        int numFeatures = 7,
        int modelDimension = 64,
        int feedForwardDim = 128,
        int numEncoderLayers = 2,
        int numDecoderLayers = 1,
        int numHeads = 4,
        double dropout = 0.1)
    {
        // Validate parameters
        if (sequenceLength < 1)
            throw new ArgumentOutOfRangeException(nameof(sequenceLength), "Sequence length must be at least 1.");
        if (predictionHorizon < 1)
            throw new ArgumentOutOfRangeException(nameof(predictionHorizon), "Prediction horizon must be at least 1.");
        if (modelDimension % numHeads != 0)
            throw new ArgumentException("Model dimension must be divisible by number of heads.");

        // Input embedding: project features to model dimension
        yield return new DenseLayer<T>(
            inputSize: numFeatures,
            outputSize: modelDimension,
            activationFunction: new GELUActivation<T>());

        // Encoder layers with exponential smoothing attention
        for (int i = 0; i < numEncoderLayers; i++)
        {
            // Self-attention layer for temporal patterns
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: sequenceLength,
                embeddingDimension: modelDimension,
                headCount: numHeads);

            // Layer normalization
            yield return new LayerNormalizationLayer<T>(modelDimension);

            // Feedforward network
            yield return new DenseLayer<T>(
                inputSize: modelDimension,
                outputSize: feedForwardDim,
                activationFunction: new GELUActivation<T>());

            yield return new DenseLayer<T>(
                inputSize: feedForwardDim,
                outputSize: modelDimension,
                activationFunction: null);

            // Dropout
            yield return new DropoutLayer<T>(dropout);

            // Layer normalization
            yield return new LayerNormalizationLayer<T>(modelDimension);
        }

        // Decoder layers
        for (int i = 0; i < numDecoderLayers; i++)
        {
            // Cross-attention for encoder-decoder connection
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: sequenceLength,
                embeddingDimension: modelDimension,
                headCount: numHeads);

            // Layer normalization
            yield return new LayerNormalizationLayer<T>(modelDimension);

            // Feedforward network
            yield return new DenseLayer<T>(
                inputSize: modelDimension,
                outputSize: feedForwardDim,
                activationFunction: new GELUActivation<T>());

            yield return new DenseLayer<T>(
                inputSize: feedForwardDim,
                outputSize: modelDimension,
                activationFunction: null);

            // Dropout
            yield return new DropoutLayer<T>(dropout);

            // Layer normalization
            yield return new LayerNormalizationLayer<T>(modelDimension);
        }

        // Output projection: maps to prediction horizon
        yield return new DenseLayer<T>(
            inputSize: modelDimension,
            outputSize: predictionHorizon * numFeatures,
            activationFunction: null);
    }

    /// <summary>
    /// Creates default layers for a Non-stationary Transformer model.
    /// </summary>
    /// <param name="architecture">Neural network architecture containing configuration.</param>
    /// <param name="sequenceLength">Input sequence length. Default: 96.</param>
    /// <param name="predictionHorizon">Number of future time steps to predict. Default: 24.</param>
    /// <param name="numFeatures">Number of input features. Default: 7.</param>
    /// <param name="modelDimension">Model embedding dimension. Default: 64.</param>
    /// <param name="feedForwardDim">Feedforward network dimension. Default: 128.</param>
    /// <param name="numEncoderLayers">Number of encoder layers. Default: 2.</param>
    /// <param name="numDecoderLayers">Number of decoder layers. Default: 1.</param>
    /// <param name="numHeads">Number of attention heads. Default: 4.</param>
    /// <param name="projectionDim">Dimension of de-stationarization projections. Default: 64.</param>
    /// <param name="dropout">Dropout rate. Default: 0.1.</param>
    /// <returns>Collection of layers for Non-stationary Transformer architecture.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Non-stationary Transformer tackles the problem of changing
    /// statistical properties (non-stationarity) in time series data by:
    ///
    /// 1. <b>Series Stationarization:</b> Normalizes input data to make it stationary
    /// 2. <b>De-stationary Attention:</b> Applies learned rescaling to attention outputs
    ///
    /// This approach preserves the benefits of attention (capturing long-range dependencies)
    /// while maintaining the original data's statistical characteristics.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultNonStationaryTransformerLayers(
        NeuralNetworkArchitecture<T> architecture,
        int sequenceLength = 96,
        int predictionHorizon = 24,
        int numFeatures = 7,
        int modelDimension = 64,
        int feedForwardDim = 128,
        int numEncoderLayers = 2,
        int numDecoderLayers = 1,
        int numHeads = 4,
        int projectionDim = 64,
        double dropout = 0.1)
    {
        // Validate parameters
        if (sequenceLength < 1)
            throw new ArgumentOutOfRangeException(nameof(sequenceLength), "Sequence length must be at least 1.");
        if (predictionHorizon < 1)
            throw new ArgumentOutOfRangeException(nameof(predictionHorizon), "Prediction horizon must be at least 1.");
        if (modelDimension % numHeads != 0)
            throw new ArgumentException("Model dimension must be divisible by number of heads.");

        // Input embedding: project features to model dimension
        yield return new DenseLayer<T>(
            inputSize: numFeatures,
            outputSize: modelDimension,
            activationFunction: new GELUActivation<T>());

        // De-stationarization projection layer (learns to capture statistics)
        yield return new DenseLayer<T>(
            inputSize: numFeatures,
            outputSize: projectionDim,
            activationFunction: new TanhActivation<T>());

        // Encoder layers with de-stationary attention
        for (int i = 0; i < numEncoderLayers; i++)
        {
            // Self-attention with de-stationary mechanism
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: sequenceLength,
                embeddingDimension: modelDimension,
                headCount: numHeads);

            // Layer normalization
            yield return new LayerNormalizationLayer<T>(modelDimension);

            // Feedforward network
            yield return new DenseLayer<T>(
                inputSize: modelDimension,
                outputSize: feedForwardDim,
                activationFunction: new GELUActivation<T>());

            yield return new DenseLayer<T>(
                inputSize: feedForwardDim,
                outputSize: modelDimension,
                activationFunction: null);

            // Dropout
            yield return new DropoutLayer<T>(dropout);

            // Layer normalization
            yield return new LayerNormalizationLayer<T>(modelDimension);
        }

        // Decoder layers
        for (int i = 0; i < numDecoderLayers; i++)
        {
            // Self-attention in decoder
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: sequenceLength,
                embeddingDimension: modelDimension,
                headCount: numHeads);

            // Layer normalization
            yield return new LayerNormalizationLayer<T>(modelDimension);

            // Cross-attention for encoder-decoder connection
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: sequenceLength,
                embeddingDimension: modelDimension,
                headCount: numHeads);

            // Layer normalization
            yield return new LayerNormalizationLayer<T>(modelDimension);

            // Feedforward network
            yield return new DenseLayer<T>(
                inputSize: modelDimension,
                outputSize: feedForwardDim,
                activationFunction: new GELUActivation<T>());

            yield return new DenseLayer<T>(
                inputSize: feedForwardDim,
                outputSize: modelDimension,
                activationFunction: null);

            // Dropout
            yield return new DropoutLayer<T>(dropout);

            // Layer normalization
            yield return new LayerNormalizationLayer<T>(modelDimension);
        }

        // Output projection: maps to prediction horizon
        yield return new DenseLayer<T>(
            inputSize: modelDimension,
            outputSize: predictionHorizon * numFeatures,
            activationFunction: null);
    }

    /// <summary>
    /// Creates default layers for a TSMixer model.
    /// </summary>
    /// <param name="architecture">Neural network architecture containing configuration.</param>
    /// <param name="sequenceLength">Input sequence length. Default: 96.</param>
    /// <param name="predictionHorizon">Number of future time steps to predict. Default: 24.</param>
    /// <param name="numFeatures">Number of input features. Default: 7.</param>
    /// <param name="hiddenDim">Hidden dimension for MLP layers. Default: 64.</param>
    /// <param name="numBlocks">Number of mixer blocks. Default: 4.</param>
    /// <param name="feedForwardExpansion">Feedforward expansion factor. Default: 2.0.</param>
    /// <param name="dropout">Dropout rate. Default: 0.1.</param>
    /// <returns>Collection of layers for TSMixer architecture.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> TSMixer is an all-MLP (Multi-Layer Perceptron) architecture
    /// that achieves excellent results without attention mechanisms. The key insight is:
    ///
    /// 1. <b>Time-Mixing:</b> MLPs that operate across the time dimension
    /// 2. <b>Feature-Mixing:</b> MLPs that operate across the feature dimension
    /// 3. <b>Alternating:</b> Each block alternates between time and feature mixing
    ///
    /// This simple approach is computationally efficient while achieving competitive accuracy.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultTSMixerLayers(
        NeuralNetworkArchitecture<T> architecture,
        int sequenceLength = 96,
        int predictionHorizon = 24,
        int numFeatures = 7,
        int hiddenDim = 64,
        int numBlocks = 4,
        double feedForwardExpansion = 2.0,
        double dropout = 0.1)
    {
        // Validate parameters
        if (sequenceLength < 1)
            throw new ArgumentOutOfRangeException(nameof(sequenceLength), "Sequence length must be at least 1.");
        if (predictionHorizon < 1)
            throw new ArgumentOutOfRangeException(nameof(predictionHorizon), "Prediction horizon must be at least 1.");
        if (numBlocks < 1)
            throw new ArgumentOutOfRangeException(nameof(numBlocks), "Number of blocks must be at least 1.");

        int expandedDim = (int)(hiddenDim * feedForwardExpansion);

        // Input projection: project features to hidden dimension
        yield return new DenseLayer<T>(
            inputSize: numFeatures,
            outputSize: hiddenDim,
            activationFunction: new ReLUActivation<T>());

        // TSMixer blocks
        for (int block = 0; block < numBlocks; block++)
        {
            // Time-mixing MLP (operates across time dimension)
            // Layer normalization before time mixing
            yield return new LayerNormalizationLayer<T>(hiddenDim);

            // Time-mixing feedforward: expand
            yield return new DenseLayer<T>(
                inputSize: sequenceLength,
                outputSize: expandedDim,
                activationFunction: new ReLUActivation<T>());

            // Time-mixing feedforward: project back
            yield return new DenseLayer<T>(
                inputSize: expandedDim,
                outputSize: sequenceLength,
                activationFunction: null);

            // Dropout
            yield return new DropoutLayer<T>(dropout);

            // Feature-mixing MLP (operates across feature dimension)
            // Layer normalization before feature mixing
            yield return new LayerNormalizationLayer<T>(hiddenDim);

            // Feature-mixing feedforward: expand
            yield return new DenseLayer<T>(
                inputSize: hiddenDim,
                outputSize: expandedDim,
                activationFunction: new ReLUActivation<T>());

            // Feature-mixing feedforward: project back
            yield return new DenseLayer<T>(
                inputSize: expandedDim,
                outputSize: hiddenDim,
                activationFunction: null);

            // Dropout
            yield return new DropoutLayer<T>(dropout);
        }

        // Final layer normalization
        yield return new LayerNormalizationLayer<T>(hiddenDim);

        // Temporal projection: maps sequence length to prediction horizon
        yield return new DenseLayer<T>(
            inputSize: sequenceLength,
            outputSize: predictionHorizon,
            activationFunction: null);

        // Output projection: maps hidden dimension to features
        yield return new DenseLayer<T>(
            inputSize: hiddenDim,
            outputSize: numFeatures,
            activationFunction: null);
    }

    /// <summary>
    /// Creates layers for DeepAR probabilistic autoregressive model.
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations.</typeparam>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="contextLength">Length of historical context window. Default: 96.</param>
    /// <param name="predictionLength">Length of forecast horizon. Default: 24.</param>
    /// <param name="numFeatures">Number of input features. Default: 1.</param>
    /// <param name="hiddenSize">Size of LSTM hidden state. Default: 40.</param>
    /// <param name="numLstmLayers">Number of stacked LSTM layers. Default: 2.</param>
    /// <param name="embeddingDim">Dimension for categorical embeddings. Default: 10.</param>
    /// <param name="dropout">Dropout rate for regularization. Default: 0.1.</param>
    /// <returns>Sequence of layers implementing the DeepAR architecture.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> DeepAR is a probabilistic forecasting model that uses LSTM
    /// (Long Short-Term Memory) networks to learn temporal patterns. Unlike point forecasts,
    /// DeepAR outputs a probability distribution, telling you not just "what" the prediction
    /// is but also "how confident" the model is.
    /// </para>
    /// <para>
    /// The architecture consists of:
    /// - Input projection layer to expand features
    /// - Stacked LSTM layers for capturing temporal dependencies
    /// - Distribution output head for probabilistic predictions
    /// </para>
    /// <para>
    /// <b>Reference:</b> Salinas et al., "DeepAR: Probabilistic Forecasting with Autoregressive
    /// Recurrent Networks", International Journal of Forecasting 2020.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultDeepARLayers(
        NeuralNetworkArchitecture<T> architecture,
        int contextLength = 96,
        int predictionLength = 24,
        int numFeatures = 1,
        int hiddenSize = 40,
        int numLstmLayers = 2,
        int embeddingDim = 10,
        double dropout = 0.1)
    {
        // Validate parameters
        if (contextLength < 1)
            throw new ArgumentOutOfRangeException(nameof(contextLength), "Context length must be at least 1.");
        if (predictionLength < 1)
            throw new ArgumentOutOfRangeException(nameof(predictionLength), "Prediction length must be at least 1.");
        if (numFeatures < 1)
            throw new ArgumentOutOfRangeException(nameof(numFeatures), "Number of features must be at least 1.");
        if (hiddenSize < 1)
            throw new ArgumentOutOfRangeException(nameof(hiddenSize), "Hidden size must be at least 1.");
        if (numLstmLayers < 1)
            throw new ArgumentOutOfRangeException(nameof(numLstmLayers), "Number of LSTM layers must be at least 1.");

        // Input size includes features + optional embedding for categorical covariates
        int inputProjectionSize = numFeatures + embeddingDim;

        // Input projection: project combined input to LSTM input size
        yield return new DenseLayer<T>(
            inputSize: inputProjectionSize,
            outputSize: hiddenSize,
            activationFunction: new ReLUActivation<T>());

        // Stacked LSTM layers for autoregressive modeling
        for (int i = 0; i < numLstmLayers; i++)
        {
            int lstmInputSize = i == 0 ? hiddenSize : hiddenSize;

            // LSTM layer with explicit type disambiguation
            yield return new LSTMLayer<T>(
                inputSize: lstmInputSize,
                hiddenSize: hiddenSize,
                inputShape: new[] { 1, contextLength, lstmInputSize },
                activation: (IActivationFunction<T>?)null,
                recurrentActivation: null,
                engine: null);

            // Dropout between LSTM layers (except after last layer)
            if (i < numLstmLayers - 1 && dropout > 0)
            {
                yield return new DropoutLayer<T>(dropout);
            }
        }

        // Layer normalization for stable training
        yield return new LayerNormalizationLayer<T>(hiddenSize);

        // Distribution parameter layers - outputs mu and sigma for Gaussian distribution
        // Mu (mean) projection
        yield return new DenseLayer<T>(
            inputSize: hiddenSize,
            outputSize: predictionLength,
            activationFunction: null);  // Linear for mean

        // Sigma (std) projection - uses softplus implicitly in forward pass for positivity
        yield return new DenseLayer<T>(
            inputSize: hiddenSize,
            outputSize: predictionLength,
            activationFunction: new SoftPlusActivation<T>());  // Ensures positive std
    }

    /// <summary>
    /// Creates layers for N-BEATS (Neural Basis Expansion Analysis for Time Series) model.
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations.</typeparam>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="lookbackWindow">Historical context window length. Default: 10.</param>
    /// <param name="forecastHorizon">Number of future steps to predict. Default: 5.</param>
    /// <param name="numStacks">Number of stacks in the architecture. Default: 2.</param>
    /// <param name="numBlocksPerStack">Number of blocks per stack. Default: 3.</param>
    /// <param name="hiddenSize">Size of hidden layers in blocks. Default: 256.</param>
    /// <param name="numHiddenLayers">Number of hidden layers per block. Default: 4.</param>
    /// <returns>Sequence of layers implementing the N-BEATS architecture.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> N-BEATS uses a stack-of-blocks architecture where each block
    /// produces both a "backcast" (explanation of the past) and a "forecast" (prediction of the future).
    /// The residuals from each block are passed to the next, allowing the model to hierarchically
    /// decompose the time series.
    /// </para>
    /// <para>
    /// Key concepts:
    /// - Blocks learn to explain patterns and produce forecasts
    /// - Residual connections pass unexplained parts to subsequent blocks
    /// - Stacks group related blocks (e.g., trend stack, seasonality stack)
    /// </para>
    /// <para>
    /// <b>Reference:</b> Oreshkin et al., "N-BEATS: Neural basis expansion analysis for
    /// interpretable time series forecasting", ICLR 2020.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultNBEATSLayers(
        NeuralNetworkArchitecture<T> architecture,
        int lookbackWindow = 10,
        int forecastHorizon = 5,
        int numStacks = 2,
        int numBlocksPerStack = 3,
        int hiddenSize = 256,
        int numHiddenLayers = 4)
    {
        // Validate parameters
        if (lookbackWindow < 1)
            throw new ArgumentOutOfRangeException(nameof(lookbackWindow), "Lookback window must be at least 1.");
        if (forecastHorizon < 1)
            throw new ArgumentOutOfRangeException(nameof(forecastHorizon), "Forecast horizon must be at least 1.");
        if (numStacks < 1)
            throw new ArgumentOutOfRangeException(nameof(numStacks), "Number of stacks must be at least 1.");
        if (numBlocksPerStack < 1)
            throw new ArgumentOutOfRangeException(nameof(numBlocksPerStack), "Number of blocks per stack must be at least 1.");
        if (hiddenSize < 1)
            throw new ArgumentOutOfRangeException(nameof(hiddenSize), "Hidden size must be at least 1.");

        // N-BEATS architecture: multiple stacks, each containing multiple blocks
        // Each block: fully connected layers -> theta output -> basis expansion -> backcast/forecast

        for (int stack = 0; stack < numStacks; stack++)
        {
            for (int block = 0; block < numBlocksPerStack; block++)
            {
                int blockInputSize = lookbackWindow;

                // Input projection for block
                yield return new DenseLayer<T>(
                    inputSize: blockInputSize,
                    outputSize: hiddenSize,
                    activationFunction: new ReLUActivation<T>());

                // Hidden layers within block
                for (int layer = 0; layer < numHiddenLayers - 1; layer++)
                {
                    yield return new DenseLayer<T>(
                        inputSize: hiddenSize,
                        outputSize: hiddenSize,
                        activationFunction: new ReLUActivation<T>());
                }

                // Theta layer for backcast (explains historical data)
                yield return new DenseLayer<T>(
                    inputSize: hiddenSize,
                    outputSize: lookbackWindow,
                    activationFunction: null);

                // Theta layer for forecast (predicts future)
                yield return new DenseLayer<T>(
                    inputSize: hiddenSize,
                    outputSize: forecastHorizon,
                    activationFunction: null);
            }
        }

        // Final output projection to aggregate forecasts
        yield return new DenseLayer<T>(
            inputSize: forecastHorizon,
            outputSize: forecastHorizon,
            activationFunction: null);
    }

    /// <summary>
    /// Creates layers for N-HiTS (Neural Hierarchical Interpolation for Time Series) model.
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations.</typeparam>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="lookbackWindow">Historical context window length. Default: 48.</param>
    /// <param name="forecastHorizon">Number of future steps to predict. Default: 24.</param>
    /// <param name="numStacks">Number of stacks in the architecture. Default: 3.</param>
    /// <param name="numBlocksPerStack">Number of blocks per stack. Default: 1.</param>
    /// <param name="hiddenSize">Size of hidden layers in blocks. Default: 512.</param>
    /// <param name="numHiddenLayers">Number of hidden layers per block. Default: 2.</param>
    /// <param name="dropout">Dropout rate for regularization. Default: 0.1.</param>
    /// <returns>Sequence of layers implementing the N-HiTS architecture.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> N-HiTS improves upon N-BEATS by using multi-rate signal sampling
    /// and hierarchical interpolation. Each stack operates at a different time resolution:
    /// - Stack 1: High-frequency patterns (fast changes)
    /// - Stack 2: Medium-frequency patterns (weekly/monthly cycles)
    /// - Stack 3: Low-frequency patterns (long-term trends)
    /// </para>
    /// <para>
    /// The hierarchical approach makes N-HiTS more efficient for long-horizon forecasting.
    /// </para>
    /// <para>
    /// <b>Reference:</b> Challu et al., "N-HiTS: Neural Hierarchical Interpolation for Time Series
    /// Forecasting", AAAI 2023.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultNHiTSLayers(
        NeuralNetworkArchitecture<T> architecture,
        int lookbackWindow = 48,
        int forecastHorizon = 24,
        int numStacks = 3,
        int numBlocksPerStack = 1,
        int hiddenSize = 512,
        int numHiddenLayers = 2,
        double dropout = 0.1)
    {
        // Validate parameters
        if (lookbackWindow < 1)
            throw new ArgumentOutOfRangeException(nameof(lookbackWindow), "Lookback window must be at least 1.");
        if (forecastHorizon < 1)
            throw new ArgumentOutOfRangeException(nameof(forecastHorizon), "Forecast horizon must be at least 1.");
        if (numStacks < 1)
            throw new ArgumentOutOfRangeException(nameof(numStacks), "Number of stacks must be at least 1.");

        // N-HiTS uses different pooling kernel sizes for each stack
        int[] defaultKernelSizes = new[] { 8, 4, 1 };

        for (int stack = 0; stack < numStacks; stack++)
        {
            int kernelSize = stack < defaultKernelSizes.Length ? defaultKernelSizes[stack] : 1;
            int pooledLookback = lookbackWindow / kernelSize;
            pooledLookback = Math.Max(1, pooledLookback);

            for (int block = 0; block < numBlocksPerStack; block++)
            {
                // Input projection for block
                yield return new DenseLayer<T>(
                    inputSize: pooledLookback,
                    outputSize: hiddenSize,
                    activationFunction: new ReLUActivation<T>());

                // Hidden layers within block
                for (int layer = 0; layer < numHiddenLayers - 1; layer++)
                {
                    yield return new DenseLayer<T>(
                        inputSize: hiddenSize,
                        outputSize: hiddenSize,
                        activationFunction: new ReLUActivation<T>());

                    // Dropout between layers
                    if (dropout > 0)
                    {
                        yield return new DropoutLayer<T>(dropout);
                    }
                }

                // Expression coefficients for interpolation
                int numCoeffs = Math.Max(1, forecastHorizon / kernelSize);

                // Backcast coefficients
                yield return new DenseLayer<T>(
                    inputSize: hiddenSize,
                    outputSize: numCoeffs,
                    activationFunction: null);

                // Forecast coefficients
                yield return new DenseLayer<T>(
                    inputSize: hiddenSize,
                    outputSize: numCoeffs,
                    activationFunction: null);
            }
        }

        // Final output projection
        yield return new DenseLayer<T>(
            inputSize: forecastHorizon,
            outputSize: forecastHorizon,
            activationFunction: null);
    }

    #endregion

    #region LSTNet Layers

    /// <summary>
    /// Creates the default layers for an LSTNet (Long Short-Term Time-series Network) model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="lookbackWindow">How many past time steps to consider (default: 168 for weekly hourly data).</param>
    /// <param name="forecastHorizon">How many future time steps to predict (default: 24).</param>
    /// <param name="numFeatures">Number of input features/variables (default: 7).</param>
    /// <param name="convolutionFilters">Number of convolutional filters (default: 100).</param>
    /// <param name="convolutionKernelSize">Size of convolution kernel (default: 6).</param>
    /// <param name="hiddenRecurrentSize">Hidden size for main recurrent layers (default: 100).</param>
    /// <param name="hiddenSkipSize">Hidden size for skip recurrent layers (default: 5).</param>
    /// <param name="skipPeriod">Skip period for Skip-RNN (default: 24).</param>
    /// <param name="dropout">Dropout rate for regularization (default: 0.2).</param>
    /// <returns>A collection of layers forming an LSTNet architecture.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> LSTNet is a specialized architecture for multivariate time series forecasting
    /// that captures patterns at different temporal scales:
    /// </para>
    /// <para>
    /// <b>Architecture Components:</b>
    /// <list type="number">
    ///     <item>
    ///         <term>Convolutional Layer</term>
    ///         <description>Extracts local features and short-term patterns from the input.
    ///         Think of it as finding small patterns like "weekend spikes" or "morning dips".</description>
    ///     </item>
    ///     <item>
    ///         <term>Recurrent Layer (GRU)</term>
    ///         <description>Captures long-term dependencies and trends. Like remembering
    ///         "sales have been growing for months".</description>
    ///     </item>
    ///     <item>
    ///         <term>Skip-RNN</term>
    ///         <description>Captures periodic patterns by looking at values from the same time
    ///         in previous periods. Like comparing today's 3 PM with yesterday's 3 PM.</description>
    ///     </item>
    ///     <item>
    ///         <term>Output Layer</term>
    ///         <description>Combines all information to produce the final forecast.</description>
    ///     </item>
    /// </list>
    /// </para>
    /// <para>
    /// <b>When to Use LSTNet:</b>
    /// <list type="bullet">
    ///     <item>Multivariate time series with multiple correlated variables</item>
    ///     <item>Data with clear periodic patterns (daily, weekly, etc.)</item>
    ///     <item>Traffic prediction, electricity load forecasting, financial data</item>
    /// </list>
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultLSTNetLayers(
        NeuralNetworkArchitecture<T> architecture,
        int lookbackWindow = 168,
        int forecastHorizon = 24,
        int numFeatures = 7,
        int convolutionFilters = 100,
        int convolutionKernelSize = 6,
        int hiddenRecurrentSize = 100,
        int hiddenSkipSize = 5,
        int skipPeriod = 24,
        double dropout = 0.2)
    {
        // Validate parameters
        if (lookbackWindow < 1)
            throw new ArgumentOutOfRangeException(nameof(lookbackWindow), "Lookback window must be at least 1.");
        if (forecastHorizon < 1)
            throw new ArgumentOutOfRangeException(nameof(forecastHorizon), "Forecast horizon must be at least 1.");
        if (numFeatures < 1)
            throw new ArgumentOutOfRangeException(nameof(numFeatures), "Number of features must be at least 1.");
        if (convolutionFilters < 1)
            throw new ArgumentOutOfRangeException(nameof(convolutionFilters), "Convolution filters must be at least 1.");
        if (convolutionKernelSize < 1)
            throw new ArgumentOutOfRangeException(nameof(convolutionKernelSize), "Convolution kernel size must be at least 1.");
        if (skipPeriod < 1)
            throw new ArgumentOutOfRangeException(nameof(skipPeriod), "Skip period must be at least 1.");
        if (dropout < 0 || dropout >= 1)
            throw new ArgumentOutOfRangeException(nameof(dropout), "Dropout must be between 0 and 1.");

        // === Component 1: Convolutional-style Feature Extraction ===
        // Uses DenseLayer to simulate 1D convolution for local feature extraction
        // In LSTNet, we process each time step's features together
        // Input: [batch, lookbackWindow * numFeatures] (flattened)
        // Output: [batch, convFilters]

        // First layer: Project input to convolution filter space
        // This acts as a learned local feature extractor
        yield return new DenseLayer<T>(
            inputSize: lookbackWindow * numFeatures,
            outputSize: convolutionFilters,
            activationFunction: new ReLUActivation<T>());

        // Dropout for regularization
        if (dropout > 0)
        {
            yield return new DropoutLayer<T>(dropout);
        }

        // Calculate effective sequence length after convolution-style processing
        int convOutputLength = lookbackWindow - convolutionKernelSize + 1;

        // === Component 2: Main Recurrent Layer (GRU) ===
        // Captures long-term temporal dependencies
        // Input: [batch, convFilters]
        // Output: [batch, hiddenRecurrentSize]

        yield return new GRULayer<T>(
            inputSize: convolutionFilters,
            hiddenSize: hiddenRecurrentSize,
            returnSequences: false,
            activation: (IActivationFunction<T>?)null,
            recurrentActivation: null);

        if (dropout > 0)
        {
            yield return new DropoutLayer<T>(dropout);
        }

        // === Component 3: Skip-RNN Layer ===
        // Captures periodic patterns by skipping through time
        // This allows the model to directly compare values at the same time in previous periods
        // For example, with skip=24, it compares 3 PM today with 3 PM yesterday

        // Calculate how many skip connections we can make
        int numSkipConnections = Math.Max(1, convOutputLength / skipPeriod);

        yield return new GRULayer<T>(
            inputSize: convolutionFilters,
            hiddenSize: hiddenSkipSize,
            returnSequences: false,
            activation: (IActivationFunction<T>?)null,
            recurrentActivation: null);

        if (dropout > 0)
        {
            yield return new DropoutLayer<T>(dropout);
        }

        // === Component 4: Highway Layer (Skip Connection) ===
        // Allows linear information to flow directly from input to output
        // This helps capture simple linear trends

        // Dense layer for highway component
        yield return new DenseLayer<T>(
            inputSize: numFeatures,
            outputSize: forecastHorizon,
            activationFunction: null);

        // === Component 5: Combination Layer ===
        // Combines outputs from GRU, Skip-RNN, and Highway
        // Total combined size: hiddenRecurrentSize + (hiddenSkipSize * numSkipConnections)
        int combinedSize = hiddenRecurrentSize + (hiddenSkipSize * numSkipConnections);

        yield return new DenseLayer<T>(
            inputSize: combinedSize,
            outputSize: forecastHorizon * numFeatures,
            activationFunction: null);

        // === Component 6: Output Projection ===
        // Final projection to get forecast for each feature
        yield return new DenseLayer<T>(
            inputSize: forecastHorizon * numFeatures,
            outputSize: forecastHorizon,
            activationFunction: null);
    }

    #endregion

    #region TCN Layers

    /// <summary>
    /// Creates the default layers for a TCN (Temporal Convolutional Network) model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="lookbackWindow">How many past time steps to consider (default: 96).</param>
    /// <param name="forecastHorizon">How many future time steps to predict (default: 24).</param>
    /// <param name="numFeatures">Number of input features (default: 1).</param>
    /// <param name="numChannels">Number of convolutional channels/filters (default: 64).</param>
    /// <param name="kernelSize">Size of convolution kernel (default: 3).</param>
    /// <param name="numLayers">Number of TCN layers (default: 8).</param>
    /// <param name="dropout">Dropout rate for regularization (default: 0.2).</param>
    /// <param name="useResidualConnections">Whether to use residual connections (default: true).</param>
    /// <returns>A collection of layers forming a TCN architecture.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> TCN (Temporal Convolutional Network) uses dilated causal convolutions
    /// to model temporal sequences efficiently:
    /// </para>
    /// <para>
    /// <b>Architecture Components:</b>
    /// <list type="number">
    ///     <item>
    ///         <term>Input Projection</term>
    ///         <description>Projects input features to the channel dimension.</description>
    ///     </item>
    ///     <item>
    ///         <term>Dilated Causal Convolution Blocks</term>
    ///         <description>Each block uses convolutions with exponentially increasing dilation
    ///         (1, 2, 4, 8, ...) to capture patterns at different time scales.</description>
    ///     </item>
    ///     <item>
    ///         <term>Residual Connections</term>
    ///         <description>Connect input to output of each block to improve gradient flow.</description>
    ///     </item>
    ///     <item>
    ///         <term>Output Projection</term>
    ///         <description>Projects the final representation to the forecast horizon.</description>
    ///     </item>
    /// </list>
    /// </para>
    /// <para>
    /// <b>Receptive Field Calculation:</b>
    /// With kernel_size k and n layers, receptive field = 1 + 2*(k-1)*(2^n - 1)
    /// For k=3, n=8: RF = 1 + 2*2*(256-1) = 1021 time steps
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultTCNLayers(
        NeuralNetworkArchitecture<T> architecture,
        int lookbackWindow = 96,
        int forecastHorizon = 24,
        int numFeatures = 1,
        int numChannels = 64,
        int kernelSize = 3,
        int numLayers = 8,
        double dropout = 0.2,
        bool useResidualConnections = true)
    {
        // Validate parameters
        if (lookbackWindow < 1)
            throw new ArgumentOutOfRangeException(nameof(lookbackWindow), "Lookback window must be at least 1.");
        if (forecastHorizon < 1)
            throw new ArgumentOutOfRangeException(nameof(forecastHorizon), "Forecast horizon must be at least 1.");
        if (numFeatures < 1)
            throw new ArgumentOutOfRangeException(nameof(numFeatures), "Number of features must be at least 1.");
        if (numChannels < 1)
            throw new ArgumentOutOfRangeException(nameof(numChannels), "Number of channels must be at least 1.");
        if (kernelSize < 2)
            throw new ArgumentOutOfRangeException(nameof(kernelSize), "Kernel size must be at least 2.");
        if (numLayers < 1)
            throw new ArgumentOutOfRangeException(nameof(numLayers), "Number of layers must be at least 1.");
        if (dropout < 0 || dropout >= 1)
            throw new ArgumentOutOfRangeException(nameof(dropout), "Dropout must be between 0 and 1.");

        // === Input Projection ===
        // Project input features to channel dimension
        yield return new DenseLayer<T>(
            inputSize: lookbackWindow * numFeatures,
            outputSize: numChannels,
            activationFunction: new ReLUActivation<T>());

        // === Dilated Causal Convolution Blocks ===
        // Each block has dilation 2^i where i is the layer index
        for (int layer = 0; layer < numLayers; layer++)
        {
            // For TCN, we simulate dilated convolutions using dense layers
            // In a real implementation, these would be causal dilated conv1d layers
            // The dilation factor doubles with each layer: 1, 2, 4, 8, 16, ...
            int dilation = 1 << layer; // 2^layer

            // First convolution in the block
            yield return new DenseLayer<T>(
                inputSize: numChannels,
                outputSize: numChannels,
                activationFunction: new ReLUActivation<T>());

            if (dropout > 0)
            {
                yield return new DropoutLayer<T>(dropout);
            }

            // Second convolution in the block
            yield return new DenseLayer<T>(
                inputSize: numChannels,
                outputSize: numChannels,
                activationFunction: new ReLUActivation<T>());

            if (dropout > 0)
            {
                yield return new DropoutLayer<T>(dropout);
            }

            // Note: Residual connections are handled in the model's Forward method,
            // not in the layer creation. The model adds the input to the output.
        }

        // === Output Projection ===
        // Project from channels to forecast horizon
        yield return new DenseLayer<T>(
            inputSize: numChannels,
            outputSize: forecastHorizon,
            activationFunction: null);
    }

    #endregion

    #region WaveNet Layers

    /// <summary>
    /// Creates the default layers for a WaveNet model adapted for time series forecasting.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="lookbackWindow">How many past time steps to consider (default: 128).</param>
    /// <param name="forecastHorizon">How many future time steps to predict (default: 24).</param>
    /// <param name="numFeatures">Number of input features (default: 1).</param>
    /// <param name="residualChannels">Number of residual channels (default: 32).</param>
    /// <param name="skipChannels">Number of skip channels (default: 256).</param>
    /// <param name="dilationDepth">Number of dilation doublings per stack (default: 8).</param>
    /// <param name="numStacks">Number of stacks (default: 2).</param>
    /// <param name="dropout">Dropout rate for regularization (default: 0.1).</param>
    /// <returns>A collection of layers forming a WaveNet architecture.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> WaveNet uses dilated causal convolutions with gated activations
    /// and dual residual/skip connections for powerful sequence modeling:
    /// </para>
    /// <para>
    /// <b>Architecture Components:</b>
    /// <list type="number">
    ///     <item>
    ///         <term>Input Projection</term>
    ///         <description>Projects input to residual channel dimension.</description>
    ///     </item>
    ///     <item>
    ///         <term>Dilated Convolution Blocks</term>
    ///         <description>Each block has gated activations and produces both
    ///         residual and skip outputs.</description>
    ///     </item>
    ///     <item>
    ///         <term>Skip Aggregation</term>
    ///         <description>Sums skip connections from all blocks.</description>
    ///     </item>
    ///     <item>
    ///         <term>Output Layers</term>
    ///         <description>Two 1x1 convolutions to produce final forecast.</description>
    ///     </item>
    /// </list>
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultWaveNetLayers(
        NeuralNetworkArchitecture<T> architecture,
        int lookbackWindow = 128,
        int forecastHorizon = 24,
        int numFeatures = 1,
        int residualChannels = 32,
        int skipChannels = 256,
        int dilationDepth = 8,
        int numStacks = 2,
        double dropout = 0.1)
    {
        // Validate parameters
        if (lookbackWindow < 1)
            throw new ArgumentOutOfRangeException(nameof(lookbackWindow), "Lookback window must be at least 1.");
        if (forecastHorizon < 1)
            throw new ArgumentOutOfRangeException(nameof(forecastHorizon), "Forecast horizon must be at least 1.");
        if (numFeatures < 1)
            throw new ArgumentOutOfRangeException(nameof(numFeatures), "Number of features must be at least 1.");
        if (residualChannels < 1)
            throw new ArgumentOutOfRangeException(nameof(residualChannels), "Residual channels must be at least 1.");
        if (skipChannels < 1)
            throw new ArgumentOutOfRangeException(nameof(skipChannels), "Skip channels must be at least 1.");
        if (dilationDepth < 1)
            throw new ArgumentOutOfRangeException(nameof(dilationDepth), "Dilation depth must be at least 1.");
        if (numStacks < 1)
            throw new ArgumentOutOfRangeException(nameof(numStacks), "Number of stacks must be at least 1.");
        if (dropout < 0 || dropout >= 1)
            throw new ArgumentOutOfRangeException(nameof(dropout), "Dropout must be between 0 and 1.");

        // === Input Projection (1x1 convolution equivalent) ===
        // Project input features to residual channels
        yield return new DenseLayer<T>(
            inputSize: lookbackWindow * numFeatures,
            outputSize: residualChannels,
            activationFunction: new TanhActivation<T>());

        // === Dilated Convolution Blocks ===
        // Each stack repeats the dilation pattern
        for (int stack = 0; stack < numStacks; stack++)
        {
            for (int layer = 0; layer < dilationDepth; layer++)
            {
                // Dilation factor: 2^layer
                int dilation = 1 << layer;

                // Filter convolution (for tanh)
                yield return new DenseLayer<T>(
                    inputSize: residualChannels,
                    outputSize: residualChannels,
                    activationFunction: new TanhActivation<T>());

                // Gate convolution (for sigmoid)
                yield return new DenseLayer<T>(
                    inputSize: residualChannels,
                    outputSize: residualChannels,
                    activationFunction: new SigmoidActivation<T>());

                // Residual connection projection (1x1 conv)
                yield return new DenseLayer<T>(
                    inputSize: residualChannels,
                    outputSize: residualChannels,
                    activationFunction: null);

                // Skip connection projection (1x1 conv)
                yield return new DenseLayer<T>(
                    inputSize: residualChannels,
                    outputSize: skipChannels,
                    activationFunction: null);

                if (dropout > 0)
                {
                    yield return new DropoutLayer<T>(dropout);
                }
            }
        }

        // === Output Layers ===
        // First output layer with ReLU
        yield return new DenseLayer<T>(
            inputSize: skipChannels,
            outputSize: skipChannels,
            activationFunction: new ReLUActivation<T>());

        // Second output layer with ReLU
        yield return new DenseLayer<T>(
            inputSize: skipChannels,
            outputSize: forecastHorizon,
            activationFunction: null);
    }

    /// <summary>
    /// Creates the default layer configuration for an MQCNN (Multi-Quantile CNN) model.
    /// </summary>
    /// <param name="architecture">The neural network architecture containing layer configurations.</param>
    /// <param name="lookbackWindow">The number of past time steps to use as input. Default is 168.</param>
    /// <param name="forecastHorizon">The number of future time steps to predict. Default is 24.</param>
    /// <param name="numFeatures">The number of input features per time step. Default is 1.</param>
    /// <param name="numQuantiles">The number of quantiles to predict. Default is 3 (e.g., P10, P50, P90).</param>
    /// <param name="encoderChannels">The number of channels in the encoder network. Default is 64.</param>
    /// <param name="decoderChannels">The number of channels in the decoder network. Default is 32.</param>
    /// <param name="numEncoderLayers">The number of encoder layers. Default is 4.</param>
    /// <param name="numDecoderLayers">The number of decoder layers. Default is 2.</param>
    /// <param name="dropout">The dropout rate for regularization. Default is 0.2.</param>
    /// <returns>An enumerable of layers configured for MQCNN.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> MQCNN predicts multiple quantiles (percentiles) simultaneously,
    /// providing uncertainty estimates with forecasts. Instead of just predicting "tomorrow's
    /// value will be 100", it predicts "there's a 10% chance it will be below 95, 50% chance
    /// below 100, and 90% chance below 105".
    /// </para>
    /// <para>
    /// The architecture has two parts:
    /// - <b>Encoder:</b> Processes the historical sequence using dilated convolutions
    /// - <b>Decoder:</b> Produces quantile predictions from the encoded context
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any parameter is out of valid range.</exception>
    public static IEnumerable<ILayer<T>> CreateDefaultMQCNNLayers(
        NeuralNetworkArchitecture<T> architecture,
        int lookbackWindow = 168,
        int forecastHorizon = 24,
        int numFeatures = 1,
        int numQuantiles = 3,
        int encoderChannels = 64,
        int decoderChannels = 32,
        int numEncoderLayers = 4,
        int numDecoderLayers = 2,
        double dropout = 0.2)
    {
        // Validate parameters
        if (lookbackWindow < 1)
            throw new ArgumentOutOfRangeException(nameof(lookbackWindow), "Lookback window must be at least 1.");
        if (forecastHorizon < 1)
            throw new ArgumentOutOfRangeException(nameof(forecastHorizon), "Forecast horizon must be at least 1.");
        if (numFeatures < 1)
            throw new ArgumentOutOfRangeException(nameof(numFeatures), "Number of features must be at least 1.");
        if (numQuantiles < 1)
            throw new ArgumentOutOfRangeException(nameof(numQuantiles), "Number of quantiles must be at least 1.");
        if (encoderChannels < 1)
            throw new ArgumentOutOfRangeException(nameof(encoderChannels), "Encoder channels must be at least 1.");
        if (decoderChannels < 1)
            throw new ArgumentOutOfRangeException(nameof(decoderChannels), "Decoder channels must be at least 1.");
        if (numEncoderLayers < 1)
            throw new ArgumentOutOfRangeException(nameof(numEncoderLayers), "Number of encoder layers must be at least 1.");
        if (numDecoderLayers < 1)
            throw new ArgumentOutOfRangeException(nameof(numDecoderLayers), "Number of decoder layers must be at least 1.");
        if (dropout < 0 || dropout >= 1)
            throw new ArgumentOutOfRangeException(nameof(dropout), "Dropout must be between 0 and 1.");

        // === Encoder: Input Projection ===
        // Project input sequence to encoder channels
        yield return new DenseLayer<T>(
            inputSize: lookbackWindow * numFeatures,
            outputSize: encoderChannels,
            activationFunction: new ReLUActivation<T>());

        // === Encoder: Dilated Causal Convolution Layers ===
        // Each layer has increasing dilation for larger receptive field
        for (int layer = 0; layer < numEncoderLayers; layer++)
        {
            // Main convolution layer
            yield return new DenseLayer<T>(
                inputSize: encoderChannels,
                outputSize: encoderChannels,
                activationFunction: new ReLUActivation<T>());

            if (dropout > 0)
            {
                yield return new DropoutLayer<T>(dropout);
            }
        }

        // === Context Layer ===
        // Compress encoder output to context representation
        yield return new DenseLayer<T>(
            inputSize: encoderChannels,
            outputSize: decoderChannels,
            activationFunction: new ReLUActivation<T>());

        // === Decoder: Quantile Prediction Layers ===
        // Process context for quantile predictions
        for (int layer = 0; layer < numDecoderLayers; layer++)
        {
            yield return new DenseLayer<T>(
                inputSize: decoderChannels,
                outputSize: decoderChannels,
                activationFunction: new ReLUActivation<T>());

            if (dropout > 0)
            {
                yield return new DropoutLayer<T>(dropout);
            }
        }

        // === Output Layer ===
        // Output: forecastHorizon * numQuantiles
        // Each time step has predictions for all quantiles
        yield return new DenseLayer<T>(
            inputSize: decoderChannels,
            outputSize: forecastHorizon * numQuantiles,
            activationFunction: null);
    }

    /// <summary>
    /// Creates the default layer configuration for a DeepState (Deep State Space) model.
    /// </summary>
    /// <param name="architecture">The neural network architecture containing layer configurations.</param>
    /// <param name="lookbackWindow">The number of past time steps to use as input. Default is 168.</param>
    /// <param name="forecastHorizon">The number of future time steps to predict. Default is 24.</param>
    /// <param name="numFeatures">The number of input features per time step. Default is 1.</param>
    /// <param name="stateDimension">The dimension of the SSM state. Default is 40.</param>
    /// <param name="hiddenDimension">The hidden dimension of the RNN encoder. Default is 50.</param>
    /// <param name="numRnnLayers">The number of RNN layers. Default is 2.</param>
    /// <param name="dropout">The dropout rate for regularization. Default is 0.1.</param>
    /// <returns>An enumerable of layers configured for DeepState.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> DeepState combines deep learning with state space models:
    /// - RNN encoder processes historical data
    /// - Encoder output parameterizes state space model matrices
    /// - State space model produces forecasts with natural uncertainty
    /// </para>
    /// <para>
    /// The architecture includes:
    /// - Input projection layer
    /// - RNN encoder (multiple layers)
    /// - SSM parameter generation layers (for F, H matrices)
    /// - State evolution and observation layers
    /// - Output projection for forecasts
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any parameter is out of valid range.</exception>
    public static IEnumerable<ILayer<T>> CreateDefaultDeepStateLayers(
        NeuralNetworkArchitecture<T> architecture,
        int lookbackWindow = 168,
        int forecastHorizon = 24,
        int numFeatures = 1,
        int stateDimension = 40,
        int hiddenDimension = 50,
        int numRnnLayers = 2,
        double dropout = 0.1)
    {
        // Validate parameters
        if (lookbackWindow < 1)
            throw new ArgumentOutOfRangeException(nameof(lookbackWindow), "Lookback window must be at least 1.");
        if (forecastHorizon < 1)
            throw new ArgumentOutOfRangeException(nameof(forecastHorizon), "Forecast horizon must be at least 1.");
        if (numFeatures < 1)
            throw new ArgumentOutOfRangeException(nameof(numFeatures), "Number of features must be at least 1.");
        if (stateDimension < 1)
            throw new ArgumentOutOfRangeException(nameof(stateDimension), "State dimension must be at least 1.");
        if (hiddenDimension < 1)
            throw new ArgumentOutOfRangeException(nameof(hiddenDimension), "Hidden dimension must be at least 1.");
        if (numRnnLayers < 1)
            throw new ArgumentOutOfRangeException(nameof(numRnnLayers), "Number of RNN layers must be at least 1.");
        if (dropout < 0 || dropout >= 1)
            throw new ArgumentOutOfRangeException(nameof(dropout), "Dropout must be between 0 and 1.");

        // === Input Projection ===
        // Project input features to hidden dimension
        yield return new DenseLayer<T>(
            inputSize: lookbackWindow * numFeatures,
            outputSize: hiddenDimension,
            activationFunction: new ReLUActivation<T>());

        // === RNN Encoder Layers ===
        // Process sequence to extract temporal patterns
        for (int layer = 0; layer < numRnnLayers; layer++)
        {
            // GRU layer (returnSequences=false for last layer)
            bool returnSeqs = layer < numRnnLayers - 1;
            yield return new GRULayer<T>(
                inputSize: hiddenDimension,
                hiddenSize: hiddenDimension,
                returnSequences: returnSeqs,
                activation: (IActivationFunction<T>?)null,
                recurrentActivation: null);

            if (dropout > 0)
            {
                yield return new DropoutLayer<T>(dropout);
            }
        }

        // === SSM Parameter Generation ===
        // Generate state transition matrix parameters (F)
        yield return new DenseLayer<T>(
            inputSize: hiddenDimension,
            outputSize: stateDimension * stateDimension,
            activationFunction: new TanhActivation<T>());

        // Generate observation matrix parameters (H)
        yield return new DenseLayer<T>(
            inputSize: hiddenDimension,
            outputSize: stateDimension,
            activationFunction: null);

        // Generate initial state
        yield return new DenseLayer<T>(
            inputSize: hiddenDimension,
            outputSize: stateDimension,
            activationFunction: new TanhActivation<T>());

        // === State Evolution Layer ===
        // Evolve state for forecast horizon
        yield return new DenseLayer<T>(
            inputSize: stateDimension,
            outputSize: stateDimension * forecastHorizon,
            activationFunction: new TanhActivation<T>());

        // === Output Projection ===
        // Project states to forecast values
        yield return new DenseLayer<T>(
            inputSize: stateDimension * forecastHorizon,
            outputSize: forecastHorizon,
            activationFunction: null);
    }

    /// <summary>
    /// Creates the default layer configuration for a DeepFactor (Deep Factor Model).
    /// </summary>
    /// <param name="architecture">The neural network architecture containing layer configurations.</param>
    /// <param name="lookbackWindow">The number of past time steps to use as input. Default is 168.</param>
    /// <param name="forecastHorizon">The number of future time steps to predict. Default is 24.</param>
    /// <param name="numFeatures">The number of input features per time step. Default is 1.</param>
    /// <param name="numFactors">The number of latent factors. Default is 10.</param>
    /// <param name="factorHiddenDim">The hidden dimension for factor model. Default is 64.</param>
    /// <param name="localHiddenDim">The hidden dimension for local model. Default is 32.</param>
    /// <param name="numFactorLayers">The number of factor model layers. Default is 2.</param>
    /// <param name="numLocalLayers">The number of local model layers. Default is 1.</param>
    /// <param name="dropout">The dropout rate for regularization. Default is 0.1.</param>
    /// <returns>An enumerable of layers configured for DeepFactor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> DeepFactor decomposes time series into:
    /// - Global factors: Patterns shared across all series
    /// - Factor loadings: How much each series responds to each factor
    /// - Local model: Series-specific residual patterns
    /// </para>
    /// <para>
    /// The architecture includes:
    /// - Factor RNN: Learns global factor dynamics
    /// - Loading layer: Maps factors to series-specific contributions
    /// - Local RNN: Captures residual series-specific patterns
    /// - Combination layer: Merges factor and local predictions
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any parameter is out of valid range.</exception>
    public static IEnumerable<ILayer<T>> CreateDefaultDeepFactorLayers(
        NeuralNetworkArchitecture<T> architecture,
        int lookbackWindow = 168,
        int forecastHorizon = 24,
        int numFeatures = 1,
        int numFactors = 10,
        int factorHiddenDim = 64,
        int localHiddenDim = 32,
        int numFactorLayers = 2,
        int numLocalLayers = 1,
        double dropout = 0.1)
    {
        // Validate parameters
        if (lookbackWindow < 1)
            throw new ArgumentOutOfRangeException(nameof(lookbackWindow), "Lookback window must be at least 1.");
        if (forecastHorizon < 1)
            throw new ArgumentOutOfRangeException(nameof(forecastHorizon), "Forecast horizon must be at least 1.");
        if (numFeatures < 1)
            throw new ArgumentOutOfRangeException(nameof(numFeatures), "Number of features must be at least 1.");
        if (numFactors < 1)
            throw new ArgumentOutOfRangeException(nameof(numFactors), "Number of factors must be at least 1.");
        if (factorHiddenDim < 1)
            throw new ArgumentOutOfRangeException(nameof(factorHiddenDim), "Factor hidden dimension must be at least 1.");
        if (localHiddenDim < 1)
            throw new ArgumentOutOfRangeException(nameof(localHiddenDim), "Local hidden dimension must be at least 1.");
        if (numFactorLayers < 1)
            throw new ArgumentOutOfRangeException(nameof(numFactorLayers), "Number of factor layers must be at least 1.");
        if (numLocalLayers < 1)
            throw new ArgumentOutOfRangeException(nameof(numLocalLayers), "Number of local layers must be at least 1.");
        if (dropout < 0 || dropout >= 1)
            throw new ArgumentOutOfRangeException(nameof(dropout), "Dropout must be between 0 and 1.");

        // === Input Projection ===
        yield return new DenseLayer<T>(
            inputSize: lookbackWindow * numFeatures,
            outputSize: factorHiddenDim,
            activationFunction: new ReLUActivation<T>());

        // === Factor Model Layers ===
        // RNN layers to learn global factor dynamics
        for (int layer = 0; layer < numFactorLayers; layer++)
        {
            bool returnSeqs = layer < numFactorLayers - 1;
            yield return new GRULayer<T>(
                inputSize: factorHiddenDim,
                hiddenSize: factorHiddenDim,
                returnSequences: returnSeqs,
                activation: (IActivationFunction<T>?)null,
                recurrentActivation: null);

            if (dropout > 0)
            {
                yield return new DropoutLayer<T>(dropout);
            }
        }

        // Factor generation layer
        yield return new DenseLayer<T>(
            inputSize: factorHiddenDim,
            outputSize: numFactors * forecastHorizon,
            activationFunction: new TanhActivation<T>());

        // === Factor Loading Layer ===
        // Learn how factors contribute to predictions
        yield return new DenseLayer<T>(
            inputSize: numFactors * forecastHorizon,
            outputSize: forecastHorizon,
            activationFunction: null);

        // === Local Model Layers ===
        // Simpler model for series-specific patterns
        yield return new DenseLayer<T>(
            inputSize: lookbackWindow * numFeatures,
            outputSize: localHiddenDim,
            activationFunction: new ReLUActivation<T>());

        for (int layer = 0; layer < numLocalLayers; layer++)
        {
            yield return new DenseLayer<T>(
                inputSize: localHiddenDim,
                outputSize: localHiddenDim,
                activationFunction: new ReLUActivation<T>());

            if (dropout > 0)
            {
                yield return new DropoutLayer<T>(dropout);
            }
        }

        // Local prediction layer
        yield return new DenseLayer<T>(
            inputSize: localHiddenDim,
            outputSize: forecastHorizon,
            activationFunction: null);

        // === Combination Layer ===
        // Merge factor and local predictions
        yield return new DenseLayer<T>(
            inputSize: forecastHorizon * 2,
            outputSize: forecastHorizon,
            activationFunction: null);
    }

    /// <summary>
    /// Creates the default layer configuration for a TimesFM (Time Series Foundation Model).
    /// </summary>
    /// <param name="architecture">The neural network architecture containing layer configurations.</param>
    /// <param name="contextLength">The context length (input sequence). Default is 512.</param>
    /// <param name="forecastHorizon">The number of future time steps to predict. Default is 96.</param>
    /// <param name="numFeatures">The number of input features per time step. Default is 1.</param>
    /// <param name="patchLength">The patch length for input tokenization. Default is 32.</param>
    /// <param name="hiddenDim">The hidden dimension of the transformer. Default is 256.</param>
    /// <param name="numLayers">The number of transformer layers. Default is 8.</param>
    /// <param name="numHeads">The number of attention heads. Default is 4.</param>
    /// <param name="dropout">The dropout rate for regularization. Default is 0.1.</param>
    /// <returns>An enumerable of layers configured for TimesFM.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> TimesFM uses a decoder-only transformer architecture:
    /// - Input is patched (grouped into chunks)
    /// - Patches are projected to hidden dimension
    /// - Transformer processes with causal (autoregressive) attention
    /// - Output head generates forecast values
    /// </para>
    /// <para>
    /// The architecture mimics the pre-trained TimesFM but can be trained from scratch
    /// in native mode or loaded from ONNX for the actual pre-trained weights.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any parameter is out of valid range.</exception>
    public static IEnumerable<ILayer<T>> CreateDefaultTimesFMLayers(
        NeuralNetworkArchitecture<T> architecture,
        int contextLength = 512,
        int forecastHorizon = 96,
        int numFeatures = 1,
        int patchLength = 32,
        int hiddenDim = 256,
        int numLayers = 8,
        int numHeads = 4,
        double dropout = 0.1)
    {
        // Validate parameters
        if (contextLength < 1)
            throw new ArgumentOutOfRangeException(nameof(contextLength), "Context length must be at least 1.");
        if (forecastHorizon < 1)
            throw new ArgumentOutOfRangeException(nameof(forecastHorizon), "Forecast horizon must be at least 1.");
        if (numFeatures < 1)
            throw new ArgumentOutOfRangeException(nameof(numFeatures), "Number of features must be at least 1.");
        if (patchLength < 1)
            throw new ArgumentOutOfRangeException(nameof(patchLength), "Patch length must be at least 1.");
        if (hiddenDim < 1)
            throw new ArgumentOutOfRangeException(nameof(hiddenDim), "Hidden dimension must be at least 1.");
        if (numLayers < 1)
            throw new ArgumentOutOfRangeException(nameof(numLayers), "Number of layers must be at least 1.");
        if (numHeads < 1)
            throw new ArgumentOutOfRangeException(nameof(numHeads), "Number of heads must be at least 1.");
        if (dropout < 0 || dropout >= 1)
            throw new ArgumentOutOfRangeException(nameof(dropout), "Dropout must be between 0 and 1.");

        int numPatches = contextLength / patchLength;
        int patchInputSize = patchLength * numFeatures;

        // === Patch Embedding ===
        // Project each patch to hidden dimension
        yield return new DenseLayer<T>(
            inputSize: contextLength * numFeatures,
            outputSize: numPatches * hiddenDim,
            activationFunction: null);

        // === Transformer Layers ===
        for (int layer = 0; layer < numLayers; layer++)
        {
            // Self-attention (approximated with dense layers)
            // Query projection
            yield return new DenseLayer<T>(
                inputSize: numPatches * hiddenDim,
                outputSize: numPatches * hiddenDim,
                activationFunction: null);

            // Key projection
            yield return new DenseLayer<T>(
                inputSize: numPatches * hiddenDim,
                outputSize: numPatches * hiddenDim,
                activationFunction: null);

            // Value projection
            yield return new DenseLayer<T>(
                inputSize: numPatches * hiddenDim,
                outputSize: numPatches * hiddenDim,
                activationFunction: null);

            // Output projection
            yield return new DenseLayer<T>(
                inputSize: numPatches * hiddenDim,
                outputSize: numPatches * hiddenDim,
                activationFunction: null);

            if (dropout > 0)
            {
                yield return new DropoutLayer<T>(dropout);
            }

            // Feed-forward network
            yield return new DenseLayer<T>(
                inputSize: numPatches * hiddenDim,
                outputSize: numPatches * hiddenDim * 4,
                activationFunction: new ReLUActivation<T>());

            yield return new DenseLayer<T>(
                inputSize: numPatches * hiddenDim * 4,
                outputSize: numPatches * hiddenDim,
                activationFunction: null);

            if (dropout > 0)
            {
                yield return new DropoutLayer<T>(dropout);
            }
        }

        // === Output Head ===
        // Project to forecast horizon
        yield return new DenseLayer<T>(
            inputSize: numPatches * hiddenDim,
            outputSize: forecastHorizon,
            activationFunction: null);
    }

    /// <summary>
    /// Creates default layer configuration for Lag-Llama time series foundation model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="contextLength">The input context length (default: 96).</param>
    /// <param name="forecastHorizon">The number of steps to forecast (default: 24).</param>
    /// <param name="numFeatures">Number of input features (default: 1).</param>
    /// <param name="numLags">Number of lag indices to use (default: 7).</param>
    /// <param name="hiddenDim">Hidden dimension of the transformer (default: 256).</param>
    /// <param name="numLayers">Number of transformer layers (default: 8).</param>
    /// <param name="numHeads">Number of attention heads (default: 4).</param>
    /// <param name="intermediateSize">Intermediate size for FFN (default: 1024).</param>
    /// <param name="dropout">Dropout rate (default: 0.1).</param>
    /// <returns>A collection of layers forming the Lag-Llama architecture.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates the Lag-Llama foundation model architecture.
    ///
    /// <b>Lag-Llama Architecture:</b>
    /// Lag-Llama adapts the Llama LLM architecture for time series:
    /// 1. <b>Lag Feature Extraction</b>: Creates features from past values at specific lags
    /// 2. <b>Input Embedding</b>: Projects lag features to hidden dimension
    /// 3. <b>Transformer Blocks</b>: Llama-style blocks with RMSNorm and SwiGLU
    /// 4. <b>Distribution Head</b>: Outputs parameters for probabilistic predictions
    ///
    /// <b>Key Innovations from Llama:</b>
    /// - RMSNorm instead of LayerNorm (faster, simpler)
    /// - SwiGLU activation in FFN (better performance)
    /// - RoPE for position encoding (better generalization)
    ///
    /// <b>Probabilistic Output:</b>
    /// Unlike point forecasts, Lag-Llama outputs distribution parameters
    /// (e.g., mean and variance for Normal, or df/loc/scale for Student-t).
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any parameter is out of valid range.</exception>
    public static IEnumerable<ILayer<T>> CreateDefaultLagLlamaLayers(
        NeuralNetworkArchitecture<T> architecture,
        int contextLength = 96,
        int forecastHorizon = 24,
        int numFeatures = 1,
        int numLags = 7,
        int hiddenDim = 256,
        int numLayers = 8,
        int numHeads = 4,
        int intermediateSize = 1024,
        double dropout = 0.1)
    {
        // Validate parameters
        if (contextLength < 1)
            throw new ArgumentOutOfRangeException(nameof(contextLength), "Context length must be at least 1.");
        if (forecastHorizon < 1)
            throw new ArgumentOutOfRangeException(nameof(forecastHorizon), "Forecast horizon must be at least 1.");
        if (numFeatures < 1)
            throw new ArgumentOutOfRangeException(nameof(numFeatures), "Number of features must be at least 1.");
        if (numLags < 1)
            throw new ArgumentOutOfRangeException(nameof(numLags), "Number of lags must be at least 1.");
        if (hiddenDim < 1)
            throw new ArgumentOutOfRangeException(nameof(hiddenDim), "Hidden dimension must be at least 1.");
        if (numLayers < 1)
            throw new ArgumentOutOfRangeException(nameof(numLayers), "Number of layers must be at least 1.");
        if (numHeads < 1)
            throw new ArgumentOutOfRangeException(nameof(numHeads), "Number of heads must be at least 1.");
        if (intermediateSize < 1)
            throw new ArgumentOutOfRangeException(nameof(intermediateSize), "Intermediate size must be at least 1.");
        if (dropout < 0 || dropout >= 1)
            throw new ArgumentOutOfRangeException(nameof(dropout), "Dropout must be between 0 and 1.");

        // Input size includes: value + lag features + time features
        int inputSize = numFeatures * (1 + numLags);
        int flattenedInputSize = contextLength * numFeatures;

        // === Input Embedding ===
        // Project lag features to hidden dimension
        yield return new ReshapeLayer<T>(
            inputShape: new[] { contextLength, numFeatures },
            outputShape: new[] { flattenedInputSize });

        yield return new DenseLayer<T>(
            inputSize: contextLength * inputSize,
            outputSize: contextLength * hiddenDim,
            activationFunction: null);

        // === Transformer Layers (Llama-style) ===
        for (int layer = 0; layer < numLayers; layer++)
        {
            // RMSNorm (approximated with batch normalization behavior)
            yield return new BatchNormalizationLayer<T>(contextLength * hiddenDim);

            // Multi-head Self-Attention (causal)
            // Query projection
            yield return new DenseLayer<T>(
                inputSize: contextLength * hiddenDim,
                outputSize: contextLength * hiddenDim,
                activationFunction: null);

            // Key projection
            yield return new DenseLayer<T>(
                inputSize: contextLength * hiddenDim,
                outputSize: contextLength * hiddenDim,
                activationFunction: null);

            // Value projection
            yield return new DenseLayer<T>(
                inputSize: contextLength * hiddenDim,
                outputSize: contextLength * hiddenDim,
                activationFunction: null);

            // Output projection
            yield return new DenseLayer<T>(
                inputSize: contextLength * hiddenDim,
                outputSize: contextLength * hiddenDim,
                activationFunction: null);

            if (dropout > 0)
            {
                yield return new DropoutLayer<T>(dropout);
            }

            // RMSNorm before FFN
            yield return new BatchNormalizationLayer<T>(contextLength * hiddenDim);

            // SwiGLU-style FFN (gate * swish(x))
            // Gate projection
            yield return new DenseLayer<T>(
                inputSize: contextLength * hiddenDim,
                outputSize: contextLength * intermediateSize,
                activationFunction: new SiLUActivation<T>());

            // Up projection
            yield return new DenseLayer<T>(
                inputSize: contextLength * hiddenDim,
                outputSize: contextLength * intermediateSize,
                activationFunction: null);

            // Down projection
            yield return new DenseLayer<T>(
                inputSize: contextLength * intermediateSize,
                outputSize: contextLength * hiddenDim,
                activationFunction: null);

            if (dropout > 0)
            {
                yield return new DropoutLayer<T>(dropout);
            }
        }

        // === Final RMSNorm ===
        yield return new BatchNormalizationLayer<T>(contextLength * hiddenDim);

        // === Distribution Output Head ===
        // For Student-t distribution: output mu, sigma, nu (3 parameters)
        // We output forecast_horizon * 3 values
        yield return new DenseLayer<T>(
            inputSize: contextLength * hiddenDim,
            outputSize: forecastHorizon * 3,
            activationFunction: null);
    }

    /// <summary>
    /// Creates default layer configuration for Chronos time series foundation model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="contextLength">The input context length (default: 512).</param>
    /// <param name="forecastHorizon">The number of steps to forecast (default: 64).</param>
    /// <param name="numTokens">Number of discrete tokens for quantization (default: 4096).</param>
    /// <param name="hiddenDim">Hidden dimension of the transformer (default: 768).</param>
    /// <param name="numLayers">Number of transformer layers (default: 12).</param>
    /// <param name="numHeads">Number of attention heads (default: 12).</param>
    /// <param name="intermediateSize">Intermediate size for FFN (default: 3072).</param>
    /// <param name="dropout">Dropout rate (default: 0.1).</param>
    /// <returns>A collection of layers forming the Chronos architecture.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates the Chronos foundation model architecture.
    ///
    /// <b>Chronos Architecture:</b>
    /// Chronos tokenizes time series and uses a language model:
    /// 1. <b>Token Embedding</b>: Maps token IDs to vectors
    /// 2. <b>Positional Encoding</b>: Adds position information
    /// 3. <b>Transformer Blocks</b>: Standard encoder-decoder or decoder-only
    /// 4. <b>Language Model Head</b>: Predicts next token probabilities
    ///
    /// <b>Tokenization Process:</b>
    /// 1. Scale time series to [-1, 1] range
    /// 2. Quantize into N bins (e.g., 4096)
    /// 3. Each bin is a "token" like words in text
    /// 4. Model predicts next token probabilities
    ///
    /// <b>Probabilistic Forecasting:</b>
    /// - Sample from predicted token distribution
    /// - Convert tokens back to values
    /// - Multiple samples give uncertainty estimates
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any parameter is out of valid range.</exception>
    public static IEnumerable<ILayer<T>> CreateDefaultChronosLayers(
        NeuralNetworkArchitecture<T> architecture,
        int contextLength = 512,
        int forecastHorizon = 64,
        int numTokens = 4096,
        int hiddenDim = 768,
        int numLayers = 12,
        int numHeads = 12,
        int intermediateSize = 3072,
        double dropout = 0.1)
    {
        // Validate parameters
        if (contextLength < 1)
            throw new ArgumentOutOfRangeException(nameof(contextLength), "Context length must be at least 1.");
        if (forecastHorizon < 1)
            throw new ArgumentOutOfRangeException(nameof(forecastHorizon), "Forecast horizon must be at least 1.");
        if (numTokens < 1)
            throw new ArgumentOutOfRangeException(nameof(numTokens), "Number of tokens must be at least 1.");
        if (hiddenDim < 1)
            throw new ArgumentOutOfRangeException(nameof(hiddenDim), "Hidden dimension must be at least 1.");
        if (numLayers < 1)
            throw new ArgumentOutOfRangeException(nameof(numLayers), "Number of layers must be at least 1.");
        if (numHeads < 1)
            throw new ArgumentOutOfRangeException(nameof(numHeads), "Number of heads must be at least 1.");
        if (intermediateSize < 1)
            throw new ArgumentOutOfRangeException(nameof(intermediateSize), "Intermediate size must be at least 1.");
        if (dropout < 0 || dropout >= 1)
            throw new ArgumentOutOfRangeException(nameof(dropout), "Dropout must be between 0 and 1.");

        // === Token Embedding ===
        // Maps token IDs (represented as one-hot) to hidden dimension
        yield return new DenseLayer<T>(
            inputSize: contextLength * numTokens,
            outputSize: contextLength * hiddenDim,
            activationFunction: null);

        // === Transformer Layers (T5/GPT style) ===
        for (int layer = 0; layer < numLayers; layer++)
        {
            // Layer normalization
            yield return new BatchNormalizationLayer<T>(contextLength * hiddenDim);

            // Self-attention
            // Query projection
            yield return new DenseLayer<T>(
                inputSize: contextLength * hiddenDim,
                outputSize: contextLength * hiddenDim,
                activationFunction: null);

            // Key projection
            yield return new DenseLayer<T>(
                inputSize: contextLength * hiddenDim,
                outputSize: contextLength * hiddenDim,
                activationFunction: null);

            // Value projection
            yield return new DenseLayer<T>(
                inputSize: contextLength * hiddenDim,
                outputSize: contextLength * hiddenDim,
                activationFunction: null);

            // Output projection
            yield return new DenseLayer<T>(
                inputSize: contextLength * hiddenDim,
                outputSize: contextLength * hiddenDim,
                activationFunction: null);

            if (dropout > 0)
            {
                yield return new DropoutLayer<T>(dropout);
            }

            // Layer normalization before FFN
            yield return new BatchNormalizationLayer<T>(contextLength * hiddenDim);

            // Feed-forward network (GeLU activation for GPT-style)
            yield return new DenseLayer<T>(
                inputSize: contextLength * hiddenDim,
                outputSize: contextLength * intermediateSize,
                activationFunction: new GELUActivation<T>());

            yield return new DenseLayer<T>(
                inputSize: contextLength * intermediateSize,
                outputSize: contextLength * hiddenDim,
                activationFunction: null);

            if (dropout > 0)
            {
                yield return new DropoutLayer<T>(dropout);
            }
        }

        // === Final Layer Norm ===
        yield return new BatchNormalizationLayer<T>(contextLength * hiddenDim);

        // === Language Model Head ===
        // Project to forecast token logits
        yield return new DenseLayer<T>(
            inputSize: contextLength * hiddenDim,
            outputSize: forecastHorizon * numTokens,
            activationFunction: null);
    }

    /// <summary>
    /// Creates the default layers for MOIRAI (Salesforce's Universal Time Series Foundation Model).
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="contextLength">The input context length.</param>
    /// <param name="forecastHorizon">The prediction horizon length.</param>
    /// <param name="numFeatures">The number of input features.</param>
    /// <param name="patchSizes">Array of patch sizes for multi-scale patching.</param>
    /// <param name="hiddenDim">Hidden dimension size.</param>
    /// <param name="numLayers">Number of transformer layers.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="intermediateSize">FFN intermediate size.</param>
    /// <param name="numMixtures">Number of mixture components for distribution output.</param>
    /// <param name="dropout">Dropout rate.</param>
    /// <returns>An enumerable of layers for MOIRAI.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> MOIRAI uses a unique multi-scale patching approach.
    /// Instead of a single patch size, it creates embeddings at multiple scales
    /// simultaneously, then processes them through a unified transformer encoder.
    /// This allows the model to capture both fine-grained patterns and long-term trends.
    /// The output is a mixture of distributions for probabilistic forecasting.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultMOIRAILayers(
        NeuralNetworkArchitecture<T> architecture,
        int contextLength = 512,
        int forecastHorizon = 96,
        int numFeatures = 1,
        int[]? patchSizes = null,
        int hiddenDim = 768,
        int numLayers = 12,
        int numHeads = 12,
        int intermediateSize = 3072,
        int numMixtures = 10,
        double dropout = 0.1)
    {
        patchSizes ??= new[] { 8, 16, 32, 64 };

        // Calculate total number of patches across all scales
        int totalPatches = 0;
        foreach (var patchSize in patchSizes)
        {
            totalPatches += contextLength / patchSize;
        }
        int flattenedInputSize = contextLength * numFeatures;
        int flattenedPatchSize = totalPatches * hiddenDim;

        // === Multi-Scale Patch Embedding ===
        // For MOIRAI, we embed patches at different scales into the same hidden dimension
        // In practice, this is done with separate embedding layers per scale
        // Here we approximate with a unified embedding
        yield return new ReshapeLayer<T>(
            inputShape: new[] { contextLength, numFeatures },
            outputShape: new[] { flattenedInputSize });

        yield return new DenseLayer<T>(
            inputSize: flattenedInputSize,
            outputSize: flattenedPatchSize,
            activationFunction: null);

        // === Positional Encoding ===
        // Add learnable positional encoding for the multi-scale patches
        yield return new BatchNormalizationLayer<T>(flattenedPatchSize);

        // === Masked Encoder Transformer Blocks ===
        for (int i = 0; i < numLayers; i++)
        {
            // Layer normalization (pre-norm architecture)
            yield return new BatchNormalizationLayer<T>(flattenedPatchSize);

            yield return new ReshapeLayer<T>(
                inputShape: new[] { flattenedPatchSize },
                outputShape: new[] { totalPatches, hiddenDim });

            // Multi-head self-attention across all scales
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: totalPatches,
                embeddingDimension: hiddenDim,
                headCount: numHeads);

            yield return new ReshapeLayer<T>(
                inputShape: new[] { totalPatches, hiddenDim },
                outputShape: new[] { flattenedPatchSize });

            if (dropout > 0)
            {
                yield return new DropoutLayer<T>(dropout);
            }

            // Layer normalization before FFN
            yield return new BatchNormalizationLayer<T>(flattenedPatchSize);

            // Feed-forward network (GELU activation)
            yield return new DenseLayer<T>(
                inputSize: flattenedPatchSize,
                outputSize: totalPatches * intermediateSize,
                activationFunction: new GELUActivation<T>());

            yield return new DenseLayer<T>(
                inputSize: totalPatches * intermediateSize,
                outputSize: flattenedPatchSize,
                activationFunction: null);

            if (dropout > 0)
            {
                yield return new DropoutLayer<T>(dropout);
            }
        }

        // === Final Layer Norm ===
        yield return new BatchNormalizationLayer<T>(flattenedPatchSize);

        // === Prediction Head ===
        // Project to forecast dimension
        yield return new DenseLayer<T>(
            inputSize: flattenedPatchSize,
            outputSize: forecastHorizon * hiddenDim,
            activationFunction: new GELUActivation<T>());

        // === Mixture Distribution Output Head ===
        // Each mixture has: weight, mean, variance (3 params per mixture per forecast step)
        int distributionParams = numMixtures * 3; // weight, mean, variance
        yield return new DenseLayer<T>(
            inputSize: forecastHorizon * hiddenDim,
            outputSize: forecastHorizon * distributionParams,
            activationFunction: null);
    }

    /// <summary>
    /// Creates the default layers for Time-LLM (LLM Reprogramming for Time Series).
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="contextLength">The input context length.</param>
    /// <param name="forecastHorizon">The prediction horizon length.</param>
    /// <param name="numFeatures">The number of input features.</param>
    /// <param name="patchLength">Length of each patch.</param>
    /// <param name="patchStride">Stride between patches.</param>
    /// <param name="llmDim">LLM hidden dimension.</param>
    /// <param name="numPrototypes">Number of text prototypes.</param>
    /// <param name="numLayers">Number of reprogramming layers.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="dropout">Dropout rate.</param>
    /// <returns>An enumerable of layers for Time-LLM.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Time-LLM reprograms frozen LLMs for time series.
    /// The architecture includes:
    /// 1. Patch embedding: Convert time series patches to embeddings
    /// 2. Text prototype layer: Bridge between time series and text domains
    /// 3. Reprogramming transformer: Learn the translation mapping
    /// 4. Simulated LLM layers: Represent the frozen LLM processing
    /// 5. Output projection: Map back to forecast values
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultTimeLLMLayers(
        NeuralNetworkArchitecture<T> architecture,
        int contextLength = 512,
        int forecastHorizon = 96,
        int numFeatures = 1,
        int patchLength = 16,
        int patchStride = 8,
        int llmDim = 768,
        int numPrototypes = 10,
        int numLayers = 2,
        int numHeads = 8,
        double dropout = 0.1)
    {
        // Calculate number of patches
        int numPatches = (contextLength - patchLength) / patchStride + 1;
        int flattenedPatchSize = numPatches * llmDim;
        int flattenedInputSize = contextLength * numFeatures;

        // === Patch Embedding ===
        // Project each patch to LLM dimension
        yield return new ReshapeLayer<T>(
            inputShape: new[] { contextLength, numFeatures },
            outputShape: new[] { flattenedInputSize });

        yield return new DenseLayer<T>(
            inputSize: flattenedInputSize,
            outputSize: flattenedPatchSize,
            activationFunction: null);

        // === Text Prototype Layer ===
        // These learned embeddings help bridge time series to text domain
        // Implemented as a dense layer that expands prototype information
        yield return new BatchNormalizationLayer<T>(flattenedPatchSize);

        // === Reprogramming Transformer Blocks ===
        // These learn to translate time series patterns to LLM-compatible representations
        for (int i = 0; i < numLayers; i++)
        {
            // Layer normalization (pre-norm)
            yield return new BatchNormalizationLayer<T>(flattenedPatchSize);

            yield return new ReshapeLayer<T>(
                inputShape: new[] { flattenedPatchSize },
                outputShape: new[] { numPatches, llmDim });

            // Self-attention for cross-patch interaction
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: numPatches,
                embeddingDimension: llmDim,
                headCount: numHeads);

            yield return new ReshapeLayer<T>(
                inputShape: new[] { numPatches, llmDim },
                outputShape: new[] { flattenedPatchSize });

            if (dropout > 0)
            {
                yield return new DropoutLayer<T>(dropout);
            }

            // FFN with GELU (matches GPT-style)
            yield return new BatchNormalizationLayer<T>(flattenedPatchSize);

            yield return new DenseLayer<T>(
                inputSize: flattenedPatchSize,
                outputSize: flattenedPatchSize * 4,
                activationFunction: new GELUActivation<T>());

            yield return new DenseLayer<T>(
                inputSize: flattenedPatchSize * 4,
                outputSize: flattenedPatchSize,
                activationFunction: null);

            if (dropout > 0)
            {
                yield return new DropoutLayer<T>(dropout);
            }
        }

        // === Simulated Frozen LLM Processing ===
        // In native mode, we simulate the LLM with additional transformer layers
        // In ONNX mode, the actual frozen LLM would be used
        for (int i = 0; i < 2; i++) // Simplified LLM simulation
        {
            yield return new BatchNormalizationLayer<T>(flattenedPatchSize);

            yield return new ReshapeLayer<T>(
                inputShape: new[] { flattenedPatchSize },
                outputShape: new[] { numPatches, llmDim });

            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: numPatches,
                embeddingDimension: llmDim,
                headCount: numHeads);

            yield return new ReshapeLayer<T>(
                inputShape: new[] { numPatches, llmDim },
                outputShape: new[] { flattenedPatchSize });

            yield return new BatchNormalizationLayer<T>(flattenedPatchSize);

            yield return new DenseLayer<T>(
                inputSize: flattenedPatchSize,
                outputSize: flattenedPatchSize * 4,
                activationFunction: new GELUActivation<T>());

            yield return new DenseLayer<T>(
                inputSize: flattenedPatchSize * 4,
                outputSize: flattenedPatchSize,
                activationFunction: null);
        }

        // === Final Layer Norm ===
        yield return new BatchNormalizationLayer<T>(flattenedPatchSize);

        // === Output Projection ===
        // Project from LLM space back to forecast values
        yield return new DenseLayer<T>(
            inputSize: flattenedPatchSize,
            outputSize: forecastHorizon * llmDim / 4,
            activationFunction: new GELUActivation<T>());

        yield return new DenseLayer<T>(
            inputSize: forecastHorizon * llmDim / 4,
            outputSize: forecastHorizon,
            activationFunction: null);
    }

    /// <summary>
    /// Creates default layers for the UniTS (Unified Time Series) model.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="architecture">The neural network architecture containing input/output specifications.</param>
    /// <param name="contextLength">The context length (input sequence length). Default: 512.</param>
    /// <param name="forecastHorizon">The forecast horizon (prediction length). Default: 96.</param>
    /// <param name="hiddenDim">The hidden dimension size. Default: 512.</param>
    /// <param name="numLayers">The number of transformer layers. Default: 6.</param>
    /// <param name="numHeads">The number of attention heads. Default: 8.</param>
    /// <param name="convKernelSizes">The convolution kernel sizes for multi-scale temporal convolution. Default: [3, 5, 7].</param>
    /// <param name="dropout">The dropout rate for regularization. Default: 0.1.</param>
    /// <param name="taskType">The task type (forecasting, classification, anomaly, imputation). Default: forecasting.</param>
    /// <param name="numClasses">The number of classes for classification task. Default: 2.</param>
    /// <returns>An enumerable of layers configured for UniTS.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> UniTS (Unified Time Series) is a universal architecture designed to
    /// handle multiple time series tasks with a single pretrained model. Instead of training
    /// separate models for forecasting, classification, anomaly detection, and imputation,
    /// UniTS learns a shared representation that transfers across all these tasks.
    /// </para>
    /// <para>
    /// The architecture combines:
    /// - Multi-scale temporal convolution (captures patterns at different time scales)
    /// - Transformer layers (captures global dependencies)
    /// - Task-specific output heads (adapts to each task type)
    /// </para>
    /// <para>
    /// <b>Reference:</b> Gao et al., "UniTS: A Unified Multi-Task Time Series Model", 2024.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultUniTSLayers(
        NeuralNetworkArchitecture<T> architecture,
        int contextLength = 512,
        int forecastHorizon = 96,
        int hiddenDim = 512,
        int numLayers = 6,
        int numHeads = 8,
        int[]? convKernelSizes = null,
        double dropout = 0.1,
        string taskType = "forecasting",
        int numClasses = 2)
    {
        convKernelSizes ??= new[] { 3, 5, 7 };

        int numFeatures = architecture.InputSize > 0 ? architecture.InputSize : 1;
        int numScales = convKernelSizes.Length;
        int flattenedInputSize = numFeatures * contextLength;
        int flattenedSize = hiddenDim * contextLength;

        // === Input Embedding ===
        // Project input features to hidden dimension
        yield return new ReshapeLayer<T>(
            inputShape: new[] { contextLength, numFeatures },
            outputShape: new[] { flattenedInputSize });

        yield return new DenseLayer<T>(
            inputSize: flattenedInputSize,
            outputSize: flattenedSize,
            activationFunction: new GELUActivation<T>());

        // === Multi-Scale Temporal Processing ===
        // Simulates multi-scale temporal convolution using dense layers with different receptive fields
        // Each scale captures patterns at different time granularities
        for (int scale = 0; scale < numScales; scale++)
        {
            // Dense layer simulating temporal convolution at this scale
            // Different scales use different hidden dimensions to capture various patterns
            int scaleHidden = hiddenDim / numScales;
            yield return new DenseLayer<T>(
                inputSize: hiddenDim * contextLength,
                outputSize: scaleHidden * contextLength,
                activationFunction: new GELUActivation<T>());

            yield return new BatchNormalizationLayer<T>(scaleHidden * contextLength);
        }

        // === Scale Aggregation ===
        // Combine multi-scale features
        yield return new DenseLayer<T>(
            inputSize: flattenedSize,
            outputSize: flattenedSize,
            activationFunction: new GELUActivation<T>());

        // === Transformer Encoder Stack ===
        // Standard transformer layers for capturing global dependencies
        for (int i = 0; i < numLayers; i++)
        {
            // Pre-normalization for stability
            yield return new BatchNormalizationLayer<T>(flattenedSize);

            yield return new ReshapeLayer<T>(
                inputShape: new[] { flattenedSize },
                outputShape: new[] { contextLength, hiddenDim });

            // Multi-head self-attention
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: contextLength,
                embeddingDimension: hiddenDim,
                headCount: numHeads);

            yield return new ReshapeLayer<T>(
                inputShape: new[] { contextLength, hiddenDim },
                outputShape: new[] { flattenedSize });

            // Dropout for regularization
            if (dropout > 0)
            {
                yield return new DropoutLayer<T>(dropout);
            }

            // Feed-forward network with GELU activation
            yield return new BatchNormalizationLayer<T>(flattenedSize);

            yield return new DenseLayer<T>(
                inputSize: flattenedSize,
                outputSize: flattenedSize * 4,
                activationFunction: new GELUActivation<T>());

            yield return new DenseLayer<T>(
                inputSize: flattenedSize * 4,
                outputSize: flattenedSize,
                activationFunction: null);

            if (dropout > 0)
            {
                yield return new DropoutLayer<T>(dropout);
            }
        }

        // === Final Layer Norm ===
        yield return new BatchNormalizationLayer<T>(flattenedSize);

        // === Task-Specific Output Heads ===
        // Different projection heads for different tasks
        switch (taskType.ToLowerInvariant())
        {
            case "classification":
                // Global pooling followed by classification head
                yield return new DenseLayer<T>(
                    inputSize: hiddenDim * contextLength,
                    outputSize: hiddenDim,
                    activationFunction: new GELUActivation<T>());

                yield return new DenseLayer<T>(
                    inputSize: hiddenDim,
                    outputSize: numClasses,
                    activationFunction: null);
                break;

            case "anomaly":
                // Reconstruction head for anomaly detection
                yield return new DenseLayer<T>(
                    inputSize: hiddenDim * contextLength,
                    outputSize: hiddenDim * contextLength / 2,
                    activationFunction: new GELUActivation<T>());

                yield return new DenseLayer<T>(
                    inputSize: hiddenDim * contextLength / 2,
                    outputSize: numFeatures * contextLength,
                    activationFunction: null);
                break;

            case "imputation":
                // Same as anomaly - reconstruct the full sequence
                yield return new DenseLayer<T>(
                    inputSize: hiddenDim * contextLength,
                    outputSize: hiddenDim * contextLength / 2,
                    activationFunction: new GELUActivation<T>());

                yield return new DenseLayer<T>(
                    inputSize: hiddenDim * contextLength / 2,
                    outputSize: numFeatures * contextLength,
                    activationFunction: null);
                break;

            case "forecasting":
            default:
                // Forecasting projection head
                yield return new DenseLayer<T>(
                    inputSize: flattenedSize,
                    outputSize: hiddenDim * forecastHorizon / 4,
                    activationFunction: new GELUActivation<T>());

                yield return new DenseLayer<T>(
                    inputSize: hiddenDim * forecastHorizon / 4,
                    outputSize: forecastHorizon,
                    activationFunction: null);
                break;
        }
    }

    /// <summary>
    /// Creates default layers for the Timer (Generative Pre-Training) model.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="architecture">The neural network architecture containing input/output specifications.</param>
    /// <param name="contextLength">The context length (input sequence length). Default: 512.</param>
    /// <param name="forecastHorizon">The forecast horizon (prediction length). Default: 96.</param>
    /// <param name="numFeatures">The number of input features. Default: 1.</param>
    /// <param name="patchLength">The patch length for tokenization. Default: 16.</param>
    /// <param name="patchStride">The patch stride. Default: 8.</param>
    /// <param name="hiddenDim">The hidden dimension size. Default: 768.</param>
    /// <param name="numLayers">The number of transformer layers. Default: 12.</param>
    /// <param name="numHeads">The number of attention heads. Default: 12.</param>
    /// <param name="dropout">The dropout rate for regularization. Default: 0.1.</param>
    /// <returns>An enumerable of layers configured for Timer.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Timer is a generative pre-training approach for time series.
    /// Like GPT for language, Timer learns to predict future values from past values,
    /// building rich temporal representations that transfer to various downstream tasks.
    /// </para>
    /// <para>
    /// The architecture follows a GPT-style decoder-only transformer:
    /// - Patch embedding converts time series to tokens
    /// - Positional encoding adds temporal position information
    /// - Causal transformer layers with masked self-attention
    /// - Autoregressive generation head for forecasting
    /// </para>
    /// <para>
    /// <b>Reference:</b> Liu et al., "Timer: Generative Pre-Training of Time Series", 2024.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultTimerLayers(
        NeuralNetworkArchitecture<T> architecture,
        int contextLength = 512,
        int forecastHorizon = 96,
        int numFeatures = 1,
        int patchLength = 16,
        int patchStride = 8,
        int hiddenDim = 768,
        int numLayers = 12,
        int numHeads = 12,
        double dropout = 0.1)
    {
        // Calculate number of patches
        int numPatches = (contextLength - patchLength) / patchStride + 1;
        int flattenedPatchSize = hiddenDim * numPatches;

        // === Patch Embedding ===
        // Convert raw time series patches to hidden dimension tokens
        yield return new DenseLayer<T>(
            inputSize: numFeatures * patchLength,
            outputSize: hiddenDim,
            activationFunction: new GELUActivation<T>());

        yield return new ReshapeLayer<T>(
            inputShape: new[] { contextLength, hiddenDim },
            outputShape: new[] { flattenedPatchSize });

        // === Positional Encoding ===
        // Add learned position information (simulated with batch norm + projection)
        yield return new BatchNormalizationLayer<T>(flattenedPatchSize);

        yield return new DenseLayer<T>(
            inputSize: flattenedPatchSize,
            outputSize: flattenedPatchSize,
            activationFunction: null);

        // === GPT-Style Decoder Transformer Stack ===
        // Causal self-attention with masked attention
        for (int i = 0; i < numLayers; i++)
        {
            // Pre-norm for stability
            yield return new BatchNormalizationLayer<T>(flattenedPatchSize);

            yield return new ReshapeLayer<T>(
                inputShape: new[] { flattenedPatchSize },
                outputShape: new[] { numPatches, hiddenDim });

            // Multi-head self-attention (causal masking applied during forward)
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: numPatches,
                embeddingDimension: hiddenDim,
                headCount: numHeads);

            yield return new ReshapeLayer<T>(
                inputShape: new[] { numPatches, hiddenDim },
                outputShape: new[] { flattenedPatchSize });

            // Dropout for regularization
            if (dropout > 0)
            {
                yield return new DropoutLayer<T>(dropout);
            }

            // Pre-norm for FFN
            yield return new BatchNormalizationLayer<T>(flattenedPatchSize);

            // Feed-forward network with GELU activation
            yield return new DenseLayer<T>(
                inputSize: flattenedPatchSize,
                outputSize: flattenedPatchSize * 4,
                activationFunction: new GELUActivation<T>());

            yield return new DenseLayer<T>(
                inputSize: flattenedPatchSize * 4,
                outputSize: flattenedPatchSize,
                activationFunction: null);

            if (dropout > 0)
            {
                yield return new DropoutLayer<T>(dropout);
            }
        }

        // === Final Layer Norm ===
        yield return new BatchNormalizationLayer<T>(flattenedPatchSize);

        // === Autoregressive Generation Head ===
        // Project to forecast space for next-token prediction
        yield return new DenseLayer<T>(
            inputSize: flattenedPatchSize,
            outputSize: hiddenDim * forecastHorizon / 4,
            activationFunction: new GELUActivation<T>());

        yield return new DenseLayer<T>(
            inputSize: hiddenDim * forecastHorizon / 4,
            outputSize: forecastHorizon,
            activationFunction: null);
    }

    /// <summary>
    /// Creates default layers for the TimeGPT-style model.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="architecture">The neural network architecture containing input/output specifications.</param>
    /// <param name="contextLength">The context length (input sequence length). Default: 512.</param>
    /// <param name="forecastHorizon">The forecast horizon (prediction length). Default: 96.</param>
    /// <param name="numFeatures">The number of input features. Default: 1.</param>
    /// <param name="hiddenDim">The hidden dimension size. Default: 1024.</param>
    /// <param name="numLayers">The number of transformer layers. Default: 24.</param>
    /// <param name="numHeads">The number of attention heads. Default: 16.</param>
    /// <param name="dropout">The dropout rate for regularization. Default: 0.0.</param>
    /// <returns>An enumerable of layers configured for TimeGPT.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> TimeGPT is a large-scale foundation model for time series forecasting.
    /// Like GPT for language, it was trained on millions of diverse time series to become a
    /// general-purpose forecaster that works zero-shot on new data.
    /// </para>
    /// <para>
    /// The architecture follows a standard transformer design:
    /// - Positional encoding for temporal information
    /// - Large transformer backbone with many layers
    /// - Projection head for generating forecasts
    /// - Optional conformal prediction for uncertainty
    /// </para>
    /// <para>
    /// <b>Reference:</b> Garza et al., "TimeGPT-1", 2023.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultTimeGPTLayers(
        NeuralNetworkArchitecture<T> architecture,
        int contextLength = 512,
        int forecastHorizon = 96,
        int numFeatures = 1,
        int hiddenDim = 1024,
        int numLayers = 24,
        int numHeads = 16,
        double dropout = 0.0)
    {
        int flattenedInputSize = contextLength * numFeatures;
        int flattenedSize = hiddenDim * contextLength;

        // === Input Embedding ===
        // Project input time series to hidden dimension
        yield return new ReshapeLayer<T>(
            inputShape: new[] { contextLength, numFeatures },
            outputShape: new[] { flattenedInputSize });

        yield return new DenseLayer<T>(
            inputSize: flattenedInputSize,
            outputSize: flattenedSize,
            activationFunction: new GELUActivation<T>());

        // === Positional Encoding ===
        // Simulated with batch normalization and projection
        yield return new BatchNormalizationLayer<T>(flattenedSize);

        yield return new DenseLayer<T>(
            inputSize: flattenedSize,
            outputSize: flattenedSize,
            activationFunction: null);

        // === Large Transformer Backbone ===
        // GPT-style transformer with many layers for foundation model capabilities
        for (int i = 0; i < numLayers; i++)
        {
            // Pre-norm for stability
            yield return new BatchNormalizationLayer<T>(flattenedSize);

            yield return new ReshapeLayer<T>(
                inputShape: new[] { flattenedSize },
                outputShape: new[] { contextLength, hiddenDim });

            // Multi-head self-attention
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: contextLength,
                embeddingDimension: hiddenDim,
                headCount: numHeads);

            yield return new ReshapeLayer<T>(
                inputShape: new[] { contextLength, hiddenDim },
                outputShape: new[] { flattenedSize });

            // Dropout (disabled for zero-shot inference)
            if (dropout > 0)
            {
                yield return new DropoutLayer<T>(dropout);
            }

            // Pre-norm for FFN
            yield return new BatchNormalizationLayer<T>(flattenedSize);

            // Large feed-forward network (4x hidden dim is standard)
            yield return new DenseLayer<T>(
                inputSize: flattenedSize,
                outputSize: flattenedSize * 4,
                activationFunction: new GELUActivation<T>());

            yield return new DenseLayer<T>(
                inputSize: flattenedSize * 4,
                outputSize: flattenedSize,
                activationFunction: null);

            if (dropout > 0)
            {
                yield return new DropoutLayer<T>(dropout);
            }
        }

        // === Final Layer Norm ===
        yield return new BatchNormalizationLayer<T>(flattenedSize);

        // === Forecast Projection Head ===
        // Project to forecast horizon (with intermediate layer for capacity)
        yield return new DenseLayer<T>(
            inputSize: flattenedSize,
            outputSize: hiddenDim * forecastHorizon / 4,
            activationFunction: new GELUActivation<T>());

        yield return new DenseLayer<T>(
            inputSize: hiddenDim * forecastHorizon / 4,
            outputSize: forecastHorizon,
            activationFunction: null);
    }

    /// <summary>
    /// Creates default layers for the Mamba (Selective State Space Model).
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="architecture">The neural network architecture containing input/output specifications.</param>
    /// <param name="contextLength">The context length (input sequence length). Default: 512.</param>
    /// <param name="forecastHorizon">The forecast horizon (prediction length). Default: 96.</param>
    /// <param name="numFeatures">The number of input features. Default: 1.</param>
    /// <param name="modelDim">The model dimension. Default: 256.</param>
    /// <param name="stateDim">The state dimension for SSM. Default: 16.</param>
    /// <param name="expandFactor">The expansion factor. Default: 2.</param>
    /// <param name="numLayers">The number of Mamba layers. Default: 4.</param>
    /// <param name="dropout">The dropout rate for regularization. Default: 0.1.</param>
    /// <returns>An enumerable of layers configured for Mamba.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Mamba is a selective state space model that achieves
    /// linear-time complexity for sequence modeling. Unlike transformers with O(n^2)
    /// attention, Mamba processes sequences in O(n) time while maintaining expressiveness
    /// through input-dependent state space parameters.
    /// </para>
    /// <para>
    /// The architecture simulates the Mamba block using available layers:
    /// - Input projection to expanded dimension
    /// - Dense layers simulating the SSM selective mechanism
    /// - Output projection back to model dimension
    /// </para>
    /// <para>
    /// <b>Reference:</b> Gu and Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2024.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultMambaLayers(
        NeuralNetworkArchitecture<T> architecture,
        int contextLength = 512,
        int forecastHorizon = 96,
        int numFeatures = 1,
        int modelDim = 256,
        int stateDim = 16,
        int expandFactor = 2,
        int numLayers = 4,
        double dropout = 0.1)
    {
        int innerDim = modelDim * expandFactor;

        // === Input Embedding ===
        yield return new DenseLayer<T>(
            inputSize: numFeatures * contextLength,
            outputSize: modelDim * contextLength,
            activationFunction: new GELUActivation<T>());

        // === Mamba Blocks ===
        for (int layer = 0; layer < numLayers; layer++)
        {
            // Pre-norm
            yield return new BatchNormalizationLayer<T>(modelDim * contextLength);

            // === Mamba Block Start ===
            // Input projection (simulates x_proj and z_proj)
            yield return new DenseLayer<T>(
                inputSize: modelDim * contextLength,
                outputSize: innerDim * contextLength,
                activationFunction: new SiLUActivation<T>());

            // Selective SSM simulation using dense layers
            // In real Mamba, this is a state space model with input-dependent parameters
            // We simulate it with a series of dense layers that capture the selective behavior

            // B projection (input to state)
            yield return new DenseLayer<T>(
                inputSize: innerDim * contextLength,
                outputSize: innerDim * stateDim,
                activationFunction: null);

            // State processing (simulates the SSM state dynamics)
            yield return new DenseLayer<T>(
                inputSize: innerDim * stateDim,
                outputSize: innerDim * stateDim,
                activationFunction: new SiLUActivation<T>());

            // C projection (state to output)
            yield return new DenseLayer<T>(
                inputSize: innerDim * stateDim,
                outputSize: innerDim * contextLength,
                activationFunction: null);

            // Gating mechanism (z branch)
            yield return new DenseLayer<T>(
                inputSize: innerDim * contextLength,
                outputSize: innerDim * contextLength,
                activationFunction: new SiLUActivation<T>());

            // Output projection
            yield return new DenseLayer<T>(
                inputSize: innerDim * contextLength,
                outputSize: modelDim * contextLength,
                activationFunction: null);

            // Dropout
            if (dropout > 0)
            {
                yield return new DropoutLayer<T>(dropout);
            }
            // === Mamba Block End ===
        }

        // === Final Layer Norm ===
        yield return new BatchNormalizationLayer<T>(modelDim * contextLength);

        // === Output Projection ===
        yield return new DenseLayer<T>(
            inputSize: modelDim * contextLength,
            outputSize: modelDim * forecastHorizon / 4,
            activationFunction: new GELUActivation<T>());

        yield return new DenseLayer<T>(
            inputSize: modelDim * forecastHorizon / 4,
            outputSize: forecastHorizon,
            activationFunction: null);
    }

    /// <summary>
    /// Creates default layers for the S4 (Structured State Space Sequence) model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="contextLength">The input sequence length (default: 1024).</param>
    /// <param name="forecastHorizon">The prediction length (default: 96).</param>
    /// <param name="modelDim">The model dimension (default: 256).</param>
    /// <param name="stateDim">The state dimension N (default: 64).</param>
    /// <param name="numLayers">Number of S4 layers (default: 6).</param>
    /// <param name="useLowRankCorrection">Whether to use DPLR (default: true).</param>
    /// <param name="lowRankRank">Rank of low-rank correction (default: 1).</param>
    /// <returns>An enumerable collection of layers for the S4 model.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> S4 (Structured State Space) is a foundational model
    /// that achieves near-linear complexity through clever mathematical structure:
    ///
    /// <b>The S4 Computation Flow:</b>
    /// 1. Input embedding to model dimension
    /// 2. For each S4 layer:
    ///    - State Space Model (SSM) using HiPPO-structured A matrix
    ///    - Compute as convolution using FFT (O(n log n))
    ///    - Residual connection and normalization
    /// 3. Output projection to forecast
    ///
    /// <b>Key Innovation - DPLR Decomposition:</b>
    /// The HiPPO matrix A is decomposed as A = diagonal + low-rank.
    /// This allows efficient computation while preserving the structure that
    /// makes HiPPO effective at compressing long sequences.
    ///
    /// <b>What Each Layer Does:</b>
    /// - Diagonal projection: The diagonal part of A (eigenvalues)
    /// - Low-rank projections: P and Q for the off-diagonal correction
    /// - State update: Simulates the state dynamics x' = Ax + Bu
    /// - Output: Computes y = Cx + Du from the state
    ///
    /// Since we don't have explicit FFT layers, we simulate the SSM
    /// computation using dense layers that learn the effective convolution kernel.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultS4Layers(
        NeuralNetworkArchitecture<T> architecture,
        int contextLength = 1024,
        int forecastHorizon = 96,
        int modelDim = 256,
        int stateDim = 64,
        int numLayers = 6,
        bool useLowRankCorrection = true,
        int lowRankRank = 1,
        int numFeatures = 1)
    {

        // === Input Embedding ===
        yield return new DenseLayer<T>(
            inputSize: contextLength * numFeatures,
            outputSize: modelDim * contextLength,
            activationFunction: new GELUActivation<T>());

        yield return new LayerNormalizationLayer<T>(modelDim * contextLength);

        // === S4 Layers ===
        for (int layer = 0; layer < numLayers; layer++)
        {
            // === SSM Block (simulated with dense layers) ===

            // B projection (input to state)
            // In S4, B projects input u into the state space
            yield return new DenseLayer<T>(
                inputSize: modelDim * contextLength,
                outputSize: stateDim * contextLength,
                activationFunction: null);

            // Diagonal component of A (discretized)
            // This simulates A_bar_diagonal * x where A_bar = discrete(A)
            yield return new DenseLayer<T>(
                inputSize: stateDim * contextLength,
                outputSize: stateDim * contextLength,
                activationFunction: new TanhActivation<T>()); // Tanh for stability (SSM eigenvalues)

            if (useLowRankCorrection)
            {
                // Low-rank correction: P projection
                yield return new DenseLayer<T>(
                    inputSize: stateDim * contextLength,
                    outputSize: lowRankRank * contextLength,
                    activationFunction: null);

                // Low-rank correction: Q^T projection (reconstructs contribution to state)
                yield return new DenseLayer<T>(
                    inputSize: lowRankRank * contextLength,
                    outputSize: stateDim * contextLength,
                    activationFunction: null);
            }

            // C projection (state to output)
            // In S4, C projects the state x back to the output
            yield return new DenseLayer<T>(
                inputSize: stateDim * contextLength,
                outputSize: modelDim * contextLength,
                activationFunction: null);

            // D (direct feedthrough)
            // Skip connection from input to output (simulated via residual)
            yield return new DenseLayer<T>(
                inputSize: modelDim * contextLength,
                outputSize: modelDim * contextLength,
                activationFunction: new GELUActivation<T>());

            // Layer normalization
            yield return new LayerNormalizationLayer<T>(modelDim * contextLength);

            // Dropout for regularization
            yield return new DropoutLayer<T>(0.1);
        }

        // === FFN Block (post-SSM processing) ===
        yield return new DenseLayer<T>(
            inputSize: modelDim * contextLength,
            outputSize: modelDim * contextLength * 2,
            activationFunction: new GELUActivation<T>());

        yield return new DenseLayer<T>(
            inputSize: modelDim * contextLength * 2,
            outputSize: modelDim * contextLength,
            activationFunction: null);

        yield return new LayerNormalizationLayer<T>(modelDim * contextLength);

        // === Output Projection ===
        yield return new DenseLayer<T>(
            inputSize: modelDim * contextLength,
            outputSize: modelDim * forecastHorizon / 4,
            activationFunction: new GELUActivation<T>());

        yield return new DenseLayer<T>(
            inputSize: modelDim * forecastHorizon / 4,
            outputSize: forecastHorizon,
            activationFunction: null);
    }

    /// <summary>
    /// Creates default layers for the TimeMachine (Time Series State Space Model) architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="contextLength">The input sequence length (default: 512).</param>
    /// <param name="forecastHorizon">The prediction horizon (default: 96).</param>
    /// <param name="modelDim">The model dimension d_model (default: 256).</param>
    /// <param name="stateDim">The SSM state dimension (default: 16).</param>
    /// <param name="numScales">Number of temporal scales/Mamba blocks (default: 4).</param>
    /// <param name="numLayers">Number of SSM layers per scale (default: 2).</param>
    /// <param name="expandFactor">Expansion factor for inner dimension (default: 2).</param>
    /// <param name="convKernelSize">Convolution kernel size (default: 4).</param>
    /// <param name="useMultiScaleAttention">Whether to use attention for scale combination (default: true).</param>
    /// <param name="numFeatures">Number of input features (default: 1).</param>
    /// <returns>A collection of layers forming the TimeMachine architecture.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> TimeMachine is a state space model designed specifically for
    /// time series forecasting. The key insight from "A Time Series is Worth 4 Mambas"
    /// is that using multiple SSM blocks at different temporal scales captures both
    /// short-term and long-term patterns effectively.
    /// </para>
    /// <para>
    /// The architecture follows this flow:
    /// 1. Input embedding with reversible normalization
    /// 2. Temporal decomposition into multiple scales
    /// 3. Each scale has its own SSM (Mamba-style) blocks
    /// 4. Multi-scale attention combines the scale outputs
    /// 5. Output projection produces forecasts
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultTimeMachineLayers(
        NeuralNetworkArchitecture<T> architecture,
        int contextLength = 512,
        int forecastHorizon = 96,
        int modelDim = 256,
        int stateDim = 16,
        int numScales = 4,
        int numLayers = 2,
        int expandFactor = 2,
        int convKernelSize = 4,
        bool useMultiScaleAttention = true,
        int numFeatures = 1)
    {
        int innerDim = modelDim * expandFactor;

        // === Input Embedding with Reversible Instance Normalization ===
        // Project input features to model dimension
        yield return new DenseLayer<T>(
            inputSize: contextLength * numFeatures,
            outputSize: modelDim * contextLength,
            activationFunction: new GELUActivation<T>());

        // Layer normalization (simulates RevIN mean/variance tracking)
        yield return new LayerNormalizationLayer<T>(modelDim * contextLength);

        // === Multi-Scale SSM Blocks (4 Mambas) ===
        // Each scale processes at different temporal granularity
        for (int scale = 0; scale < numScales; scale++)
        {
            // Downsampling factor for this scale (scale 0 = finest, scale n-1 = coarsest)
            // Scale-specific effective sequence length: contextLength / (2^scale)
            int scaleSeqLen = Math.Max(contextLength / (1 << scale), 1);
            int scaleDim = modelDim * scaleSeqLen;

            // === Temporal Decomposition for this scale ===
            // Simulates downsampling via dense projection
            // (Conv1DLayer doesn't exist, so we use dense layers to approximate)
            yield return new DenseLayer<T>(
                inputSize: modelDim * contextLength,
                outputSize: scaleDim,
                activationFunction: new GELUActivation<T>());

            yield return new LayerNormalizationLayer<T>(scaleDim);

            // === SSM Layers for this scale ===
            for (int layer = 0; layer < numLayers; layer++)
            {
                // === Mamba-style SSM Block ===

                // Input projection to expanded dimension
                yield return new DenseLayer<T>(
                    inputSize: scaleDim,
                    outputSize: innerDim * scaleSeqLen,
                    activationFunction: new SiLUActivation<T>());

                // Local context processing (simulates 1D convolution with dense layer)
                yield return new DenseLayer<T>(
                    inputSize: innerDim * scaleSeqLen,
                    outputSize: innerDim * scaleSeqLen,
                    activationFunction: new SiLUActivation<T>());

                // B projection (input to state)
                yield return new DenseLayer<T>(
                    inputSize: innerDim * scaleSeqLen,
                    outputSize: stateDim * scaleSeqLen,
                    activationFunction: null);

                // Selective state update (A diagonal)
                yield return new DenseLayer<T>(
                    inputSize: stateDim * scaleSeqLen,
                    outputSize: stateDim * scaleSeqLen,
                    activationFunction: new TanhActivation<T>());

                // C projection (state to output)
                yield return new DenseLayer<T>(
                    inputSize: stateDim * scaleSeqLen,
                    outputSize: innerDim * scaleSeqLen,
                    activationFunction: null);

                // Output projection back to model dimension
                yield return new DenseLayer<T>(
                    inputSize: innerDim * scaleSeqLen,
                    outputSize: scaleDim,
                    activationFunction: null);

                // Layer normalization and dropout
                yield return new LayerNormalizationLayer<T>(scaleDim);
                yield return new DropoutLayer<T>(0.1);
            }

            // === Upsampling back to original sequence length ===
            if (scale > 0)
            {
                yield return new DenseLayer<T>(
                    inputSize: scaleDim,
                    outputSize: modelDim * contextLength,
                    activationFunction: new GELUActivation<T>());
            }
        }

        // === Multi-Scale Fusion ===
        if (useMultiScaleAttention)
        {
            // Attention-based fusion of multi-scale outputs
            // Projects all scales to same dimension for attention
            yield return new DenseLayer<T>(
                inputSize: modelDim * contextLength * numScales,
                outputSize: modelDim * contextLength,
                activationFunction: new GELUActivation<T>());

            // Simple self-attention mechanism for scale weighting
            yield return new DenseLayer<T>(
                inputSize: modelDim * contextLength,
                outputSize: numScales,
                activationFunction: new SoftmaxActivation<T>());

            // Apply attention weights (simplified as dense)
            yield return new DenseLayer<T>(
                inputSize: numScales,
                outputSize: modelDim * contextLength,
                activationFunction: null);
        }

        yield return new LayerNormalizationLayer<T>(modelDim * contextLength);

        // === Reversible De-normalization ===
        // (Learned affine transformation to restore original scale)
        yield return new DenseLayer<T>(
            inputSize: modelDim * contextLength,
            outputSize: modelDim * contextLength,
            activationFunction: null);

        // === Output Projection ===
        yield return new DenseLayer<T>(
            inputSize: modelDim * contextLength,
            outputSize: modelDim * forecastHorizon / 4,
            activationFunction: new GELUActivation<T>());

        yield return new DenseLayer<T>(
            inputSize: modelDim * forecastHorizon / 4,
            outputSize: forecastHorizon,
            activationFunction: null);
    }

    /// <summary>
    /// Creates default layers for the HiPPO (High-order Polynomial Projection Operators) architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="contextLength">The input sequence length (default: 512).</param>
    /// <param name="forecastHorizon">The prediction horizon (default: 96).</param>
    /// <param name="modelDim">The model dimension d_model (default: 256).</param>
    /// <param name="stateDim">The HiPPO state dimension/polynomial order (default: 64).</param>
    /// <param name="numLayers">Number of HiPPO layers (default: 4).</param>
    /// <param name="useNormalization">Whether to use layer normalization (default: true).</param>
    /// <param name="numFeatures">Number of input features (default: 1).</param>
    /// <returns>A collection of layers forming the HiPPO architecture.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> HiPPO (High-order Polynomial Projection Operators) provides
    /// the theoretical foundation for state space models like S4 and Mamba.
    /// </para>
    /// <para>
    /// The architecture follows this flow:
    /// 1. Input embedding to model dimension
    /// 2. For each HiPPO layer:
    ///    - B projection (input to polynomial state)
    ///    - A matrix application (HiPPO state evolution)
    ///    - C projection (polynomial state to output)
    ///    - D feedthrough (skip connection)
    ///    - Normalization and dropout
    /// 3. Output projection to forecast horizon
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultHippoLayers(
        NeuralNetworkArchitecture<T> architecture,
        int contextLength = 512,
        int forecastHorizon = 96,
        int modelDim = 256,
        int stateDim = 64,
        int numLayers = 4,
        bool useNormalization = true,
        int numFeatures = 1)
    {
        // === Input Embedding ===
        yield return new DenseLayer<T>(
            inputSize: contextLength * numFeatures,
            outputSize: modelDim * contextLength,
            activationFunction: new GELUActivation<T>());

        if (useNormalization)
        {
            yield return new LayerNormalizationLayer<T>(modelDim * contextLength);
        }

        // === HiPPO Layers ===
        for (int layer = 0; layer < numLayers; layer++)
        {
            // === HiPPO Block ===
            // Each block implements: x' = Ax + Bu, y = Cx + Du

            // B projection (input to polynomial state space)
            // B maps input u to state contribution
            yield return new DenseLayer<T>(
                inputSize: modelDim * contextLength,
                outputSize: stateDim * contextLength,
                activationFunction: null);

            // A matrix application (HiPPO state evolution)
            // This simulates the HiPPO matrix A that defines optimal memory
            // In HiPPO-LegS: A[i,j] = -sqrt(2i+1)*sqrt(2j+1) if i>j, -(2i+1) if i==j
            yield return new DenseLayer<T>(
                inputSize: stateDim * contextLength,
                outputSize: stateDim * contextLength,
                activationFunction: new TanhActivation<T>()); // Tanh for stability

            // Second A application for deeper state evolution
            yield return new DenseLayer<T>(
                inputSize: stateDim * contextLength,
                outputSize: stateDim * contextLength,
                activationFunction: new TanhActivation<T>());

            // C projection (polynomial state to output)
            // C reads out the polynomial coefficients to produce output
            yield return new DenseLayer<T>(
                inputSize: stateDim * contextLength,
                outputSize: modelDim * contextLength,
                activationFunction: null);

            // D feedthrough (skip connection from input to output)
            // Allows direct information flow bypassing the state
            yield return new DenseLayer<T>(
                inputSize: modelDim * contextLength,
                outputSize: modelDim * contextLength,
                activationFunction: new GELUActivation<T>());

            // Normalization and dropout
            if (useNormalization)
            {
                yield return new LayerNormalizationLayer<T>(modelDim * contextLength);
            }
            yield return new DropoutLayer<T>(0.1);
        }

        // === FFN Block (post-HiPPO processing) ===
        yield return new DenseLayer<T>(
            inputSize: modelDim * contextLength,
            outputSize: modelDim * contextLength * 2,
            activationFunction: new GELUActivation<T>());

        yield return new DenseLayer<T>(
            inputSize: modelDim * contextLength * 2,
            outputSize: modelDim * contextLength,
            activationFunction: null);

        if (useNormalization)
        {
            yield return new LayerNormalizationLayer<T>(modelDim * contextLength);
        }

        // === Output Projection ===
        yield return new DenseLayer<T>(
            inputSize: modelDim * contextLength,
            outputSize: modelDim * forecastHorizon / 4,
            activationFunction: new GELUActivation<T>());

        yield return new DenseLayer<T>(
            inputSize: modelDim * forecastHorizon / 4,
            outputSize: forecastHorizon,
            activationFunction: null);
    }

    /// <summary>
    /// Creates default layers for the TimeGrad (Diffusion for Time Series) architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="contextLength">The input sequence length (default: 168).</param>
    /// <param name="forecastHorizon">The prediction horizon (default: 24).</param>
    /// <param name="hiddenDim">The RNN hidden dimension (default: 64).</param>
    /// <param name="numRnnLayers">Number of RNN layers (default: 2).</param>
    /// <param name="denoisingDim">The denoising network dimension (default: 128).</param>
    /// <param name="numDiffusionSteps">Number of diffusion steps (default: 100).</param>
    /// <param name="numFeatures">Number of input features (default: 1).</param>
    /// <returns>A collection of layers forming the TimeGrad architecture.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> TimeGrad uses a diffusion process for probabilistic forecasting.
    /// The architecture combines:
    /// 1. RNN encoder to process historical data
    /// 2. Denoising network conditioned on RNN hidden state
    /// 3. Multiple diffusion steps for noise removal
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultTimeGradLayers(
        NeuralNetworkArchitecture<T> architecture,
        int contextLength = 168,
        int forecastHorizon = 24,
        int hiddenDim = 64,
        int numRnnLayers = 2,
        int denoisingDim = 128,
        int numDiffusionSteps = 100,
        int numFeatures = 1)
    {
        // === Input Embedding ===
        yield return new DenseLayer<T>(
            inputSize: contextLength * numFeatures,
            outputSize: hiddenDim * contextLength,
            activationFunction: new GELUActivation<T>());

        yield return new LayerNormalizationLayer<T>(hiddenDim * contextLength);

        // === RNN Encoder ===
        // Simulated with dense layers (actual LSTM would need special handling)
        for (int i = 0; i < numRnnLayers; i++)
        {
            // Forget gate simulation
            yield return new DenseLayer<T>(
                inputSize: hiddenDim * contextLength,
                outputSize: hiddenDim * contextLength,
                activationFunction: new SigmoidActivation<T>());

            // Input gate simulation
            yield return new DenseLayer<T>(
                inputSize: hiddenDim * contextLength,
                outputSize: hiddenDim * contextLength,
                activationFunction: new SigmoidActivation<T>());

            // Cell state update
            yield return new DenseLayer<T>(
                inputSize: hiddenDim * contextLength,
                outputSize: hiddenDim * contextLength,
                activationFunction: new TanhActivation<T>());

            // Output projection
            yield return new DenseLayer<T>(
                inputSize: hiddenDim * contextLength,
                outputSize: hiddenDim * contextLength,
                activationFunction: null);

            yield return new LayerNormalizationLayer<T>(hiddenDim * contextLength);
            yield return new DropoutLayer<T>(0.1);
        }

        // === Denoising Network ===
        // Takes noisy forecast + RNN hidden state + timestep embedding

        // Timestep embedding projection
        yield return new DenseLayer<T>(
            inputSize: hiddenDim * contextLength,
            outputSize: denoisingDim,
            activationFunction: new SiLUActivation<T>());

        // Denoising blocks (simplified UNet-like structure)
        // Down path
        yield return new DenseLayer<T>(
            inputSize: denoisingDim,
            outputSize: denoisingDim * 2,
            activationFunction: new SiLUActivation<T>());

        yield return new DenseLayer<T>(
            inputSize: denoisingDim * 2,
            outputSize: denoisingDim * 4,
            activationFunction: new SiLUActivation<T>());

        yield return new LayerNormalizationLayer<T>(denoisingDim * 4);

        // Up path
        yield return new DenseLayer<T>(
            inputSize: denoisingDim * 4,
            outputSize: denoisingDim * 2,
            activationFunction: new SiLUActivation<T>());

        yield return new DenseLayer<T>(
            inputSize: denoisingDim * 2,
            outputSize: denoisingDim,
            activationFunction: new SiLUActivation<T>());

        yield return new LayerNormalizationLayer<T>(denoisingDim);

        // === Noise Prediction Head ===
        yield return new DenseLayer<T>(
            inputSize: denoisingDim,
            outputSize: denoisingDim / 2,
            activationFunction: new GELUActivation<T>());

        yield return new DenseLayer<T>(
            inputSize: denoisingDim / 2,
            outputSize: forecastHorizon,
            activationFunction: null);
    }

    /// <summary>
    /// Creates the default layer configuration for CSDI (Conditional Score-based Diffusion for Imputation).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> CSDI uses a transformer-based score network to learn the gradient
    /// of the log probability (the "score") for imputing missing values. The architecture consists of:
    ///
    /// 1. <b>Input Processing:</b> Concatenates observed values with mask indicating missing positions
    /// 2. <b>Time/Feature Embeddings:</b> Positional encodings for both time and feature dimensions
    /// 3. <b>Transformer Blocks:</b> Self-attention to capture dependencies across all positions
    /// 4. <b>Residual Blocks:</b> Process diffusion timestep and predict noise/score
    /// 5. <b>Output Head:</b> Predicts noise for the reverse diffusion process
    ///
    /// The key innovation of CSDI is conditioning on observed values: the model only predicts
    /// noise for missing positions while keeping observed values fixed during sampling.
    /// </para>
    /// </remarks>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="sequenceLength">Length of the input time series (default: 100).</param>
    /// <param name="numFeatures">Number of features/variables (default: 1).</param>
    /// <param name="hiddenDim">Hidden dimension for the score network (default: 64).</param>
    /// <param name="numResidualLayers">Number of residual blocks (default: 4).</param>
    /// <param name="numHeads">Number of attention heads (default: 8).</param>
    /// <param name="timeEmbeddingDim">Dimension of diffusion timestep embeddings (default: 128).</param>
    /// <param name="featureEmbeddingDim">Dimension of feature embeddings (default: 16).</param>
    /// <returns>An enumerable collection of layers for the CSDI model.</returns>
    public static IEnumerable<ILayer<T>> CreateDefaultCSDILayers(
        NeuralNetworkArchitecture<T> architecture,
        int sequenceLength = 100,
        int numFeatures = 1,
        int hiddenDim = 64,
        int numResidualLayers = 4,
        int numHeads = 8,
        int timeEmbeddingDim = 128,
        int featureEmbeddingDim = 16)
    {
        // Input size: values + mask (both of shape [seqLen * numFeatures])
        int inputSize = sequenceLength * numFeatures * 2;
        int flatSize = sequenceLength * numFeatures;

        // === Input Projection ===
        // Project concatenated (values, mask) to hidden dimension
        yield return new DenseLayer<T>(
            inputSize: inputSize,
            outputSize: hiddenDim * flatSize / numFeatures,
            activationFunction: new GELUActivation<T>());

        yield return new LayerNormalizationLayer<T>(hiddenDim * flatSize / numFeatures);

        // === Time Embedding Network ===
        // Process diffusion timestep through sinusoidal embedding + MLP
        yield return new DenseLayer<T>(
            inputSize: hiddenDim * flatSize / numFeatures,
            outputSize: timeEmbeddingDim,
            activationFunction: new SiLUActivation<T>());

        yield return new DenseLayer<T>(
            inputSize: timeEmbeddingDim,
            outputSize: timeEmbeddingDim,
            activationFunction: new SiLUActivation<T>());

        // === Feature Embedding ===
        // Learned embeddings for each feature dimension
        yield return new DenseLayer<T>(
            inputSize: timeEmbeddingDim,
            outputSize: featureEmbeddingDim * numFeatures,
            activationFunction: null);

        // === Transformer Encoder Blocks ===
        // Self-attention to capture temporal and cross-feature dependencies
        for (int i = 0; i < 2; i++)
        {
            // Self-attention (simulated with dense layers)
            yield return new DenseLayer<T>(
                inputSize: featureEmbeddingDim * numFeatures,
                outputSize: hiddenDim * numHeads,
                activationFunction: null); // Q projection

            yield return new DenseLayer<T>(
                inputSize: hiddenDim * numHeads,
                outputSize: hiddenDim * numHeads,
                activationFunction: null); // K projection

            yield return new DenseLayer<T>(
                inputSize: hiddenDim * numHeads,
                outputSize: hiddenDim * numHeads,
                activationFunction: null); // V projection

            yield return new DenseLayer<T>(
                inputSize: hiddenDim * numHeads,
                outputSize: hiddenDim,
                activationFunction: null); // Output projection

            yield return new LayerNormalizationLayer<T>(hiddenDim);
            yield return new DropoutLayer<T>(0.1);

            // Feedforward network
            yield return new DenseLayer<T>(
                inputSize: hiddenDim,
                outputSize: hiddenDim * 4,
                activationFunction: new GELUActivation<T>());

            yield return new DenseLayer<T>(
                inputSize: hiddenDim * 4,
                outputSize: hiddenDim,
                activationFunction: null);

            yield return new LayerNormalizationLayer<T>(hiddenDim);
            yield return new DropoutLayer<T>(0.1);
        }

        // === Residual Diffusion Blocks ===
        // Process timestep-conditioned features through residual blocks
        for (int i = 0; i < numResidualLayers; i++)
        {
            // First dense in residual block
            yield return new DenseLayer<T>(
                inputSize: hiddenDim,
                outputSize: hiddenDim * 2,
                activationFunction: new SiLUActivation<T>());

            // Timestep conditioning (would be added in actual implementation)
            yield return new DenseLayer<T>(
                inputSize: hiddenDim * 2,
                outputSize: hiddenDim * 2,
                activationFunction: new SiLUActivation<T>());

            // Second dense with residual connection
            yield return new DenseLayer<T>(
                inputSize: hiddenDim * 2,
                outputSize: hiddenDim,
                activationFunction: null);

            yield return new LayerNormalizationLayer<T>(hiddenDim);
        }

        // === Score/Noise Prediction Head ===
        // Predicts noise for reverse diffusion
        yield return new DenseLayer<T>(
            inputSize: hiddenDim,
            outputSize: hiddenDim,
            activationFunction: new GELUActivation<T>());

        yield return new DenseLayer<T>(
            inputSize: hiddenDim,
            outputSize: flatSize,
            activationFunction: null); // Output noise prediction for all positions
    }

    /// <summary>
    /// Creates the default layer configuration for TSDiff (Time Series Diffusion).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> TSDiff uses a U-Net style architecture with residual blocks
    /// and self-attention for denoising. The architecture includes:
    ///
    /// 1. <b>Input Embedding:</b> Projects input to hidden dimension
    /// 2. <b>Time Embedding:</b> Sinusoidal embeddings for diffusion timestep
    /// 3. <b>Downsampling Path:</b> Progressively reduce temporal resolution
    /// 4. <b>Middle Block:</b> Self-attention at lowest resolution
    /// 5. <b>Upsampling Path:</b> Progressively restore temporal resolution
    /// 6. <b>Output Head:</b> Predict noise for denoising
    ///
    /// The self-guidance mechanism uses intermediate predictions to refine generation.
    /// </para>
    /// </remarks>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="sequenceLength">Total sequence length (default: 192).</param>
    /// <param name="numFeatures">Number of features/variables (default: 1).</param>
    /// <param name="hiddenDim">Hidden dimension (default: 128).</param>
    /// <param name="numResidualBlocks">Number of residual blocks (default: 8).</param>
    /// <param name="numAttentionHeads">Number of attention heads (default: 4).</param>
    /// <returns>An enumerable collection of layers for the TSDiff model.</returns>
    public static IEnumerable<ILayer<T>> CreateDefaultTSDiffLayers(
        NeuralNetworkArchitecture<T> architecture,
        int sequenceLength = 192,
        int numFeatures = 1,
        int hiddenDim = 128,
        int numResidualBlocks = 8,
        int numAttentionHeads = 4)
    {
        int flatSize = sequenceLength * numFeatures;

        // === Input Embedding ===
        // Project input (noisy sequence) to hidden dimension
        yield return new DenseLayer<T>(
            inputSize: flatSize,
            outputSize: hiddenDim,
            activationFunction: new SiLUActivation<T>());

        yield return new LayerNormalizationLayer<T>(hiddenDim);

        // === Time Embedding Network ===
        // Process diffusion timestep
        yield return new DenseLayer<T>(
            inputSize: hiddenDim,
            outputSize: hiddenDim * 2,
            activationFunction: new SiLUActivation<T>());

        yield return new DenseLayer<T>(
            inputSize: hiddenDim * 2,
            outputSize: hiddenDim,
            activationFunction: new SiLUActivation<T>());

        // === Downsampling Path ===
        int currentDim = hiddenDim;
        for (int i = 0; i < numResidualBlocks / 2; i++)
        {
            // Residual block
            yield return new DenseLayer<T>(
                inputSize: currentDim,
                outputSize: currentDim,
                activationFunction: new SiLUActivation<T>());

            // Group normalization simulation
            yield return new LayerNormalizationLayer<T>(currentDim);

            yield return new DenseLayer<T>(
                inputSize: currentDim,
                outputSize: currentDim,
                activationFunction: new SiLUActivation<T>());

            yield return new DropoutLayer<T>(0.1);

            // Downsample (double channels)
            if (i < numResidualBlocks / 2 - 1)
            {
                yield return new DenseLayer<T>(
                    inputSize: currentDim,
                    outputSize: currentDim * 2,
                    activationFunction: null);
                currentDim *= 2;
            }
        }

        // === Middle Block with Self-Attention ===
        // Self-attention (simulated)
        yield return new DenseLayer<T>(
            inputSize: currentDim,
            outputSize: currentDim * numAttentionHeads,
            activationFunction: null); // Q

        yield return new DenseLayer<T>(
            inputSize: currentDim * numAttentionHeads,
            outputSize: currentDim * numAttentionHeads,
            activationFunction: null); // K

        yield return new DenseLayer<T>(
            inputSize: currentDim * numAttentionHeads,
            outputSize: currentDim * numAttentionHeads,
            activationFunction: null); // V

        yield return new DenseLayer<T>(
            inputSize: currentDim * numAttentionHeads,
            outputSize: currentDim,
            activationFunction: null); // Output projection

        yield return new LayerNormalizationLayer<T>(currentDim);

        // Middle residual block
        yield return new DenseLayer<T>(
            inputSize: currentDim,
            outputSize: currentDim,
            activationFunction: new SiLUActivation<T>());

        yield return new LayerNormalizationLayer<T>(currentDim);

        yield return new DenseLayer<T>(
            inputSize: currentDim,
            outputSize: currentDim,
            activationFunction: new SiLUActivation<T>());

        // === Upsampling Path ===
        for (int i = 0; i < numResidualBlocks / 2; i++)
        {
            // Upsample (halve channels)
            if (i > 0)
            {
                yield return new DenseLayer<T>(
                    inputSize: currentDim,
                    outputSize: currentDim / 2,
                    activationFunction: null);
                currentDim /= 2;
            }

            // Residual block
            yield return new DenseLayer<T>(
                inputSize: currentDim,
                outputSize: currentDim,
                activationFunction: new SiLUActivation<T>());

            yield return new LayerNormalizationLayer<T>(currentDim);

            yield return new DenseLayer<T>(
                inputSize: currentDim,
                outputSize: currentDim,
                activationFunction: new SiLUActivation<T>());

            yield return new DropoutLayer<T>(0.1);
        }

        // === Output Head ===
        yield return new DenseLayer<T>(
            inputSize: currentDim,
            outputSize: hiddenDim,
            activationFunction: new GELUActivation<T>());

        yield return new DenseLayer<T>(
            inputSize: hiddenDim,
            outputSize: flatSize,
            activationFunction: null); // Output noise prediction
    }

    /// <summary>
    /// Creates the default layer configuration for DiffusionTS (Interpretable Diffusion for Time Series).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> DiffusionTS uses a decomposition-based architecture that generates
    /// trend, seasonal, and residual components separately:
    ///
    /// 1. <b>Trend Network:</b> Smooth, low-frequency component (moving average style)
    /// 2. <b>Seasonal Network:</b> Periodic patterns with fourier-like features
    /// 3. <b>Residual Network:</b> Irregular/noise component
    /// 4. <b>Fusion Module:</b> Combines all components coherently
    ///
    /// Each component network has different capacity appropriate for its complexity.
    /// </para>
    /// </remarks>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="sequenceLength">Input sequence length (default: 168).</param>
    /// <param name="forecastHorizon">Forecast horizon (default: 24).</param>
    /// <param name="numFeatures">Number of features (default: 1).</param>
    /// <param name="hiddenDim">Main hidden dimension (default: 64).</param>
    /// <param name="trendHiddenDim">Trend network hidden dim (default: 32).</param>
    /// <param name="seasonalHiddenDim">Seasonal network hidden dim (default: 48).</param>
    /// <returns>An enumerable collection of layers for the DiffusionTS model.</returns>
    public static IEnumerable<ILayer<T>> CreateDefaultDiffusionTSLayers(
        NeuralNetworkArchitecture<T> architecture,
        int sequenceLength = 168,
        int forecastHorizon = 24,
        int numFeatures = 1,
        int hiddenDim = 64,
        int trendHiddenDim = 32,
        int seasonalHiddenDim = 48)
    {
        int contextLength = sequenceLength;
        int flatContextSize = contextLength * numFeatures;
        int flatForecastSize = forecastHorizon * numFeatures;

        // === Input Embedding ===
        yield return new DenseLayer<T>(
            inputSize: flatContextSize,
            outputSize: hiddenDim,
            activationFunction: new GELUActivation<T>());

        yield return new LayerNormalizationLayer<T>(hiddenDim);

        // === Trend Network (smooth, low capacity) ===
        // Trend should be smooth, so use fewer parameters
        yield return new DenseLayer<T>(
            inputSize: hiddenDim,
            outputSize: trendHiddenDim,
            activationFunction: new GELUActivation<T>());

        yield return new LayerNormalizationLayer<T>(trendHiddenDim);

        yield return new DenseLayer<T>(
            inputSize: trendHiddenDim,
            outputSize: trendHiddenDim,
            activationFunction: new GELUActivation<T>());

        // Trend output
        yield return new DenseLayer<T>(
            inputSize: trendHiddenDim,
            outputSize: flatForecastSize,
            activationFunction: null);

        // === Seasonal Network (periodic patterns) ===
        // Project back to hidden for seasonal processing
        yield return new DenseLayer<T>(
            inputSize: flatForecastSize,
            outputSize: seasonalHiddenDim,
            activationFunction: new GELUActivation<T>());

        yield return new LayerNormalizationLayer<T>(seasonalHiddenDim);

        // Fourier-like feature extraction (Tanh for bounded periodic-like output)
        yield return new DenseLayer<T>(
            inputSize: seasonalHiddenDim,
            outputSize: seasonalHiddenDim * 2,
            activationFunction: new TanhActivation<T>());

        yield return new DenseLayer<T>(
            inputSize: seasonalHiddenDim * 2,
            outputSize: seasonalHiddenDim,
            activationFunction: new GELUActivation<T>());

        yield return new LayerNormalizationLayer<T>(seasonalHiddenDim);

        // Seasonal output
        yield return new DenseLayer<T>(
            inputSize: seasonalHiddenDim,
            outputSize: flatForecastSize,
            activationFunction: null);

        // === Residual Network (irregular patterns) ===
        // Project for residual processing
        yield return new DenseLayer<T>(
            inputSize: flatForecastSize,
            outputSize: hiddenDim,
            activationFunction: new SiLUActivation<T>());

        yield return new LayerNormalizationLayer<T>(hiddenDim);
        yield return new DropoutLayer<T>(0.1);

        yield return new DenseLayer<T>(
            inputSize: hiddenDim,
            outputSize: hiddenDim,
            activationFunction: new SiLUActivation<T>());

        yield return new LayerNormalizationLayer<T>(hiddenDim);

        // Residual output
        yield return new DenseLayer<T>(
            inputSize: hiddenDim,
            outputSize: flatForecastSize,
            activationFunction: null);

        // === Fusion Module ===
        // Combine trend + seasonal + residual
        // Takes concatenated components and produces final output
        yield return new DenseLayer<T>(
            inputSize: flatForecastSize,
            outputSize: hiddenDim,
            activationFunction: new GELUActivation<T>());

        yield return new LayerNormalizationLayer<T>(hiddenDim);

        yield return new DenseLayer<T>(
            inputSize: hiddenDim,
            outputSize: flatForecastSize,
            activationFunction: null); // Final noise/output prediction
    }

    /// <summary>
    /// Creates default layers for the ScoreGrad (Score-based Gradient) model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="sequenceLength">Length of input sequence (default: 168).</param>
    /// <param name="forecastHorizon">Number of steps to forecast (default: 24).</param>
    /// <param name="numFeatures">Number of input features (default: 1).</param>
    /// <param name="hiddenDim">Hidden layer dimension (default: 128).</param>
    /// <param name="numLayers">Number of score network layers (default: 4).</param>
    /// <returns>Collection of layers for the ScoreGrad architecture.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> ScoreGrad architecture has these key components:
    ///
    /// <b>1. Input Embedding:</b>
    /// - Projects input time series and noise level to hidden dimension
    /// - Combines context, noisy forecast, and sigma embedding
    ///
    /// <b>2. Score Network Core:</b>
    /// - Multiple residual dense blocks
    /// - Each block: Dense → GELU → LayerNorm → Dense → Skip connection
    /// - Learns the score function ∇_x log p(x|σ)
    ///
    /// <b>3. Noise Level Conditioning:</b>
    /// - Sinusoidal embedding of sigma value
    /// - Allows network to output different scores at different noise levels
    /// - Critical for annealed Langevin sampling
    ///
    /// <b>4. Output Head:</b>
    /// - Projects hidden state back to forecast dimension
    /// - Outputs the score (gradient direction toward higher probability)
    ///
    /// <b>Key Design Choices:</b>
    /// - Residual connections for gradient flow
    /// - Layer normalization for stability
    /// - GELU activation for smooth gradients
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultScoreGradLayers(
        NeuralNetworkArchitecture<T> architecture,
        int sequenceLength = 168,
        int forecastHorizon = 24,
        int numFeatures = 1,
        int hiddenDim = 128,
        int numLayers = 4)
    {
        int contextSize = (sequenceLength - forecastHorizon) * numFeatures;
        int forecastSize = forecastHorizon * numFeatures;
        int noiseEmbedDim = hiddenDim / 4; // Smaller embedding for noise level

        // Total input: context + noisy forecast + noise embedding
        int totalInputSize = contextSize + forecastSize + noiseEmbedDim;

        // === Input Projection ===
        // Combine all inputs into hidden dimension
        yield return new DenseLayer<T>(
            inputSize: totalInputSize,
            outputSize: hiddenDim,
            activationFunction: new GELUActivation<T>());

        yield return new LayerNormalizationLayer<T>(hiddenDim);

        // === Score Network Core (Residual Blocks) ===
        // Multiple dense blocks that learn the score function
        for (int i = 0; i < numLayers; i++)
        {
            // First dense layer of residual block
            yield return new DenseLayer<T>(
                inputSize: hiddenDim,
                outputSize: hiddenDim * 2,
                activationFunction: new GELUActivation<T>());

            // Second dense layer (reduce back to hidden dim for residual connection)
            yield return new DenseLayer<T>(
                inputSize: hiddenDim * 2,
                outputSize: hiddenDim,
                activationFunction: new GELUActivation<T>());

            yield return new LayerNormalizationLayer<T>(hiddenDim);
        }

        // === Score Output Head ===
        // Project to forecast dimension - this is the score (gradient)
        yield return new DenseLayer<T>(
            inputSize: hiddenDim,
            outputSize: hiddenDim / 2,
            activationFunction: new GELUActivation<T>());

        yield return new DenseLayer<T>(
            inputSize: hiddenDim / 2,
            outputSize: forecastSize,
            activationFunction: null); // Score output (can be any value, not bounded)
    }

    /// <summary>
    /// Creates default layers for the STGNN (Spatio-Temporal Graph Neural Network) model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="sequenceLength">Length of input sequence (default: 12).</param>
    /// <param name="forecastHorizon">Number of steps to forecast (default: 12).</param>
    /// <param name="numNodes">Number of nodes in the graph (default: 207).</param>
    /// <param name="numFeatures">Number of features per node (default: 1).</param>
    /// <param name="hiddenDim">Hidden layer dimension (default: 64).</param>
    /// <param name="numSpatialLayers">Number of graph convolution layers (default: 2).</param>
    /// <param name="numTemporalLayers">Number of temporal convolution layers (default: 2).</param>
    /// <returns>Collection of layers for the STGNN architecture.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> STGNN architecture alternates between spatial and temporal processing:
    ///
    /// <b>1. Input Embedding:</b>
    /// - Projects node features to hidden dimension
    /// - Prepares data for graph operations
    ///
    /// <b>2. ST-Conv Blocks (repeated):</b>
    /// Each block contains:
    /// - Temporal Conv: Captures time patterns (1D conv along time axis)
    /// - Spatial Conv: Aggregates neighbor info (graph convolution)
    /// - Temporal Conv: Another temporal pass for ST fusion
    /// - Layer Norm + Residual connection
    ///
    /// <b>3. Output Layer:</b>
    /// - Projects to forecast dimension
    /// - Outputs predictions for all nodes
    ///
    /// <b>Key Design Choices:</b>
    /// - Sandwich structure: Temporal-Spatial-Temporal captures complex ST dependencies
    /// - Residual connections prevent gradient vanishing
    /// - Layer normalization stabilizes training
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultSTGNNLayers(
        NeuralNetworkArchitecture<T> architecture,
        int sequenceLength = 12,
        int forecastHorizon = 12,
        int numNodes = 207,
        int numFeatures = 1,
        int hiddenDim = 64,
        int numSpatialLayers = 2,
        int numTemporalLayers = 2)
    {
        // Input size: nodes * sequence * features (flattened)
        int inputSize = numNodes * sequenceLength * numFeatures;
        int outputSize = numNodes * forecastHorizon * numFeatures;

        // === Input Embedding ===
        // Project input features to hidden dimension
        yield return new DenseLayer<T>(
            inputSize: inputSize,
            outputSize: numNodes * hiddenDim,
            activationFunction: new GELUActivation<T>());

        yield return new LayerNormalizationLayer<T>(numNodes * hiddenDim);

        // === ST-Conv Blocks ===
        // Each block: Temporal -> Spatial -> Temporal
        int numBlocks = Math.Max(numSpatialLayers, numTemporalLayers);
        for (int block = 0; block < numBlocks; block++)
        {
            // Temporal convolution (simulated with dense layers for time processing)
            yield return new DenseLayer<T>(
                inputSize: numNodes * hiddenDim,
                outputSize: numNodes * hiddenDim,
                activationFunction: new GELUActivation<T>());

            // Spatial convolution (graph aggregation simulated with dense)
            // In practice, this would use actual graph convolution with adjacency matrix
            yield return new DenseLayer<T>(
                inputSize: numNodes * hiddenDim,
                outputSize: numNodes * hiddenDim,
                activationFunction: new GELUActivation<T>());

            // Second temporal convolution
            yield return new DenseLayer<T>(
                inputSize: numNodes * hiddenDim,
                outputSize: numNodes * hiddenDim,
                activationFunction: new GELUActivation<T>());

            yield return new LayerNormalizationLayer<T>(numNodes * hiddenDim);
        }

        // === Output Projection ===
        // Project to forecast dimension
        yield return new DenseLayer<T>(
            inputSize: numNodes * hiddenDim,
            outputSize: numNodes * hiddenDim / 2,
            activationFunction: new GELUActivation<T>());

        yield return new DenseLayer<T>(
            inputSize: numNodes * hiddenDim / 2,
            outputSize: outputSize,
            activationFunction: null);
    }

    /// <summary>
    /// Creates default layers for TemporalGCN (Temporal Graph Convolutional Network).
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="sequenceLength">Number of input time steps.</param>
    /// <param name="forecastHorizon">Number of future time steps to predict.</param>
    /// <param name="numNodes">Number of nodes in the graph.</param>
    /// <param name="numFeatures">Number of features per node.</param>
    /// <param name="hiddenDim">Hidden dimension for GCN and temporal layers.</param>
    /// <param name="numGCNLayers">Number of graph convolution layers.</param>
    /// <param name="numTemporalLayers">Number of temporal (recurrent) layers.</param>
    /// <returns>An enumerable of layers configured for TemporalGCN.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates layers for TemporalGCN which combines
    /// graph convolution with recurrent neural networks to capture both spatial
    /// dependencies (through GCN) and temporal patterns (through GRU/LSTM).
    /// </para>
    /// <para>
    /// The architecture alternates between:
    /// 1. Graph convolution layers that aggregate neighbor information
    /// 2. Temporal recurrent layers that model sequence patterns
    /// This allows the network to learn how spatial patterns evolve over time.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultTemporalGCNLayers(
        NeuralNetworkArchitecture<T> architecture,
        int sequenceLength = 12,
        int forecastHorizon = 12,
        int numNodes = 207,
        int numFeatures = 1,
        int hiddenDim = 64,
        int numGCNLayers = 2,
        int numTemporalLayers = 1)
    {
        // Input size: nodes * sequence * features (flattened)
        int inputSize = numNodes * sequenceLength * numFeatures;
        int outputSize = numNodes * forecastHorizon * numFeatures;

        // === Input Projection ===
        // Project input features to hidden dimension
        yield return new DenseLayer<T>(
            inputSize: inputSize,
            outputSize: numNodes * hiddenDim,
            activationFunction: new ReLUActivation<T>());

        // === GCN Layers ===
        // Stack graph convolution layers for spatial aggregation
        for (int gcn = 0; gcn < numGCNLayers; gcn++)
        {
            // Graph convolution (simulated with dense + aggregation pattern)
            // Each GCN layer aggregates information from k-hop neighbors
            yield return new DenseLayer<T>(
                inputSize: numNodes * hiddenDim,
                outputSize: numNodes * hiddenDim,
                activationFunction: new ReLUActivation<T>());

            yield return new LayerNormalizationLayer<T>(numNodes * hiddenDim);

            // Dropout for regularization
            yield return new DropoutLayer<T>(0.3);
        }

        // === Temporal GRU Layers ===
        // Process temporal sequence with recurrent layers
        for (int t = 0; t < numTemporalLayers; t++)
        {
            // GRU layer for temporal modeling (using dense layers as approximation)
            // Real implementation would use proper GRU cells with gating
            yield return new DenseLayer<T>(
                inputSize: numNodes * hiddenDim,
                outputSize: numNodes * hiddenDim,
                activationFunction: new TanhActivation<T>());

            // Update gate approximation
            yield return new DenseLayer<T>(
                inputSize: numNodes * hiddenDim,
                outputSize: numNodes * hiddenDim,
                activationFunction: new SigmoidActivation<T>());

            yield return new LayerNormalizationLayer<T>(numNodes * hiddenDim);
        }

        // === Spatio-Temporal Fusion ===
        // Combine spatial and temporal features
        yield return new DenseLayer<T>(
            inputSize: numNodes * hiddenDim,
            outputSize: numNodes * hiddenDim,
            activationFunction: new ReLUActivation<T>());

        yield return new LayerNormalizationLayer<T>(numNodes * hiddenDim);

        // === Output Projection ===
        // Project to forecast horizon
        yield return new DenseLayer<T>(
            inputSize: numNodes * hiddenDim,
            outputSize: numNodes * hiddenDim / 2,
            activationFunction: new ReLUActivation<T>());

        yield return new DenseLayer<T>(
            inputSize: numNodes * hiddenDim / 2,
            outputSize: outputSize,
            activationFunction: null);
    }

    /// <summary>
    /// Creates default layers for MTGNN (Multivariate Time-series Graph Neural Network).
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="sequenceLength">Number of input time steps.</param>
    /// <param name="forecastHorizon">Number of future time steps to predict.</param>
    /// <param name="numNodes">Number of nodes (variables/time series).</param>
    /// <param name="numFeatures">Number of features per node.</param>
    /// <param name="hiddenDim">Hidden dimension for processing layers.</param>
    /// <param name="nodeEmbeddingDim">Dimension of node embeddings for graph learning.</param>
    /// <param name="numLayers">Number of graph-temporal processing layers.</param>
    /// <param name="mixHopDepth">Depth of mix-hop propagation.</param>
    /// <returns>An enumerable of layers configured for MTGNN.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates layers for MTGNN which automatically learns
    /// the graph structure while performing spatio-temporal forecasting. The key components are:
    /// </para>
    /// <para>
    /// 1. Node embedding layers that learn representations for graph structure discovery
    /// 2. Mix-hop propagation layers for multi-scale spatial aggregation
    /// 3. Dilated temporal convolution layers for multi-scale temporal patterns
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultMTGNNLayers(
        NeuralNetworkArchitecture<T> architecture,
        int sequenceLength = 12,
        int forecastHorizon = 12,
        int numNodes = 207,
        int numFeatures = 1,
        int hiddenDim = 32,
        int nodeEmbeddingDim = 40,
        int numLayers = 3,
        int mixHopDepth = 2)
    {
        // Input size: nodes * sequence * features (flattened)
        int inputSize = numNodes * sequenceLength * numFeatures;
        int outputSize = numNodes * forecastHorizon * numFeatures;

        // === Input Projection ===
        // Project input to hidden dimension
        yield return new DenseLayer<T>(
            inputSize: inputSize,
            outputSize: numNodes * hiddenDim,
            activationFunction: new ReLUActivation<T>());

        yield return new LayerNormalizationLayer<T>(numNodes * hiddenDim);

        // === Node Embedding Layer ===
        // Create learnable node embeddings for adaptive graph learning
        yield return new DenseLayer<T>(
            inputSize: numNodes * hiddenDim,
            outputSize: numNodes * nodeEmbeddingDim,
            activationFunction: new TanhActivation<T>());

        // === Graph-Temporal Processing Layers ===
        for (int layer = 0; layer < numLayers; layer++)
        {
            // Mix-hop graph convolution (simulated with dense layers)
            // Aggregates 1-hop, 2-hop, ... k-hop neighbors
            for (int hop = 0; hop < mixHopDepth; hop++)
            {
                yield return new DenseLayer<T>(
                    inputSize: numNodes * (hop == 0 ? nodeEmbeddingDim : hiddenDim),
                    outputSize: numNodes * hiddenDim,
                    activationFunction: new ReLUActivation<T>());
            }

            // Dilated temporal convolution (simulated with dense)
            // Different dilation rates capture different temporal scales
            yield return new DenseLayer<T>(
                inputSize: numNodes * hiddenDim,
                outputSize: numNodes * hiddenDim,
                activationFunction: new TanhActivation<T>());

            // Gated skip connection
            yield return new DenseLayer<T>(
                inputSize: numNodes * hiddenDim,
                outputSize: numNodes * hiddenDim,
                activationFunction: new SigmoidActivation<T>());

            yield return new LayerNormalizationLayer<T>(numNodes * hiddenDim);
            yield return new DropoutLayer<T>(0.3);
        }

        // === Output Projection ===
        // Project to forecast dimension
        yield return new DenseLayer<T>(
            inputSize: numNodes * hiddenDim,
            outputSize: numNodes * hiddenDim,
            activationFunction: new ReLUActivation<T>());

        yield return new DenseLayer<T>(
            inputSize: numNodes * hiddenDim,
            outputSize: outputSize,
            activationFunction: null);
    }

    /// <summary>
    /// Creates default layers for GraphWaveNet (Graph WaveNet for spatial-temporal modeling).
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="sequenceLength">Number of input time steps.</param>
    /// <param name="forecastHorizon">Number of future time steps to predict.</param>
    /// <param name="numNodes">Number of nodes in the graph.</param>
    /// <param name="numFeatures">Number of input features per node.</param>
    /// <param name="residualChannels">Number of residual channels.</param>
    /// <param name="skipChannels">Number of skip connection channels.</param>
    /// <param name="endChannels">Number of end (output) channels.</param>
    /// <param name="numBlocks">Number of temporal convolution blocks.</param>
    /// <param name="layersPerBlock">Number of layers per block.</param>
    /// <returns>An enumerable of layers configured for GraphWaveNet.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates layers for GraphWaveNet which combines:
    /// </para>
    /// <para>
    /// 1. Adaptive graph learning via node embeddings
    /// 2. Diffusion convolution for bidirectional spatial propagation
    /// 3. Gated dilated convolutions (WaveNet-style) for temporal patterns
    /// 4. Skip connections from each layer to the output
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultGraphWaveNetLayers(
        NeuralNetworkArchitecture<T> architecture,
        int sequenceLength = 12,
        int forecastHorizon = 12,
        int numNodes = 207,
        int numFeatures = 2,
        int residualChannels = 32,
        int skipChannels = 256,
        int endChannels = 512,
        int numBlocks = 4,
        int layersPerBlock = 2)
    {
        // Input size: nodes * sequence * features (flattened)
        int inputSize = numNodes * sequenceLength * numFeatures;
        int outputSize = numNodes * forecastHorizon;

        // === Input Projection ===
        // Project input to residual channels
        yield return new DenseLayer<T>(
            inputSize: inputSize,
            outputSize: numNodes * residualChannels,
            activationFunction: new ReLUActivation<T>());

        yield return new LayerNormalizationLayer<T>(numNodes * residualChannels);

        // === WaveNet-style Blocks ===
        for (int block = 0; block < numBlocks; block++)
        {
            for (int layer = 0; layer < layersPerBlock; layer++)
            {
                // Dilated convolution (filter branch) - captures temporal patterns
                yield return new DenseLayer<T>(
                    inputSize: numNodes * residualChannels,
                    outputSize: numNodes * residualChannels,
                    activationFunction: new TanhActivation<T>());

                // Gate convolution - controls information flow
                yield return new DenseLayer<T>(
                    inputSize: numNodes * residualChannels,
                    outputSize: numNodes * residualChannels,
                    activationFunction: new SigmoidActivation<T>());

                // Diffusion convolution placeholder (spatial aggregation)
                yield return new DenseLayer<T>(
                    inputSize: numNodes * residualChannels,
                    outputSize: numNodes * residualChannels,
                    activationFunction: new ReLUActivation<T>());

                yield return new LayerNormalizationLayer<T>(numNodes * residualChannels);

                // Skip connection projection
                yield return new DenseLayer<T>(
                    inputSize: numNodes * residualChannels,
                    outputSize: numNodes * skipChannels / numBlocks,
                    activationFunction: null);

                yield return new DropoutLayer<T>(0.3);
            }
        }

        // === Output Processing ===
        // Combine skip connections and project to output
        yield return new DenseLayer<T>(
            inputSize: numNodes * skipChannels / numBlocks,
            outputSize: numNodes * endChannels,
            activationFunction: new ReLUActivation<T>());

        yield return new DenseLayer<T>(
            inputSize: numNodes * endChannels,
            outputSize: outputSize,
            activationFunction: null);
    }

    /// <summary>
    /// Creates the default layers for a DCRNN (Diffusion Convolutional Recurrent Neural Network).
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="numNodes">Number of nodes in the graph (sensors/locations).</param>
    /// <param name="numFeatures">Number of input features per node.</param>
    /// <param name="hiddenDimension">Hidden dimension for DCGRU cells.</param>
    /// <param name="numEncoderLayers">Number of encoder DCGRU layers.</param>
    /// <param name="numDecoderLayers">Number of decoder DCGRU layers.</param>
    /// <param name="forecastHorizon">Number of future time steps to predict.</param>
    /// <param name="diffusionSteps">Number of diffusion steps (K).</param>
    /// <returns>A collection of layers for the DCRNN architecture.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> DCRNN (Diffusion Convolutional Recurrent Neural Network) was designed
    /// specifically for traffic forecasting. It has two main innovations:
    /// </para>
    /// <para>
    /// <b>1. Diffusion Convolution:</b> Instead of standard graph convolution, DCRNN models traffic
    /// flow as a diffusion process - like how a drop of dye spreads through water. This is done
    /// using random walk matrices that capture how influence propagates through the network.
    /// </para>
    /// <para>
    /// <b>2. Encoder-Decoder Architecture:</b> The encoder reads the historical data and compresses
    /// it into a context vector. The decoder then generates predictions one step at a time, using
    /// both the context and its own previous outputs.
    /// </para>
    /// <para>
    /// <b>3. Scheduled Sampling:</b> During training, the model gradually transitions from using
    /// ground truth inputs (teacher forcing) to using its own predictions. This helps the model
    /// learn to handle its own errors during inference.
    /// </para>
    /// <para>
    /// The architecture mirrors the original paper: encoder DCGRU layers followed by decoder
    /// DCGRU layers, with a final projection layer for output.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultDCRNNLayers(
        NeuralNetworkArchitecture<T> architecture,
        int numNodes = 207,
        int numFeatures = 2,
        int hiddenDimension = 64,
        int numEncoderLayers = 2,
        int numDecoderLayers = 2,
        int forecastHorizon = 12,
        int diffusionSteps = 2)
    {
        // === Input Embedding ===
        // Project input features to hidden dimension
        yield return new DenseLayer<T>(
            inputSize: numNodes * numFeatures,
            outputSize: numNodes * hiddenDimension,
            activationFunction: null);

        // === Encoder DCGRU Layers ===
        // Each layer is a GRU where matrix multiplications are replaced with diffusion convolution
        // Note: Diffusion convolution is applied in the model itself, not in layers
        for (int i = 0; i < numEncoderLayers; i++)
        {
            // GRU layer for temporal processing
            yield return new GRULayer<T>(
                inputSize: hiddenDimension,
                hiddenSize: hiddenDimension,
                returnSequences: true,
                activation: (IActivationFunction<T>?)null,
                recurrentActivation: null);

            // Dense layer for spatial mixing (simplified diffusion approximation)
            yield return new DenseLayer<T>(
                inputSize: numNodes * hiddenDimension,
                outputSize: numNodes * hiddenDimension,
                activationFunction: new ReLUActivation<T>());
        }

        // === Decoder DCGRU Layers ===
        // Decoder generates predictions autoregressively
        for (int i = 0; i < numDecoderLayers; i++)
        {
            // GRU layer for temporal processing
            yield return new GRULayer<T>(
                inputSize: hiddenDimension,
                hiddenSize: hiddenDimension,
                returnSequences: true,
                activation: (IActivationFunction<T>?)null,
                recurrentActivation: null);

            // Dense layer for spatial mixing
            yield return new DenseLayer<T>(
                inputSize: numNodes * hiddenDimension,
                outputSize: numNodes * hiddenDimension,
                activationFunction: new ReLUActivation<T>());
        }

        // === Output Projection ===
        // Project hidden states to output predictions
        yield return new DenseLayer<T>(
            inputSize: numNodes * hiddenDimension,
            outputSize: numNodes * forecastHorizon,
            activationFunction: null);
    }

    /// <summary>
    /// Creates default layers for RelationalGCN (Relational Graph Convolutional Network).
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="numNodes">Number of nodes (entities) in the knowledge graph.</param>
    /// <param name="numFeatures">Number of input features per node.</param>
    /// <param name="numRelations">Number of relation types in the graph.</param>
    /// <param name="hiddenDimension">Hidden dimension for R-GCN layers.</param>
    /// <param name="numLayers">Number of R-GCN layers.</param>
    /// <param name="numBases">Number of bases for basis decomposition.</param>
    /// <param name="forecastHorizon">Number of time steps to forecast.</param>
    /// <returns>An enumerable of layers configured for RelationalGCN.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> RelationalGCN extends Graph Convolutional Networks to handle
    /// multi-relational data where different types of edges (relations) exist between nodes.
    /// This is essential for knowledge graphs and heterogeneous networks.
    /// </para>
    /// <para>
    /// The key insight is that different relationship types should have different transformation
    /// weights. For example, in a financial network, a "supplies-to" relationship is fundamentally
    /// different from a "competes-with" relationship.
    /// </para>
    /// <para>
    /// R-GCN uses basis decomposition for parameter efficiency: instead of having a full weight
    /// matrix for each relation, relation weights are linear combinations of shared basis matrices.
    /// This dramatically reduces parameters when you have many relation types.
    /// </para>
    /// <para>
    /// The formula: W_r = sum_b (a_rb * B_b) where B_b are shared bases and a_rb are
    /// relation-specific combination coefficients.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultRelationalGCNLayers(
        NeuralNetworkArchitecture<T> architecture,
        int numNodes = 100,
        int numFeatures = 16,
        int numRelations = 10,
        int hiddenDimension = 64,
        int numLayers = 2,
        int numBases = 5,
        int forecastHorizon = 12)
    {
        // === Input Embedding ===
        // Project input node features to hidden dimension
        yield return new DenseLayer<T>(
            inputSize: numFeatures,
            outputSize: hiddenDimension,
            activationFunction: new ReLUActivation<T>());

        // === Basis Matrices ===
        // Shared basis matrices for parameter-efficient relation-specific transformations
        // In practice, these are learned as part of the model's forward pass
        for (int b = 0; b < numBases; b++)
        {
            yield return new DenseLayer<T>(
                inputSize: hiddenDimension,
                outputSize: hiddenDimension,
                activationFunction: null);
        }

        // === Relation Coefficient Layers ===
        // Each relation has coefficients for combining basis matrices
        // This creates R-GCN's W_r = sum_b (a_rb * B_b) structure
        yield return new DenseLayer<T>(
            inputSize: numRelations * numBases,
            outputSize: numRelations * numBases,
            activationFunction: null);

        // === R-GCN Layers ===
        // Stacked relational graph convolution layers
        for (int layer = 0; layer < numLayers; layer++)
        {
            // Graph aggregation layer - aggregates messages from neighbors per relation
            // Then combines across relations
            yield return new DenseLayer<T>(
                inputSize: numNodes * hiddenDimension,
                outputSize: numNodes * hiddenDimension,
                activationFunction: new ReLUActivation<T>());

            // Self-loop transformation - nodes also consider their own features
            yield return new DenseLayer<T>(
                inputSize: hiddenDimension,
                outputSize: hiddenDimension,
                activationFunction: null);

            // Dropout layer for regularization
            yield return new DropoutLayer<T>(dropoutRate: 0.2);
        }

        // === Temporal Processing ===
        // For time series forecasting, add temporal layers
        yield return new GRULayer<T>(
            inputSize: hiddenDimension,
            hiddenSize: hiddenDimension,
            returnSequences: true,
            activation: (IActivationFunction<T>?)null,
            recurrentActivation: null);

        // === Output Projection ===
        // Project to forecast horizon
        yield return new DenseLayer<T>(
            inputSize: numNodes * hiddenDimension,
            outputSize: numNodes * forecastHorizon,
            activationFunction: null);
    }

    /// <summary>
    /// Creates default layers for FinBERT (Financial BERT) model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="vocabularySize">Size of the token vocabulary.</param>
    /// <param name="maxSequenceLength">Maximum sequence length in tokens.</param>
    /// <param name="hiddenDimension">Hidden dimension for transformer layers.</param>
    /// <param name="numAttentionHeads">Number of attention heads.</param>
    /// <param name="intermediateDimension">Feed-forward intermediate dimension.</param>
    /// <param name="numLayers">Number of transformer layers.</param>
    /// <param name="numSentimentClasses">Number of output sentiment classes.</param>
    /// <param name="dropoutRate">Dropout rate for regularization.</param>
    /// <returns>An enumerable of layers configured for FinBERT.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> FinBERT is a BERT-based model fine-tuned for financial sentiment
    /// analysis. BERT (Bidirectional Encoder Representations from Transformers) uses
    /// self-attention to understand context from both directions of the text.
    /// </para>
    /// <para>
    /// The architecture follows the original BERT design:
    /// 1. Token embeddings convert tokens to vectors
    /// 2. Position embeddings encode sequence position
    /// 3. Multiple transformer layers with self-attention
    /// 4. Classification head for sentiment prediction
    /// </para>
    /// <para>
    /// FinBERT adds financial domain knowledge through pre-training and fine-tuning on
    /// financial text, enabling it to understand financial terminology and sentiment nuances.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultFinBERTLayers(
        NeuralNetworkArchitecture<T> architecture,
        int vocabularySize = 30522,
        int maxSequenceLength = 512,
        int hiddenDimension = 768,
        int numAttentionHeads = 12,
        int intermediateDimension = 3072,
        int numLayers = 12,
        int numSentimentClasses = 3,
        double dropoutRate = 0.1)
    {
        // === Token Embedding Layer ===
        // Maps token IDs to dense vectors
        yield return new EmbeddingLayer<T>(
            vocabularySize: vocabularySize,
            embeddingDimension: hiddenDimension);

        // === Position Embedding Layer ===
        // Adds position information to token embeddings
        yield return new EmbeddingLayer<T>(
            vocabularySize: maxSequenceLength,
            embeddingDimension: hiddenDimension);

        // === Layer Normalization ===
        // Normalizes embeddings before transformer layers
        yield return new LayerNormalizationLayer<T>(
            featureSize: hiddenDimension);

        // === Dropout ===
        yield return new DropoutLayer<T>(dropoutRate: dropoutRate);

        // === Transformer Layers ===
        // Stack of transformer encoder blocks
        for (int i = 0; i < numLayers; i++)
        {
            // Multi-head self-attention
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: maxSequenceLength,
                embeddingDimension: hiddenDimension,
                headCount: numAttentionHeads);

            // Add & Norm after attention
            yield return new LayerNormalizationLayer<T>(
                featureSize: hiddenDimension);

            // Feed-forward network
            yield return new DenseLayer<T>(
                inputSize: hiddenDimension,
                outputSize: intermediateDimension,
                activationFunction: new GELUActivation<T>());

            yield return new DenseLayer<T>(
                inputSize: intermediateDimension,
                outputSize: hiddenDimension,
                activationFunction: null);

            // Dropout in feed-forward
            yield return new DropoutLayer<T>(dropoutRate: dropoutRate);

            // Add & Norm after feed-forward
            yield return new LayerNormalizationLayer<T>(
                featureSize: hiddenDimension);
        }

        // === Pooler ===
        // Takes [CLS] token representation for classification
        yield return new DenseLayer<T>(
            inputSize: hiddenDimension,
            outputSize: hiddenDimension,
            activationFunction: new TanhActivation<T>());

        // === Classification Head ===
        // Final dropout before classification
        yield return new DropoutLayer<T>(dropoutRate: dropoutRate);

        // Classification layer
        yield return new DenseLayer<T>(
            inputSize: hiddenDimension,
            outputSize: numSentimentClasses,
            activationFunction: null);

        // Softmax for class probabilities (applied in model's forward pass)
    }

    /// <summary>
    /// Creates the default layer configuration for SEC-BERT (Securities and Exchange Commission BERT).
    /// </summary>
    /// <param name="architecture">The neural network architecture containing base configuration.</param>
    /// <param name="vocabularySize">Size of the token vocabulary (default: 30522 for BERT-base).</param>
    /// <param name="maxSequenceLength">Maximum sequence length in tokens (default: 512).</param>
    /// <param name="hiddenDimension">Hidden dimension size (default: 768 for BERT-base).</param>
    /// <param name="numAttentionHeads">Number of attention heads (default: 12).</param>
    /// <param name="intermediateDimension">Feed-forward intermediate dimension (default: 3072).</param>
    /// <param name="numLayers">Number of transformer layers (default: 12).</param>
    /// <param name="numClasses">Number of output classes (default: 2 for binary classification).</param>
    /// <param name="dropoutRate">Dropout rate for regularization (default: 0.1).</param>
    /// <param name="taskType">Task type: classification, ner, or qa (default: classification).</param>
    /// <returns>An enumerable of layers comprising the SEC-BERT architecture.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> SEC-BERT is a BERT model specialized for understanding SEC filings,
    /// including 10-K annual reports, 10-Q quarterly reports, 8-K current reports, and other
    /// regulatory documents. The model understands formal legal and accounting language used
    /// in securities filings.
    /// </para>
    /// <para>
    /// SEC-BERT differs from general FinBERT in several ways:
    /// - Pre-trained on millions of SEC filing documents
    /// - Understands regulatory terminology like "material adverse effect", "going concern"
    /// - Can detect subtle changes in disclosure language over time
    /// - Supports multiple tasks: classification, named entity recognition (NER), and question answering (QA)
    /// </para>
    /// <para>
    /// The architecture follows BERT-base with:
    /// 1. Token and position embeddings
    /// 2. 12 transformer encoder layers with self-attention
    /// 3. Task-specific output head (classification, NER, or QA)
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultSECBERTLayers(
        NeuralNetworkArchitecture<T> architecture,
        int vocabularySize = 30522,
        int maxSequenceLength = 512,
        int hiddenDimension = 768,
        int numAttentionHeads = 12,
        int intermediateDimension = 3072,
        int numLayers = 12,
        int numClasses = 2,
        double dropoutRate = 0.1,
        string taskType = "classification")
    {
        // === Token Embedding Layer ===
        // Maps token IDs to dense vectors - trained on SEC filing vocabulary
        yield return new EmbeddingLayer<T>(
            vocabularySize: vocabularySize,
            embeddingDimension: hiddenDimension);

        // === Position Embedding Layer ===
        // Adds position information to understand document structure
        yield return new EmbeddingLayer<T>(
            vocabularySize: maxSequenceLength,
            embeddingDimension: hiddenDimension);

        // === Layer Normalization ===
        // Normalizes embeddings before transformer layers
        yield return new LayerNormalizationLayer<T>(
            featureSize: hiddenDimension);

        // === Dropout ===
        yield return new DropoutLayer<T>(dropoutRate: dropoutRate);

        // === Transformer Layers ===
        // Stack of transformer encoder blocks for understanding SEC filing language
        for (int i = 0; i < numLayers; i++)
        {
            // Multi-head self-attention - captures relationships across filing sections
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: maxSequenceLength,
                embeddingDimension: hiddenDimension,
                headCount: numAttentionHeads);

            // Add & Norm after attention
            yield return new LayerNormalizationLayer<T>(
                featureSize: hiddenDimension);

            // Feed-forward network for regulatory language understanding
            yield return new DenseLayer<T>(
                inputSize: hiddenDimension,
                outputSize: intermediateDimension,
                activationFunction: new GELUActivation<T>());

            yield return new DenseLayer<T>(
                inputSize: intermediateDimension,
                outputSize: hiddenDimension,
                activationFunction: null);

            // Dropout in feed-forward
            yield return new DropoutLayer<T>(dropoutRate: dropoutRate);

            // Add & Norm after feed-forward
            yield return new LayerNormalizationLayer<T>(
                featureSize: hiddenDimension);
        }

        // === Task-Specific Output Head ===
        // Different heads based on task type
        if (taskType == "classification")
        {
            // Classification: [CLS] token pooling then classification
            yield return new DenseLayer<T>(
                inputSize: hiddenDimension,
                outputSize: hiddenDimension,
                activationFunction: new TanhActivation<T>());

            yield return new DropoutLayer<T>(dropoutRate: dropoutRate);

            yield return new DenseLayer<T>(
                inputSize: hiddenDimension,
                outputSize: numClasses,
                activationFunction: null);
        }
        else if (taskType == "ner")
        {
            // NER: Token-level classification for each position
            yield return new DropoutLayer<T>(dropoutRate: dropoutRate);

            // Output for each token position
            yield return new DenseLayer<T>(
                inputSize: hiddenDimension,
                outputSize: numClasses,  // Entity types (B-ORG, I-ORG, B-MONEY, etc.)
                activationFunction: null);
        }
        else if (taskType == "qa")
        {
            // QA: Start and end position prediction
            yield return new DropoutLayer<T>(dropoutRate: dropoutRate);

            // Combined start/end position prediction
            yield return new DenseLayer<T>(
                inputSize: hiddenDimension,
                outputSize: 2,  // Start and end logits
                activationFunction: null);
        }
        else
        {
            // Default to classification
            yield return new DenseLayer<T>(
                inputSize: hiddenDimension,
                outputSize: hiddenDimension,
                activationFunction: new TanhActivation<T>());

            yield return new DropoutLayer<T>(dropoutRate: dropoutRate);

            yield return new DenseLayer<T>(
                inputSize: hiddenDimension,
                outputSize: numClasses,
                activationFunction: null);
        }
    }

    /// <summary>
    /// Creates the default layer configuration for FinancialBERT (domain-adapted financial BERT).
    /// </summary>
    /// <param name="architecture">The neural network architecture containing base configuration.</param>
    /// <param name="vocabularySize">Size of the token vocabulary (default: 30522 for BERT-base).</param>
    /// <param name="maxSequenceLength">Maximum sequence length in tokens (default: 512).</param>
    /// <param name="hiddenDimension">Hidden dimension size (default: 768 for BERT-base).</param>
    /// <param name="numAttentionHeads">Number of attention heads (default: 12).</param>
    /// <param name="intermediateDimension">Feed-forward intermediate dimension (default: 3072).</param>
    /// <param name="numLayers">Number of transformer layers (default: 12).</param>
    /// <param name="numClasses">Number of output classes (default: 3 for sentiment).</param>
    /// <param name="dropoutRate">Dropout rate for regularization (default: 0.1).</param>
    /// <param name="taskType">Task type: sentiment, topic, entity, multi (default: sentiment).</param>
    /// <returns>An enumerable of layers comprising the FinancialBERT architecture.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> FinancialBERT is a domain-adapted BERT for comprehensive financial
    /// text analysis. It covers broader financial applications than specialized models like
    /// FinBERT (sentiment-focused) or SEC-BERT (regulatory-focused).
    /// </para>
    /// <para>
    /// FinancialBERT supports multiple task types:
    /// - sentiment: Positive/negative/neutral classification
    /// - topic: Categorize by financial topic (M&amp;A, earnings, guidance, etc.)
    /// - entity: Named entity recognition
    /// - multi: Multi-task learning with multiple output heads
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultFinancialBERTLayers(
        NeuralNetworkArchitecture<T> architecture,
        int vocabularySize = 30522,
        int maxSequenceLength = 512,
        int hiddenDimension = 768,
        int numAttentionHeads = 12,
        int intermediateDimension = 3072,
        int numLayers = 12,
        int numClasses = 3,
        double dropoutRate = 0.1,
        string taskType = "sentiment")
    {
        // === Token Embedding Layer ===
        yield return new EmbeddingLayer<T>(
            vocabularySize: vocabularySize,
            embeddingDimension: hiddenDimension);

        // === Position Embedding Layer ===
        yield return new EmbeddingLayer<T>(
            vocabularySize: maxSequenceLength,
            embeddingDimension: hiddenDimension);

        // === Layer Normalization ===
        yield return new LayerNormalizationLayer<T>(
            featureSize: hiddenDimension);

        // === Dropout ===
        yield return new DropoutLayer<T>(dropoutRate: dropoutRate);

        // === Transformer Layers ===
        for (int i = 0; i < numLayers; i++)
        {
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: maxSequenceLength,
                embeddingDimension: hiddenDimension,
                headCount: numAttentionHeads);

            yield return new LayerNormalizationLayer<T>(
                featureSize: hiddenDimension);

            yield return new DenseLayer<T>(
                inputSize: hiddenDimension,
                outputSize: intermediateDimension,
                activationFunction: new GELUActivation<T>());

            yield return new DenseLayer<T>(
                inputSize: intermediateDimension,
                outputSize: hiddenDimension,
                activationFunction: null);

            yield return new DropoutLayer<T>(dropoutRate: dropoutRate);

            yield return new LayerNormalizationLayer<T>(
                featureSize: hiddenDimension);
        }

        // === Task-Specific Output Head ===
        // Pooler
        yield return new DenseLayer<T>(
            inputSize: hiddenDimension,
            outputSize: hiddenDimension,
            activationFunction: new TanhActivation<T>());

        yield return new DropoutLayer<T>(dropoutRate: dropoutRate);

        // Classification head
        yield return new DenseLayer<T>(
            inputSize: hiddenDimension,
            outputSize: numClasses,
            activationFunction: null);
    }

    /// <summary>
    /// Creates the default layer configuration for FinBERT-Tone (sentiment-focused FinBERT).
    /// </summary>
    /// <param name="architecture">The neural network architecture containing base configuration.</param>
    /// <param name="vocabularySize">Size of the token vocabulary (default: 30522 for BERT-base).</param>
    /// <param name="maxSequenceLength">Maximum sequence length in tokens (default: 512).</param>
    /// <param name="hiddenDimension">Hidden dimension size (default: 768 for BERT-base).</param>
    /// <param name="numAttentionHeads">Number of attention heads (default: 12).</param>
    /// <param name="intermediateDimension">Feed-forward intermediate dimension (default: 3072).</param>
    /// <param name="numLayers">Number of transformer layers (default: 12).</param>
    /// <param name="numToneClasses">Number of tone classes (default: 5 for fine-grained sentiment).</param>
    /// <param name="dropoutRate">Dropout rate for regularization (default: 0.1).</param>
    /// <returns>An enumerable of layers comprising the FinBERT-Tone architecture.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> FinBERT-Tone is a variant of FinBERT specifically focused on
    /// capturing fine-grained sentiment and tone in financial text. It uses 5 classes
    /// to capture more nuanced sentiment (very negative, negative, neutral, positive, very positive).
    /// </para>
    /// <para>
    /// FinBERT-Tone is particularly useful for:
    /// - Earnings call tone analysis
    /// - Management communication sentiment
    /// - Forward-looking statement sentiment
    /// - Analyst report tone classification
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultFinBERTToneLayers(
        NeuralNetworkArchitecture<T> architecture,
        int vocabularySize = 30522,
        int maxSequenceLength = 512,
        int hiddenDimension = 768,
        int numAttentionHeads = 12,
        int intermediateDimension = 3072,
        int numLayers = 12,
        int numToneClasses = 5,
        double dropoutRate = 0.1)
    {
        // === Token Embedding Layer ===
        yield return new EmbeddingLayer<T>(
            vocabularySize: vocabularySize,
            embeddingDimension: hiddenDimension);

        // === Position Embedding Layer ===
        yield return new EmbeddingLayer<T>(
            vocabularySize: maxSequenceLength,
            embeddingDimension: hiddenDimension);

        // === Layer Normalization ===
        yield return new LayerNormalizationLayer<T>(
            featureSize: hiddenDimension);

        // === Dropout ===
        yield return new DropoutLayer<T>(dropoutRate: dropoutRate);

        // === Transformer Layers ===
        for (int i = 0; i < numLayers; i++)
        {
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: maxSequenceLength,
                embeddingDimension: hiddenDimension,
                headCount: numAttentionHeads);

            yield return new LayerNormalizationLayer<T>(
                featureSize: hiddenDimension);

            yield return new DenseLayer<T>(
                inputSize: hiddenDimension,
                outputSize: intermediateDimension,
                activationFunction: new GELUActivation<T>());

            yield return new DenseLayer<T>(
                inputSize: intermediateDimension,
                outputSize: hiddenDimension,
                activationFunction: null);

            yield return new DropoutLayer<T>(dropoutRate: dropoutRate);

            yield return new LayerNormalizationLayer<T>(
                featureSize: hiddenDimension);
        }

        // === Tone Classification Head ===
        // Pooler with tanh
        yield return new DenseLayer<T>(
            inputSize: hiddenDimension,
            outputSize: hiddenDimension,
            activationFunction: new TanhActivation<T>());

        yield return new DropoutLayer<T>(dropoutRate: dropoutRate);

        // 5-class tone classifier
        yield return new DenseLayer<T>(
            inputSize: hiddenDimension,
            outputSize: numToneClasses,
            activationFunction: null);
    }

    /// <summary>
    /// Creates the default layer configuration for FinGPT (Financial GPT).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> FinGPT uses a GPT-style decoder architecture for financial NLP.
    /// Unlike BERT (encoder-only), GPT uses causal attention where each token can only
    /// attend to previous tokens, making it suitable for generation tasks.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultFinGPTLayers(
        NeuralNetworkArchitecture<T> architecture,
        int vocabularySize = 50257,
        int maxSequenceLength = 2048,
        int hiddenDimension = 768,
        int numAttentionHeads = 12,
        int intermediateDimension = 3072,
        int numLayers = 12,
        int numClasses = 3,
        double dropoutRate = 0.1)
    {
        // Token embeddings
        yield return new EmbeddingLayer<T>(vocabularySize, hiddenDimension);
        yield return new EmbeddingLayer<T>(maxSequenceLength, hiddenDimension);

        yield return new DropoutLayer<T>(dropoutRate: dropoutRate);

        // GPT transformer blocks
        for (int i = 0; i < numLayers; i++)
        {
            yield return new LayerNormalizationLayer<T>(featureSize: hiddenDimension);
            yield return new MultiHeadAttentionLayer<T>(maxSequenceLength, hiddenDimension, numAttentionHeads);
            yield return new LayerNormalizationLayer<T>(featureSize: hiddenDimension);
            yield return new DenseLayer<T>(hiddenDimension, intermediateDimension, (IActivationFunction<T>)new GELUActivation<T>());
            yield return new DenseLayer<T>(intermediateDimension, hiddenDimension, (IActivationFunction<T>?)null);
            yield return new DropoutLayer<T>(dropoutRate: dropoutRate);
        }

        yield return new LayerNormalizationLayer<T>(featureSize: hiddenDimension);

        // Classification head
        yield return new DenseLayer<T>(hiddenDimension, numClasses, (IActivationFunction<T>?)null);
    }

    /// <summary>
    /// Creates the default layer configuration for BloombergGPT-style model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> BloombergGPT is a large financial LLM. This creates a smaller
    /// version suitable for fine-tuning on specific tasks. The architecture follows
    /// standard decoder-only transformer design with financial domain adaptation.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultBloombergGPTLayers(
        NeuralNetworkArchitecture<T> architecture,
        int vocabularySize = 50257,
        int maxSequenceLength = 2048,
        int hiddenDimension = 1024,
        int numAttentionHeads = 16,
        int intermediateDimension = 4096,
        int numLayers = 24,
        int numClasses = 3,
        double dropoutRate = 0.1)
    {
        // Token and position embeddings
        yield return new EmbeddingLayer<T>(vocabularySize, hiddenDimension);
        yield return new EmbeddingLayer<T>(maxSequenceLength, hiddenDimension);

        yield return new DropoutLayer<T>(dropoutRate: dropoutRate);

        // Transformer blocks
        for (int i = 0; i < numLayers; i++)
        {
            yield return new LayerNormalizationLayer<T>(featureSize: hiddenDimension);
            yield return new MultiHeadAttentionLayer<T>(maxSequenceLength, hiddenDimension, numAttentionHeads);
            yield return new LayerNormalizationLayer<T>(featureSize: hiddenDimension);
            yield return new DenseLayer<T>(hiddenDimension, intermediateDimension, (IActivationFunction<T>)new GELUActivation<T>());
            yield return new DenseLayer<T>(intermediateDimension, hiddenDimension, (IActivationFunction<T>?)null);
            yield return new DropoutLayer<T>(dropoutRate: dropoutRate);
        }

        yield return new LayerNormalizationLayer<T>(featureSize: hiddenDimension);

        // Classification head
        yield return new DenseLayer<T>(hiddenDimension, numClasses, (IActivationFunction<T>?)null);
    }

    /// <summary>
    /// Creates the default layer configuration for FinMA (Financial Multi-Agent).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> FinMA uses a transformer backbone with multiple output heads
    /// for different agent tasks. The base architecture processes input, and specialized
    /// heads handle different aspects of financial analysis.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultFinMALayers(
        NeuralNetworkArchitecture<T> architecture,
        int vocabularySize = 32000,
        int maxSequenceLength = 2048,
        int hiddenDimension = 768,
        int numAttentionHeads = 12,
        int intermediateDimension = 3072,
        int numLayers = 12,
        int numClasses = 3,
        double dropoutRate = 0.1)
    {
        // Token and position embeddings
        yield return new EmbeddingLayer<T>(vocabularySize, hiddenDimension);
        yield return new EmbeddingLayer<T>(maxSequenceLength, hiddenDimension);

        yield return new LayerNormalizationLayer<T>(featureSize: hiddenDimension);
        yield return new DropoutLayer<T>(dropoutRate: dropoutRate);

        // Shared transformer backbone
        for (int i = 0; i < numLayers; i++)
        {
            yield return new MultiHeadAttentionLayer<T>(maxSequenceLength, hiddenDimension, numAttentionHeads);
            yield return new LayerNormalizationLayer<T>(featureSize: hiddenDimension);
            yield return new DenseLayer<T>(hiddenDimension, intermediateDimension, (IActivationFunction<T>)new GELUActivation<T>());
            yield return new DenseLayer<T>(intermediateDimension, hiddenDimension, (IActivationFunction<T>?)null);
            yield return new DropoutLayer<T>(dropoutRate: dropoutRate);
            yield return new LayerNormalizationLayer<T>(featureSize: hiddenDimension);
        }

        // Multi-agent classification head (primary output)
        yield return new DenseLayer<T>(hiddenDimension, hiddenDimension, (IActivationFunction<T>)new TanhActivation<T>());
        yield return new DropoutLayer<T>(dropoutRate: dropoutRate);
        yield return new DenseLayer<T>(hiddenDimension, numClasses, (IActivationFunction<T>?)null);
    }

    /// <summary>
    /// Creates the default layer configuration for InvestLM (Investment Language Model).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> InvestLM is designed for investment-related NLP tasks.
    /// The architecture is similar to other financial LLMs but with focus on
    /// investment terminology and reasoning.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultInvestLMLayers(
        NeuralNetworkArchitecture<T> architecture,
        int vocabularySize = 32000,
        int maxSequenceLength = 2048,
        int hiddenDimension = 768,
        int numAttentionHeads = 12,
        int intermediateDimension = 3072,
        int numLayers = 12,
        int numClasses = 3,
        double dropoutRate = 0.1)
    {
        // Token and position embeddings
        yield return new EmbeddingLayer<T>(vocabularySize, hiddenDimension);
        yield return new EmbeddingLayer<T>(maxSequenceLength, hiddenDimension);

        yield return new LayerNormalizationLayer<T>(featureSize: hiddenDimension);
        yield return new DropoutLayer<T>(dropoutRate: dropoutRate);

        // Transformer blocks
        for (int i = 0; i < numLayers; i++)
        {
            yield return new MultiHeadAttentionLayer<T>(maxSequenceLength, hiddenDimension, numAttentionHeads);
            yield return new LayerNormalizationLayer<T>(featureSize: hiddenDimension);
            yield return new DenseLayer<T>(hiddenDimension, intermediateDimension, (IActivationFunction<T>)new GELUActivation<T>());
            yield return new DenseLayer<T>(intermediateDimension, hiddenDimension, (IActivationFunction<T>?)null);
            yield return new DropoutLayer<T>(dropoutRate: dropoutRate);
            yield return new LayerNormalizationLayer<T>(featureSize: hiddenDimension);
        }

        // Investment recommendation head
        yield return new DenseLayer<T>(hiddenDimension, hiddenDimension, (IActivationFunction<T>)new TanhActivation<T>());
        yield return new DropoutLayer<T>(dropoutRate: dropoutRate);
        yield return new DenseLayer<T>(hiddenDimension, numClasses, (IActivationFunction<T>?)null);
    }

    #endregion

    #region Factor Models

    /// <summary>
    /// Creates default layers for an AlphaFactorModel.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="numFeatures">Number of input features per asset.</param>
    /// <param name="hiddenDimension">Dimension of hidden layers.</param>
    /// <param name="numFactors">Number of latent factors to learn.</param>
    /// <param name="numAssets">Number of assets for output.</param>
    /// <param name="dropoutRate">Dropout rate for regularization.</param>
    /// <returns>Enumerable of layers forming the AlphaFactorModel architecture.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates the neural network layers for learning alpha factors:
    /// 1. Feature encoder: Transforms raw features into learned representations
    /// 2. Factor extractor: Learns a set of uncorrelated factors
    /// 3. Alpha predictor: Combines factors to predict excess returns
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultAlphaFactorLayers(
        NeuralNetworkArchitecture<T> architecture,
        int numFeatures = 50,
        int hiddenDimension = 128,
        int numFactors = 10,
        int numAssets = 500,
        double dropoutRate = 0.1)
    {
        // Feature encoder
        yield return new DenseLayer<T>(numFeatures, hiddenDimension, (IActivationFunction<T>)new ReLUActivation<T>());
        yield return new BatchNormalizationLayer<T>(hiddenDimension);
        yield return new DropoutLayer<T>(dropoutRate: dropoutRate);

        // Hidden layer
        yield return new DenseLayer<T>(hiddenDimension, hiddenDimension, (IActivationFunction<T>)new ReLUActivation<T>());
        yield return new BatchNormalizationLayer<T>(hiddenDimension);
        yield return new DropoutLayer<T>(dropoutRate: dropoutRate);

        // Factor extractor
        yield return new DenseLayer<T>(hiddenDimension, numFactors, (IActivationFunction<T>)new TanhActivation<T>());
        yield return new BatchNormalizationLayer<T>(numFactors);

        // Alpha predictor
        yield return new DenseLayer<T>(numFactors, hiddenDimension, (IActivationFunction<T>)new ReLUActivation<T>());
        yield return new DenseLayer<T>(hiddenDimension, numAssets, (IActivationFunction<T>?)null);
    }

    /// <summary>
    /// Creates default layers for a FactorVAE model.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="hiddenDimension">Dimension of hidden layers.</param>
    /// <param name="latentDimension">Dimension of latent space.</param>
    /// <param name="numFactors">Number of factors to disentangle.</param>
    /// <param name="dropoutRate">Dropout rate for regularization.</param>
    /// <returns>Enumerable of layers forming the FactorVAE architecture.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> FactorVAE uses a variational autoencoder to learn disentangled factors.
    /// The encoder compresses data to a latent space, and the decoder reconstructs it.
    /// The "disentanglement" means each latent dimension captures a different factor.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultFactorVAELayers(
        NeuralNetworkArchitecture<T> architecture,
        int numFeatures = 50,
        int hiddenDimension = 128,
        int latentDimension = 32,
        int numFactors = 10,
        double dropoutRate = 0.1)
    {
        // Encoder
        yield return new DenseLayer<T>(numFeatures, hiddenDimension, (IActivationFunction<T>)new ReLUActivation<T>());
        yield return new BatchNormalizationLayer<T>(hiddenDimension);
        yield return new DropoutLayer<T>(dropoutRate: dropoutRate);

        yield return new DenseLayer<T>(hiddenDimension, hiddenDimension, (IActivationFunction<T>)new ReLUActivation<T>());
        yield return new BatchNormalizationLayer<T>(hiddenDimension);

        // Latent space (mean and log-variance)
        yield return new DenseLayer<T>(hiddenDimension, latentDimension * 2, (IActivationFunction<T>?)null);

        // Factor discriminator for disentanglement
        yield return new DenseLayer<T>(latentDimension, hiddenDimension, (IActivationFunction<T>)new LeakyReLUActivation<T>());
        yield return new DenseLayer<T>(hiddenDimension, numFactors, (IActivationFunction<T>?)null);

        // Decoder
        yield return new DenseLayer<T>(latentDimension, hiddenDimension, (IActivationFunction<T>)new ReLUActivation<T>());
        yield return new BatchNormalizationLayer<T>(hiddenDimension);

        yield return new DenseLayer<T>(hiddenDimension, hiddenDimension, (IActivationFunction<T>)new ReLUActivation<T>());
        yield return new DenseLayer<T>(hiddenDimension, numFeatures, (IActivationFunction<T>?)null);
    }

    /// <summary>
    /// Creates default layers for a FactorTransformer model.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="hiddenDimension">Dimension of hidden layers.</param>
    /// <param name="numFactors">Number of factors.</param>
    /// <param name="headCount">Number of attention heads.</param>
    /// <param name="numLayers">Number of transformer layers.</param>
    /// <param name="sequenceLength">Length of input sequences.</param>
    /// <param name="dropoutRate">Dropout rate for regularization.</param>
    /// <returns>Enumerable of layers forming the FactorTransformer architecture.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> FactorTransformer uses transformer attention to model
    /// relationships between assets and time, extracting factors that capture
    /// cross-sectional and temporal patterns.
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultFactorTransformerLayers(
        NeuralNetworkArchitecture<T> architecture,
        int numFeatures = 50,
        int hiddenDimension = 128,
        int numFactors = 10,
        int headCount = 4,
        int numLayers = 2,
        int sequenceLength = 60,
        double dropoutRate = 0.1)
    {
        // Input embedding
        yield return new DenseLayer<T>(numFeatures, hiddenDimension, (IActivationFunction<T>)new ReLUActivation<T>());

        // Positional encoding is handled inside transformer, add layer norm
        yield return new LayerNormalizationLayer<T>(hiddenDimension);

        // Transformer encoder layers
        for (int i = 0; i < numLayers; i++)
        {
            // Multi-head self-attention
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: sequenceLength,
                embeddingDimension: hiddenDimension,
                headCount: headCount);
            yield return new LayerNormalizationLayer<T>(hiddenDimension);

            // Feed-forward network
            yield return new DenseLayer<T>(hiddenDimension, hiddenDimension * 4, (IActivationFunction<T>)new GELUActivation<T>());
            yield return new DropoutLayer<T>(dropoutRate: dropoutRate);
            yield return new DenseLayer<T>(hiddenDimension * 4, hiddenDimension, (IActivationFunction<T>?)null);
            yield return new LayerNormalizationLayer<T>(hiddenDimension);
        }

        // Factor extraction head
        yield return new DenseLayer<T>(hiddenDimension, hiddenDimension, (IActivationFunction<T>)new ReLUActivation<T>());
        yield return new DenseLayer<T>(hiddenDimension, numFactors, (IActivationFunction<T>)new TanhActivation<T>());

        // Alpha prediction head
        yield return new DenseLayer<T>(numFactors, hiddenDimension, (IActivationFunction<T>)new ReLUActivation<T>());
        yield return new DenseLayer<T>(hiddenDimension, 1, (IActivationFunction<T>?)null);
    }

    #endregion

    #region Risk Models

    /// <summary>
    /// Creates default layers for a NeuralVaR model.
    /// </summary>
    public static IEnumerable<ILayer<T>> CreateDefaultNeuralVaRLayers(
        NeuralNetworkArchitecture<T> architecture,
        int numFeatures = 50,
        int hiddenDimension = 128,
        int sequenceLength = 60,
        double dropoutRate = 0.1)
    {
        // LSTM for temporal patterns
        yield return new LSTMLayer<T>(
            inputSize: numFeatures,
            hiddenSize: hiddenDimension,
            inputShape: new[] { 1, sequenceLength, numFeatures },
            activation: (IActivationFunction<T>?)null,
            recurrentActivation: null,
            engine: null);

        // Dense layers for risk estimation
        yield return new DenseLayer<T>(hiddenDimension, hiddenDimension, (IActivationFunction<T>)new ReLUActivation<T>());
        yield return new BatchNormalizationLayer<T>(hiddenDimension);
        yield return new DropoutLayer<T>(dropoutRate: dropoutRate);

        yield return new DenseLayer<T>(hiddenDimension, hiddenDimension / 2, (IActivationFunction<T>)new ReLUActivation<T>());
        yield return new BatchNormalizationLayer<T>(hiddenDimension / 2);

        // Output layer for VaR estimate
        yield return new DenseLayer<T>(hiddenDimension / 2, 1, (IActivationFunction<T>?)null);
    }

    /// <summary>
    /// Creates default layers for a NeuralCVaR model.
    /// </summary>
    public static IEnumerable<ILayer<T>> CreateDefaultNeuralCVaRLayers(
        NeuralNetworkArchitecture<T> architecture,
        int numFeatures = 50,
        int hiddenDimension = 128,
        int sequenceLength = 60,
        double dropoutRate = 0.1)
    {
        // LSTM for temporal patterns
        yield return new LSTMLayer<T>(
            inputSize: numFeatures,
            hiddenSize: hiddenDimension,
            inputShape: new[] { 1, sequenceLength, numFeatures },
            activation: (IActivationFunction<T>?)null,
            recurrentActivation: null,
            engine: null);

        // Dense layers
        yield return new DenseLayer<T>(hiddenDimension, hiddenDimension, (IActivationFunction<T>)new ReLUActivation<T>());
        yield return new BatchNormalizationLayer<T>(hiddenDimension);
        yield return new DropoutLayer<T>(dropoutRate: dropoutRate);

        yield return new DenseLayer<T>(hiddenDimension, hiddenDimension / 2, (IActivationFunction<T>)new ReLUActivation<T>());

        // Output both VaR and CVaR
        yield return new DenseLayer<T>(hiddenDimension / 2, 2, (IActivationFunction<T>?)null);
    }

    /// <summary>
    /// Creates default layers for a TabNet model.
    /// </summary>
    public static IEnumerable<ILayer<T>> CreateDefaultTabNetLayers(
        NeuralNetworkArchitecture<T> architecture,
        int numFeatures = 50,
        int hiddenDimension = 128,
        int numSteps = 3,
        int numClasses = 2,
        double dropoutRate = 0.1)
    {
        // Feature transformer (shared)
        yield return new DenseLayer<T>(numFeatures, hiddenDimension, (IActivationFunction<T>)new ReLUActivation<T>());
        yield return new BatchNormalizationLayer<T>(hiddenDimension);

        // Decision steps
        for (int i = 0; i < numSteps; i++)
        {
            yield return new DenseLayer<T>(hiddenDimension, hiddenDimension, (IActivationFunction<T>)new ReLUActivation<T>());
            yield return new BatchNormalizationLayer<T>(hiddenDimension);
            yield return new DropoutLayer<T>(dropoutRate: dropoutRate);
        }

        // Output layer
        yield return new DenseLayer<T>(hiddenDimension, numClasses, (IActivationFunction<T>)new SoftmaxActivation<T>());
    }

    /// <summary>
    /// Creates default layers for a SAINT model.
    /// </summary>
    public static IEnumerable<ILayer<T>> CreateDefaultSAINTLayers(
        NeuralNetworkArchitecture<T> architecture,
        int numFeatures = 50,
        int hiddenDimension = 128,
        int numHeads = 4,
        int numLayers = 2,
        int sequenceLength = 1,
        int numClasses = 2,
        double dropoutRate = 0.1)
    {
        // Input embedding
        yield return new DenseLayer<T>(numFeatures, hiddenDimension, (IActivationFunction<T>)new ReLUActivation<T>());
        yield return new LayerNormalizationLayer<T>(hiddenDimension);

        // Transformer layers with inter-sample attention
        for (int i = 0; i < numLayers; i++)
        {
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: sequenceLength,
                embeddingDimension: hiddenDimension,
                headCount: numHeads);
            yield return new LayerNormalizationLayer<T>(hiddenDimension);

            yield return new DenseLayer<T>(hiddenDimension, hiddenDimension * 4, (IActivationFunction<T>)new GELUActivation<T>());
            yield return new DropoutLayer<T>(dropoutRate: dropoutRate);
            yield return new DenseLayer<T>(hiddenDimension * 4, hiddenDimension, (IActivationFunction<T>?)null);
            yield return new LayerNormalizationLayer<T>(hiddenDimension);
        }

        // Classification head
        yield return new DenseLayer<T>(hiddenDimension, hiddenDimension / 2, (IActivationFunction<T>)new ReLUActivation<T>());
        yield return new DenseLayer<T>(hiddenDimension / 2, numClasses, (IActivationFunction<T>)new SoftmaxActivation<T>());
    }

    /// <summary>
    /// Creates default layers for a TabTransformer model.
    /// </summary>
    public static IEnumerable<ILayer<T>> CreateDefaultTabTransformerLayers(
        NeuralNetworkArchitecture<T> architecture,
        int numFeatures = 50,
        int hiddenDimension = 128,
        int numHeads = 4,
        int numLayers = 2,
        int sequenceLength = 1,
        int numClasses = 2,
        double dropoutRate = 0.1)
    {
        // Column embedding
        yield return new DenseLayer<T>(numFeatures, hiddenDimension, (IActivationFunction<T>)new ReLUActivation<T>());
        yield return new LayerNormalizationLayer<T>(hiddenDimension);

        // Transformer encoder for categorical columns
        for (int i = 0; i < numLayers; i++)
        {
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: sequenceLength,
                embeddingDimension: hiddenDimension,
                headCount: numHeads);
            yield return new LayerNormalizationLayer<T>(hiddenDimension);

            yield return new DenseLayer<T>(hiddenDimension, hiddenDimension * 4, (IActivationFunction<T>)new GELUActivation<T>());
            yield return new DropoutLayer<T>(dropoutRate: dropoutRate);
            yield return new DenseLayer<T>(hiddenDimension * 4, hiddenDimension, (IActivationFunction<T>?)null);
            yield return new LayerNormalizationLayer<T>(hiddenDimension);
        }

        // MLP head
        yield return new DenseLayer<T>(hiddenDimension, hiddenDimension, (IActivationFunction<T>)new ReLUActivation<T>());
        yield return new DropoutLayer<T>(dropoutRate: dropoutRate);
        yield return new DenseLayer<T>(hiddenDimension, numClasses, (IActivationFunction<T>)new SoftmaxActivation<T>());
    }

    /// <summary>
    /// Creates default layers for a GANDALF (Gated Additive Neural Decision Forest) model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="outputSize">Number of output dimensions.</param>
    /// <param name="numTrees">Number of soft decision trees in the ensemble (default: 20).</param>
    /// <param name="treeDepth">Depth of each decision tree (default: 6).</param>
    /// <param name="gatingHiddenDim">Hidden dimension for gating network (default: 128).</param>
    /// <param name="numGatingLayers">Number of gating network hidden layers (default: 2).</param>
    /// <param name="leafDimension">Output dimension per leaf node (default: 1).</param>
    /// <param name="temperature">Temperature for soft tree decisions (default: 1.0).</param>
    /// <param name="initScale">Initialization scale for tree parameters (default: 0.01).</param>
    /// <param name="useBatchNorm">Whether to use batch normalization (default: true).</param>
    /// <param name="dropoutRate">Dropout rate for regularization (default: 0.1).</param>
    /// <returns>A collection of layers forming the GANDALF network.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> GANDALF combines learned feature gating with soft decision trees:
    ///
    /// 1. **Gating Network**: Learns which features are important for each prediction
    ///    - Multiple fully connected layers with ReLU activation
    ///    - Final sigmoid layer produces [0,1] importance weights per feature
    ///
    /// 2. **Soft Decision Tree Ensemble**: Trees with differentiable splits
    ///    - Each tree has learnable split weights and leaf values
    ///    - "Soft" decisions allow partial paths (differentiable for backprop)
    ///
    /// 3. **Output Projection**: Aggregates tree outputs to final prediction
    /// </para>
    /// <para>
    /// Reference: "GANDALF: Gated Adaptive Network for Deep Automated Learning of Features" (2022)
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultGANDALFLayers(
        NeuralNetworkArchitecture<T> architecture,
        int numFeatures = 50,
        int outputSize = 1,
        int numTrees = 20,
        int treeDepth = 6,
        int gatingHiddenDim = 128,
        int numGatingLayers = 2,
        int leafDimension = 1,
        double temperature = 1.0,
        double initScale = 0.01,
        bool useBatchNorm = true,
        double dropoutRate = 0.1)
    {
        // ============================================
        // 1. GATING NETWORK - learns feature importance
        // ============================================
        int prevDim = numFeatures;

        // Gating hidden layers with ReLU activation
        for (int i = 0; i < numGatingLayers; i++)
        {
            yield return new DenseLayer<T>(prevDim, gatingHiddenDim, (IActivationFunction<T>)new ReLUActivation<T>());

            if (useBatchNorm)
            {
                yield return new BatchNormalizationLayer<T>(gatingHiddenDim);
            }

            if (dropoutRate > 0)
            {
                yield return new DropoutLayer<T>(dropoutRate: dropoutRate);
            }

            prevDim = gatingHiddenDim;
        }

        // Gating output layer - sigmoid produces [0,1] feature importance weights
        yield return new DenseLayer<T>(prevDim, numFeatures, (IActivationFunction<T>)new SigmoidActivation<T>());

        // ============================================
        // 2. SOFT DECISION TREE ENSEMBLE
        // ============================================
        // Each tree processes the gated features independently
        for (int t = 0; t < numTrees; t++)
        {
            yield return new SoftTreeLayer<T>(
                inputDim: numFeatures,
                depth: treeDepth,
                outputDim: leafDimension,
                temperature: temperature,
                initScale: initScale);
        }

        // ============================================
        // 3. OUTPUT PROJECTION
        // ============================================
        // Aggregate tree outputs to final prediction dimension
        int treeTotalOutputDim = numTrees * leafDimension;

        if (treeTotalOutputDim != outputSize)
        {
            yield return new DenseLayer<T>(treeTotalOutputDim, outputSize, (IActivationFunction<T>?)null);
        }
    }

    /// <summary>
    /// Creates default layers for a NODE (Neural Oblivious Decision Ensembles) model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="numTrees">Number of oblivious decision trees.</param>
    /// <param name="treeDepth">Depth of each tree.</param>
    /// <param name="treeOutputDim">Output dimension per tree.</param>
    /// <param name="outputSize">Final output size.</param>
    /// <param name="useBatchNorm">Whether to use batch normalization.</param>
    /// <param name="dropoutRate">Dropout rate for regularization.</param>
    /// <returns>A collection of layers forming the NODE network.</returns>
    public static IEnumerable<ILayer<T>> CreateDefaultNODELayers(
        NeuralNetworkArchitecture<T> architecture,
        int numFeatures = 50,
        int numTrees = 20,
        int treeDepth = 6,
        int treeOutputDim = 3,
        int outputSize = 1,
        bool useBatchNorm = true,
        double dropoutRate = 0.0)
    {
        // Optional input batch normalization
        if (useBatchNorm)
        {
            yield return new BatchNormalizationLayer<T>(numFeatures);
        }

        // Feature preprocessing layer
        yield return new DenseLayer<T>(numFeatures, numFeatures * 2, (IActivationFunction<T>)new ReLUActivation<T>());

        if (useBatchNorm)
        {
            yield return new BatchNormalizationLayer<T>(numFeatures * 2);
        }

        // Soft tree ensemble (using SoftTreeLayer)
        for (int t = 0; t < numTrees; t++)
        {
            yield return new SoftTreeLayer<T>(
                inputDim: numFeatures * 2,
                depth: treeDepth,
                outputDim: treeOutputDim,
                temperature: 1.0,
                initScale: 0.01);
        }

        // Aggregate tree outputs
        int totalTreeOutput = numTrees * treeOutputDim;
        yield return new DenseLayer<T>(totalTreeOutput, outputSize, (IActivationFunction<T>?)null);
    }

    /// <summary>
    /// Creates default layers for an AutoInt (Automatic Feature Interaction) model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="embeddingDimension">Embedding dimension for features.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="numLayers">Number of interacting layers.</param>
    /// <param name="numClasses">Number of output classes.</param>
    /// <param name="dropoutRate">Dropout rate for regularization.</param>
    /// <returns>A collection of layers forming the AutoInt network.</returns>
    public static IEnumerable<ILayer<T>> CreateDefaultAutoIntLayers(
        NeuralNetworkArchitecture<T> architecture,
        int numFeatures = 50,
        int embeddingDimension = 16,
        int numHeads = 2,
        int numLayers = 3,
        int numClasses = 2,
        double dropoutRate = 0.0)
    {
        // Feature embedding layer
        yield return new DenseLayer<T>(numFeatures, numFeatures * embeddingDimension, (IActivationFunction<T>)new ReLUActivation<T>());
        yield return new LayerNormalizationLayer<T>(numFeatures * embeddingDimension);

        // Multi-head self-attention layers for feature interactions
        for (int i = 0; i < numLayers; i++)
        {
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: numFeatures,
                embeddingDimension: embeddingDimension,
                headCount: numHeads);
            yield return new LayerNormalizationLayer<T>(numFeatures * embeddingDimension);

            if (dropoutRate > 0)
            {
                yield return new DropoutLayer<T>(dropoutRate: dropoutRate);
            }
        }

        // MLP head for final prediction
        yield return new DenseLayer<T>(numFeatures * embeddingDimension, 64, (IActivationFunction<T>)new ReLUActivation<T>());
        yield return new DenseLayer<T>(64, 32, (IActivationFunction<T>)new ReLUActivation<T>());
        yield return new DenseLayer<T>(32, numClasses, (IActivationFunction<T>)new SoftmaxActivation<T>());
    }

    /// <summary>
    /// Creates default layers for a Mambular (State Space Model for Tabular) model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="embeddingDimension">Embedding dimension.</param>
    /// <param name="stateDimension">State dimension for SSM.</param>
    /// <param name="numLayers">Number of Mamba layers.</param>
    /// <param name="numClasses">Number of output classes.</param>
    /// <param name="dropoutRate">Dropout rate for regularization.</param>
    /// <returns>A collection of layers forming the Mambular network.</returns>
    public static IEnumerable<ILayer<T>> CreateDefaultMambularLayers(
        NeuralNetworkArchitecture<T> architecture,
        int numFeatures = 50,
        int embeddingDimension = 32,
        int stateDimension = 16,
        int numLayers = 4,
        int numClasses = 2,
        double dropoutRate = 0.1)
    {
        // Feature embedding
        yield return new DenseLayer<T>(numFeatures, embeddingDimension, (IActivationFunction<T>)new ReLUActivation<T>());
        yield return new LayerNormalizationLayer<T>(embeddingDimension);

        // Mamba-style layers (approximated with dense + gating)
        for (int i = 0; i < numLayers; i++)
        {
            // Expand dimension
            yield return new DenseLayer<T>(embeddingDimension, embeddingDimension * 2, (IActivationFunction<T>)new SiLUActivation<T>());

            // State space processing (simplified)
            yield return new DenseLayer<T>(embeddingDimension * 2, embeddingDimension * 2, (IActivationFunction<T>?)null);

            // Contract back
            yield return new DenseLayer<T>(embeddingDimension * 2, embeddingDimension, (IActivationFunction<T>?)null);
            yield return new LayerNormalizationLayer<T>(embeddingDimension);

            if (dropoutRate > 0)
            {
                yield return new DropoutLayer<T>(dropoutRate: dropoutRate);
            }
        }

        // MLP head
        yield return new DenseLayer<T>(embeddingDimension, 64, (IActivationFunction<T>)new ReLUActivation<T>());
        yield return new DenseLayer<T>(64, 32, (IActivationFunction<T>)new ReLUActivation<T>());
        yield return new DenseLayer<T>(32, numClasses, (IActivationFunction<T>)new SoftmaxActivation<T>());
    }

    /// <summary>
    /// Creates default layers for a TabDPT (Tabular Data Pre-Training) model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="embeddingDimension">Embedding dimension.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="numLayers">Number of transformer layers.</param>
    /// <param name="numClasses">Number of output classes.</param>
    /// <param name="dropoutRate">Dropout rate for regularization.</param>
    /// <returns>A collection of layers forming the TabDPT network.</returns>
    public static IEnumerable<ILayer<T>> CreateDefaultTabDPTLayers(
        NeuralNetworkArchitecture<T> architecture,
        int numFeatures = 50,
        int embeddingDimension = 128,
        int numHeads = 4,
        int numLayers = 6,
        int numClasses = 2,
        double dropoutRate = 0.1)
    {
        // Input projection
        yield return new DenseLayer<T>(numFeatures, embeddingDimension, (IActivationFunction<T>)new ReLUActivation<T>());
        yield return new LayerNormalizationLayer<T>(embeddingDimension);

        // Transformer encoder layers
        for (int i = 0; i < numLayers; i++)
        {
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: 1,
                embeddingDimension: embeddingDimension,
                headCount: numHeads);
            yield return new LayerNormalizationLayer<T>(embeddingDimension);

            yield return new DenseLayer<T>(embeddingDimension, embeddingDimension * 4, (IActivationFunction<T>)new GELUActivation<T>());
            yield return new DropoutLayer<T>(dropoutRate: dropoutRate);
            yield return new DenseLayer<T>(embeddingDimension * 4, embeddingDimension, (IActivationFunction<T>?)null);
            yield return new LayerNormalizationLayer<T>(embeddingDimension);
        }

        // Output head
        yield return new DenseLayer<T>(embeddingDimension, 64, (IActivationFunction<T>)new ReLUActivation<T>());
        yield return new DenseLayer<T>(64, numClasses, (IActivationFunction<T>)new SoftmaxActivation<T>());
    }

    /// <summary>
    /// Creates default layers for a TabPFN (Prior-Fitted Network) model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="embeddingDimension">Embedding dimension.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="numLayers">Number of transformer layers.</param>
    /// <param name="numClasses">Number of output classes.</param>
    /// <param name="dropoutRate">Dropout rate for regularization.</param>
    /// <returns>A collection of layers forming the TabPFN network.</returns>
    public static IEnumerable<ILayer<T>> CreateDefaultTabPFNLayers(
        NeuralNetworkArchitecture<T> architecture,
        int numFeatures = 50,
        int embeddingDimension = 128,
        int numHeads = 4,
        int numLayers = 12,
        int numClasses = 2,
        double dropoutRate = 0.0)
    {
        // Input projection
        yield return new DenseLayer<T>(numFeatures, embeddingDimension, (IActivationFunction<T>)new GELUActivation<T>());
        yield return new LayerNormalizationLayer<T>(embeddingDimension);

        // Deep transformer encoder (TabPFN uses many layers)
        for (int i = 0; i < numLayers; i++)
        {
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: 1,
                embeddingDimension: embeddingDimension,
                headCount: numHeads);
            yield return new LayerNormalizationLayer<T>(embeddingDimension);

            yield return new DenseLayer<T>(embeddingDimension, embeddingDimension * 4, (IActivationFunction<T>)new GELUActivation<T>());
            if (dropoutRate > 0)
            {
                yield return new DropoutLayer<T>(dropoutRate: dropoutRate);
            }
            yield return new DenseLayer<T>(embeddingDimension * 4, embeddingDimension, (IActivationFunction<T>?)null);
            yield return new LayerNormalizationLayer<T>(embeddingDimension);
        }

        // Output head
        yield return new DenseLayer<T>(embeddingDimension, 64, (IActivationFunction<T>)new GELUActivation<T>());
        yield return new DenseLayer<T>(64, numClasses, (IActivationFunction<T>)new SoftmaxActivation<T>());
    }

    /// <summary>
    /// Creates default layers for a TabM (Parameter-Efficient Ensemble) model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="hiddenDimensions">Array of hidden layer dimensions.</param>
    /// <param name="numClasses">Number of output classes.</param>
    /// <param name="dropoutRate">Dropout rate for regularization.</param>
    /// <returns>A collection of layers forming the TabM network.</returns>
    public static IEnumerable<ILayer<T>> CreateDefaultTabMLayers(
        NeuralNetworkArchitecture<T> architecture,
        int numFeatures = 50,
        int[]? hiddenDimensions = null,
        int numClasses = 2,
        double dropoutRate = 0.1)
    {
        hiddenDimensions ??= [256, 256];

        int prevDim = numFeatures;

        // Hidden layers
        foreach (int hiddenDim in hiddenDimensions)
        {
            yield return new DenseLayer<T>(prevDim, hiddenDim, (IActivationFunction<T>)new ReLUActivation<T>());
            yield return new LayerNormalizationLayer<T>(hiddenDim);
            if (dropoutRate > 0)
            {
                yield return new DropoutLayer<T>(dropoutRate: dropoutRate);
            }
            prevDim = hiddenDim;
        }

        // Output layer
        yield return new DenseLayer<T>(prevDim, numClasses, (IActivationFunction<T>)new SoftmaxActivation<T>());
    }

    /// <summary>
    /// Creates default layers for an FT-Transformer (Feature Tokenizer + Transformer) model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="embeddingDimension">Embedding dimension for feature tokens.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="numLayers">Number of transformer layers.</param>
    /// <param name="numClasses">Number of output classes.</param>
    /// <param name="dropoutRate">Dropout rate for regularization.</param>
    /// <returns>A collection of layers forming the FT-Transformer network.</returns>
    public static IEnumerable<ILayer<T>> CreateDefaultFTTransformerLayers(
        NeuralNetworkArchitecture<T> architecture,
        int numFeatures = 50,
        int embeddingDimension = 192,
        int numHeads = 8,
        int numLayers = 3,
        int numClasses = 2,
        double dropoutRate = 0.1)
    {
        // Feature tokenization (embedding each feature)
        yield return new DenseLayer<T>(numFeatures, embeddingDimension, (IActivationFunction<T>)new ReLUActivation<T>());
        yield return new LayerNormalizationLayer<T>(embeddingDimension);

        // Transformer encoder layers
        for (int i = 0; i < numLayers; i++)
        {
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: 1,
                embeddingDimension: embeddingDimension,
                headCount: numHeads);
            yield return new LayerNormalizationLayer<T>(embeddingDimension);

            // ReGLU-style feed-forward (using GELU approximation)
            yield return new DenseLayer<T>(embeddingDimension, embeddingDimension * 4, (IActivationFunction<T>)new GELUActivation<T>());
            yield return new DropoutLayer<T>(dropoutRate: dropoutRate);
            yield return new DenseLayer<T>(embeddingDimension * 4, embeddingDimension, (IActivationFunction<T>?)null);
            yield return new LayerNormalizationLayer<T>(embeddingDimension);
        }

        // CLS token aggregation and classification head
        yield return new DenseLayer<T>(embeddingDimension, embeddingDimension / 2, (IActivationFunction<T>)new ReLUActivation<T>());
        yield return new DenseLayer<T>(embeddingDimension / 2, numClasses, (IActivationFunction<T>)new SoftmaxActivation<T>());
    }

    /// <summary>
    /// Creates default layers for a TabR (Retrieval-Augmented) model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="embeddingDimension">Embedding dimension.</param>
    /// <param name="numLayers">Number of MLP layers.</param>
    /// <param name="numClasses">Number of output classes.</param>
    /// <param name="dropoutRate">Dropout rate for regularization.</param>
    /// <returns>A collection of layers forming the TabR network.</returns>
    public static IEnumerable<ILayer<T>> CreateDefaultTabRLayers(
        NeuralNetworkArchitecture<T> architecture,
        int numFeatures = 50,
        int embeddingDimension = 256,
        int numLayers = 4,
        int numClasses = 2,
        double dropoutRate = 0.1)
    {
        int prevDim = numFeatures;

        // Feature encoder MLP
        for (int i = 0; i < numLayers; i++)
        {
            int nextDim = i == 0 ? embeddingDimension : embeddingDimension;
            yield return new DenseLayer<T>(prevDim, nextDim, (IActivationFunction<T>)new ReLUActivation<T>());
            yield return new LayerNormalizationLayer<T>(nextDim);
            if (dropoutRate > 0)
            {
                yield return new DropoutLayer<T>(dropoutRate: dropoutRate);
            }
            prevDim = nextDim;
        }

        // Context encoding (simplified - full implementation would include retrieval)
        yield return new DenseLayer<T>(embeddingDimension, embeddingDimension, (IActivationFunction<T>)new ReLUActivation<T>());
        yield return new LayerNormalizationLayer<T>(embeddingDimension);

        // Classification head
        yield return new DenseLayer<T>(embeddingDimension, embeddingDimension / 2, (IActivationFunction<T>)new ReLUActivation<T>());
        yield return new DenseLayer<T>(embeddingDimension / 2, numClasses, (IActivationFunction<T>)new SoftmaxActivation<T>());
    }

    /// <summary>
    /// Creates default layers for a NeuralStressTest model.
    /// </summary>
    public static IEnumerable<ILayer<T>> CreateDefaultNeuralStressTestLayers(
        NeuralNetworkArchitecture<T> architecture,
        int numFeatures = 50,
        int hiddenDimension = 128,
        int numScenarios = 10,
        double dropoutRate = 0.1)
    {
        // Scenario encoder
        yield return new DenseLayer<T>(numFeatures, hiddenDimension, (IActivationFunction<T>)new ReLUActivation<T>());
        yield return new BatchNormalizationLayer<T>(hiddenDimension);
        yield return new DropoutLayer<T>(dropoutRate: dropoutRate);

        yield return new DenseLayer<T>(hiddenDimension, hiddenDimension, (IActivationFunction<T>)new ReLUActivation<T>());
        yield return new BatchNormalizationLayer<T>(hiddenDimension);
        yield return new DropoutLayer<T>(dropoutRate: dropoutRate);

        // Impact predictor
        yield return new DenseLayer<T>(hiddenDimension, hiddenDimension / 2, (IActivationFunction<T>)new ReLUActivation<T>());
        yield return new DenseLayer<T>(hiddenDimension / 2, numScenarios, (IActivationFunction<T>?)null);
    }

    #endregion

    #region Portfolio Models

    /// <summary>
    /// Creates default layers for a DeepPortfolioManager model.
    /// </summary>
    public static IEnumerable<ILayer<T>> CreateDefaultDeepPortfolioLayers(
        NeuralNetworkArchitecture<T> architecture,
        int numFeatures = 50,
        int hiddenDimension = 128,
        int numAssets = 100,
        double dropoutRate = 0.1)
    {
        // Feature encoder
        yield return new DenseLayer<T>(numFeatures, hiddenDimension, (IActivationFunction<T>)new ReLUActivation<T>());
        yield return new BatchNormalizationLayer<T>(hiddenDimension);
        yield return new DropoutLayer<T>(dropoutRate: dropoutRate);

        yield return new DenseLayer<T>(hiddenDimension, hiddenDimension, (IActivationFunction<T>)new ReLUActivation<T>());
        yield return new BatchNormalizationLayer<T>(hiddenDimension);
        yield return new DropoutLayer<T>(dropoutRate: dropoutRate);

        // Portfolio weights output with softmax for valid allocation
        yield return new DenseLayer<T>(hiddenDimension, numAssets, (IActivationFunction<T>)new SoftmaxActivation<T>());
    }

    /// <summary>
    /// Creates default layers for a HierarchicalRiskParity model.
    /// </summary>
    public static IEnumerable<ILayer<T>> CreateDefaultHierarchicalRiskParityLayers(
        NeuralNetworkArchitecture<T> architecture,
        int numFeatures = 50,
        int hiddenDimension = 128,
        int numAssets = 100,
        double dropoutRate = 0.1)
    {
        // Covariance estimation network
        yield return new DenseLayer<T>(numFeatures, hiddenDimension, (IActivationFunction<T>)new ReLUActivation<T>());
        yield return new BatchNormalizationLayer<T>(hiddenDimension);

        yield return new DenseLayer<T>(hiddenDimension, hiddenDimension, (IActivationFunction<T>)new ReLUActivation<T>());
        yield return new BatchNormalizationLayer<T>(hiddenDimension);

        // Weight prediction
        yield return new DenseLayer<T>(hiddenDimension, numAssets, (IActivationFunction<T>)new SoftmaxActivation<T>());
    }

    /// <summary>
    /// Creates default layers for an AttentionAllocation model.
    /// </summary>
    public static IEnumerable<ILayer<T>> CreateDefaultAttentionAllocationLayers(
        NeuralNetworkArchitecture<T> architecture,
        int numFeatures = 50,
        int hiddenDimension = 128,
        int numHeads = 4,
        int numAssets = 100,
        int sequenceLength = 60,
        double dropoutRate = 0.1)
    {
        // Input embedding
        yield return new DenseLayer<T>(numFeatures, hiddenDimension, (IActivationFunction<T>)new ReLUActivation<T>());
        yield return new LayerNormalizationLayer<T>(hiddenDimension);

        // Attention for cross-asset relationships
        yield return new MultiHeadAttentionLayer<T>(
            sequenceLength: sequenceLength,
            embeddingDimension: hiddenDimension,
            headCount: numHeads);
        yield return new LayerNormalizationLayer<T>(hiddenDimension);

        yield return new DenseLayer<T>(hiddenDimension, hiddenDimension * 2, (IActivationFunction<T>)new GELUActivation<T>());
        yield return new DropoutLayer<T>(dropoutRate: dropoutRate);
        yield return new DenseLayer<T>(hiddenDimension * 2, hiddenDimension, (IActivationFunction<T>?)null);
        yield return new LayerNormalizationLayer<T>(hiddenDimension);

        // Allocation output
        yield return new DenseLayer<T>(hiddenDimension, numAssets, (IActivationFunction<T>)new SoftmaxActivation<T>());
    }

    /// <summary>
    /// Creates default layers for a BlackLittermanNeural model.
    /// </summary>
    public static IEnumerable<ILayer<T>> CreateDefaultBlackLittermanNeuralLayers(
        NeuralNetworkArchitecture<T> architecture,
        int numFeatures = 50,
        int hiddenDimension = 128,
        int numAssets = 100,
        double dropoutRate = 0.1)
    {
        // View generator network
        yield return new DenseLayer<T>(numFeatures, hiddenDimension, (IActivationFunction<T>)new ReLUActivation<T>());
        yield return new BatchNormalizationLayer<T>(hiddenDimension);
        yield return new DropoutLayer<T>(dropoutRate: dropoutRate);

        yield return new DenseLayer<T>(hiddenDimension, hiddenDimension, (IActivationFunction<T>)new ReLUActivation<T>());
        yield return new BatchNormalizationLayer<T>(hiddenDimension);

        // Output: expected returns and confidence
        yield return new DenseLayer<T>(hiddenDimension, numAssets * 2, (IActivationFunction<T>?)null);
    }

    #endregion

    #region Volatility Models

    /// <summary>
    /// Creates default layers for a NeuralGARCH model.
    /// </summary>
    public static IEnumerable<ILayer<T>> CreateDefaultNeuralGARCHLayers(
        NeuralNetworkArchitecture<T> architecture,
        int lookbackWindow,
        int numAssets,
        int hiddenSize = 64,
        int numLayers = 2,
        double dropoutRate = 0.1)
    {
        int inputSize = architecture.CalculatedInputSize;
        int layers = Math.Max(1, numLayers);

        yield return new DenseLayer<T>(inputSize, hiddenSize, (IActivationFunction<T>)new ReLUActivation<T>());

        for (int i = 1; i < layers; i++)
        {
            yield return new DenseLayer<T>(hiddenSize, hiddenSize, (IActivationFunction<T>)new ReLUActivation<T>());
        }

        yield return new DropoutLayer<T>(dropoutRate: dropoutRate);

        // SoftPlus keeps outputs positive for volatility
        yield return new DenseLayer<T>(hiddenSize, numAssets, (IActivationFunction<T>)new SoftPlusActivation<T>());
    }

    /// <summary>
    /// Creates default layers for a RealizedVolatilityTransformer model.
    /// </summary>
    public static IEnumerable<ILayer<T>> CreateDefaultRealizedVolatilityTransformerLayers(
        NeuralNetworkArchitecture<T> architecture,
        int sequenceLength,
        int numAssets,
        int hiddenSize = 128,
        int numHeads = 4,
        int numLayers = 2,
        double dropoutRate = 0.1)
    {
        int inputSize = architecture.CalculatedInputSize;
        int layers = Math.Max(1, numLayers);

        // Input projection
        yield return new DenseLayer<T>(inputSize, hiddenSize, (IActivationFunction<T>)new ReLUActivation<T>());
        yield return new LayerNormalizationLayer<T>(hiddenSize);

        // Transformer encoder for temporal patterns
        for (int i = 0; i < layers; i++)
        {
            yield return new MultiHeadAttentionLayer<T>(
                sequenceLength: sequenceLength,
                embeddingDimension: hiddenSize,
                headCount: numHeads);
            yield return new LayerNormalizationLayer<T>(hiddenSize);

            yield return new DenseLayer<T>(hiddenSize, hiddenSize * 4, (IActivationFunction<T>)new GELUActivation<T>());
            yield return new DenseLayer<T>(hiddenSize * 4, hiddenSize, (IActivationFunction<T>?)null);
            yield return new DropoutLayer<T>(dropoutRate: dropoutRate);
        }

        // Volatility prediction head
        yield return new DenseLayer<T>(hiddenSize, numAssets, (IActivationFunction<T>)new SoftPlusActivation<T>());
    }

    /// <summary>
    /// Creates default layers for an OpenSora video generation model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="height">Output frame height.</param>
    /// <param name="width">Output frame width.</param>
    /// <param name="channels">Number of output channels (typically 3 for RGB).</param>
    /// <param name="hiddenDim">Hidden dimension of the DiT blocks.</param>
    /// <param name="numLayers">Number of DiT transformer layers.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <returns>A collection of layers forming the OpenSora architecture.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> OpenSora is a video generation model that uses Diffusion
    /// Transformers (DiT) to generate videos from text descriptions or images.
    /// The architecture consists of:
    /// </para>
    /// <para>
    /// <list type="bullet">
    /// <item><b>Patch Embedding:</b> Converts spatiotemporal video patches into embeddings.</item>
    /// <item><b>DiT Blocks:</b> Transformer layers with multi-head self-attention and FFN.</item>
    /// <item><b>Text/Time Projections:</b> Project conditioning signals into the hidden space.</item>
    /// <item><b>VAE Encoder/Decoder:</b> Compress images to latent space and back.</item>
    /// </list>
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultOpenSoraLayers(
        NeuralNetworkArchitecture<T> architecture,
        int height = 256,
        int width = 256,
        int channels = 3,
        int hiddenDim = 1152,
        int numLayers = 28,
        int numHeads = 16)
    {
        int latentH = height / 8;
        int latentW = width / 8;
        int latentDim = 4;
        int featH = latentH / 2;
        int featW = latentW / 2;
        int headDim = hiddenDim / numHeads;

        // Patch embedding (2x2x2 spatiotemporal patches)
        yield return new ConvolutionalLayer<T>(latentDim, latentH, latentW, hiddenDim, 2, 2, 0);

        // DiT blocks with QKV, attention projection, and FFN layers
        for (int i = 0; i < numLayers; i++)
        {
            // QKV projection (combined Q, K, V projection)
            yield return new ConvolutionalLayer<T>(hiddenDim, featH, featW, hiddenDim * 3, 1, 1, 0);

            // Attention output projection
            yield return new ConvolutionalLayer<T>(hiddenDim, featH, featW, hiddenDim, 1, 1, 0);

            // FFN with expansion (4x hidden dim as per transformer standard)
            yield return new ConvolutionalLayer<T>(hiddenDim, featH, featW, hiddenDim * 4, 1, 1, 0);
            yield return new ConvolutionalLayer<T>(hiddenDim * 4, featH, featW, hiddenDim, 1, 1, 0);
        }

        // Text projection (from CLIP-like encoder)
        yield return new ConvolutionalLayer<T>(768, 1, 1, hiddenDim, 1, 1, 0);

        // Time embedding
        yield return new ConvolutionalLayer<T>(1, 1, 1, hiddenDim, 1, 1, 0);

        // Final layer (predict noise)
        yield return new ConvolutionalLayer<T>(hiddenDim, featH, featW, latentDim * 4, 1, 1, 0);

        // VAE decoder
        yield return new ConvolutionalLayer<T>(latentDim, latentH, latentW, 256, 3, 1, 1);
        yield return new ConvolutionalLayer<T>(256, latentH * 2, latentW * 2, 128, 3, 1, 1);
        yield return new ConvolutionalLayer<T>(128, latentH * 4, latentW * 4, 64, 3, 1, 1);
        yield return new ConvolutionalLayer<T>(64, height, width, channels, 3, 1, 1);

        // VAE encoder (reverse of decoder for learned image compression)
        yield return new ConvolutionalLayer<T>(channels, height, width, 64, 3, 2, 1);
        yield return new ConvolutionalLayer<T>(64, height / 2, width / 2, 128, 3, 2, 1);
        yield return new ConvolutionalLayer<T>(128, height / 4, width / 4, 256, 3, 2, 1);
        yield return new ConvolutionalLayer<T>(256, latentH, latentW, latentDim, 3, 1, 1);
    }

    /// <summary>
    /// Creates default layers for a FILM (Frame Interpolation for Large Motion) model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="height">Input frame height.</param>
    /// <param name="width">Input frame width.</param>
    /// <param name="channels">Number of input channels (typically 3 for RGB).</param>
    /// <param name="numScales">Number of pyramid scales for multi-scale processing.</param>
    /// <param name="numFeatures">Base number of feature channels.</param>
    /// <returns>A collection of layers forming the FILM architecture.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> FILM generates smooth intermediate frames between two input frames,
    /// even when there's significant motion between them. The architecture includes:
    /// </para>
    /// <para>
    /// <list type="bullet">
    /// <item><b>Feature Extractor:</b> Extracts features from both input frames.</item>
    /// <item><b>Pyramid Layers:</b> Multi-scale feature pyramid for handling large motions.</item>
    /// <item><b>Flow Estimator:</b> Bi-directional optical flow estimation.</item>
    /// <item><b>Fusion Layers:</b> Combines features for frame synthesis.</item>
    /// </list>
    /// </para>
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultFILMLayers(
        NeuralNetworkArchitecture<T> architecture,
        int height = 256,
        int width = 256,
        int channels = 3,
        int numScales = 7,
        int numFeatures = 64)
    {
        // Multi-scale feature extractor (shared for both frames)
        yield return new ConvolutionalLayer<T>(channels, height, width, numFeatures, 3, 1, 1);
        yield return new ConvolutionalLayer<T>(numFeatures, height, width, numFeatures, 3, 1, 1);
        yield return new ConvolutionalLayer<T>(numFeatures, height, width, numFeatures * 2, 3, 2, 1);

        // Pyramid layers for each scale
        int currentH = height / 2;
        int currentW = width / 2;
        int currentC = numFeatures * 2;

        for (int s = 0; s < numScales - 1; s++)
        {
            yield return new ConvolutionalLayer<T>(currentC, currentH, currentW, Math.Min(currentC * 2, 512), 3, 2, 1);
            currentH /= 2;
            currentW /= 2;
            currentC = Math.Min(currentC * 2, 512);
            if (currentH < 4 || currentW < 4) break;
        }

        // Bi-directional flow estimator
        int flowInputC = numFeatures * 2 * 2; // Concatenated features from both frames
        yield return new ConvolutionalLayer<T>(flowInputC, height / 2, width / 2, numFeatures * 2, 3, 1, 1);
        yield return new ConvolutionalLayer<T>(numFeatures * 2, height / 2, width / 2, numFeatures, 3, 1, 1);
        yield return new ConvolutionalLayer<T>(numFeatures, height / 2, width / 2, 4, 3, 1, 1); // 4 = 2 flows x 2 coords

        // Flow refinement
        yield return new ConvolutionalLayer<T>(4 + numFeatures, height / 2, width / 2, 4, 3, 1, 1);

        // Occlusion estimator
        yield return new ConvolutionalLayer<T>(flowInputC + 4, height / 2, width / 2, 2, 3, 1, 1);

        // Feature fusion for synthesis
        int fusionInputC = numFeatures * 2 * 2 + 4 + 2; // Features + flow + occlusion
        yield return new ConvolutionalLayer<T>(fusionInputC, height / 2, width / 2, numFeatures * 2, 3, 1, 1);
        yield return new ConvolutionalLayer<T>(numFeatures * 2, height / 2, width / 2, numFeatures, 3, 1, 1);

        // Synthesis head
        yield return new ConvolutionalLayer<T>(numFeatures, height, width, channels, 3, 1, 1);
    }

    #endregion
}

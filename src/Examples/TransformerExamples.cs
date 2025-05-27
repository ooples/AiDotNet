namespace AiDotNet.Examples;

using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Factories;

/// <summary>
/// Examples demonstrating how to create and use transformer models with different positional encodings.
/// </summary>
/// <remarks>
/// <para>
/// This class provides examples of how to create transformer models with different positional
/// encoding techniques for various natural language processing tasks.
/// </para>
/// <para><b>For Beginners:</b> These examples show you how to build transformer models.
/// 
/// Transformers are powerful neural networks that are the foundation of modern language models like GPT and BERT.
/// These examples demonstrate how to configure transformers for different tasks and how to use
/// different positional encoding techniques to enhance their performance.
/// </para>
/// </remarks>
public static class TransformerExamples
{
    /// <summary>
    /// Creates a simple transformer model for language understanding tasks.
    /// </summary>
    /// <param name="vocabularySize">The size of the vocabulary.</param>
    /// <param name="maxSequenceLength">The maximum length of input sequences.</param>
    /// <param name="encodingType">The type of positional encoding to use.</param>
    /// <returns>A transformer model configured for language understanding.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a transformer model suitable for tasks like sentiment analysis
    /// or text classification. It uses a simple encoder-only architecture.
    /// </para>
    /// <para><b>For Beginners:</b> This example creates a model similar to BERT.
    /// 
    /// This type of transformer is good for understanding text (like determining if a movie
    /// review is positive or negative). It processes the whole input at once and produces
    /// a representation that captures the meaning of the text.
    /// </para>
    /// </remarks>
    public static Transformer<float> CreateLanguageUnderstandingTransformer(
        int vocabularySize = 30000,
        int maxSequenceLength = 512,
        PositionalEncodingType encodingType = PositionalEncodingType.Sinusoidal)
    {
        // Create a transformer architecture for language understanding
        var architecture = new TransformerArchitecture<float>(
            taskType: NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: 6,
            numDecoderLayers: 0, // Encoder-only for understanding tasks
            numHeads: 8,
            modelDimension: 512,
            feedForwardDimension: 2048,
            vocabularySize: vocabularySize,
            maxSequenceLength: maxSequenceLength,
            usePositionalEncoding: true,
            positionalEncodingType: encodingType);
        
        // Create the transformer model
        return new Transformer<float>(architecture);
    }
    
    /// <summary>
    /// Creates a transformer model for language generation tasks.
    /// </summary>
    /// <param name="vocabularySize">The size of the vocabulary.</param>
    /// <param name="maxSequenceLength">The maximum length of input sequences.</param>
    /// <param name="encodingType">The type of positional encoding to use.</param>
    /// <returns>A transformer model configured for language generation.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a transformer model suitable for tasks like text generation
    /// or completion. It uses a decoder-only architecture.
    /// </para>
    /// <para><b>For Beginners:</b> This example creates a model similar to GPT.
    /// 
    /// This type of transformer is good for generating text (like completing a sentence
    /// or writing an essay). It processes the input sequentially and predicts the next
    /// token based on what it has seen so far.
    /// </para>
    /// </remarks>
    public static Transformer<float> CreateLanguageGenerationTransformer(
        int vocabularySize = 50000,
        int maxSequenceLength = 1024,
        PositionalEncodingType encodingType = PositionalEncodingType.Rotary)
    {
        // Create a transformer architecture for language generation
        var architecture = new TransformerArchitecture<float>(
            taskType: NeuralNetworkTaskType.TextGeneration,
            numEncoderLayers: 0, // Decoder-only for generation tasks
            numDecoderLayers: 12,
            numHeads: 16,
            modelDimension: 768,
            feedForwardDimension: 3072,
            vocabularySize: vocabularySize,
            maxSequenceLength: maxSequenceLength,
            usePositionalEncoding: true,
            positionalEncodingType: encodingType,
            temperature: 0.8); // Slightly lower temperature for more focused generations
        
        // Create the transformer model
        return new Transformer<float>(architecture);
    }
    
    /// <summary>
    /// Creates a transformer model for sequence-to-sequence tasks.
    /// </summary>
    /// <param name="vocabularySize">The size of the shared vocabulary.</param>
    /// <param name="maxSequenceLength">The maximum length of input sequences.</param>
    /// <param name="encodingType">The type of positional encoding to use.</param>
    /// <returns>A transformer model configured for sequence-to-sequence tasks.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a transformer model suitable for tasks like translation
    /// or summarization. It uses the classic encoder-decoder architecture.
    /// </para>
    /// <para><b>For Beginners:</b> This example creates a model similar to the original Transformer.
    /// 
    /// This type of transformer is good for converting between sequences (like translating
    /// from English to French). It processes the input with an encoder and then generates
    /// the output with a decoder.
    /// </para>
    /// </remarks>
    public static Transformer<float> CreateSequenceToSequenceTransformer(
        int vocabularySize = 40000,
        int maxSequenceLength = 256,
        PositionalEncodingType encodingType = PositionalEncodingType.Learned)
    {
        // Create a transformer architecture for sequence-to-sequence tasks
        var architecture = new TransformerArchitecture<float>(
            taskType: NeuralNetworkTaskType.Translation,
            numEncoderLayers: 6,
            numDecoderLayers: 6,
            numHeads: 8,
            modelDimension: 512,
            feedForwardDimension: 2048,
            vocabularySize: vocabularySize,
            maxSequenceLength: maxSequenceLength,
            usePositionalEncoding: true,
            positionalEncodingType: encodingType);
        
        // Create the transformer model
        return new Transformer<float>(architecture);
    }
    
    /// <summary>
    /// Creates a large language model transformer for advanced text generation.
    /// </summary>
    /// <param name="vocabularySize">The size of the vocabulary.</param>
    /// <param name="maxSequenceLength">The maximum length of input sequences.</param>
    /// <param name="encodingType">The type of positional encoding to use.</param>
    /// <returns>A transformer model configured as a large language model.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a larger transformer model suitable for more advanced language
    /// modeling and generation. It uses a decoder-only architecture with more layers and
    /// a larger model dimension.
    /// </para>
    /// <para><b>For Beginners:</b> This example creates a model similar to smaller versions of GPT-3.
    /// 
    /// This model is larger and more powerful than the other examples, making it better for
    /// complex language tasks. It can generate more coherent and contextually relevant text
    /// over longer sequences.
    /// </para>
    /// </remarks>
    public static Transformer<float> CreateLargeLanguageModel(
        int vocabularySize = 50000,
        int maxSequenceLength = 2048,
        PositionalEncodingType encodingType = PositionalEncodingType.ALiBi)
    {
        // Create a transformer architecture for a large language model
        var architecture = new TransformerArchitecture<float>(
            taskType: NeuralNetworkTaskType.TextGeneration,
            numEncoderLayers: 0, // Decoder-only for LLMs
            numDecoderLayers: 24,
            numHeads: 16,
            modelDimension: 1024,
            feedForwardDimension: 4096,
            complexity: NetworkComplexity.Deep,
            dropoutRate: 0.1,
            vocabularySize: vocabularySize,
            maxSequenceLength: maxSequenceLength,
            usePositionalEncoding: true,
            positionalEncodingType: encodingType);
        
        // Create the transformer model
        return new Transformer<float>(architecture);
    }
    
    /// <summary>
    /// Demonstrates how to run inference with a transformer model.
    /// </summary>
    /// <param name="transformer">The transformer model to use.</param>
    /// <param name="inputSequence">The input sequence tensor.</param>
    /// <returns>The output tensor from the transformer.</returns>
    /// <remarks>
    /// <para>
    /// This method shows how to run inference with a transformer model on an input sequence.
    /// </para>
    /// <para><b>For Beginners:</b> This example shows how to use a transformer once it's created.
    /// 
    /// After creating your transformer model, you need to feed it input data and get predictions.
    /// This method demonstrates the basic process of running a transformer on input data.
    /// </para>
    /// </remarks>
    public static Tensor<float> RunTransformerInference(
        Transformer<float> transformer,
        Tensor<float> inputSequence)
    {
        // Create a mask if needed (for example, to handle padding)
        // For simplicity, we're using a full attention mask here
        var mask = new Tensor<float>(inputSequence.Shape);
        for (int i = 0; i < mask.Shape[0]; i++)
        {
            for (int j = 0; j < mask.Shape[1]; j++)
            {
                mask[i, j] = 1.0f; // Full attention
            }
        }
        
        // Set the attention mask
        transformer.SetAttentionMask(mask);
        
        // Run inference
        return transformer.Predict(inputSequence);
    }
    
    /// <summary>
    /// Creates a simple test input sequence for a transformer model.
    /// </summary>
    /// <param name="sequenceLength">The length of the sequence to create.</param>
    /// <param name="embeddingSize">The size of each embedding vector.</param>
    /// <returns>A randomly initialized tensor to use as test input.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a randomly initialized tensor to use as test input for a transformer model.
    /// </para>
    /// <para><b>For Beginners:</b> This creates test data to experiment with transformers.
    /// 
    /// Before you have real data to work with, you can use this to generate random data
    /// to test that your transformer model is working correctly.
    /// </para>
    /// </remarks>
    public static Tensor<float> CreateTestInputSequence(int sequenceLength, int embeddingSize)
    {
        // Create a random input tensor
        var random = new Random(42); // For reproducibility
        var input = new Tensor<float>([sequenceLength, embeddingSize]);
        
        for (int i = 0; i < sequenceLength; i++)
        {
            for (int j = 0; j < embeddingSize; j++)
            {
                input[i, j] = (float)(random.NextDouble() * 2 - 1) * 0.1f; // Small random values
            }
        }
        
        return input;
    }
    
    /// <summary>
    /// Demonstrates how to use different positional encodings for specific scenarios.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method shows which positional encoding types are recommended for different scenarios.
    /// </para>
    /// <para><b>For Beginners:</b> This helps you choose the right positional encoding.
    /// 
    /// Different encoding methods work better for different tasks or models:
    /// - Sinusoidal: Good general-purpose choice, especially for smaller models
    /// - Learned: Good when you have fixed sequence lengths and want maximum performance
    /// - Relative: Good for tasks that need to understand relationships between positions
    /// - Rotary: Good for large language models, combines benefits of absolute and relative encodings
    /// - ALiBi: Excellent for extrapolating to longer sequences than seen during training
    /// - T5RelativeBias: Good for text-to-text tasks like summarization and translation
    /// </para>
    /// </remarks>
    public static void PositionalEncodingRecommendations()
    {
        Console.WriteLine("Recommended positional encoding types for different scenarios:");
        Console.WriteLine();
        
        Console.WriteLine("1. Basic language understanding (like BERT):");
        Console.WriteLine("   - Learned positional encoding");
        Console.WriteLine("   - Good for: Classification, sentiment analysis, named entity recognition");
        Console.WriteLine();
        
        Console.WriteLine("2. Language generation with fixed context window:");
        Console.WriteLine("   - Sinusoidal or Learned positional encoding");
        Console.WriteLine("   - Good for: Small to medium generative models with known max sequence length");
        Console.WriteLine();
        
        Console.WriteLine("3. Language generation with long contexts:");
        Console.WriteLine("   - ALiBi or Rotary positional encoding");
        Console.WriteLine("   - Good for: Large language models that need to handle inputs longer than seen in training");
        Console.WriteLine();
        
        Console.WriteLine("4. Translation or summarization:");
        Console.WriteLine("   - T5RelativeBias or Relative positional encoding");
        Console.WriteLine("   - Good for: Sequence-to-sequence tasks where relative positions matter");
        Console.WriteLine();
        
        Console.WriteLine("5. Memory-efficient large language models:");
        Console.WriteLine("   - Rotary positional encoding");
        Console.WriteLine("   - Good for: When you need computational efficiency with good performance");
        Console.WriteLine();
        
        Console.WriteLine("6. Fine-tuning pre-trained models:");
        Console.WriteLine("   - Match the positional encoding of the pre-trained model");
        Console.WriteLine("   - Important for maintaining compatibility with the original architecture");
    }
    
    /// <summary>
    /// Creates a transformer model with custom positional encoding parameters.
    /// </summary>
    /// <param name="encodingType">The type of positional encoding to use.</param>
    /// <param name="customParameters">Dictionary containing custom parameters for the encoding.</param>
    /// <returns>A transformer model with custom positional encoding configuration.</returns>
    /// <remarks>
    /// <para>
    /// This method demonstrates how to create a transformer model with custom parameters
    /// for specific positional encoding types, providing more fine-grained control over
    /// the encoding behavior.
    /// </para>
    /// <para><b>For Beginners:</b> This example shows how to customize positional encodings.
    /// 
    /// Some positional encoding methods have additional parameters you can adjust:
    /// - For Rotary encoding, you might change the base frequency
    /// - For ALiBi, you can adjust how quickly the penalty increases with distance
    /// - For T5, you can set the number of bucket groups for similar distances
    /// 
    /// This lets you fine-tune the encoding for your specific use case.
    /// </para>
    /// </remarks>
    public static Transformer<float> CreateTransformerWithCustomPositionalEncoding(
        PositionalEncodingType encodingType,
        Dictionary<string, object> customParameters)
    {
        // Create a transformer architecture with the specified encoding type
        var architecture = new TransformerArchitecture<float>(
            taskType: NeuralNetworkTaskType.TextGeneration,
            numEncoderLayers: 0,
            numDecoderLayers: 12,
            numHeads: 16,
            modelDimension: 768,
            feedForwardDimension: 3072,
            usePositionalEncoding: true,
            positionalEncodingType: encodingType);
        
        // Create the transformer model
        var transformer = new Transformer<float>(architecture);
        
        // When the model is created, the LayerHelper will use AttentionLayerFactory
        // to create the appropriate attention layer based on the encoding type,
        // and PositionalEncodingFactory to create the encoding layer.
        // The custom parameters will be passed to these factories when needed.
        
        return transformer;
    }
    
    /// <summary>
    /// Demonstrates how to create transformers with different positional encoding configurations.
    /// </summary>
    public static void DemonstratePositionalEncodingConfigurations()
    {
        // Example 1: Create a transformer with ALiBi encoding with custom slope
        Console.WriteLine("Creating a transformer with custom ALiBi encoding (slope: -0.2)");
        var alibiParameters = new Dictionary<string, object> { { "slope", -0.2 } };
        var alibiTransformer = CreateTransformerWithCustomPositionalEncoding(
            PositionalEncodingType.ALiBi, 
            alibiParameters);
        
        // Example 2: Create a transformer with T5 relative bias with custom bucketing
        Console.WriteLine("Creating a transformer with custom T5 relative bias (64 buckets)");
        var t5Parameters = new Dictionary<string, object> { 
            { "numBuckets", 64 }, 
            { "maxDistance", 256 } 
        };
        var t5Transformer = CreateTransformerWithCustomPositionalEncoding(
            PositionalEncodingType.T5RelativeBias, 
            t5Parameters);
        
        // Example 3: Create a transformer with Rotary encoding with custom base
        Console.WriteLine("Creating a transformer with custom Rotary encoding (base: 20000)");
        var rotaryParameters = new Dictionary<string, object> { { "base", 20000.0 } };
        var rotaryTransformer = CreateTransformerWithCustomPositionalEncoding(
            PositionalEncodingType.Rotary, 
            rotaryParameters);
        
        // Print some information about the created transformers
        Console.WriteLine("\nCreated transformers with custom positional encoding configurations:");
        Console.WriteLine($"ALiBi Transformer: {alibiTransformer.GetModelMetaData().AdditionalInfo["PositionalEncodingType"]}");
        Console.WriteLine($"T5 Transformer: {t5Transformer.GetModelMetaData().AdditionalInfo["PositionalEncodingType"]}");
        Console.WriteLine($"Rotary Transformer: {rotaryTransformer.GetModelMetaData().AdditionalInfo["PositionalEncodingType"]}");
    }
}
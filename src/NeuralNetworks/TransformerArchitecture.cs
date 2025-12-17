namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Defines the architecture configuration for a Transformer neural network.
/// </summary>
/// <remarks>
/// <para>
/// The TransformerArchitecture class encapsulates all the hyperparameters and configuration options
/// needed to define a Transformer neural network. It includes settings for the encoder and decoder stacks,
/// attention mechanisms, model dimensions, and other key aspects that determine the network's structure
/// and behavior.
/// </para>
/// <para>
/// Transformers are particularly effective for sequence-based tasks like natural language processing,
/// translation, text summarization, and other tasks that benefit from understanding the relationships
/// between elements in a sequence.
/// </para>
/// <para><b>For Beginners:</b> Think of this class as a blueprint for building a Transformer.
/// 
/// Just like building a house requires decisions about how many rooms, how big each room should be, 
/// and what materials to use, building a Transformer requires decisions about:
/// - How many layers of processing to include
/// - How much information to process at once
/// - How to connect different parts of the network
/// 
/// This class stores all those decisions in one place, making it easier to create Transformer
/// networks with different capabilities for different tasks.
/// </para>
/// </remarks>
/// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
public class TransformerArchitecture<T> : NeuralNetworkArchitecture<T>
{
    /// <summary>
    /// Gets the number of encoder layers in the Transformer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Encoder layers process the input sequence and create representations that capture the meaning
    /// and context of each element in the sequence. Each encoder layer typically consists of a
    /// self-attention mechanism followed by a feed-forward network.
    /// </para>
    /// <para><b>For Beginners:</b> Encoder layers are like processing stations that understand your input data.
    /// 
    /// More encoder layers mean:
    /// - The network can understand more complex patterns
    /// - It can capture deeper relationships between words or elements
    /// - The model becomes more powerful but also more computationally expensive
    /// 
    /// For example, with text processing, more encoder layers help the model better understand
    /// the subtle meanings and connections between words in a sentence.
    /// </para>
    /// </remarks>
    public int NumEncoderLayers { get; }

    /// <summary>
    /// Gets the number of decoder layers in the Transformer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Decoder layers are used to generate output sequences based on both the encoded input and
    /// the previously generated elements of the output. Each decoder layer typically includes
    /// both self-attention and cross-attention mechanisms followed by a feed-forward network.
    /// </para>
    /// <para><b>For Beginners:</b> Decoder layers are like writing stations that create your output data.
    /// 
    /// Decoder layers:
    /// - Generate responses or translations one element at a time
    /// - Look at both what they've already generated and the original input
    /// - Help ensure the output is coherent and relevant to the input
    /// 
    /// For example, in a translation task, decoder layers would generate the translated text
    /// word by word, making sure each new word fits with both the original text and the 
    /// translation so far.
    /// </para>
    /// </remarks>
    public int NumDecoderLayers { get; }

    /// <summary>
    /// Gets the number of attention heads in each multi-head attention layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Attention heads allow the model to focus on different aspects of the input simultaneously.
    /// Each head can learn to attend to different types of relationships or patterns in the data.
    /// Having multiple heads improves the model's ability to capture diverse aspects of the input.
    /// </para>
    /// <para><b>For Beginners:</b> Attention heads are like different perspectives on the same information.
    /// 
    /// Think of attention heads as different people reading the same text:
    /// - One person might focus on the main characters
    /// - Another might focus on the setting or time period
    /// - A third might focus on the emotional tone
    /// 
    /// By combining these different perspectives, the model gets a more complete understanding.
    /// More heads generally means the model can capture more nuanced relationships,
    /// but also requires more computation.
    /// </para>
    /// </remarks>
    public int NumHeads { get; }

    /// <summary>
    /// Gets the dimension of the model's internal representations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The model dimension determines the size of the vectors used to represent each element in the
    /// input and output sequences. It affects the model's capacity to store information and its
    /// ability to capture complex patterns in the data.
    /// </para>
    /// <para><b>For Beginners:</b> Model dimension is like the size of the "memory" for each word or element.
    /// 
    /// A larger model dimension means:
    /// - Each word or element carries more information
    /// - The model can represent more subtle meanings and relationships
    /// - The network becomes more powerful but requires more memory and computation
    /// 
    /// For example, a model dimension of 512 means each word is represented by 512 numbers,
    /// which can capture various aspects of its meaning, grammatical role, and relationships
    /// with other words.
    /// </para>
    /// </remarks>
    public int ModelDimension { get; }

    /// <summary>
    /// Gets the dimension of the feed-forward networks within the Transformer layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Each Transformer layer contains a feed-forward network that processes the output of the
    /// attention mechanism. The feed-forward dimension determines the intermediate size of this
    /// network, which affects its capacity to transform and process the attended information.
    /// </para>
    /// <para><b>For Beginners:</b> Feed-forward dimension is like the processing power in each layer.
    /// 
    /// The feed-forward networks:
    /// - Process the information after the attention mechanism has focused on relevant parts
    /// - Transform the data to extract useful patterns and relationships
    /// - Usually have a larger dimension than the model dimension to allow for more processing capacity
    /// 
    /// A larger feed-forward dimension generally allows the model to learn more complex transformations,
    /// but requires more computation. It's common for this value to be 4 times the model dimension.
    /// </para>
    /// </remarks>
    public int FeedForwardDimension { get; }

    /// <summary>
    /// Gets the dropout rate used for regularization in the Transformer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Dropout is a regularization technique that helps prevent overfitting by randomly "dropping out"
    /// (setting to zero) a portion of the neurons during training. The dropout rate specifies the
    /// probability of each neuron being dropped.
    /// </para>
    /// <para><b>For Beginners:</b> Dropout rate is like training with random challenges to improve resilience.
    /// 
    /// Think of dropout as training a team where:
    /// - During practice, random team members sit out temporarily
    /// - This forces the remaining members to adapt and work without relying too much on specific teammates
    /// - When the full team plays together later, they perform better and more robustly
    /// 
    /// A typical dropout rate is around 0.1 (10%), meaning each neuron has a 10% chance of being temporarily
    /// disabled during training. This helps prevent the model from becoming too dependent on specific
    /// neurons, making it more robust.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; }

    /// <summary>
    /// Gets the maximum length of input sequences that the Transformer can process.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This parameter limits the length of input sequences that the model can handle. It affects
    /// the memory requirements and computational cost of the model, as longer sequences require
    /// more resources to process. It also determines the maximum range of dependencies that the
    /// model can capture.
    /// </para>
    /// <para><b>For Beginners:</b> Maximum sequence length is like the longest text the model can read at once.
    /// 
    /// This setting:
    /// - Limits how long your input can be (like number of words in a sentence)
    /// - Affects how much memory the model needs
    /// - Determines how far apart relationships can be captured
    /// 
    /// For example, a max sequence length of 512 means the model can process texts up to 512 tokens long
    /// (roughly 384-512 words in English). Longer texts would need to be broken into smaller chunks.
    /// Common values range from 512 to 2048, with larger models supporting longer sequences.
    /// </para>
    /// </remarks>
    public int MaxSequenceLength { get; }

    /// <summary>
    /// Gets the size of the vocabulary for text-based tasks.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For natural language processing tasks, this parameter determines the number of unique tokens
    /// (typically words or subwords) that the model can recognize and generate. A larger vocabulary
    /// allows for more precise representation of language but increases the model size.
    /// </para>
    /// <para><b>For Beginners:</b> Vocabulary size is like the dictionary the model uses.
    /// 
    /// The vocabulary size:
    /// - Defines how many different words or word pieces the model knows
    /// - Affects the model's ability to understand rare or specialized terms
    /// - Impacts the memory requirements of the model
    /// 
    /// For example, a vocabulary size of 30,000 means the model can recognize and use 30,000 different
    /// words or subword units. Larger vocabularies help with specialized terminology but make the model
    /// larger. Common sizes range from 10,000 to 50,000 for general language models.
    /// </para>
    /// </remarks>
    public int VocabularySize { get; }

    /// <summary>
    /// Gets a value indicating whether positional encoding is used in the Transformer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Transformers don't inherently capture the order of elements in a sequence, as they process all
    /// elements in parallel. Positional encoding adds information about the position of each element
    /// to its representation, allowing the model to understand the sequence order.
    /// </para>
    /// <para><b>For Beginners:</b> Positional encoding is like adding page numbers to a book.
    /// 
    /// Without positional encoding:
    /// - The model would see all words at once, but wouldn't know their order
    /// - "The dog chased the cat" and "The cat chased the dog" would look the same
    /// 
    /// Positional encoding adds location information to each word, so the model knows which word
    /// comes first, second, and so on. This is usually set to true, as order is critical for
    /// understanding most sequences.
    /// </para>
    /// </remarks>
    public bool UsePositionalEncoding { get; }

    /// <summary>
    /// Gets the temperature parameter used for controlling randomness in text generation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Temperature is a parameter that controls the randomness of text generation in models like
    /// Transformers. A higher temperature increases randomness and creativity, while a lower
    /// temperature makes the model more deterministic and focused on the most likely outputs.
    /// </para>
    /// <para><b>For Beginners:</b> Temperature is like a creativity dial for text generation.
    /// 
    /// Think of temperature as controlling how "creative" or "predictable" the model's outputs are:
    /// - Low temperature (e.g., 0.3): More focused, predictable responses
    /// - Medium temperature (e.g., 1.0): Balanced between predictability and creativity
    /// - High temperature (e.g., 1.5): More diverse, surprising, and creative outputs
    /// 
    /// For example, when generating story ideas:
    /// - Low temperature might give conventional plots
    /// - High temperature might give unusual or unexpected storylines
    /// 
    /// This parameter is only relevant for text generation tasks, not for classification or other tasks.
    /// </para>
    /// </remarks>
    public double Temperature { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="TransformerArchitecture{T}"/> class with the specified parameters.
    /// </summary>
    /// <param name="inputType">The type of input the network will process (e.g., text, image).</param>
    /// <param name="taskType">The type of task the network will perform (e.g., classification, generation).</param>
    /// <param name="numEncoderLayers">The number of encoder layers in the Transformer.</param>
    /// <param name="numDecoderLayers">The number of decoder layers in the Transformer.</param>
    /// <param name="numHeads">The number of attention heads in each multi-head attention layer.</param>
    /// <param name="modelDimension">The dimension of the model's internal representations.</param>
    /// <param name="feedForwardDimension">The dimension of the feed-forward networks within the Transformer layers.</param>
    /// <param name="complexity">The overall complexity level of the network. Defaults to Medium.</param>
    /// <param name="inputSize">The size of the input for simple vector inputs. Defaults to 0.</param>
    /// <param name="inputHeight">The height of the input for 2D inputs like images. Defaults to 0.</param>
    /// <param name="inputWidth">The width of the input for 2D inputs like images. Defaults to 0.</param>
    /// <param name="inputDepth">The depth of the input for multi-channel inputs. Defaults to 1.</param>
    /// <param name="outputSize">The size of the output of the network. Defaults to 0.</param>
    /// <param name="dropoutRate">The dropout rate used for regularization. Defaults to 0.1.</param>
    /// <param name="maxSequenceLength">The maximum length of input sequences. Defaults to 512.</param>
    /// <param name="vocabularySize">The size of the vocabulary for text-based tasks. Defaults to 0.</param>
    /// <param name="usePositionalEncoding">Whether to use positional encoding. Defaults to true.</param>
    /// <param name="temperature">The temperature parameter for text generation. Defaults to 1.0.</param>
    /// <param name="layers">Optional custom layers for the network. Defaults to null.</param>
    /// <param name="rbmLayers">Optional Restricted Boltzmann Machine layers for the network. Defaults to null.</param>
    /// <remarks>
    /// <para>
    /// This constructor initializes a new TransformerArchitecture with the specified parameters, which will
    /// define the structure and behavior of a Transformer neural network. It passes the basic network
    /// parameters to the base NeuralNetworkArchitecture class and initializes the Transformer-specific
    /// parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor is where you set all the options for your Transformer.
    /// 
    /// When creating a new Transformer architecture, you need to decide:
    /// - What kind of input it will process (text, images, etc.)
    /// - What task it will perform (translation, classification, etc.)
    /// - How big and powerful the model should be
    /// - How it will handle and process sequences
    /// 
    /// Many parameters have default values that work well for common cases, so you only need to specify
    /// the ones that are important for your specific task.
    /// 
    /// Think of it like configuring a new computer - you specify the components and settings
    /// that matter for what you'll be using it for.
    /// </para>
    /// </remarks>
    public TransformerArchitecture(
        InputType inputType,
        NeuralNetworkTaskType taskType,
        int numEncoderLayers,
        int numDecoderLayers,
        int numHeads,
        int modelDimension,
        int feedForwardDimension,
        NetworkComplexity complexity = NetworkComplexity.Medium,
        int inputSize = 0,
        int outputSize = 0,
        double dropoutRate = 0.1,
        int maxSequenceLength = 512,
        int vocabularySize = 0,
        bool usePositionalEncoding = true,
        double temperature = 1.0,
        List<ILayer<T>>? layers = null)
        : base(
            inputType: inputType,
            taskType: taskType,
            complexity: complexity,
            inputSize: inputSize,
            outputSize: outputSize,
            layers: layers)
    {
        NumEncoderLayers = numEncoderLayers;
        NumDecoderLayers = numDecoderLayers;
        NumHeads = numHeads;
        ModelDimension = modelDimension;
        FeedForwardDimension = feedForwardDimension;
        DropoutRate = dropoutRate;
        MaxSequenceLength = maxSequenceLength;
        VocabularySize = vocabularySize;
        UsePositionalEncoding = usePositionalEncoding;
        Temperature = temperature;
    }
}

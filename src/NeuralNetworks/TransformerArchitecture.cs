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
    /// Gets the number of warmup steps for the default Noam (Vaswani 2017)
    /// learning-rate schedule attached to the Transformer's default Adam
    /// optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The Vaswani 2017 §5.3 recipe uses a 4000-step linear warmup followed
    /// by inverse-sqrt decay. That paper trained for 100k+ steps on WMT
    /// translation, so 4000 (~4% of total) was a small fraction. For shorter
    /// training budgets this default keeps the model in warmup forever and
    /// the LR never reaches its peak — set this to roughly <c>0.1 ·
    /// total_steps</c> for fine-tuning or short training runs.
    /// </para>
    /// <para>
    /// Ignored when an explicit optimizer is passed to the Transformer
    /// constructor (the user-supplied optimizer's own scheduler config wins).
    /// </para>
    /// </remarks>
    public int WarmupSteps { get; }

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
    /// Strategy for collapsing the encoder's
    /// <c>[batch, seq, dim]</c> hidden states into a single
    /// <c>[batch, dim]</c> vector before the classification head, when
    /// the task is single-label per sequence. Defaults to
    /// <see cref="SequencePoolingMode.LastToken"/> for token-input
    /// architectures (<see cref="VocabularySize"/> &gt; 0) and to
    /// <see cref="SequencePoolingMode.MeanPool"/> for continuous-input
    /// architectures.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Picking the wrong mode silently destroys position-specific signal.
    /// MeanPool over a token-LM context maps every prefix to roughly the
    /// same averaged hidden state, which makes the model converge to
    /// <c>~uniform / V</c> softmax — the bug tracked in
    /// <a href="https://github.com/ooples/AiDotNet/issues/1232">AiDotNet#1232</a>.
    /// </para>
    /// </remarks>
    public SequencePoolingMode SequencePooling { get; }

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
    /// <param name="sequencePooling">
    /// How to collapse the sequence dimension when producing model output.
    /// <c>null</c> defers to a sensible default per <c>TaskType</c>:
    /// token-input architectures (<c>vocabularySize &gt; 0</c>) get
    /// <c>SequencePoolingMode.LastToken</c> (matches the GPT / Llama /
    /// Mistral output-head convention), continuous-input architectures get
    /// <c>SequencePoolingMode.MeanPool</c> (correct for document-level
    /// classification). Pass an explicit value to override.
    /// </param>
    /// <param name="layers">Optional custom layers for the network. Defaults to null.</param>
    /// <param name="warmupSteps">
    /// Number of warmup steps for the Vaswani 2017 Noam learning-rate
    /// schedule used by the Transformer's default Adam-with-Noam
    /// optimizer (β₁=0.9, β₂=0.98, ε=1e-9). LR ramps linearly from a
    /// tiny value to peak across the first <paramref name="warmupSteps"/>
    /// batches, then decays as t<sup>-0.5</sup>. Defaults to 4000
    /// (paper-canonical). For training budgets too small to warm up
    /// over (under ~100 steps), drop the schedule entirely and use a
    /// constant LR — the constructor rejects values ≤ 0 with
    /// <c>ArgumentOutOfRangeException</c>.
    /// </param>
    /// <param name="randomSeed">
    /// Optional seed for deterministic layer-weight initialization.
    /// When provided, every weighted layer in the constructed network
    /// receives a per-layer seed derived from this value (see
    /// <c>LayerHelper</c>'s seeded-RNG plumbing) so identical seeds
    /// produce identical initial weight tensors. Defaults to <c>null</c>
    /// — uses the framework's secure non-deterministic RNG, suitable
    /// for production training.
    /// </param>
    /// <remarks>
    /// <para>
    /// This constructor initializes a new TransformerArchitecture with the specified parameters, which will
    /// define the structure and behavior of a Transformer neural network. It passes the basic network
    /// parameters to the base NeuralNetworkArchitecture class and initializes the Transformer-specific
    /// parameters.
    /// </para>
    /// <para>
    /// <b>Breaking change (closes #1382):</b> when <paramref name="layers"/> is non-null AND
    /// non-empty, <paramref name="numEncoderLayers"/> and <paramref name="numDecoderLayers"/>
    /// MUST be 0. The constructor throws <see cref="ArgumentException"/> otherwise. Previously
    /// these parameters were silently ignored when <c>layers:</c> was supplied (the custom list
    /// REPLACES the auto-built encoder/decoder block), which presented as a model with 0 trainable
    /// parameters or a first-batch shape-mismatch crash inside the loss function. Migration:
    /// </para>
    /// <list type="bullet">
    /// <item><b>If you want auto-built encoder blocks composed AROUND your custom layers</b> —
    /// not supported. The <c>layers:</c> contract is "consumer owns the entire forward graph".
    /// Include your own <see cref="NeuralNetworks.Layers.MultiHeadAttentionLayer{T}"/> /
    /// feed-forward / norm layers explicitly in the list.</item>
    /// <item><b>If you intentionally want the custom list to be the whole graph</b> — pass
    /// <c>numEncoderLayers: 0, numDecoderLayers: 0</c> alongside your <c>layers:</c> list.</item>
    /// <item><b>If you want the default Vaswani-style encoder</b> — omit <c>layers:</c> (pass
    /// <c>null</c>) and the constructor uses your <c>numEncoderLayers</c> /
    /// <c>numDecoderLayers</c> / <c>numHeads</c> to build a standard layer stack.</item>
    /// </list>
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
    /// <summary>
    /// Binary-compatible overload preserving the pre-PR-#1270 ctor signature.
    /// Forwards to the canonical ctor with the new <c>warmupSteps</c> and
    /// <c>randomSeed</c> parameters defaulted (4000, null), matching the
    /// behaviour callers would see if they upgraded source-only. Closes
    /// review-comment #1270.yYt1 (binary-breaking change to the
    /// already-public constructor signature).
    /// </summary>
    public TransformerArchitecture(
        InputType inputType,
        NeuralNetworkTaskType taskType,
        int numEncoderLayers,
        int numDecoderLayers,
        int numHeads,
        int modelDimension,
        int feedForwardDimension,
        NetworkComplexity complexity,
        int inputSize,
        int outputSize,
        double dropoutRate,
        int maxSequenceLength,
        int vocabularySize,
        bool usePositionalEncoding,
        double temperature,
        SequencePoolingMode? sequencePooling,
        List<ILayer<T>>? layers)
        : this(
            inputType: inputType,
            taskType: taskType,
            numEncoderLayers: numEncoderLayers,
            numDecoderLayers: numDecoderLayers,
            numHeads: numHeads,
            modelDimension: modelDimension,
            feedForwardDimension: feedForwardDimension,
            complexity: complexity,
            inputSize: inputSize,
            outputSize: outputSize,
            dropoutRate: dropoutRate,
            maxSequenceLength: maxSequenceLength,
            vocabularySize: vocabularySize,
            usePositionalEncoding: usePositionalEncoding,
            temperature: temperature,
            sequencePooling: sequencePooling,
            layers: layers,
            warmupSteps: 4000,
            randomSeed: null)
    {
    }

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
        SequencePoolingMode? sequencePooling = null,
        List<ILayer<T>>? layers = null,
        int warmupSteps = 4000,
        int? randomSeed = null)
        : base(
            inputType: inputType,
            taskType: taskType,
            complexity: complexity,
            inputSize: inputSize,
            outputSize: outputSize,
            layers: layers)
    {
        // Validate warmupSteps at the architecture boundary so a 0/negative
        // value fails immediately and is attributed to this parameter,
        // instead of bubbling up later as an exception from the
        // NoamSchedule ctor during Transformer construction or training
        // (where the call stack makes the root cause harder to find).
        // Closes review-comment #1269.vuR5 / .vzGk.
        if (warmupSteps <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(warmupSteps),
                warmupSteps,
                "warmupSteps must be positive — the Vaswani 2017 Noam schedule has no defined behavior for non-positive warmup. " +
                "For a budget too small to warm up over (less than ~100 steps), drop the schedule entirely and use a constant LR.");
        }

        // Closes #1382: when a custom `layers:` list is provided, the
        // Transformer's InitializeLayers path uses ONLY that list — the
        // auto-built encoder block (numEncoderLayers × MultiHeadAttention,
        // numDecoderLayers × DecoderLayer, head, etc) is NOT composed
        // with it. Previously the structural parameters were silently
        // accepted and ignored, leaving the user to discover the
        // miswiring as either:
        //   (a) a zero-trainable-parameter model that "trains" vacuously
        //       (the HRE-substitution consumer reproducer in #1382), OR
        //   (b) a shape mismatch on the very first batch when the
        //       custom chain's final shape doesn't match outputSize.
        // Both surface FAR from the constructor where the mistake was
        // made. Fail-fast here with a diagnostic that names the actual
        // contract: layers: REPLACES the auto-built block, so structural
        // parameters that would have driven that block must be left at
        // their no-op defaults when layers: is supplied.
        if (layers is not null && layers.Count > 0)
        {
            if (numEncoderLayers > 0)
            {
                throw new ArgumentException(
                    $"TransformerArchitecture cannot accept both a custom 'layers:' list ({layers.Count} layers) " +
                    $"and numEncoderLayers={numEncoderLayers}. Providing layers: REPLACES the auto-built encoder " +
                    "block (numEncoderLayers × MultiHeadAttention + feed-forward + norm); the structural parameters " +
                    "would be silently ignored otherwise, leaving the model with 0 trainable parameters and no " +
                    "optimizer signal. Either (a) pass numEncoderLayers: 0 and include your own attention blocks " +
                    "in the layers: list, or (b) omit layers: to get the default encoder. See #1382.",
                    nameof(numEncoderLayers));
            }
            if (numDecoderLayers > 0)
            {
                throw new ArgumentException(
                    $"TransformerArchitecture cannot accept both a custom 'layers:' list ({layers.Count} layers) " +
                    $"and numDecoderLayers={numDecoderLayers}. Same reasoning as numEncoderLayers above — providing " +
                    "layers: REPLACES the auto-built decoder block. Either (a) pass numDecoderLayers: 0 and include " +
                    "your own decoder blocks in the layers: list, or (b) omit layers: to get the default decoder. " +
                    "See #1382.",
                    nameof(numDecoderLayers));
            }
        }

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
        WarmupSteps = warmupSteps;
        RandomSeed = randomSeed;
        // Default sequence pooling: token-input architectures (vocabSize > 0)
        // are autoregressive LMs by convention — use the LAST position's
        // hidden state, matching GPT / Llama / Mistral output heads.
        // Continuous-input architectures default to MEAN POOLING which is
        // the right choice for document-level sequence classification.
        // Callers can override either default via the explicit parameter.
        SequencePooling = sequencePooling ?? (vocabularySize > 0
            ? SequencePoolingMode.LastToken
            : SequencePoolingMode.MeanPool);
    }

    /// <summary>
    /// Infers the correct classification task type from the shape of the target data.
    /// </summary>
    /// <param name="targetShape">The shape of the target tensor (e.g., [numSamples] or [numSamples, seqLen]).</param>
    /// <param name="inputSeqLen">The sequence length of the input data.</param>
    /// <returns>
    /// <see cref="NeuralNetworkTaskType.TokenClassification"/> if targets have a sequence
    /// dimension matching the input (per-position labels);
    /// <see cref="NeuralNetworkTaskType.SequenceClassification"/> if targets are 1D
    /// (one label per sequence).
    /// </returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this helper before constructing a Transformer to automatically
    /// pick the right task type based on your target data shape. If your targets have one label
    /// per token position (like NER or POS tagging), it returns TokenClassification. If your
    /// targets have one label per whole sequence (like sentiment analysis), it returns
    /// SequenceClassification.
    /// </para>
    /// </remarks>
    internal static NeuralNetworkTaskType InferClassificationTaskType(int[] targetShape, int inputSeqLen)
    {
        if (targetShape is null || targetShape.Length == 0)
            throw new ArgumentException("Target shape must not be null or empty.", nameof(targetShape));
        if (inputSeqLen <= 0)
            throw new ArgumentOutOfRangeException(nameof(inputSeqLen), "Input sequence length must be positive.");

        if (targetShape.Length >= 2 && targetShape[1] == inputSeqLen)
        {
            // Target has shape [numSamples, seqLen] → per-position labels
            return NeuralNetworkTaskType.TokenClassification;
        }

        // Target has shape [numSamples] → one label per sequence
        return NeuralNetworkTaskType.SequenceClassification;
    }

    /// <summary>
    /// Validates that the configured task type is consistent with the target data shape.
    /// Throws a descriptive exception if there is a mismatch.
    /// </summary>
    /// <param name="taskType">The configured task type.</param>
    /// <param name="targetShape">The shape of the target tensor.</param>
    /// <param name="inputSeqLen">The sequence length of the input.</param>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the task type is SequenceClassification but targets have per-position labels,
    /// or when the task type is TokenClassification but targets have per-sequence labels.
    /// </exception>
    internal static void ValidateTaskTypeVsTargetShape(
        NeuralNetworkTaskType taskType, int[] targetShape, int inputSeqLen)
    {
        if (targetShape is null || targetShape.Length == 0)
            throw new ArgumentException("Target shape must not be null or empty.", nameof(targetShape));
        if (inputSeqLen <= 0)
            throw new ArgumentOutOfRangeException(nameof(inputSeqLen), "Input sequence length must be positive.");

        bool hasPerPositionTargets = targetShape.Length >= 2 && targetShape[1] == inputSeqLen;

        if (taskType == NeuralNetworkTaskType.SequenceClassification && hasPerPositionTargets)
        {
            throw new InvalidOperationException(
                $"Task type mismatch: SequenceClassification was configured, but target shape " +
                $"[{string.Join(", ", targetShape)}] has a sequence dimension matching the input " +
                $"sequence length ({inputSeqLen}). This indicates per-position labels. " +
                $"Use NeuralNetworkTaskType.TokenClassification instead, or use " +
                $"TransformerArchitecture.InferClassificationTaskType() to auto-detect.");
        }

        if (taskType == NeuralNetworkTaskType.TokenClassification && !hasPerPositionTargets)
        {
            throw new InvalidOperationException(
                $"Task type mismatch: TokenClassification was configured, but target shape " +
                $"[{string.Join(", ", targetShape)}] does not have a sequence dimension matching " +
                $"the input sequence length ({inputSeqLen}). TokenClassification requires " +
                $"per-position labels with shape [batch, {inputSeqLen}, ...]. " +
                $"Use NeuralNetworkTaskType.SequenceClassification instead, or use " +
                $"TransformerArchitecture.InferClassificationTaskType() to auto-detect.");
        }
    }
}

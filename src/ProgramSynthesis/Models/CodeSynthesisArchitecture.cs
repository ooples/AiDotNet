using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ProgramSynthesis.Enums;

namespace AiDotNet.ProgramSynthesis.Models;

/// <summary>
/// Defines the architecture configuration for code synthesis and understanding models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// CodeSynthesisArchitecture extends the neural network architecture with code-specific
/// parameters such as programming language, maximum code length, vocabulary size, and
/// synthesis strategy. It serves as a blueprint for building code models like CodeBERT,
/// GraphCodeBERT, and CodeT5.
/// </para>
/// <para><b>For Beginners:</b> This is a blueprint for building AI models that understand code.
///
/// Just like TransformerArchitecture defines how to build a general transformer,
/// CodeSynthesisArchitecture defines how to build models specifically for:
/// - Understanding code
/// - Generating code
/// - Translating between programming languages
/// - Finding bugs
/// - Completing code
///
/// It includes all the settings needed to build these specialized code models,
/// like which programming language to work with and how much code it can handle.
/// </para>
/// </remarks>
public class CodeSynthesisArchitecture<T> : NeuralNetworkArchitecture<T>
{
    /// <summary>
    /// Gets the type of synthesis approach to use.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Specifies whether to use neural, symbolic, hybrid, or genetic programming
    /// approaches for code synthesis.
    /// </para>
    /// <para><b>For Beginners:</b> This chooses the strategy for generating code.
    ///
    /// Different approaches work better for different problems:
    /// - Neural: Good for learning from examples
    /// - Symbolic: Good for following rules
    /// - Hybrid: Combines both approaches
    /// - GeneticProgramming: Good for optimization problems
    /// </para>
    /// </remarks>
    public SynthesisType SynthesisType { get; }

    /// <summary>
    /// Gets the target programming language.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Specifies which programming language the model is designed to work with.
    /// </para>
    /// <para><b>For Beginners:</b> This is which programming language the model knows.
    ///
    /// Like a translator specializing in French or Spanish, code models often
    /// specialize in specific languages like Python or Java.
    /// </para>
    /// </remarks>
    public ProgramLanguage TargetLanguage { get; }

    /// <summary>
    /// Gets the number of encoder layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The number of transformer encoder layers used to process and understand code.
    /// More layers allow for deeper understanding but require more computation.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how deeply the model analyzes code.
    ///
    /// More encoder layers mean:
    /// - Better understanding of complex code patterns
    /// - Can capture more subtle relationships
    /// - Takes more time and memory to process
    ///
    /// Typical values: 6-12 layers for code models.
    /// </para>
    /// </remarks>
    public int NumEncoderLayers { get; }

    /// <summary>
    /// Gets the number of decoder layers (for generation tasks).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The number of transformer decoder layers used to generate code.
    /// Only relevant for encoder-decoder models like CodeT5.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how the model generates code.
    ///
    /// Decoder layers are used when the model needs to create new code:
    /// - For code completion
    /// - For code translation
    /// - For code generation from descriptions
    ///
    /// Not all models need decoders - some only understand code (encoders only).
    /// </para>
    /// </remarks>
    public int NumDecoderLayers { get; }

    /// <summary>
    /// Gets the number of attention heads.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The number of parallel attention mechanisms in each layer. More heads
    /// allow the model to focus on different aspects of code simultaneously.
    /// </para>
    /// <para><b>For Beginners:</b> This is how many different things the model looks at simultaneously.
    ///
    /// Multiple attention heads let the model focus on:
    /// - Variable definitions
    /// - Function calls
    /// - Control flow
    /// - Data dependencies
    /// All at the same time!
    ///
    /// Typical values: 8-16 heads.
    /// </para>
    /// </remarks>
    public int NumHeads { get; }

    /// <summary>
    /// Gets the model dimension (embedding size).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The size of the vector used to represent each token in the code.
    /// Larger dimensions can capture more information but require more memory.
    /// </para>
    /// <para><b>For Beginners:</b> This is how much information each code piece holds.
    ///
    /// Each word/token in code is represented by a vector of numbers.
    /// This dimension controls the size of that vector:
    /// - Larger: Can capture more nuanced meaning
    /// - Smaller: Faster but less detailed
    ///
    /// Typical values: 256-768 for code models.
    /// </para>
    /// </remarks>
    public int ModelDimension { get; }

    /// <summary>
    /// Gets the feed-forward network dimension.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The size of the intermediate layer in the feed-forward networks within
    /// each transformer layer. Usually 2-4 times the model dimension.
    /// </para>
    /// <para><b>For Beginners:</b> This is the processing power in each layer.
    ///
    /// After attention, each layer has a feed-forward network that processes
    /// the information. This dimension controls its size:
    /// - Larger: More processing power
    /// - Smaller: Faster but less capable
    ///
    /// Typical: 4 × ModelDimension (e.g., if ModelDim is 512, this would be 2048).
    /// </para>
    /// </remarks>
    public int FeedForwardDimension { get; }

    /// <summary>
    /// Gets the maximum sequence length (in tokens).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The maximum number of code tokens the model can process at once.
    /// Longer sequences capture more context but require more memory and computation.
    /// </para>
    /// <para><b>For Beginners:</b> This is the maximum length of code the model can handle.
    ///
    /// Code is broken into tokens (like words). This limits how many tokens:
    /// - 512 tokens: ~200-400 lines of code
    /// - 1024 tokens: ~400-800 lines of code
    /// - 2048 tokens: ~800-1600 lines of code
    ///
    /// Longer files need to be split into chunks.
    /// </para>
    /// </remarks>
    public int MaxSequenceLength { get; }

    /// <summary>
    /// Gets the vocabulary size.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The number of unique tokens (keywords, operators, identifiers, etc.) in
    /// the model's vocabulary. Larger vocabularies can represent more code patterns.
    /// </para>
    /// <para><b>For Beginners:</b> This is the model's dictionary size for code.
    ///
    /// How many different code tokens the model knows:
    /// - Keywords: if, for, while, class, etc.
    /// - Operators: +, -, ==, etc.
    /// - Common identifiers and patterns
    ///
    /// Typical values: 30,000-50,000 tokens for code models.
    /// </para>
    /// </remarks>
    public int VocabularySize { get; }

    /// <summary>
    /// Gets the dropout rate for regularization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The probability of dropping neurons during training to prevent overfitting.
    /// </para>
    /// <para><b>For Beginners:</b> This helps prevent the model from memorizing too much.
    ///
    /// Dropout randomly disables some neurons during training, which:
    /// - Prevents overfitting (memorizing training data)
    /// - Makes the model more robust
    /// - Improves generalization to new code
    ///
    /// Typical value: 0.1 (10% of neurons randomly disabled during training).
    /// </para>
    /// </remarks>
    public double DropoutRate { get; }

    /// <summary>
    /// Gets the maximum allowed program length for synthesis.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Limits the size of programs that can be synthesized, measured in
    /// abstract syntax tree nodes or lines of code.
    /// </para>
    /// <para><b>For Beginners:</b> This limits how long generated programs can be.
    ///
    /// Prevents the AI from creating huge, unwieldy programs. Like a word limit
    /// on an essay - keeps the output manageable and focused.
    /// </para>
    /// </remarks>
    public int MaxProgramLength { get; }

    /// <summary>
    /// Gets whether to use positional encoding.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Determines if positional information should be added to token embeddings
    /// to help the model understand code order and structure.
    /// </para>
    /// <para><b>For Beginners:</b> This helps the model understand code order.
    ///
    /// Without this, the model wouldn't know if "a = b" comes before or after "b = 5".
    /// Positional encoding adds location information so the model understands:
    /// - Which line comes first
    /// - How far apart two statements are
    /// - The sequential structure of code
    ///
    /// Usually set to true for code models.
    /// </para>
    /// </remarks>
    public bool UsePositionalEncoding { get; }

    /// <summary>
    /// Gets whether to use data flow information (for GraphCodeBERT-style models).
    /// </summary>
    /// <remarks>
    /// <para>
    /// If true, the model will use graph-based representations that capture
    /// data flow between variables and functions, not just sequential structure.
    /// </para>
    /// <para><b>For Beginners:</b> This makes the model understand how data flows through code.
    ///
    /// Beyond just reading code line by line, this tracks:
    /// - Which variables depend on which others
    /// - How data flows from one function to another
    /// - The relationships between different parts of code
    ///
    /// Like understanding not just the words in a recipe, but how ingredients
    /// flow from one step to the next. Used in GraphCodeBERT models.
    /// </para>
    /// </remarks>
    public bool UseDataFlow { get; }

    /// <summary>
    /// Gets the code task type this architecture is optimized for.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Specifies the primary task this model will perform, which affects the
    /// model structure and training approach.
    /// </para>
    /// <para><b>For Beginners:</b> This is the main job the model will do.
    ///
    /// Code models can do many things:
    /// - Complete code as you type
    /// - Find bugs
    /// - Translate between languages
    /// - Generate documentation
    ///
    /// This setting optimizes the model for one specific task.
    /// </para>
    /// </remarks>
    public CodeTask CodeTaskType { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="CodeSynthesisArchitecture{T}"/> class.
    /// </summary>
    /// <param name="synthesisType">The type of synthesis approach.</param>
    /// <param name="targetLanguage">The target programming language.</param>
    /// <param name="codeTaskType">The primary code task type.</param>
    /// <param name="numEncoderLayers">Number of encoder layers.</param>
    /// <param name="numDecoderLayers">Number of decoder layers.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="modelDimension">Size of token embeddings.</param>
    /// <param name="feedForwardDimension">Size of feed-forward layers.</param>
    /// <param name="maxSequenceLength">Maximum input sequence length.</param>
    /// <param name="vocabularySize">Size of the code vocabulary.</param>
    /// <param name="maxProgramLength">Maximum length of synthesized programs.</param>
    /// <param name="dropoutRate">Dropout rate for regularization.</param>
    /// <param name="usePositionalEncoding">Whether to use positional encoding.</param>
    /// <param name="useDataFlow">Whether to use data flow analysis.</param>
    /// <param name="complexity">Overall network complexity.</param>
    /// <param name="inputSize">Input size (calculated from vocabulary).</param>
    /// <param name="outputSize">Output size (calculated from task).</param>
    /// <param name="layers">Optional custom layers.</param>
    /// <remarks>
    /// <para>
    /// Creates a new code synthesis architecture with the specified parameters.
    /// This configuration will be used to build code understanding and generation models.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor sets up all the parameters for a code model.
    ///
    /// When creating a code model, you specify:
    /// - What approach to use (neural, symbolic, etc.)
    /// - Which language to work with
    /// - What task to perform
    /// - How big and powerful the model should be
    ///
    /// Many parameters have sensible defaults, so you only need to set the ones
    /// that matter for your specific use case.
    /// </para>
    /// </remarks>
    public CodeSynthesisArchitecture(
        SynthesisType synthesisType,
        ProgramLanguage targetLanguage,
        CodeTask codeTaskType,
        int numEncoderLayers = 6,
        int numDecoderLayers = 0,
        int numHeads = 8,
        int modelDimension = 512,
        int feedForwardDimension = 2048,
        int maxSequenceLength = 512,
        int vocabularySize = 50000,
        int maxProgramLength = 100,
        double dropoutRate = 0.1,
        bool usePositionalEncoding = true,
        bool useDataFlow = false,
        NetworkComplexity complexity = NetworkComplexity.Medium,
        int inputSize = 0,
        int outputSize = 0,
        List<ILayer<T>>? layers = null)
        : base(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.SequenceToSequence,
            complexity: complexity,
            inputSize: inputSize > 0 ? inputSize : vocabularySize,
            outputSize: outputSize > 0 ? outputSize : vocabularySize,
            layers: layers)
    {
        SynthesisType = synthesisType;
        TargetLanguage = targetLanguage;
        CodeTaskType = codeTaskType;
        NumEncoderLayers = numEncoderLayers;
        NumDecoderLayers = numDecoderLayers;
        NumHeads = numHeads;
        ModelDimension = modelDimension;
        FeedForwardDimension = feedForwardDimension;
        MaxSequenceLength = maxSequenceLength;
        VocabularySize = vocabularySize;
        MaxProgramLength = maxProgramLength;
        DropoutRate = dropoutRate;
        UsePositionalEncoding = usePositionalEncoding;
        UseDataFlow = useDataFlow;
    }
}

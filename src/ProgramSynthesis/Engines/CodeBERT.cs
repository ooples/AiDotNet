using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Models;
using AiDotNet.Tokenization.Interfaces;

namespace AiDotNet.ProgramSynthesis.Engines;

/// <summary>
/// CodeBERT is a bimodal pre-trained model for programming and natural languages.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// CodeBERT is designed to understand both code and natural language. It uses a
/// transformer-based encoder architecture pre-trained on code-documentation pairs
/// from GitHub. It excels at tasks like code search, code documentation generation,
/// and code completion.
/// </para>
/// <para><b>For Beginners:</b> CodeBERT is an AI that understands programming languages.
///
/// Just like BERT understands English, CodeBERT understands code. It's been trained
/// on millions of code examples from GitHub and can:
/// - Understand what code does
/// - Find similar code
/// - Complete code as you write
/// - Generate documentation
/// - Translate between code and descriptions
///
/// Think of it as an AI that's read millions of lines of code and learned the
/// patterns of good programming, just like you learn language by reading books.
/// </para>
/// </remarks>
public class CodeBERT<T> : CodeModelBase<T>
{
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <summary>
    /// Initializes a new instance of the <see cref="CodeBERT{T}"/> class.
    /// </summary>
    /// <param name="architecture">The architecture configuration for the model.</param>
    /// <param name="lossFunction">Optional loss function (defaults to cross-entropy for code tasks).</param>
    /// <param name="optimizer">Optional optimizer (defaults to Adam optimizer).</param>
    /// <param name="tokenizer">Optional tokenizer (defaults to a safe built-in tokenizer).</param>
    /// <remarks>
    /// <para>
    /// Creates a new CodeBERT model with the specified architecture. The model will
    /// be initialized with encoder layers suitable for code understanding tasks.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new CodeBERT model.
    ///
    /// You provide:
    /// - Architecture: The blueprint (size, layers, etc.)
    /// - Loss function: How to measure mistakes (optional)
    /// - Optimizer: How to improve from mistakes (optional)
    /// - Tokenizer: How to convert code into tokens (optional)
    ///
    /// Like setting up a new student with a curriculum and teaching method.
    /// </para>
    /// </remarks>
    public CodeBERT(
        CodeSynthesisArchitecture<T> architecture,
        ILossFunction<T>? lossFunction = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ITokenizer? tokenizer = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), tokenizer)
    {
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        InitializeLayersCore();
    }

    /// <summary>
    /// Initializes the layers of the CodeBERT model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Sets up the encoder layers including embeddings, positional encoding,
    /// multi-head attention, and feed-forward networks based on the architecture.
    /// </para>
    /// <para><b>For Beginners:</b> This builds the internal structure of CodeBERT.
    ///
    /// Creates all the layers that process code:
    /// - Embedding layer: Converts code tokens to numbers
    /// - Attention layers: Let the model focus on important parts
    /// - Processing layers: Transform and analyze the code
    ///
    /// Like assembling the components of a machine according to the blueprint.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        InitializeLayersCore();
    }

    private void InitializeLayersCore()
    {
        if (Layers.Count > 0)
        {
            return;
        }

        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
            return;
        }

        CodeTransformerLayerFactory.AddEmbeddingAndPosition(Layers, CodeArchitecture);
        CodeTransformerLayerFactory.AddEncoderBlocks(Layers, CodeArchitecture);
        CodeTransformerLayerFactory.AddOutputProjection(Layers, CodeArchitecture);
    }

    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        TrainWithOptimizer(input, expectedOutput, _optimizer);
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return CreateTransformerModelMetadata(
            modelName: "CodeBERT",
            extraInfo: null,
            optimizerName: _optimizer.GetType().Name);
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        CodeModelArchitectureSerialization.Write(
            writer,
            CodeArchitecture,
            includeUseDataFlow: false,
            includeEncoderDecoderLayerCounts: false);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        CodeModelArchitectureSerialization.ReadAndValidate(
            reader,
            CodeArchitecture,
            modelName: "CodeBERT",
            includeUseDataFlow: false,
            includeEncoderDecoderLayerCounts: false);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new CodeBERT<T>(CodeArchitecture, LossFunction, optimizer: null, tokenizer: Tokenizer);
    }
}

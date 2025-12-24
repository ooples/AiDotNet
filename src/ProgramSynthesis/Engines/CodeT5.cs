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
/// CodeT5 is an encoder-decoder model for code understanding and generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// CodeT5 is based on the T5 (Text-To-Text Transfer Transformer) architecture adapted
/// for code. It uses an encoder-decoder structure that can handle both code understanding
/// and generation tasks. It's particularly effective for code translation, summarization,
/// and generation from natural language descriptions.
/// </para>
/// <para><b>For Beginners:</b> CodeT5 can both understand AND generate code.
///
/// Unlike CodeBERT which mainly understands code, CodeT5 can also create it:
/// - Understand: Read and analyze code (encoder)
/// - Generate: Write new code (decoder)
///
/// This makes it powerful for tasks like:
/// - Translating Python to Java
/// - Generating code from English descriptions
/// - Creating documentation from code
/// - Fixing bugs by rewriting code
///
/// Think of it as both a reader and a writer, not just a reader.
/// </para>
/// </remarks>
public class CodeT5<T> : CodeModelBase<T>
{
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <summary>
    /// Gets the number of encoder layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The encoder processes and understands the input code or text.
    /// </para>
    /// <para><b>For Beginners:</b> Encoder layers read and understand the input.
    ///
    /// These layers analyze and comprehend what you give the model,
    /// like reading comprehension in school.
    /// </para>
    /// </remarks>
    public int NumEncoderLayers => CodeArchitecture.NumEncoderLayers;

    /// <summary>
    /// Gets the number of decoder layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The decoder generates the output code based on the encoder's understanding.
    /// </para>
    /// <para><b>For Beginners:</b> Decoder layers write the output.
    ///
    /// After understanding the input (encoder), these layers generate
    /// the response, like writing an essay based on your understanding.
    /// </para>
    /// </remarks>
    public int NumDecoderLayers => CodeArchitecture.NumDecoderLayers;

    /// <summary>
    /// Initializes a new instance of the <see cref="CodeT5{T}"/> class.
    /// </summary>
    /// <param name="architecture">The architecture configuration.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <param name="optimizer">Optional optimizer.</param>
    /// <param name="tokenizer">Optional tokenizer (defaults to a safe built-in tokenizer).</param>
    /// <remarks>
    /// <para>
    /// Creates a new CodeT5 model with encoder-decoder architecture. The model
    /// can both understand existing code and generate new code.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new CodeT5 model.
    ///
    /// CodeT5 needs both encoder and decoder layers, so make sure your
    /// architecture specifies both (NumEncoderLayers and NumDecoderLayers).
    /// </para>
    /// </remarks>
    public CodeT5(
        CodeSynthesisArchitecture<T> architecture,
        ILossFunction<T>? lossFunction = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ITokenizer? tokenizer = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), tokenizer)
    {
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        InitializeLayersCore();
    }

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
        CodeTransformerLayerFactory.AddDecoderBlocks(Layers, CodeArchitecture);
        CodeTransformerLayerFactory.AddOutputProjection(Layers, CodeArchitecture);
    }

    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        TrainWithOptimizer(input, expectedOutput, _optimizer);
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return CreateTransformerModelMetadata(
            modelName: "CodeT5",
            extraInfo: new Dictionary<string, object>
            {
                { "NumDecoderLayers", CodeArchitecture.NumDecoderLayers }
            },
            optimizerName: _optimizer.GetType().Name);
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        CodeModelArchitectureSerialization.Write(
            writer,
            CodeArchitecture,
            includeUseDataFlow: false,
            includeEncoderDecoderLayerCounts: true);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        CodeModelArchitectureSerialization.ReadAndValidate(
            reader,
            CodeArchitecture,
            modelName: "CodeT5",
            includeUseDataFlow: false,
            includeEncoderDecoderLayerCounts: true);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new CodeT5<T>(CodeArchitecture, LossFunction, optimizer: null, tokenizer: Tokenizer);
    }
}

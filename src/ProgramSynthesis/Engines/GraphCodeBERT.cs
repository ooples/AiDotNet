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
/// GraphCodeBERT extends CodeBERT by incorporating data flow analysis.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// GraphCodeBERT combines source code with data flow information to better understand
/// code semantics. It uses graph neural networks to model the relationships between
/// variables, functions, and data dependencies in code.
/// </para>
/// <para><b>For Beginners:</b> GraphCodeBERT understands how data flows through code.
///
/// While CodeBERT reads code like text, GraphCodeBERT also understands:
/// - Which variables depend on which others
/// - How data flows from one function to another
/// - The relationships and connections in code structure
///
/// Think of it like understanding a city:
/// - CodeBERT sees the streets and buildings (structure)
/// - GraphCodeBERT also sees how traffic flows and which roads connect (data flow)
///
/// This deeper understanding helps with tasks like bug detection and code optimization.
/// </para>
/// </remarks>
public class GraphCodeBERT<T> : CodeModelBase<T>
{
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <summary>
    /// Gets whether this model uses data flow analysis.
    /// </summary>
    /// <remarks>
    /// <para>
    /// GraphCodeBERT's key differentiator is its use of data flow graphs to
    /// understand code beyond just sequential structure.
    /// </para>
    /// <para><b>For Beginners:</b> This shows whether the model tracks how data moves.
    ///
    /// When true, the model doesn't just read code line by line - it builds a map
    /// of how data flows between different parts of the code, giving deeper understanding.
    /// </para>
    /// </remarks>
    public bool UsesDataFlow => CodeArchitecture.UseDataFlow;

    /// <summary>
    /// Initializes a new instance of the <see cref="GraphCodeBERT{T}"/> class.
    /// </summary>
    /// <param name="architecture">The architecture configuration (should have UseDataFlow=true).</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <param name="optimizer">Optional optimizer.</param>
    /// <param name="tokenizer">Optional tokenizer (defaults to a safe built-in tokenizer).</param>
    /// <remarks>
    /// <para>
    /// Creates a new GraphCodeBERT model with data flow analysis capabilities.
    /// The architecture should have UseDataFlow set to true to enable graph-based processing.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new GraphCodeBERT model.
    ///
    /// Similar to CodeBERT, but with extra capabilities to understand data flow.
    /// Make sure the architecture has UseDataFlow enabled to get the full benefit.
    /// </para>
    /// </remarks>
    public GraphCodeBERT(
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
        CodeTransformerLayerFactory.AddGraphConvolutionIfEnabled(Layers, CodeArchitecture);
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
            modelName: "GraphCodeBERT",
            extraInfo: new Dictionary<string, object>
            {
                { "UseDataFlow", CodeArchitecture.UseDataFlow }
            },
            optimizerName: _optimizer.GetType().Name);
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        CodeModelArchitectureSerialization.Write(
            writer,
            CodeArchitecture,
            includeUseDataFlow: true,
            includeEncoderDecoderLayerCounts: false);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        CodeModelArchitectureSerialization.ReadAndValidate(
            reader,
            CodeArchitecture,
            modelName: "GraphCodeBERT",
            includeUseDataFlow: true,
            includeEncoderDecoderLayerCounts: false);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new GraphCodeBERT<T>(CodeArchitecture, LossFunction, optimizer: null, tokenizer: Tokenizer);
    }
}

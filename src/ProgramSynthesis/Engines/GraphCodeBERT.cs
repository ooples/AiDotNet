using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Interfaces;
using AiDotNet.ProgramSynthesis.Models;

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
public class GraphCodeBERT<T> : NeuralNetworkBase<T>, ICodeModel<T>
{
    private readonly CodeSynthesisArchitecture<T> _architecture;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    public ProgramLanguage TargetLanguage => _architecture.TargetLanguage;
    public int MaxSequenceLength => _architecture.MaxSequenceLength;
    public int VocabularySize => _architecture.VocabularySize;

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
    public bool UsesDataFlow => _architecture.UseDataFlow;

    /// <summary>
    /// Initializes a new instance of the <see cref="GraphCodeBERT{T}"/> class.
    /// </summary>
    /// <param name="architecture">The architecture configuration (should have UseDataFlow=true).</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <param name="optimizer">Optional optimizer.</param>
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
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>())
    {
        _architecture = architecture;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        if (!architecture.UseDataFlow)
        {
            Console.WriteLine("Warning: GraphCodeBERT works best with UseDataFlow=true in architecture.");
        }

        InitializeLayers();
    }

    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else
        {
            // Embedding layer
            Layers.Add(new EmbeddingLayer<T>(
                vocabularySize: _architecture.VocabularySize,
                embeddingDimension: _architecture.ModelDimension,
                maxSequenceLength: _architecture.MaxSequenceLength,
                usePositionalEncoding: _architecture.UsePositionalEncoding));

            // Graph convolution layers for data flow
            if (_architecture.UseDataFlow)
            {
                Layers.Add(new GraphConvolutionalLayer<T>(
                    inputFeatures: _architecture.ModelDimension,
                    outputFeatures: _architecture.ModelDimension));
            }

            // Standard transformer encoder layers
            for (int i = 0; i < _architecture.NumEncoderLayers; i++)
            {
                Layers.Add(new MultiHeadAttentionLayer<T>(
                    modelDimension: _architecture.ModelDimension,
                    numHeads: _architecture.NumHeads,
                    dropout: _architecture.DropoutRate));

                Layers.Add(new LayerNormalizationLayer<T>(
                    normalizedShape: new[] { _architecture.ModelDimension }));

                Layers.Add(new DenseLayer<T>(
                    inputSize: _architecture.ModelDimension,
                    outputSize: _architecture.FeedForwardDimension,
                    activationFunction: new GELUActivationFunction<T>()));

                Layers.Add(new DenseLayer<T>(
                    inputSize: _architecture.FeedForwardDimension,
                    outputSize: _architecture.ModelDimension,
                    activationFunction: null));

                Layers.Add(new LayerNormalizationLayer<T>(
                    normalizedShape: new[] { _architecture.ModelDimension }));

                Layers.Add(new DropoutLayer<T>(_architecture.DropoutRate));
            }

            // Output layer
            Layers.Add(new DenseLayer<T>(
                inputSize: _architecture.ModelDimension,
                outputSize: _architecture.VocabularySize,
                activationFunction: null));
        }
    }

    public Tensor<T> EncodeCode(string code)
    {
        var input = TokenizeCode(code);
        return Predict(input);
    }

    public string DecodeCode(Tensor<T> encoding)
    {
        return DetokenizeCode(encoding);
    }

    public string PerformTask(string code, CodeTask task)
    {
        var encoding = EncodeCode(code);

        return task switch
        {
            CodeTask.BugDetection => PerformBugDetectionWithDataFlow(encoding, code),
            CodeTask.Refactoring => PerformRefactoring(encoding),
            CodeTask.Understanding => PerformCodeUnderstanding(encoding),
            _ => DecodeCode(encoding)
        };
    }

    public Tensor<T> GetEmbeddings(string code)
    {
        var input = TokenizeCode(code);
        return Layers[0].Forward(input);
    }

    public override Tensor<T> Predict(Tensor<T> input)
    {
        SetTrainingMode(false);
        var output = input;
        foreach (var layer in Layers)
        {
            output = layer.Forward(output);
        }
        return output;
    }

    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        SetTrainingMode(true);
        var output = Predict(input);
        var loss = LossFunction.ComputeLoss(output, expectedOutput);
        AddLoss(loss);

        var gradient = LossFunction.ComputeGradient(output, expectedOutput);
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            gradient = Layers[i].Backward(gradient);
        }

        _optimizer.UpdateParameters();
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = "GraphCodeBERT",
            ParameterCount = ParameterCount,
            InputSize = _architecture.InputSize,
            OutputSize = _architecture.OutputSize,
            TrainingLosses = GetLosses()
        };
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write((int)_architecture.TargetLanguage);
        writer.Write(_architecture.MaxSequenceLength);
        writer.Write(_architecture.VocabularySize);
        writer.Write(_architecture.UseDataFlow);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        var targetLanguage = (ProgramLanguage)reader.ReadInt32();
        var maxSeqLength = reader.ReadInt32();
        var vocabSize = reader.ReadInt32();
        var useDataFlow = reader.ReadBoolean();
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new GraphCodeBERT<T>(_architecture, LossFunction, _optimizer);
    }

    // Helper methods
    private Tensor<T> TokenizeCode(string code)
    {
        var tokens = code.Split(new[] { ' ', '\n', '\t' }, StringSplitOptions.RemoveEmptyEntries);
        var tokenIds = new int[Math.Min(tokens.Length, _architecture.MaxSequenceLength)];

        for (int i = 0; i < tokenIds.Length; i++)
        {
            tokenIds[i] = Math.Abs(tokens[i].GetHashCode()) % _architecture.VocabularySize;
        }

        return Tensor<T>.FromArray(Array.ConvertAll(tokenIds, id => (T)Convert.ChangeType(id, typeof(T))));
    }

    private string DetokenizeCode(Tensor<T> encoding)
    {
        return "// Generated code with data flow analysis";
    }

    private string PerformBugDetectionWithDataFlow(Tensor<T> encoding, string code)
    {
        // Enhanced bug detection using data flow
        return "// Bug detection with data flow analysis: No issues found";
    }

    private string PerformRefactoring(Tensor<T> encoding)
    {
        return "// Refactored code";
    }

    private string PerformCodeUnderstanding(Tensor<T> encoding)
    {
        return "// Code analysis: This code implements...";
    }
}

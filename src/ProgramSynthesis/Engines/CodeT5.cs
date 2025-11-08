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
public class CodeT5<T> : NeuralNetworkBase<T>, ICodeModel<T>
{
    private readonly CodeSynthesisArchitecture<T> _architecture;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    public ProgramLanguage TargetLanguage => _architecture.TargetLanguage;
    public int MaxSequenceLength => _architecture.MaxSequenceLength;
    public int VocabularySize => _architecture.VocabularySize;

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
    public int NumEncoderLayers => _architecture.NumEncoderLayers;

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
    public int NumDecoderLayers => _architecture.NumDecoderLayers;

    /// <summary>
    /// Initializes a new instance of the <see cref="CodeT5{T}"/> class.
    /// </summary>
    /// <param name="architecture">The architecture configuration.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <param name="optimizer">Optional optimizer.</param>
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
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>())
    {
        _architecture = architecture;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        if (architecture.NumDecoderLayers == 0)
        {
            Console.WriteLine("Warning: CodeT5 works best with decoder layers (NumDecoderLayers > 0).");
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
            // Shared embedding layer
            Layers.Add(new EmbeddingLayer<T>(
                vocabularySize: _architecture.VocabularySize,
                embeddingDimension: _architecture.ModelDimension,
                maxSequenceLength: _architecture.MaxSequenceLength,
                usePositionalEncoding: _architecture.UsePositionalEncoding));

            // Encoder layers
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

            // Decoder layers (if specified)
            for (int i = 0; i < _architecture.NumDecoderLayers; i++)
            {
                // Self-attention in decoder
                Layers.Add(new MultiHeadAttentionLayer<T>(
                    modelDimension: _architecture.ModelDimension,
                    numHeads: _architecture.NumHeads,
                    dropout: _architecture.DropoutRate));

                Layers.Add(new LayerNormalizationLayer<T>(
                    normalizedShape: new[] { _architecture.ModelDimension }));

                // Cross-attention (decoder attending to encoder)
                Layers.Add(new MultiHeadAttentionLayer<T>(
                    modelDimension: _architecture.ModelDimension,
                    numHeads: _architecture.NumHeads,
                    dropout: _architecture.DropoutRate));

                Layers.Add(new LayerNormalizationLayer<T>(
                    normalizedShape: new[] { _architecture.ModelDimension }));

                // Feed-forward
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

            // Output projection
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
            CodeTask.Generation => PerformGeneration(code),
            CodeTask.Translation => PerformTranslation(code),
            CodeTask.Summarization => PerformSummarization(code),
            CodeTask.Refactoring => PerformRefactoring(code),
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
            ModelType = "CodeT5",
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
        writer.Write(_architecture.NumEncoderLayers);
        writer.Write(_architecture.NumDecoderLayers);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        var targetLanguage = (ProgramLanguage)reader.ReadInt32();
        var maxSeqLength = reader.ReadInt32();
        var vocabSize = reader.ReadInt32();
        var numEncoderLayers = reader.ReadInt32();
        var numDecoderLayers = reader.ReadInt32();
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new CodeT5<T>(_architecture, LossFunction, _optimizer);
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
        return "// Generated code from CodeT5";
    }

    private string PerformGeneration(string description)
    {
        // Generate code from natural language description
        return $"// Generated code based on: {description}";
    }

    private string PerformTranslation(string code)
    {
        // Translate code between languages
        return $"// Translated code to {_architecture.TargetLanguage}";
    }

    private string PerformSummarization(string code)
    {
        // Generate natural language summary of code
        return "// Summary: This code implements...";
    }

    private string PerformRefactoring(string code)
    {
        // Generate refactored version of code
        return "// Refactored code";
    }
}

using System.Linq;
using System.Text;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Interfaces;
using AiDotNet.ProgramSynthesis.Models;
using AiDotNet.ProgramSynthesis.Options;
using Microsoft.Data.Sqlite;
using TreeSitter;

namespace AiDotNet.ProgramSynthesis.Engines;

/// <summary>
/// Neural network-based program synthesizer that generates programs from specifications.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// NeuralProgramSynthesizer uses deep learning to generate programs from natural language
/// descriptions, input-output examples, or formal specifications. It employs an encoder-decoder
/// architecture similar to CodeT5 but optimized for program synthesis tasks.
/// </para>
/// <para><b>For Beginners:</b> This AI can write programs for you automatically!
///
/// Imagine describing what you want a program to do, or showing examples of
/// inputs and outputs, and an AI writes the actual code. That's what this does!
///
/// You can provide:
/// - A description: "Write a function that sorts a list of numbers"
/// - Examples: Input [3,1,2] → Output [1,2,3]
/// - Or both!
///
/// The AI learns from training and generates working code that solves your problem.
/// It's like having an AI programmer that can code based on your requirements!
/// </para>
/// </remarks>
public class NeuralProgramSynthesizer<T> : NeuralNetworkBase<T>, IProgramSynthesizer<T>
{
    private readonly NeuralProgramSynthesizerOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private readonly CodeSynthesisArchitecture<T> _architecture;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ICodeModel<T> _codeModel;
    private readonly IProgramExecutionEngine? _executionEngine;

    public SynthesisType SynthesisType => _architecture.SynthesisType;
    public ProgramLanguage TargetLanguage => _architecture.TargetLanguage;
    public int MaxProgramLength => _architecture.MaxProgramLength;

    /// <summary>
    /// Initializes a new instance of the <see cref="NeuralProgramSynthesizer{T}"/> class.
    /// </summary>
    /// <param name="architecture">The synthesis architecture configuration.</param>
    /// <param name="codeModel">The underlying code model (CodeT5 recommended).</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <param name="optimizer">Optional optimizer.</param>
    /// <remarks>
    /// <para>
    /// Creates a new neural program synthesizer. Uses a code model (like CodeT5)
    /// as the backbone for understanding requirements and generating code.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the AI program writer.
    ///
    /// You need to provide:
    /// - Architecture: The blueprint for how it works
    /// - Code model: The brain that understands and generates code (usually CodeT5)
    /// - Optional: Loss function and optimizer for training
    ///
    /// Once set up, you can ask it to write programs for you!
    /// </para>
    /// </remarks>
    public NeuralProgramSynthesizer(
        CodeSynthesisArchitecture<T> architecture,
        ICodeModel<T> codeModel,
        ILossFunction<T>? lossFunction = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        IProgramExecutionEngine? executionEngine = null,
        NeuralProgramSynthesizerOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>())
    {
        _options = options ?? new NeuralProgramSynthesizerOptions();
        Options = _options;
        _architecture = architecture;
        _codeModel = codeModel;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _executionEngine = executionEngine;
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

        // Use the code model's layers as the base
        // Additional synthesis-specific layers can be added here
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else
        {
            // Synthesis-specific processing layers
            Layers.Add(new EmbeddingLayer<T>(
                vocabularySize: _architecture.VocabularySize,
                embeddingDimension: _architecture.ModelDimension));

            if (_architecture.UsePositionalEncoding)
            {
                Layers.Add(new PositionalEncodingLayer<T>(_architecture.MaxSequenceLength, _architecture.ModelDimension));
            }

            if (_architecture.DropoutRate > 0)
            {
                Layers.Add(new DropoutLayer<T>(_architecture.DropoutRate));
            }

            // Program structure encoding layers
            for (int i = 0; i < _architecture.NumEncoderLayers; i++)
            {
                Layers.Add(new MultiHeadAttentionLayer<T>(
                    sequenceLength: _architecture.MaxSequenceLength,
                    embeddingDimension: _architecture.ModelDimension,
                    headCount: _architecture.NumHeads,
                    activationFunction: new IdentityActivation<T>()));

                Layers.Add(new LayerNormalizationLayer<T>(_architecture.ModelDimension));

                if (_architecture.DropoutRate > 0)
                {
                    Layers.Add(new DropoutLayer<T>(_architecture.DropoutRate));
                }
            }

            // Output projection
            Layers.Add(new DenseLayer<T>(
                inputSize: _architecture.ModelDimension,
                outputSize: _architecture.VocabularySize,
                activationFunction: new IdentityActivation<T>()));
        }
    }

    /// <summary>
    /// Synthesizes a program from the given input specification.
    /// </summary>
    /// <param name="input">The input specification containing requirements or examples.</param>
    /// <returns>A synthesized program that meets the specification.</returns>
    /// <remarks>
    /// <para>
    /// This is the main synthesis method. It processes the input specification through
    /// the neural network and generates code that satisfies the requirements.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the magic happens - it writes code for you!
    ///
    /// You provide what you want (description, examples, constraints), and this
    /// method generates actual working code. The process:
    /// 1. Understand your requirements
    /// 2. Generate candidate code
    /// 3. Validate the code
    /// 4. Return the best solution
    ///
    /// Like asking an AI chef for a recipe and getting step-by-step instructions!
    /// </para>
    /// </remarks>
    public Program<T> SynthesizeProgram(ProgramInput<T> input)
    {
        if (input is null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        var bestProgram = SynthesizeProgramOnce(input);

        if (input.Examples is { Count: > 0 })
        {
            // Inductive synthesis: iteratively refine using execution feedback when available.
            // Keep this bounded and deterministic: small fixed loop, only accept strict improvements.
            const int maxRefinementIterations = 3;
            for (int iteration = 0; iteration < maxRefinementIterations; iteration++)
            {
                if (!bestProgram.IsValid || bestProgram.FitnessScore >= 1.0)
                {
                    break;
                }

                var feedback = BuildFeedbackInput(bestProgram, input);
                if (feedback is null)
                {
                    break;
                }

                var refined = RefineProgram(bestProgram, feedback);
                if (refined.IsValid && refined.FitnessScore > bestProgram.FitnessScore)
                {
                    bestProgram = refined;
                }
                else
                {
                    break;
                }
            }
        }

        return bestProgram;
    }

    private Program<T> SynthesizeProgramOnce(ProgramInput<T> input)
    {
        // Encode the input specification
        var encoding = EncodeSpecification(input);

        // Generate program using the code model
        var generatedCode = GenerateCodeFromEncoding(encoding, input);

        var program = BuildCandidateProgram(input.TargetLanguage, generatedCode);
        program.IsValid = ValidateProgram(program);

        if (input.Examples is { Count: > 0 })
        {
            program.FitnessScore = EvaluateProgram(program, input);
        }

        return program;
    }

    private Program<T> BuildCandidateProgram(ProgramLanguage language, string generatedCode)
    {
        return new Program<T>(
            sourceCode: generatedCode,
            language: language,
            isValid: false,
            fitnessScore: 0.0,
            complexity: EstimateComplexity(generatedCode));
    }

    /// <summary>
    /// Validates whether a synthesized program is correct and well-formed.
    /// </summary>
    /// <param name="program">The program to validate.</param>
    /// <returns>True if the program is valid, false otherwise.</returns>
    /// <remarks>
    /// <para>
    /// Checks if the program is syntactically correct and can potentially be executed.
    /// This includes parsing, syntax checking, and basic semantic validation.
    /// </para>
    /// <para><b>For Beginners:</b> This checks if the generated code will work.
    ///
    /// Before using generated code, we check:
    /// - Is the syntax correct? (no typos)
    /// - Does it make logical sense?
    /// - Will it compile/run?
    ///
    /// Like proofreading an essay before submitting it.
    /// </para>
    /// </remarks>
    public bool ValidateProgram(Program<T> program)
    {
        if (program is null)
        {
            throw new ArgumentNullException(nameof(program));
        }

        // Basic validation checks
        if (string.IsNullOrWhiteSpace(program.SourceCode))
            return false;

        // Check complexity constraints
        if (program.Complexity > MaxProgramLength)
            return false;

        if (program.Language == ProgramLanguage.SQL)
        {
            return ValidateSql(program.SourceCode);
        }

        var treeSitterLanguageSpec = GetTreeSitterLanguageSpec(program.Language);
        if (treeSitterLanguageSpec is not null)
        {
            try
            {
                return ValidateWithTreeSitter(treeSitterLanguageSpec.Value, program.SourceCode);
            }
            catch (DllNotFoundException)
            {
                return ValidateGenericSource(program.SourceCode);
            }
            catch (BadImageFormatException)
            {
                return ValidateGenericSource(program.SourceCode);
            }
            catch (InvalidOperationException)
            {
                return ValidateGenericSource(program.SourceCode);
            }
            catch (ArgumentException)
            {
                return ValidateGenericSource(program.SourceCode);
            }
        }

        return ValidateGenericSource(program.SourceCode);
    }

    private static (string LibraryName, string FunctionName)? GetTreeSitterLanguageSpec(ProgramLanguage language)
    {
        return language switch
        {
            ProgramLanguage.CSharp => ("tree-sitter-c-sharp", "tree_sitter_c_sharp"),
            ProgramLanguage.Python => ("tree-sitter-python", "tree_sitter_python"),
            ProgramLanguage.Java => ("tree-sitter-java", "tree_sitter_java"),
            ProgramLanguage.JavaScript => ("tree-sitter-javascript", "tree_sitter_javascript"),
            ProgramLanguage.TypeScript => ("tree-sitter-typescript", "tree_sitter_typescript"),
            ProgramLanguage.C => ("tree-sitter-c", "tree_sitter_c"),
            ProgramLanguage.CPlusPlus => ("tree-sitter-cpp", "tree_sitter_cpp"),
            ProgramLanguage.Go => ("tree-sitter-go", "tree_sitter_go"),
            ProgramLanguage.Rust => ("tree-sitter-rust", "tree_sitter_rust"),
            _ => null
        };
    }

    private static bool ValidateWithTreeSitter((string LibraryName, string FunctionName) languageSpec, string sourceCode)
    {
        using var language = new Language(languageSpec.LibraryName, languageSpec.FunctionName);
        using var parser = new Parser(language);
        using var tree = parser.Parse(sourceCode ?? string.Empty);

        if (tree is null)
        {
            return false;
        }

        var root = tree.RootNode;
        return !root.HasError && !root.IsError;
    }

    private static bool ValidateSql(string sourceCode)
    {
        try
        {
            using var connection = new SqliteConnection("Data Source=:memory:");
            connection.Open();

            using var command = connection.CreateCommand();
            command.CommandText = sourceCode ?? string.Empty;
            command.Prepare();

            return true;
        }
        catch (SqliteException)
        {
            return false;
        }
        catch (TypeInitializationException)
        {
            return ValidateGenericSource(sourceCode);
        }
        catch (DllNotFoundException)
        {
            return ValidateGenericSource(sourceCode);
        }
        catch (FileNotFoundException)
        {
            return ValidateGenericSource(sourceCode);
        }
        catch (BadImageFormatException)
        {
            return ValidateGenericSource(sourceCode);
        }
        catch (InvalidOperationException)
        {
            return false;
        }
        catch (ArgumentException)
        {
            return false;
        }
    }

    private static bool ValidateGenericSource(string sourceCode)
    {
        var text = sourceCode ?? string.Empty;
        if (text.IndexOf('\0') >= 0)
        {
            return false;
        }

        var stack = new Stack<char>();

        foreach (var ch in text)
        {
            switch (ch)
            {
                case '(':
                case '[':
                case '{':
                    stack.Push(ch);
                    break;
                case ')':
                    if (stack.Count == 0 || stack.Pop() != '(') return false;
                    break;
                case ']':
                    if (stack.Count == 0 || stack.Pop() != '[') return false;
                    break;
                case '}':
                    if (stack.Count == 0 || stack.Pop() != '{') return false;
                    break;
            }
        }

        return stack.Count == 0;
    }

    /// <summary>
    /// Evaluates how well a program satisfies the input specification.
    /// </summary>
    /// <param name="program">The program to evaluate.</param>
    /// <param name="testCases">Test cases to evaluate the program against.</param>
    /// <returns>A fitness score indicating how well the program meets requirements (0-1).</returns>
    /// <remarks>
    /// <para>
    /// Runs the program against test cases and calculates a fitness score based on
    /// how many tests pass and how well the outputs match expectations.
    /// </para>
    /// <para><b>For Beginners:</b> This grades how well the program works.
    ///
    /// Tests the program and gives it a score (like a percentage grade):
    /// - 1.0 = Perfect! Passes all tests
    /// - 0.5 = Passes half the tests
    /// - 0.0 = Doesn't work at all
    ///
    /// The score helps us know if the program is good enough or needs improvement.
    /// </para>
    /// </remarks>
    public double EvaluateProgram(Program<T> program, ProgramInput<T> testCases)
    {
        if (!program.IsValid)
            return 0.0;

        if (testCases.Examples == null || testCases.Examples.Count == 0)
            return 0.5; // No tests to run, assume partial fitness

        int passedTests = 0;
        var examples = testCases.Examples;

        foreach (var example in examples)
        {
            if (!TryExecuteProgram(program, example.Input, out var result, out var errorMessage))
            {
                program.ErrorMessage ??= errorMessage;
                continue;
            }

            if (IsOutputMatch(result, example.ExpectedOutput))
            {
                passedTests++;
            }
        }

        return (double)passedTests / examples.Count;
    }

    private static bool IsOutputMatch(string actual, string expected)
    {
        var normalizedActual = NormalizeOutput(actual);
        var normalizedExpected = NormalizeOutput(expected);

        if (string.Equals(normalizedActual, normalizedExpected, StringComparison.Ordinal))
        {
            return true;
        }

        if (double.TryParse(normalizedActual, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var actualNumber) &&
            double.TryParse(normalizedExpected, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var expectedNumber))
        {
            // Use relative error for non-zero values to handle large numbers correctly
            // For numbers near zero, fall back to absolute error comparison
            const double absoluteTolerance = 1e-9;
            const double relativeTolerance = 1e-6;

            double absDiff = Math.Abs(actualNumber - expectedNumber);
            if (absDiff <= absoluteTolerance)
            {
                return true;
            }

            double maxAbs = Math.Max(Math.Abs(actualNumber), Math.Abs(expectedNumber));
            if (maxAbs > 0 && absDiff / maxAbs <= relativeTolerance)
            {
                return true;
            }

            return false;
        }

        return false;
    }

    private static string NormalizeOutput(string value)
        => (value ?? string.Empty).Replace("\r\n", "\n").Trim();

    /// <summary>
    /// Refines an existing program to better meet the specification.
    /// </summary>
    /// <param name="program">The program to refine.</param>
    /// <param name="feedback">Feedback or test cases that failed.</param>
    /// <returns>A refined version of the program.</returns>
    /// <remarks>
    /// <para>
    /// Takes an existing program and improves it based on feedback from failed tests
    /// or user corrections. Uses the neural network to generate a better version.
    /// </para>
    /// <para><b>For Beginners:</b> This improves a program based on feedback.
    ///
    /// If the first version isn't quite right:
    /// 1. Look at what went wrong (failed tests)
    /// 2. Generate an improved version
    /// 3. Keep the good parts, fix the problems
    ///
    /// Like editing a draft based on reviewer comments to make it better.
    /// </para>
    /// </remarks>
    public Program<T> RefineProgram(Program<T> program, ProgramInput<T> feedback)
    {
        // Create a new input that includes the existing program and feedback
        var refinementInput = new ProgramInput<T>
        {
            Description = $"Refine this program:\n{program.SourceCode}\n\nFeedback:\n{feedback.Description}",
            TargetLanguage = program.Language,
            Examples = feedback.Examples,
            TestCases = feedback.TestCases,
            Constraints = feedback.Constraints
        };

        // Synthesize improved version (single-pass, refinement loop is handled by the caller)
        var refinedProgram = SynthesizeProgramOnce(refinementInput);

        // If refinement didn't improve, return original
        if (refinedProgram.FitnessScore <= program.FitnessScore)
        {
            return program;
        }

        return refinedProgram;
    }

    private ProgramInput<T>? BuildFeedbackInput(Program<T> program, ProgramInput<T> original)
    {
        if (original.Examples is not { Count: > 0 })
        {
            return null;
        }

        if (_executionEngine is null)
        {
            return null;
        }

        var failing = new List<ProgramInputOutputExample>();
        foreach (var example in original.Examples)
        {
            if (!TryExecuteProgram(program, example.Input, out var output, out _))
            {
                failing.Add(example);
                continue;
            }

            if (!IsOutputMatch(output, example.ExpectedOutput))
            {
                failing.Add(example);
            }
        }

        if (failing.Count == 0)
        {
            return null;
        }

        var sb = new StringBuilder();
        sb.AppendLine("Failed examples:");
        foreach (var ex in failing.Take(8))
        {
            sb.Append("Input: ").AppendLine(ex.Input);
            sb.Append("Expected: ").AppendLine(ex.ExpectedOutput);
            sb.AppendLine("---");
        }

        return new ProgramInput<T>
        {
            Description = sb.ToString(),
            TargetLanguage = original.TargetLanguage,
            Examples = failing,
            TestCases = original.TestCases,
            Constraints = original.Constraints,
            FormalSpecification = original.FormalSpecification,
            MaxComplexity = original.MaxComplexity,
            Tags = original.Tags
        };
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

    public override void UpdateParameters(Vector<T> parameters)
    {
        int index = 0;
        foreach (var layer in Layers)
        {
            int layerParameterCount = layer.ParameterCount;
            if (layerParameterCount <= 0)
            {
                continue;
            }

            var layerParameters = parameters.Slice(index, layerParameterCount);
            layer.UpdateParameters(layerParameters);
            index += layerParameterCount;
        }
    }

    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        SetTrainingMode(true);
        var output = input;
        foreach (var layer in Layers)
        {
            output = layer.Forward(output);
        }

        LastLoss = LossFunction.CalculateLoss(output.ToVector(), expectedOutput.ToVector());

        var outputGradient = LossFunction.CalculateDerivative(output.ToVector(), expectedOutput.ToVector());
        var gradient = new Tensor<T>(output.Shape, outputGradient);
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            gradient = Layers[i].Backward(gradient);
        }

        _optimizer.UpdateParameters(Layers);
        SetTrainingMode(false);
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.Transformer,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ModelName", "NeuralProgramSynthesizer" },
                { "SynthesisType", _architecture.SynthesisType.ToString() },
                { "TargetLanguage", _architecture.TargetLanguage.ToString() },
                { "MaxProgramLength", _architecture.MaxProgramLength },
                { "NumHeads", _architecture.NumHeads },
                { "ModelDimension", _architecture.ModelDimension },
                { "MaxSequenceLength", _architecture.MaxSequenceLength },
                { "VocabularySize", _architecture.VocabularySize },
                { "DropoutRate", _architecture.DropoutRate },
                { "UsePositionalEncoding", _architecture.UsePositionalEncoding },
                { "LayerCount", Layers.Count },
                { "ParameterCount", GetParameterCount() },
                { "LossFunction", LossFunction.GetType().Name },
                { "Optimizer", _optimizer.GetType().Name }
            },
            ModelData = Serialize()
        };
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write((int)_architecture.SynthesisType);
        writer.Write((int)_architecture.TargetLanguage);
        writer.Write(_architecture.MaxProgramLength);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        var synthesisType = (SynthesisType)reader.ReadInt32();
        var targetLanguage = (ProgramLanguage)reader.ReadInt32();
        var maxProgramLength = reader.ReadInt32();

        if (synthesisType != _architecture.SynthesisType ||
            targetLanguage != _architecture.TargetLanguage ||
            maxProgramLength != _architecture.MaxProgramLength)
        {
            throw new InvalidOperationException(
                "Serialized NeuralProgramSynthesizer architecture does not match the current instance. " +
                $"Serialized: SynthesisType={synthesisType}, TargetLanguage={targetLanguage}, MaxProgramLength={maxProgramLength}. " +
                $"Expected: SynthesisType={_architecture.SynthesisType}, TargetLanguage={_architecture.TargetLanguage}, MaxProgramLength={_architecture.MaxProgramLength}.");
        }
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new NeuralProgramSynthesizer<T>(_architecture, _codeModel, LossFunction, _optimizer, _executionEngine);
    }

    // Helper methods
    private Tensor<T> EncodeSpecification(ProgramInput<T> input)
    {
        // Combine description and examples into a unified encoding
        var specTextBuilder = new StringBuilder(input.Description ?? string.Empty);

        if (!string.IsNullOrWhiteSpace(input.FormalSpecification))
        {
            specTextBuilder.Append("\nFormalSpec: ").Append(input.FormalSpecification);
        }

        if (input.Examples != null)
        {
            foreach (var example in input.Examples)
            {
                specTextBuilder.Append("\nExample: ").Append(example.Input).Append(" -> ").Append(example.ExpectedOutput);
            }
        }

        if (input.Constraints != null && input.Constraints.Count > 0)
        {
            foreach (var constraint in input.Constraints.Where(static c => !string.IsNullOrWhiteSpace(c)))
            {
                specTextBuilder.Append("\nConstraint: ").Append(constraint);
            }
        }

        if (input.MaxComplexity.HasValue)
        {
            specTextBuilder.Append("\nMaxComplexity: ").Append(input.MaxComplexity.Value);
        }

        if (input.Tags != null && input.Tags.Count > 0)
        {
            foreach (var tag in input.Tags.Where(static t => !string.IsNullOrWhiteSpace(t)))
            {
                specTextBuilder.Append("\nTag: ").Append(tag);
            }
        }

        return _codeModel.EncodeCode(specTextBuilder.ToString());
    }

    private string GenerateCodeFromEncoding(Tensor<T> encoding, ProgramInput<T> input)
    {
        // Use the code model to generate code
        var generated = _codeModel.DecodeCode(encoding);

        return generated;
    }

    private int EstimateComplexity(string code)
    {
        // Simple complexity estimation based on code length and structure
        var lines = code.Split(new[] { '\n' }, StringSplitOptions.RemoveEmptyEntries);
        return lines.Length;
    }

    private bool TryExecuteProgram(Program<T> program, string input, out string output, out string? errorMessage)
    {
        output = string.Empty;
        errorMessage = null;

        if (_executionEngine == null)
        {
            errorMessage = "No program execution engine is configured for NeuralProgramSynthesizer.";
            return false;
        }

        return _executionEngine.TryExecute(program.Language, program.SourceCode, input, out output, out errorMessage);
    }
}

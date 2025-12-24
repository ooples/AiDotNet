using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Models;

namespace AiDotNet.ProgramSynthesis.Interfaces;

/// <summary>
/// Represents a program synthesis engine capable of automatically generating programs.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// IProgramSynthesizer defines the interface for models that can automatically generate
/// programs from specifications, examples, or natural language descriptions. This is a
/// key component of automated programming and AI-assisted development.
/// </para>
/// <para><b>For Beginners:</b> A program synthesizer is like an AI programmer.
///
/// Imagine describing what you want a program to do, and an AI writes the code for you.
/// That's what a program synthesizer does. You provide:
/// - Examples of inputs and desired outputs
/// - A description in plain English
/// - Or formal specifications
///
/// And the synthesizer creates a working program that meets your requirements.
/// This is like having an AI assistant that can code for you!
/// </para>
/// </remarks>
public interface IProgramSynthesizer<T> : IFullModel<T, Tensor<T>, Tensor<T>>
{
    /// <summary>
    /// Gets the type of synthesis approach used by this synthesizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Different synthesis approaches have different strengths. Neural methods are
    /// creative, symbolic methods are precise, and hybrid methods combine both.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you how the AI generates programs.
    ///
    /// Different approaches are like different problem-solving strategies:
    /// - Neural: Learns from examples (like learning by watching)
    /// - Symbolic: Uses logic and rules (like following instructions)
    /// - Genetic: Evolves solutions (like natural selection)
    /// </para>
    /// </remarks>
    SynthesisType SynthesisType { get; }

    /// <summary>
    /// Gets the target programming language for synthesis.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Specifies which programming language the synthesized programs will be written in.
    /// </para>
    /// <para><b>For Beginners:</b> This is the language the AI will write code in.
    ///
    /// Just like you choose whether to write in English or Spanish, this specifies
    /// which programming language the generated code will use (Python, Java, etc.).
    /// </para>
    /// </remarks>
    ProgramLanguage TargetLanguage { get; }

    /// <summary>
    /// Gets the maximum allowed length for synthesized programs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This limits the complexity and size of generated programs, measured in tokens
    /// or abstract syntax tree nodes.
    /// </para>
    /// <para><b>For Beginners:</b> This limits how long/complex the generated code can be.
    ///
    /// Like a word limit on an essay, this prevents the AI from generating programs
    /// that are too large or complex. Helps ensure the code stays manageable.
    /// </para>
    /// </remarks>
    int MaxProgramLength { get; }

    /// <summary>
    /// Synthesizes a program from the given input specification.
    /// </summary>
    /// <param name="input">The input specification containing requirements or examples.</param>
    /// <returns>A synthesized program that meets the specification.</returns>
    /// <remarks>
    /// <para>
    /// This is the core synthesis method that generates a complete program from the
    /// provided input specification. The input can contain examples, natural language
    /// descriptions, or formal specifications.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the magic happens - it creates a program for you!
    ///
    /// You provide what you want (examples, description, etc.), and this method
    /// generates actual working code that does what you asked for. Like asking
    /// an AI chef for a recipe and getting step-by-step cooking instructions.
    /// </para>
    /// </remarks>
    Program<T> SynthesizeProgram(ProgramInput<T> input);

    /// <summary>
    /// Validates whether a synthesized program is correct and well-formed.
    /// </summary>
    /// <param name="program">The program to validate.</param>
    /// <returns>True if the program is valid, false otherwise.</returns>
    /// <remarks>
    /// <para>
    /// Validation checks syntactic correctness, semantic validity, and whether
    /// the program compiles or can be executed.
    /// </para>
    /// <para><b>For Beginners:</b> This checks if the generated code is valid and will work.
    ///
    /// Before using generated code, we need to check:
    /// - Is the syntax correct? (no typos or grammar errors)
    /// - Does it make sense? (logical consistency)
    /// - Will it compile/run? (can the computer execute it)
    ///
    /// Like proofreading before submitting an essay.
    /// </para>
    /// </remarks>
    bool ValidateProgram(Program<T> program);

    /// <summary>
    /// Evaluates how well a program satisfies the input specification.
    /// </summary>
    /// <param name="program">The program to evaluate.</param>
    /// <param name="testCases">Test cases to evaluate the program against.</param>
    /// <returns>A fitness score indicating how well the program meets requirements (0-1, higher is better).</returns>
    /// <remarks>
    /// <para>
    /// Evaluation tests the program against provided test cases and returns a score
    /// indicating how well it performs. This is crucial for iterative refinement.
    /// </para>
    /// <para><b>For Beginners:</b> This grades how well the generated program works.
    ///
    /// Just like a teacher grades homework, this checks how well the program solves
    /// the problem. It runs tests and gives a score (like a percentage):
    /// - 1.0 = Perfect, passes all tests
    /// - 0.5 = Passes half the tests
    /// - 0.0 = Doesn't work at all
    /// </para>
    /// </remarks>
    double EvaluateProgram(Program<T> program, ProgramInput<T> testCases);

    /// <summary>
    /// Refines an existing program to better meet the specification.
    /// </summary>
    /// <param name="program">The program to refine.</param>
    /// <param name="feedback">Feedback or test cases that failed.</param>
    /// <returns>A refined version of the program.</returns>
    /// <remarks>
    /// <para>
    /// Refinement takes an existing program and improves it based on feedback,
    /// such as failed test cases or user corrections. This enables iterative improvement.
    /// </para>
    /// <para><b>For Beginners:</b> This improves a program based on feedback.
    ///
    /// If the first version isn't quite right, this method improves it. Like editing
    /// a draft based on reviewer comments - it takes the feedback and creates a
    /// better version. Keeps the good parts and fixes the problems.
    /// </para>
    /// </remarks>
    Program<T> RefineProgram(Program<T> program, ProgramInput<T> feedback);
}

using AiDotNet.LinearAlgebra;
using AiDotNet.ProgramSynthesis.Enums;

namespace AiDotNet.ProgramSynthesis.Models;

/// <summary>
/// Represents a synthesized program with its source code and metadata.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// The Program class encapsulates a synthesized or analyzed program, including its
/// source code, the programming language it's written in, validation status, and
/// optional execution metrics.
/// </para>
/// <para><b>For Beginners:</b> This class represents a computer program created by AI.
///
/// Think of this as a container that holds:
/// - The actual code (like a recipe holds instructions)
/// - What language it's written in (Python, Java, etc.)
/// - Whether the code is valid and will run
/// - How well it performs
/// - An optional numerical representation that AI can work with
///
/// Just like a recipe card has the recipe, cooking time, and difficulty level,
/// this class holds a program and all its important information.
/// </para>
/// </remarks>
public class Program<T>
{
    /// <summary>
    /// Gets or sets the source code of the program.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The actual program text in the target programming language. This is the
    /// human-readable code that can be executed or compiled.
    /// </para>
    /// <para><b>For Beginners:</b> This is the actual code - the instructions the computer will follow.
    ///
    /// Just like a recipe has step-by-step cooking instructions, this contains
    /// the step-by-step commands that tell the computer what to do.
    /// </para>
    /// </remarks>
    public string SourceCode { get; set; }

    /// <summary>
    /// Gets or sets the programming language of the program.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Specifies which programming language the source code is written in.
    /// This affects how the code should be interpreted, compiled, or executed.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you which programming language was used.
    ///
    /// Just like knowing whether a recipe is in English or French, this tells you
    /// whether the code is in Python, Java, C#, etc. Different languages have
    /// different rules and syntax.
    /// </para>
    /// </remarks>
    public ProgramLanguage Language { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether the program is syntactically and semantically valid.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Indicates whether the program passes validation checks, including syntax
    /// correctness and semantic validity. A valid program can potentially be executed.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you if the code is correct and will run.
    ///
    /// Like checking a recipe for mistakes before cooking:
    /// - Are all ingredients listed? (syntax)
    /// - Do the instructions make sense? (semantics)
    /// - Will following this recipe actually work? (validity)
    ///
    /// If IsValid is true, the code should run without errors.
    /// </para>
    /// </remarks>
    public bool IsValid { get; set; }

    /// <summary>
    /// Gets or sets the fitness score of the program.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A value between 0 and 1 indicating how well the program satisfies the
    /// synthesis requirements. Higher values indicate better performance.
    /// 1.0 means perfect, 0.0 means complete failure.
    /// </para>
    /// <para><b>For Beginners:</b> This is like a grade showing how well the program works.
    ///
    /// Think of it as a score from 0% to 100%:
    /// - 1.0 (100%): Perfect! Passes all tests
    /// - 0.75 (75%): Pretty good, passes most tests
    /// - 0.5 (50%): Mediocre, passes half the tests
    /// - 0.0 (0%): Doesn't work at all
    ///
    /// Higher scores mean the program better solves the problem you gave it.
    /// </para>
    /// </remarks>
    public double FitnessScore { get; set; }

    /// <summary>
    /// Gets or sets the complexity measure of the program.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A metric indicating the complexity of the program, which could be based on
    /// various factors like number of statements, cyclomatic complexity, or
    /// abstract syntax tree size.
    /// </para>
    /// <para><b>For Beginners:</b> This measures how complicated the program is.
    ///
    /// Just like recipes can be simple (toast) or complex (soufflé), programs
    /// have different complexity levels. This number tells you:
    /// - Low values: Simple, short programs that are easy to understand
    /// - High values: Complex, longer programs with many steps
    ///
    /// Usually, simpler programs (lower complexity) are better when they
    /// solve the same problem.
    /// </para>
    /// </remarks>
    public int Complexity { get; set; }

    /// <summary>
    /// Gets or sets the encoded representation of the program.
    /// </summary>
    /// <remarks>
    /// <para>
    /// An optional numerical encoding of the program that can be used by neural
    /// networks for further processing or refinement.
    /// </para>
    /// <para><b>For Beginners:</b> This is a numerical version of the code for AI to work with.
    ///
    /// Computers and AI work better with numbers than text. This is the program
    /// converted into a numerical form that AI can easily process, like converting
    /// a photo into pixels. The original code is still in SourceCode - this is
    /// just an alternative representation for computation.
    /// </para>
    /// </remarks>
    public Tensor<T>? Encoding { get; set; }

    /// <summary>
    /// Gets or sets any error messages from compilation or execution attempts.
    /// </summary>
    /// <remarks>
    /// <para>
    /// If the program failed validation or execution, this contains the error
    /// messages explaining what went wrong.
    /// </para>
    /// <para><b>For Beginners:</b> This explains what's wrong if the program doesn't work.
    ///
    /// When code has problems, we need to know why. This stores error messages like:
    /// - "Syntax error on line 5: missing semicolon"
    /// - "Variable 'x' not defined"
    ///
    /// These help debug and fix the program, like having someone point out
    /// exactly what's wrong with a recipe.
    /// </para>
    /// </remarks>
    public string? ErrorMessage { get; set; }

    /// <summary>
    /// Gets or sets execution time in milliseconds if the program was executed.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Records how long the program took to execute, which can be useful for
    /// performance comparison and optimization.
    /// </para>
    /// <para><b>For Beginners:</b> This is how long the program takes to run.
    ///
    /// Measured in milliseconds (1000 milliseconds = 1 second). Helps answer:
    /// - Is this program fast or slow?
    /// - Which of two programs is faster?
    ///
    /// Lower execution time is usually better - it means the program finishes faster.
    /// </para>
    /// </remarks>
    public double? ExecutionTimeMs { get; set; }

    /// <summary>
    /// Initializes a new instance of the <see cref="Program{T}"/> class.
    /// </summary>
    /// <param name="sourceCode">The source code of the program.</param>
    /// <param name="language">The programming language.</param>
    /// <param name="isValid">Whether the program is valid.</param>
    /// <param name="fitnessScore">The fitness score (default is 0.0).</param>
    /// <param name="complexity">The complexity measure (default is 0).</param>
    /// <remarks>
    /// <para>
    /// Creates a new Program instance with the specified source code and metadata.
    /// This constructor is typically used when creating a synthesized program.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new program object.
    ///
    /// When the AI generates or processes code, it creates a Program object
    /// to store all the information. You need to provide:
    /// - The actual code (required)
    /// - What language it's in (required)
    /// - Whether it's valid (required)
    /// - Optional: fitness score and complexity
    ///
    /// Think of it like filling out a form with all the program's details.
    /// </para>
    /// </remarks>
    public Program(
        string sourceCode,
        ProgramLanguage language,
        bool isValid = false,
        double fitnessScore = 0.0,
        int complexity = 0)
    {
        SourceCode = sourceCode;
        Language = language;
        IsValid = isValid;
        FitnessScore = fitnessScore;
        Complexity = complexity;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Program{T}"/> class with default values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Creates an empty Program instance. Useful when the program will be
    /// populated later or when deserializing.
    /// </para>
    /// <para><b>For Beginners:</b> This creates an empty program instance.
    ///
    /// Sometimes you need to create a Program object before you have all the
    /// information. This creates an empty one that you can fill in later,
    /// like having a blank form to fill out gradually.
    /// </para>
    /// </remarks>
    public Program()
    {
        SourceCode = string.Empty;
        Language = ProgramLanguage.Generic;
        IsValid = false;
        FitnessScore = 0.0;
        Complexity = 0;
    }

    /// <summary>
    /// Returns a string representation of the program.
    /// </summary>
    /// <returns>A string containing the source code.</returns>
    /// <remarks>
    /// <para>
    /// Provides a string representation of the Program for display purposes.
    /// </para>
    /// <para><b>For Beginners:</b> This converts the program to a readable string.
    ///
    /// When you need to display or print the program, this method returns
    /// the source code as a string. Useful for debugging and logging.
    /// </para>
    /// </remarks>
    public override string ToString()
    {
        return $"[{Language}] Valid: {IsValid}, Fitness: {FitnessScore:F2}, Complexity: {Complexity}\n{SourceCode}";
    }
}

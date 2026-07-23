using AiDotNet.LinearAlgebra;
using AiDotNet.ProgramSynthesis.Enums;

namespace AiDotNet.ProgramSynthesis.Models;

/// <summary>
/// Represents the input specification for program synthesis.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// ProgramInput encapsulates all the information needed to synthesize a program,
/// including natural language descriptions, input-output examples, formal specifications,
/// and constraints.
/// </para>
/// <para><b>For Beginners:</b> This class describes what you want the program to do.
///
/// When you want AI to create a program for you, you need to tell it what you want.
/// This class lets you provide that information in different ways:
/// - Describe it in plain English
/// - Give examples of inputs and expected outputs
/// - Specify constraints (like "must run in under 1 second")
///
/// Think of it like ordering at a restaurant - you tell the chef what you want,
/// and they create the dish. This is how you tell the AI what program you want.
/// </para>
/// </remarks>
public class ProgramInput<T>
{
    /// <summary>
    /// Gets or sets the natural language description of the desired program.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A plain-English description of what the program should do. This can be
    /// used by neural synthesis methods to understand the user's intent.
    /// </para>
    /// <para><b>For Beginners:</b> This is where you describe what you want in plain English.
    ///
    /// Just like telling someone:
    /// "I need a function that takes a list of numbers and returns the average"
    ///
    /// No programming knowledge needed - just explain what you want the program
    /// to accomplish.
    /// </para>
    /// </remarks>
    public string? Description { get; set; }

    /// <summary>
    /// Gets or sets the target programming language for synthesis.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Specifies which programming language the synthesized program should be written in.
    /// </para>
    /// <para><b>For Beginners:</b> This is which programming language you want the code in.
    ///
    /// Like choosing whether you want instructions in English or Spanish, this
    /// tells the AI whether to generate code in Python, Java, C#, etc.
    /// </para>
    /// </remarks>
    public ProgramLanguage TargetLanguage { get; set; }

    /// <summary>
    /// Gets or sets the input-output examples for inductive synthesis.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A list of example inputs and their expected outputs. The synthesizer learns
    /// from these examples to generate a program that generalizes to new inputs.
    /// Each example includes an input and the expected output.
    /// </para>
    /// <para><b>For Beginners:</b> These are examples showing what the program should do.
    ///
    /// Instead of explaining, you can show examples:
    /// - Input: [1, 2, 3] → Output: 6 (sum)
    /// - Input: [4, 5] → Output: 9 (sum)
    /// - Input: [10] → Output: 10 (sum)
    ///
    /// The AI figures out the pattern from your examples. Like teaching by example
    /// rather than explaining - show what you want, and the AI learns the rule.
    /// </para>
    /// </remarks>
    public List<ProgramInputOutputExample>? Examples { get; set; }

    /// <summary>
    /// Gets or sets the formal specification in logic or a domain-specific language.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A formal, mathematical specification of the program's behavior. This is used
    /// by deductive synthesis methods to construct provably correct programs.
    /// </para>
    /// <para><b>For Beginners:</b> This is a precise mathematical description (advanced).
    ///
    /// This is more advanced - it's a very precise, formal way to describe what
    /// the program should do using mathematical logic. Like a detailed blueprint
    /// with exact specifications. Most users will use Description or Examples instead.
    /// </para>
    /// </remarks>
    public string? FormalSpecification { get; set; }

    /// <summary>
    /// Gets or sets constraints that the synthesized program must satisfy.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A list of constraints or requirements for the program, such as:
    /// - Performance requirements ("must run in O(n) time")
    /// - Resource limits ("must use less than 1MB memory")
    /// - Style requirements ("must use functional programming style")
    /// </para>
    /// <para><b>For Beginners:</b> These are rules the program must follow.
    ///
    /// Beyond just working correctly, you might have specific requirements:
    /// - "Must be fast"
    /// - "Should be easy to read"
    /// - "Can't use certain functions"
    ///
    /// Like telling a chef: "Make it vegetarian and gluten-free."
    /// These constraints ensure the program meets your specific needs.
    /// </para>
    /// </remarks>
    public List<string>? Constraints { get; set; }

    /// <summary>
    /// Gets or sets the maximum allowed complexity for the synthesized program.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Limits how complex the generated program can be. This helps ensure the
    /// synthesizer produces simple, understandable code when possible.
    /// </para>
    /// <para><b>For Beginners:</b> This limits how complicated the program can be.
    ///
    /// Sometimes simple is better. This sets a maximum complexity level:
    /// - Low value: Forces simple solutions
    /// - High value: Allows complex solutions if needed
    ///
    /// Like asking for a simple recipe instead of a gourmet one - both might
    /// work, but simple is often better for learning and maintaining.
    /// </para>
    /// </remarks>
    public int? MaxComplexity { get; set; }

    /// <summary>
    /// Gets or sets the timeout for program synthesis in milliseconds.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Specifies how long the synthesizer should attempt to find a solution
    /// before giving up. Prevents indefinite computation on difficult problems.
    /// </para>
    /// <para><b>For Beginners:</b> This is how long the AI has to find a solution.
    ///
    /// Measured in milliseconds (1000ms = 1 second). Sometimes finding the perfect
    /// program takes too long. This sets a time limit:
    /// - 5000ms (5 seconds): Quick attempt, might not find best solution
    /// - 60000ms (1 minute): More thorough search
    ///
    /// Like giving up on a crossword puzzle after 10 minutes - sometimes you
    /// need to move on even if you haven't finished.
    /// </para>
    /// </remarks>
    public int? TimeoutMs { get; set; }

    /// <summary>
    /// Gets or sets the test cases for program validation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Additional test cases (beyond the examples) used to validate the correctness
    /// of synthesized programs. Each test case includes an input and the expected output.
    /// </para>
    /// <para><b>For Beginners:</b> These are additional tests to verify the program works.
    ///
    /// While Examples teach the AI, TestCases verify the result:
    /// - Examples: "Learn from these"
    /// - TestCases: "Prove you got it right with these"
    ///
    /// Like the difference between practice problems and an exam - test cases
    /// help ensure the program truly works correctly.
    /// </para>
    /// </remarks>
    public List<ProgramInputOutputExample>? TestCases { get; set; }

    /// <summary>
    /// Gets or sets an encoded representation of the input for neural processing.
    /// </summary>
    /// <remarks>
    /// <para>
    /// An optional numerical encoding of the input specification that can be
    /// directly processed by neural networks.
    /// </para>
    /// <para><b>For Beginners:</b> This is a numerical version for AI processing.
    ///
    /// Neural networks work with numbers, not text. This is an optional field
    /// where the input can be pre-converted to numbers. Usually generated
    /// automatically - you don't need to provide this yourself.
    /// </para>
    /// </remarks>
    public Tensor<T>? Encoding { get; set; }

    /// <summary>
    /// Gets or sets metadata tags for categorizing or filtering synthesis tasks.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Optional tags that can be used to categorize the synthesis task, track
    /// experiments, or provide additional context to the synthesizer.
    /// </para>
    /// <para><b>For Beginners:</b> These are labels for organizing synthesis tasks.
    ///
    /// Like hashtags or folders, these help organize and categorize:
    /// - "sorting", "algorithm", "beginner"
    /// - "web-scraping", "python", "advanced"
    ///
    /// Useful for tracking different types of synthesis tasks and experiments.
    /// </para>
    /// </remarks>
    public List<string>? Tags { get; set; }

    /// <summary>
    /// Initializes a new instance of the <see cref="ProgramInput{T}"/> class.
    /// </summary>
    /// <param name="description">The natural language description.</param>
    /// <param name="targetLanguage">The target programming language.</param>
    /// <param name="examples">Optional input-output examples.</param>
    /// <param name="constraints">Optional constraints.</param>
    /// <remarks>
    /// <para>
    /// Creates a new ProgramInput with the essential information needed for synthesis.
    /// Additional properties can be set after construction.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new specification for what program you want.
    ///
    /// Provide at minimum:
    /// - A description of what you want
    /// - Which language to use
    /// - Optionally: examples and constraints
    ///
    /// Like filling out an order form for a custom program.
    /// </para>
    /// </remarks>
    public ProgramInput(
        string? description = null,
        ProgramLanguage targetLanguage = ProgramLanguage.Generic,
        List<ProgramInputOutputExample>? examples = null,
        List<string>? constraints = null)
    {
        Description = description;
        TargetLanguage = targetLanguage;
        Examples = examples;
        Constraints = constraints;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="ProgramInput{T}"/> class with default values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Creates an empty ProgramInput that can be populated later.
    /// </para>
    /// <para><b>For Beginners:</b> Creates an empty specification to fill in later.
    ///
    /// Sometimes you want to create the object first and add details later.
    /// This creates an empty form you can fill in step by step.
    /// </para>
    /// </remarks>
    public ProgramInput()
    {
        TargetLanguage = ProgramLanguage.Generic;
    }

    /// <summary>
    /// Adds an input-output example to the Examples list.
    /// </summary>
    /// <param name="input">The example input.</param>
    /// <param name="expectedOutput">The expected output for this input.</param>
    /// <remarks>
    /// <para>
    /// Convenience method to add examples one at a time instead of creating
    /// the entire list upfront.
    /// </para>
    /// <para><b>For Beginners:</b> This adds one example at a time.
    ///
    /// Instead of providing all examples at once, you can add them one by one:
    /// programInput.AddExample("[1,2,3]", "6");
    /// programInput.AddExample("[4,5]", "9");
    ///
    /// Easier than creating the list yourself.
    /// </para>
    /// </remarks>
    public void AddExample(string input, string expectedOutput)
    {
        Examples ??= new List<ProgramInputOutputExample>();
        Examples.Add(new ProgramInputOutputExample { Input = input, ExpectedOutput = expectedOutput });
    }

    /// <summary>
    /// Adds a test case to the TestCases list.
    /// </summary>
    /// <param name="input">The test input.</param>
    /// <param name="expectedOutput">The expected output for this input.</param>
    /// <remarks>
    /// <para>
    /// Convenience method to add test cases one at a time.
    /// </para>
    /// <para><b>For Beginners:</b> This adds one test case at a time.
    ///
    /// Similar to AddExample, but for test cases that verify correctness:
    /// programInput.AddTestCase("[10,20]", "30");
    /// </para>
    /// </remarks>
    public void AddTestCase(string input, string expectedOutput)
    {
        TestCases ??= new List<ProgramInputOutputExample>();
        TestCases.Add(new ProgramInputOutputExample { Input = input, ExpectedOutput = expectedOutput });
    }

    /// <summary>
    /// Adds a constraint to the Constraints list.
    /// </summary>
    /// <param name="constraint">The constraint to add.</param>
    /// <remarks>
    /// <para>
    /// Convenience method to add constraints one at a time.
    /// </para>
    /// <para><b>For Beginners:</b> This adds one constraint at a time.
    ///
    /// Add requirements one by one:
    /// programInput.AddConstraint("Must run in O(n) time");
    /// programInput.AddConstraint("Should not use recursion");
    /// </para>
    /// </remarks>
    public void AddConstraint(string constraint)
    {
        Constraints ??= new List<string>();
        Constraints.Add(constraint);
    }
}

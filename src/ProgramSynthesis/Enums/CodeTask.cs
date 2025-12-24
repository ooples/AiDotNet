namespace AiDotNet.ProgramSynthesis.Enums;

/// <summary>
/// Defines the different types of code-related tasks that can be performed.
/// </summary>
/// <remarks>
/// <para>
/// This enumeration categorizes the various operations that can be performed on code,
/// from understanding and generation to transformation and quality assurance.
/// </para>
/// <para><b>For Beginners:</b> These are different things you might want to do with code.
///
/// Just like you can do different things with text (read, write, translate, summarize),
/// you can do different things with code. This enum lists all the code-related tasks
/// the system can help with.
/// </para>
/// </remarks>
public enum CodeTask
{
    /// <summary>
    /// Code completion task - suggesting how to complete partial code.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Code completion predicts and suggests the next tokens or statements based on
    /// the existing code context. Similar to autocomplete in text editors.
    /// </para>
    /// <para><b>For Beginners:</b> Code completion is like autocomplete for programming.
    ///
    /// When you start typing code, the system suggests how to complete it, just like
    /// your phone suggests words when you're texting. This saves time and reduces errors.
    /// </para>
    /// </remarks>
    Completion,

    /// <summary>
    /// Code generation task - creating new code from specifications or descriptions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Code generation creates complete code implementations from high-level descriptions,
    /// requirements, or examples. This can range from single functions to entire programs.
    /// </para>
    /// <para><b>For Beginners:</b> Code generation creates code from descriptions.
    ///
    /// You describe what you want in plain English (or provide examples), and the system
    /// writes the code for you. Like asking a chef to make a dish from a description.
    /// </para>
    /// </remarks>
    Generation,

    /// <summary>
    /// Code translation task - converting code from one language to another.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Code translation transforms programs written in one programming language into
    /// equivalent programs in another language, preserving functionality.
    /// </para>
    /// <para><b>For Beginners:</b> Code translation converts code between languages.
    ///
    /// Like translating a book from English to Spanish, this converts code from one
    /// programming language to another (like Python to Java) while keeping the same functionality.
    /// </para>
    /// </remarks>
    Translation,

    /// <summary>
    /// Code summarization task - generating natural language descriptions of code.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Code summarization creates concise natural language descriptions that explain
    /// what a piece of code does, helping with documentation and code understanding.
    /// </para>
    /// <para><b>For Beginners:</b> Code summarization explains what code does in plain English.
    ///
    /// It reads code and writes a human-readable description of what the code does,
    /// like creating a book summary from the full text.
    /// </para>
    /// </remarks>
    Summarization,

    /// <summary>
    /// Bug detection task - identifying potential errors and issues in code.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Bug detection analyzes code to find errors, vulnerabilities, and potential issues
    /// that could cause the program to fail or behave incorrectly.
    /// </para>
    /// <para><b>For Beginners:</b> Bug detection finds mistakes in code.
    ///
    /// Like proofreading a document, this examines code to find errors before they
    /// cause problems. It can catch typos, logic errors, and security vulnerabilities.
    /// </para>
    /// </remarks>
    BugDetection,

    /// <summary>
    /// Bug fixing task - automatically repairing identified bugs in code.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Bug fixing not only identifies bugs but also suggests or automatically applies
    /// corrections to fix the identified issues.
    /// </para>
    /// <para><b>For Beginners:</b> Bug fixing automatically corrects errors in code.
    ///
    /// After finding bugs, this goes a step further and actually fixes them, like
    /// spell-check that not only finds typos but corrects them too.
    /// </para>
    /// </remarks>
    BugFixing,

    /// <summary>
    /// Code refactoring task - improving code structure without changing functionality.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Code refactoring restructures existing code to improve readability, maintainability,
    /// or performance while preserving its external behavior.
    /// </para>
    /// <para><b>For Beginners:</b> Refactoring makes code better without changing what it does.
    ///
    /// Like reorganizing a messy room - everything stays the same but becomes easier to
    /// find and use. Makes code cleaner, easier to understand, and easier to modify later.
    /// </para>
    /// </remarks>
    Refactoring,

    /// <summary>
    /// Code understanding task - analyzing and comprehending code semantics.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Code understanding involves analyzing code to extract semantic information,
    /// identify patterns, understand control flow, and grasp the program's logic.
    /// </para>
    /// <para><b>For Beginners:</b> Code understanding means figuring out what code does.
    ///
    /// This involves reading and analyzing code to understand its purpose, how it works,
    /// and what it accomplishes. Like reading comprehension for programming.
    /// </para>
    /// </remarks>
    Understanding,

    /// <summary>
    /// Test generation task - automatically creating test cases for code.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Test generation creates test cases that verify the correctness of code by
    /// checking various inputs and expected outputs.
    /// </para>
    /// <para><b>For Beginners:</b> Test generation creates checks to verify code works correctly.
    ///
    /// It automatically writes tests that check if your code does what it's supposed to do.
    /// Like creating a checklist to make sure all features of a product work correctly.
    /// </para>
    /// </remarks>
    TestGeneration,

    /// <summary>
    /// Code documentation task - generating documentation for code.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Code documentation creates explanatory comments and documentation that describe
    /// what code does, how to use it, and important implementation details.
    /// </para>
    /// <para><b>For Beginners:</b> Documentation creates guides and explanations for code.
    ///
    /// It generates comments, user guides, and API documentation that explain how to use
    /// the code. Like writing an instruction manual for a product.
    /// </para>
    /// </remarks>
    Documentation,

    /// <summary>
    /// Code search task - finding relevant code based on queries.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Code search finds relevant code snippets or functions based on natural language
    /// queries or code patterns, helping developers find reusable code.
    /// </para>
    /// <para><b>For Beginners:</b> Code search finds code that does what you need.
    ///
    /// You describe what you're looking for (like "function to sort a list"), and it
    /// finds existing code that does that. Like a search engine for code.
    /// </para>
    /// </remarks>
    Search,

    /// <summary>
    /// Clone detection task - identifying duplicate or similar code.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Clone detection finds instances of duplicated or highly similar code, which can
    /// indicate opportunities for refactoring or potential plagiarism.
    /// </para>
    /// <para><b>For Beginners:</b> Clone detection finds copied or repeated code.
    ///
    /// It identifies places where the same or very similar code appears multiple times,
    /// which often means the code could be simplified by reusing one version.
    /// </para>
    /// </remarks>
    CloneDetection,

    /// <summary>
    /// Code review task - analyzing code quality and suggesting improvements.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Code review evaluates code for quality, adherence to best practices, potential
    /// issues, and suggests improvements or changes.
    /// </para>
    /// <para><b>For Beginners:</b> Code review checks code quality and suggests improvements.
    ///
    /// Like having an experienced programmer review your code, this examines your code
    /// for problems, style issues, and opportunities to make it better.
    /// </para>
    /// </remarks>
    CodeReview
}

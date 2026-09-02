namespace AiDotNet.Configuration;

/// <summary>Names the executable and argument template used to run one language inside the program sandbox.</summary>
/// <remarks>
/// <para>
/// A specification is a command line with holes in it. <see cref="RunArgumentTemplate"/> is expanded by replacing
/// <c>{source}</c> with the full path of the file the sandbox wrote and <c>{workspace}</c> with the directory that
/// holds it; both replacements are shell-quoted for the target platform, so a workspace path containing spaces is
/// handled correctly and a path can never break out of its argument. Nothing else in the template is interpreted,
/// so any flags the interpreter needs can be written literally.
/// </para>
/// <para>
/// <see cref="CompileArgumentTemplate"/> is the separate command that compiles or type-checks without running.
/// Leaving it <c>null</c> is a statement of fact rather than a gap: a request that sets
/// <see cref="AiDotNet.ProgramSynthesis.Execution.ProgramExecuteRequest.CompileOnly"/> against a language with no
/// compile command is refused instead of quietly executing the program, because silently running code that the
/// caller asked only to compile is exactly the failure this sandbox exists to prevent.
/// </para>
/// <para><b>For Beginners:</b> To run a Python file you type <c>python myfile.py</c>. This class stores those two
/// halves separately: the program to launch (<c>python</c>) and the arguments to give it (<c>"{source}"</c>, where
/// <c>{source}</c> stands for whichever temporary file the sandbox just wrote). Because it is only a template, you
/// can point it at a specific interpreter on your machine, add flags such as <c>-X utf8</c>, or plug in a language
/// this library has never heard of, without changing any code.</para>
/// </remarks>
public sealed class ProgramInterpreterSpecification
{
    /// <summary>The placeholder replaced by the quoted full path of the written source file.</summary>
    public const string SourcePlaceholder = "{source}";

    /// <summary>The placeholder replaced by the quoted full path of the isolated workspace directory.</summary>
    public const string WorkspacePlaceholder = "{workspace}";

    /// <summary>Initializes an interpreter specification.</summary>
    /// <param name="executable">The interpreter or compiler to launch; a bare name is resolved against the pinned sandbox path.</param>
    /// <param name="runArgumentTemplate">The argument template used to run the program, containing <see cref="SourcePlaceholder"/>.</param>
    /// <param name="compileArgumentTemplate">The argument template used to compile without running, or <c>null</c> when the language has no such command.</param>
    /// <exception cref="ArgumentNullException"><paramref name="executable"/> or <paramref name="runArgumentTemplate"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="executable"/> is empty or white space.</exception>
    public ProgramInterpreterSpecification(
        string executable,
        string runArgumentTemplate,
        string? compileArgumentTemplate = null)
    {
        if (executable is null) throw new ArgumentNullException(nameof(executable));
        if (runArgumentTemplate is null) throw new ArgumentNullException(nameof(runArgumentTemplate));
        if (string.IsNullOrWhiteSpace(executable))
            throw new ArgumentException("The interpreter executable cannot be empty or white space.", nameof(executable));

        Executable = executable.Trim();
        RunArgumentTemplate = runArgumentTemplate;
        CompileArgumentTemplate = compileArgumentTemplate;
    }

    /// <summary>Gets the interpreter or compiler to launch.</summary>
    public string Executable { get; }

    /// <summary>Gets the argument template used to run a program.</summary>
    public string RunArgumentTemplate { get; }

    /// <summary>Gets the argument template used to compile a program without running it, when one exists.</summary>
    public string? CompileArgumentTemplate { get; }

    /// <summary>Gets whether this specification can satisfy a compile-only request.</summary>
    public bool SupportsCompileOnly => !string.IsNullOrWhiteSpace(CompileArgumentTemplate);

    /// <summary>Expands a template against one workspace, quoting both substituted paths.</summary>
    /// <param name="template">The template to expand.</param>
    /// <param name="sourcePath">The full path of the written source file.</param>
    /// <param name="workspacePath">The full path of the isolated workspace directory.</param>
    /// <param name="quoteWithDoubleQuotes">
    /// <c>true</c> to wrap substituted paths in double quotes, which is how a command line is parsed on Windows and
    /// by the .NET process launcher; <c>false</c> to use POSIX single-quote escaping for a shell command line.
    /// </param>
    /// <returns>The expanded argument string.</returns>
    /// <exception cref="ArgumentNullException">Any argument is <c>null</c>.</exception>
    public static string Expand(
        string template,
        string sourcePath,
        string workspacePath,
        bool quoteWithDoubleQuotes)
    {
        if (template is null) throw new ArgumentNullException(nameof(template));
        if (sourcePath is null) throw new ArgumentNullException(nameof(sourcePath));
        if (workspacePath is null) throw new ArgumentNullException(nameof(workspacePath));

        string source = quoteWithDoubleQuotes ? QuoteForCommandLine(sourcePath) : QuoteForPosixShell(sourcePath);
        string workspace = quoteWithDoubleQuotes ? QuoteForCommandLine(workspacePath) : QuoteForPosixShell(workspacePath);
        return template
            .Replace(SourcePlaceholder, source)
            .Replace(WorkspacePlaceholder, workspace);
    }

    /// <summary>Wraps a value in double quotes, escaping embedded quotes and trailing backslashes.</summary>
    /// <param name="value">The value to quote.</param>
    /// <returns>A token that survives the .NET and Windows command-line parsers intact.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="value"/> is <c>null</c>.</exception>
    public static string QuoteForCommandLine(string value)
    {
        if (value is null) throw new ArgumentNullException(nameof(value));

        var builder = new System.Text.StringBuilder(value.Length + 8);
        builder.Append('"');
        int pendingBackslashes = 0;
        foreach (char character in value)
        {
            if (character == '\\')
            {
                pendingBackslashes++;
                continue;
            }

            if (character == '"')
            {
                builder.Append('\\', (pendingBackslashes * 2) + 1);
                pendingBackslashes = 0;
                builder.Append('"');
                continue;
            }

            builder.Append('\\', pendingBackslashes);
            pendingBackslashes = 0;
            builder.Append(character);
        }

        builder.Append('\\', pendingBackslashes * 2);
        builder.Append('"');
        return builder.ToString();
    }

    /// <summary>Wraps a value in POSIX single quotes so a shell treats it as one literal token.</summary>
    /// <param name="value">The value to quote.</param>
    /// <returns>A single-quoted token safe to embed in a <c>/bin/sh -c</c> script.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="value"/> is <c>null</c>.</exception>
    public static string QuoteForPosixShell(string value)
    {
        if (value is null) throw new ArgumentNullException(nameof(value));
        return "'" + value.Replace("'", "'\\''") + "'";
    }
}

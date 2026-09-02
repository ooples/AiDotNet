using AiDotNet.ProgramSynthesis.Enums;

namespace AiDotNet.Configuration;

/// <summary>Configures the boundary, the limits, and the per-language commands used to execute untrusted programs.</summary>
/// <remarks>
/// <para>
/// This is the single place a caller states how much a candidate program is allowed to do. <see cref="Mode"/>
/// chooses the execution boundary, <see cref="Limits"/> carries the resource ceilings enforced at that boundary, and
/// <see cref="Interpreters"/> maps each <see cref="ProgramLanguage"/> onto the command that runs it. Nothing here
/// starts a process on its own: the options are inert until they are handed to
/// <see cref="AiDotNet.ProgramSynthesis.Execution.ProcessProgramExecutionEngine"/> or to another engine, which is
/// why the core library gains no Python, Node, Docker, or network dependency by shipping them.
/// </para>
/// <para>
/// <see cref="Validate"/> refuses configurations that cannot be honoured, and refuses
/// <see cref="ProgramSandboxMode.InProcessUnsafe"/> outright unless <see cref="AllowUnsafeInProcessExecution"/> has
/// also been set, so a host cannot reach the unsafe mode by editing one field of a configuration file. The default
/// interpreter table covers Python, JavaScript, and C# with the same commands the reasoning-layer code verifier
/// uses, except that they run the written file directly rather than interpolating it into a shell expression, and
/// each entry that can be type-checked without running carries a separate compile-only command.
/// </para>
/// <para><b>For Beginners:</b> Think of this as the rules-of-the-house form you fill in before letting generated
/// code run on your machine: where it may run, how long it may run, how much memory it may use, and which program
/// on your computer should be used to run each language. The defaults are safe and small. The one field worth
/// reading twice is <see cref="AllowUnsafeInProcessExecution"/>, which turns off the protection entirely; leave it
/// alone unless you fully control the code being executed.</para>
/// </remarks>
public sealed class ProgramSandboxOptions
{
    private readonly Dictionary<ProgramLanguage, ProgramInterpreterSpecification> _interpreters;

    /// <summary>Initializes sandbox options with safe defaults and the built-in interpreter table.</summary>
    public ProgramSandboxOptions()
    {
        _interpreters = CreateDefaultInterpreters();
    }

    /// <summary>Gets or sets the execution boundary used to run candidate programs.</summary>
    public ProgramSandboxMode Mode { get; set; } = ProgramSandboxMode.OutOfProcessWorker;

    /// <summary>Gets or sets the resource limits enforced on every execution.</summary>
    public ProgramSandboxLimitOptions Limits { get; set; } = new();

    /// <summary>Gets the per-language commands used to run and to compile candidate programs.</summary>
    /// <remarks>
    /// Populated with the built-in defaults on construction. Assign an entry to override a language, or add one for
    /// a language the defaults do not cover; a language with no entry is refused rather than guessed at.
    /// </remarks>
    public IDictionary<ProgramLanguage, ProgramInterpreterSpecification> Interpreters => _interpreters;

    /// <summary>Gets or sets the directory under which isolated workspaces are created, or <c>null</c> for the system temporary directory.</summary>
    /// <remarks>
    /// Each execution creates a fresh subdirectory here, writes exactly one source file into it, uses it as the
    /// child process's working directory, and deletes it when the execution finishes, including after a timeout.
    /// </remarks>
    public string? WorkingDirectory { get; set; }

    /// <summary>Gets or sets whether <see cref="ProgramSandboxMode.InProcessUnsafe"/> is permitted.</summary>
    /// <remarks>
    /// Defaults to <c>false</c>. While it is <c>false</c>, <see cref="Validate"/> throws for that mode, so an
    /// unsafe configuration cannot be reached by accident or by a single edit to a configuration file.
    /// </remarks>
    public bool AllowUnsafeInProcessExecution { get; set; }

    /// <summary>Builds the default per-language interpreter table.</summary>
    /// <returns>
    /// A table covering <see cref="ProgramLanguage.Python"/>, <see cref="ProgramLanguage.JavaScript"/>, and
    /// <see cref="ProgramLanguage.CSharp"/>. The Python executable is <c>python</c> on Windows and <c>python3</c>
    /// elsewhere, matching how each platform actually names the interpreter.
    /// </returns>
    /// <remarks>
    /// The C# entry launches the <c>dotnet-script</c> global tool, which is not installed by default; override that
    /// entry when evolving C#. Python and JavaScript additionally carry a compile-only command
    /// (<c>-m py_compile</c> and <c>--check</c>) so a compile-only request can be honoured without running anything.
    /// </remarks>
    public static Dictionary<ProgramLanguage, ProgramInterpreterSpecification> CreateDefaultInterpreters()
    {
        string python = IsWindowsHost() ? "python" : "python3";
        return new Dictionary<ProgramLanguage, ProgramInterpreterSpecification>
        {
            [ProgramLanguage.Python] = new(
                python,
                ProgramInterpreterSpecification.SourcePlaceholder,
                "-m py_compile " + ProgramInterpreterSpecification.SourcePlaceholder),
            [ProgramLanguage.JavaScript] = new(
                "node",
                ProgramInterpreterSpecification.SourcePlaceholder,
                "--check " + ProgramInterpreterSpecification.SourcePlaceholder),
            [ProgramLanguage.CSharp] = new(
                "dotnet",
                "script " + ProgramInterpreterSpecification.SourcePlaceholder)
        };
    }

    /// <summary>Overrides or adds the command used for one language.</summary>
    /// <param name="language">The language the command applies to.</param>
    /// <param name="specification">The executable and argument templates to use.</param>
    /// <exception cref="ArgumentNullException"><paramref name="specification"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="language"/> is not a defined value.</exception>
    public void SetInterpreter(ProgramLanguage language, ProgramInterpreterSpecification specification)
    {
        if (specification is null) throw new ArgumentNullException(nameof(specification));
        if (!Enum.IsDefined(typeof(ProgramLanguage), language))
            throw new ArgumentOutOfRangeException(nameof(language), language, "Value must be a defined language.");
        _interpreters[language] = specification;
    }

    /// <summary>Looks up the command configured for one language.</summary>
    /// <param name="language">The language to look up.</param>
    /// <param name="specification">The configured command when one exists.</param>
    /// <returns><c>true</c> when the language has a configured command.</returns>
    public bool TryGetInterpreter(ProgramLanguage language, out ProgramInterpreterSpecification? specification) =>
        _interpreters.TryGetValue(language, out specification);

    /// <summary>Creates an independent copy so a running engine is unaffected by later mutation.</summary>
    /// <returns>A new instance carrying the same mode, cloned limits, a copied interpreter table, and the same flags.</returns>
    public ProgramSandboxOptions Clone()
    {
        var clone = new ProgramSandboxOptions
        {
            Mode = Mode,
            Limits = Limits is null ? new ProgramSandboxLimitOptions() : Limits.Clone(),
            WorkingDirectory = WorkingDirectory,
            AllowUnsafeInProcessExecution = AllowUnsafeInProcessExecution
        };

        clone._interpreters.Clear();
        foreach (KeyValuePair<ProgramLanguage, ProgramInterpreterSpecification> entry in _interpreters)
        {
            clone._interpreters[entry.Key] = entry.Value;
        }

        return clone;
    }

    /// <summary>Rejects a configuration that cannot be honoured or that silently disables the sandbox.</summary>
    /// <exception cref="ArgumentOutOfRangeException"><see cref="Mode"/> is not a defined value, or a limit is impossible.</exception>
    /// <exception cref="ArgumentException">
    /// <see cref="Limits"/> is <c>null</c>, an interpreter entry is <c>null</c> or lacks the
    /// <see cref="ProgramInterpreterSpecification.SourcePlaceholder"/> token, <see cref="WorkingDirectory"/> is
    /// white space, or <see cref="Mode"/> is <see cref="ProgramSandboxMode.InProcessUnsafe"/> without
    /// <see cref="AllowUnsafeInProcessExecution"/>.
    /// </exception>
    public void Validate()
    {
        if (!Enum.IsDefined(typeof(ProgramSandboxMode), Mode))
            throw new ArgumentOutOfRangeException(nameof(Mode), Mode, "Value must be a defined sandbox mode.");
        if (Mode == ProgramSandboxMode.InProcessUnsafe && !AllowUnsafeInProcessExecution)
        {
            throw new ArgumentException(
                "ProgramSandboxMode.InProcessUnsafe executes untrusted program text inside this process. " +
                "Set AllowUnsafeInProcessExecution to true to acknowledge that before selecting it.",
                nameof(Mode));
        }

        if (Limits is null) throw new ArgumentException("Limits cannot be null.", nameof(Limits));
        Limits.Validate();

        if (WorkingDirectory is not null && string.IsNullOrWhiteSpace(WorkingDirectory))
            throw new ArgumentException("WorkingDirectory cannot be white space; use null for the temporary directory.", nameof(WorkingDirectory));

        foreach (KeyValuePair<ProgramLanguage, ProgramInterpreterSpecification> entry in _interpreters)
        {
            if (entry.Value is null)
                throw new ArgumentException($"The interpreter for {entry.Key} cannot be null.", nameof(Interpreters));
            if (entry.Value.RunArgumentTemplate.IndexOf(
                    ProgramInterpreterSpecification.SourcePlaceholder, StringComparison.Ordinal) < 0)
            {
                throw new ArgumentException(
                    $"The run command for {entry.Key} must contain the {ProgramInterpreterSpecification.SourcePlaceholder} placeholder.",
                    nameof(Interpreters));
            }

            string? compile = entry.Value.CompileArgumentTemplate;
            if (compile is not null &&
                compile.IndexOf(ProgramInterpreterSpecification.SourcePlaceholder, StringComparison.Ordinal) < 0)
            {
                throw new ArgumentException(
                    $"The compile command for {entry.Key} must contain the {ProgramInterpreterSpecification.SourcePlaceholder} placeholder.",
                    nameof(Interpreters));
            }
        }
    }

    private static bool IsWindowsHost()
    {
#if NET5_0_OR_GREATER
        return OperatingSystem.IsWindows();
#else
        return Environment.OSVersion.Platform == PlatformID.Win32NT;
#endif
    }
}

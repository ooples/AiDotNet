using System.Globalization;

namespace AiDotNet.Configuration;

/// <summary>Settings for a chat client that runs a command-line program instead of calling an HTTP endpoint.</summary>
/// <remarks>
/// <para>
/// Some models are reached through a command rather than an API: a locally installed agent, a vendor's own CLI, a
/// wrapper script that holds credentials the process should never see. This describes how to invoke one.
/// </para>
/// <para><b>For Beginners:</b> Give it the program to run and any fixed arguments. The prompt is handed over on
/// standard input rather than on the command line, so a long conversation cannot overflow the operating system's
/// limit on how long a command may be.</para>
/// </remarks>
public sealed class ProcessChatClientOptions
{
    /// <summary>The largest permitted total length of the composed argument string.</summary>
    /// <remarks>
    /// Well below the operating-system limit on purpose. The point is not to get as close as possible; it is to fail
    /// with a clear message rather than have a command truncated into something that runs and means something else.
    /// </remarks>
    public const int MaxArgumentLength = 8_000;

    private IList<string>? _arguments;

    /// <summary>Gets or sets the executable to run.</summary>
    public string FileName { get; set; } = string.Empty;

    /// <summary>Gets or sets the fixed arguments passed before any that this class adds.</summary>
    public IList<string> Arguments
    {
        get => _arguments ??= new List<string>();
        set => _arguments = value;
    }

    /// <summary>Gets or sets the working directory, or <c>null</c> to inherit the current one.</summary>
    public string? WorkingDirectory { get; set; }

    /// <summary>Gets or sets the per-call spending cap in US dollars, or <c>null</c> for none.</summary>
    /// <remarks>
    /// <para>
    /// Passed to the program as <see cref="BudgetArgumentName"/> followed by the amount, formatted with the invariant
    /// culture so a machine with a comma decimal separator sends the same text as one with a point. Whether the cap
    /// is honoured is the program's business; what this guarantees is that the program is told.
    /// </para>
    /// <para>
    /// A cap matters more here than for an HTTP client, because a subprocess model can be an agent that decides for
    /// itself how much work one prompt deserves, and an evolution run makes thousands of prompts.
    /// </para>
    /// </remarks>
    public double? MaxBudgetUsd { get; set; }

    /// <summary>Gets or sets the argument name that carries <see cref="MaxBudgetUsd"/>. Defaults to <c>--max-budget-usd</c>.</summary>
    public string BudgetArgumentName { get; set; } = "--max-budget-usd";

    /// <summary>Gets or sets how long one call may run before it is killed. Defaults to five minutes.</summary>
    public TimeSpan Timeout { get; set; } = TimeSpan.FromMinutes(5);

    /// <summary>Gets or sets the largest reply read from the program, in characters. Defaults to 1 MB.</summary>
    public int MaxOutputChars { get; set; } = 1024 * 1024;

    /// <summary>Gets or sets the identifier the client reports. Defaults to <c>process</c>.</summary>
    public string ModelId { get; set; } = "process";

    /// <summary>Creates an independent copy so a running client is unaffected by later mutation.</summary>
    /// <returns>A new instance carrying the same settings and a copied argument list.</returns>
    public ProcessChatClientOptions Clone() => new()
    {
        FileName = FileName,
        _arguments = _arguments is null ? null : new List<string>(_arguments),
        WorkingDirectory = WorkingDirectory,
        MaxBudgetUsd = MaxBudgetUsd,
        BudgetArgumentName = BudgetArgumentName,
        Timeout = Timeout,
        MaxOutputChars = MaxOutputChars,
        ModelId = ModelId
    };

    /// <summary>Builds the full argument list, budget included.</summary>
    /// <returns>The arguments in the order they are passed.</returns>
    /// <exception cref="ArgumentException">The composed arguments exceed <see cref="MaxArgumentLength"/>.</exception>
    public IReadOnlyList<string> BuildArguments()
    {
        var arguments = new List<string>(Arguments.Count + 2);
        foreach (string argument in Arguments)
            if (argument is not null) arguments.Add(argument);

        if (MaxBudgetUsd.HasValue)
        {
            arguments.Add(BudgetArgumentName);
            arguments.Add(MaxBudgetUsd.Value.ToString("0.####", CultureInfo.InvariantCulture));
        }

        int length = arguments.Sum(argument => argument.Length + 1);
        if (length > MaxArgumentLength)
        {
            throw new ArgumentException(
                "The composed command line is " + length.ToString(CultureInfo.InvariantCulture) +
                " characters, above the limit of " + MaxArgumentLength.ToString(CultureInfo.InvariantCulture) +
                ". Move the long values into a file or an environment variable; the prompt itself is already sent " +
                "on standard input and is not part of this.",
                nameof(Arguments));
        }

        return arguments;
    }

    /// <summary>Rejects settings that cannot produce a runnable command.</summary>
    /// <exception cref="ArgumentException">The executable, the budget argument name, or the identifier is blank.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The budget, timeout, or output bound is out of range.</exception>
    public void Validate()
    {
        if (string.IsNullOrWhiteSpace(FileName))
            throw new ArgumentException("The executable to run cannot be blank.", nameof(FileName));
        if (Arguments.Any(argument => argument is null))
            throw new ArgumentException("Arguments cannot contain a null entry.", nameof(Arguments));
        if (string.IsNullOrWhiteSpace(BudgetArgumentName))
            throw new ArgumentException("The budget argument name cannot be blank.", nameof(BudgetArgumentName));
        if (string.IsNullOrWhiteSpace(ModelId))
            throw new ArgumentException("The reported model identifier cannot be blank.", nameof(ModelId));

        if (MaxBudgetUsd is { } budget && (double.IsNaN(budget) || double.IsInfinity(budget) || budget <= 0))
        {
            throw new ArgumentOutOfRangeException(nameof(MaxBudgetUsd), budget,
                "Value must be a finite positive amount; leave it null for no cap.");
        }

        if (Timeout <= TimeSpan.Zero)
            throw new ArgumentOutOfRangeException(nameof(Timeout), Timeout, "Value must be positive.");
        if (MaxOutputChars <= 0)
            throw new ArgumentOutOfRangeException(nameof(MaxOutputChars), MaxOutputChars, "Value must be positive.");

        BuildArguments();
    }
}

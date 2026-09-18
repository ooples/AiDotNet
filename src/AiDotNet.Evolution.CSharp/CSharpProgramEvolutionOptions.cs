using System.Globalization;
using AiDotNet.Evolution;

namespace AiDotNet.Evolution.CSharp;

/// <summary>Bounds and explicitly declared model/build prices for the opt-in C# proposal loop.</summary>
/// <remarks>Defaults are synthetic work units, not money. Supply trusted reference assemblies and a pinned
/// model/configuration identity. Compiler cancellation is cooperative; this is not an OS security sandbox.</remarks>
public sealed class CSharpProgramEvolutionOptions
{
    /// <summary>Gets or sets trusted assembly paths, copied into bounded owned metadata images at configuration time.</summary>
    public IList<string> ReferencePaths { get; set; } = new List<string>();
    /// <summary>Gets or sets the expected runtime/reference-target identity, such as a pinned .NET reference pack.</summary>
    public string TargetIdentity { get; set; } = string.Empty;
    /// <summary>Gets or sets a pinned model/provider/tool configuration identity, not just a mutable model alias.</summary>
    public string ModelVersionIdentity { get; set; } = string.Empty;
    /// <summary>Gets or sets the operator identity. Use distinct identities for different compilers on one ledger.</summary>
    public string Id { get; set; } = "csharp-compiler-guided";
    /// <summary>Gets or sets the caller-owned directory for complete per-attempt patch/build evidence.</summary>
    /// <remarks>Records contain unredacted source, prompts and model output. Use a private, access-controlled
    /// directory with a retention policy; do not put credentials or sealed test data in search inputs.</remarks>
    public string AuditDirectory { get; set; } = string.Empty;
    /// <summary>Gets or sets the source bound, at most 65,536 UTF-16 characters.</summary>
    public int MaxSourceChars { get; set; } = 65_536;
    /// <summary>Gets or sets the JSON response bound, at most 262,144 characters.</summary>
    public int MaxResponseChars { get; set; } = 131_072;
    /// <summary>Gets or sets the number of repairs after the first attempt, from zero through seven.</summary>
    public int MaxRepairs { get; set; } = 2;
    /// <summary>Gets or sets the maximum non-overlapping edits in one response, from one through sixteen.</summary>
    public int MaxEdits { get; set; } = 8;
    /// <summary>Gets or sets the bounded syntax catalog size, from one through sixty-four.</summary>
    public int MaxCatalogNodes { get; set; } = 32;
    /// <summary>Gets or sets the declared per-request input-token maximum.</summary>
    public int MaxInputTokens { get; set; } = 65_536;
    /// <summary>Gets or sets the provider's requested output-token maximum.</summary>
    public int MaxOutputTokens { get; set; } = 4_096;
    /// <summary>Gets or sets the cooperative per-compiler-operation timeout, from one through thirty seconds.</summary>
    public int CompilationTimeoutSeconds { get; set; } = 10;
    /// <summary>Gets or sets the common cost-unit identity, shared with program evaluation.</summary>
    public string CostUnitVersionHash { get; set; } = "csharp-synthetic-work-v1";
    /// <summary>Gets or sets the fixed charge for loading and hashing the trusted reference bundle.</summary>
    public decimal SetupCostUnits { get; set; } = 0.1m;
    /// <summary>Gets or sets the fixed charge for each dispatched model request, including unusable answers.</summary>
    public decimal ModelCallCostUnits { get; set; } = 1m;
    /// <summary>Gets or sets the charge for each provider-reported input token.</summary>
    public decimal InputTokenCostUnits { get; set; } = 0.00001m;
    /// <summary>Gets or sets the charge for each provider-reported output token.</summary>
    public decimal OutputTokenCostUnits { get; set; } = 0.00002m;
    /// <summary>Gets or sets the fixed charge for each dispatched compiler emit, including failed builds.</summary>
    public decimal CompilationCostUnits { get; set; } = 0.25m;
    /// <summary>Gets or sets the fixed charge for each syntax preparation or patch-validation attempt.</summary>
    public decimal ParseCostUnits { get; set; } = 0.01m;
    /// <summary>Gets or sets the fixed charge for committing each bounded evidence record.</summary>
    public decimal AuditCostUnits { get; set; } = 0.01m;

    internal CSharpProgramEvolutionOptions Snapshot()
    {
        var copy = (CSharpProgramEvolutionOptions)MemberwiseClone();
        if (ReferencePaths is null) throw new ArgumentException("ReferencePaths is required.");
        copy.ReferencePaths = ReferencePaths.Take(65).ToArray();
        if (copy.ReferencePaths.Count is < 1 or > 64 || copy.ReferencePaths.Any(string.IsNullOrWhiteSpace))
            throw new ArgumentException("Supply one through sixty-four trusted reference assembly paths.");
        foreach (string value in new[] { Id, TargetIdentity, ModelVersionIdentity, CostUnitVersionHash })
        {
            if (string.IsNullOrWhiteSpace(value) || value.Length > 256 || value.Any(char.IsControl))
                throw new ArgumentException("Compiler, target, model and cost-unit identities must be bounded printable strings.");
            new System.Text.UTF8Encoding(false, true).GetByteCount(value);
        }
        if (Id.Length > 64) throw new ArgumentException("The compiler operator identity must not exceed 64 characters.");
        if (string.IsNullOrWhiteSpace(AuditDirectory)) throw new ArgumentException("An audit directory is required.");
        copy.AuditDirectory = Path.GetFullPath(AuditDirectory);
        if (MaxSourceChars is < 256 or > 65_536 || MaxResponseChars is < 256 or > 262_144 ||
            MaxRepairs is < 0 or > 7 || MaxEdits is < 1 or > 16 || MaxCatalogNodes is < 1 or > 64 ||
            MaxInputTokens is < 1 or > 1_048_576 || MaxOutputTokens is < 1 or > 65_536 || CompilationTimeoutSeconds is < 1 or > 30)
            throw new ArgumentOutOfRangeException(nameof(CSharpProgramEvolutionOptions), "A compiler/proposal bound is outside its supported range.");
        foreach (decimal price in new[] { SetupCostUnits, ModelCallCostUnits, CompilationCostUnits, ParseCostUnits, AuditCostUnits })
            if (price <= 0 || price > 1_000_000_000m) throw new ArgumentOutOfRangeException(nameof(CSharpProgramEvolutionOptions), "Fixed work prices must be positive and bounded.");
        foreach (decimal price in new[] { InputTokenCostUnits, OutputTokenCostUnits })
            if (price < 0 || price > 1_000_000m) throw new ArgumentOutOfRangeException(nameof(CSharpProgramEvolutionOptions), "Token prices must be nonnegative and bounded.");
        return copy;
    }

    internal string ConfigurationHash => EvolutionHash.Combine(new[]
    {
        "csharp-proposal-options-v1", Id, TargetIdentity, ModelVersionIdentity, CostUnitVersionHash,
        MaxSourceChars.ToString(CultureInfo.InvariantCulture), MaxResponseChars.ToString(CultureInfo.InvariantCulture),
        MaxRepairs.ToString(CultureInfo.InvariantCulture), MaxEdits.ToString(CultureInfo.InvariantCulture),
        MaxCatalogNodes.ToString(CultureInfo.InvariantCulture), MaxInputTokens.ToString(CultureInfo.InvariantCulture),
        MaxOutputTokens.ToString(CultureInfo.InvariantCulture), CompilationTimeoutSeconds.ToString(CultureInfo.InvariantCulture),
        SetupCostUnits.ToString(CultureInfo.InvariantCulture), ModelCallCostUnits.ToString(CultureInfo.InvariantCulture),
        CompilationCostUnits.ToString(CultureInfo.InvariantCulture), ParseCostUnits.ToString(CultureInfo.InvariantCulture),
        InputTokenCostUnits.ToString(CultureInfo.InvariantCulture), OutputTokenCostUnits.ToString(CultureInfo.InvariantCulture),
        AuditCostUnits.ToString(CultureInfo.InvariantCulture)
    });
}

using AiDotNet.Agentic.Models;
using AiDotNet.Configuration;
using AiDotNet.Evolution.Programs;
using AiDotNet.Interfaces;
using AiDotNet.ProgramSynthesis.Enums;

namespace AiDotNet.Evolution.CSharp;

/// <summary>Configures compiler-guided C# proposals through the existing program-evolution facade.</summary>
public static class CSharpProgramEvolutionExtensions
{
    /// <summary>Installs a bounded syntax-edit/emit/repair loop and a shared proposal/evaluation resource ledger.</summary>
    /// <remarks>
    /// Configuration loads trusted reference metadata and charges setup before any model call. The caller supplies
    /// the chat client, execution isolation, search-visible tests and independent correctness checks. Compilation
    /// does not execute candidates and is not a correctness, API-behavior or held-out-performance proof.
    /// No key, provider, network connection or subscription is inferred. Automatic resource checkpoint/resume is
    /// currently refused by the facade. Use a fresh operator, ledger and evidence directory for an independent run.
    /// </remarks>
    public static IAiModelBuilder<T, TInput, TOutput> ConfigureCSharpProgramEvolution<T, TInput, TOutput>(
        this IAiModelBuilder<T, TInput, TOutput> builder, IChatClient<T> client, ProgramEvolutionOptions programOptions,
        CSharpProgramEvolutionOptions compilerOptions, ProgramEvolutionResourceOptions resources)
    {
        if (builder is null) throw new ArgumentNullException(nameof(builder));
        if (client is null) throw new ArgumentNullException(nameof(client));
        if (programOptions is null) throw new ArgumentNullException(nameof(programOptions));
        if (compilerOptions is null) throw new ArgumentNullException(nameof(compilerOptions));
        if (resources is null) throw new ArgumentNullException(nameof(resources));
        var program = programOptions.Clone();
        if (program.CustomVariation is not null) throw new ArgumentException("A custom proposal loop is already configured.", nameof(programOptions));
        if (program.HasEnabledProvenance) throw new ArgumentException("The compiler loop owns its bounded evidence directory; disable the built-in provenance sink.", nameof(programOptions));
        if (program.Language == ProgramLanguage.Generic) program.Language = ProgramLanguage.CSharp;
        if (program.Language != ProgramLanguage.CSharp) throw new ArgumentException("CSharp program language is required.", nameof(programOptions));
        var compiler = compilerOptions.Snapshot();
        if (!string.Equals(compiler.CostUnitVersionHash, resources.CostUnitVersionHash, StringComparison.Ordinal))
            throw new ArgumentException("Compiler and evaluator cost-unit identities must match.", nameof(resources));
        program.ResourceAccounting = resources;
        program.Validate();
        if (program.Engine.Resume || program.Engine.CheckpointInterval > 0)
            throw new NotSupportedException("Compiler accounting requires coordinated ledger/engine checkpoints; automatic resume is not yet supported.");
        program.CustomVariation = CreateCSharpProgramVariation(client, program, compiler, resources);
        return builder.ConfigureProgramEvolution(program);
    }

    /// <summary>Creates a metered compiler-guided arm for an explicitly configured program portfolio.</summary>
    /// <remarks>Charges compiler setup immediately. Use distinct compiler IDs and audit directories per arm,
    /// the same evaluator/proposal cost identity, and a caller-owned shared ledger. Compilation, repairs and
    /// audit work remain in the proposal receipt. No model is contacted during construction.</remarks>
    public static MeteredProgramVariationOperator CreateCSharpProgramVariation<T>(IChatClient<T> client,
        ProgramEvolutionOptions programOptions, CSharpProgramEvolutionOptions compilerOptions, ProgramEvolutionResourceOptions resources)
    {
        if (client is null) throw new ArgumentNullException(nameof(client));
        if (programOptions is null) throw new ArgumentNullException(nameof(programOptions));
        if (compilerOptions is null) throw new ArgumentNullException(nameof(compilerOptions));
        if (resources is null) throw new ArgumentNullException(nameof(resources));
        var compiler = compilerOptions.Snapshot();
        if (!string.Equals(compiler.CostUnitVersionHash, resources.CostUnitVersionHash, StringComparison.Ordinal))
            throw new ArgumentException("Compiler and evaluator cost-unit identities must match.", nameof(resources));
        var program = programOptions.Clone();
        if (program.Language == ProgramLanguage.Generic) program.Language = ProgramLanguage.CSharp;
        if (program.Language != ProgramLanguage.CSharp) throw new ArgumentException("CSharp program language is required.", nameof(programOptions));
        program.Validate();
        var source = CSharpProposalSource<T>.Create(client, compiler, program, resources.Ledger);
        return new(source, resources.Ledger, source.MaximumProposalResources, source.CostUnitVersionHash);
    }
}

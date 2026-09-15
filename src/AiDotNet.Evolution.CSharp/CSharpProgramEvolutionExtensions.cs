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
        var source = CSharpProposalSource<T>.Create(client, compiler, program, resources.Ledger);
        program.CustomVariation = new CompilerVariation<T>(source, resources.Ledger);
        return builder.ConfigureProgramEvolution(program);
    }

    private sealed class CompilerVariation<T> : IProgramVariationOperator, IOutcomeAwareVariationOperator<ProgramGenome>, IEvolutionProposalCostProvider
    {
        private readonly CSharpProposalSource<T> _source;
        private readonly ResourceMeteredVariationOperator<ProgramGenome> _metered;
        internal CompilerVariation(CSharpProposalSource<T> source, EvolutionResourceLedger ledger)
        {
            _source = source;
            _metered = new(source, ledger, source.MaximumProposalResources, source.CostUnitVersionHash);
        }
        public string Id => _metered.Id;
        public string VersionHash => _metered.VersionHash;
        public string CostUnitVersionHash => _metered.CostUnitVersionHash;
        public ValueTask<ProgramGenome> ProposeAsync(EvolutionVariationContext<ProgramGenome> context, CancellationToken cancellationToken = default) => _metered.ProposeAsync(context, cancellationToken);
        public void Observe(EvolutionEvaluation evaluation, EvolutionArchiveInsertionResult? insertionResult) => _metered.Observe(evaluation, insertionResult);
        public string CaptureState() => _metered.CaptureState();
        public void RestoreState(string state) => _metered.RestoreState(state);
        public EvolutionProposalCost GetProposalCost(long generation) => _metered.GetProposalCost(generation);
        public ProgramEvolutionLlmUsage GetUsage() => _source.GetUsage();
    }
}

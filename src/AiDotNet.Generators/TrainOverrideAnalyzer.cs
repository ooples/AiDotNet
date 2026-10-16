using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Diagnostics;

namespace AiDotNet.Generators;

/// <summary>
/// Keeps training in the base class: a concrete neural network may not override <c>Train(Tensor, Tensor)</c>.
/// </summary>
/// <remarks>
/// <para>
/// <c>NeuralNetworkBase.Train</c> is the one training step: the ONNX and disposed guards, batch-dim promotion,
/// micro-batching, the out-of-memory and device-fault retries, the model's optimizer, and the shared fused session.
/// A model supplies what differs through declarations instead: its optimizer (the generated <c>TrainingOptimizer</c>),
/// why it cannot train (<c>TrainingUnavailableReason</c>), how its input or target is prepared
/// (<c>PrepareTrainingPair</c>, <c>PrepareInputForTraining</c>), a non-standard objective
/// (<c>DefineTrainingObjectives</c>), and a step that is not a single gradient objective (<c>TryApplyLearningRule</c>:
/// contrastive divergence, a Hebbian rule, a closed-form solve, a multi-network orchestration). About 700 overrides that re-did the base step, and skipped half of it, were
/// deleted when this rule was added.
/// </para>
/// <para>
/// Abstract family bases may still override <c>Train</c>. The overrides that predate the rule are listed in
/// <see cref="TrainOverrideBaseline"/>, which may only shrink: a listed type that stops overriding is AIDN108.
/// </para>
/// </remarks>
[DiagnosticAnalyzer(LanguageNames.CSharp)]
public class TrainOverrideAnalyzer : DiagnosticAnalyzer
{
    private const string NeuralNetworkBasePrefix = "AiDotNet.NeuralNetworks.NeuralNetworkBase<";

    private static readonly DiagnosticDescriptor NewTrainOverride = new(
        "AIDN107",
        "A model overrides Train instead of declaring its training step",
        "'{0}' overrides Train. The base class runs every model's training step; declare what differs instead: "
            + "DefineTrainingObjectives for a non-standard objective, TryApplyLearningRule for a non-gradient rule, PrepareTrainingPair or PrepareInputForTraining for "
            + "the data, TrainingUnavailableReason for a guard. Its optimizer comes from its _optimizer field.",
        "AiDotNet.Design",
        DiagnosticSeverity.Warning,
        isEnabledByDefault: true,
        description: "A Train override bypasses the base step's guards, batch-dim promotion, micro-batching, retries and "
            + "declared objectives.");

    private static readonly DiagnosticDescriptor StaleBaselineEntry = new(
        "AIDN108",
        "Train-override baseline entry no longer overrides",
        "'{0}' is listed in TrainOverrideBaseline but no longer overrides Train; remove its line so the baseline stays an exact inventory",
        "AiDotNet.Design",
        DiagnosticSeverity.Warning,
        isEnabledByDefault: true);

    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics
        => ImmutableArray.Create(NewTrainOverride, StaleBaselineEntry);

    public override void Initialize(AnalysisContext context)
    {
        context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
        context.EnableConcurrentExecution();
        context.RegisterCompilationStartAction(start =>
        {
            if (start.Compilation.AssemblyName != "AiDotNet") return;
            var overriding = new ConcurrentDictionary<string, byte>(System.StringComparer.Ordinal);
            start.RegisterSymbolAction(symbolContext =>
            {
                var type = (INamedTypeSymbol)symbolContext.Symbol;
                if (type.IsAbstract || type.TypeKind != TypeKind.Class || !IsNeuralNetwork(type)) return;
                var train = type.GetMembers("Train").OfType<IMethodSymbol>().FirstOrDefault(IsTensorTrainOverride);
                if (train is null) return;

                string name = type.ToDisplayString();
                overriding[name] = 0;
                if (TrainOverrideBaseline.Types.Contains(name)) return;
                var location = train.Locations.FirstOrDefault(l => l.IsInSource) ?? Location.None;
                symbolContext.ReportDiagnostic(Diagnostic.Create(NewTrainOverride, location, name));
            }, SymbolKind.NamedType);

            start.RegisterCompilationEndAction(end =>
            {
                foreach (var listed in TrainOverrideBaseline.Types.Where(t => !overriding.ContainsKey(t)).OrderBy(t => t))
                    end.ReportDiagnostic(Diagnostic.Create(StaleBaselineEntry, Location.None, listed));
            });
        });
    }

    private static bool IsTensorTrainOverride(IMethodSymbol method)
        => method.IsOverride
            && method.Parameters.Length == 2
            && method.ReturnsVoid
            && method.Parameters.All(p => p.Type is INamedTypeSymbol { Name: "Tensor", TypeArguments.Length: 1 });

    private static bool IsNeuralNetwork(INamedTypeSymbol type)
    {
        for (var current = type.BaseType; current is not null; current = current.BaseType)
        {
            if (current.OriginalDefinition.ToDisplayString().StartsWith(NeuralNetworkBasePrefix, System.StringComparison.Ordinal))
                return true;
        }

        return false;
    }
}

/// <summary>
/// The concrete networks that overrode <c>Train</c> when <see cref="TrainOverrideAnalyzer"/> was added (2026-10-16).
/// May only shrink.
/// </summary>
internal static class TrainOverrideBaseline
{
    internal static readonly HashSet<string> Types = new HashSet<string>(System.StringComparer.Ordinal)
    {
    };
}
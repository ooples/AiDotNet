# AiDotNet.Evolution.CSharp

Opt-in compiler-guided C# algorithm evolution for AiDotNet on .NET 8 and .NET 10.

Configure `AiModelBuilder` with `ConfigureCSharpProgramEvolution`: supply your chat client, program options,
pinned reference assemblies, model/target identities, a private evidence directory and a shared resource ledger.
The loop applies bounded syntax-addressed patches, emits with Roslyn and uses bounded compiler feedback for repair.
Failed work remains charged; incomplete usage retains a conservative reserved maximum.

Compilation is not a security sandbox, correctness proof or performance result. Supply an isolated execution engine
and independent correctness checks, and keep final held-out tests outside search. Evidence contains unredacted source,
prompts and model replies; protect the directory accordingly. Automatic ledger/engine checkpoint resume is not supported.

The package does not choose a provider, API key or paid service. Default prices are synthetic work units, not money.
Use matching compatible AiDotNet and AiDotNet.Evolution versions; the original Evolution `0.1.0-preview.1` lacks
the required resource-ledger APIs.

See [configuration, contracts and limitations](https://github.com/ooples/AiDotNet/blob/master/docs/evolution-csharp.md).

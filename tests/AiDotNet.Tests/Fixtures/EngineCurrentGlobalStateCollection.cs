using Xunit;

namespace AiDotNet.Tests.Fixtures;

/// <summary>
/// Test classes that set <c>AiDotNetEngine.Current</c>, a process-wide static, and so must not run beside any other
/// class.
/// </summary>
/// <remarks>
/// Ten classes named this collection, but nothing defined it. A <c>[Collection]</c> name with no definition only
/// serializes its own members; the collection still ran in parallel with every other one. An unrelated model trained
/// at the same time could have its operations dispatched to the engine a GPU test had swapped in: an FTTransformer
/// classifier test was measured learning nothing (its loss pinned at ln 3) beside QuantumStateEncodingRegressionTests.
/// </remarks>
[CollectionDefinition(Name, DisableParallelization = true)]
public class EngineCurrentGlobalStateCollection
{
    /// <summary>Name used in <c>[Collection(...)]</c> attributes on classes that set <c>AiDotNetEngine.Current</c>.</summary>
    public const string Name = "EngineCurrentGlobalState";
}

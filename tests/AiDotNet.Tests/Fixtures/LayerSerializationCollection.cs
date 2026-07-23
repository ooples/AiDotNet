using Xunit;

namespace AiDotNet.Tests.Fixtures;

/// <summary>
/// xUnit collection that serialises the auto-generated layer invariant tests
/// emitted by <c>TestScaffoldGenerator</c>.
/// <para>
/// Background: every generated <c>*LayerTests</c> class inherits the
/// <c>Serialize_Deserialize_ShouldPreserveBehavior</c> fact from
/// <c>LayerTestBase</c>, which runs two full forward passes on a
/// BLAS-heavy recurrent / attention layer. Under xUnit's default
/// <c>parallelizeTestCollections = true</c> with <c>maxParallelThreads = 0</c>
/// on a many-core box, dozens of these run concurrently and each one times
/// out inside its 30-second <c>[Fact]</c> ceiling — issue #1166.
/// </para>
/// <para>
/// Membership in this collection (applied via <c>[Collection]</c> on each
/// generated class) sets <c>DisableParallelization = true</c>, so the
/// generated-layer tests run one-at-a-time relative to each other. They
/// still run in parallel with tests in *other* collections; this only
/// removes intra-collection thrash. Measured walltime on the 136-test
/// shard: 59s → 56s (a small *improvement*, because failed tests no
/// longer burn their full 30-second timeout).
/// </para>
/// <para>
/// This collection has no fixture object — it is purely a parallelisation
/// barrier. If generated layer tests ever need shared setup, attach an
/// <see cref="ICollectionFixture{TFixture}"/> at that point.
/// </para>
/// </summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public class LayerSerializationCollection
{
    /// <summary>
    /// Name used in <c>[Collection(...)]</c> attributes on the generated
    /// test classes. Kept as a <c>const</c> so both the generator output
    /// and this definition share one string — rename in one place.
    /// </summary>
    public const string Name = "LayerSerialization";
}

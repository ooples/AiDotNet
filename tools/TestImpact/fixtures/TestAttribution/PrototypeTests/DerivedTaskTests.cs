using Xunit;

namespace PrototypeTests;

public sealed class DerivedTaskTests
{
    private sealed class CustomTask(Action action) : Task(action);
    private sealed class CustomTaskOfInt(Func<int> action) : Task<int>(action);
    private static CustomTask Create() => new(() => { });
    private static CustomTaskOfInt CreateGeneric() => new(() => 42);

    [Fact, Trait("Scenario", "DerivedTask")]
    public void UnfinishedDerivedReturn() => _ = Create();

    [Fact, Trait("Scenario", "DerivedGenericTask")]
    public void UnfinishedDerivedGenericReturn() => _ = CreateGeneric();

    [Fact, Trait("Scenario", "DerivedTaskComplete")]
    public async Task CompletedTasksKeepTheirOriginalReturnTypes()
    {
        using CustomTask task = Create();
        task.RunSynchronously();
        using CustomTaskOfInt generic = CreateGeneric();
        generic.RunSynchronously();
        await task;
        Assert.Equal(42, await generic);
    }
}

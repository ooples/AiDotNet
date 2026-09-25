using System.Threading.Tasks.Sources;
using Xunit;

namespace PrototypeTests;

public sealed class ValueTaskTests
{
    // No Task backing object exists, so the ordinary Task observer cannot
    // accidentally make these controls pass. No covered code needs to execute.
    private sealed class PendingSource : IValueTaskSource, IValueTaskSource<int>
    {
        public ValueTask Create() => new(this, 0);
        public ValueTask<int> CreateGeneric() => new(this, 0);
        public ValueTaskSourceStatus GetStatus(short token) => ValueTaskSourceStatus.Pending;
        public void OnCompleted(Action<object?> continuation, object? state, short token, ValueTaskSourceOnCompletedFlags flags) { }
        void IValueTaskSource.GetResult(short token) => throw new InvalidOperationException("Never completes.");
        int IValueTaskSource<int>.GetResult(short token) => throw new InvalidOperationException("Never completes.");
    }

    [Fact, Trait("Scenario", "ValueTask")]
    public void Unfinished() => _ = new PendingSource().Create();

    [Fact, Trait("Scenario", "GenericValueTask")]
    public void UnfinishedGeneric() => _ = new PendingSource().CreateGeneric();

    [Fact, Trait("Scenario", "ValueTaskConstructor")]
    public void DirectConstructor() => _ = new ValueTask(new PendingSource(), 0);
}

using AiDotNet.Evolution;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution;

public sealed class FanOutEvolutionObserverTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task AFailureDoesNotSuppressTheOtherObserver(bool firstFails)
    {
        var expected = new InvalidOperationException("authored failure");
        var first = new Observer(firstFails ? expected : null);
        var second = new Observer(firstFails ? null : expected);
        var actual = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            new FanOutEvolutionObserver<int>(first, second).OnEventAsync(new EvolutionEvent<int>(EvolutionEventKind.Stopped, 0)).AsTask());
        Assert.Same(expected, actual);
        Assert.Equal(1, first.Calls); Assert.Equal(1, second.Calls);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task BothCausesArePreservedEvenWhenOneIsFatal(bool firstFatal)
    {
        var fatal = new OutOfMemoryException("synthetic fatal exception; no memory exhaustion performed");
        var recoverable = new InvalidOperationException("authored failure");
        var first = new Observer(firstFatal ? fatal : recoverable);
        var second = new Observer(firstFatal ? recoverable : fatal);
        var actual = await Assert.ThrowsAsync<AggregateException>(() =>
            new FanOutEvolutionObserver<int>(first, second).OnEventAsync(new EvolutionEvent<int>(EvolutionEventKind.Stopped, 0)).AsTask());
        Assert.Contains(fatal, actual.InnerExceptions);
        Assert.Contains(recoverable, actual.InnerExceptions);
        Assert.Equal(1, first.Calls); Assert.Equal(1, second.Calls);
    }

    [Fact]
    public async Task CancellationIsNotRewrittenWhenItIsTheOnlyFailure()
    {
        var canceled = new OperationCanceledException();
        var second = new Observer(null);
        Assert.Same(canceled, await Assert.ThrowsAsync<OperationCanceledException>(() =>
            new FanOutEvolutionObserver<int>(new Observer(canceled), second).OnEventAsync(new EvolutionEvent<int>(EvolutionEventKind.Stopped, 0)).AsTask()));
        Assert.Equal(1, second.Calls);
    }

    private sealed class Observer : IEvolutionObserver<int>
    {
        private readonly Exception? _failure;
        internal Observer(Exception? failure) => _failure = failure;
        internal int Calls { get; private set; }
        public ValueTask OnEventAsync(EvolutionEvent<int> item, CancellationToken cancellationToken = default)
        {
            Calls++;
            if (_failure is not null) throw _failure;
            return default;
        }
    }
}

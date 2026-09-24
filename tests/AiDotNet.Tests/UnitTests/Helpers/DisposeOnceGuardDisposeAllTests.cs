using System;
using AiDotNet.Helpers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Helpers;

/// <summary>
/// <see cref="DisposeOnceGuard.DisposeAll"/> releases every resource, then reports failures without changing
/// what a caller could catch before: one failure is rethrown as the ORIGINAL exception (type and stack preserved),
/// and only two or more are wrapped in an <see cref="AggregateException"/>.
/// </summary>
public sealed class DisposeOnceGuardDisposeAllTests
{
    private sealed class DistinctDisposeException : Exception
    {
        public DistinctDisposeException(string message) : base(message)
        {
        }
    }

    private sealed class Resource : IDisposable
    {
        private readonly string? _failure;
        public Resource(string? failure = null) => _failure = failure;
        public int DisposeCalls { get; private set; }

        public void Dispose()
        {
            DisposeCalls++;
            if (_failure is not null) ThrowFromDispose(_failure);
        }

        private static void ThrowFromDispose(string message) => throw new DistinctDisposeException(message);
    }

    [Fact]
    public void A_single_failure_is_rethrown_unchanged_after_every_resource_is_released()
    {
        var before = new Resource();
        var failing = new Resource("only failure");
        var after = new Resource();

        var thrown = Assert.Throws<DistinctDisposeException>(
            () => DisposeOnceGuard.DisposeAll(new object?[] { before, failing, after }, "owner"));

        Assert.Equal("only failure", thrown.Message);
        // The stack still points at the code that threw, not only at the rethrow site.
        Assert.Contains("ThrowFromDispose", thrown.StackTrace ?? string.Empty);
        Assert.Equal(1, before.DisposeCalls);
        Assert.Equal(1, failing.DisposeCalls);
        Assert.Equal(1, after.DisposeCalls);
    }

    [Fact]
    public void Two_failures_are_wrapped_in_one_aggregate_after_every_resource_is_released()
    {
        var first = new Resource("first");
        var healthy = new Resource();
        var second = new Resource("second");

        var thrown = Assert.Throws<AggregateException>(
            () => DisposeOnceGuard.DisposeAll(new object?[] { first, healthy, second }, "owner"));

        Assert.Equal(new[] { "first", "second" }, new[] { thrown.InnerExceptions[0].Message, thrown.InnerExceptions[1].Message });
        Assert.All(thrown.InnerExceptions, e => Assert.IsType<DistinctDisposeException>(e));
        Assert.Equal(1, healthy.DisposeCalls);
    }

    [Fact]
    public void No_failure_throws_nothing_and_releases_each_resource_once()
    {
        var shared = new Resource();

        DisposeOnceGuard.DisposeAll(new object?[] { shared, null, "not disposable", shared }, "owner");

        Assert.Equal(1, shared.DisposeCalls);
    }
}

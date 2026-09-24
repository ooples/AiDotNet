using System;
using System.Reflection;
using AiDotNet.Finance.Trading.Agents;
using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// <see cref="RecurrentPolicyAgent{T}"/> owns an LSTM cell layer -- pool-rented weights and a registered
/// engine state -- but was not disposable, so a caller had no way to release it short of the garbage collector.
/// </summary>
public sealed class RecurrentPolicyAgentDisposalTests
{
    [Fact]
    public void Dispose_releases_the_recurrent_cell_it_owns()
    {
        var agent = new RecurrentPolicyAgent<double>(stateDim: 3, actionDim: 2, hidden: 4, seed: 1);
        _ = agent.SelectAction(new Vector<double>(new[] { 0.1, 0.2, 0.3 }), explore: false);
        var cell = Assert.IsAssignableFrom<IDisposable>(
            typeof(RecurrentPolicyAgent<double>)
                .GetField("_cell", BindingFlags.Instance | BindingFlags.NonPublic)
                ?.GetValue(agent));

        var disposable = Assert.IsAssignableFrom<IDisposable>(agent);
        disposable.Dispose();

        // False means the cell was already released through the once-only guard.
        Assert.False(DisposeOnceGuard.TryDispose(cell), "The agent's LSTM cell was still live after the agent was disposed.");
        Assert.Null(Record.Exception(() => disposable.Dispose()));
    }
}

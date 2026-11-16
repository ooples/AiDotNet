using System;
using System.Collections.Generic;
using AiDotNet.FederatedLearning.Trainers;
using AiDotNet.Models;
using AiDotNet.Tests.Helpers;
using Microsoft.Extensions.Logging;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

/// <summary>
/// A client whose samples cannot be hashed is left out of the private set intersection, and the failure is logged
/// rather than swallowed.
/// </summary>
public class InMemoryFederatedTrainerLoggingTests
{
    [Fact]
    public void ComputeClientSampleIdentities_LogsAClientItCannotHash_AndKeepsTheOthers()
    {
        var logger = new RecordingLogger();
        var model = new MockFullModel(_ => new Vector<double>(2), parameterCount: 2);
        var trainer = new InMemoryFederatedTrainer<double, Matrix<double>, Vector<double>>(
            new FederatedNoOpOptimizer(model), randomSeed: 1, logger: logger);

        var readable = new Matrix<double>(2, 2);
        readable[0, 0] = 1.0;
        readable[1, 1] = 2.0;
        var clients = new Dictionary<int, FederatedClientDataset<Matrix<double>, Vector<double>>>
        {
            [0] = new FederatedClientDataset<Matrix<double>, Vector<double>>(readable, new Vector<double>(2), 2),
            [1] = new FederatedClientDataset<Matrix<double>, Vector<double>>(TestNull.Of<Matrix<double>>(), new Vector<double>(2), 2),
        };

        var identities = trainer.ComputeClientSampleIdentities(clients);

        Assert.Equal(2, identities[0].Count);
        Assert.False(identities.ContainsKey(1));
        var entry = Assert.Single(logger.Entries);
        Assert.Equal(Microsoft.Extensions.Logging.LogLevel.Warning, entry.Level);
        Assert.NotNull(entry.Exception);
    }

    [Fact]
    public void Trainer_WithoutALogger_StillFallsBack()
    {
        var model = new MockFullModel(_ => new Vector<double>(2), parameterCount: 2);
        var trainer = new InMemoryFederatedTrainer<double, Matrix<double>, Vector<double>>(new FederatedNoOpOptimizer(model), randomSeed: 1);
        var clients = new Dictionary<int, FederatedClientDataset<Matrix<double>, Vector<double>>>
        {
            [1] = new FederatedClientDataset<Matrix<double>, Vector<double>>(TestNull.Of<Matrix<double>>(), new Vector<double>(2), 2),
        };

        Assert.Empty(trainer.ComputeClientSampleIdentities(clients));
    }

    private sealed class RecordingLogger : ILogger
    {
        public List<(Microsoft.Extensions.Logging.LogLevel Level, Exception? Exception)> Entries { get; } = new List<(Microsoft.Extensions.Logging.LogLevel Level, Exception? Exception)>();

        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;

        public bool IsEnabled(Microsoft.Extensions.Logging.LogLevel logLevel) => true;

        public void Log<TState>(Microsoft.Extensions.Logging.LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter)
            => Entries.Add((logLevel, exception));
    }
}

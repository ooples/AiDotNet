using AiDotNet.Evolution;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution;

public sealed class EvolutionCheckpointStoreTests
{
    [Fact]
    public async Task InMemoryStoreRejectsRollbackAndReturnsDefensiveCopies()
    {
        var store = new InMemoryEvolutionCheckpointStore();
        await store.SaveAsync(new EvolutionCheckpoint("run", 2, "compat", "two"));

        EvolutionCheckpoint loaded = Assert.IsType<EvolutionCheckpoint>(await store.LoadLatestAsync("run"));

        Assert.Equal(2, loaded.Sequence);
        Assert.Equal("two", loaded.Payload);
        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            store.SaveAsync(new EvolutionCheckpoint("run", 1, "compat", "one")));
    }

    [Fact]
    public async Task JsonStoreFallsBackToPreviousValidSnapshot()
    {
        using var directory = new TemporaryDirectory();
        string path = Path.Combine(directory.Path, "checkpoint.json");
        var store = new JsonEvolutionCheckpointStore(path);
        await store.SaveAsync(new EvolutionCheckpoint("run", 1, "compat", "one"));
        await store.SaveAsync(new EvolutionCheckpoint("run", 2, "compat", "two"));
        await store.SaveAsync(new EvolutionCheckpoint("run", 3, "compat", "three"));

        File.WriteAllText(path, "{corrupt");
        EvolutionCheckpoint loaded = Assert.IsType<EvolutionCheckpoint>(await store.LoadLatestAsync("run"));

        Assert.Equal(2, loaded.Sequence);
        Assert.Equal("two", loaded.Payload);
        Assert.Empty(Directory.EnumerateFiles(directory.Path, "*.tmp"));
    }

    [Fact]
    public async Task JsonStoreRejectsWrongRunAndInvalidChecksum()
    {
        using var directory = new TemporaryDirectory();
        string path = Path.Combine(directory.Path, "checkpoint.json");
        var store = new JsonEvolutionCheckpointStore(path);
        await store.SaveAsync(new EvolutionCheckpoint("run", 1, "compat", "payload"));

        await Assert.ThrowsAsync<InvalidDataException>(() => store.LoadLatestAsync("other"));

        string json = File.ReadAllText(path).Replace("payload", "tampered");
        File.WriteAllText(path, json);
        await Assert.ThrowsAsync<InvalidDataException>(() => store.LoadLatestAsync("run"));
    }

    [Fact]
    public async Task StoresRejectSameSequenceForkAndCompatibilityDrift()
    {
        var memory = new InMemoryEvolutionCheckpointStore();
        await memory.SaveAsync(new EvolutionCheckpoint("run", 1, "compat", "one"));
        await memory.SaveAsync(new EvolutionCheckpoint("run", 1, "compat", "one"));
        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            memory.SaveAsync(new EvolutionCheckpoint("run", 1, "compat", "fork")));
        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            memory.SaveAsync(new EvolutionCheckpoint("run", 2, "different", "two")));

        using var directory = new TemporaryDirectory();
        var json = new JsonEvolutionCheckpointStore(Path.Combine(directory.Path, "checkpoint.json"));
        await json.SaveAsync(new EvolutionCheckpoint("run", 1, "compat", "one"));
        await json.SaveAsync(new EvolutionCheckpoint("run", 1, "compat", "one"));
        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            json.SaveAsync(new EvolutionCheckpoint("run", 1, "compat", "fork")));
        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            json.SaveAsync(new EvolutionCheckpoint("run", 2, "different", "two")));
    }

    [Fact]
    public async Task JsonStoreLoadsPreviousWhenPrimaryIsMissing()
    {
        using var directory = new TemporaryDirectory();
        string path = Path.Combine(directory.Path, "checkpoint.json");
        var store = new JsonEvolutionCheckpointStore(path);
        await store.SaveAsync(new EvolutionCheckpoint("run", 1, "compat", "one"));
        await store.SaveAsync(new EvolutionCheckpoint("run", 2, "compat", "two"));
        File.Delete(path);

        EvolutionCheckpoint loaded = Assert.IsType<EvolutionCheckpoint>(await store.LoadLatestAsync("run"));

        Assert.Equal(1, loaded.Sequence);
        Assert.Equal("one", loaded.Payload);
    }

    [Fact]
    public async Task JsonStoreRefusesToOverwriteCorruptPrimary()
    {
        using var directory = new TemporaryDirectory();
        string path = Path.Combine(directory.Path, "checkpoint.json");
        var store = new JsonEvolutionCheckpointStore(path);
        await store.SaveAsync(new EvolutionCheckpoint("run", 1, "compat", "one"));
        File.WriteAllText(path, "{corrupt");

        await Assert.ThrowsAsync<InvalidDataException>(() =>
            store.SaveAsync(new EvolutionCheckpoint("run", 2, "compat", "two")));
        Assert.Equal("{corrupt", File.ReadAllText(path));
    }

    [Fact]
    public async Task JsonStoreEnforcesEncodedSizeLimit()
    {
        using var directory = new TemporaryDirectory();
        var store = new JsonEvolutionCheckpointStore(Path.Combine(directory.Path, "checkpoint.json"), 128);

        await Assert.ThrowsAsync<InvalidDataException>(() =>
            store.SaveAsync(new EvolutionCheckpoint("run", 1, "compat", new string('x', 256))));
    }

    [Fact]
    public async Task JsonStoreDetectsEnvelopeMetadataTampering()
    {
        using var directory = new TemporaryDirectory();
        string path = Path.Combine(directory.Path, "checkpoint.json");
        var store = new JsonEvolutionCheckpointStore(path);
        await store.SaveAsync(new EvolutionCheckpoint("run", 1, "compat", "payload"));
        string original = File.ReadAllText(path);
        string tampered = original.Replace("\"Sequence\": 1", "\"Sequence\": 2");
        Assert.NotEqual(original, tampered);
        File.WriteAllText(path, tampered);

        await Assert.ThrowsAsync<InvalidDataException>(() => store.LoadLatestAsync("run"));
    }
}

internal sealed class TemporaryDirectory : IDisposable
{
    public TemporaryDirectory()
    {
        Path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "aidotnet-evolution-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path);
    }

    public string Path { get; }

    public void Dispose() => Directory.Delete(Path, recursive: true);
}

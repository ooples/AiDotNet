using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using AiDotNet.Models;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Models;

public sealed class ReadonlyNumericStateContractTests
{
    public enum StorageKind { Tensor, Vector, Matrix }
    public enum IncompatibleState { Shape, NullSource, NullDestination }

    public ReadonlyNumericStateContractTests() => TestModuleInitializer.EnsureInitialized();

    [Theory]
    [InlineData(StorageKind.Tensor)]
    [InlineData(StorageKind.Vector)]
    [InlineData(StorageKind.Matrix)]
    public void InPlaceRestorePreservesConstructionOwnedReferenceAndValues(StorageKind kind)
    {
        using var source = new Storage(kind, 13);
        using var destination = new Storage(kind, 7);
        object? alias = destination.Reference;
        var payload = Save(source.Registry);
        Restore(destination.Registry, payload);
        Assert.Same(alias, destination.Reference);
        Assert.Equal(source.Values(), destination.Values());
    }

    [Theory]
    [InlineData(StorageKind.Tensor, IncompatibleState.Shape)]
    [InlineData(StorageKind.Vector, IncompatibleState.Shape)]
    [InlineData(StorageKind.Matrix, IncompatibleState.Shape)]
    [InlineData(StorageKind.Tensor, IncompatibleState.NullSource)]
    [InlineData(StorageKind.Vector, IncompatibleState.NullSource)]
    [InlineData(StorageKind.Matrix, IncompatibleState.NullSource)]
    [InlineData(StorageKind.Tensor, IncompatibleState.NullDestination)]
    [InlineData(StorageKind.Vector, IncompatibleState.NullDestination)]
    [InlineData(StorageKind.Matrix, IncompatibleState.NullDestination)]
    public void IncompatibleStateFailsBeforeAnyDestinationMutation(StorageKind kind, IncompatibleState mismatch)
    {
        using var source = new Storage(kind, 13, missing: mismatch == IncompatibleState.NullSource);
        using var destination = new Storage(kind, 7, differentShape: mismatch == IncompatibleState.Shape,
            missing: mismatch == IncompatibleState.NullDestination);
        object? alias = destination.Reference;
        var before = destination.Values();
        var error = Assert.Throws<InvalidDataException>(() => Restore(destination.Registry, Save(source.Registry)));
        Assert.Contains("storage", error.Message, StringComparison.Ordinal);
        Assert.Same(alias, destination.Reference);
        Assert.Equal(before, destination.Values());
    }

    [Theory]
    [InlineData(StorageKind.Tensor)]
    [InlineData(StorageKind.Vector)]
    [InlineData(StorageKind.Matrix)]
    public void MatchingAbsentStorageRemainsAbsent(StorageKind kind)
    {
        using var source = new Storage(kind, 0, missing: true);
        using var destination = new Storage(kind, 0, missing: true);
        Restore(destination.Registry, Save(source.Registry));
        Assert.Null(destination.Reference);
    }

    [Fact]
    public void TensorRestorePreservesStorageDeviceAndInvalidatesCachedVersions()
    {
        using var source = new Tensor<float>(new[] { 2, 2 });
        source.CopyFromArray(new[] { 13f, 14f, 15f, 16f });
        using var destination = new Tensor<float>(new[] { 2, 2 });
        var alias = destination;
        var device = destination.Device;
        Assert.True(MemoryMarshal.TryGetArray((ReadOnlyMemory<float>)destination.Data, out var before));
        int version = destination.Version;
        var saved = new ModelStateRegistry<float>();
        saved.DeclareInPlace("storage", () => source);
        var loaded = new ModelStateRegistry<float>();
        loaded.DeclareInPlace("storage", () => destination);
        Restore(loaded, Save(saved));
        Assert.Same(alias, destination);
        Assert.Equal(device, destination.Device);
        Assert.True(destination.Version > version);
        Assert.True(MemoryMarshal.TryGetArray((ReadOnlyMemory<float>)destination.Data, out var after));
        Assert.Same(before.Array, after.Array);
        Assert.Equal(before.Offset, after.Offset);
        Assert.Equal(source.ToArray(), alias.ToArray());
    }

    [Fact]
    public void TensorViewRestoreUsesLogicalStridesAndKeepsTheViewIdentity()
    {
        using var owner = new Tensor<float>(new[] { 2, 2 });
        using var view = owner.Transpose(new[] { 1, 0 });
        Assert.True(view.IsView);
        Assert.False(view.IsContiguous);
        var alias = view;
        using var source = new Tensor<float>(new[] { 2, 2 });
        source.CopyFromArray(new[] { 13f, 14f, 15f, 16f });
        var saved = new ModelStateRegistry<float>();
        saved.DeclareInPlace("storage", () => source);
        var loaded = new ModelStateRegistry<float>();
        loaded.DeclareInPlace("storage", () => view);
        Restore(loaded, Save(saved));
        Assert.Same(alias, view);
        Assert.Equal(source.ToArray(), view.ToArray());
        Assert.Equal(new[] { 13f, 15f, 14f, 16f }, owner.ToArray());
    }

    [Fact]
    public void TensorCowRestorePreservesObjectIdentityWithoutMutatingTheOriginalOwner()
    {
        using var owner = new Tensor<float>(new[] { 2, 2 });
        owner.CopyFromArray(new[] { 1f, 2f, 3f, 4f });
        using var destination = Assert.IsType<Tensor<float>>(owner.CloneShared());
        var alias = destination;
        // Read the existing diagnostic only; do not force storage materialization through a
        // writable array accessor, which would detach COW before the operation being tested.
        var shared = typeof(TensorBase<float>).GetProperty("IsCowShared", BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException("The tensor COW diagnostic is unavailable.");
        Assert.True(shared.GetValue(owner) is true);
        Assert.True(shared.GetValue(destination) is true);
        Assert.Equal(owner.ToArray(), destination.ToArray());
        using var source = new Tensor<float>(new[] { 2, 2 });
        source.CopyFromArray(new[] { 13f, 14f, 15f, 16f });
        var saved = new ModelStateRegistry<float>();
        saved.DeclareInPlace("storage", () => source);
        var loaded = new ModelStateRegistry<float>();
        loaded.DeclareInPlace("storage", () => destination);
        Restore(loaded, Save(saved));
        Assert.Same(alias, destination);
        Assert.Equal(source.ToArray(), destination.ToArray());
        Assert.True(shared.GetValue(destination) is false);
        Assert.Equal(new[] { 1f, 2f, 3f, 4f }, owner.ToArray());
        owner[0] = 99;
        Assert.Equal(13f, destination[0]);
    }

    [Theory]
    [InlineData(3)]
    [InlineData(5)]
    public void InconsistentStoredTensorLengthCannotPartiallyOverwriteDestination(int storedLength)
    {
        using var destination = new Tensor<float>(new[] { 4 });
        destination.CopyFromArray(new[] { 7f, 8f, 9f, 10f });
        var registry = new ModelStateRegistry<float>();
        registry.DeclareInPlace("storage", () => destination);
        using var block = new MemoryStream();
        using (var writer = new BinaryWriter(block, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            writer.Write(1); // rank
            writer.Write(4); // shape
            writer.Write(storedLength);
            for (int index = 0; index < storedLength; index++) writer.Write((double)(13 + index));
        }
        using var envelope = new MemoryStream();
        using (var writer = new BinaryWriter(envelope, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            writer.Write(1); // named entries
            writer.Write("storage");
            writer.Write(checked((int)block.Length));
            writer.Write(block.ToArray());
        }
        Assert.Throws<InvalidDataException>(() => Restore(registry, envelope.ToArray()));
        Assert.Equal(new[] { 7f, 8f, 9f, 10f }, destination.ToArray());
    }

    private static byte[] Save(ModelStateRegistry<float> registry)
    {
        using var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true))
            registry.WriteAll(writer);
        return stream.ToArray();
    }

    private static void Restore(ModelStateRegistry<float> registry, byte[] payload)
    {
        using var stream = new MemoryStream(payload);
        using var reader = new BinaryReader(stream);
        registry.ReadAll(reader);
        Assert.Equal(stream.Length, stream.Position);
    }

    private sealed class Storage : IDisposable
    {
        private readonly Tensor<float>? _tensor;
        private readonly Vector<float>? _vector;
        private readonly Matrix<float>? _matrix;
        public ModelStateRegistry<float> Registry { get; } = new();
        public object? Reference => (object?)_tensor ?? (object?)_vector ?? _matrix;

        public Storage(StorageKind kind, float value, bool differentShape = false, bool missing = false)
        {
            switch (kind)
            {
                case StorageKind.Tensor:
                    if (!missing)
                    {
                        _tensor = new Tensor<float>(differentShape ? new[] { 1, 4 } : new[] { 2, 2 });
                        _tensor.CopyFromArray(Enumerable.Range(0, 4).Select(index => value + index).ToArray());
                    }
                    Registry.DeclareInPlace("storage", () => _tensor);
                    break;
                case StorageKind.Vector:
                    if (!missing) _vector = new Vector<float>(Enumerable.Range(0, differentShape ? 5 : 4)
                        .Select(index => value + index).ToArray());
                    Registry.DeclareInPlace("storage", () => _vector);
                    break;
                case StorageKind.Matrix:
                    if (!missing)
                    {
                        _matrix = new Matrix<float>(differentShape ? 1 : 2, differentShape ? 4 : 2);
                        for (int row = 0; row < _matrix.Rows; row++)
                            for (int column = 0; column < _matrix.Columns; column++)
                                _matrix[row, column] = value + row * _matrix.Columns + column;
                    }
                    Registry.DeclareInPlace("storage", () => _matrix);
                    break;
                default: throw new ArgumentOutOfRangeException(nameof(kind));
            }
        }

        public float[] Values()
        {
            if (_tensor is not null) return _tensor.ToArray();
            if (_vector is not null) return _vector.ToArray();
            return _matrix is not null ? _matrix.AsSpan().ToArray() : Array.Empty<float>();
        }
        public void Dispose() => _tensor?.Dispose();
    }
}

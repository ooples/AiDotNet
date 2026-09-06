using System;
using AiDotNet.Models;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Models;

/// <summary>
/// Regression tests for #1830 — <see cref="ModelMetadata{T}.ModelData"/> must not be produced
/// until something actually reads it.
/// </summary>
/// <remarks>
/// <para>
/// Serializing a model is a licensed persistence operation, and <c>AiModelResult</c>'s constructor
/// captures metadata for EVERY model it wraps (<c>AiModelResult.cs:1710</c>,
/// <c>ModelMetaData = Model?.GetModelMetadata() ?? new()</c>). While model implementations built
/// the bytes eagerly — <c>ModelData = this.Serialize()</c> inside an object initializer — merely
/// calling <c>AiModelBuilder.BuildAsync</c> ran a serialization the caller never asked for, so an
/// expired trial threw <c>LicenseRequiredException</c> out of the primary training entry point.
/// </para>
/// <para>
/// Deferring is strictly better than catching the exception, which is what the pre-existing
/// <c>SerializeForMetadata</c> wrapper does: it also stops every build from serializing the whole
/// model and discarding the bytes unread, and it lets the licence error surface at the point a
/// caller genuinely asks for the weights instead of silently yielding an empty array.
/// </para>
/// </remarks>
public class ModelMetadataLazyModelDataTests
{
    [Fact]
    public void ModelDataProvider_IsNotInvokedUntilModelDataIsRead()
    {
        int invocations = 0;
        var metadata = new ModelMetadata<double>
        {
            ModelDataProvider = () => { invocations++; return new byte[] { 1, 2, 3 }; },
        };

        // The whole point: constructing metadata must not serialize.
        Assert.Equal(0, invocations);
        Assert.False(metadata.IsModelDataMaterialized);

        // Reading other properties must not either — infrastructure inspects metadata freely.
        Assert.NotNull(metadata.AdditionalInfo);
        Assert.Equal(0, invocations);

        Assert.Equal(new byte[] { 1, 2, 3 }, metadata.ModelData);
        Assert.Equal(1, invocations);
        Assert.True(metadata.IsModelDataMaterialized);
    }

    [Fact]
    public void ModelData_IsMaterializedOnlyOnce()
    {
        int invocations = 0;
        var metadata = new ModelMetadata<double>
        {
            ModelDataProvider = () => { invocations++; return new byte[] { 7 }; },
        };

        _ = metadata.ModelData;
        _ = metadata.ModelData;
        _ = metadata.ModelData;

        Assert.Equal(1, invocations);
    }

    [Fact]
    public void AThrowingProvider_IsNotRetriedOnEveryRead()
    {
        // An expired licence is the expected failure, and it is raised from Serialize() deep inside
        // the provider. Retrying on each read would turn one licence check into an unbounded number
        // of them, and would re-throw from property getters that callers treat as cheap.
        int invocations = 0;
        var metadata = new ModelMetadata<double>
        {
            ModelDataProvider = () => { invocations++; throw new InvalidOperationException("boom"); },
        };

        Assert.Throws<InvalidOperationException>(() => metadata.ModelData);
        Assert.Equal(1, invocations);

        // Second read does not run the provider again; the metadata is simply empty.
        Assert.Empty(metadata.ModelData);
        Assert.Equal(1, invocations);
    }

    [Fact]
    public void DirectAssignment_StillWorksAndDiscardsAPendingProvider()
    {
        // Back-compat: `ModelData = bytes` is the long-standing shape and must keep working.
        int invocations = 0;
        var metadata = new ModelMetadata<double>
        {
            ModelDataProvider = () => { invocations++; return new byte[] { 9, 9 }; },
        };

        metadata.ModelData = new byte[] { 4, 5 };

        Assert.Equal(new byte[] { 4, 5 }, metadata.ModelData);
        Assert.Equal(0, invocations);
        Assert.True(metadata.IsModelDataMaterialized);
    }

    [Fact]
    public void SetModelDataProvider_DefersOnAnAlreadyConstructedInstance()
    {
        // The statement form, used where the metadata object already exists (e.g. MegaTTS2).
        int invocations = 0;
        var metadata = new ModelMetadata<double>();
        metadata.SetModelDataProvider(() => { invocations++; return new byte[] { 8 }; });

        Assert.Equal(0, invocations);
        Assert.Equal(new byte[] { 8 }, metadata.ModelData);
        Assert.Equal(1, invocations);
    }

    [Fact]
    public void DefaultMetadata_HasEmptyModelDataAndIsMaterialized()
    {
        var metadata = new ModelMetadata<double>();

        Assert.Empty(metadata.ModelData);
        Assert.True(metadata.IsModelDataMaterialized);
    }
}

using AiDotNet.Models;
using AiDotNet.Models.Options;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Models;

/// <summary>
/// Issue #1993 asks whether the golden copy-constructor pattern applies to every options class.
/// The question it was blocked on is answerable from how cloning actually works now: models no
/// longer hand-write <c>CreateNewInstance</c>, so nothing calls an options copy constructor —
/// <see cref="CloneEngine.CopyConfiguration"/> walks the generated clone plan and copies each
/// property, deeply for arrays and collections and <b>by reference</b> for everything else.
/// </summary>
/// <remarks>
/// <para>
/// That makes the real residual defect precise, and much narrower than 630 classes: a property
/// whose type is itself an options type is copied by reference, so a configuration and its copy
/// share that nested object and a mutation through one is visible through the other. 50 settable
/// properties across <c>src/Models/Options</c> are in that shape.
/// </para>
/// </remarks>
public sealed class NestedOptionsCloneSharingTests
{
    /// <summary>
    /// A copied configuration must not share its nested options object with the original.
    /// </summary>
    /// <remarks>
    /// Control arm. <c>QuantileRegressionOptions</c> has a copy constructor that deep-copies
    /// <c>SolverOptions</c>, so this passed before the fix too; it is here to show the fix did not
    /// break the path that already worked. The new fallback is exercised by
    /// <see cref="APlainOptionsClassAlsoGetsAnIndependentNestedCopy"/>, which fails without it.
    /// </remarks>
    [Fact(Timeout = 60000)]
    public async Task ACopiedConfigurationDoesNotShareItsNestedOptions()
    {
        await Task.Yield();

        var original = new QuantileRegressionOptions<double>
        {
            SolverOptions = new SimplexSolverOptions { MaxIterations = 123 },
        };

        var copy = (QuantileRegressionOptions<double>)CloneEngine.CopyConfiguration(original);
        copy.SolverOptions.MaxIterations = 999;

        Assert.Equal(123, original.SolverOptions.MaxIterations);
    }

    /// <summary>
    /// Control arm: the copy must actually carry the configured value, so the test above cannot
    /// pass on a copy that simply dropped the nested object back to its default.
    /// </summary>
    [Fact(Timeout = 60000)]
    public async Task ACopiedConfigurationCarriesTheNestedValue()
    {
        await Task.Yield();

        var original = new QuantileRegressionOptions<double>
        {
            SolverOptions = new SimplexSolverOptions { MaxIterations = 123 },
        };

        var copy = (QuantileRegressionOptions<double>)CloneEngine.CopyConfiguration(original);

        Assert.NotNull(copy.SolverOptions);
        Assert.Equal(123, copy.SolverOptions.MaxIterations);
    }

    /// <summary>
    /// The same guarantee for a type from the measured gap: <c>TeeOptions</c> is one of the 11
    /// that own a nested options property with no copy constructor, and it is a plain class rather
    /// than a <c>ModelOptions</c> derivative, so no base-class fix could have reached it.
    /// </summary>
    [Fact(Timeout = 60000)]
    public async Task APlainOptionsClassAlsoGetsAnIndependentNestedCopy()
    {
        await Task.Yield();

        var original = new TeeOptions();
        original.Attestation.MaxQuoteAgeSec = 42;

        var copy = (TeeOptions)CloneEngine.CopyConfiguration(original);
        Assert.Equal(42, copy.Attestation.MaxQuoteAgeSec);

        copy.Attestation.MaxQuoteAgeSec = 7;
        Assert.Equal(42, original.Attestation.MaxQuoteAgeSec);
        Assert.NotSame(original.Attestation, copy.Attestation);
    }
}

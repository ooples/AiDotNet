using System;
using System.IO;
using AiDotNet.Helpers;
using Xunit;

namespace AiDotNet.Tests.Helpers;

/// <summary>
/// Tests the online-validation attestation that closes the "block the license server → unbounded ValidationPending
/// → free persistence" bypass. The attestation must: honour a recent record for the SAME key, reject a different
/// key, reject when absent (never validated), reject once the window lapses, and reject a tampered file.
/// </summary>
public sealed class OnlineValidationAttestationTests : IDisposable
{
    private readonly string _dir;
    private readonly string _path;
    private readonly IDisposable _override;

    public OnlineValidationAttestationTests()
    {
        _dir = Path.Combine(Path.GetTempPath(), "aidn-att-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_dir);
        _path = Path.Combine(_dir, "validation.att");
        _override = OnlineValidationAttestation.OverridePathForTesting(_path);
    }

    public void Dispose()
    {
        _override.Dispose();
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    [Fact]
    public void Recorded_key_is_valid_within_window()
    {
        OnlineValidationAttestation.Record("aidn.abc.def");
        Assert.True(OnlineValidationAttestation.HasValidWithin("aidn.abc.def", TimeSpan.FromDays(30)));
    }

    [Fact]
    public void Different_key_is_not_honoured()
    {
        OnlineValidationAttestation.Record("aidn.abc.def");
        Assert.False(OnlineValidationAttestation.HasValidWithin("AIDN-PROD-somethingelse", TimeSpan.FromDays(30)));
    }

    [Fact]
    public void Absent_attestation_fails_closed()
    {
        // Never recorded — this is the bypass case (blocked server, never validated) and MUST be denied.
        Assert.False(OnlineValidationAttestation.HasValidWithin("aidn.abc.def", TimeSpan.FromDays(30)));
    }

    [Fact]
    public void Lapsed_window_is_rejected()
    {
        OnlineValidationAttestation.Record("aidn.abc.def");
        // A zero-length window means "must have validated in the last instant" — the just-written record is already
        // outside it, proving the time bound is enforced (a permanently-blocked server eventually fails closed).
        Assert.False(OnlineValidationAttestation.HasValidWithin("aidn.abc.def", TimeSpan.Zero));
    }

    [Fact]
    public void Tampered_file_is_rejected()
    {
        OnlineValidationAttestation.Record("aidn.abc.def");
        // Flip the signature line — the HMAC no longer verifies, so the record is rejected.
        var lines = File.ReadAllText(_path);
        File.WriteAllText(_path, lines + "TAMPER");
        Assert.False(OnlineValidationAttestation.HasValidWithin("aidn.abc.def", TimeSpan.FromDays(30)));

        // Garbage content is also rejected.
        File.WriteAllText(_path, "not-a-valid-envelope");
        Assert.False(OnlineValidationAttestation.HasValidWithin("aidn.abc.def", TimeSpan.FromDays(30)));
    }
}

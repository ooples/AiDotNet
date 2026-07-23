using System.Globalization;
using System.Security.Cryptography;
using System.Text;

namespace AiDotNet.Helpers;

/// <summary>
/// Tamper-evident, machine-bound record that a license key has completed at least one SUCCESSFUL ONLINE validation
/// against the license server.
/// </summary>
/// <remarks>
/// <para><b>Why:</b> the persistence gate honours <c>LicenseKeyStatus.ValidationPending</c> — the status returned
/// when the license server is unreachable and the key was never validated in this process — by ALLOWING the
/// operation during an offline grace window. On its own that is a bypass: simply blocking the license server (a
/// firewall rule, an <c>/etc/hosts</c> entry) yields <c>ValidationPending</c> forever and therefore unbounded
/// unlicensed persistence. This attestation closes it: <c>ValidationPending</c> is honoured ONLY when the key has a
/// recorded successful ONLINE validation within <see cref="AttestationValidity"/>. A key that has ever validated
/// online — e.g. a community/open-source key — keeps working fully offline for <see cref="AttestationValidity"/>
/// after each success, so genuinely-offline licensed users are not inconvenienced; a never-validated key that can't
/// reach the server is denied.</para>
/// <para><b>Protection:</b> the plaintext key is NEVER written — only its SHA-256 hash. The record is HMAC-SHA256
/// signed with a key derived from the machine fingerprint (same scheme as <see cref="TrialStateManager"/>), so the
/// file cannot be forged, its timestamp cannot be back-dated, and it cannot be copied to another machine. Signed
/// (aidn.*) keys are unaffected — they verify offline by HMAC and return <c>Active</c>, never <c>ValidationPending</c>.</para>
/// </remarks>
internal static class OnlineValidationAttestation
{
    /// <summary>How long a successful online validation lets the key keep operating while the server is
    /// unreachable. Generous (a licensed user validates online at least monthly) but bounded, so a permanently
    /// blocked server eventually fails closed.</summary>
    internal static readonly TimeSpan AttestationValidity = TimeSpan.FromDays(30);

    private static readonly object FileLock = new();
    private static string? _pathOverrideForTests;

    private static string GetPath() => _pathOverrideForTests ?? Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".aidotnet", "validation.att");

    /// <summary>TEST-ONLY: redirects the attestation file so tests don't touch the real user profile.</summary>
    internal static IDisposable OverridePathForTesting(string path)
    {
        var previous = _pathOverrideForTests;
        _pathOverrideForTests = path;
        return new PathOverrideScope(previous);
    }

    /// <summary>Records a successful ONLINE validation of <paramref name="licenseKey"/>. Best-effort: an IO failure
    /// never breaks the validation that just succeeded.</summary>
    public static void Record(string licenseKey)
    {
        if (string.IsNullOrWhiteSpace(licenseKey))
        {
            return;
        }

        try
        {
            string data = KeyHash(licenseKey) + "|" +
                DateTimeOffset.UtcNow.ToUnixTimeSeconds().ToString(CultureInfo.InvariantCulture);
            string payload = data + "\n" + Sign(data);
            string path = GetPath();
            string? dir = Path.GetDirectoryName(path);
            lock (FileLock)
            {
                if (!string.IsNullOrEmpty(dir))
                {
                    Directory.CreateDirectory(dir);
                }

                File.WriteAllText(path, payload);
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Trace.TraceWarning(
                "OnlineValidationAttestation: failed to record: " + ex.GetType().Name + ": " + ex.Message);
        }
    }

    /// <summary>Returns true when a valid, machine-matched, key-matched attestation exists whose timestamp is within
    /// <paramref name="window"/> of now (and not implausibly in the future). Fails closed on any error.</summary>
    public static bool HasValidWithin(string licenseKey, TimeSpan window)
    {
        if (string.IsNullOrWhiteSpace(licenseKey))
        {
            return false;
        }

        try
        {
            string path = GetPath();
            string payload;
            lock (FileLock)
            {
                if (!File.Exists(path))
                {
                    return false;
                }

                payload = File.ReadAllText(path);
            }

            int newline = payload.IndexOf('\n');
            if (newline <= 0)
            {
                return false;
            }

            string data = payload[..newline];
            string signature = payload[(newline + 1)..].Trim();

            // Tamper / wrong-machine check: the HMAC is keyed to this machine, so a copied or edited file fails here.
            if (!ConstantTimeEquals(Sign(data), signature))
            {
                return false;
            }

            string[] parts = data.Split('|');
            if (parts.Length != 2)
            {
                return false;
            }

            // The attestation must belong to THIS license key.
            if (!ConstantTimeEquals(parts[0], KeyHash(licenseKey)))
            {
                return false;
            }

            if (!long.TryParse(parts[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out long unix))
            {
                return false;
            }

            DateTimeOffset when = DateTimeOffset.FromUnixTimeSeconds(unix);
            DateTimeOffset now = DateTimeOffset.UtcNow;

            // A timestamp in the future means the clock was rolled back (or the record forged) — reject.
            if (when > now.AddMinutes(5))
            {
                return false;
            }

            return when + window > now;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>SHA-256 hash of the key (domain-separated) so the plaintext key never touches disk.</summary>
    private static string KeyHash(string key)
    {
        byte[] bytes = Encoding.UTF8.GetBytes("license-attest:" + key);
#if NET471
        using var sha = SHA256.Create();
        return Convert.ToBase64String(sha.ComputeHash(bytes));
#else
        return Convert.ToBase64String(SHA256.HashData(bytes));
#endif
    }

    /// <summary>HMAC-SHA256 with a machine-derived key (same domain-separated scheme as TrialStateManager) — binds
    /// the attestation to this machine and makes the timestamp unforgeable.</summary>
    private static string Sign(string data)
    {
        string machineId = MachineFingerprint.GetMachineId();
        byte[] keyMaterial = Encoding.UTF8.GetBytes("AiDotNet.OnlineValidation.v1:" + machineId);
#if NET471
        using var sha = SHA256.Create();
        byte[] signingKey = sha.ComputeHash(keyMaterial);
#else
        byte[] signingKey = SHA256.HashData(keyMaterial);
#endif
        using var hmac = new HMACSHA256(signingKey);
        return Convert.ToBase64String(hmac.ComputeHash(Encoding.UTF8.GetBytes(data)));
    }

    private static bool ConstantTimeEquals(string a, string b)
    {
        if (a.Length != b.Length)
        {
            return false;
        }

        int result = 0;
        for (int i = 0; i < a.Length; i++)
        {
            result |= a[i] ^ b[i];
        }

        return result == 0;
    }

    private sealed class PathOverrideScope : IDisposable
    {
        private readonly string? _previous;
        public PathOverrideScope(string? previous) => _previous = previous;
        public void Dispose() => _pathOverrideForTests = _previous;
    }
}

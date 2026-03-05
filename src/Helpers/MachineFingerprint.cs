using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;

namespace AiDotNet.Helpers;

/// <summary>
/// Generates an advisory, deterministic machine identifier used for soft device-binding telemetry.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This class produces a stable hash that identifies the current machine.
/// It is used for advisory seat-counting (how many machines are using a license) but is never used
/// to block or enforce access. It is purely informational.</para>
///
/// <para><b>Platform strategies:</b></para>
/// <list type="bullet">
/// <item><description>Windows: reads <c>MachineGuid</c> from the registry</description></item>
/// <item><description>Linux: reads <c>/etc/machine-id</c></description></item>
/// <item><description>macOS: runs <c>ioreg</c> for <c>IOPlatformUUID</c></description></item>
/// <item><description>Fallback: SHA-256 of hostname + username + OS description</description></item>
/// </list>
/// </remarks>
internal static class MachineFingerprint
{
    /// <summary>
    /// Returns a deterministic, hex-encoded SHA-256 fingerprint for the current machine.
    /// </summary>
    public static string GetMachineId()
    {
        string? raw = GetPlatformMachineId();
        if (string.IsNullOrWhiteSpace(raw))
        {
            raw = GetFallbackId();
        }

        return HashToHex(raw ?? "unknown");
    }

    private static string? GetPlatformMachineId()
    {
#if NET471
        // .NET Framework — use fallback (registry APIs differ; keep it simple)
        return null;
#else
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            return ReadWindowsMachineGuid();
        }

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            return ReadLinuxMachineId();
        }

        if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            return ReadMacOsPlatformUuid();
        }

        return null;
#endif
    }

#if !NET471
    [System.Runtime.Versioning.SupportedOSPlatform("windows")]
    private static string? ReadWindowsMachineGuid()
    {
        try
        {
            using var key = Microsoft.Win32.Registry.LocalMachine.OpenSubKey(
                @"SOFTWARE\Microsoft\Cryptography");
            return key?.GetValue("MachineGuid") as string;
        }
        catch
        {
            return null;
        }
    }

    [System.Runtime.Versioning.SupportedOSPlatform("linux")]
    private static string? ReadLinuxMachineId()
    {
        try
        {
            const string path = "/etc/machine-id";
            if (File.Exists(path))
            {
                string content = File.ReadAllText(path).Trim();
                if (content.Length > 0)
                {
                    return content;
                }
            }
        }
        catch
        {
            // Ignore
        }

        return null;
    }

    private static string? ReadMacOsPlatformUuid()
    {
        try
        {
            var psi = new System.Diagnostics.ProcessStartInfo
            {
                FileName = "ioreg",
                Arguments = "-rd1 -c IOPlatformExpertDevice",
                RedirectStandardOutput = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            using var process = System.Diagnostics.Process.Start(psi);
            if (process is null)
            {
                return null;
            }

            string output = process.StandardOutput.ReadToEnd();
            if (!process.WaitForExit(5000))
            {
                try { process.Kill(); } catch { /* best effort */ }
                return null;
            }

            // Parse IOPlatformUUID from output
            const string marker = "\"IOPlatformUUID\" = \"";
            int start = output.IndexOf(marker, StringComparison.Ordinal);
            if (start >= 0)
            {
                start += marker.Length;
                int end = output.IndexOf('"', start);
                if (end > start)
                {
                    return output.Substring(start, end - start);
                }
            }
        }
        catch
        {
            // Ignore
        }

        return null;
    }
#endif

    private static string GetFallbackId()
    {
        string hostname;
        string username;
        try { hostname = System.Environment.MachineName; } catch { hostname = "unknown"; }
        try { username = System.Environment.UserName; } catch { username = "unknown"; }
        string os = RuntimeInformation.OSDescription ?? "unknown";
        return $"{hostname}|{username}|{os}";
    }

    private static string HashToHex(string input)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(input);

#if NET471
        using var sha = SHA256.Create();
        byte[] hash = sha.ComputeHash(bytes);
#else
        byte[] hash = SHA256.HashData(bytes);
#endif

        var sb = new StringBuilder(hash.Length * 2);
        for (int i = 0; i < hash.Length; i++)
        {
            sb.Append(hash[i].ToString("x2"));
        }

        return sb.ToString();
    }
}

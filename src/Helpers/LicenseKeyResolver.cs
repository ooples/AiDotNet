using AiDotNet.Models;

namespace AiDotNet.Helpers;

/// <summary>
/// Resolves a license key string from multiple sources using a priority-based fallback chain.
/// </summary>
/// <remarks>
/// <para><b>Resolution order:</b></para>
/// <list type="number">
/// <item><description>Explicit <see cref="AiDotNetLicenseKey"/> instance (key property)</description></item>
/// <item><description><c>AIDOTNET_LICENSE_KEY</c> environment variable</description></item>
/// <item><description><c>~/.aidotnet/license.key</c> file (first non-empty line)</description></item>
/// </list>
/// <para>Returns null when no source provides a key.</para>
/// </remarks>
internal static class LicenseKeyResolver
{
    /// <summary>
    /// The environment variable name checked for a license key.
    /// </summary>
    internal const string EnvVarName = "AIDOTNET_LICENSE_KEY";

    /// <summary>
    /// The file name (relative to the user home directory) checked for a license key.
    /// </summary>
    internal const string LicenseFileName = ".aidotnet/license.key";

    /// <summary>
    /// Resolves a license key string from the fallback chain.
    /// </summary>
    /// <param name="licenseKey">An explicit license key object, or null.</param>
    /// <returns>The resolved key string, or null if none was found.</returns>
    public static string? Resolve(AiDotNetLicenseKey? licenseKey)
    {
        // 1. Explicit license key object
        string? explicitKey = licenseKey?.Key;
        if (explicitKey is not null && explicitKey.Trim().Length > 0)
        {
            return explicitKey.Trim();
        }

        // 2. Environment variable
        string? envValue = System.Environment.GetEnvironmentVariable(EnvVarName);
        if (envValue is not null && envValue.Trim().Length > 0)
        {
            return envValue.Trim();
        }

        // 3. File in user home directory
        string? fileValue = ReadLicenseFile();
        if (fileValue is not null && fileValue.Trim().Length > 0)
        {
            return fileValue.Trim();
        }

        return null;
    }

    private static string? ReadLicenseFile()
    {
        try
        {
            string home = System.Environment.GetFolderPath(System.Environment.SpecialFolder.UserProfile);
            if (string.IsNullOrWhiteSpace(home))
            {
                return null;
            }

            string path = Path.Combine(home, LicenseFileName);
            if (!File.Exists(path))
            {
                return null;
            }

            // Read first non-empty, non-comment line
            foreach (string line in File.ReadLines(path))
            {
                string trimmed = line.Trim();
                if (trimmed.Length > 0 && !trimmed.StartsWith("#", StringComparison.Ordinal))
                {
                    return trimmed;
                }
            }
        }
        catch (IOException ex)
        {
            System.Diagnostics.Trace.TraceWarning("LicenseKeyResolver: failed to read license file: {0}", ex.Message);
        }
        catch (UnauthorizedAccessException ex)
        {
            System.Diagnostics.Trace.TraceWarning("LicenseKeyResolver: insufficient permissions for license file: {0}", ex.Message);
        }

        return null;
    }
}

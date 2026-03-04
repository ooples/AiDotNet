using System.Reflection;

namespace AiDotNet.Helpers;

/// <summary>
/// Provides access to the build-time signing key embedded as a resource during official CI/CD builds.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> When AiDotNet is built through the official CI/CD pipeline, a secret signing key
/// is embedded into the DLL as a resource. This key is used to derive encryption keys that are unique
/// to official builds. Fork or dev builds will not have this key, which means they derive different
/// encryption keys and cannot decrypt models encrypted by official builds.
///
/// This is Layer 1 of the three-layer obfuscation system.
/// </remarks>
internal static class BuildKeyProvider
{
    private const string ResourceName = "AiDotNet.BuildKey";
    private static byte[]? _cachedKey;
    private static bool _loaded;
    private static readonly object _lock = new();

    /// <summary>
    /// Gets whether this is an official build with an embedded build key.
    /// </summary>
    internal static bool IsOfficialBuild
    {
        get
        {
            var key = GetBuildKey();
            return key.Length > 0;
        }
    }

    /// <summary>
    /// Gets the embedded build key, or an empty array if not present (dev/fork build).
    /// </summary>
    internal static byte[] GetBuildKey()
    {
        if (_loaded)
        {
            return _cachedKey ?? Array.Empty<byte>();
        }

        lock (_lock)
        {
            if (_loaded)
            {
                return _cachedKey ?? Array.Empty<byte>();
            }

            try
            {
                var assembly = typeof(BuildKeyProvider).Assembly;
                using var stream = assembly.GetManifestResourceStream(ResourceName);
                if (stream is null || stream.Length == 0)
                {
                    _cachedKey = null;
                    return Array.Empty<byte>();
                }

                var buffer = new byte[stream.Length];
                int bytesRead = 0;
                while (bytesRead < buffer.Length)
                {
                    int read = stream.Read(buffer, bytesRead, buffer.Length - bytesRead);
                    if (read == 0)
                    {
                        break;
                    }

                    bytesRead += read;
                }

                // Guard against partial reads — treat as missing key
                if (bytesRead < buffer.Length)
                {
                    _cachedKey = null;
                    return Array.Empty<byte>();
                }

                // Publish to cache after full read to avoid exposing partial data
                _cachedKey = buffer;
                return _cachedKey;
            }
            catch
            {
                _cachedKey = null;
                return Array.Empty<byte>();
            }
            finally
            {
                _loaded = true;
            }
        }
    }
}

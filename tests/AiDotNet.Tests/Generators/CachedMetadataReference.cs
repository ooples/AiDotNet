using System.Collections.Concurrent;
using Microsoft.CodeAnalysis;

namespace AiDotNet.Tests.Generators;

/// <summary>
/// One <see cref="MetadataReference"/> per assembly file for the whole test process.
/// </summary>
/// <remarks>
/// The generator and analyzer tests compile against every loaded assembly, including the 44 MB AiDotNet.dll, and used to
/// create a fresh reference for each file on every compilation. Roslyn caches an assembly's metadata and symbol tables
/// per reference instance, so each test re-read and re-mapped all of them, and the mappings are released only when a
/// finalizer runs. Across the Unassigned - 01 shard that native memory accumulated past the 16 GB runner. References
/// are immutable and designed to be shared between compilations; the files are build outputs that do not change while
/// the process runs.
/// </remarks>
internal static class CachedMetadataReference
{
    private static readonly ConcurrentDictionary<string, MetadataReference> Cache = new(StringComparer.OrdinalIgnoreCase);

    internal static MetadataReference FromFile(string path) =>
        Cache.GetOrAdd(path, static p => MetadataReference.CreateFromFile(p));
}

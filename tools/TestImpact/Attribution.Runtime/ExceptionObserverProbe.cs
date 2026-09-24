using System.Reflection;
using System.Runtime.ExceptionServices;
using System.Security.Cryptography;

namespace AttributionRuntime;

public enum ExceptionObserverState { Unknown, NoneObserved, Present }

// Boundary observation only. Absence here is not proof that another thread or
// the body cannot install a callback later. Body/lifetime contracts must prove
// that separately. The catalog binds the backing-field layout to exact runtime
// bytes; unknown runtimes never inherit an absent-observer claim.
internal static class ExceptionObserverProbe
{
    internal const string RuntimeHash = "1125acc8106c43fc8bad2d203c4c4485df6182d292846c2fff415c1040c54678";
    private static readonly FieldInfo? FirstChance = Bind(typeof(object).Assembly);

    internal static ExceptionObserverState Capture()
    {
        try
        {
            return FirstChance is null ? ExceptionObserverState.Unknown :
                FirstChance.GetValue(null) is null ? ExceptionObserverState.NoneObserved : ExceptionObserverState.Present;
        }
        catch (Exception error) when (error is MemberAccessException or TargetException or TargetInvocationException or NotSupportedException)
        {
            return ExceptionObserverState.Unknown;
        }
    }

    internal static FieldInfo? Bind(Assembly runtime)
    {
        try
        {
            if (!ReferenceEquals(runtime, typeof(object).Assembly)) return null;
            string file = runtime.Location;
            for (FileSystemInfo? entry = new FileInfo(Path.GetFullPath(file)); entry is not null;
                 entry = entry is FileInfo leaf ? leaf.Directory : ((DirectoryInfo)entry).Parent)
                if (!entry.Exists || (entry.Attributes & FileAttributes.ReparsePoint) != 0 || entry.LinkTarget is not null) return null;
            using var stream = File.OpenRead(file);
            if (Convert.ToHexStringLower(SHA256.HashData(stream)) != RuntimeHash) return null;
            FieldInfo? field = typeof(AppContext).GetField(nameof(AppDomain.FirstChanceException), BindingFlags.Static | BindingFlags.NonPublic);
            return field is { IsStatic: true, IsPrivate: true } && field.DeclaringType == typeof(AppContext) &&
                field.FieldType == typeof(EventHandler<FirstChanceExceptionEventArgs>) ? field : null;
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or ArgumentException or
            NotSupportedException or MemberAccessException)
        {
            return null;
        }
    }
}

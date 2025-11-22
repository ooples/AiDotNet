// Compatibility shim for init-only setters in .NET Framework 4.6.2
// This type is required for C# 9+ init accessors to work in older frameworks
// See: https://github.com/dotnet/runtime/issues/45510

namespace System.Runtime.CompilerServices
{
    /// <summary>
    /// Reserved for use by the compiler for tracking metadata.
    /// This class allows the use of init-only setters in .NET Framework 4.6.2.
    /// </summary>
    internal static class IsExternalInit
    {
    }
}

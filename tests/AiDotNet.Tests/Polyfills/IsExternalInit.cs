#if !NET5_0_OR_GREATER

namespace System.Runtime.CompilerServices;

/// <summary>
/// Marker the compiler requires to emit <c>init</c> accessors, which positional records rely on.
/// </summary>
/// <remarks>
/// <para>
/// .NET Framework does not ship this type, so every assembly that uses records on net471 has to
/// supply it. Both AiDotNet and AiDotNet.Tensors already do — each as an <c>internal</c> type — and
/// both grant this test assembly <c>InternalsVisibleTo</c>. That made two candidates visible from
/// here and the net471 leg of the build failed outright:
/// </para>
/// <code>
///   error CS8356: Predefined type 'System.Runtime.CompilerServices.IsExternalInit' is declared in
///   multiple referenced assemblies: 'AiDotNet' and 'AiDotNet.Tensors'
/// </code>
/// <para>
/// A declaration in the CURRENT assembly takes precedence over any number of referenced ones, so
/// owning the type here resolves the ambiguity for good — and keeps resolving it for any record
/// added to these tests later, rather than only for the single one that happened to trip it.
/// Compiled out on net5.0+, where the framework supplies the real type.
/// </para>
/// </remarks>
internal sealed class IsExternalInit : Attribute
{
}

#endif

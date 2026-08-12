using System.Runtime.CompilerServices;

namespace AiDotNet.Tests.Helpers;

/// <summary>
/// Compares reference types by object identity on every target framework supported by the tests.
/// </summary>
internal sealed class ReferenceIdentityComparer<T> : IEqualityComparer<T> where T : class
{
    internal static readonly ReferenceIdentityComparer<T> Instance = new();

    private ReferenceIdentityComparer()
    {
    }

    public bool Equals(T? x, T? y) => ReferenceEquals(x, y);

    public int GetHashCode(T obj) => RuntimeHelpers.GetHashCode(obj);
}

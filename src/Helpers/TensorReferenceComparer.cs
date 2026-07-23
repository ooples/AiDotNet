using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace AiDotNet.Helpers;

/// <summary>
/// Compares objects by reference identity, not by value equality.
/// Used for Dictionary keys that must distinguish separate tensor instances
/// even if they contain identical data.
/// </summary>
/// <remarks>
/// This replaces <c>System.Collections.Generic.ReferenceEqualityComparer</c> which
/// is only available in .NET 5+. This implementation works on all target frameworks
/// including net471.
/// </remarks>
internal sealed class TensorReferenceComparer<T> : IEqualityComparer<T> where T : class
{
    public static readonly TensorReferenceComparer<T> Instance = new();

    public bool Equals(T? x, T? y) => ReferenceEquals(x, y);

    public int GetHashCode(T obj) => RuntimeHelpers.GetHashCode(obj);
}

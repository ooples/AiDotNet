using System;
using System.Collections.Generic;
using System.Reflection;

namespace AiDotNet.Models;

/// <summary>The source of an explicitly bound constructor argument.</summary>
public enum CloneConstructorArgumentKind
{
    /// <summary>Read the recorded declaring members from the original object.</summary>
    MemberPath,
    /// <summary>Use the selected constructor parameter's own declared default.</summary>
    ParameterDefault
}

/// <summary>
/// Retains the declaring identity of constructor state, including private or shadowed members.
/// Open generic declaring types are rebound to the corresponding closed base during cloning.
/// </summary>
public sealed class CloneConstructorArgumentBinding
{
    /// <summary>Creates a binding to one or more readable fields/properties.</summary>
    public CloneConstructorArgumentBinding(string parameterName, IReadOnlyList<MemberInfo> members)
    {
        if (string.IsNullOrEmpty(parameterName)) throw new ArgumentException("A parameter name is required.", nameof(parameterName));
        if (members is null) throw new ArgumentNullException(nameof(members));
        if (members.Count == 0) throw new ArgumentException("A member path cannot be empty.", nameof(members));
        var copied = new MemberInfo[members.Count];
        for (int i = 0; i < copied.Length; i++)
        {
            var member = members[i];
            bool readable = member is FieldInfo { IsStatic: false }
                || member is PropertyInfo property && property.GetMethod is { IsStatic: false }
                    && property.GetIndexParameters().Length == 0;
            if (!readable || member.DeclaringType is null)
                throw new ArgumentException("Bindings require instance fields or readable non-indexed properties.", nameof(members));
            copied[i] = member;
        }
        ParameterName = parameterName;
        Kind = CloneConstructorArgumentKind.MemberPath;
        Members = Array.AsReadOnly(copied);
    }

    private CloneConstructorArgumentBinding(string parameterName)
    {
        if (string.IsNullOrEmpty(parameterName)) throw new ArgumentException("A parameter name is required.", nameof(parameterName));
        ParameterName = parameterName;
        Kind = CloneConstructorArgumentKind.ParameterDefault;
        Members = Array.Empty<MemberInfo>();
    }

    /// <summary>Creates a binding to the parameter's own optional default.</summary>
    public static CloneConstructorArgumentBinding UseDefault(string parameterName) => new(parameterName);

    /// <summary>The actual constructor parameter name, not a heuristic member alias.</summary>
    public string ParameterName { get; }

    /// <summary>Whether the argument uses recorded members or its declared default.</summary>
    public CloneConstructorArgumentKind Kind { get; }

    /// <summary>The immutable, declaring-type-aware path from the original instance.</summary>
    public IReadOnlyList<MemberInfo> Members { get; }
}

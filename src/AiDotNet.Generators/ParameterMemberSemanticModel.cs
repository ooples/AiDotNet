using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace AiDotNet.Generators;

/// <summary>
/// The single semantic classifier shared by parameter generators and diagnostics.
/// It deliberately classifies declarations only; tensor type, nullability and field names never
/// invent trainability.
/// </summary>
internal static class ParameterMemberSemanticModel
{
    internal const string TrainableAttribute = "AiDotNet.Attributes.TrainableParameterAttribute";
    internal const string FittedAttribute = "AiDotNet.Attributes.FittedParameterAttribute";
    internal const string FrozenAttribute = "AiDotNet.Attributes.FrozenParameterAttribute";
    internal const string BufferAttribute = "AiDotNet.Attributes.BufferAttribute";
    internal const string ScratchAttribute = "AiDotNet.Attributes.ScratchAttribute";
    internal const string AliasAttribute = "AiDotNet.Attributes.ParameterAliasAttribute";
    internal const string ExternalAttribute = "AiDotNet.Attributes.ExternalStateAttribute";

    internal enum Kind
    {
        Unclassified,
        Trainable,
        Fitted,
        Frozen,
        Buffer,
        Scratch,
        Alias,
        External,
        Conflicting
    }

    internal readonly struct Classification
    {
        internal Classification(Kind kind, IReadOnlyList<string> declarations)
        {
            Kind = kind;
            Declarations = declarations;
        }

        internal Kind Kind { get; }
        internal IReadOnlyList<string> Declarations { get; }
        internal bool IsDeclared => Kind != Kind.Unclassified && Kind != Kind.Conflicting;
    }

    internal static Classification Classify(ISymbol member)
    {
        var declarations = new List<(Kind Kind, string Name)>();
        foreach (var attribute in member.GetAttributes())
        {
            string? name = attribute.AttributeClass?.ToDisplayString();
            _ = TryGetKind(attribute, out Kind kind);
            if (kind != Kind.Unclassified) declarations.Add((kind, name!));
        }

        if (declarations.Count == 0)
            return new Classification(Kind.Unclassified, Array.Empty<string>());

        var distinct = declarations.Select(item => item.Kind).Distinct().ToArray();
        return new Classification(
            distinct.Length == 1 ? distinct[0] : Kind.Conflicting,
            declarations.Select(item => item.Name).ToArray());
    }

    /// <summary>
    /// Classifies a declaration together with existing imperative registration APIs. A call named
    /// RegisterTrainableParameter/RegisterBuffer/RegisterParameterComponent is an explicit semantic
    /// declaration, not inference from storage shape. This lets the analyzer validate legacy code
    /// without forcing duplicate attributes while generators migrate that plumbing to declarations.
    /// </summary>
    internal static Classification ClassifyWithRegistrations(
        ISymbol member,
        IReadOnlyDictionary<string, Classification> registrations)
    {
        var attributed = Classify(member);
        if (!registrations.TryGetValue(member.Name, out var registered)) return attributed;

        // [ParameterAlias("stable-id")] intentionally accompanies a manual stable registration
        // of the same storage: the attribute tells generation not to register it a second time.
        // The alias validator independently proves that the named registration exists.
        if (attributed.Kind == Kind.Alias) return attributed;

        var kinds = new List<Kind>();
        var names = new List<string>();
        if (attributed.Kind == Kind.Conflicting)
        {
            kinds.Add(Kind.Conflicting);
            names.AddRange(attributed.Declarations);
        }
        else if (attributed.Kind != Kind.Unclassified)
        {
            kinds.Add(attributed.Kind);
            names.AddRange(attributed.Declarations);
        }

        kinds.Add(registered.Kind);
        names.AddRange(registered.Declarations);

        var distinct = kinds.Distinct().ToArray();
        return new Classification(
            distinct.Length == 1 ? distinct[0] : Kind.Conflicting,
            names.Distinct(StringComparer.Ordinal).ToArray());
    }

    internal static bool TryGetKind(AttributeData attribute, out Kind kind)
    {
        kind = attribute.AttributeClass?.ToDisplayString() switch
        {
            TrainableAttribute => Kind.Trainable,
            FittedAttribute => Kind.Fitted,
            FrozenAttribute => Kind.Frozen,
            BufferAttribute => Kind.Buffer,
            ScratchAttribute => Kind.Scratch,
            AliasAttribute => Kind.Alias,
            ExternalAttribute => Kind.External,
            _ => Kind.Unclassified
        };
        return kind != Kind.Unclassified;
    }

    /// <summary>
    /// True for raw numeric storage whose meaning cannot be inferred safely. Collections are
    /// unwrapped recursively because a list of tensors is just as ambiguous as one tensor.
    /// </summary>
    internal static bool IsNumericStateStorage(ITypeSymbol type)
    {
        ITypeSymbol probe = type;
        for (int depth = 0; depth < 4; depth++)
        {
            if (probe is IArrayTypeSymbol array)
            {
                probe = array.ElementType;
                continue;
            }

            if (probe is not INamedTypeSymbol named) break;
            string open = named.OriginalDefinition.ToDisplayString();
            if (open.StartsWith("AiDotNet.Tensors.LinearAlgebra.Tensor<", StringComparison.Ordinal)
                || open.StartsWith("AiDotNet.Tensors.LinearAlgebra.Matrix<", StringComparison.Ordinal)
                || open.StartsWith("AiDotNet.Tensors.LinearAlgebra.Vector<", StringComparison.Ordinal))
                return true;

            if (named.TypeArguments.Length == 1 && IsSequence(open))
            {
                probe = named.TypeArguments[0];
                continue;
            }
            if (named.TypeArguments.Length == 2 && IsDictionary(open))
            {
                probe = named.TypeArguments[1];
                continue;
            }
            break;
        }
        return false;
    }

    internal static bool IsConventionGradient(
        IFieldSymbol field,
        INamedTypeSymbol owner,
        IReadOnlyDictionary<string, Classification> registrations)
    {
        string suffix;
        if (field.Name.EndsWith("Gradients", StringComparison.Ordinal)) suffix = "Gradients";
        else if (field.Name.EndsWith("Gradient", StringComparison.Ordinal)) suffix = "Gradient";
        else return false;

        string parameterName = field.Name.Substring(0, field.Name.Length - suffix.Length);
        foreach (var candidate in owner.GetMembers(parameterName).OfType<IFieldSymbol>())
        {
            var classification = ClassifyWithRegistrations(candidate, registrations);
            if (classification.Kind == Kind.Trainable) return true;
        }
        return false;
    }

    internal static ITypeSymbol? GetMemberType(ISymbol member) => member switch
    {
        IFieldSymbol field when field.AssociatedSymbol is null => field.Type,
        IPropertySymbol property when property.GetMethod is not null => property.Type,
        _ => null
    };

    internal static bool IsNullable(ISymbol member)
    {
        var type = GetMemberType(member);
        return member switch
        {
            IFieldSymbol field => field.NullableAnnotation == NullableAnnotation.Annotated
                || type?.NullableAnnotation == NullableAnnotation.Annotated,
            IPropertySymbol property => property.NullableAnnotation == NullableAnnotation.Annotated
                || type?.NullableAnnotation == NullableAnnotation.Annotated,
            _ => false
        };
    }

    internal static bool HasExplicitDeferredAvailability(ISymbol member, Kind kind)
    {
        foreach (var attribute in member.GetAttributes())
        {
            if (!TryGetKind(attribute, out var declaredKind) || declaredKind != kind) continue;
            foreach (var argument in attribute.NamedArguments)
            {
                if (argument.Key == "Optional" && argument.Value.Value is bool optional && optional)
                    return true;
                if (argument.Key == "Availability" && argument.Value.Value is int availability
                    && availability != 0)
                    return true;
            }
        }
        return kind == Kind.Fitted || kind == Kind.External || kind == Kind.Scratch;
    }

    internal static string? GetAliasTarget(ISymbol member)
    {
        foreach (var attribute in member.GetAttributes())
        {
            if (!TryGetKind(attribute, out var kind) || kind != Kind.Alias) continue;
            if (attribute.ConstructorArguments.Length > 0
                && attribute.ConstructorArguments[0].Value is string target)
                return target;
        }
        return null;
    }

    internal static IReadOnlyDictionary<string, Classification> GetRegistrationClassifications(
        INamedTypeSymbol owner)
    {
        var memberNames = new HashSet<string>(owner.GetMembers()
            .Where(member => member is IFieldSymbol or IPropertySymbol)
            .Select(member => member.Name), StringComparer.Ordinal);
        var declarations = new Dictionary<string, List<(Kind Kind, string Name)>>(StringComparer.Ordinal);
        foreach (var syntaxReference in owner.DeclaringSyntaxReferences)
        {
            if (syntaxReference.GetSyntax() is not ClassDeclarationSyntax declaration) continue;
            foreach (var invocation in declaration.DescendantNodes().OfType<InvocationExpressionSyntax>())
            {
                string? callName = invocation.Expression switch
                {
                    IdentifierNameSyntax identifier => identifier.Identifier.ValueText,
                    MemberAccessExpressionSyntax access => access.Name.Identifier.ValueText,
                    _ => null
                };
                if (callName is not "RegisterTrainableParameter"
                    and not "RegisterBuffer"
                    and not "RegisterParameterComponent") continue;
                Kind kind = callName switch
                {
                    "RegisterTrainableParameter" => Kind.Trainable,
                    "RegisterBuffer" => Kind.Buffer,
                    _ => ParameterComponentRole(invocation)
                };
                foreach (string memberName in invocation.ArgumentList.DescendantNodesAndSelf()
                    .OfType<IdentifierNameSyntax>()
                    .Select(identifier => identifier.Identifier.ValueText)
                    .Where(memberNames.Contains)
                    .Distinct(StringComparer.Ordinal))
                {
                    if (!declarations.TryGetValue(memberName, out var memberDeclarations))
                    {
                        memberDeclarations = new List<(Kind Kind, string Name)>();
                        declarations.Add(memberName, memberDeclarations);
                    }
                    memberDeclarations.Add((kind, callName));
                }
            }
        }

        var result = new Dictionary<string, Classification>(StringComparer.Ordinal);
        foreach (var pair in declarations)
        {
            var distinct = pair.Value.Select(item => item.Kind).Distinct().ToArray();
            result.Add(pair.Key, new Classification(
                distinct.Length == 1 ? distinct[0] : Kind.Conflicting,
                pair.Value.Select(item => item.Name).Distinct(StringComparer.Ordinal).ToArray()));
        }
        return result;
    }

    private static Kind ParameterComponentRole(InvocationExpressionSyntax invocation)
    {
        foreach (var access in invocation.ArgumentList.DescendantNodes()
            .OfType<MemberAccessExpressionSyntax>())
        {
            if (!access.Expression.ToString().EndsWith("ParameterSlotRole", StringComparison.Ordinal))
                continue;
            return access.Name.Identifier.ValueText switch
            {
                "LearnedState" => Kind.Fitted,
                "Frozen" => Kind.Frozen,
                "Buffer" => Kind.Buffer,
                "Scratch" => Kind.Scratch,
                "Alias" => Kind.Alias,
                "External" => Kind.External,
                _ => Kind.Trainable
            };
        }
        return Kind.Trainable;
    }

    private static bool IsSequence(string open) =>
        open.StartsWith("System.Collections.Generic.List<", StringComparison.Ordinal)
        || open.StartsWith("System.Collections.Generic.IList<", StringComparison.Ordinal)
        || open.StartsWith("System.Collections.Generic.IReadOnlyList<", StringComparison.Ordinal)
        || open.StartsWith("System.Collections.Generic.ICollection<", StringComparison.Ordinal)
        || open.StartsWith("System.Collections.Generic.IEnumerable<", StringComparison.Ordinal);

    private static bool IsDictionary(string open) =>
        open.StartsWith("System.Collections.Generic.Dictionary<", StringComparison.Ordinal)
        || open.StartsWith("System.Collections.Generic.IDictionary<", StringComparison.Ordinal)
        || open.StartsWith("System.Collections.Generic.IReadOnlyDictionary<", StringComparison.Ordinal);
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;

namespace AiDotNet.Generators;

public partial class ClonePlanGenerator
{
    private sealed class ConstructorArgumentSource
    {
        public static readonly ConstructorArgumentSource Default = new(Array.Empty<ISymbol>());
        public ConstructorArgumentSource(IReadOnlyList<ISymbol> members) => Members = members;
        public IReadOnlyList<ISymbol> Members { get; }
        public string MemberName => Members.Count == 0 ? UseDefault : string.Join(".", Members.Select(member => member.Name));
    }

    private sealed class ConstructorSourceCandidate
    {
        public ConstructorSourceCandidate(IMethodSymbol constructor, IReadOnlyList<ConstructorArgumentSource> arguments)
        {
            Constructor = constructor;
            Arguments = arguments;
        }
        public IMethodSymbol Constructor { get; }
        public IReadOnlyList<ConstructorArgumentSource> Arguments { get; }
    }

    private static bool HasShadowedSource(INamedTypeSymbol type, ConstructorSourceCandidate candidate)
    {
        foreach (var argument in candidate.Arguments)
        {
            INamedTypeSymbol? owner = type;
            foreach (var selected in argument.Members)
            {
                ISymbol? first = null;
                for (var current = owner; current is not null && first is null; current = current.BaseType)
                {
                    var named = current.GetMembers(selected.Name);
                    first = named.FirstOrDefault(member => member is IPropertySymbol { IsStatic: false, IsIndexer: false } property
                        && property.GetMethod is not null)
                        ?? named.FirstOrDefault(member => member is IFieldSymbol { IsStatic: false });
                }
                if (!SymbolEqualityComparer.Default.Equals(first, selected)) return true;
                owner = MemberType(selected) as INamedTypeSymbol;
            }
        }
        return false;
    }

    private static string EmitConstructorBindings(INamedTypeSymbol type, IReadOnlyList<ConstructorSourceCandidate> candidates)
    {
        var result = new StringBuilder("new IReadOnlyList<CloneConstructorArgumentBinding>[] { ");
        foreach (var candidate in candidates)
        {
            result.Append("new CloneConstructorArgumentBinding[] { ");
            for (int i = 0; i < candidate.Arguments.Count; i++)
            {
                var argument = candidate.Arguments[i];
                string name = Microsoft.CodeAnalysis.CSharp.SymbolDisplay.FormatLiteral(candidate.Constructor.Parameters[i].Name, quote: true);
                if (argument.Members.Count == 0)
                {
                    result.Append("CloneConstructorArgumentBinding.UseDefault(").Append(name).Append("), ");
                    continue;
                }
                result.Append("new CloneConstructorArgumentBinding(").Append(name)
                    .Append(", BindMembers(t, new (int Depth, string Name, MemberTypes Kind)[] { ");
                INamedTypeSymbol? owner = type;
                foreach (var member in argument.Members)
                {
                    int depth = 0;
                    while (owner is not null && !SymbolEqualityComparer.Default.Equals(owner, member.ContainingType))
                    {
                        owner = owner.BaseType;
                        depth++;
                    }
                    if (owner is null)
                        throw new InvalidOperationException("The selected clone member is not in its source inheritance chain.");
                    result.Append('(').Append(depth).Append(", ")
                        .Append(Microsoft.CodeAnalysis.CSharp.SymbolDisplay.FormatLiteral(member.Name, quote: true))
                        .Append(", MemberTypes.").Append(member is IFieldSymbol ? "Field" : "Property").Append("), ");
                    owner = MemberType(member) as INamedTypeSymbol;
                }
                result.Append("})), ");
            }
            result.Append("}, ");
        }
        return result.Append('}').ToString();
    }

    private static ITypeSymbol? MemberType(ISymbol member) => member switch
    {
        IFieldSymbol field => field.Type,
        IPropertySymbol property => property.Type,
        _ => null
    };

    private static void EmitBindingMemberResolver(StringBuilder sb)
    {
        sb.AppendLine("    private static MemberInfo[] BindMembers(Type root, (int Depth, string Name, MemberTypes Kind)[] steps)");
        sb.AppendLine("    {");
        sb.AppendLine("        const BindingFlags flags = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.DeclaredOnly;");
        sb.AppendLine("        var result = new MemberInfo[steps.Length];");
        sb.AppendLine("        for (int i = 0; i < steps.Length; i++)");
        sb.AppendLine("        {");
        sb.AppendLine("            var owner = root;");
        sb.AppendLine("            for (int depth = 0; depth < steps[i].Depth; depth++)");
        sb.AppendLine("                owner = owner.BaseType ?? throw new InvalidOperationException(\"A generated clone owner is missing.\");");
        sb.AppendLine("            MemberInfo? member = steps[i].Kind switch");
        sb.AppendLine("            {");
        sb.AppendLine("                MemberTypes.Field => owner.GetField(steps[i].Name, flags),");
        sb.AppendLine("                MemberTypes.Property => owner.GetProperty(steps[i].Name, flags),");
        sb.AppendLine("                _ => throw new InvalidOperationException(\"Unsupported generated clone member kind.\")");
        sb.AppendLine("            };");
        sb.AppendLine("            result[i] = member ?? throw new MissingMemberException(owner.FullName, steps[i].Name);");
        sb.AppendLine("            root = member switch");
        sb.AppendLine("            {");
        sb.AppendLine("                FieldInfo field => field.FieldType,");
        sb.AppendLine("                PropertyInfo property => property.PropertyType,");
        sb.AppendLine("                _ => throw new InvalidOperationException(\"Unsupported generated clone member kind.\")");
        sb.AppendLine("            };");
        sb.AppendLine("        }");
        sb.AppendLine("        return result;");
        sb.AppendLine("    }");
    }
}

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Threading;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Text;

namespace AiDotNet.Generators;

/// <summary>
/// Roslyn incremental generator that enforces, at build time, that every fluent
/// <c>Configure*</c> method on the facade actually does something with what it was given.
/// </summary>
/// <remarks>
/// <para>
/// The bug class this exists to kill: a <c>Configure*</c> method accepts a value, stores it in a
/// private field, returns <c>this</c>, and nothing ever reads that field. The caller gets no
/// exception and no warning -- just a model built as if they had never called the method.
/// </para>
/// <para>
/// <b>Why this is cheap.</b> The fields are private, so reads can only appear inside the declaring
/// type's partial declarations. The syntax-provider transform resolves that exact symbol and
/// immediately projects each declaration into an immutable, structurally equatable value model.
/// No <see cref="SyntaxNode"/> or <see cref="ISymbol"/> survives into the incremental pipeline, so
/// unrelated edits do not invalidate this analysis or retain compilations in memory.
/// </para>
/// <para>
/// <b>Severity policy.</b> Ships as Warning, matching AIDN070-076. Ratchet to Error once the
/// backlog is zero.
/// </para>
/// <para>
/// <b>Rule-id range.</b> AIDN096-097 follow the parameter-automation block AIDN090-095 and are
/// uniquely owned by facade configuration validation.
/// </para>
/// </remarks>
[Generator]
public class FacadeConfigurationValidationGenerator : IIncrementalGenerator
{
    private const string Category = "AiDotNet.FacadeConfiguration";
    private const string FacadeTypeName = "AiModelBuilder";
    private const string FacadeNamespace = "AiDotNet";
    private const int FacadeArity = 3;

    internal static readonly DiagnosticDescriptor ConfiguredValueNeverRead = new(
        id: "AIDN096",
        title: "Configure* method stores a value nothing ever reads",
        messageFormat: "'{0}' assigns '{1}', but nothing ever reads it -- the configuration is silently dropped",
        category: Category,
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true,
        description: "A fluent Configure* method must have an observable effect. Assigning a private field that " +
                     "no code reads means the caller's configuration is accepted and discarded with no error: the " +
                     "model is built as if the method had never been called. Either consume the field where the " +
                     "model is built, route the value into the pipeline, or delete the field if the method already " +
                     "takes effect by another route.");

    internal static readonly DiagnosticDescriptor ConfiguredValueOnlyExposed = new(
        id: "AIDN097",
        title: "Configured value is only reachable through an accessor nobody calls",
        messageFormat: "'{0}' is read only by '{1}', which has no callers -- the configuration is still effectively unused",
        category: Category,
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true,
        description: "Exposing a configured field through a property does not by itself make the configuration take " +
                     "effect. If the accessor has no callers, the value is stored, exposed, and still never acted on. " +
                     "Consume it, or remove the configuration surface that promises it is honoured.");

    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        var facadeParts = context.SyntaxProvider.CreateSyntaxProvider(
                predicate: static (node, _) => node is ClassDeclarationSyntax declaration
                    && declaration.Identifier.ValueText == FacadeTypeName,
                transform: static (syntaxContext, cancellationToken) =>
                    CreateFacadePart(syntaxContext, cancellationToken))
            .Where(static part => part is not null)
            .Select(static (part, _) => part!)
            .Collect();

        context.RegisterSourceOutput(facadeParts, static (productionContext, parts) =>
            Analyze(productionContext, parts));
    }

    private static FacadePart? CreateFacadePart(
        GeneratorSyntaxContext context,
        CancellationToken cancellationToken)
    {
        var declaration = (ClassDeclarationSyntax)context.Node;
        var declaredSymbol = context.SemanticModel.GetDeclaredSymbol(declaration, cancellationToken);
        if (declaredSymbol is null
            || declaredSymbol.Name != FacadeTypeName
            || declaredSymbol.Arity != FacadeArity
            || declaredSymbol.ContainingNamespace.ToDisplayString() != FacadeNamespace)
        {
            return null;
        }

        string symbolKey = declaredSymbol.OriginalDefinition.ToDisplayString(
            SymbolDisplayFormat.FullyQualifiedFormat);
        var fields = ImmutableArray.CreateBuilder<FieldModel>();
        var configureWrites = ImmutableArray.CreateBuilder<NamePair>();
        var realReads = ImmutableArray.CreateBuilder<string>();
        var accessorReads = ImmutableArray.CreateBuilder<NamePair>();
        var accessorCalls = ImmutableArray.CreateBuilder<string>();
        var accessorReferenceStarts = new HashSet<int>();

        foreach (var fieldDeclaration in declaration.Members.OfType<FieldDeclarationSyntax>())
        {
            foreach (var variable in fieldDeclaration.Declaration.Variables)
            {
                if (context.SemanticModel.GetDeclaredSymbol(variable, cancellationToken) is not IFieldSymbol fieldSymbol
                    || fieldSymbol.DeclaredAccessibility != Accessibility.Private
                    || !BelongsToFacade(fieldSymbol.ContainingType, declaredSymbol))
                {
                    continue;
                }

                fields.Add(new FieldModel(fieldSymbol.Name, SourceLocation.From(variable.Identifier)));
            }
        }

        foreach (var property in declaration.Members.OfType<PropertyDeclarationSyntax>())
        {
            ExpressionSyntax? expression = GetAccessorExpression(property);
            if (expression is null) continue;

            if (context.SemanticModel.GetSymbolInfo(expression, cancellationToken).Symbol is not IFieldSymbol fieldSymbol
                || fieldSymbol.DeclaredAccessibility != Accessibility.Private
                || !BelongsToFacade(fieldSymbol.ContainingType, declaredSymbol))
            {
                continue;
            }

            accessorReads.Add(new NamePair(fieldSymbol.Name, property.Identifier.ValueText));
            foreach (var identifier in expression.DescendantNodesAndSelf().OfType<IdentifierNameSyntax>())
            {
                if (context.SemanticModel.GetSymbolInfo(identifier, cancellationToken).Symbol is IFieldSymbol referenced
                    && SymbolEqualityComparer.Default.Equals(referenced, fieldSymbol))
                {
                    accessorReferenceStarts.Add(identifier.SpanStart);
                }
            }
        }

        foreach (var identifier in declaration.DescendantNodes().OfType<IdentifierNameSyntax>())
        {
            ISymbol? referencedSymbol = context.SemanticModel.GetSymbolInfo(identifier, cancellationToken).Symbol;

            if (referencedSymbol is IFieldSymbol fieldSymbol
                && fieldSymbol.DeclaredAccessibility == Accessibility.Private
                && BelongsToFacade(fieldSymbol.ContainingType, declaredSymbol))
            {
                if (accessorReferenceStarts.Contains(identifier.SpanStart)) continue;

                if (IsSimpleAssignmentTarget(identifier))
                {
                    string? methodName = identifier.FirstAncestorOrSelf<MethodDeclarationSyntax>()
                        ?.Identifier.ValueText;
                    if (methodName is not null
                        && methodName.StartsWith("Configure", StringComparison.Ordinal))
                    {
                        configureWrites.Add(new NamePair(fieldSymbol.Name, methodName));
                    }
                    continue;
                }

                realReads.Add(fieldSymbol.Name);
                continue;
            }

            if (referencedSymbol is IPropertySymbol propertySymbol
                && BelongsToFacade(propertySymbol.ContainingType, declaredSymbol)
                && !IsWithinNameOfExpression(identifier)
                && !IsSimpleAssignmentTarget(identifier))
            {
                accessorCalls.Add(propertySymbol.Name);
            }
        }

        return new FacadePart(
            symbolKey,
            fields.ToImmutable(),
            configureWrites.ToImmutable(),
            realReads.ToImmutable(),
            accessorReads.ToImmutable(),
            accessorCalls.ToImmutable());
    }

    private static bool BelongsToFacade(INamedTypeSymbol? candidate, INamedTypeSymbol facade) =>
        candidate is not null
        && SymbolEqualityComparer.Default.Equals(candidate.OriginalDefinition, facade.OriginalDefinition);

    private static ExpressionSyntax? GetAccessorExpression(PropertyDeclarationSyntax property)
    {
        if (property.ExpressionBody is not null) return property.ExpressionBody.Expression;

        var getter = property.AccessorList?.Accessors
            .FirstOrDefault(accessor => accessor.IsKind(SyntaxKind.GetAccessorDeclaration));
        if (getter?.ExpressionBody is not null) return getter.ExpressionBody.Expression;
        if (getter?.Body?.Statements.Count == 1
            && getter.Body.Statements[0] is ReturnStatementSyntax returnStatement)
        {
            return returnStatement.Expression;
        }

        return null;
    }

    private static bool IsSimpleAssignmentTarget(IdentifierNameSyntax identifier)
    {
        ExpressionSyntax target = identifier;
        if (identifier.Parent is MemberAccessExpressionSyntax memberAccess
            && memberAccess.Name == identifier)
        {
            target = memberAccess;
        }

        return target.Parent is AssignmentExpressionSyntax assignment
            && assignment.Left == target
            && assignment.IsKind(SyntaxKind.SimpleAssignmentExpression);
    }

    private static bool IsWithinNameOfExpression(IdentifierNameSyntax identifier)
    {
        var argument = identifier.FirstAncestorOrSelf<ArgumentSyntax>();
        return argument?.Parent?.Parent is InvocationExpressionSyntax invocation
            && invocation.Expression is IdentifierNameSyntax nameOfIdentifier
            && nameOfIdentifier.Identifier.ValueText == "nameof";
    }

    private static void Analyze(
        SourceProductionContext context,
        ImmutableArray<FacadePart> parts)
    {
        if (parts.IsDefaultOrEmpty) return;

        var facades = new Dictionary<string, FacadeAggregate>(StringComparer.Ordinal);
        foreach (var part in parts)
        {
            if (!facades.TryGetValue(part.SymbolKey, out var facade))
            {
                facade = new FacadeAggregate();
                facades.Add(part.SymbolKey, facade);
            }

            foreach (var field in part.Fields)
                if (!facade.Fields.ContainsKey(field.Name)) facade.Fields.Add(field.Name, field);
            foreach (var write in part.ConfigureWrites)
                if (!facade.AssigningMethods.ContainsKey(write.Left))
                    facade.AssigningMethods.Add(write.Left, write.Right);
            foreach (string read in part.RealReads) facade.RealReads.Add(read);
            foreach (var accessorRead in part.AccessorReads)
                if (!facade.AccessorReads.ContainsKey(accessorRead.Left))
                    facade.AccessorReads.Add(accessorRead.Left, accessorRead.Right);
            foreach (string accessorCall in part.AccessorCalls) facade.AccessorCalls.Add(accessorCall);
        }

        foreach (var facade in facades.Values)
        {
            foreach (var pair in facade.Fields)
            {
                string fieldName = pair.Key;
                FieldModel field = pair.Value;
                if (!facade.AssigningMethods.TryGetValue(fieldName, out string methodName)) continue;
                if (facade.RealReads.Contains(fieldName)) continue;

                if (facade.AccessorReads.TryGetValue(fieldName, out string accessorName))
                {
                    if (!facade.AccessorCalls.Contains(accessorName))
                    {
                        context.ReportDiagnostic(Diagnostic.Create(
                            ConfiguredValueOnlyExposed,
                            field.Location.ToLocation(),
                            fieldName,
                            accessorName));
                    }
                    continue;
                }

                context.ReportDiagnostic(Diagnostic.Create(
                    ConfiguredValueNeverRead,
                    field.Location.ToLocation(),
                    methodName,
                    fieldName));
            }
        }
    }

    private sealed class FacadeAggregate
    {
        internal Dictionary<string, FieldModel> Fields { get; } =
            new(StringComparer.Ordinal);
        internal Dictionary<string, string> AssigningMethods { get; } =
            new(StringComparer.Ordinal);
        internal HashSet<string> RealReads { get; } = new(StringComparer.Ordinal);
        internal Dictionary<string, string> AccessorReads { get; } =
            new(StringComparer.Ordinal);
        internal HashSet<string> AccessorCalls { get; } = new(StringComparer.Ordinal);
    }

    private sealed class FacadePart : IEquatable<FacadePart>
    {
        internal FacadePart(
            string symbolKey,
            ImmutableArray<FieldModel> fields,
            ImmutableArray<NamePair> configureWrites,
            ImmutableArray<string> realReads,
            ImmutableArray<NamePair> accessorReads,
            ImmutableArray<string> accessorCalls)
        {
            SymbolKey = symbolKey;
            Fields = fields;
            ConfigureWrites = configureWrites;
            RealReads = realReads;
            AccessorReads = accessorReads;
            AccessorCalls = accessorCalls;
        }

        internal string SymbolKey { get; }
        internal ImmutableArray<FieldModel> Fields { get; }
        internal ImmutableArray<NamePair> ConfigureWrites { get; }
        internal ImmutableArray<string> RealReads { get; }
        internal ImmutableArray<NamePair> AccessorReads { get; }
        internal ImmutableArray<string> AccessorCalls { get; }

        public bool Equals(FacadePart? other) =>
            other is not null
            && SymbolKey == other.SymbolKey
            && Fields.SequenceEqual(other.Fields)
            && ConfigureWrites.SequenceEqual(other.ConfigureWrites)
            && RealReads.SequenceEqual(other.RealReads, StringComparer.Ordinal)
            && AccessorReads.SequenceEqual(other.AccessorReads)
            && AccessorCalls.SequenceEqual(other.AccessorCalls, StringComparer.Ordinal);

        public override bool Equals(object? obj) => Equals(obj as FacadePart);

        public override int GetHashCode()
        {
            unchecked
            {
                int hash = SymbolKey.GetHashCode();
                foreach (var item in Fields) hash = hash * 31 + item.GetHashCode();
                foreach (var item in ConfigureWrites) hash = hash * 31 + item.GetHashCode();
                foreach (var item in RealReads) hash = hash * 31 + item.GetHashCode();
                foreach (var item in AccessorReads) hash = hash * 31 + item.GetHashCode();
                foreach (var item in AccessorCalls) hash = hash * 31 + item.GetHashCode();
                return hash;
            }
        }
    }

    private readonly struct FieldModel : IEquatable<FieldModel>
    {
        internal FieldModel(string name, SourceLocation location)
        {
            Name = name;
            Location = location;
        }

        internal string Name { get; }
        internal SourceLocation Location { get; }
        public bool Equals(FieldModel other) => Name == other.Name && Location.Equals(other.Location);
        public override bool Equals(object? obj) => obj is FieldModel other && Equals(other);
        public override int GetHashCode() => unchecked(Name.GetHashCode() * 397 ^ Location.GetHashCode());
    }

    private readonly struct NamePair : IEquatable<NamePair>
    {
        internal NamePair(string left, string right)
        {
            Left = left;
            Right = right;
        }

        internal string Left { get; }
        internal string Right { get; }
        public bool Equals(NamePair other) => Left == other.Left && Right == other.Right;
        public override bool Equals(object? obj) => obj is NamePair other && Equals(other);
        public override int GetHashCode() => unchecked(Left.GetHashCode() * 397 ^ Right.GetHashCode());
    }

    private readonly struct SourceLocation : IEquatable<SourceLocation>
    {
        private SourceLocation(
            string path,
            int spanStart,
            int spanLength,
            int startLine,
            int startCharacter,
            int endLine,
            int endCharacter)
        {
            Path = path;
            SpanStart = spanStart;
            SpanLength = spanLength;
            StartLine = startLine;
            StartCharacter = startCharacter;
            EndLine = endLine;
            EndCharacter = endCharacter;
        }

        private string Path { get; }
        private int SpanStart { get; }
        private int SpanLength { get; }
        private int StartLine { get; }
        private int StartCharacter { get; }
        private int EndLine { get; }
        private int EndCharacter { get; }

        internal static SourceLocation From(SyntaxToken token)
        {
            FileLinePositionSpan lines = token.SyntaxTree.GetLineSpan(token.Span);
            return new SourceLocation(
                token.SyntaxTree.FilePath ?? string.Empty,
                token.SpanStart,
                token.Span.Length,
                lines.StartLinePosition.Line,
                lines.StartLinePosition.Character,
                lines.EndLinePosition.Line,
                lines.EndLinePosition.Character);
        }

        internal Location ToLocation() => Location.Create(
            Path,
            new TextSpan(SpanStart, SpanLength),
            new LinePositionSpan(
                new LinePosition(StartLine, StartCharacter),
                new LinePosition(EndLine, EndCharacter)));

        public bool Equals(SourceLocation other) =>
            Path == other.Path
            && SpanStart == other.SpanStart
            && SpanLength == other.SpanLength
            && StartLine == other.StartLine
            && StartCharacter == other.StartCharacter
            && EndLine == other.EndLine
            && EndCharacter == other.EndCharacter;

        public override bool Equals(object? obj) => obj is SourceLocation other && Equals(other);

        public override int GetHashCode()
        {
            unchecked
            {
                int hash = Path.GetHashCode();
                hash = hash * 31 + SpanStart;
                hash = hash * 31 + SpanLength;
                hash = hash * 31 + StartLine;
                hash = hash * 31 + StartCharacter;
                hash = hash * 31 + EndLine;
                hash = hash * 31 + EndCharacter;
                return hash;
            }
        }
    }
}

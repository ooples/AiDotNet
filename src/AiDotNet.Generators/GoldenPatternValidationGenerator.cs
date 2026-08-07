using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace AiDotNet.Generators;

/// <summary>
/// Roslyn incremental generator that enforces the project's golden patterns at BUILD time,
/// rather than relying on a reviewer to spot them in a diff.
/// </summary>
/// <remarks>
/// <para>
/// Every rule here is mechanically decidable, which is exactly why it belongs in an analyzer
/// instead of a code review: an analyzer sees 100% of the code on every build, costs nothing,
/// is never rate-limited, and cannot be merged around. Reviewers are then free to spend their
/// attention on the judgement calls an analyzer genuinely cannot make -- whether an
/// implementation is faithful to its paper, whether a test asserts something meaningful.
/// </para>
/// <para>
/// <b>Severity policy.</b> These rules ship as Warning, matching the precedent set by
/// AIDN010/011/012: the existing violation backlog is large, and turning them straight to Error
/// would break the build on day one and force a blanket suppression, which is worse than a
/// warning that is actually read. Ratchet each rule to Error once its backlog reaches zero.
/// </para>
/// <para>
/// <b>Scope.</b> The analyzer is referenced by AiDotNet.csproj, so it runs over production
/// source only; test projects consume the library as a package reference and are unaffected.
/// </para>
/// </remarks>
[Generator]
public class GoldenPatternValidationGenerator : IIncrementalGenerator
{
    private const string Category = "AiDotNet.GoldenPattern";

    internal static readonly DiagnosticDescriptor CopyConstructorMissesProperty = new(
        id: "AIDN070",
        title: "Options copy constructor does not copy every property",
        messageFormat: "Copy constructor of '{0}' does not copy property '{1}' -- clones will silently revert it to its default",
        category: Category,
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true,
        description: "An Options copy constructor must assign every settable property declared on the type. " +
                     "A missed property is silent data loss: the clone keeps the default while the original keeps " +
                     "the configured value, which typically shows up as a model that trains correctly but whose " +
                     "clone does not.");

    internal static readonly DiagnosticDescriptor NullForgivingOperator = new(
        id: "AIDN071",
        title: "Null-forgiving operator is not permitted",
        messageFormat: "Null-forgiving operator '!' suppresses a nullable warning instead of handling null",
        category: Category,
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true,
        description: "Use an explicit null check, a null-coalescing default, or proper initialization. " +
                     "The '!' operator hides the warning without removing the NullReferenceException.");

    internal static readonly DiagnosticDescriptor RawRandomConstruction = new(
        id: "AIDN072",
        title: "Use RandomHelper instead of new Random()",
        messageFormat: "'new Random(...)' is not cryptographically secure and is seeded from the clock; use RandomHelper.CreateSeededRandom or RandomHelper.CreateSecureRandom",
        category: Category,
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true,
        description: "System.Random is not cryptographically secure, and instances created close together can " +
                     "share a seed. Use RandomHelper.CreateSeededRandom(seed) when reproducibility is required, " +
                     "RandomHelper.CreateSecureRandom() otherwise.");

    internal static readonly DiagnosticDescriptor RegexWithoutTimeout = new(
        id: "AIDN073",
        title: "Regex without a timeout (ReDoS)",
        messageFormat: "'{0}' has no matchTimeout argument; a catastrophically backtracking pattern can hang the process",
        category: Category,
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true,
        description: "Pass an explicit TimeSpan matchTimeout so a pathological pattern or hostile input fails fast " +
                     "instead of consuming the thread indefinitely.");

    internal static readonly DiagnosticDescriptor NotImplemented = new(
        id: "AIDN074",
        title: "NotImplementedException in production code",
        messageFormat: "'{0}' is a stub -- production code must have a complete implementation",
        category: Category,
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true,
        description: "Throwing NotImplementedException ships a method that fails at runtime. Implement it, or " +
                     "throw NotSupportedException when the operation is genuinely not applicable to this type.");

    internal static readonly DiagnosticDescriptor ConsoleUsedForLogging = new(
        id: "AIDN075",
        title: "Console output used instead of a logging abstraction",
        messageFormat: "'{0}' writes to the console; use ILogger so the message reaches monitoring",
        category: Category,
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true,
        description: "Console output is not captured by exception tracking or monitoring. Use ILogger, and never " +
                     "surface raw exception text to users.");

    internal static readonly DiagnosticDescriptor SwallowedException = new(
        id: "AIDN076",
        title: "Catch block swallows the exception",
        messageFormat: "Catch block discards the exception without logging it",
        category: Category,
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true,
        description: "Every catch must call Logger.LogError/LogWarning with the exception so developers can " +
                     "diagnose it. An empty catch makes the failure invisible.");

    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        var optionsFindings = context.SyntaxProvider.CreateSyntaxProvider(
                predicate: static (node, _) => node is ClassDeclarationSyntax cds
                    && (cds.Identifier.ValueText.EndsWith("Options", System.StringComparison.Ordinal)
                        || cds.Identifier.ValueText.EndsWith("Config", System.StringComparison.Ordinal)),
                transform: static (ctx, _) => AnalyzeOptionsCopyConstructor(ctx))
            .Where(static f => f.Count > 0);

        var nodeFindings = context.SyntaxProvider.CreateSyntaxProvider(
                predicate: static (node, _) => IsInterestingNode(node),
                transform: static (ctx, _) => AnalyzeNode(ctx))
            .Where(static f => f is not null);

        context.RegisterSourceOutput(optionsFindings.Collect(), static (spc, batches) =>
        {
            foreach (var batch in batches)
                foreach (var f in batch)
                    Report(spc, f);
        });

        context.RegisterSourceOutput(nodeFindings.Collect(), static (spc, findings) =>
        {
            foreach (var f in findings)
            {
                if (f is not null) Report(spc, f);
            }
        });
    }

    private static void Report(SourceProductionContext spc, Finding f) =>
        spc.ReportDiagnostic(Diagnostic.Create(f.Descriptor, f.Location, f.Args));

    /// <summary>A single rule violation, carried from the transform phase to the output phase.</summary>
    internal sealed class Finding : System.IEquatable<Finding>
    {
        internal Finding(DiagnosticDescriptor descriptor, Location location, params string[] args)
        {
            Descriptor = descriptor;
            Location = location;
            Args = args;
        }

        internal DiagnosticDescriptor Descriptor { get; }
        internal Location Location { get; }
        internal string[] Args { get; }

        // Value equality so incremental-pipeline values compare by content, not identity — otherwise
        // every transform run yields reference-distinct Findings and Collect() treats any edit as
        // changed input, re-reporting diagnostics that did not actually change.
        public bool Equals(Finding? other)
        {
            if (other is null) return false;
            if (Descriptor.Id != other.Descriptor.Id) return false;
            if (!Location.Equals(other.Location)) return false;
            if (Args.Length != other.Args.Length) return false;
            for (int i = 0; i < Args.Length; i++)
                if (!string.Equals(Args[i], other.Args[i], System.StringComparison.Ordinal)) return false;
            return true;
        }

        public override bool Equals(object? obj) => Equals(obj as Finding);

        public override int GetHashCode()
        {
            unchecked
            {
                int hash = Descriptor.Id.GetHashCode();
                hash = (hash * 397) ^ Location.GetHashCode();
                foreach (var arg in Args) hash = (hash * 397) ^ (arg?.GetHashCode() ?? 0);
                return hash;
            }
        }
    }

    /// <summary>
    /// A readonly <see cref="ImmutableArray{T}"/> wrapper with structural equality. Incremental
    /// pipeline values are compared with <see cref="EqualityComparer{T}.Default"/>, and
    /// <c>ImmutableArray&lt;T&gt;.Equals</c> compares the underlying array by reference, so a bare
    /// list/array cached value never compares equal across runs; this wrapper compares by content.
    /// </summary>
    internal readonly struct EquatableArray<T> : System.IEquatable<EquatableArray<T>>, IEnumerable<T>
        where T : System.IEquatable<T>
    {
        private readonly ImmutableArray<T> _items;

        internal EquatableArray(ImmutableArray<T> items) => _items = items;

        internal int Count => _items.IsDefault ? 0 : _items.Length;

        public bool Equals(EquatableArray<T> other)
        {
            var a = _items.IsDefault ? ImmutableArray<T>.Empty : _items;
            var b = other._items.IsDefault ? ImmutableArray<T>.Empty : other._items;
            if (a.Length != b.Length) return false;
            for (int i = 0; i < a.Length; i++)
                if (!a[i].Equals(b[i])) return false;
            return true;
        }

        public override bool Equals(object? obj) => obj is EquatableArray<T> other && Equals(other);

        public override int GetHashCode()
        {
            if (_items.IsDefault) return 0;
            unchecked
            {
                int hash = 17;
                foreach (var item in _items) hash = (hash * 397) ^ (item?.GetHashCode() ?? 0);
                return hash;
            }
        }

        public IEnumerator<T> GetEnumerator()
        {
            var items = _items.IsDefault ? ImmutableArray<T>.Empty : _items;
            foreach (var item in items) yield return item;
        }

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();
    }

    private static bool IsInterestingNode(SyntaxNode node) =>
        node is PostfixUnaryExpressionSyntax { RawKind: (int)SyntaxKind.SuppressNullableWarningExpression }
             or BaseObjectCreationExpressionSyntax
             or InvocationExpressionSyntax
             or CatchClauseSyntax;

    private static Finding? AnalyzeNode(GeneratorSyntaxContext ctx)
    {
        switch (ctx.Node)
        {
            case PostfixUnaryExpressionSyntax postfix
                when postfix.IsKind(SyntaxKind.SuppressNullableWarningExpression):
                return new Finding(NullForgivingOperator, postfix.GetLocation());

            case BaseObjectCreationExpressionSyntax creation:
                return AnalyzeObjectCreation(ctx, creation);

            case InvocationExpressionSyntax invocation:
                return AnalyzeInvocation(ctx, invocation);

            case CatchClauseSyntax catchClause:
                return AnalyzeCatch(catchClause);
        }

        return null;
    }

    private static Finding? AnalyzeObjectCreation(GeneratorSyntaxContext ctx, BaseObjectCreationExpressionSyntax creation)
    {
        var type = ctx.SemanticModel.GetSymbolInfo(creation).Symbol?.ContainingType
                   ?? ctx.SemanticModel.GetTypeInfo(creation).Type as INamedTypeSymbol;
        var name = type?.ToDisplayString();

        switch (name)
        {
            case "System.Random":
                return new Finding(RawRandomConstruction, creation.GetLocation());

            case "System.NotImplementedException":
                // A virtual member whose whole body is this throw is abstract-by-convention: it
                // exists to be overridden and throws precisely so a type that overrides NEITHER it
                // nor its predecessor fails loudly. Inventing a default there would be worse than
                // throwing. LayerBase.ForwardTraced is the case in point - virtual only so a
                // migration can proceed one layer at a time without a broken build in between.
                return IsAbstractByConvention(creation)
                    ? null
                    : new Finding(NotImplemented, creation.GetLocation(), EnclosingMemberName(creation));

            case "System.Text.RegularExpressions.Regex" when !HasTimeSpanArgument(ctx, creation.ArgumentList):
                return new Finding(RegexWithoutTimeout, creation.GetLocation(), "new Regex(...)");
        }

        return null;
    }

    private static Finding? AnalyzeInvocation(GeneratorSyntaxContext ctx, InvocationExpressionSyntax invocation)
    {
        if (ctx.SemanticModel.GetSymbolInfo(invocation).Symbol is not IMethodSymbol method)
            return null;

        var owner = method.ContainingType?.ToDisplayString();

        if (owner == "System.Console" && (method.Name == "WriteLine" || method.Name == "Write"))
        {
            return new Finding(ConsoleUsedForLogging, invocation.GetLocation(), $"Console.{method.Name}");
        }

        // Console.Error.WriteLine / Console.Out.WriteLine bind to System.IO.TextWriter (the type of
        // the Console.Error/Out properties), so the invocation's owner is TextWriter, not Console.
        // Catch them by inspecting the receiver: a Write/WriteLine whose target expression is the
        // System.Console.Error or .Out property. (A bare `method.Name == "Error"` never matched —
        // Error is a property, not a method, so GetSymbolInfo never yields a method named Error.)
        if (owner == "System.IO.TextWriter" &&
            (method.Name == "WriteLine" || method.Name == "Write") &&
            invocation.Expression is MemberAccessExpressionSyntax consoleWrite &&
            ctx.SemanticModel.GetSymbolInfo(consoleWrite.Expression).Symbol is IPropertySymbol writer &&
            writer.ContainingType?.ToDisplayString() == "System.Console" &&
            (writer.Name == "Error" || writer.Name == "Out"))
        {
            return new Finding(ConsoleUsedForLogging, invocation.GetLocation(), $"Console.{writer.Name}.{method.Name}");
        }

        // Static Regex helpers take the timeout as their last parameter; without it the default is infinite.
        if (owner == "System.Text.RegularExpressions.Regex" && method.IsStatic &&
            method.Name is "Match" or "Matches" or "IsMatch" or "Replace" or "Split" &&
            !HasTimeSpanArgument(ctx, invocation.ArgumentList))
        {
            return new Finding(RegexWithoutTimeout, invocation.GetLocation(), $"Regex.{method.Name}(...)");
        }

        return null;
    }

    /// <summary>
    /// Flags a catch that discards the exception with no indication that doing so was intended.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Deliberately narrow, because "empty catch" alone is not evidence of a bug. Two signals mark
    /// a discard as considered rather than accidental, and either one exempts it:
    /// </para>
    /// <list type="bullet">
    /// <item><description>a FILTERED catch (<c>catch (ArgumentException)</c>) - the author named
    /// the one failure they expect, so anything else still propagates;</description></item>
    /// <item><description>an explanatory comment in the body - someone wrote down why the failure
    /// is acceptable here.</description></item>
    /// </list>
    /// <para>
    /// What is left is a bare <c>catch { }</c> with nothing in it, which swallows every exception
    /// and says nothing about why. That is the case that hides bugs. The prevailing legitimate use
    /// in this codebase is a best-effort warm-up probe whose real failure surfaces later on the
    /// actual Train/Predict call - flagging those trains people to ignore the rule.
    /// </para>
    /// </remarks>
    private static Finding? AnalyzeCatch(CatchClauseSyntax catchClause)
    {
        var statements = catchClause.Block?.Statements;
        if (statements is null || statements.Value.Count > 0)
            return null;

        // A filtered catch states which failure was expected.
        if (catchClause.Declaration is not null)
            return null;

        // A comment states why the failure is acceptable.
        if (catchClause.Block is not null && HasComment(catchClause.Block))
            return null;

        return new Finding(SwallowedException, catchClause.CatchKeyword.GetLocation());
    }

    private static bool HasComment(SyntaxNode node)
    {
        foreach (var trivia in node.DescendantTrivia(descendIntoTrivia: true))
        {
            if (trivia.IsKind(SyntaxKind.SingleLineCommentTrivia) ||
                trivia.IsKind(SyntaxKind.MultiLineCommentTrivia))
            {
                return true;
            }
        }

        return false;
    }

    private static bool HasTimeSpanArgument(GeneratorSyntaxContext ctx, BaseArgumentListSyntax? arguments)
    {
        if (arguments is null) return false;

        foreach (var argument in arguments.Arguments)
        {
            var type = ctx.SemanticModel.GetTypeInfo(argument.Expression).Type;
            if (type?.ToDisplayString() == "System.TimeSpan")
                return true;
        }

        return false;
    }

    /// <summary>
    /// True when this throw is the ENTIRE body of a virtual member - an abstract-by-convention
    /// member rather than an unfinished one.
    /// </summary>
    /// <remarks>
    /// Deliberately narrow. It requires BOTH that the member is virtual (an override or a plain
    /// method with a stub body is still a real finding) AND that the throw is the whole body (a
    /// virtual method that does work and throws on one branch is still a real finding).
    /// </remarks>
    private static bool IsAbstractByConvention(SyntaxNode node)
    {
        for (var current = node.Parent; current is not null; current = current.Parent)
        {
            if (current is not MethodDeclarationSyntax method)
            {
                // Stop at the first member boundary that is not a method: a property or
                // constructor throwing NotImplementedException is a genuine stub.
                if (current is MemberDeclarationSyntax) return false;
                continue;
            }

            if (!method.Modifiers.Any(SyntaxKind.VirtualKeyword)) return false;

            // Expression body: `=> throw new NotImplementedException(...)`.
            if (method.ExpressionBody is not null)
                return method.ExpressionBody.Expression is ThrowExpressionSyntax;

            // Block body: a single throw statement and nothing else.
            var statements = method.Body?.Statements;
            return statements is { Count: 1 } && statements.Value[0] is ThrowStatementSyntax;
        }

        return false;
    }

    private static string EnclosingMemberName(SyntaxNode node)
    {
        for (var current = node.Parent; current is not null; current = current.Parent)
        {
            switch (current)
            {
                case MethodDeclarationSyntax method: return method.Identifier.ValueText;
                case PropertyDeclarationSyntax property: return property.Identifier.ValueText;
                case ConstructorDeclarationSyntax constructor: return constructor.Identifier.ValueText;
            }
        }

        return "<member>";
    }

    /// <summary>
    /// Reports every settable property an Options copy constructor forgets to assign.
    /// </summary>
    /// <remarks>
    /// Only properties DECLARED on the type are checked. Inherited properties are the base type's
    /// own copy constructor's responsibility, and flagging them here would report the same omission
    /// once per derived type.
    /// </remarks>
    private static EquatableArray<Finding> AnalyzeOptionsCopyConstructor(GeneratorSyntaxContext ctx)
    {
        var declaration = (ClassDeclarationSyntax)ctx.Node;

        var name = declaration.Identifier.ValueText;
        if (!name.EndsWith("Options", System.StringComparison.Ordinal) &&
            !name.EndsWith("Config", System.StringComparison.Ordinal))
            return default;

        if (ctx.SemanticModel.GetDeclaredSymbol(declaration) is not INamedTypeSymbol symbol)
            return default;

        var copyConstructor = FindCopyConstructor(declaration, ctx, symbol);
        if (copyConstructor is null)
            return default;

        // ': this(other)' hands the whole copy to another constructor on this same type, which is
        // then the one that gets checked. ': base(other)' only covers the BASE type's properties,
        // so it does not excuse this type from copying the properties it declares itself.
        if (copyConstructor.Initializer.IsKind(SyntaxKind.ThisConstructorInitializer))
            return default;

        var assigned = CollectAssignedMemberNames(copyConstructor);

        var findings = new List<Finding>();
        foreach (var member in symbol.GetMembers().OfType<IPropertySymbol>())
        {
            if (member.IsStatic || member.IsIndexer || member.SetMethod is null) continue;
            if (member.DeclaredAccessibility == Accessibility.Private) continue;
            if (assigned.Contains(NormalizeMemberName(member.Name))) continue;

            var location = member.Locations.FirstOrDefault(l => l.IsInSource)
                           ?? copyConstructor.Identifier.GetLocation();
            findings.Add(new Finding(CopyConstructorMissesProperty, location, name, member.Name));
        }

        return new EquatableArray<Finding>(findings.ToImmutableArray());
    }

    private static ConstructorDeclarationSyntax? FindCopyConstructor(
        ClassDeclarationSyntax declaration, GeneratorSyntaxContext ctx, INamedTypeSymbol symbol)
    {
        foreach (var constructor in declaration.Members.OfType<ConstructorDeclarationSyntax>())
        {
            var parameters = constructor.ParameterList.Parameters;
            if (parameters.Count != 1) continue;

            var parameterType = parameters[0].Type is null
                ? null
                : ctx.SemanticModel.GetTypeInfo(parameters[0].Type!).Type;

            if (parameterType is null) continue;

            // Compare on the unbound definition so Options<T> copy constructors match.
            if (SymbolEqualityComparer.Default.Equals(
                    parameterType.OriginalDefinition, symbol.OriginalDefinition))
            {
                return constructor;
            }
        }

        return null;
    }

    private static HashSet<string> CollectAssignedMemberNames(ConstructorDeclarationSyntax constructor)
    {
        var assigned = new HashSet<string>(System.StringComparer.Ordinal);

        SyntaxNode? body = constructor.Body;
        body ??= constructor.ExpressionBody;
        if (body is null) return assigned;

        foreach (var assignment in body.DescendantNodes().OfType<AssignmentExpressionSyntax>())
        {
            switch (assignment.Left)
            {
                case IdentifierNameSyntax identifier:
                    assigned.Add(NormalizeMemberName(identifier.Identifier.ValueText));
                    break;
                case MemberAccessExpressionSyntax { Expression: ThisExpressionSyntax } member:
                    assigned.Add(NormalizeMemberName(member.Name.Identifier.ValueText));
                    break;
            }
        }

        return assigned;
    }

    /// <summary>
    /// Folds a property name and its backing-field spellings onto one key, so a copy constructor
    /// that assigns <c>_hiddenDimension</c> counts as copying <c>HiddenDimension</c>.
    /// </summary>
    /// <remarks>
    /// Without this, every manually-backed property in the codebase reports as uncopied: the
    /// prevailing style assigns backing fields directly in copy constructors, which copies the
    /// property just as effectively as assigning through the setter.
    /// </remarks>
    private static string NormalizeMemberName(string name)
    {
        var trimmed = name;
        if (trimmed.StartsWith("m_", System.StringComparison.Ordinal))
            trimmed = trimmed.Substring(2);
        trimmed = trimmed.TrimStart('_');
        return trimmed.ToLowerInvariant();
    }
}

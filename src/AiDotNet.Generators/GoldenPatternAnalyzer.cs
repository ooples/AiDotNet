using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

namespace AiDotNet.Generators;

/// <summary>
/// Roslyn analyzer that enforces the project's golden patterns at BUILD time,
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
[DiagnosticAnalyzer(LanguageNames.CSharp)]
public sealed class GoldenPatternAnalyzer : DiagnosticAnalyzer
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

    internal static readonly DiagnosticDescriptor OptimizerOwnsRandomGenerator = new(
        id: "AIDN077",
        title: "Optimizer builds its own random generator",
        messageFormat: "'{0}' creates a generator of its own; draw from the inherited OptimizerBase.Random, which is built from Options.Seed",
        category: Category,
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true,
        description: "OptimizerBase derives its Random from Options.Seed and re-derives it when options are " +
                     "restored, so an optimizer that owns a private generator silently ignores the seed the " +
                     "caller configured - every run unrepeatable, on the path a model actually uses. " +
                     "ParticleSwarmOptimizer and SimulatedAnnealingOptimizer both did exactly that. Draw from " +
                     "the inherited Random instead; there is then nothing left to forget to seed.");

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

    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics => ImmutableArray.Create(
        CopyConstructorMissesProperty, NullForgivingOperator, RawRandomConstruction, OptimizerOwnsRandomGenerator,
        RegexWithoutTimeout, NotImplemented, ConsoleUsedForLogging, SwallowedException);

    // Every rule that binds an invocation binds a method of one of these names, and an invocation's simple name is
    // the method's own (an alias cannot rename a method); every rule that binds a creation names one of these types.
    // Filtering on syntax first matters: as an incremental generator this bound every invocation and creation in
    // the assembly (a few hundred thousand nodes; 117 s of the net471 compile) and cached an entry for each.
    private static readonly HashSet<string> BoundMethodNames = new(System.StringComparer.Ordinal)
    {
        "WriteLine", "Write", "Match", "Matches", "IsMatch", "Replace", "Split", "CreateSecureRandom", "CreateSeededRandom",
    };

    private static readonly string[] BoundTypeNames = { "Random", "NotImplementedException", "Regex" };

    public override void Initialize(AnalysisContext context)
    {
        // Source generators' output is not checked, as it was not when these rules ran as a generator (which sees
        // only the hand-written trees); no hand-written file here is marked generated.
        context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
        context.EnableConcurrentExecution();
        context.RegisterCompilationStartAction(start =>
        {
            // A creation can name a bound type through a using alias (file, namespace or global), so the alias
            // names join the filter. An alias's target cannot itself be an alias, so one pass finds them all.
            var createdNames = new HashSet<string>(BoundTypeNames, System.StringComparer.Ordinal);
            foreach (var tree in start.Compilation.SyntaxTrees)
            {
                foreach (var directive in tree.GetRoot(start.CancellationToken)
                             .DescendantNodes(n => n is CompilationUnitSyntax || n is BaseNamespaceDeclarationSyntax)
                             .OfType<UsingDirectiveSyntax>())
                {
                    if (directive.Alias is not null && directive.Name is not null
                        && System.Array.IndexOf(BoundTypeNames, RightmostName(directive.Name)) >= 0)
                        createdNames.Add(directive.Alias.Name.Identifier.ValueText);
                }
            }

            start.RegisterSyntaxNodeAction(c => Report(c, new Finding(NullForgivingOperator, c.Node.GetLocation())),
                SyntaxKind.SuppressNullableWarningExpression);
            start.RegisterSyntaxNodeAction(c =>
            {
                var creation = (BaseObjectCreationExpressionSyntax)c.Node;
                if (creation is ObjectCreationExpressionSyntax explicitCreation
                    && RightmostName(explicitCreation.Type) is { } typeName && !createdNames.Contains(typeName)) return;
                Report(c, AnalyzeObjectCreation(c.SemanticModel, creation));
            }, SyntaxKind.ObjectCreationExpression, SyntaxKind.ImplicitObjectCreationExpression);
            start.RegisterSyntaxNodeAction(c =>
            {
                var invocation = (InvocationExpressionSyntax)c.Node;
                if (InvokedName(invocation.Expression) is { } methodName && !BoundMethodNames.Contains(methodName)) return;
                Report(c, AnalyzeInvocation(c.SemanticModel, invocation));
            }, SyntaxKind.InvocationExpression);
            start.RegisterSyntaxNodeAction(c => Report(c, AnalyzeCatch((CatchClauseSyntax)c.Node)), SyntaxKind.CatchClause);
            start.RegisterSyntaxNodeAction(c =>
            {
                foreach (var finding in AnalyzeOptionsCopyConstructor(c.SemanticModel, (ClassDeclarationSyntax)c.Node))
                    Report(c, finding);
            }, SyntaxKind.ClassDeclaration);
        });
    }

    /// <summary>The name a type reference ends in, or null for a form that has none (arrays, pointers, tuples).</summary>
    private static string? RightmostName(TypeSyntax type) => type switch
    {
        SimpleNameSyntax simple => simple.Identifier.ValueText,
        QualifiedNameSyntax qualified => qualified.Right.Identifier.ValueText,
        AliasQualifiedNameSyntax aliased => aliased.Name.Identifier.ValueText,
        _ => null,
    };

    /// <summary>The method name an invocation spells, or null when its target is not a name (then it is bound).</summary>
    private static string? InvokedName(ExpressionSyntax expression) => expression switch
    {
        SimpleNameSyntax simple => simple.Identifier.ValueText,
        MemberAccessExpressionSyntax access => access.Name.Identifier.ValueText,
        MemberBindingExpressionSyntax binding => binding.Name.Identifier.ValueText,
        _ => null,
    };

    private static void Report(SyntaxNodeAnalysisContext context, Finding? f)
    {
        if (f is not null) context.ReportDiagnostic(Diagnostic.Create(f.Descriptor, f.Location, f.Args));
    }

    /// <summary>A single rule violation.</summary>
    internal sealed class Finding
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
    }

    /// <summary>
    /// Whether this node sits inside a type that inherits <c>AiDotNet.Optimizers.OptimizerBase</c>'s
    /// seeded generator.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <c>OptimizerBase</c> itself is excluded: it is the one type that must build the generator,
    /// and it is where <c>Options.Seed</c> is honoured on behalf of everything below it.
    /// </para>
    /// <para>
    /// Resolved as a SYMBOL rather than by simple name. Matching <c>current.Name == "OptimizerBase"</c>
    /// would fire on any unrelated hierarchy that happens to use the name - a consumer's own
    /// OptimizerBase, or a test double - and report AIDN077 against code that inherits no seeded
    /// generator and has nothing to draw from instead.
    /// </para>
    /// </remarks>
    private static bool IsInsideOptimizer(SemanticModel model, SyntaxNode node)
    {
        var declaration = node.FirstAncestorOrSelf<TypeDeclarationSyntax>();
        if (declaration is null) return false;
        if (model.GetDeclaredSymbol(declaration) is not INamedTypeSymbol type) return false;

        // Arity is part of the metadata name: OptimizerBase<T, TInput, TOutput>.
        var optimizerBase = model.Compilation
            .GetTypeByMetadataName("AiDotNet.Optimizers.OptimizerBase`3");
        if (optimizerBase is null) return false;

        for (var current = type.BaseType; current is not null; current = current.BaseType)
        {
            if (SymbolEqualityComparer.Default.Equals(current.OriginalDefinition, optimizerBase))
                return true;
        }

        return false;
    }

    private static Finding? AnalyzeObjectCreation(SemanticModel model, BaseObjectCreationExpressionSyntax creation)
    {
        var type = model.GetSymbolInfo(creation).Symbol?.ContainingType
                   ?? model.GetTypeInfo(creation).Type as INamedTypeSymbol;
        var name = type?.ToDisplayString();

        switch (name)
        {
            case "System.Random":
                return IsInsideOptimizer(model, creation)
                    ? new Finding(OptimizerOwnsRandomGenerator, creation.GetLocation(), "new Random(...)")
                    : new Finding(RawRandomConstruction, creation.GetLocation());

            case "System.NotImplementedException":
                // A virtual member whose whole body is this throw is abstract-by-convention: it
                // exists to be overridden and throws precisely so a type that overrides NEITHER it
                // nor its predecessor fails loudly. Inventing a default there would be worse than
                // throwing. LayerBase.ForwardTraced is the case in point - virtual only so a
                // migration can proceed one layer at a time without a broken build in between.
                return IsAbstractByConvention(creation)
                    ? null
                    : new Finding(NotImplemented, creation.GetLocation(), EnclosingMemberName(creation));

            case "System.Text.RegularExpressions.Regex" when !HasTimeSpanArgument(model, creation.ArgumentList):
                return new Finding(RegexWithoutTimeout, creation.GetLocation(), "new Regex(...)");
        }

        return null;
    }

    private static Finding? AnalyzeInvocation(SemanticModel model, InvocationExpressionSyntax invocation)
    {
        if (model.GetSymbolInfo(invocation).Symbol is not IMethodSymbol method)
            return null;

        var owner = method.ContainingType?.ToDisplayString();

        // RandomHelper is the right answer everywhere EXCEPT inside an optimizer, which already
        // inherits a generator built from Options.Seed. Calling it there is how the seed gets lost.
        if (owner == "AiDotNet.Tensors.Helpers.RandomHelper"
            && (method.Name == "CreateSecureRandom" || method.Name == "CreateSeededRandom")
            && IsInsideOptimizer(model, invocation))
        {
            return new Finding(
                OptimizerOwnsRandomGenerator, invocation.GetLocation(), $"RandomHelper.{method.Name}()");
        }

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
            model.GetSymbolInfo(consoleWrite.Expression).Symbol is IPropertySymbol writer &&
            writer.ContainingType?.ToDisplayString() == "System.Console" &&
            (writer.Name == "Error" || writer.Name == "Out"))
        {
            return new Finding(ConsoleUsedForLogging, invocation.GetLocation(), $"Console.{writer.Name}.{method.Name}");
        }

        // Static Regex helpers take the timeout as their last parameter; without it the default is infinite.
        if (owner == "System.Text.RegularExpressions.Regex" && method.IsStatic &&
            method.Name is "Match" or "Matches" or "IsMatch" or "Replace" or "Split" &&
            !HasTimeSpanArgument(model, invocation.ArgumentList))
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

    private static bool HasTimeSpanArgument(SemanticModel model, BaseArgumentListSyntax? arguments)
    {
        if (arguments is null) return false;

        foreach (var argument in arguments.Arguments)
        {
            var type = model.GetTypeInfo(argument.Expression).Type;
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
    private static List<Finding> AnalyzeOptionsCopyConstructor(SemanticModel model, ClassDeclarationSyntax declaration)
    {
        var findings = new List<Finding>();

        var name = declaration.Identifier.ValueText;
        if (!name.EndsWith("Options", System.StringComparison.Ordinal) &&
            !name.EndsWith("Config", System.StringComparison.Ordinal))
            return findings;

        if (model.GetDeclaredSymbol(declaration) is not INamedTypeSymbol symbol)
            return findings;

        var copyConstructor = FindCopyConstructor(declaration, model, symbol);
        if (copyConstructor is null)
            return findings;

        // ': this(other)' hands the whole copy to another constructor on this same type, which is
        // then the one that gets checked. ': base(other)' only covers the BASE type's properties,
        // so it does not excuse this type from copying the properties it declares itself.
        if (copyConstructor.Initializer.IsKind(SyntaxKind.ThisConstructorInitializer))
            return findings;

        var assigned = CollectAssignedMemberNames(copyConstructor);

        foreach (var member in symbol.GetMembers().OfType<IPropertySymbol>())
        {
            if (member.IsStatic || member.IsIndexer || member.SetMethod is null) continue;
            if (member.DeclaredAccessibility == Accessibility.Private) continue;
            if (assigned.Contains(NormalizeMemberName(member.Name))) continue;

            var location = member.Locations.FirstOrDefault(l => l.IsInSource)
                           ?? copyConstructor.Identifier.GetLocation();
            findings.Add(new Finding(CopyConstructorMissesProperty, location, name, member.Name));
        }

        return findings;
    }

    private static ConstructorDeclarationSyntax? FindCopyConstructor(
        ClassDeclarationSyntax declaration, SemanticModel model, INamedTypeSymbol symbol)
    {
        foreach (var constructor in declaration.Members.OfType<ConstructorDeclarationSyntax>())
        {
            var parameters = constructor.ParameterList.Parameters;
            if (parameters.Count != 1) continue;

            var parameterType = parameters[0].Type is null
                ? null
                : model.GetTypeInfo((parameters[0].Type ?? throw new System.InvalidOperationException("parameters[0].Type is null where it is required."))).Type;

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

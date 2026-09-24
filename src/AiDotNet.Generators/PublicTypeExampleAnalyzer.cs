using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

namespace AiDotNet.Generators;

/// <summary>
/// Reports a public type whose documentation carries no <c>&lt;example&gt;</c> block.
/// </summary>
/// <remarks>
/// <para>
/// An example is the part of the documentation a reader actually copies, and it is the only part a machine
/// can check: prose cannot be compiled, but an <c>&lt;example&gt;</c> can, and the docs job does exactly
/// that. A type with no example is therefore a type whose documentation nothing verifies -- its summary can
/// describe a constructor that no longer exists and no build will ever notice.
/// </para>
/// <para>
/// This is not hypothetical. Around 900 of this library's examples could not compile at all: they named
/// enum members that did not exist, a <c>LatentDiffusionOptions</c> type that was never declared, and
/// MathNet's <c>Matrix.Build.Dense</c> rather than anything in AiDotNet. Some were wrong in ways a reader
/// would not spot -- one classifier example paired six rows of features with labels {0,0,0,1,1,1}, which
/// describes a separable set, but as written the two classes came out identical. They survived because
/// nothing compiled them, and nothing compiled them partly because many types had no example to compile.
/// </para>
/// <para>
/// Reported at <see cref="DiagnosticSeverity.Info"/>, so it guides a type being written now without
/// failing the build over the types that predate the rule. The line that cannot be crossed is held in CI
/// instead, where the example counts are ratcheted in both directions -- the number that compile may not
/// fall and the number that fail may not rise, so a newly added broken example is a red build.
/// </para>
/// </remarks>
[DiagnosticAnalyzer(LanguageNames.CSharp)]
public class PublicTypeExampleAnalyzer : DiagnosticAnalyzer
{
    /// <summary>A public type documented without an example block.</summary>
    private static readonly DiagnosticDescriptor MissingExample = new(
        "ADN0066",
        "Public type has no <example> in its documentation",
        "'{0}' is public but its documentation has no <example> block, so nothing compiles against this "
            + "type's documented usage and it is free to drift. Add an <example><code> showing the type "
            + "being constructed and used, self-contained enough to paste into a new file -- the docs job "
            + "compiles it against the real assemblies",
        "AiDotNet.Documentation",
        DiagnosticSeverity.Info,
        isEnabledByDefault: true,
        description: "Examples are the only documentation a build can verify; a type without one has "
            + "documentation nothing checks.");

    /// <inheritdoc />
    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics
        => ImmutableArray.Create(MissingExample);

    /// <inheritdoc />
    public override void Initialize(AnalysisContext context)
    {
        context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
        context.EnableConcurrentExecution();
        context.RegisterSyntaxNodeAction(
            Analyze,
            SyntaxKind.ClassDeclaration,
            SyntaxKind.StructDeclaration,
            SyntaxKind.InterfaceDeclaration,
            SyntaxKind.RecordDeclaration);
    }

    private static void Analyze(SyntaxNodeAnalysisContext context)
    {
        var declaration = (TypeDeclarationSyntax)context.Node;

        // Only public types: internal ones are not part of anyone's copy-and-paste surface.
        if (!declaration.Modifiers.Any(SyntaxKind.PublicKeyword)) return;

        // A nested type is documented through its container in practice; requiring its own example would
        // fire on option bags and enums-in-classes without telling the author anything useful.
        if (declaration.Parent is TypeDeclarationSyntax) return;

        // A partial type may carry its example on any part, so only report the part holding the doc
        // comment, and stay silent when this part has no documentation at all.
        var trivia = declaration.GetLeadingTrivia();
        bool hasDocComment = trivia.Any(t =>
            t.IsKind(SyntaxKind.SingleLineDocumentationCommentTrivia) ||
            t.IsKind(SyntaxKind.MultiLineDocumentationCommentTrivia));
        if (!hasDocComment) return;

        string docText = trivia.ToFullString();
        if (docText.IndexOf("<example", System.StringComparison.OrdinalIgnoreCase) >= 0) return;
        if (docText.IndexOf("<inheritdoc", System.StringComparison.OrdinalIgnoreCase) >= 0) return;

        context.ReportDiagnostic(Diagnostic.Create(
            MissingExample,
            declaration.Identifier.GetLocation(),
            declaration.Identifier.ValueText));
    }
}

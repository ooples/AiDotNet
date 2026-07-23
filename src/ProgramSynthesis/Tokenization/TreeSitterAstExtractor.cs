using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Models;
using TreeSitter;

namespace AiDotNet.ProgramSynthesis.Tokenization;

internal static class TreeSitterAstExtractor
{
    public static bool TryExtractAst(
        string code,
        ProgramLanguage language,
        CodeTokenizationPipelineOptions options,
        out List<CodeAstNode> nodes,
        out List<CodeAstEdge> edges)
    {
        nodes = new List<CodeAstNode>();
        edges = new List<CodeAstEdge>();

        if (options is null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        if (!options.IncludeAst)
        {
            return false;
        }

        var languageSpec = GetTreeSitterLanguageSpec(language);
        if (languageSpec is null)
        {
            return false;
        }

        var text = code ?? string.Empty;
        var lineStarts = CodeSpanBuilder.ComputeLineStarts(text);
        var byteToChar = BuildByteToCharBoundaries(text);

        using var tsLanguage = new Language(languageSpec.Value.LibraryName, languageSpec.Value.FunctionName);
        using var parser = new Parser(tsLanguage);
        using var tree = parser.Parse(text);

        if (tree is null)
        {
            return false;
        }

        var root = tree.RootNode;
        if (root.IsError)
        {
            return false;
        }

        int maxNodes = Math.Max(1, options.MaxAstNodes);

        var stack = new Stack<(Node Node, int? ParentId)>();
        stack.Push((root, null));

        var nextId = 1;

        while (stack.Count > 0 && nodes.Count < maxNodes)
        {
            var (node, parentId) = stack.Pop();

            var nodeId = nextId++;
            var span = CreateSpan(text, lineStarts, byteToChar, node.StartIndex, node.EndIndex);

            nodes.Add(new CodeAstNode
            {
                NodeId = nodeId,
                ParentNodeId = parentId,
                Language = language,
                Kind = node.Type ?? string.Empty,
                Span = span
            });

            if (parentId.HasValue)
            {
                edges.Add(new CodeAstEdge { ParentNodeId = parentId.Value, ChildNodeId = nodeId });
            }

            if (node.Children is null || node.Children.Count == 0)
            {
                continue;
            }

            for (int i = node.Children.Count - 1; i >= 0; i--)
            {
                stack.Push((node.Children[i], nodeId));
            }
        }

        return nodes.Count > 0;
    }

    private static (string LibraryName, string FunctionName)? GetTreeSitterLanguageSpec(ProgramLanguage language)
    {
        return language switch
        {
            ProgramLanguage.CSharp => ("tree-sitter-c-sharp", "tree_sitter_c_sharp"),
            ProgramLanguage.Python => ("tree-sitter-python", "tree_sitter_python"),
            ProgramLanguage.Java => ("tree-sitter-java", "tree_sitter_java"),
            ProgramLanguage.JavaScript => ("tree-sitter-javascript", "tree_sitter_javascript"),
            ProgramLanguage.TypeScript => ("tree-sitter-typescript", "tree_sitter_typescript"),
            ProgramLanguage.C => ("tree-sitter-c", "tree_sitter_c"),
            ProgramLanguage.CPlusPlus => ("tree-sitter-cpp", "tree_sitter_cpp"),
            ProgramLanguage.Go => ("tree-sitter-go", "tree_sitter_go"),
            ProgramLanguage.Rust => ("tree-sitter-rust", "tree_sitter_rust"),
            _ => null
        };
    }

    private static CodeSpan CreateSpan(
        string text,
        IReadOnlyList<int> lineStarts,
        IReadOnlyList<(int ByteOffset, int CharOffset)> byteToCharBoundaries,
        int startByteOffset,
        int endByteOffset)
    {
        var startCharOffset = ByteOffsetToCharOffset(byteToCharBoundaries, startByteOffset);
        var endCharOffset = ByteOffsetToCharOffset(byteToCharBoundaries, endByteOffset);

        startCharOffset = Math.Max(0, Math.Min(startCharOffset, text.Length));
        endCharOffset = Math.Max(startCharOffset, Math.Min(endCharOffset, text.Length));

        return CodeSpanBuilder.CreateSpan(lineStarts, startCharOffset, endCharOffset);
    }

    private static List<(int ByteOffset, int CharOffset)> BuildByteToCharBoundaries(string text)
    {
        var boundaries = new List<(int ByteOffset, int CharOffset)>(Math.Max(16, text.Length + 1))
        {
            (0, 0)
        };

        int byteOffset = 0;
        int charOffset = 0;

        while (charOffset < text.Length)
        {
            int charsConsumed;
            int codePoint;

            var ch = text[charOffset];
            if (char.IsHighSurrogate(ch) &&
                charOffset + 1 < text.Length &&
                char.IsLowSurrogate(text[charOffset + 1]))
            {
                charsConsumed = 2;
                codePoint = char.ConvertToUtf32(ch, text[charOffset + 1]);
            }
            else
            {
                charsConsumed = 1;
                codePoint = ch;
            }

            byteOffset += Utf8ByteCount(codePoint);
            charOffset += charsConsumed;
            boundaries.Add((byteOffset, charOffset));
        }

        return boundaries;
    }

    private static int Utf8ByteCount(int codePoint)
    {
        if (codePoint <= 0x7F)
        {
            return 1;
        }

        if (codePoint <= 0x7FF)
        {
            return 2;
        }

        if (codePoint <= 0xFFFF)
        {
            return 3;
        }

        return 4;
    }

    private static int ByteOffsetToCharOffset(IReadOnlyList<(int ByteOffset, int CharOffset)> boundaries, int byteOffset)
    {
        if (byteOffset <= 0)
        {
            return 0;
        }

        int lo = 0;
        int hi = boundaries.Count - 1;
        int best = 0;

        while (lo <= hi)
        {
            int mid = lo + ((hi - lo) / 2);
            var candidate = boundaries[mid].ByteOffset;

            if (candidate == byteOffset)
            {
                return boundaries[mid].CharOffset;
            }

            if (candidate < byteOffset)
            {
                best = mid;
                lo = mid + 1;
            }
            else
            {
                hi = mid - 1;
            }
        }

        return boundaries[best].CharOffset;
    }
}

using System.Security.Cryptography;
using Mono.Cecil.Cil;

internal enum SourceDocumentMatch { Exact, EmbeddedLineEndings, Rejected }

internal static class SourceDocumentVerifier
{
    internal static SourceDocumentMatch Match(Document document, byte[] checkout)
    {
        if (MatchesChecksum(document, checkout)) return SourceDocumentMatch.Exact;
        var embedded = document.CustomDebugInformations.OfType<EmbeddedSourceDebugInformation>().Select(info => info.Content).ToList();
        if (document.EmbeddedSource is { Length: > 0 } bytes) embedded.Add(bytes);
        // Do not silently choose one of several conflicting source payloads.
        if (embedded.Count == 0 || embedded.Any(source => source is null || !source.AsSpan().SequenceEqual(embedded[0])) ||
            !MatchesChecksum(document, embedded[0])) return SourceDocumentMatch.Rejected;
        return SameLineEndingContent(embedded[0], checkout) ? SourceDocumentMatch.EmbeddedLineEndings : SourceDocumentMatch.Rejected;
    }

    internal static bool MatchesChecksum(Document document, byte[] bytes)
    {
        byte[]? checksum = document.HashAlgorithm switch
        {
            DocumentHashAlgorithm.SHA256 => SHA256.HashData(bytes),
            DocumentHashAlgorithm.SHA1 => SHA1.HashData(bytes),
            _ => null
        };
        return checksum is not null && checksum.AsSpan().SequenceEqual(document.Hash);
    }

    private static bool SameLineEndingContent(ReadOnlySpan<byte> left, ReadOnlySpan<byte> right)
    {
        // No whitespace, encoding, Unicode, BOM, bare-CR or final-newline
        // normalization. CRLF and LF retain identical source line positions.
        // Runtime string/constant differences still participate in body and
        // metadata hashing; source equivalence never implies binary equivalence.
        int a = 0, b = 0;
        while (a < left.Length && b < right.Length)
        {
            if (left[a] == '\r' && a + 1 < left.Length && left[a + 1] == '\n') a++;
            if (right[b] == '\r' && b + 1 < right.Length && right[b + 1] == '\n') b++;
            if (left[a++] != right[b++]) return false;
        }
        return a == left.Length && b == right.Length;
    }
}

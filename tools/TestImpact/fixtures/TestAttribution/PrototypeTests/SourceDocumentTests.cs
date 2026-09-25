using System.Security.Cryptography;
using System.Text;
using Mono.Cecil.Cil;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "SourceImpact")]
public sealed class SourceDocumentTests
{
    private static Document Document(string source, bool embed = true)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(source);
        var document = new Document("Source.cs") { HashAlgorithm = DocumentHashAlgorithm.SHA256, Hash = SHA256.HashData(bytes) };
        if (embed) document.CustomDebugInformations.Add(new EmbeddedSourceDebugInformation(bytes, true));
        return document;
    }

    [Fact]
    public void CompilerBytesExplainMixedCheckoutLineEndings()
    {
        const string compiled = "first\r\nsecond\nthird\r\n";
        Document document = Document(compiled);
        Assert.Equal(SourceDocumentMatch.Exact, SourceDocumentVerifier.Match(document, Encoding.UTF8.GetBytes(compiled)));
        Assert.Equal(SourceDocumentMatch.EmbeddedLineEndings, SourceDocumentVerifier.Match(document, "first\nsecond\nthird\n"u8.ToArray()));
        Assert.Equal(SourceDocumentMatch.EmbeddedLineEndings, SourceDocumentVerifier.Match(document, "first\r\nsecond\r\nthird\r\n"u8.ToArray()));
    }

    [Theory]
    [InlineData("first\nchanged\n")]
    [InlineData("first \nsecond\n")]
    [InlineData("first\rsecond\r")]
    [InlineData("first\nsecond")]
    [InlineData("\ufefffirst\nsecond\n")]
    public void OnlyLineEndingConversionIsAccepted(string checkout)
    {
        Assert.Equal(SourceDocumentMatch.Rejected, SourceDocumentVerifier.Match(Document("first\r\nsecond\r\n"), Encoding.UTF8.GetBytes(checkout)));
    }

    [Fact]
    public void MissingOrForgedEmbeddedBytesCannotExplainMismatch()
    {
        byte[] checkout = "first\nsecond\n"u8.ToArray();
        Document document = Document("first\r\nsecond\r\n", embed: false);
        Assert.Equal(SourceDocumentMatch.Rejected, SourceDocumentVerifier.Match(document, checkout));
        document.CustomDebugInformations.Add(new EmbeddedSourceDebugInformation(checkout, true));
        Assert.Equal(SourceDocumentMatch.Rejected, SourceDocumentVerifier.Match(document, checkout));
        document = Document("first\r\nsecond\r\n");
        document.CustomDebugInformations.Add(new EmbeddedSourceDebugInformation(checkout, true));
        Assert.Equal(SourceDocumentMatch.Rejected, SourceDocumentVerifier.Match(document, checkout));
    }

    [Fact]
    public void UnsupportedChecksumDoesNotAcceptEvenIdenticalBytes()
    {
        Document document = Document("source\n");
        document.HashAlgorithm = DocumentHashAlgorithm.None;
        Assert.Equal(SourceDocumentMatch.Rejected, SourceDocumentVerifier.Match(document, "source\n"u8.ToArray()));
    }

    [Fact]
    public void LineEndingsInsideRuntimeStringsStillChangeBodyIdentity()
    {
        string Hash(string literal)
        {
            using var assembly = Mono.Cecil.AssemblyDefinition.CreateAssembly(new("LiteralFixture", new(1, 0)), "LiteralFixture", Mono.Cecil.ModuleKind.Dll);
            var type = new Mono.Cecil.TypeDefinition("Fixture", "Literal", Mono.Cecil.TypeAttributes.Public, assembly.MainModule.TypeSystem.Object);
            assembly.MainModule.Types.Add(type);
            var method = new Mono.Cecil.MethodDefinition("Value", Mono.Cecil.MethodAttributes.Public | Mono.Cecil.MethodAttributes.Static, assembly.MainModule.TypeSystem.String);
            type.Methods.Add(method);
            method.Body.Instructions.Add(Instruction.Create(OpCodes.Ldstr, literal));
            method.Body.Instructions.Add(Instruction.Create(OpCodes.Ret));
            using var output = new MemoryStream();
            assembly.Write(output);
            byte[] binary = output.ToArray();
            using var reread = Mono.Cecil.AssemblyDefinition.ReadAssembly(new MemoryStream(binary));
            using var pe = new System.Reflection.PortableExecutable.PEReader(new MemoryStream(binary));
            return SourceSnapshotReader.BodyHash(pe, reread.MainModule.Types.Single(owner => owner.Name == "Literal").Methods.Single());
        }
        Assert.NotEqual(Hash("first\r\nsecond"), Hash("first\nsecond"));
    }
}

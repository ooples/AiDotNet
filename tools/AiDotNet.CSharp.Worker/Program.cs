using System.Runtime.Loader;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;

namespace AiDotNet.CSharp.Worker;

// A compiler/interpreter, NOT a security sandbox. The supervisor must provide the execution boundary.
internal static class Program
{
    private const int MaximumSourceBytes = 262144;
    private const int MaximumSourceChars = 65536;
    private const int MaximumImageBytes = 8 * 1024 * 1024;

    public static int Main(string[] args)
    {
        if (args.Length is < 2 or > 3 || args[0] != "--source" ||
            (args.Length == 3 && args[2] != "--compile-only"))
            return Fail(64, "Usage: AiDotNet.CSharp.Worker --source <file> [--compile-only]");

        try
        {
            string source = ReadSource(args[1]);
            string[] paths = ((string?)AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES") ?? "")
                .Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries);
            if (paths.Length is 0 or > 512) return Fail(70, "Runtime reference catalog unavailable or too large.");
            var references = paths.Select(path => MetadataReference.CreateFromFile(path));
            var tree = CSharpSyntaxTree.ParseText(source, new CSharpParseOptions(LanguageVersion.CSharp12), path: "candidate.cs");
            var compilation = CSharpCompilation.Create("candidate", new[] { tree }, references,
                new CSharpCompilationOptions(OutputKind.ConsoleApplication, optimizationLevel: OptimizationLevel.Release,
                    allowUnsafe: false, deterministic: true, concurrentBuild: false));
            using var image = new BoundedImageStream();
            using var compilationTimeout = new CancellationTokenSource(TimeSpan.FromSeconds(20));
            var emitted = compilation.Emit(image, cancellationToken: compilationTimeout.Token);
            if (!emitted.Success)
            {
                foreach (var diagnostic in emitted.Diagnostics.Where(value => value.Severity == DiagnosticSeverity.Error).Take(8))
                    Console.Error.WriteLine($"{diagnostic.Id}@{diagnostic.Location.SourceSpan.Start}:{diagnostic.Location.SourceSpan.Length}");
                return 65;
            }

            // In particular, module initializers and static constructors cannot run on this path.
            if (args.Length == 3) return 0;

            image.Position = 0;
            var assembly = AssemblyLoadContext.Default.LoadFromStream(image);
            var entry = assembly.EntryPoint ?? throw new InvalidOperationException("No entry point.");
            object? result = entry.Invoke(null, entry.GetParameters().Length == 0 ? null : new object[] { Array.Empty<string>() });
            // Roslyn emits a synchronous CLR entry-point bridge for Task/Task<int> Main methods.
            return result is int exitCode ? exitCode : 0;
        }
        catch (Exception exception) when (exception is not OutOfMemoryException and not StackOverflowException and not AccessViolationException)
        {
            // Do not forward candidate text, exception messages, paths, stack traces or #line directives.
            return Fail(exception is OperationCanceledException ? 75 : 70, "C# worker failed.");
        }
    }

    private static string ReadSource(string path)
    {
        using var file = new FileStream(Path.GetFullPath(path), FileMode.Open, FileAccess.Read, FileShare.Read);
        if (file.Length > MaximumSourceBytes) throw new InvalidDataException();
        using var reader = new StreamReader(file, new UTF8Encoding(false, true), detectEncodingFromByteOrderMarks: false);
        var characters = new char[MaximumSourceChars + 1];
        int count = reader.ReadBlock(characters, 0, characters.Length);
        if (count > MaximumSourceChars || count == 0) throw new InvalidDataException();
        return new string(characters, 0, count);
    }

    private static int Fail(int code, string message) { Console.Error.WriteLine(message); return code; }

    internal sealed class BoundedImageStream : MemoryStream
    {
        public override void Write(byte[] buffer, int offset, int count) { Check(count); base.Write(buffer, offset, count); }
        public override void Write(ReadOnlySpan<byte> buffer) { Check(buffer.Length); base.Write(buffer); }
        public override void WriteByte(byte value) { Check(1); base.WriteByte(value); }
        public override void SetLength(long value)
        {
            if (value > MaximumImageBytes) throw new InvalidDataException();
            base.SetLength(value);
        }
        private void Check(int count)
        {
            if (Position > MaximumImageBytes - count) throw new InvalidDataException();
        }
    }
}

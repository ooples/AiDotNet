using System.Buffers.Binary;
using System.IO.Pipes;
using Xunit;

namespace AiDotNet.Evolve.Cli.Tests;

public sealed class LocalRunControlTests
{
    [Fact]
    public async Task LocalClientCanInspectAndRequestOnlyNamedControls()
    {
        string session = Guid.NewGuid().ToString("N");
        var requests = new List<string>();
        await using var service = new LocalRunControl(session, command =>
        {
            requests.Add(command);
            return "{\"request\":\"" + command + "\"}";
        });
        foreach (string command in new[] { "inspect", "pause", "cancel" })
            Assert.Equal("{\"request\":\"" + command + "\"}", await LocalRunControl.SendAsync(session, command));
        Assert.Equal(new[] { "inspect", "pause", "cancel" }, requests);
        await Assert.ThrowsAsync<ArgumentException>(() => LocalRunControl.SendAsync(session, "execute"));
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(0)]
    [InlineData(33)]
    [InlineData(int.MaxValue)]
    public async Task InvalidLengthCannotAllocateAnUnboundedFrameOrDisableTheNextClient(int length)
    {
        string session = Guid.NewGuid().ToString("N");
        int calls = 0;
        await using var service = new LocalRunControl(session, _ => { calls++; return "{}"; });
        using (var pipe = new NamedPipeClientStream(".", LocalRunControl.PipeName(session), PipeDirection.InOut,
                   PipeOptions.Asynchronous | PipeOptions.CurrentUserOnly))
        {
            await pipe.ConnectAsync(3000);
            var frame = new byte[4];
            BinaryPrimitives.WriteInt32LittleEndian(frame, length);
            await pipe.WriteAsync(frame);
            await pipe.FlushAsync();
            using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(5));
            try { Assert.Equal(0, await pipe.ReadAsync(new byte[1], timeout.Token)); }
            catch (IOException) { } // Windows may report a broken pipe rather than EOF.
        }
        Assert.Equal("{}", await LocalRunControl.SendAsync(session, "inspect"));
        Assert.Equal(1, calls);
    }

    [Fact]
    public async Task DuplicateBindingFailsAndDisposalReleasesTheName()
    {
        string session = Guid.NewGuid().ToString("N");
        var first = new LocalRunControl(session, _ => "{}");
        try { Assert.ThrowsAny<IOException>(() => new LocalRunControl(session, _ => "{}")); }
        finally { await first.DisposeAsync(); }
        await using var next = new LocalRunControl(session, _ => "{}");
        Assert.Equal("{}", await LocalRunControl.SendAsync(session, "inspect"));
    }

    [Theory]
    [InlineData("")]
    [InlineData("../run")]
    [InlineData("a\nb")]
    [InlineData("résumé")]
    public void InvalidSessionNamesAreRejected(string name) => Assert.Throws<ArgumentException>(() => LocalRunControl.PipeName(name));
}

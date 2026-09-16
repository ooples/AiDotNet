<# Invoked by Receive-RequiredArtifact.ps1 -SelfTest; exercises its actual curl transfer loop. #>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not ('ArtifactResumeFixture' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

public enum ArtifactResumeScenario
{
    Interrupted, TimedOut, IgnoredRange, RejectedRange, WrongRange,
    RateLimited, Corrupt, AlreadyComplete, Exhausted, PermissionDenied, DuplicateRange,
    WildcardTotal, ResumedCorrupt, ResumedCorruptExhausted
}

public sealed class ArtifactResumeFixture : IDisposable
{
    private readonly TcpListener listener = new TcpListener(IPAddress.Loopback, 0);
    private readonly Dictionary<ArtifactResumeScenario, List<long>> requests = new Dictionary<ArtifactResumeScenario, List<long>>();
    private readonly object gate = new object();
    private readonly Task server;
    private volatile bool stopped;
    public byte[] Payload { get; } = Encoding.ASCII.GetBytes(new string('a', 1024) + new string('b', 1024) + new string('c', 1024) + new string('d', 1024));
    public int Port { get; }
    public ArtifactResumeFixture()
    {
        listener.Start();
        Port = ((IPEndPoint)listener.LocalEndpoint).Port;
        server = Task.Run(Serve);
    }
    public long[] Offsets(ArtifactResumeScenario scenario)
    {
        lock (gate) { return requests.TryGetValue(scenario, out var values) ? values.ToArray() : Array.Empty<long>(); }
    }
    private void Serve()
    {
        while (!stopped)
        {
            try
            {
                using (var client = listener.AcceptTcpClient())
                using (var stream = client.GetStream())
                using (var reader = new StreamReader(stream, Encoding.ASCII, false, 1024, true))
                {
                    var request = reader.ReadLine() ?? throw new IOException("Missing request");
                    var scenario = (ArtifactResumeScenario)int.Parse(request.Split(' ')[1].TrimStart('/'));
                    long offset = 0;
                    string header;
                    while (!string.IsNullOrEmpty(header = reader.ReadLine()))
                    {
                        if (header.StartsWith("Range: bytes=", StringComparison.OrdinalIgnoreCase))
                            offset = long.Parse(header.Substring(13).TrimEnd('-'));
                        if (header.StartsWith("Authorization:", StringComparison.OrdinalIgnoreCase))
                            throw new InvalidOperationException("Fixture must never receive credentials");
                    }
                    int attempt;
                    lock (gate)
                    {
                        if (!requests.TryGetValue(scenario, out var values)) requests[scenario] = values = new List<long>();
                        values.Add(offset);
                        attempt = values.Count;
                    }
                    int status = offset > 0 ? 206 : 200;
                    int start = (int)offset;
                    int count = Payload.Length - start;
                    int declared = count;
                    string extra = offset > 0 ? $"Content-Range: bytes {offset}-{Payload.Length - 1}/{Payload.Length}\r\n" : "";
                    byte[] bytes = Payload;
                    if (scenario == ArtifactResumeScenario.PermissionDenied)
                    {
                        status = 403; start = 0; count = declared = 0; extra = "";
                    }
                    else if (scenario == ArtifactResumeScenario.AlreadyComplete ||
                        (scenario == ArtifactResumeScenario.RejectedRange && attempt == 2))
                    {
                        status = 416; start = 0; count = declared = 0;
                        extra = $"Content-Range: bytes */{Payload.Length}\r\n";
                    }
                    else if (scenario == ArtifactResumeScenario.RateLimited && attempt == 2)
                    {
                        status = 503; bytes = Encoding.ASCII.GetBytes("unavailable"); start = 0; count = declared = bytes.Length; extra = "";
                    }
                    else if (scenario == ArtifactResumeScenario.Corrupt)
                    {
                        bytes = (byte[])Payload.Clone(); bytes[0] = (byte)'z';
                    }
                    else if (attempt == 1 || scenario == ArtifactResumeScenario.Exhausted)
                    {
                        count = Math.Min(1024, count); // Close early while advertising the full range.
                    }
                    else if (scenario == ArtifactResumeScenario.IgnoredRange)
                    {
                        status = 200; start = 0; count = declared = Payload.Length; extra = "";
                    }
                    else if (scenario == ArtifactResumeScenario.WrongRange)
                    {
                        extra = $"Content-Range: bytes {offset + 1}-{Payload.Length - 1}/{Payload.Length}\r\n";
                    }
                    else if (scenario == ArtifactResumeScenario.DuplicateRange) { extra += extra; }
                    else if (scenario == ArtifactResumeScenario.WildcardTotal)
                    {
                        extra = $"Content-Range: bytes {offset}-{Payload.Length - 1}/*\r\n";
                    }
                    else if ((scenario == ArtifactResumeScenario.ResumedCorrupt ||
                        scenario == ArtifactResumeScenario.ResumedCorruptExhausted) && offset > 0)
                    {
                        bytes = (byte[])Payload.Clone(); bytes[start] = (byte)'z'; // A different representation's suffix.
                    }
                    var response = Encoding.ASCII.GetBytes($"HTTP/1.1 {status} Fixture\r\nContent-Length: {declared}\r\n{extra}Connection: close\r\n\r\n");
                    stream.Write(response, 0, response.Length);
                    stream.Write(bytes, start, count);
                    stream.Flush();
                    if (scenario == ArtifactResumeScenario.TimedOut && attempt == 1) Thread.Sleep(1500);
                }
            }
            catch (IOException) { if (stopped) return; }
            catch (InvalidOperationException) when (stopped) { return; }
            catch (SocketException) { if (stopped) return; throw; }
        }
    }
    public void Dispose()
    {
        stopped = true;
        listener.Stop();
        server.GetAwaiter().GetResult();
    }
}
'@
}

$fixture = [ArtifactResumeFixture]::new()
$directory = Join-Path ([IO.Path]::GetTempPath()) ('artifact-resume-' + [guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($directory) | Out-Null
$archive = Join-Path $directory 'artifact.zip'
$expected = 'sha256:' + [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($fixture.Payload)).ToLowerInvariant()
try {
    foreach ($scenario in [Enum]::GetValues([ArtifactResumeScenario])) {
        if (Test-Path -LiteralPath $archive) { Remove-Item -LiteralPath $archive }
        if ($scenario -eq [ArtifactResumeScenario]::AlreadyComplete) {
            [IO.File]::WriteAllBytes($archive, $fixture.Payload)
        }
        # Only TimedOut needs a request to outlive curl's --max-time; a one-second limit elsewhere would let a
        # slow loopback response add an unplanned retry and break the offset assertions below.
        $timeout = if ($scenario -eq [ArtifactResumeScenario]::TimedOut) { 1 } else { 30 }
        $attempts = if ($scenario -eq [ArtifactResumeScenario]::ResumedCorruptExhausted) { 2 } else { 3 }
        $failure = $null
        try {
            Receive-ArtifactArchive -Url "http://127.0.0.1:$($fixture.Port)/$([int]$scenario)" `
                -Archive $archive -Expected $expected -RequestHeaders @() -Attempts $attempts `
                -WorkflowRunId 100 -MatrixJobIndex 0 -RequestTimeoutSeconds $timeout -Delay { param($seconds) }
        }
        catch { $failure = $_.Exception.Message }
        $expectedFailure = switch ($scenario) {
            ([ArtifactResumeScenario]::Corrupt) { 'SHA-256' }
            ([ArtifactResumeScenario]::ResumedCorruptExhausted) { 'SHA-256' }
            ([ArtifactResumeScenario]::Exhausted) { 'download failed' }
            ([ArtifactResumeScenario]::PermissionDenied) { 'HTTP=403' }
            default { $null }
        }
        if ($null -ne $expectedFailure) {
            Assert-True ($null -ne $failure -and $failure.Contains($expectedFailure)) "$scenario did not fail closed: $failure"
        }
        else {
            Assert-True ($null -eq $failure) "$scenario failed: $failure"
            Assert-True (Test-ArtifactDigest $archive $expected) "$scenario produced an incorrect archive"
        }
        $expectedOffsets = switch ($scenario) {
            ([ArtifactResumeScenario]::AlreadyComplete) { @(4096) }
            ([ArtifactResumeScenario]::PermissionDenied) { @(0) }
            ([ArtifactResumeScenario]::Corrupt) { @(0) }
            ([ArtifactResumeScenario]::RejectedRange) { @(0, 1024, 0) }
            # A range that cannot continue the prefix, or a resumed archive failing its digest, restarts at 0.
            ([ArtifactResumeScenario]::WrongRange) { @(0, 1024, 0) }
            ([ArtifactResumeScenario]::DuplicateRange) { @(0, 1024, 0) }
            ([ArtifactResumeScenario]::ResumedCorrupt) { @(0, 1024, 0) }
            ([ArtifactResumeScenario]::RateLimited) { @(0, 1024, 1024) }
            ([ArtifactResumeScenario]::Exhausted) { @(0, 1024, 2048) }
            default { @(0, 1024) }
        }
        Assert-True (($fixture.Offsets($scenario) -join ',') -ceq ($expectedOffsets -join ',')) `
            "$scenario did not use the expected resume offsets"
    }
    Write-Host "Artifact resume transport: $([Enum]::GetValues([ArtifactResumeScenario]).Length) real-curl loopback scenarios passed."
}
finally {
    $fixture.Dispose()
    foreach ($file in @($archive, "$archive.response", "$archive.headers")) {
        if (Test-Path -LiteralPath $file) { Remove-Item -LiteralPath $file }
    }
    Remove-Item -LiteralPath $directory
}

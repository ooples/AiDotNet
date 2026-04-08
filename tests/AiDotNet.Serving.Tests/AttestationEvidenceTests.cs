using AiDotNet.Serving.Models;
using Xunit;

namespace AiDotNet.Serving.Tests;

public class AttestationEvidenceTests
{
    [Fact]
    public void Defaults_AreEmptyStrings()
    {
        var evidence = new AttestationEvidence();

        Assert.Equal(string.Empty, evidence.Platform);
        Assert.Equal(string.Empty, evidence.TeeType);
        Assert.Equal(string.Empty, evidence.Nonce);
        Assert.Equal(string.Empty, evidence.AttestationToken);
    }

    [Fact]
    public void Properties_CanBeSet()
    {
        var evidence = new AttestationEvidence
        {
            Platform = "Windows",
            TeeType = "TDX",
            Nonce = "n",
            AttestationToken = "token"
        };

        Assert.Equal("Windows", evidence.Platform);
        Assert.Equal("TDX", evidence.TeeType);
        Assert.Equal("n", evidence.Nonce);
        Assert.Equal("token", evidence.AttestationToken);
    }
}


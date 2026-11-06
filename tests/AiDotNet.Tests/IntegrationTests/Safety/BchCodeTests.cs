using System;
using System.Linq;
using AiDotNet.Safety.Watermarking;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Safety;

/// <summary>StegaStamp's BCH code: GF(2^7), polynomial 137, t = 5, 56 data bits.</summary>
public class BchCodeTests
{
    [Fact]
    public void StegaStampCode_HasThePapersLayout()
    {
        var code = BchCode.StegaStamp();
        Assert.Equal(56, code.DataBits);
        Assert.Equal(35, code.ParityBits);
        Assert.Equal(5, code.Capacity);
        Assert.Equal(91, code.CodewordBits);
    }

    [Fact]
    public void UpToFiveFlippedBits_AreCorrected_AtAnyPositions()
    {
        var code = BchCode.StegaStamp();
        var rng = RandomHelper.CreateSeededRandom(17);
        for (int trial = 0; trial < 300; trial++)
        {
            var data = Enumerable.Range(0, code.DataBits).Select(_ => rng.Next(2) == 1).ToArray();
            var codeword = code.Encode(data);
            Assert.Equal(data, codeword.Take(code.DataBits).ToArray());
            int errors = trial % 6;
            foreach (int i in Enumerable.Range(0, code.CodewordBits).OrderBy(_ => rng.Next()).Take(errors)) codeword[i] = !codeword[i];

            var decoded = code.Decode(codeword);
            Assert.True(decoded.HasValue, $"Trial {trial}: {errors} errors were not corrected.");
            Assert.Equal(data, decoded!.Value.Data);
            Assert.Equal(errors, decoded.Value.Corrected);
        }
    }

    [Fact]
    public void BeyondCapacity_TheDecoderUsuallyRefuses_AndNeverClaimsMoreThanItCanLocate()
    {
        var code = BchCode.StegaStamp();
        var rng = RandomHelper.CreateSeededRandom(23);
        int refused = 0, wrong = 0;
        for (int trial = 0; trial < 300; trial++)
        {
            var data = Enumerable.Range(0, code.DataBits).Select(_ => rng.Next(2) == 1).ToArray();
            var codeword = code.Encode(data);
            foreach (int i in Enumerable.Range(0, code.CodewordBits).OrderBy(_ => rng.Next()).Take(8)) codeword[i] = !codeword[i];
            var decoded = code.Decode(codeword);
            if (!decoded.HasValue) refused++;
            else
            {
                Assert.InRange(decoded.Value.Corrected, 1, code.Capacity);
                if (!decoded.Value.Data.SequenceEqual(data)) wrong++;
            }
        }

        // With 8 errors the received word lies within distance 5 of another codeword only rarely.
        Assert.True(refused > 250, $"Only {refused} of 300 eight-error words were refused ({wrong} miscorrected).");
    }
}

using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Helpers;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.Helpers;

/// <summary>ZlibDeflate reproduces Python's zlib.compress (zlib 1.3.2, level 6) byte for byte
/// (tools/reference-data/zlib_reference.py): empty, short, repetitive, multi-script, random, window-sliding and
/// multi-block inputs.</summary>
public class ZlibDeflateTests
{
    [Fact(Timeout = 120000)]
    public async Task Compress_MatchesPythonZlib()
    {
        await Task.Yield();
        const string fileName = "zlib_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "Helpers", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "Helpers", "ReferenceData", fileName);
        int index = 0;
        foreach (var c in JObject.Parse(File.ReadAllText(path))["cases"]!)
        {
            var input = Convert.FromBase64String((string)c["input"]!);
            var expected = Convert.FromBase64String((string)c["output"]!);
            var actual = ZlibDeflate.Compress(input);
            int first = Enumerable.Range(0, Math.Min(expected.Length, actual.Length)).FirstOrDefault(i => expected[i] != actual[i], -1);
            Assert.True(expected.SequenceEqual(actual), $"case {index} ({input.Length} bytes): zlib {expected.Length} bytes, port {actual.Length}, first difference at {first}");
            index++;
        }
    }
}

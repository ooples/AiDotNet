"""Check the actual C# report schema and Python reader without training models.

Run with: python -m unittest discover -s benchmarks/AiDotNet.PyTorchParity/pytorch -p test_report_serialization.py
Requires the .NET 10 SDK; no AiDotNet rebuild or Python ML packages are needed.
"""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import compare


class ReportSerializationTests(unittest.TestCase):
    def test_endpoint_losses_survive_both_readers(self):
        source = (Path(__file__).resolve().parents[1] / "Program.cs").read_text(encoding="utf-8")
        # Compile the production records and options, not a duplicate of the schema.
        records = source[source.index("internal sealed record BenchmarkReport("):]
        records = records[:records.index("/// <summary>")]
        with tempfile.TemporaryDirectory(prefix="parity-report-") as temp:
            root = Path(temp)
            (root / "ReportCheck.csproj").write_text(
                '<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup>'
                '<OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework>'
                '<ImplicitUsings>enable</ImplicitUsings><Nullable>enable</Nullable>'
                '</PropertyGroup></Project>', encoding="utf-8")
            (root / "Program.cs").write_text(r'''
using System.Text.Json;

double?[] losses = [null, 1.25, double.NaN, double.PositiveInfinity, double.NegativeInfinity];
var reports = new List<TrainingReport>();
foreach (var loss in losses)
{
    var report = new TrainingReport([1, 2], 3, 2, 2, 1.5, 2.5, 4, .1, .2,
        new ResourceReport(10, 12, null), loss, loss);
    var json = JsonSerializer.Serialize(report, JsonOptions.Default);
    var restored = JsonSerializer.Deserialize<TrainingReport>(json, JsonOptions.Default)
        ?? throw new Exception("Report was lost");
    if (!Nullable.Equals(loss, restored.FirstLoss) || !Nullable.Equals(loss, restored.LastLoss))
        throw new Exception("Loss failed to round-trip: " + json);
    if (restored.TotalSeconds != 3 || restored.Resources.ProcessRssMbPeak != 10)
        throw new Exception("Report metrics were lost");
    reports.Add(report);
}
File.WriteAllText(args[0], JsonSerializer.Serialize(reports, JsonOptions.Default));
''' + records, encoding="utf-8")
            output = root / "reports.json"
            result = subprocess.run(
                ["dotnet", "run", "--project", str(root / "ReportCheck.csproj"),
                 "--configuration", "Release", "--", str(output)],
                capture_output=True, text=True, encoding="utf-8", timeout=120)
            self.assertEqual(0, result.returncode, result.stdout + result.stderr)
            # Reject non-standard bare NaN/Infinity tokens: the wire format must be valid JSON.
            reports = json.loads(output.read_text(encoding="utf-8"),
                                 parse_constant=lambda value: self.fail(f"Invalid JSON number: {value}"))

        for report, loss in zip(reports, [None, 1.25, "NaN", "Infinity", "-Infinity"], strict=True):
            with self.subTest(loss=loss):
                self.assertEqual(loss, report["FirstLoss"])
                self.assertEqual(loss, report["LastLoss"])
                rows = compare._index({"Results": [{
                    "Model": "mlp", "Device": "cpu", "Training": report, "Inference": []
                }]})
                self.assertEqual(500, rows["mlp"].training.median)
                self.assertEqual(375, rows["mlp"].training.p25)
                self.assertEqual(625, rows["mlp"].training.p75)

class InferenceBatchParsingTests(unittest.TestCase):
    """benchmark.py must accept exactly what the C# harness's NumberStyles.Integer parse accepts."""

    def setUp(self):
        try:
            import benchmark
        except ImportError as error:  # benchmark.py imports torch
            self.skipTest(f"benchmark.py needs PyTorch: {error}")
        self.parse = benchmark.parse_inference_batches

    def test_accepts_what_csharp_accepts(self):
        self.assertEqual(self.parse("1,8,32"), [1, 8, 32])
        self.assertEqual(self.parse(" 4 , +5"), [4, 5])
        self.assertEqual(self.parse("2147483647"), [2147483647])

    def test_rejects_what_csharp_rejects(self):
        for text in ["1_0", "2147483648", "\u0661", "1.5", "0", "-3", "1,,2", ""]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.parse(text)

if __name__ == "__main__":
    unittest.main()

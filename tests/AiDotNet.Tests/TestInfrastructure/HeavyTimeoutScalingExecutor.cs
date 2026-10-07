#if !AIDOTNET_TEST_ATTRIBUTION
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Reflection;
using Xunit.Abstractions;
using Xunit.Sdk;

namespace AiDotNet.Tests.TestInfrastructure;

/// <summary>
/// Gives HeavyTimeout tests a longer per-test xUnit timeout when the nightly heavy lane asks for it (#2087).
/// </summary>
/// <remarks>
/// <para>
/// The <c>[Fact(Timeout = 120000)]</c> budgets are the PR gate's, where HeavyTimeout tests never run. In the nightly
/// lane a paper-scale model's training invariant legitimately needs longer: a 477.6M-parameter U-Net's
/// Training_ShouldReducePredictionError costs about 230 s on the 4-core runner. Timeouts are attribute constants, so
/// the lane sets <c>AIDOTNET_HEAVY_TEST_TIMEOUT_SCALE</c> and this executor multiplies the timeout of every test case
/// carrying <c>Category=HeavyTimeout</c> before it runs. Assertions are untouched, other tests keep their budget, and a
/// genuine hang is still stopped by the lane's per-chunk cap and blame-hang detector.
/// </para>
/// <para>
/// xUnit v2 exposes a test case's timeout with a protected setter only, so it is set through reflection. If a future
/// xUnit drops the setter the scale is reported and skipped rather than silently assumed.
/// </para>
/// </remarks>
internal sealed class HeavyTimeoutScalingExecutor : XunitTestFrameworkExecutor
{
    internal const string ScaleVariable = "AIDOTNET_HEAVY_TEST_TIMEOUT_SCALE";
    private const string CategoryTrait = "Category";
    private const string HeavyTimeoutCategory = "HeavyTimeout";

    private static readonly MethodInfo? TimeoutSetter =
        typeof(XunitTestCase).GetProperty(nameof(XunitTestCase.Timeout))?.GetSetMethod(nonPublic: true);

    public HeavyTimeoutScalingExecutor(AssemblyName assemblyName, ISourceInformationProvider sourceInformationProvider,
        IMessageSink diagnosticMessageSink)
        : base(assemblyName, sourceInformationProvider, diagnosticMessageSink)
    {
    }

    internal static int ReadScale() => ParseScale(Environment.GetEnvironmentVariable(ScaleVariable));

    internal static int ParseScale(string? raw) =>
        int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out int scale) && scale > 1 ? scale : 1;

    protected override void RunTestCases(IEnumerable<IXunitTestCase> testCases, IMessageSink executionMessageSink,
        ITestFrameworkExecutionOptions executionOptions)
    {
        int scale = ReadScale();
        if (scale > 1)
        {
            if (TimeoutSetter is null)
            {
                DiagnosticMessageSink.OnMessage(new DiagnosticMessage(
                    $"{ScaleVariable}={scale} ignored: XunitTestCase.Timeout has no setter in this xUnit version."));
            }
            else
            {
                testCases = testCases.ToList();
                foreach (var testCase in testCases)
                {
                    if (testCase is XunitTestCase xunitCase && xunitCase.Timeout > 0 && IsHeavyTimeout(testCase.Traits))
                        TimeoutSetter.Invoke(xunitCase, new object[] { (int)Math.Min(int.MaxValue, (long)xunitCase.Timeout * scale) });
                }
            }
        }

        base.RunTestCases(testCases, executionMessageSink, executionOptions);
    }

    internal static bool IsHeavyTimeout(Dictionary<string, List<string>>? traits) =>
        traits is not null
        && traits.TryGetValue(CategoryTrait, out var categories)
        && categories.Contains(HeavyTimeoutCategory, StringComparer.Ordinal);
}
#endif

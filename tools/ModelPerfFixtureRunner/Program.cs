// Copyright (c) AiDotNet. All rights reserved.

using System.Reflection;
using Xunit.Abstractions;

namespace AiDotNet.Tools.ModelPerfFixtureRunner;

/// <summary>
/// Runs one exact generated model fixture without asking VSTest/xUnit to rediscover the repository's
/// roughly 75,000 unrelated tests. The outer shard runner owns OS-process isolation and the hard
/// timeout; this host owns the same initialize/test/dispose lifecycle xUnit applies to the fixture.
/// </summary>
internal static class Program
{
    private static async Task<int> Main(string[] args)
    {
        if (args.Length == 2 && args[0] == "--validate-inventory")
            return ValidateInventory(args[1]);

        if (args.Length != 1 || string.IsNullOrWhiteSpace(args[0]))
        {
            Console.Error.WriteLine("usage: ModelPerfFixtureRunner <fully-qualified-fixture-type>");
            return 2;
        }

        object? fixture = null;
        int exitCode = 0;
        try
        {
            Assembly tests = Assembly.Load("AiDotNetTests");
            Type fixtureType = tests.GetType(args[0], throwOnError: true, ignoreCase: false)!;
            fixture = ConstructFixture(fixtureType);

            await InvokeLifecycleMethod(fixture, "InitializeAsync").ConfigureAwait(false);
            MethodInfo census = fixtureType.GetMethod(
                "ModelPerformanceCensus",
                BindingFlags.Instance | BindingFlags.Public)
                ?? throw new MissingMethodException(fixtureType.FullName, "ModelPerformanceCensus");
            await AwaitResult(census.Invoke(fixture, null)).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            Exception diagnostic = ex is TargetInvocationException { InnerException: not null } invocation
                ? invocation.InnerException!
                : ex;
            Console.Error.WriteLine(diagnostic);
            exitCode = 1;
        }
        finally
        {
            if (fixture is not null)
            {
                try
                {
                    MethodInfo? disposeAsync = fixture.GetType().GetMethod(
                        "DisposeAsync", BindingFlags.Instance | BindingFlags.Public);
                    if (disposeAsync is not null)
                        await AwaitResult(disposeAsync.Invoke(fixture, null)).ConfigureAwait(false);
                    else if (fixture is IDisposable disposable)
                        disposable.Dispose();
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"Fixture disposal failed: {ex}");
                    exitCode = 1;
                }
            }
        }

        return exitCode;
    }

    private static object ConstructFixture(Type fixtureType)
    {
        ConstructorInfo[] constructors = fixtureType.GetConstructors(BindingFlags.Instance | BindingFlags.Public);
        ConstructorInfo? parameterless = constructors.FirstOrDefault(c => c.GetParameters().Length == 0);
        if (parameterless is not null)
            return parameterless.Invoke(null);

        ConstructorInfo? outputConstructor = constructors.FirstOrDefault(c =>
            c.GetParameters() is [{ ParameterType: var type }] && type == typeof(ITestOutputHelper));
        if (outputConstructor is not null)
            return outputConstructor.Invoke([NullTestOutputHelper.Instance]);

        throw new InvalidOperationException(
            $"Performance fixture '{fixtureType.FullName}' must have either a public parameterless " +
            "constructor or the standard single ITestOutputHelper constructor.");
    }

    private static int ValidateInventory(string path)
    {
        Assembly tests = Assembly.Load("AiDotNetTests");
        int count = 0;
        foreach (string line in File.ReadLines(path))
        {
            string testName = line.Trim();
            if (!testName.EndsWith(".ModelPerformanceCensus", StringComparison.Ordinal)) continue;
            string fixtureName = testName[..^".ModelPerformanceCensus".Length];
            Type fixtureType = tests.GetType(fixtureName, throwOnError: true, ignoreCase: false)!;
            bool supportedConstructor = fixtureType.GetConstructors(BindingFlags.Instance | BindingFlags.Public)
                .Any(constructor => constructor.GetParameters().Length == 0
                    || constructor.GetParameters() is [{ ParameterType: var type }]
                    && type == typeof(ITestOutputHelper));
            if (fixtureType.IsAbstract || !supportedConstructor)
                throw new InvalidOperationException(
                    $"Performance fixture '{fixtureName}' must be concrete and use a supported constructor.");
            if (fixtureType.GetMethod("ModelPerformanceCensus", BindingFlags.Instance | BindingFlags.Public) is null)
                throw new MissingMethodException(fixtureName, "ModelPerformanceCensus");
            count++;
        }

        if (count == 0)
            throw new InvalidOperationException($"No ModelPerformanceCensus fixtures were found in '{path}'.");
        Console.WriteLine($"Validated {count} exact-fixture performance host target(s).");
        return 0;
    }

    private static async Task InvokeLifecycleMethod(object fixture, string methodName)
    {
        MethodInfo? method = fixture.GetType().GetMethod(
            methodName, BindingFlags.Instance | BindingFlags.Public);
        if (method is not null)
            await AwaitResult(method.Invoke(fixture, null)).ConfigureAwait(false);
    }

    private static async Task AwaitResult(object? result)
    {
        switch (result)
        {
            case Task task:
                await task.ConfigureAwait(false);
                break;
            case ValueTask valueTask:
                await valueTask.ConfigureAwait(false);
                break;
            case null:
                break;
            default:
                throw new InvalidOperationException(
                    $"Fixture lifecycle returned unsupported awaitable '{result.GetType().FullName}'.");
        }
    }

    private sealed class NullTestOutputHelper : ITestOutputHelper
    {
        public static NullTestOutputHelper Instance { get; } = new();
        public void WriteLine(string message) { }
        public void WriteLine(string format, params object[] args) { }
    }
}

param(
    [Parameter(Mandatory = $true)]
    [string] $PackagePath
)

$ErrorActionPreference = 'Stop'
$resolvedPackage = (Resolve-Path -LiteralPath $PackagePath).Path
$packageDirectory = Split-Path -Parent $resolvedPackage
$nugetOrgSource = 'https://api.nuget.org/v3/index.json'
$probeRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("aidotnet-generator-probe-" + [guid]::NewGuid().ToString('N'))
$probePackages = Join-Path $probeRoot '.packages'
$nugetConfig = Join-Path $probeRoot 'NuGet.Config'

try {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [System.IO.Compression.ZipFile]::OpenRead($resolvedPackage)
    try {
        $generatorEntry = $archive.Entries | Where-Object {
            $_.FullName -eq 'analyzers/dotnet/cs/AiDotNet.Generators.dll'
        }
        if ($null -eq $generatorEntry) {
            throw "Package does not contain analyzers/dotnet/cs/AiDotNet.Generators.dll."
        }

        $generatorStream = $generatorEntry.Open()
        $sha256 = [System.Security.Cryptography.SHA256]::Create()
        try {
            $packagedGeneratorHash = [System.BitConverter]::ToString(
                $sha256.ComputeHash($generatorStream)).Replace('-', '')
        }
        finally {
            $sha256.Dispose()
            $generatorStream.Dispose()
        }

        $nuspecEntry = $archive.Entries | Where-Object { $_.FullName -like '*.nuspec' } |
            Select-Object -First 1
        if ($null -eq $nuspecEntry) {
            throw 'Package does not contain a NuGet manifest.'
        }

        $reader = [System.IO.StreamReader]::new($nuspecEntry.Open())
        try {
            [xml] $nuspec = $reader.ReadToEnd()
        }
        finally {
            $reader.Dispose()
        }

        $packageId = $nuspec.SelectSingleNode(
            "/*[local-name()='package']/*[local-name()='metadata']/*[local-name()='id']").InnerText
        $packageVersion = $nuspec.SelectSingleNode(
            "/*[local-name()='package']/*[local-name()='metadata']/*[local-name()='version']").InnerText
        if ([string]::IsNullOrWhiteSpace($packageId) -or [string]::IsNullOrWhiteSpace($packageVersion)) {
            throw 'Package manifest does not contain an id and version.'
        }
    }
    finally {
        $archive.Dispose()
    }

    New-Item -ItemType Directory -Path $probeRoot | Out-Null
    $escapedPackageDirectory = [System.Security.SecurityElement]::Escape($packageDirectory)
    $configText = @"
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <clear />
    <add key="package-under-test" value="$escapedPackageDirectory" />
    <add key="nuget.org" value="$nugetOrgSource" protocolVersion="3" />
  </packageSources>
</configuration>
"@
    Set-Content -LiteralPath $nugetConfig -Value $configText -Encoding utf8

    & dotnet new classlib --framework net8.0 --output $probeRoot --no-restore
    if ($LASTEXITCODE -ne 0) { throw 'dotnet new failed.' }

    # Pin the exact archive under test. Without --version, a configured remote feed could
    # satisfy the request with a different package and turn this probe into a false positive.
    & dotnet add $probeRoot package $packageId --version $packageVersion --no-restore
    if ($LASTEXITCODE -ne 0) { throw 'Adding the packed AiDotNet package failed.' }

    & dotnet restore $probeRoot --configfile $nugetConfig --packages $probePackages `
        --force --no-http-cache --verbosity minimal
    if ($LASTEXITCODE -ne 0) { throw 'Restoring the packed AiDotNet package failed.' }

    $restoredGenerator = Join-Path $probePackages $packageId.ToLowerInvariant()
    $restoredGenerator = Join-Path $restoredGenerator $packageVersion.ToLowerInvariant()
    $restoredGenerator = Join-Path $restoredGenerator 'analyzers'
    $restoredGenerator = Join-Path $restoredGenerator 'dotnet'
    $restoredGenerator = Join-Path $restoredGenerator 'cs'
    $restoredGenerator = Join-Path $restoredGenerator 'AiDotNet.Generators.dll'
    if (!(Test-Path -LiteralPath $restoredGenerator)) {
        throw 'The restored package does not expose AiDotNet.Generators.dll as a C# analyzer.'
    }
    if ((Get-FileHash -LiteralPath $restoredGenerator -Algorithm SHA256).Hash -ne $packagedGeneratorHash) {
        throw 'Restore selected a different generator binary than the package under test.'
    }

    $probeSource = @'
using AiDotNet.NeuralNetworks.Layers;

namespace PackedConsumer;

[AiDotNet.Attributes.ElementWiseShape]
public sealed partial class PackedLayer<T> : LayerBase<T>
{
    private readonly int _units;
    private readonly bool _useBias;

    public PackedLayer(int units, bool useBias = true) : base([units], [units])
    {
        _units = units;
        _useBias = useBias;
    }

    public override bool SupportsTraining => false;
    public override void ResetState() { }
}

public static class GeneratorProof
{
    // This type is emitted into THIS assembly by LayerStateGenerator. The package's runtime
    // reflection fallback cannot make this compile, so a successful build proves the analyzer
    // asset flowed automatically through PackageReference.
    public static int FactoryCount => AiDotNet.Serialization.GeneratedLayerFactories<double>.Count;
}
'@
    Set-Content -LiteralPath (Join-Path $probeRoot 'Class1.cs') -Value $probeSource -Encoding utf8

    & dotnet build $probeRoot --no-restore --configuration Release --verbosity minimal -nologo -clp:ErrorsOnly
    if ($LASTEXITCODE -ne 0) {
        throw 'The PackageReference consumer did not compile with generated layer factories.'
    }
}
finally {
    if (Test-Path -LiteralPath $probeRoot) {
        $resolvedProbe = [System.IO.Path]::GetFullPath($probeRoot)
        $resolvedTemp = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
        if (!$resolvedProbe.StartsWith($resolvedTemp, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to remove probe outside the system temp directory: $resolvedProbe"
        }
        Remove-Item -LiteralPath $resolvedProbe -Recurse -Force
    }
}

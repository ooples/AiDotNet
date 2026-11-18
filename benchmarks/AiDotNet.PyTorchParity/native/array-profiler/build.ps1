param([Parameter(Mandatory=$true)][string]$OutputDirectory)
$ErrorActionPreference = 'Stop'
$taskOutput = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Path $taskOutput -Force | Out-Null
$taskHeaders = Join-Path $taskOutput 'headers'
New-Item -ItemType Directory -Path $taskHeaders -Force | Out-Null
$taskCommit = '60629d14374c56f1cb51819049ad1fa529307f8d'
$taskSources = @{
    'corprof.h' = @('src/coreclr/pal/prebuilt/inc/corprof.h', '6892e8f1c8492290d19cd9c9b3f73d840f02b05cdcee8ba344ab3614ad66406a')
    'cor.h' = @('src/coreclr/inc/cor.h', '8be143fbedafa9ccc27bc8dbf1e292046fbe3c1e902d808bb9aa289f53a2cde8')
    'corhdr.h' = @('src/coreclr/inc/corhdr.h', 'ac7eca591deb91f913d77f6877386a986bdfd9f8bab342e660d2fc26e7d212f4')
    'corerror.h' = @('src/coreclr/pal/prebuilt/inc/corerror.h', '852b0559ce5d040abaf35f0f03c59da23c5e2ed9559188a0d9648da8bc2805e0')
}
foreach ($taskName in $taskSources.Keys) {
    $taskFile = Join-Path $taskHeaders $taskName
    if (-not (Test-Path -LiteralPath $taskFile)) {
        Invoke-WebRequest -Uri ('https://raw.githubusercontent.com/dotnet/runtime/' + $taskCommit + '/' + $taskSources[$taskName][0]) -OutFile $taskFile
    }
    if ((Get-FileHash -LiteralPath $taskFile -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskSources[$taskName][1]) {
        throw ('Pinned CoreCLR header hash differs: ' + $taskName)
    }
}
$taskVsWhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$taskVs = & $taskVsWhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $taskVs) { throw 'MSVC x64 Build Tools were not found.' }
$taskVcvars = Join-Path $taskVs 'VC/Auxiliary/Build/vcvars64.bat'
$taskToolsVersion = (Get-Content -LiteralPath (Join-Path $taskVs 'VC/Auxiliary/Build/Microsoft.VCToolsVersion.default.txt') -Raw).Trim()
$taskCompiler = Join-Path $taskVs ('VC/Tools/MSVC/' + $taskToolsVersion + '/bin/Hostx64/x64/cl.exe')
$taskDll = Join-Path $taskOutput 'AiDotNet.ArrayProfiler.dll'
$taskCommand = Join-Path $taskOutput 'build.cmd'
$taskSource = Join-Path $PSScriptRoot 'ArrayProfiler.cpp'
$taskText = '@echo off' + "`r`n" + 'call "' + $taskVcvars + '" >nul' + "`r`n" +
    'if errorlevel 1 exit /b %errorlevel%' + "`r`n" +
    'cl /nologo /LD /O2 /std:c++17 /EHsc /W4 /WX /wd4100 /I"' + $taskHeaders + '" "' + $taskSource +
    '" /Fo"' + (Join-Path $taskOutput 'ArrayProfiler.obj') + '" /Fe"' + $taskDll + '" ole32.lib /link /DEF:"' + (Join-Path $PSScriptRoot 'ArrayProfiler.def') + '"' + "`r`n"
[IO.File]::WriteAllText($taskCommand, $taskText)
& $env:ComSpec /d /c $taskCommand 2>&1 | Tee-Object -FilePath (Join-Path $taskOutput 'build.log')
if ($LASTEXITCODE -ne 0) { throw 'Array profiler native compilation failed.' }
$taskReceipt = [ordered]@{
    schema = 'clr-array-profiler-build-v1'; runtime_headers_commit = $taskCommit
    source_sha256 = (Get-FileHash -LiteralPath $taskSource -Algorithm SHA256).Hash.ToLowerInvariant()
    stubs_sha256 = (Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'CallbackStubs.inc') -Algorithm SHA256).Hash.ToLowerInvariant()
    numeric_counters_sha256 = (Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'NumericArrayCounters.h') -Algorithm SHA256).Hash.ToLowerInvariant()
    legacy_snapshot_slots = 7; numeric_abi = 2; numeric_snapshot_slots = 1370
    dll_sha256 = (Get-FileHash -LiteralPath $taskDll -Algorithm SHA256).Hash.ToLowerInvariant()
    compiler_path = $taskCompiler
    compiler_file_version = (Get-Item -LiteralPath $taskCompiler).VersionInfo.FileVersion
    compiler_sha256 = (Get-FileHash -LiteralPath $taskCompiler -Algorithm SHA256).Hash.ToLowerInvariant()
    exports_sha256 = (Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'ArrayProfiler.def') -Algorithm SHA256).Hash.ToLowerInvariant()
    command = $taskText; headers = @{}
}
Get-ChildItem -LiteralPath $taskHeaders -File | ForEach-Object {
    $taskReceipt.headers[$_.Name] = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
}
$taskReceipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $taskOutput 'build-receipt.json') -Encoding utf8

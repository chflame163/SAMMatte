#requires -Version 5.1
<#
.SYNOPSIS
    SAMMatte Windows 环境一键搭建（等价于 Linux 的 setup_linux.sh）。

.DESCRIPTION
    1) 找到可用的 Python（3.10+，推荐 3.12）
    2) 在项目目录内创建 venv（默认 <项目>\venv）
    3) 安装 torch / torchvision（CUDA 轮子，默认 cu128）
    4) 安装 requirements.txt，并把 diffusers 等关键包对齐到已知可用版本
    5) 确认 ffmpeg（tools\ffmpeg\ffmpeg.exe 或系统 PATH，缺失时自动下载 gyan.dev 构建）
    6) 运行环境自检

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File setup_windows.ps1
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File setup_windows.ps1 -TorchVariant cu126 -SkipTorch
#>
[CmdletBinding()]
param(
    [string]$PythonExe = '',
    [string]$VenvDir = '',
    [string]$PipIndex = 'https://pypi.tuna.tsinghua.edu.cn/simple',
    [string]$TorchVariant = 'cu128',
    [string]$TorchVersion = '2.7.1',
    [string]$TorchvisionVersion = '0.22.1',
    [string]$FfmpegVersion = '8.1.1',
    [switch]$SkipTorch,
    [switch]$SkipDeps,
    [switch]$SkipFfmpeg,
    [switch]$SkipCheck,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch { }

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $VenvDir) { $VenvDir = Join-Path $Root 'venv' }
$VenvPy = Join-Path $VenvDir 'Scripts\python.exe'
$Pins = @('diffusers==0.35.1', 'accelerate==1.14.0', 'timm==1.0.27', 'ftfy==6.1.1', 'opencv-python==4.13.0.92')

function Write-Step([string]$Message) {
    Write-Host ''
    Write-Host "== $Message ==" -ForegroundColor Cyan
}

function Invoke-Build {
    param(
        [string]$File,
        [string[]]$Arguments = @(),
        [string]$Label = '',
        [switch]$AllowFail
    )
    $display = if ($Label) { $Label } else { "$File $($Arguments -join ' ')" }
    Write-Host "  > $display"
    if ($DryRun) { return $true }
    & $File @Arguments
    if ($LASTEXITCODE -ne 0) {
        if ($AllowFail) {
            Write-Host "  ! 该步骤失败（退出码 $LASTEXITCODE），继续后续步骤。" -ForegroundColor Yellow
            return $false
        }
        throw "命令失败（退出码 $LASTEXITCODE）：$display"
    }
    return $true
}

function Invoke-Capture {
    param([string]$File, [string[]]$Arguments = @())
    try {
        $out = & $File @Arguments 2>$null
        if ($LASTEXITCODE -ne 0) { return '' }
        return (($out | Out-String).Trim())
    } catch {
        return ''
    }
}

function Get-PythonCandidates {
    $list = @()
    if ($PythonExe) {
        $list += , @{ Exe = $PythonExe; Prefix = @() }
        return $list
    }
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        foreach ($flag in @('-3.12', '-3.11', '-3')) {
            $list += , @{ Exe = $launcher.Source; Prefix = @($flag) }
        }
    }
    foreach ($name in @('python', 'python3')) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) { $list += , @{ Exe = $cmd.Source; Prefix = @() } }
    }
    return $list
}

function Probe-Python($Candidate) {
    # PowerShell passes native args with embedded double quotes stripped, so the python
    # one-liner below only uses single quotes (doubled here for PS escaping).
    $code = 'import sys; print(str(sys.version_info[0]) + ''.'' + str(sys.version_info[1]) + ''|'' + sys.version.split()[0] + ''|'' + sys.executable)'
    $probeArgs = @($Candidate.Prefix) + @('-c', $code)
    $raw = Invoke-Capture -File $Candidate.Exe -Arguments $probeArgs
    if (-not $raw) { return $null }
    $firstLine = ($raw -split "`n")[0].Trim()
    $parts = $firstLine -split '\|'
    if ($parts.Count -lt 3) { return $null }
    $versionPair = $parts[0] -split '\.'
    if ($versionPair.Count -lt 2) { return $null }
    return [pscustomobject]@{
        Exe     = $Candidate.Exe
        Prefix  = $Candidate.Prefix
        Major   = [int]$versionPair[0]
        Minor   = [int]$versionPair[1]
        Version = $parts[1]
        Path    = $parts[2]
    }
}

Write-Host 'SAMMatte Windows 环境搭建' -ForegroundColor Green
Write-Host "项目目录 : $Root"
Write-Host "虚拟环境 : $VenvDir"
Write-Host "PyPI 索引: $PipIndex"
Write-Host "torch    : $TorchVersion / torchvision $TorchvisionVersion / $TorchVariant"
if ($DryRun) { Write-Host '[DryRun] 只打印将要执行的命令，不做任何修改。' -ForegroundColor Yellow }

Write-Step '1/6 查找 Python'
$python = $null
foreach ($candidate in Get-PythonCandidates) {
    $probed = Probe-Python $candidate
    if (-not $probed) {
        Write-Host "  跳过（不可用）: $($candidate.Exe)"
        continue
    }
    if ($probed.Major -eq 3 -and $probed.Minor -ge 10) {
        if ($probed.Major -eq 3 -and $probed.Minor -gt 13) {
            Write-Host "  跳过（版本过新）: $($probed.Path) $($probed.Version)"
            continue
        }
        $python = $probed
        break
    }
    Write-Host "  跳过（需要 3.10+）: $($probed.Path) $($probed.Version)"
}
if (-not $python) {
    Write-Host ''
    Write-Host '[ERROR] 没有找到可用的 Python 3.10+。' -ForegroundColor Red
    Write-Host '        安装方式（任选其一）：'
    Write-Host '          1) https://www.python.org/downloads/ 安装 Python 3.12，安装时勾选 "Add python.exe to PATH"'
    Write-Host '             （注意取消 "Use py launcher" 之外的默认项无妨，但不要把 python 指向 Microsoft Store 别名）'
    Write-Host '          2) conda create -n sammatte python=3.12 && conda activate sammatte'
    Write-Host '             然后用 -PythonExe 指定该环境的 python.exe'
    Write-Host '        也可以直接复用已配置好的环境：powershell -File setup_windows.ps1 -PythonExe "<...>\python.exe" -SkipTorch'
    exit 1
}
Write-Host "  使用 Python : $($python.Version)  $($python.Path)"
if ($python.Minor -ne 12) {
    Write-Host '  提示：项目推荐 Python 3.12；当前不是 3.12，若出现依赖编译问题请改用 3.12。' -ForegroundColor Yellow
}

Write-Step '2/6 创建 venv'
if (Test-Path $VenvPy) {
    Write-Host "  已存在，跳过创建: $VenvPy"
} else {
    $venvArgs = @($python.Prefix) + @('-m', 'venv', $VenvDir)
    if (-not (Invoke-Build -File $python.Exe -Arguments $venvArgs -AllowFail)) {
        throw "创建 venv 失败。若使用 Microsoft Store 版 Python，请改用官网安装包或 conda。"
    }
}

if (-not $DryRun) {
    if (-not (Test-Path $VenvPy)) {
        throw "venv 创建后未找到 $VenvPy"
    }
}

Write-Step '3/6 确保 pip 可用'
$pipCheck = Invoke-Capture -File $VenvPy -Arguments @('-m', 'pip', '--version')
if ($pipCheck) {
    Write-Host "  $pipCheck"
} else {
    if (-not (Invoke-Build -File $VenvPy -Arguments @('-m', 'ensurepip', '--upgrade') -AllowFail)) {
        $getPip = Join-Path $env:TEMP 'sammatte-get-pip.py'
        Write-Host '  ensurepip 不可用，改用 get-pip.py'
        Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile $getPip -UseBasicParsing
        Invoke-Build -File $VenvPy -Arguments @($getPip, '--index-url', $PipIndex) | Out-Null
    }
}
Invoke-Build -File $VenvPy -Arguments @('-m', 'pip', 'install', '-U', 'pip', 'setuptools', 'wheel', '--index-url', $PipIndex) | Out-Null

Write-Step '4/6 安装 PyTorch 与项目依赖'
if ($SkipTorch) {
    Write-Host '  已跳过 torch/torchvision 安装（-SkipTorch）'
} else {
    $currentTorch = Invoke-Capture -File $VenvPy -Arguments @('-c', 'import torch;print(torch.__version__)')
    if ($currentTorch) {
        Write-Host "  检测到已安装 torch $currentTorch，跳过（如需强制安装请先删除 venv 或去掉 -SkipTorch 前先卸载）。" -ForegroundColor Yellow
    } else {
        Invoke-Build -File $VenvPy -Arguments @(
            '-m', 'pip', 'install', "torch==$TorchVersion", "torchvision==$TorchvisionVersion",
            '--index-url', "https://download.pytorch.org/whl/$TorchVariant"
        ) | Out-Null
    }
}

if ($SkipDeps) {
    Write-Host '  已跳过 requirements 安装（-SkipDeps）'
} else {
    Invoke-Build -File $VenvPy -Arguments @('-m', 'pip', 'install', '-r', (Join-Path $Root 'requirements.txt'), '--index-url', $PipIndex) | Out-Null
    Write-Host '  对齐关键包版本（VideoMaMa 依赖 diffusers 内部符号，版本必须固定）'
    $pinArgs = @('-m', 'pip', 'install', '--index-url', $PipIndex) + $Pins
    Invoke-Build -File $VenvPy -Arguments $pinArgs -AllowFail | Out-Null
}

Write-Step '5/6 ffmpeg'
$bundledFfmpeg = Join-Path $Root 'tools\ffmpeg\ffmpeg.exe'
$systemFfmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if (Test-Path $bundledFfmpeg) {
    $ver = Invoke-Capture -File $bundledFfmpeg -Arguments @('-version')
    Write-Host "  已存在: $bundledFfmpeg"
    if ($ver) { Write-Host ("  " + (($ver -split "`n")[0])) }
} elseif ($systemFfmpeg) {
    Write-Host "  使用系统 PATH 中的 ffmpeg: $($systemFfmpeg.Source)"
} elseif ($SkipFfmpeg) {
    Write-Host '  已跳过 ffmpeg 下载（-SkipFfmpeg）。预览/导出的 H.264 重编码会失败。' -ForegroundColor Yellow
} else {
    $zipUrl = "https://github.com/GyanD/codexffmpeg/releases/download/$FfmpegVersion/ffmpeg-$FfmpegVersion-essentials_build.zip"
    $zipPath = Join-Path $env:TEMP "sammatte-ffmpeg-$FfmpegVersion.zip"
    $extractDir = Join-Path $env:TEMP "sammatte-ffmpeg-$FfmpegVersion"
    Write-Host "  下载 ffmpeg（gyan.dev essentials 构建，与项目自带版本一致）"
    if ($DryRun) {
        Write-Host "  [DryRun] $zipUrl -> $bundledFfmpeg"
    } else {
        try {
            Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath -UseBasicParsing
        } catch {
            Write-Host '  Invoke-WebRequest 失败，改用 curl.exe' -ForegroundColor Yellow
            Invoke-Build -File 'curl.exe' -Arguments @('-fL', '--retry', '3', '-o', $zipPath, $zipUrl) | Out-Null
        }
        if (Test-Path $extractDir) { Remove-Item -Recurse -Force $extractDir }
        Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force
        $binDir = Get-ChildItem -Path $extractDir -Recurse -Directory -Filter 'bin' | Select-Object -First 1
        if (-not $binDir) { throw 'ffmpeg 压缩包结构异常，未找到 bin 目录。' }
        New-Item -ItemType Directory -Force -Path (Split-Path $bundledFfmpeg) | Out-Null
        foreach ($name in @('ffmpeg.exe', 'ffprobe.exe', 'ffplay.exe')) {
            $src = Join-Path $binDir.FullName $name
            if (Test-Path $src) { Copy-Item $src (Join-Path (Split-Path $bundledFfmpeg) $name) -Force }
        }
        Remove-Item -Force $zipPath -ErrorAction SilentlyContinue
        Remove-Item -Recurse -Force $extractDir -ErrorAction SilentlyContinue
        Write-Host "  已安装到: $bundledFfmpeg"
    }
}

Write-Step '6/6 环境自检'
if ($SkipCheck -or $DryRun) {
    Write-Host '  已跳过（-SkipCheck / -DryRun）'
} else {
    $checkScript = @'
import os
import sys

print("python       ", sys.version.split()[0])
print("python_exe   ", sys.executable)
import numpy
import cv2
import torch
import torchvision
import transformers
import diffusers
import timm
import accelerate
import psutil
print("torch        ", torch.__version__, "cuda_runtime", torch.version.cuda, "cuda_available", torch.cuda.is_available())
print("torchvision  ", torchvision.__version__)
print("numpy        ", numpy.__version__)
print("cv2          ", cv2.__version__)
print("transformers ", transformers.__version__)
print("diffusers    ", diffusers.__version__)
print("timm         ", timm.__version__)
print("accelerate   ", accelerate.__version__)
try:
    import sam3.model_builder  # noqa: F401
    print("sam3_import  OK")
except Exception as exc:
    print("sam3_import  FAILED:", type(exc).__name__, exc)
sys.path.insert(0, os.path.join(os.environ["SAMMATTE_ROOT"], "models", "VideoMaMa"))
try:
    import pipeline_svd_mask  # noqa: F401
    print("videomama_pipeline_import OK")
except Exception as exc:
    print("videomama_pipeline_import FAILED:", type(exc).__name__, exc)
from sam31_webapp.service import describe_ffmpeg
print(describe_ffmpeg())
'@
    $checkPath = Join-Path $env:TEMP 'sammatte_selfcheck.py'
    Set-Content -Path $checkPath -Value $checkScript -Encoding UTF8
    $env:PYTHONPATH = "$Root\app;$Root\models\sam3"
    $env:SAMMATTE_ROOT = $Root
    Invoke-Build -File $VenvPy -Arguments @($checkPath) -AllowFail | Out-Null
}

Write-Host ''
Write-Host '完成。后续使用：' -ForegroundColor Green
Write-Host "  前台启动 : run_SAMMatte.bat          （会自动优先使用 $VenvPy）"
Write-Host '  环境自检 : check_runtime.bat'
Write-Host '  局域网访问（其他电脑可连）:'
Write-Host '    set SAM31_HOST=0.0.0.0'
Write-Host '    set SAM31_PORT=8765'
Write-Host '    run_SAMMatte.bat'

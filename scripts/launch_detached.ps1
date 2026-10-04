# Start a long job detached from the terminal or session that launches it, so that closing that session does not kill it
# (a grid killed this way lost 19 GPU-hours on 2026-09-30). The job runs from the repository root with output in logs\<name>.log.
# Stop a grid gracefully with a STOP file in its experiment directory (flybrain/harness/runner.py), never by killing the process.
# Usage: pwsh scripts/launch_detached.ps1 -Name s1b_grid -Command "experiments\p3_s1b_sparse\run.py --compile --device cuda --data_workers 5"
#        optional -Env "HF_HUB_OFFLINE=1" (semicolon-separated NAME=value pairs)
param([Parameter(Mandatory)][string]$Name, [Parameter(Mandatory)][string]$Command, [string]$Env = "")

$root = Split-Path -Parent $PSScriptRoot
$python = (Get-Command python).Source
if ($python -like "*shims*") { $python = (& python -c "import sys; print(sys.executable)").Trim() }   # pyenv shims are batch files: use the real interpreter
New-Item -ItemType Directory -Force (Join-Path $root "logs") | Out-Null
$log = Join-Path $root "logs\$Name.log"
$sets = ($Env -split ";" | Where-Object { $_ } | ForEach-Object { "set `"$_`" && " }) -join ""
$path = $env:PATH                                                                    # the launching shell's PATH (compilers, CUDA) is passed on explicitly
$cmd = "cmd.exe /c `"cd /d `"$root`" && set `"PATH=$path`" && $sets`"$python`" $Command > `"$log`" 2>&1`""
# Hidden: without this the job opens a visible terminal window, and closing that window kills the job ("window-CLOSE event";
# it killed both jobs on 2026-10-01). A hidden job is seen in Task Manager as python.exe, in nvidia-smi, and in its log.
$hidden = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = $root; ProcessStartupInformation = $hidden }
if ($r.ReturnValue -ne 0) { throw "launch failed with code $($r.ReturnValue)" }
"started '$Name' (cmd pid $($r.ProcessId)); log: $log"

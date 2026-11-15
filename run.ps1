# run.ps1 — воспроизводит метрики из otvet.txt
$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "=== Support Triage: воспроизведение метрик ===" -ForegroundColor Cyan

$csvFiles = @(
    "rusentiment_preselected_posts.csv",
    "rusentiment_test.csv",
    "support_tickets_v2.csv"
)

# 1. Ищем собранный exe рекурсивно от папки скрипта
$exe = Get-ChildItem -Path $scriptDir -Filter "WorkSHVEC.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $exe) {
    Write-Host "ОШИБКА: WorkSHVEC.exe не найден." -ForegroundColor Red
    Write-Host "  Соберите проект в Visual Studio (Build -> Build Solution, Ctrl+Shift+B)." -ForegroundColor Red
    exit 1
}
$exeDir = $exe.DirectoryName
Write-Host "Найден exe: $($exe.FullName)" -ForegroundColor Green

# 2. Для каждого CSV: если уже в папке exe — ок, иначе ищем рекурсивно и копируем
foreach ($name in $csvFiles) {
    $inExeDir = Join-Path $exeDir $name
    if (Test-Path $inExeDir) {
        Write-Host "OK: $name уже в папке exe" -ForegroundColor Green
        continue
    }
    $found = Get-ChildItem -Path $scriptDir -Filter $name -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($found) {
        Copy-Item $found.FullName -Destination $exeDir -Force
        Write-Host "OK: $name скопирован из $($found.DirectoryName)" -ForegroundColor Green
    } else {
        Write-Host "ВНИМАНИЕ: $name не найден нигде в проекте." -ForegroundColor Yellow
        Write-Host "  Положите его в любую папку проекта и запустите снова." -ForegroundColor Yellow
    }
}

# 3. Запускаем и сохраняем вывод в otvet.txt
$outPath = Join-Path $scriptDir "otvet.txt"
Write-Host "Запускаю и сохраняю в otvet.txt..." -ForegroundColor Yellow
Push-Location $exeDir
& $exe.FullName *> $outPath
Pop-Location

Write-Host "Готово. Метрики в: $outPath" -ForegroundColor Green
Write-Host "Открыть: notepad otvet.txt" -ForegroundColor Cyan
# Full uncertainty workflow for one country:
#   preprocess -> validate -> LHS experiments -> Sobol experiments -> analyses
# Usage:  powershell -File run_country.ps1 -Country DNK [-Workers 6]

param(
    [Parameter(Mandatory = $true)][string]$Country,
    [int]$Workers = 6
)

$py = "$env:USERPROFILE\.venvs\miraca_uq\Scripts\python.exe"
Set-Location $PSScriptRoot

$steps = @(
    @("-m", "src.preprocess", "--country", $Country),
    @("-m", "src.validate", "--country", $Country),
    @("-m", "src.run_experiments", "--country", $Country, "--n", "3000", "--workers", "$Workers"),
    @("-m", "src.run_experiments", "--country", $Country, "--sampler", "sobol", "--n", "512", "--workers", "$Workers"),
    @("-m", "src.analyze", "--country", $Country),
    @("-m", "src.analyze_sobol", "--country", $Country)
)

foreach ($s in $steps) {
    Write-Output ""
    Write-Output "=== $Country :: python $($s -join ' ') ==="
    & $py @s
    if ($LASTEXITCODE -ne 0) {
        Write-Output "STEP FAILED (exit $LASTEXITCODE) - aborting $Country"
        exit 1
    }
}
Write-Output ""
Write-Output "=== $Country workflow complete ==="

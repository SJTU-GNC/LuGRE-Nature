$ErrorActionPreference = 'Stop'
$figSource = 'D:/Reference/Paper/2026/Nature_LuGRE/Fig1_R3.pptx'
$figProject = (Resolve-Path -LiteralPath 'H:/LUGRE').Path
$figBuild = Join-Path $figProject 'figures_R3/Fig1_R3_export_20261013'
$figStamp = (Get-Item -LiteralPath $figSource).LastWriteTime.ToString('yyyyMMdd_HHmmss')
$figSnapshot = Join-Path $figBuild ('Fig1_R3_source_' + $figStamp + '.pptx')
$figPng = Join-Path $figBuild ('Fig1_R3_' + $figStamp + '_600dpi.png')
$figSourceHash = (Get-FileHash -LiteralPath $figSource -Algorithm SHA256).Hash
if (-not (Test-Path -LiteralPath $figSnapshot)) {
    Copy-Item -LiteralPath $figSource -Destination $figSnapshot
}
if ((Get-FileHash -LiteralPath $figSnapshot -Algorithm SHA256).Hash -ne $figSourceHash) {
    throw 'Source changed since the snapshot. Preserve both versions and review before export.'
}
$figApp = $null
$figDeck = $null
$figFontEdits = [System.Collections.Generic.List[object]]::new()
$figLineEdits = [System.Collections.Generic.List[object]]::new()
$figSpacingEdits = [System.Collections.Generic.List[object]]::new()
function Set-FigureExportTypography($figShape, [double]$figScale) {
    if ($figShape.Type -eq 6) {
        for ($figItem = 1; $figItem -le $figShape.GroupItems.Count; $figItem++) {
            Set-FigureExportTypography $figShape.GroupItems.Item($figItem) $figScale
        }
    }
    if ($figShape.HasTextFrame -eq -1 -and $figShape.TextFrame.HasText -eq -1) {
        $figText = $figShape.TextFrame.TextRange
        $figOriginalString = $figText.Text
        $figText.Font.Name = 'Arial'
        $figIsPanel = $figOriginalString.Trim() -cmatch '^[abcdef]$'
        for ($figRunIndex = 1; $figRunIndex -le $figText.Runs().Count; $figRunIndex++) {
            $figRun = $figText.Runs($figRunIndex, 1)
            $figOriginalSize = [double]$figRun.Font.Size
            $figFinalSize = if ($figIsPanel) { 8.0 } else { [Math]::Max(5.0, [Math]::Min(7.0, $figOriginalSize * $figScale)) }
            $figRun.Font.Size = $figFinalSize / $figScale
        }
        if ($figIsPanel) { $figText.Font.Bold = -1; $figText.Font.Italic = 0 }
        if ($figText.Text -cne $figOriginalString) { throw 'Unexpected text-content change.' }
        $figFontEdits.Add([ordered]@{shape=$figShape.Name; text=$figOriginalString; font='Arial'; panel_label=$figIsPanel})
    }
    if ($figShape.Line.Visible -eq -1 -and $figShape.Line.Weight -gt 0) {
        $figOriginalWeight = [double]$figShape.Line.Weight
        if ($figOriginalWeight * $figScale -lt 0.25) {
            $figShape.Line.Weight = 0.25 / $figScale
            $figLineEdits.Add([ordered]@{shape=$figShape.Name; original_pt=$figOriginalWeight; export_final_pt=0.25})
        }
    }
    if ($figShape.HasChart -eq -1) {
        $figChart = $figShape.Chart
        $figChart.ChartArea.Font.Name = 'Arial'
        $figChart.ChartArea.Font.Size = 6.63 / $figScale
        for ($figAxisType = 1; $figAxisType -le 2; $figAxisType++) {
            $figAxis = $figChart.Axes($figAxisType)
            $figAxis.TickLabels.Font.Name = 'Arial'
            $figAxis.TickLabels.Font.Size = 6.63 / $figScale
            if ($figAxis.HasTitle) {
                $figAxis.AxisTitle.Font.Name = 'Arial'
                $figAxis.AxisTitle.Font.Size = 6.63 / $figScale
            }
            if ($figAxis.Format.Line.Weight * $figScale -lt 0.25) {
                $figAxis.Format.Line.Weight = 0.25 / $figScale
            }
        }
    }
}
try {
    $figApp = New-Object -ComObject PowerPoint.Application
    # ReadOnly=True; Untitled=False; WithWindow=False. Do not close the user's application.
    $figDeck = $figApp.Presentations.Open($figSnapshot, -1, 0, 0)
    if ($figDeck.Slides.Count -ne 1) { throw 'Expected exactly one Figure 1 slide.' }
    $figWidth = [double]$figDeck.PageSetup.SlideWidth
    $figHeight = [double]$figDeck.PageSetup.SlideHeight
    $figWidthMm = 183.0
    $figDpi = 600
    $figScale = ($figWidthMm / 25.4 * 72) / $figWidth
    for ($figShapeIndex = 1; $figShapeIndex -le $figDeck.Slides.Item(1).Shapes.Count; $figShapeIndex++) {
        Set-FigureExportTypography $figDeck.Slides.Item(1).Shapes.Item($figShapeIndex) $figScale
    }
    $figPhaseLabel = $null
    $figLeoLabel = $null
    foreach ($figShape in $figDeck.Slides.Item(1).Shapes) {
        if ($figShape.HasTextFrame -eq -1 -and $figShape.TextFrame.HasText -eq -1) {
            $figLabel = $figShape.TextFrame.TextRange.Text.Trim()
            if ($figShape.Top -gt $figHeight * 0.6 -and $figShape.Left -lt $figWidth / 2) {
                if ($figLabel -eq 'Phase screen') { $figPhaseLabel = $figShape }
                if ($figLabel -eq 'LEO-RO') { $figLeoLabel = $figShape }
            }
            if ($figLabel -like 'L1/E1 limit*' -or $figLabel -like 'L5/E5 limit*') {
                $figShift = 0.4 * 72 / 25.4 / $figScale
                $figShape.Top -= $figShift
                $figSpacingEdits.Add([ordered]@{text=$figLabel; final_vertical_shift_mm=-0.4})
            }
        }
    }
    if ($null -eq $figPhaseLabel -or $null -eq $figLeoLabel) { throw 'Expected panel e labels are missing.' }
    $figGap = 0.8 * 72 / 25.4 / $figScale
    $figPhaseRight = $figPhaseLabel.TextFrame.TextRange.BoundLeft + $figPhaseLabel.TextFrame.TextRange.BoundWidth
    $figLabelShift = [Math]::Max(0, $figPhaseRight + $figGap - $figLeoLabel.TextFrame.TextRange.BoundLeft)
    $figLeoLabel.Left += $figLabelShift
    $figSpacingEdits.Add([ordered]@{text='LEO-RO (panel e)'; final_horizontal_shift_mm=$figLabelShift*$figScale*25.4/72})
    $figWidthPx = [int][Math]::Round($figWidthMm / 25.4 * $figDpi)
    $figHeightPx = [int][Math]::Round($figWidthPx * $figHeight / $figWidth)
    $figDeck.Slides.Item(1).Export($figPng, 'PNG', $figWidthPx, $figHeightPx)
    $figReport = [ordered]@{
        revision = 'R3'
        intended_figure = 'Main Figure 1'
        source = $figSource
        source_sha256 = $figSourceHash
        source_saved_time = (Get-Item -LiteralPath $figSource).LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss')
        snapshot = $figSnapshot
        renderer = 'Microsoft PowerPoint native Slide.Export'
        powerpoint_version = $figApp.Version
        slides = $figDeck.Slides.Count
        source_slide_size_pt = @($figWidth, $figHeight)
        intended_width_mm = $figWidthMm
        intended_height_mm = $figWidthMm * $figHeight / $figWidth
        intended_dpi = $figDpi
        pixels = @($figWidthPx, $figHeightPx)
        png = $figPng
        source_edit = $false
        raster_resampling = $false
        export_scope = 'Latest complete existing slide; in-memory publication typography only; original file is not modified'
        typeface = 'Arial'
        final_text_size_pt = @(5, 7)
        final_panel_label_size_pt = 8
        minimum_native_line_weight_pt = 0.25
        native_text_shapes = $figFontEdits.Count
        strengthened_native_strokes = $figLineEdits.Count
        font_changes = $figFontEdits
        line_changes = $figLineEdits
        spacing_changes = $figSpacingEdits
    }
    $figReport | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $figBuild ('Fig1_R3_export_record_' + $figStamp + '.json')) -Encoding UTF8
    [PSCustomObject]$figReport | Select-Object revision,source_saved_time,source_sha256,pixels,png,native_text_shapes,strengthened_native_strokes | ConvertTo-Json -Depth 4
}
finally {
    if ($null -ne $figDeck) {
        $figDeck.Saved = -1
        $figDeck.Close()
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($figDeck)
    }
    if ($null -ne $figApp) {
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($figApp)
    }
}
if ((Get-FileHash -LiteralPath $figSource -Algorithm SHA256).Hash -ne $figSourceHash) {
    throw 'Source content changed during export; inspect the newer saved source before delivering.'
}

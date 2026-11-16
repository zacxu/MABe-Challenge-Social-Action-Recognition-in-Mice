param([switch]$Render)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$sourcePath = Join-Path $projectRoot 'SC4000 Second Draft downloaded.docx'
$auditRoot = Join-Path $PSScriptRoot 'audit'
$mediaRoot = Join-Path $auditRoot 'media'
New-Item -ItemType Directory -Path $mediaRoot -Force | Out-Null
$hashBefore = (Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash
$archive = [IO.Compression.ZipFile]::OpenRead($sourcePath)
$reader = [IO.StreamReader]::new($archive.GetEntry('word/document.xml').Open())
[xml]$documentXml = $reader.ReadToEnd()
$reader.Dispose()
$ns = [Xml.XmlNamespaceManager]::new($documentXml.NameTable)
$ns.AddNamespace('w', 'http://schemas.openxmlformats.org/wordprocessingml/2006/main')
$ns.AddNamespace('m', 'http://schemas.openxmlformats.org/officeDocument/2006/math')
$ns.AddNamespace('a', 'http://schemas.openxmlformats.org/drawingml/2006/main')
$records = [Collections.Generic.List[object]]::new()
$lines = [Collections.Generic.List[string]]::new()
$blockIndex = 0
$paragraphIndex = 0
$tableIndex = 0
foreach ($block in $documentXml.SelectNodes('//w:body/*', $ns)) {
    if ($block.LocalName -eq 'p') {
        $text = ($block.SelectNodes('.//w:t | .//m:t', $ns) | ForEach-Object { $_.InnerText }) -join ''
        $styleNode = $block.SelectSingleNode('./w:pPr/w:pStyle', $ns)
        $style = if ($styleNode) { $styleNode.GetAttribute('val', $ns.LookupNamespace('w')) } else { '' }
        $records.Add([pscustomobject]@{ kind='paragraph'; block=$blockIndex; paragraph=$paragraphIndex; style=$style; text=$text })
        if ($text.Trim()) { $lines.Add(('[P{0:D3}; B{1:D3}; {2}] {3}' -f $paragraphIndex, $blockIndex, $style, $text)) }
        $paragraphIndex++
    } elseif ($block.LocalName -eq 'tbl') {
        $tableIndex++
        $rows = @($block.SelectNodes('./w:tr', $ns) | ForEach-Object {
            $cells = @($_.SelectNodes('./w:tc', $ns) | ForEach-Object {
                ($_.SelectNodes('.//w:t | .//m:t', $ns) | ForEach-Object { $_.InnerText }) -join ''
            })
            ,$cells
        })
        $records.Add([pscustomobject]@{ kind='table'; block=$blockIndex; table=$tableIndex; rows=$rows })
        $lines.Add(('[TABLE {0}; B{1:D3}]' -f $tableIndex, $blockIndex))
        foreach ($row in $rows) { $lines.Add('| ' + ($row -join ' | ') + ' |') }
    }
    $blockIndex++
}
$media = @()
foreach ($entry in $archive.Entries) {
    if ($entry.FullName -like 'word/media/*' -and $entry.Name) {
        $destination = Join-Path $mediaRoot $entry.Name
        $stream = [IO.File]::Create($destination)
        $inputStream = $entry.Open()
        try { $inputStream.CopyTo($stream) } finally { $inputStream.Dispose(); $stream.Dispose() }
        $media += [pscustomobject]@{ source=$entry.FullName; path=$destination; bytes=$entry.Length }
    }
}
$archive.Dispose()
[IO.File]::WriteAllText((Join-Path $auditRoot 'draft_blocks.json'), (ConvertTo-Json -InputObject $records.ToArray() -Depth 12), [Text.UTF8Encoding]::new($false))
[IO.File]::WriteAllLines((Join-Path $auditRoot 'draft_text.txt'), $lines, [Text.UTF8Encoding]::new($false))
$renderEvidence = $null
if ($Render) {
    # Windows fallback: native Word read-only export. Never save the source.
    $word = $null
    $document = $null
    $closeApplication = $false
    try {
        $word = New-Object -ComObject Word.Application
        $closeApplication = $word.Documents.Count -eq 0
        if ($closeApplication) { $word.Visible = $false; $word.DisplayAlerts = 0; $word.AutomationSecurity = 3 }
        $document = $word.Documents.Open($sourcePath, $false, $true, $false)
        $document.Repaginate()
        $pageCount = $document.ComputeStatistics(2)
        $pdfPath = Join-Path $auditRoot 'draft_word_render.pdf'
        $document.ExportAsFixedFormat($pdfPath, 17)
        $renderEvidence = [pscustomobject]@{ method='Microsoft Word read-only ExportAsFixedFormat'; pages=$pageCount; pdf=$pdfPath; word_version=$word.Version }
    } finally {
        if ($document) { $document.Close(0); [void][Runtime.InteropServices.Marshal]::ReleaseComObject($document) }
        if ($word) {
            if ($closeApplication) { $word.Quit() }
            [void][Runtime.InteropServices.Marshal]::ReleaseComObject($word)
        }
    }
}
$hashAfter = (Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash
if ($hashBefore -ne $hashAfter) { throw 'Source draft changed during read-only inspection.' }
$evidence = [pscustomobject]@{
    source=$sourcePath; source_sha256=$hashBefore; source_unchanged=($hashBefore -eq $hashAfter)
    paragraph_count=$paragraphIndex; table_count=$tableIndex; media=$media; render=$renderEvidence
    extraction_note='Body paragraph indices count all direct paragraphs; drawings may expose duplicated fallback text. Confirm visual defects from rendering, not raw extraction alone.'
}
[IO.File]::WriteAllText((Join-Path $auditRoot 'inspection_manifest.json'), (ConvertTo-Json $evidence -Depth 8), [Text.UTF8Encoding]::new($false))
ConvertTo-Json $evidence -Depth 8

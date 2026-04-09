# download_model.ps1

$model_dir = "local_models"
$model_name = "Gemma-3-TAIDE-12b-Chat-Q4_K_M.gguf"
$url = "https://huggingface.co/nctu6/Gemma-3-TAIDE-12b-Chat-GGUF/resolve/main/$model_name"

# Fix Join-Path for older PowerShell versions
$target_dir = Join-Path $PWD $model_dir
$target_path = Join-Path $target_dir $model_name

if (-not (Test-Path $model_dir)) {
    New-Item -ItemType Directory -Path $model_dir
    Write-Host "Created directory: $model_dir"
}

Write-Host "Starting download of $model_name (~4.9GB)..."
Write-Host "This may take some time depending on your internet speed."

try {
    # Using bitsadmin for a more robust download
    Write-Host "Starting BITS job (recommended for large files)..."
    # Ensure -Description is not causing issues, and use -Priority foreground for speed
    Start-BitsTransfer -Source $url -Destination $target_path -Priority Foreground -ErrorAction Stop
    Write-Host "Download complete! Model saved to: $target_path"
}
catch {
    Write-Host "BITS transfer failed (Error: $_), trying direct download with WebClient..."
    try {
        # Using .NET WebClient which is better for very large files than Invoke-WebRequest
        $client = New-Object System.Net.WebClient
        $client.DownloadFile($url, $target_path)
        Write-Host "Download complete (via WebClient)!"
    }
    catch {
        Write-Host "Direct download failed, trying Invoke-WebRequest with basic parsing..."
        try {
            Invoke-WebRequest -Uri $url -OutFile $target_path -UseBasicParsing -ErrorAction Stop
        }
        catch {
            Write-Host "All download methods failed: $_"
            Write-Host "Please manually download the model from: $url"
            Write-Host "And save it to: $target_path"
        }
    }
}

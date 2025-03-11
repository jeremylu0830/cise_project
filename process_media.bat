@echo off

:: Move to the directory where this script is located
cd /d "%~dp0"

:: Enter the uploads folder
cd uploads

:: Enable delayed variable expansion
setlocal enabledelayedexpansion

:: Initialize a counter
set count=1

:: Process all files in the current directory (*.*).
:: If you only want to process specific extensions (e.g., *.jpg, *.png, *.mp4),
:: just replace (*.*) with (*.jpg *.png *.mp4), etc.
for %%f in (*.*) do (
    :: Extract the file extension
    set "ext=%%~xf"
    :: Rename the file using the counter + extension
    ren "%%f" "!count!!ext!"
    :: Increment the counter by 1
    set /a count+=1
)

echo Batch script execution complete. Files have been renamed.
exit /b 0
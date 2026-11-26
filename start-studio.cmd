Attempting to perform the InitializeDefaultDrives operation on the 'FileSystem' provider failed.
@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-studio.ps1" %*
if errorlevel 1 pause


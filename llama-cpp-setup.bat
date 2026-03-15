@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

set "LLAMA_REPO_URL=https://github.com/ggml-org/llama.cpp.git"
set "LLAMA_DIR=%SCRIPT_DIR%\llama.cpp"
set "BUILD_DIR=%SCRIPT_DIR%\build-vulkan"
set "DEPS_DIR=%SCRIPT_DIR%\_deps"
set "DOWNLOAD_DIR=%SCRIPT_DIR%\_downloads"
set "LOCAL_VULKAN_SDK=%DEPS_DIR%\VulkanSDK"
set "VULKAN_INSTALLER=%DOWNLOAD_DIR%\vulkan_sdk.exe"
set "VULKAN_LATEST_URL=https://sdk.lunarg.com/sdk/download/latest/windows/vulkan_sdk.exe"
set "CLEAN_BUILD=0"

:parse_args
if /I "%~1"=="--help" goto :usage
if /I "%~1"=="-h" goto :usage
if /I "%~1"=="--clean" (
    set "CLEAN_BUILD=1"
    shift
    goto :parse_args
)
if not "%~1"=="" (
    echo [ERROR] Unsupported argument: %~1
    echo.
    goto :usage
)

echo ========================================
echo   llama.cpp Vulkan Setup Script
echo ========================================
echo.
echo [INFO] Working directory: %SCRIPT_DIR%
echo [INFO] Target repo: %LLAMA_REPO_URL%
echo [INFO] Target source dir: %LLAMA_DIR%
if "%CLEAN_BUILD%"=="1" (
    echo [INFO] Clean build mode: enabled
)
echo.

call :require_command git
if errorlevel 1 exit /b 1

call :require_command cmake
if errorlevel 1 exit /b 1

call :require_vs2022
if errorlevel 1 exit /b 1

call :sync_llama_repo
if errorlevel 1 exit /b 1

if "%CLEAN_BUILD%"=="1" (
    echo.
    echo [INFO] Cleaning build directory per --clean option...
    if exist "%BUILD_DIR%" (
        call :run rmdir /s /q "%BUILD_DIR%"
        if errorlevel 1 (
            echo [ERROR] Failed to remove build directory: %BUILD_DIR%
            exit /b 1
        )
    )
    echo [OK] Build directory cleaned.
)

call :ensure_vulkan_sdk
if errorlevel 1 exit /b 1

call :configure_vulkan_env
if errorlevel 1 exit /b 1

call :build_llama
if errorlevel 1 exit /b 1

call :run_vulkan_smoke_test
if errorlevel 1 exit /b 1

echo.
echo [SUCCESS] llama.cpp Vulkan setup completed successfully.
echo [SUCCESS] Source: %LLAMA_DIR%
echo [SUCCESS] Build:  %BUILD_DIR%
exit /b 0

:usage
echo Usage: %~nx0 [OPTIONS]
echo.
echo What it does:
echo   1. Clone or update the latest official llama.cpp source into:
echo      %LLAMA_DIR%
echo   2. Detect a usable Vulkan SDK and install one locally if needed.
echo   3. Build llama.cpp with the Vulkan backend on Windows 11 using VS 2022.
echo   4. Run a basic Vulkan backend smoke test to verify the build.
echo.
echo Options:
echo   --clean    Remove existing build directory before building (force rebuild)
echo   --help     Show this help message
echo.
echo Notes:
echo   - This script assumes git, cmake, and Visual Studio 2022 C++ tools are already installed.
echo   - Vulkan SDK is installed locally under:
echo      %LOCAL_VULKAN_SDK%
echo   - GPU drivers are not installed by this script. If no Vulkan device is detected,
echo     install or update your GPU vendor's Windows driver and rerun the script.
exit /b 0

:require_command
where "%~1" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Required tool not found in PATH: %~1
    echo [ERROR] Please install it first and rerun this script.
    exit /b 1
)
echo [OK] Found %~1
exit /b 0

:require_vs2022
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VS2022_PATH="

if not exist "%VSWHERE%" (
    echo [ERROR] vswhere.exe was not found.
    echo [ERROR] Please install Visual Studio 2022 with Desktop development with C++.
    exit /b 1
)

for /f "usebackq delims=" %%I in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "& '%VSWHERE%' -version '[17.0,18.0)' -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath"`) do (
    if not defined VS2022_PATH set "VS2022_PATH=%%I"
)

if not defined VS2022_PATH (
    echo [ERROR] Visual Studio 2022 C++ build tools were not found.
    echo [ERROR] Please install Visual Studio 2022 with Desktop development with C++.
    exit /b 1
)

echo [OK] Found Visual Studio 2022: %VS2022_PATH%
exit /b 0

:sync_llama_repo
echo.
echo [INFO] Syncing official llama.cpp repository...

if exist "%LLAMA_DIR%\.git" (
    set "CURRENT_REMOTE="
    for /f "usebackq delims=" %%I in (`git -C "%LLAMA_DIR%" remote get-url origin 2^>nul`) do (
        if not defined CURRENT_REMOTE set "CURRENT_REMOTE=%%I"
    )

    if not defined CURRENT_REMOTE (
        echo [ERROR] Existing folder looks like a git repo, but origin remote is missing:
        echo [ERROR]   %LLAMA_DIR%
        exit /b 1
    )

    if /I not "!CURRENT_REMOTE!"=="%LLAMA_REPO_URL%" (
        echo [ERROR] Existing repo at %LLAMA_DIR% does not point to the official llama.cpp remote.
        echo [ERROR] Current origin: !CURRENT_REMOTE!
        echo [ERROR] Expected origin: %LLAMA_REPO_URL%
        exit /b 1
    )

    call :run git -C "%LLAMA_DIR%" fetch origin --prune --tags
    if errorlevel 1 exit /b 1

    set "DEFAULT_BRANCH="
    for /f "usebackq delims=" %%I in (`git -C "%LLAMA_DIR%" symbolic-ref refs/remotes/origin/HEAD 2^>nul`) do (
        if not defined DEFAULT_BRANCH set "DEFAULT_BRANCH=%%I"
    )

    if not defined DEFAULT_BRANCH (
        echo [ERROR] Could not resolve origin/HEAD for %LLAMA_DIR%.
        exit /b 1
    )

    set "DEFAULT_BRANCH=!DEFAULT_BRANCH:refs/remotes/origin/=!"
    echo [INFO] Updating existing clone on branch: !DEFAULT_BRANCH!

    call :run git -C "%LLAMA_DIR%" checkout "!DEFAULT_BRANCH!"
    if errorlevel 1 exit /b 1

    call :run git -C "%LLAMA_DIR%" pull --ff-only origin "!DEFAULT_BRANCH!"
    if errorlevel 1 exit /b 1
    exit /b 0
)

if exist "%LLAMA_DIR%" (
    echo [ERROR] Target path already exists but is not a git repository:
    echo [ERROR]   %LLAMA_DIR%
    echo [ERROR] Move or delete it, then rerun the script.
    exit /b 1
)

call :ensure_dir "%SCRIPT_DIR%"
if errorlevel 1 exit /b 1

call :run git clone --depth 1 "%LLAMA_REPO_URL%" "%LLAMA_DIR%"
if errorlevel 1 exit /b 1

exit /b 0

:ensure_vulkan_sdk
echo.
echo [INFO] Checking Vulkan SDK...

if defined VULKAN_SDK (
    call :validate_vulkan_sdk "%VULKAN_SDK%"
    if not errorlevel 1 (
        set "ACTIVE_VULKAN_SDK=%VULKAN_SDK%"
        echo [OK] Using existing VULKAN_SDK from environment: !ACTIVE_VULKAN_SDK!
        exit /b 0
    )
)

call :validate_vulkan_sdk "%LOCAL_VULKAN_SDK%"
if not errorlevel 1 (
    set "ACTIVE_VULKAN_SDK=%LOCAL_VULKAN_SDK%"
    echo [OK] Using locally cached Vulkan SDK: !ACTIVE_VULKAN_SDK!
    exit /b 0
)

call :ensure_dir "%DOWNLOAD_DIR%"
if errorlevel 1 exit /b 1

call :ensure_dir "%DEPS_DIR%"
if errorlevel 1 exit /b 1

echo [INFO] Downloading latest Vulkan SDK installer...
call :run powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest -UseBasicParsing '%VULKAN_LATEST_URL%' -OutFile '%VULKAN_INSTALLER%'"
if errorlevel 1 (
    echo [ERROR] Failed to download the Vulkan SDK installer.
    exit /b 1
)

echo [INFO] Installing Vulkan SDK locally to %LOCAL_VULKAN_SDK%
call :run "%VULKAN_INSTALLER%" --root "%LOCAL_VULKAN_SDK%" --accept-licenses --default-answer --confirm-command install copy_only=1
if errorlevel 1 (
    echo [ERROR] Vulkan SDK installation failed.
    exit /b 1
)

call :validate_vulkan_sdk "%LOCAL_VULKAN_SDK%"
if errorlevel 1 (
    echo [ERROR] Vulkan SDK appears incomplete after installation: %LOCAL_VULKAN_SDK%
    exit /b 1
)

set "ACTIVE_VULKAN_SDK=%LOCAL_VULKAN_SDK%"
echo [OK] Vulkan SDK installed locally: %ACTIVE_VULKAN_SDK%
exit /b 0

:validate_vulkan_sdk
if "%~1"=="" exit /b 1
if not exist "%~1\Include\vulkan\vulkan.h" exit /b 1
if not exist "%~1\Lib\vulkan-1.lib" exit /b 1
if not exist "%~1\Bin\glslc.exe" exit /b 1
exit /b 0

:configure_vulkan_env
if not defined ACTIVE_VULKAN_SDK (
    echo [ERROR] ACTIVE_VULKAN_SDK is not set.
    exit /b 1
)

set "VULKAN_SDK=%ACTIVE_VULKAN_SDK%"
set "VK_SDK_PATH=%ACTIVE_VULKAN_SDK%"
set "CMAKE_PREFIX_PATH=%ACTIVE_VULKAN_SDK%;%CMAKE_PREFIX_PATH%"
set "PATH=%ACTIVE_VULKAN_SDK%\Bin;%PATH%"

echo [OK] Vulkan SDK environment configured: %ACTIVE_VULKAN_SDK%
exit /b 0

:build_llama
echo.
echo [INFO] Configuring llama.cpp for Vulkan...
call :run cmake -S "%LLAMA_DIR%" -B "%BUILD_DIR%" -G "Visual Studio 17 2022" -A x64 -DGGML_VULKAN=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=OFF
if errorlevel 1 (
    echo [ERROR] CMake configuration failed.
    exit /b 1
)

echo [INFO] Building Release solution...
call :run cmake --build "%BUILD_DIR%" --config Release --parallel
if errorlevel 1 (
    echo [ERROR] Build failed.
    exit /b 1
)

call :find_binary "llama-cli.exe" LLAMA_DEVICE_TOOL
if errorlevel 1 (
    call :find_binary "llama-completion.exe" LLAMA_DEVICE_TOOL
    if errorlevel 1 (
        echo [ERROR] Could not find a llama tool that supports --list-devices.
        echo [ERROR] Expected one of: llama-cli.exe, llama-completion.exe
        exit /b 1
    )
)

call :find_binary "test-backend-ops.exe" TEST_BACKEND_OPS
if errorlevel 1 exit /b 1

echo [OK] Build artifacts located.
echo [OK] Device tool: %LLAMA_DEVICE_TOOL%
echo [OK] test-backend-ops: %TEST_BACKEND_OPS%
exit /b 0

:find_binary
set "%~2="
for %%D in ("%BUILD_DIR%\bin\Release" "%BUILD_DIR%\bin" "%BUILD_DIR%\Release") do (
    if exist "%%~D\%~1" (
        set "%~2=%%~D\%~1"
        exit /b 0
    )
)

echo [ERROR] Could not find %~1 under build output: %BUILD_DIR%
exit /b 1

:run_vulkan_smoke_test
set "DEVICE_FILE=%BUILD_DIR%\llama-devices.txt"
set "VULKAN_DEVICE_TMP=%BUILD_DIR%\vulkan_device.tmp"
set "VULKAN_DEVICE="

echo.
echo [INFO] Enumerating available llama.cpp devices...
call :run "%LLAMA_DEVICE_TOOL%" --list-devices > "%DEVICE_FILE%"
if errorlevel 1 (
    echo [ERROR] Failed to query llama.cpp devices.
    exit /b 1
)

for /f "tokens=1 delims=:" %%A in ('findstr /R /C:"^  Vulkan" "%DEVICE_FILE%"') do (
    for /f "tokens=*" %%B in ("%%A") do (
        if not defined VULKAN_DEVICE set "VULKAN_DEVICE=%%B"
    )
)

if not defined VULKAN_DEVICE (
    echo [ERROR] Build completed, but llama.cpp did not detect any Vulkan device.
    echo [ERROR] The Vulkan SDK does not install GPU drivers.
    echo [ERROR] Install or update your GPU vendor's Vulkan-capable Windows driver, then rerun the script.
    exit /b 1
)

echo [OK] Detected Vulkan backend device: %VULKAN_DEVICE%
echo [INFO] Running a basic Vulkan backend test (MUL_MAT operation)...
call :run "%TEST_BACKEND_OPS%" test -b "%VULKAN_DEVICE%" -o MUL_MAT
if errorlevel 1 (
    echo [ERROR] Vulkan backend smoke test failed.
    exit /b 1
)

echo.
echo ========================================
echo   Vulkan Backend Smoke Test PASSED
echo ========================================
exit /b 0

:ensure_dir
if exist "%~1" exit /b 0
mkdir "%~1" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to create directory: %~1
    exit /b 1
)
exit /b 0

:run
echo [RUN] %*
%*
exit /b %errorlevel%

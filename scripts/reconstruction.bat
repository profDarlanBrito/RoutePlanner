@echo off
set SCRIPT_PATH=%1
set PATH=%SCRIPT_PATH%\lib;%PATH%
set QT_PLUGIN_PATH=%SCRIPT_PATH%\lib\plugins;%QT_PLUGIN_PATH%
set COLMAP=%SCRIPT_PATH%\bin\colmap

set WORKSPACE_PATH=%2
set IMAGES_PATH=%WORKSPACE_PATH%\%3
set MAX_IMAGE_SIZE=1600
set IS_MULTPLE_MODELS=1

if not exist %WORKSPACE_PATH% (
    echo "Invalid workspace folder"
    Exit /b
)

if not exist %IMAGES_PATH% (
    echo "Invalid image folder"
    Exit /b
)

:automatic_reconstructor
@REM {low, medium, high, extreme}
@echo Automatic Reconstructor
%COLMAP% automatic_reconstructor ^
    --workspace_path %WORKSPACE_PATH% ^
    --image_path %IMAGES_PATH% ^
    --quality medium ^
    --camera_model PINHOLE

:save_statistics
@echo Save statistics about reconstructions
for /d %%A in (%WORKSPACE_PATH%\sparse\*) do (
    %COLMAP% model_analyzer ^
        --path %%A ^
        > %%A\stats.txt 2>&1

    %COLMAP% model_converter ^
        --input_path %%A ^
        --output_path %%A\model.nvm ^
        --output_type NVM
)

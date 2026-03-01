@echo off
REM Update these paths to match your local setup
set "TRAIN_DATA=data\train\flair"
set "TRAIN_GTS=data\train\gt"
set "VAL_DATA=data\dev_in\flair"
set "VAL_GTS=data\dev_in\gt"
set "BASE_SAVE_DIR=experiments_unet"

FOR %%S IN (1 2 3) DO (
    echo ----------------------------------------------------------------
    echo Running training for Seed %%S
    echo ----------------------------------------------------------------
    
    REM Ensure save directory exists
    if not exist "%BASE_SAVE_DIR%\seed%%S" mkdir "%BASE_SAVE_DIR%\seed%%S"

    python src\train.py ^
    --seed %%S ^
    --path_train_data "%TRAIN_DATA%" ^
    --path_train_gts "%TRAIN_GTS%" ^
    --path_val_data "%VAL_DATA%" ^
    --path_val_gts "%VAL_GTS%" ^
    --path_save "%BASE_SAVE_DIR%\seed%%S"
)

echo Done.
pause
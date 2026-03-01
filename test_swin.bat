@echo off
REM Update these paths to match your local setup
REM Using dev_in (validation set) for evaluation by default
set "TEST_DATA=data\dev_in\flair"
set "TEST_GTS=data\dev_in\gt"
set "TEST_BM=data\dev_in\fg_mask"
set "MODEL_DIR=experiments_swin"

echo ----------------------------------------------------------------
echo Running evaluation (Swin UNETR)
echo ----------------------------------------------------------------

python src\test_swin.py ^
--path_model "%MODEL_DIR%" ^
--path_data "%TEST_DATA%" ^
--path_gts "%TEST_GTS%" ^
--path_bm "%TEST_BM%" ^
--threshold 0.35

echo Done.
pause
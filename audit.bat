@echo off
REM Audit script to compare UNet and Swin UNETR on Ljubljana (dev_out)

set "AUDIT_DATA=data\dev_out\flair"
set "AUDIT_GTS=data\dev_out\gt"
set "AUDIT_BM=data\dev_out\fg_mask"

set "UNET_DIR=experiments_unet"
set "SWIN_DIR=experiments_swin"

echo ----------------------------------------------------------------
echo Running Failure Mode Audit (UNet vs Swin UNETR)
echo ----------------------------------------------------------------

python src\audit.py ^
--path_unet "%UNET_DIR%" ^
--path_swin "%SWIN_DIR%" ^
--path_data "%AUDIT_DATA%" ^
--path_gts "%AUDIT_GTS%" ^
--path_bm "%AUDIT_BM%"

pause
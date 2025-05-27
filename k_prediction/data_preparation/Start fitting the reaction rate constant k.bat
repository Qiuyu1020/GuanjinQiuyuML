@echo off
call C:\ProgramData\anaconda3\Scripts\activate.bat ML
cd /d D:\DESKTOP\GuanjinQiuyuML\data_preparation
streamlit run local_web_k_fitting.py
pause

@echo off
call C:\ProgramData\anaconda3\Scripts\activate.bat ML
cd /d D:\DESKTOP\GuanjinQiuyuML\data preparation
streamlit run local_web_input_mass_g.py
pause
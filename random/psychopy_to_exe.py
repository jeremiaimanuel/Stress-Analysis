from cx_Freeze import setup, Executable

setup(

    name="MAT Jp 5 min",
    version = "1",
    executeables=[Executable("v20241205_arithmetictest_5mins_jp.py",base="gui")],

)
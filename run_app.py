# run_app.py
import streamlit.web.cli as stcli
import os
import sys

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

if __name__ == "__main__":
    # 設置 Streamlit 參數
    # 指向打包後的 app.py 路徑
    app_path = resource_path("app.py")
    
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--global.developmentMode=false",
    ]
    
    # 執行 Streamlit
    sys.exit(stcli.main())

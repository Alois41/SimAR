"""
Build script for windows, build result inside dist folder
"""
if __name__ == '__main__':
    from subprocess import check_output
    check_output("pyinstaller main.spec -y", shell=True)

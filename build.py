from subprocess import check_output
check_output("pyinstaller run.spec -y", shell=True)

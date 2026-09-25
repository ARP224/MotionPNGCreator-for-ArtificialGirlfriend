' Launches the given program with its window hidden (window style 0).
'
' Why this exists: venv launchers created by newer uv versions
' (.venv\Scripts\pythonw.exe "trampolines") are console-subsystem
' executables, so they open a console window even though they are
' named pythonw. Running them through WScript.Shell with style 0
' hides that console regardless of how the launcher was built.
'
' Usage: wscript //nologo launch_hidden.vbs <exe> [args...]
Option Explicit
Dim sh, cmd, i
Set sh = CreateObject("WScript.Shell")
cmd = ""
For i = 0 To WScript.Arguments.Count - 1
    If i > 0 Then cmd = cmd & " "
    cmd = cmd & """" & WScript.Arguments(i) & """"
Next
sh.Run cmd, 0, False

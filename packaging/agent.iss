#define AppVersion "0.2.0"
#ifndef PayloadDir
  #define PayloadDir "..\.local\agent-build\dist\IBPPhotoAgent"
#endif
[Setup]
AppId={{B406A86D-9254-4BC0-A856-942C45AC3D11}
AppName=IBP Studio Photo Agent
AppVersion={#AppVersion}
AppPublisher=IBP community build
DefaultDirName={localappdata}\Programs\IBPPhotoAgent
DefaultGroupName=IBP Photo Agent
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\.local\release
OutputBaseFilename=IBP-Photo-Agent-0.2.0-win-x64-setup
Compression=lzma2/fast
CloseApplications=yes
RestartApplications=no
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\IBPPhotoAgent.exe
LicenseFile=..\LICENSE
DisableProgramGroupPage=yes
[Files]
Source: "{#PayloadDir}\*"; DestDir: "{app}"; Excludes: "*.gguf"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#PayloadDir}\models\*.gguf"; DestDir: "{app}\models"; Flags: ignoreversion nocompression
[Icons]
Name: "{group}\IBP Photo Agent"; Filename: "{app}\IBPPhotoAgent.exe"
Name: "{group}\ImageBatchProcessor Studio"; Filename: "{app}\IBPPhotoAgent.exe"
Name: "{userdesktop}\IBP Photo Agent"; Filename: "{app}\IBPPhotoAgent.exe"
[Run]
Filename: "{app}\IBPPhotoAgent.exe"; Description: "Launch IBP Photo Agent"; Flags: nowait postinstall skipifsilent

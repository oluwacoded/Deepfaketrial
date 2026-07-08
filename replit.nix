{pkgs}: {
  deps = [
    pkgs.ffmpeg
    pkgs.xorg.libICE
    pkgs.xorg.libSM
    pkgs.xorg.libXext
    pkgs.xorg.libX11
    pkgs.glib
    pkgs.libGL
    pkgs.xorg.libxcb
  ];
}

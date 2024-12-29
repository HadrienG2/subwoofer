# Requirements

Because this project uses features specific to your CPU model, it is not easily
amenable to binary distribution. The recommended way to use it is therefore to
roll out an optimized local build for your machine.

For this, you are going to need `rustup` and the `libhwloc` C library along with
the associated pkg-config file. The latter should either be installed
system-wide or be reachable via your `PKG_CONFIG_PATH`.

On Unices like Linux and macOS, you can install `libhwloc` and the tooling
needed to link it with Rust code by running the following commands...

- **macOS:**
  ```bash
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"  \
  && brew install hwloc pkgconf
  ```
- **Debian/Ubuntu/Mint:**
  ```bash
  sudo apt-get update  \
  && sudo apt-get install build-essential libhwloc-dev libudev-dev pkg-config
  ```
- **Fedora:**
  ```bash
  sudo dnf makecache --refresh   \
  && sudo dnf group install c-development   \
  && sudo dnf install hwloc-devel libudev-devel pkg-config
  ```
- **RHEL/Alma/Rocky:**
  ```bash
  sudo dnf makecache --refresh  \
  && sudo dnf groupinstall "Devlopment tools"  \
  && sudo dnf install epel-release  \
  && sudo /usr/bin/crb enable  \
  && sudo dnf makecache --refresh  \
  && sudo dnf install hwloc-devel libudev-devel pkg-config
  ```
- **Arch:**
  ```bash
  sudo pacman -Syu base-devel libhwloc pkg-config
  ```
- **openSUSE:**
  ```bash
  sudo zypper ref  \
  && sudo zypper in -t pattern devel_C_C++  \
  && sudo zypper in hwloc-devel libudev-devel pkg-config
  ```

...then you can install `rustup` and bring it into your current shell's
environment with those commands:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain none  \
&& . "$HOME/.cargo/env"
```

The required nightly Rust toolchain will then be installed automatically on the
first run of one of the `cargo bench` commands discussed in the data acquisition
section.

On Windows, I would recommend using [Windows Subsystem for
Linux](https://learn.microsoft.com/en-us/windows/wsl/install) (aka WSL) and
following the instructions for Ubuntu above because...

- WSL offers a much better software developer user experience than native
  Windows development.
- Contrary to what you may think, the underlying Linux virtual machine will not
  get in the way of precise CPU microbenchmarking due to the magic of 
  [VT-x](https://fr.wikipedia.org/wiki/Intel_VT)/[AMD-V](https://fr.wikipedia.org/wiki/Advanced_Micro_Devices#Pacifica/AMD-V).

...but if you really want a native Windows development environment, please take
inspiration from [these setup
instructions](https://numerical-rust-cpu-d1379d.pages.math.cnrs.fr/setup/windows.html)
that I wrote for a numerical computing course. They will tell you how to set up
everything needed for this benchmark, plus HDF5. You can leave out HDF5 for the
sake of minimalism, but this will prevent you from using the suggested
installation testing procedure.

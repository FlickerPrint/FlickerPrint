from setuptools import setup
import json
import subprocess
import os
import platform
import warnings

def locked_requirements(section):
    """Look through the 'Pipfile.lock' to fetch requirements by section."""
    with open('Pipfile.lock') as pip_file:
        pipfile_json = json.load(pip_file)

    if section not in pipfile_json:
        print("{0} section missing from Pipfile.lock".format(section))
        return []

    return [package + detail.get('version', "")
            for package, detail in pipfile_json[section].items()]





def check_working_Java_install():
    """
    Check to see if Java is already installed. If it is and the system is Unix-like, also check to see whether it is available in the PATH.
    """
    # Java is availavle on install in Windows so we can check for it directly.
    # In macOS and Linux, we need to ensure that JAVA_HOME and PATH are set correctly.
    if platform.system() == "Windows":
        try:
            java_version_output = subprocess.run(['java', '-version'], check=True, capture_output=True, text=True).stderr
            return_val = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return_val = False
    elif platform.system() == "Darwin":  # macOS
        try:
            java_version_output = subprocess.run(['java', '-version'], check=True, capture_output=True, text=True).stderr
            # On macOS, ensure JAVA_HOME is set correctly
            java_home = str(os.environ["JAVA_HOME"])
            if java_home is not None or java_home != "":
                if "Contents/Home" not in java_home:
                    # JAVA_HOME may not be correct
                    warnings.warn(f"\n########\nJAVA_HOME does not have the standard format so may not be set correctly.\nPlease check that it is set correctly.\n########\n")
                return_val = True
            else:
                return_val = False
        except (subprocess.CalledProcessError, KeyError):
            return_val = False
    else: # Linux or other Unix-like systems
        try:
            java_version_output = subprocess.run(['java', '-version'], check=True, capture_output=True, text=True).stderr
            # Ensure JAVA_HOME is set correctly
            java_home = str(os.environ["JAVA_HOME"])
            if java_home is not None or java_home != "":
                if "jdk" not in java_home:
                    # JAVA_HOME may not be correct
                    warnings.warn(f"\n########\nJAVA_HOME does not have the standard format so may not be set correctly.\nPlease check that it is set correctly.\n########\n")
                return_val = True
            else:
                return_val = False
        except (subprocess.CalledProcessError, KeyError):
            return_val = False

    return return_val

def check_default_Java_locations():  
    system = platform.system()
    updated_path = None
    java_home = None
    if system == "Windows":
        try:
            java_home = subprocess.run(
                ["where", "java"], capture_output=True, text=True
            ).stdout.strip()
        except subprocess.CalledProcessError:
            java_home = None
        if java_home:
            java_home = os.path.dirname(os.path.dirname(java_home))
            os.environ["JAVA_HOME"] = java_home
            updated_path = java_home + ";$PATH" + os.environ.get("PATH", "")
            os.environ["PATH"] = updated_path
    
    elif system == "Darwin":  # macOS
        try:
            java_home = subprocess.run(
                ["/usr/libexec/java_home"], capture_output=True, text=True
            ).stdout.strip()
        except subprocess.CalledProcessError:
            java_home = None
        if java_home:
            os.environ["JAVA_HOME"] = java_home
            updated_path = os.path.join(java_home, "bin") + ":$PATH"
            os.environ["PATH"] = updated_path

    elif system == "Linux":
        try:
            java_home = subprocess.run(
                ["readlink", "-f", "$(which java)"], shell=True, capture_output=True, text=True
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            java_home = None
        if java_home:
            java_home = os.path.dirname(os.path.dirname(java_home))
            if "bin" in java_home:
                java_home = java_home.split("bin")[0].rstrip("/")
            os.environ["JAVA_HOME"] = java_home
            updated_path = os.path.join(java_home, "bin") + ":$PATH"
            os.environ["PATH"] = updated_path
    else:
        print(f"Unsupported operating system: {system}\nPlease refer to the FlickerPrint documentation to ensure that Java is installed and that the\nJAVA_HOME and PATH environment variables are set correctly.")

    return java_home, updated_path


def update_profile(java_home: str, updated_path: str):
    """
    Update the user's profile file with the JAVA_HOME and PATH variables.
    This function is designed to work on Windows, macOS, and Linux, for most common installations.

    We check for profiles in the order that they are most likely to be used.
    In macOS, if no profile is found, we create .zprofile as the default.
    """
    profile_path = None
    if platform.system() == "Darwin":  # macOS
        profile_path = os.path.expanduser("~/.zprofile")
        if not os.path.exists(profile_path):
            profile_path = os.path.expanduser("~/.zshrc")
        if not os.path.exists(profile_path):
            profile_path = os.path.expanduser("~/.bash_profile")
        if not os.path.exists(profile_path):
            profile_path = os.path.expanduser("~/.bashrc")
        if not os.path.exists(profile_path):
            profile_path = os.path.expanduser("~/.profile")
        else:
            profile_path = os.path.expanduser("~/.zprofile")

    else:  # Linux or other Unix-like systems
        profile_path = os.path.expanduser("~/.bash_profile")
        if not os.path.exists(profile_path):
            profile_path = os.path.expanduser("~/.bashrc")
        if not os.path.exists(profile_path):
            profile_path = os.path.expanduser("~/.profile")

    if profile_path is None:
        print(f"Unable to determine the profile file path.\nPlease refer to the FlickerPrint documentation to ensure that\nthe JAVA_HOME and PATH environment variables are set correctly\nfor your shell profile.")
        print(f"##############################################################\n")
        return False

    with open(profile_path, "a+") as profile_file:
        profile_file.write(f"\nexport JAVA_HOME={java_home}\n")
        profile_file.write(f"export PATH={updated_path}\n")
        print(f"JAVA_HOME and PATH environment variables have been set in\n{profile_path}.Please restart your terminal\nor run 'source {profile_path}' to apply the changes.")
        print(f"##############################################################\n")



def check_Java():
    """
    Check if Java is installed and available in the ``PATH``.
    If not, check for a default installation and set ``JAVA_HOME`` and ``PATH``.
    
    This function is designed to work on Windows, macOS, and Linux, for most common installations.
    """
    if check_working_Java_install():
        # Java is available and working
        return True
    else:
        # Check for Java installation in the default locations
        # If none found, then prompt the user to install Java
        # If it is found, update the profile with the JAVA_HOME and PATH variables
        print(f"\n\n##############################################################")
        print(f"Java is not available in the PATH. Attempting to locate Java...")
        java_home, updated_path = check_default_Java_locations()
        if updated_path is None:
            print(f"Unable to determine the location of your Java installation.\nPlease ensure that you have installed Java before continuing.\nFor more information, refer to the FlickerPrint documentation.")
            print(f"##############################################################\n")
            return False
        else:
            # Update the profile:
            update_profile(java_home, updated_path)
            return False


# Check other platform-specific requirements

def check_xcode():
    """
    Check if Xcode is installed on macOS.
    If not, prompt the user to install it.
    """
    try:
        subprocess.run(['xcode-select', '--install'], check=True, capture_output=True)
        print(f"\n\n##############################################################")
        print(f"Xcode developer tools are not installed.\nAttempting installation...")
        print(f"##############################################################\n")
        return False
    except subprocess.CalledProcessError:
        return True
    
def check_Windows_Visual_Studio():
    """
    Check to see whether Visual Studio is installed on Windows.
    Strictly speeking, we need C++ tools and .NET but it is easiest to check for Visual Studio.
    If not installed, prompt the user to install it.
    """
    vswhere_path = os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe")
    
    if not os.path.exists(vswhere_path):
        return_val =  False
    else:
        try:
            result = subprocess.run(
                [vswhere_path, "-latest", "-products", "*", "-requires", "Microsoft.Component.MSBuild"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False
            )
            return_val = bool(result.stdout.strip())
        except :
            return_val = False

    if return_val:
        return True
    else:
        print(f"\n\n##############################################################")
        print(f"The Windows C++ and .Net tools are not installed.\nThese are required for compiling some of the dependencies of\nthe package.\nPlease install them using the Visual Studio Installer.\nFor more information, refer to the FlickerPrint documentation.")
        print(f"##############################################################\n")
        return False



def check_requirements():
    """Check to ensure that the non-Python requirements are met.
    This includes Java, Xcode (macOS), C++ compiler (Windows), and .NET (Windows).

    Returns
    -------
        bool: True if all requirements are met, False otherwise.
    """

    err = True
    if platform.system() == "Windows":
        if not check_Windows_Visual_Studio():
            err = False
    elif platform.system() == "Darwin":  # macOS
        if not check_xcode():
            err = False
    # Java required for all platforms
    if not check_Java():
        err = False

    return err



# #########

# Run Setup

# #########


# Check for all requirements
if not check_requirements():
    print(f"\n\n##############################################################")
    print(f"Please ensure that all requirements are met before proceeding.\nDetails of the software you need to install are printed above.")
    print(f"##############################################################\n\n")
    exit(1)

setup(name='flickerprint',
      version=0.3,
      author='Carl Jones, Jack Law, Thomas Williamson, Fynn Wolf, Endre Tønnessen',
      maintainer='Thomas Williamson',
      python_requires=">=3.9, <3.12",
      install_requires=["numpy",
                        "python-javabridge>=4.0.4",
                        "python-bioformats>=4.1.0",
                        "matplotlib",
                        "scikit-image",
                        "pandas",
                        "seaborn",
                        "h5py>=3.11.1",
                        "tqdm",
                        "argh",
                        "wget",
                        "strictyaml>=1.7.3",
                        "tables>=3.9.1",
                        "opencv-python>=4.10.0.84",
                        "exifread>=3.0.0",
                        "tensorflow>=2.16.2",
                        "shiny==1.2.1",
                        "more-itertools",
                        "trieste",
                        "shinyswatch",
                        "jinja2",
                        "cmake"],
      entry_points={
        'console_scripts': ['flickerprint=flickerprint.workflow.manager:main'
        ]},
      package_data={"flickerprint": ["common/defaults.yaml"]},
      include_package_data=True,
      )

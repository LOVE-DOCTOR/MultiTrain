import sys
import subprocess
from setuptools import setup
from setuptools.command.install import install


class CustomInstallCommand(install):
    """Customized install command that runs 'brew install libomp' on macOS."""

    def run(self):
        if sys.platform == "darwin":
            homebrew_installed = False
            try:
                # Check if Homebrew is installed by trying to call 'brew --version'
                subprocess.check_call(["brew", "--version"])
                homebrew_installed = True
            except Exception as e:
                print(
                    "Homebrew not found. Please install Homebrew to automatically install libomp."
                )

            if homebrew_installed:
                try:
                    subprocess.check_call(["brew", "install", "libomp"])
                except Exception as e:
                    print(
                        "Warning: Failed to install libomp using Homebrew. Please ensure libomp is installed.",
                        e,
                    )

        # Continue with the normal installation process
        install.run(self)


setup(
    cmdclass={"install": CustomInstallCommand},
)

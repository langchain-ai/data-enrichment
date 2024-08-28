import os
import shutil
from setuptools import setup
from setuptools.command.build_py import build_py


class CustomBuildPy(build_py):
    def run(self):
        build_py.run(self)
        build_dir = self.build_lib
        nested_dir = os.path.join(build_dir, "langgraph", "templates")
        os.makedirs(nested_dir, exist_ok=True)

        # Move all directories from src to nested structure
        for item in os.listdir(build_dir):
            item_path = os.path.join(build_dir, item)
            if os.path.isdir(item_path) and item != "langgraph":
                shutil.move(item_path, os.path.join(nested_dir, item))

        open(os.path.join(build_dir, "langgraph", "__init__.py"), "a").close()
        open(os.path.join(nested_dir, "__init__.py"), "a").close()

        print(
            "Custom build completed. Packages restructured under langgraph.templates."
        )


setup(
    cmdclass={
        "build_py": CustomBuildPy,
    },
)

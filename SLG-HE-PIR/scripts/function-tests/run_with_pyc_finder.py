"""Bootstrap loader for running scripts when .py source files are missing.

Discovers all `__pycache__/<name>.cpython-312.pyc` files in the repo and resolves
imports via a custom MetaPathFinder so that `two_epoch_test` and friends can be
invoked as if the .py files were present.

Usage:
    python run_with_pyc_finder.py scripts.function_tests.two_epoch_test --epochs 20 ...

Or just import this file from another wrapper script to enable the finder.
"""
import os
import sys
import importlib.machinery
import importlib.util


_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_HERE)  # parent of scripts/


class PycMetaFinder:
    """Resolve any import to <repo>/<dotted>/__pycache__/<leaf>.cpython-312.pyc.

    Falls back to standard import machinery if no .pyc is found.
    """

    def __init__(self, repo_root: str):
        self.repo_root = repo_root

    def find_spec(self, name, path=None, target=None):
        parts = name.split(".")
        candidates = []
        # 1) top-level package: <repo>/<name>/__init__.pyc
        candidates.append(os.path.join(self.repo_root, parts[0], "__init__.pyc"))
        # 2) top-level module: <repo>/<name>.pyc
        candidates.append(os.path.join(self.repo_root, parts[0] + ".pyc"))
        # 3) submodule inside package (only if name has dots)
        if len(parts) > 1:
            # module: <repo>/<package>/<submodule>.pyc
            candidates.append(
                os.path.join(self.repo_root, *parts[:-1], parts[-1] + ".pyc")
            )
            # module: <repo>/<package>/__pycache__/<submodule>.cpython-312.pyc
            candidates.append(
                os.path.join(self.repo_root, *parts[:-1],
                             "__pycache__", parts[-1] + ".cpython-312.pyc")
            )
            # package sub-init: <repo>/<package>/<subpackage>/__pycache__/__init__.cpython-312.pyc
            candidates.append(
                os.path.join(self.repo_root, *parts,
                             "__pycache__", "__init__.cpython-312.pyc")
            )
            # nested package: <repo>/<package>/<subpackage>/__init__.pyc
            candidates.append(
                os.path.join(self.repo_root, *parts, "__init__.pyc")
            )
        for c in candidates:
            if c and os.path.exists(c):
                spec = importlib.util.spec_from_file_location(
                    name, c, loader=importlib.machinery.SourcelessFileLoader(name, c),
                )
                # A package init ends with __init__.pyc — mark it so submodule imports work
                if c.endswith("__init__.pyc") or c.endswith("__init__.cpython-312.pyc"):
                    pkg_dir = os.path.dirname(c)
                    if c.endswith("__init__.cpython-312.pyc"):
                        pkg_dir = os.path.dirname(pkg_dir)
                    spec.submodule_search_locations = [pkg_dir]
                return spec
        return None  # let default finders handle it


def install_pyc_finder(repo_root: str = REPO_ROOT) -> PycMetaFinder:
    finder = PycMetaFinder(repo_root)
    sys.path.insert(0, repo_root)
    sys.meta_path.insert(0, finder)
    return finder


if __name__ == "__main__":
    install_pyc_finder()
    if len(sys.argv) < 2:
        print("usage: python run_with_pyc_finder.py <module.to.run> [args...]", file=sys.stderr)
        sys.exit(1)
    target = sys.argv[1]
    sys.argv = sys.argv[1:]
    # Run the target module as __main__ so argparse sees the right sys.argv
    spec = importlib.util.spec_from_file_location(
        "__main__",
        # resolve target path: try .pyc with sys.path lookup
        # if user gives "scripts.function_tests.two_epoch_test", import it then exec as main
        None,
    )
    # easier: import the module normally then re-execute its __dict__ as __main__
    mod_name = target
    # Import using the meta path finder
    import importlib
    mod = importlib.import_module(mod_name)
    # Re-execute its main() under __main__
    sys.modules["__main__"] = mod
    # hand control to mod.main() if it exists; else Python already ran the top-level
    if hasattr(mod, "main"):
        mod.main()
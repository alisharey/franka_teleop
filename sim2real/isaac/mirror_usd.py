"""Mirrors a USD asset and everything it references from Isaac's asset server to a local folder
(the compute nodes' connection to the server is flaky; Isaac Sim then fails to load assets).

    python mirror_usd.py <url of the .usd> <local folder>
Run with Isaac Lab's python (needs pxr). Prints the local path of the main file. The Panda for
cup_demos.py / replica.py (--robot-usd) goes to sim2real/assets/ (git-ignored), e.g. from IsaacLab/:
    ./isaaclab.sh -p <repo>/sim2real/isaac/mirror_usd.py \
        https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd \
        <repo>/sim2real/assets/panda
"""
import posixpath
import subprocess
import sys
import time
from pathlib import Path

from pxr import Sdf


def fetch(url, path, tries=8):
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(tries):
        result = subprocess.run(['curl', '-sf', '--retry', '3', '-o', str(path), url])
        if result.returncode == 0 and path.stat().st_size > 0:
            return True
        time.sleep(1 + attempt)
    return False


def main():
    url, root = sys.argv[1], Path(sys.argv[2])
    base = url.rsplit('/', 1)[0]
    todo, done = [posixpath.basename(url)], set()
    while todo:
        relative = posixpath.normpath(todo.pop())
        if relative in done:
            continue
        done.add(relative)
        local = root / relative
        if not local.exists() and not fetch(base + '/' + relative, local):
            print('FAILED', relative)
            continue
        if local.suffix not in ('.usd', '.usda', '.usdc'):
            continue
        layer = Sdf.Layer.FindOrOpen(str(local))
        if layer is None:
            continue
        for dependency in set(layer.GetCompositionAssetDependencies()) | set(layer.externalReferences):
            if not dependency or '://' in dependency or dependency.startswith('/'):
                continue
            todo.append(posixpath.join(posixpath.dirname(relative), dependency))
    print('mirrored %d files to %s' % (len(done), root))
    print('MAIN', root / posixpath.basename(url))


main()

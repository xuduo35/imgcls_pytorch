import os
import shutil
import matplotlib

print(matplotlib.matplotlib_fname())
print(matplotlib.get_cachedir())
print("wget https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf")
print(f"rm -f {matplotlib.get_cachedir()}/*")
fontsdir = os.path.join(os.path.dirname(os.path.join(matplotlib.matplotlib_fname())), "fonts/ttf")
print(f"cp ./SimHei.ttf {fontsdir}")

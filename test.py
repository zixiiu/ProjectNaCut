import os

for root, dirs, files in os.walk("F:\\Testv2\\11336264"):
    for name in files:
        if name.endswith('.mp4'):
            print(root.split("\\")[-1])
            print(os.path.join(root, name))


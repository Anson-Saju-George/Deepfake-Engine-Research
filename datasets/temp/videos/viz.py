import os

def tree_dirs_only(root, prefix=""):
    total_dirs = 0

    items = sorted(os.listdir(root))
    dirs = [d for d in items if os.path.isdir(os.path.join(root, d))]

    for i, d in enumerate(dirs):
        path = os.path.join(root, d)
        connector = "└── " if i == len(dirs) - 1 else "├── "

        print(prefix + connector + d)

        sub_dirs = tree_dirs_only(
            path,
            prefix + ("    " if i == len(dirs) - 1 else "│   ")
        )
        total_dirs += 1 + sub_dirs

    return total_dirs


def count_files(root):
    total_files = 0
    for _, _, files in os.walk(root):
        total_files += len(files)
    return total_files


if __name__ == "__main__":
    root = "."
    print(root)

    dirs = tree_dirs_only(root)
    files = count_files(root)

    print(f"\n{dirs} directories, {files} files")
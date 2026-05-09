import os.path


def split_path(path: str) -> tuple[str, str]:
    """ Splits a path into an outer part pointing to an archive and an inner part pointing to a file inside the archive.

    The inner part may be empty if the input path does not point to a file inside an archive.
    :param path: The path to a file in an archive that shall be split into an outer and inner part.
    :return: A pair of strings representing the outer and inner part of the input path.
    """
    archive_extensions = [".zip", ".tar"]
    for extension in archive_extensions:
        split_point = path.find(f"{extension}{os.path.sep}")
        if split_point != -1:
            split_point += len(extension)
            return path[:split_point], path[split_point+len(os.path.sep):]
    return path, ""

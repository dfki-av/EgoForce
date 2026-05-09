##############################################################################
# Copyright (c) 2023 DFKI GmbH - All Rights Reserved
# Written by Stephan Krauß <Stephan.Krauss@dfki.de>, March 2023
##############################################################################
"""doc
# datapipes.archive_processor: Archive Processor

> Allows to (pre-)process datasets stored in archives without extracting the files to disk
"""
from __future__ import annotations

import io
import os.path
import tarfile
import zipfile
from abc import ABCMeta, abstractmethod
from multiprocessing import Pool, RLock
from typing import IO

from tqdm import tqdm

import datapipes.utils.term_colors as tc


class ArchiveProcessor(metaclass=ABCMeta):
    def __init__(self,
                 num_processes: int = 8,
                 compression: int = zipfile.ZIP_STORED,
                 compress_level: int | None = None,
                 print_progress_bar: bool = True,
                 continue_on_errors: bool = False):
        """ Creates a new ArchiveProcessor.

        :param num_processes: The number of processes that may be used to process archives in parallel.
        :param compression: The compression method that shall be used. Defaults to ZIP_STORED (no compression) which is
          suitable for compressed images. For other file formats ZIP_DEFLATED may be considered for compressed storage.
        :param compress_level: If ZIP_DEFLATED is used, the compress_level determines how much CPU time is used to try
          to increase the compression ratio.
        :param print_progress_bar: Whether to print a progress bar indicating the progress of the individual conversion
          processes or not. If set to False only messages indicating that an archive has been processed completely will
          be printed.
        :param continue_on_errors: Whether to continue processing an archive if exceptions are thrown in the processing
          function.
        """
        self.in_dir = None
        self.out_dir = None
        self.num_processes = num_processes
        self.compression = compression
        self.compress_level = compress_level
        self.print_progress_bar = print_progress_bar
        self.continue_on_errors = continue_on_errors

    def process_dataset(self, in_dir: str, out_dir: str) -> None:
        """ Processes the archives of a dataset.

        :param in_dir: The directory containing the zip or tar archives of the dataset that shall be (pre-)processed.
        :param out_dir: The directory where the archives containing the (pre-)processed files of the dataset shall be
          stored.
        """
        if not os.path.exists(in_dir):
            raise RuntimeError(f"Input directory does not exist: {in_dir}")
        self.in_dir = os.path.normpath(in_dir)
        self.out_dir = os.path.normpath(out_dir)

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        input_archives = []

        # traverse the specified directory and its subdirectories
        for root_dir, dir_names, file_names in os.walk(in_dir):
            for file_name in file_names:
                src_path = os.path.join(root_dir, file_name)
                extension = os.path.splitext(file_name)[1]
                if extension in [".tar", ".zip"] and self.include_archive(src_path):
                    full_path = os.path.join(root_dir, src_path)
                    input_archives.append((os.path.getsize(full_path), full_path))
                else:
                    print("Skipping", src_path)

        # sort files by descending file size to improve load distribution
        input_archives.sort(key=lambda x: x[0], reverse=True)
        input_archives = [(idx, x[1]) for idx, x in enumerate(input_archives)]

        num_archives = len(input_archives)
        num_processes = min(self.num_processes, num_archives)

        tqdm.set_lock(RLock())  # for managing output contention
        all_bad_files = []
        if num_processes > 1:
            print(f"Processing {num_archives} archives using {num_processes} sub-processes...", flush=True)
            with Pool(num_processes, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as p:
                bad_files = p.map(self._process_archive, input_archives)
                for item in bad_files:
                    all_bad_files.extend(item)
        else:
            print(f"Processing {num_archives} archives using 0 sub-processes...")
            for x in input_archives:
                all_bad_files.extend(self._process_archive(x))
        if len(all_bad_files) == 0:
            print("Done")
        else:
            tc.print_c(f"\nFinished with {len(all_bad_files)} errors. Processing the following files failed:",
                       tc.Color.RED, flush=True)
            for file_path in all_bad_files:
                tc.print_c(file_path, tc.Color.RED, flush=True)

    def _process_archive(self, item: tuple[int, str]) -> list[str]:
        """ Processes the files in an individual archive of a dataset.

        :param item: A tuple of an index and the path to source zip or tar archive. The index is used to identify the
          progress bar associated with the archive.
        """
        index, src_path = item
        # re-create folder structure from inside in_dir in out_dir and place output zip in same relative location
        out_dir = os.path.join(self.out_dir, os.path.relpath(os.path.split(src_path)[0], start=self.in_dir))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        output_archive_path = os.path.join(out_dir, self.transform_archive_name(os.path.basename(src_path)))
        # ensure the file extension of the output archive is always ".zip"
        if not output_archive_path.endswith(".zip"):
            output_archive_path += ".zip"
        bad_files = []
        with zipfile.ZipFile(output_archive_path, "w", self.compression, True, self.compress_level) as output_archive:
            extension = os.path.splitext(src_path)[1]
            if extension == ".tar":
                with tarfile.open(src_path, "r") as in_file:
                    if self.print_progress_bar:
                        description = os.path.relpath(src_path, self.in_dir)
                        files_in_archive = tqdm(in_file, desc=description, position=index)
                    else:
                        files_in_archive = in_file
                    for tar_info in files_in_archive:
                        try:
                            if tar_info.isfile() and self.include_file(src_path, tar_info.name):
                                with in_file.extractfile(tar_info) as data:
                                    outputs = self.process_file(src_path, tar_info.name, data)
                                    for out_name, out_data in outputs:
                                        # Seek to the beginning of the file buffer,
                                        # otherwise we will write an empty file.
                                        out_data.seek(0)
                                        output_archive.writestr(out_name, out_data.read())
                        except BaseException as e:
                            archive_path = os.path.relpath(src_path, self.in_dir)
                            bad_files.append(f"{archive_path}/{tar_info.name}  error: {e}")
                            if not self.continue_on_errors:
                                break
            elif extension == ".zip":
                with zipfile.ZipFile(src_path, "r") as in_file:
                    if self.print_progress_bar:
                        description = os.path.relpath(src_path, self.in_dir)
                        files_in_archive = tqdm(in_file.infolist(), desc=description, position=index)
                    else:
                        files_in_archive = in_file.infolist()
                    for zip_info in files_in_archive:
                        try:
                            if not zip_info.filename.endswith("/") and self.include_file(src_path, zip_info.filename):
                                with in_file.open(zip_info) as data:
                                    outputs = self.process_file(src_path, zip_info.filename, data)
                                    for out_name, out_data in outputs:
                                        # Seek to the beginning of the file buffer,
                                        # otherwise we will write an empty file.
                                        out_data.seek(0)
                                        output_archive.writestr(out_name, out_data.read())
                        except BaseException as e:
                            archive_path = os.path.relpath(src_path, self.in_dir)
                            bad_files.append(f"{archive_path}/{zip_info.filename}  error: {e}")
                            if not self.continue_on_errors:
                                break
        if not self.print_progress_bar:
            print(f"Finished processing {src_path} with {len(bad_files)} errors", flush=True)
        return bad_files

    def include_archive(self, archive_path: str) -> bool:
        """ Returns for a given archive path whether this archive shall be processed or not.

        Can be used to avoid processing irrelevant archives.
        The default implementation accepts all archives.
        Overwrite this method in a subclass to change the behavior.

        :param archive_path: The path to the archive in question.
        :return: Whether the archive shall be processed (True) or not (False).
        """
        return True

    def include_file(self, archive_path: str, filepath: str) -> bool:
        """ Whether to include a file in an archive in the processing.

        Can be used to avoid extracting and processing files that are not needed or that are damaged.
        The default implementation accepts all files.
        Overwrite this method in a subclass to change the behavior.

        :param archive_path: The path to the archive containing the file.
        :param filepath: The filename or path of the file inside the archive.
        """
        return True

    @abstractmethod
    def transform_archive_name(self, src_archive_name: str) -> str:
        """ Transforms the name of the input archive to the name that shall be used for the output archive.

        :param src_archive_name: The name of the input archive.
        :return: The name of the output archive.
        """
        return src_archive_name

    @abstractmethod
    def process_file(self,
                     src_archive_path: str,
                     filepath: str,
                     file_data: IO[bytes]) -> list[tuple[str, io.StringIO | io.BytesIO]]:
        """ This function performs the actual processing of a file read from an input archive and returns the resulting
         processed file(s).

        :param src_archive_path: The path to the archive that contains the file that shall be processed. May be used
          to alter behavior based on the archive containing the file.
        :param filepath: The path of the file in the input archive.
        :param file_data: A buffer from which the file content can be read.
        :return: A list of files that resulted from the (pre-)processing. Can be an empty list.
        """
        raise NotImplementedError()

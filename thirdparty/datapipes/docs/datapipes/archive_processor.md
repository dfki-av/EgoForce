[Back to Overview](../README.md)

# datapipes.archive_processor: Archive Processor

> Allows to (pre-)process datasets stored in archives without extracting the files to disk

## *class* **ArchiveProcessor**(metaclass=ABCMeta) [[src]](../../datapipes/archive_processor.py#L25)
Creates a new ArchiveProcessor.

* **num_processes**: The number of processes that may be used to process archives in parallel.
* **compression**: The compression method that shall be used. Defaults to ZIP_STORED (no compression) which is
  suitable for compressed images. For other file formats ZIP_DEFLATED may be considered for compressed storage.
* **compress_level**: If ZIP_DEFLATED is used, the compress_level determines how much CPU time is used to try
  to increase the compression ratio.
* **print_progress_bar**: Whether to print a progress bar indicating the progress of the individual conversion
  processes or not. If set to False only messages indicating that an archive has been processed completely will
  be printed.
* **continue_on_errors**: Whether to continue processing an archive if exceptions are thrown in the processing
  function.

### *def* **process_dataset** [[src]](../../datapipes/archive_processor.py#L53)
Processes the archives of a dataset.

* **in_dir**: The directory containing the zip or tar archives of the dataset that shall be (pre-)processed.
* **out_dir**: The directory where the archives containing the (pre-)processed files of the dataset shall be
  stored.

### *def* **include_archive** [[src]](../../datapipes/archive_processor.py#L174)
Returns for a given archive path whether this archive shall be processed or not.

Can be used to avoid processing irrelevant archives.
The default implementation accepts all archives.
Overwrite this method in a subclass to change the behavior.

* **archive_path**: The path to the archive in question.
* **returns**: Whether the archive shall be processed (True) or not (False).

### *def* **include_file** [[src]](../../datapipes/archive_processor.py#L186)
Whether to include a file in an archive in the processing.

Can be used to avoid extracting and processing files that are not needed or that are damaged.
The default implementation accepts all files.
Overwrite this method in a subclass to change the behavior.

* **archive_path**: The path to the archive containing the file.
* **filepath**: The filename or path of the file inside the archive.

### *def* **transform_archive_name** [[src]](../../datapipes/archive_processor.py#L199)
Transforms the name of the input archive to the name that shall be used for the output archive.

* **src_archive_name**: The name of the input archive.
* **returns**: The name of the output archive.

### *def* **process_file** [[src]](../../datapipes/archive_processor.py#L208)
This function performs the actual processing of a file read from an input archive and returns the resulting
 processed file(s).

* **src_archive_path**: The path to the archive that contains the file that shall be processed. May be used
  to alter behavior based on the archive containing the file.
* **filepath**: The path of the file in the input archive.
* **file_data**: A buffer from which the file content can be read.
* **returns**: A list of files that resulted from the (pre-)processing. Can be an empty list.

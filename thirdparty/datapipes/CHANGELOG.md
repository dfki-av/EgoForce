# Changelog

## 3.7.0

* Added validation of input paths
* Added validation to requested sample component names
* Added support for length-limiting of single shard datasets
* Improved in-memory dataset and UnrealEgo examples
* Added template for converting webdatasets
* Added function to collate samples to a single dict
* Updated documentation

## 3.6.0

* Added example for use of in-memory datasets
* Updated readme to point to new example for in-memory datasets
* Added validation for user defined values when creating FileInfo object
* Added more explanations on the add_component_fn parameter
* Updated estimation of reading efficiency

## 3.5.1

* Fixed crash if torchdata is not available
* Optimized sample access in InMemoryDataset
* Improved error message in InMemoryDataset

## 3.5.0

* Removed dependency on torchdata
* Improved and cleaned the main readme file
* Updated FaceSynthetics and UnrealEgo examples to show the latest features
* Added template for creation ds_info.json files
* Added missing links in API documentation
* Fixed pyTorch dependency (version 1.12.1 was specified but 1.13.0 is actually needed)

## 3.4.0

* Added decoders for message pack and pickle
* Improved handling of errors in ArchiveProcessor by printing the error for each failed file
* Made split_path function public to help with parsing paths during dataset conversion
* Shortened message when printing version during dataset conversion

## 3.3.1

* Fixed installation as a pip package

## 3.3.0

* Added two ways to convert a dataset with a single function call
* Added UnrealEgo2 example using one of the new functions to convert the dataset
* Changed template to use the single function call conversion
* Improved documentation and error reporting
* Added commercial use field to ds_info file creation code
* ArchiveProcessor will now recreate the source dataset's folder structure in the output directory
* Updated the Readme file to provide an easier start with converting datasets

## 3.2.1

* Fixed writing shards message being printed multiple times

## 3.2.0

* Cleaned console output of dataset converter
* Improved formatting of console output in dataset converter and archive processor
* Small improvement to Unreal Ego example conversion script
* Updated API documentation

## 3.1.3

* Fixed bug in pipeline synchronization for multi-GPU training which occurred when there were more workers than shards 

## 3.1.2

* Fixed issues with pyTorch 2.1.x
* Fixed bugs in multi-GPU synchronization code which limits pipelines to the same length
* Accept shuffle_buffer_size explicitly set to None and treat it as 0
* Cleaned code of SequenceAwareShuffler

## 3.1.1

* Fixed crash if max_samples is set in single GPU scenarios
* Fixed crash if temporal_sliding_window_size is explicitly set to None

## 3.1.0

### New features

* New metadata format (version 2), which adds information of the length of sequences in sequence datasets
* Automatic synchronization of pipeline lengths to prevent stalling of GPUs in multi-GPU scenarios for sequence and non-sequence datasets

### Changes

* The max_samples parameter is now limiting the number of samples per GPU instead of per DataLoader worker
* Use of GPU parameter during pipeline creation is deprecated (and no longer needed)
* Added warning in case a sequence pipeline is constructed for pre-shuffled datasets
* Added warning in case a sequence pipeline is constructed using the old metadata format version
* Updated UnrealEgo example
* Removed outdated H2O example
* Updated comments and documentation

## 3.0.1

* Added several performance optimizations for the InMemoryDataset
* Added checks to verify that dataset path provided to InMemoryDataset exists 

## 3.0.0

### Backwards Incompatible API Changes

* The API of ArchiveProcessor changed. The method "preprocess_file" was renamed to "process_file". The parameter filename was renamed to filepath.

### New features

* Added an implementation that serves small datasets from DRAM

### Other changes

* The BasePipelineCreator's get_tar_files_for_subsets now also accepts a single subset as the first parameter
* The variable sequence_name in the dataset conversion template was renamed to sequence_id to align it with the naming convention.
* Updated documentation
* Updated readme

## 2.1.3

* Fixed ComponentAdder not constructing the file path correctly

## 2.1.2

* Fixed get_ids_from_path not working correctly if sequence IDs are not used in the dataset
* Changed ArchiveProcessor to use the name of the input as default name for the output archive 

## 2.1.1

* Updated readme and API documentation to reflect recent changes

## 2.1.0

* Added support for multiple workers for single shard datasets
* Added simple shard balancing strategy and enabled targeting specific shard sizes for sequence datasets
* Reduces the default threshold for small datasets that are merged into a single shard to 16 GB
* Various improvements for the ArchiveProcessor including improved error handling

## 2.0.1

* Added code example for conversion of the KITTI object dataset
* Fixed bug when converting small datasets containing sequence data not creating single shards
* Updated readme file

## 2.0.0

### Backwards Incompatible API Changes

* The decoder has been renamed to Dispatcher to reflect its broader applicability.
At the same time is behavior and interfaces have changed slightly:
  * It now passes the path to each file instead of only the file extension.
  * This means it is no longer compatible with the decoders that are built into torchvision. Torchvision's ImageHandler can be replaced by ImageDecoder which is shipped with datapipes.
  * If you wrote your own decoders that make use of the file extension, you need adjust it to extract the extension from the path. (Helper functions are available in `dataset_path_utils.py`)
  * The constructor accepts additional arguments. If you are passing a custom key function, then you now need to pass it as a keyword parameter.
  * Imports need to be adjusted as follows:  
`from datapipes.utils.decoder import Decoder` → `from datapipes.utils.dispatcher import Dispatcher`

* The actual decoders have been moved to their own directory. The imports need to be adjusted as follows:  
`from datapipes.{utils → decoders}.image_decoder import ImageDecoder`  
`from datapipes.{utils → decoders}.json_decoder import decode_json`  
`from datapipes.{utils → decoders}.pointcloud_decoder import PointcloudDecoder`

* The file that contains the ArchiveProcessor has been renamed to match the class name.
Imports need to be adjusted as follows:  
`from datapipes.{dataset_preprocessor → archive_processor} import ArchiveProcessor`

* The collation function `collate_batch_as_dict` has been renamed to `collate_dict`, because the name was misleading.
The data must already be a dict. Use the new function `collate_batch_as_dicts` to collate from tuples to dicts. 

* The function for grouping components from multiple pipelines `group_fn` has been moved from `collation_functions.py` to `dataset_path_utils.py` and renamed to `get_sample_id`.
This should not affect user as the function is typically only used internally by the pipeline creator classes.

* Some internal functions and data structures are no longer exposed by default.
If you are using any of them you need to adjust your imports. (An "_" was added to the beginning of the name.)

* Sequence pipelines wrap all data loaded from files in a new StateWrapper.
This is necessary to be able to avoid applying the same operation multiple times to the same object due to the replication of objects caused by the sliding window filter.

### Other New Features

* The collations functions have been extended significantly. (1) `collate_batch_as_dicts` collates a list of tuples to
dictionaries. (2) `collate_sequences_as_tuples` collates list of tuples containing sequences to tuples, while (3)
`collate_sequences_as_dict` produces dictionaries instead. Furthermore, collating numpy arrays is now faster.
* The pipeline creator classes now optionally accept a function that provides additional components for a sample.
This way additional components (e.g., labels stored in a single annotation file) can be added seamlessly to the pipeline.
* Added a more memory-efficient alternative shuffling method for sequence pipelines.
* In case of decoding errors ImageDecoder now informs the user which file caused the error.
This helps to identify broken image files.
Alternatively, you can choose to decode incomplete images and train with (some) garbage data.
Not recommended. ;-)
* The Dispatcher class is now able to work on dictionaries as well as tuples.
This way it can be used to apply per-component transformations to samples even after the user has converted them to dictionaries.

## 1.7.1

* Reverted back to setup.py to restore support for Ubuntu <23.04

## 1.7.0

* Added missing documentation for SequencePipelineCreator
* Added option to allow decoding of incomplete (truncated) image files

## 1.6.2

* Fixed filetree creation on Windows
* Added support for TAR archives as source format in filetree creation
* Updated readme

## 1.6.1

* Switched from setup.py to pyproject.toml
* Fixed original paths in filetree during metadata creation
* Dropped check for file extension in ImageDecoder
* Updated documentation

## 1.6.0

* Metadata for mounting a dataset via datapipeFS is now created automatically during dataset conversion
* Support for querying the original paths of sample components has been removed
* Added support for reading datasets consisting of a single shard with multiple DataLoader workers
* Minor issues in the code have been fixed which in some cases could have caused problems in the future
* Updated readme to provide more information on dependencies and installation

## 1.5.1

* Empty component maps are now accepted, resulting in all components being stored together
* Never skip the root input directory when using path filter functions
* Updated the container compatibility table

## 1.5.0

* Improved shard size suggestions for datasets with short sequences
* Fixed path handling in dataset conversion code on Windows
* Added progress bars to the ArchiveProcessor 

## 1.4.1

* Fixed formatting issues
* Relaxed version requirements on torch and torchdata
* Updated documentation

## 1.4.0

* Integrated a custom grouping implementation that can handle samples with varying number of components better and provides additional features
* Added option to specify the minimum number of components each samples is supposed to have
* Added option to choose whether you wish to receive an error if a sample does not have this minimum number of components or have such samples dropped silently

## 1.3.2

* Fixed default value for setting the minimally required number of components during file grouping 

## 1.3.1

* Added option to manually specify the minimal number of components that each component is expected to have

## 1.3.0

* Added option to limit the number of samples emitted by a pipeline
* Added script for pre-processing files in dataset archives and storing the results in zip archives

## 1.2.1

* Fixed computation of reading efficiency

## 1.2.0

* Added the possibility to skip entire directories and archives when converting datasets
* Stabilized order of component groups (and hence pipes) during dataset streaming
* Stabilized order of components during writing of metadata
* Do not check for duplicates across subset boundaries by default during dataset conversion

## 1.1.0

* Improved test for duplicate samples and provide more details in case duplicates are found
* Updated FaceSynthetics example to work with new datapipes code
* Updated documentation

## 1.0.2

* Fixed a corner case in sample to shard assignment
* Updated readme
* Fixed issue regarding type hints and Python versions below 3.9 in pointcloud_decoder.py

## 1.0.1

* Fixed image decoder failing to decode 16 bit single channel images
* Fixed component filtering in nuscenes example
* Switched to using ImageDecoder instead of ImageHandler in nuscenes example

## 1.0.0

* Changed which information is encoded into the dataset infos
* Split the write_ds_info() function into encode_ds_info() and write_ds_info()
* Extended the dataset conversion template to also cover sequence data (i.e. when multiple consecutive frames are needed for training)
* Added an improved image decoder which adds support for WebP and 16-bit single channel images
* Removed the GrayscaleDecoder and replaced its use with the new ImageDecoder 

## 0.2.1

* Fixed some type hints 

## 0.2.0

* Added more hints on how to create the component map
* Print version number in dataset conversion template
* Provide a relative path to the user supplied function that converts paths to FileInfo objects
* Added template for in-memory datasets

## 0.1.1:

* Fixed crash if no tar file size is set
* Use correct units (MiB, GiB) for file sizes when printing messages
* Fixed dataset conversion template to not exclude all files by default
* Removed confusing example in PathToFilerConvert.\_\_init\_\_
* Added comments to fields in PathToFilerConvert
* Fixed import of BasePipeline creator in template and UnrealEgo example
* Fixed parts of documentation that were not up-to-date

## 0.1.0:

* Initial release

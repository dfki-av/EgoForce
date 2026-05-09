import os.path

from argparse import ArgumentParser

from datapipes import dataset_converter


def main(out_path: str):
    # TODO fill in the information about the dataset
    ds_info = dataset_converter.encode_ds_info(
        short_name="",      # short name
        full_name="",       # full name of the dataset
        sensors=[""],       # type of sensor(s), e.g. "rgb", "depth"
        camera_setup="",    # e.g. "mono", "stereo", "matrix", or "custom"
        nature_of_data="",  # e.g. "synthetic", "real", "mixed"
        tasks=[""],         # e.g. "image classification"
        project_page="",    # e.g. "https://www.example.com/dataset"
        code_repo="",       # e.g. "https://github.com/example"
        paper_url="",       # e.g. "https://www.example.com/paper"
        license_name="",    # e.g. "GPL", "MIT", "3-BSD", "non-commercial"
        commercial_use="",  # whether commercial use is allowed or not, "yes" or "no" (or "unknown")
        converted_by=[""])  # name of the person that sharded the dataset
    infos = [ds_info]  # if multiple datasets are stored together you can provide multiple ds_info objects here
    dataset_converter.write_ds_info(infos, out_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--out_dir",
                        type=str,
                        default="",
                        help="The path to the directory where the dataset info file will be written to.")
    args = parser.parse_args()
    out_dir = os.path.realpath("~") if args.out_dir == "" else args.out_dir
    main(out_dir)

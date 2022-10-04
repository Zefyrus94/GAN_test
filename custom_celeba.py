#si ripetono in train.py
import os
from os import listdir
from os.path import isfile, join
#fine train.py
from collections import namedtuple
from typing import Any, Callable, List, Optional, Union, Tuple, Iterator
import PIL
#
from torchvision.datasets.utils import download_file_from_google_drive, check_integrity, verify_str_arg, extract_archive
from torchvision.datasets.vision import VisionDataset
import requests
import itertools
import re
#per ora salvo in root, ho solo un dataset
class CelebACustom(VisionDataset):
    file_list = [
        # File ID                           	MD5 Hash 	Filename
        ("10kVEvYnNgrp1leZYqttXx81RyP4HhV2d", 	None, 		"reduced_celeba.zip"),
    ]
    def __init__(
        self,
        root: str = 'data',#Downloads/datasets
        transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform)
        if download:
            self.download()
        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")
        
        self.total = 60000
        self.filename = [f for f in listdir(self.root) if isfile(join(self.root, f))]
    def _check_integrity(self) -> bool:
        for (_, md5, filename) in self.file_list:
            fpath = os.path.join(self.root, filename)
            _, ext = os.path.splitext(filename)
            check_integrity(fpath, md5)
            if ext not in [".zip", ".7z"] and not check_integrity(fpath, md5):
                return False
        return os.path.isdir(self.root)
    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        for (file_id, md5, filename) in self.file_list:
            print("Downloading file from Google Drive...")
            download_file_from_google_drive_copy(file_id, self.root, filename, md5)
        print("Extracting file...")
        extract_archive(os.path.join(self.root, "reduced_celeba.zip"))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = PIL.Image.open(os.path.join(self.root, self.filename[index]))
        if self.transform is not None:
            X = self.transform(X)
        return X,X

    def __len__(self) -> int:
        return self.total
#debug download_file_from_google_drive scarica male lo zip
def download_file_from_google_drive_copy(file_id: str, root: str, filename: Optional[str] = None, md5: Optional[str] = None):
    """Download a Google Drive file from  and place it in root.
    Args:
        file_id (str): id of file to be downloaded
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the id of the file.
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    # Based on https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    #print("download_file_from_google_drive2")
    root = os.path.expanduser(root)
    if not filename:
        filename = file_id
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    if check_integrity(fpath, md5):
        print(f"Using downloaded {'and verified ' if md5 else ''}file: {fpath}")
        return

    url = "https://drive.google.com/uc"
    params = dict(id=file_id, export="download")
    with requests.Session() as session:
        print("url",url,"params",params)
        response = session.get(url, params=params, stream=True)

        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                token = value
                break
        else:
            api_response, content = _extract_gdrive_api_response(response)
            token = "t" if api_response == "Virus scan warning" else None

        if token is not None:
            response = session.get(url, params=dict(params, confirm=token), stream=True)
            api_response, content = _extract_gdrive_api_response(response)

        if api_response == "Quota exceeded":
            raise RuntimeError(
                f"The daily quota of the file {filename} is exceeded and it "
                f"can't be downloaded. This is a limitation of Google Drive "
                f"and can only be overcome by trying again later."
            )

        _save_response_content(content, fpath)

    # In case we deal with an unhandled GDrive API response, the file should be smaller than 10kB and contain only text
    if os.stat(fpath).st_size < 10 * 1024:
        with contextlib.suppress(UnicodeDecodeError), open(fpath) as fh:
            text = fh.read()
            # Regular expression to detect HTML. Copied from https://stackoverflow.com/a/70585604
            if re.search(r"</?\s*[a-z-][^>]*\s*>|(&(?:[\w\d]+|#\d+|#x[a-f\d]+);)", text):
                warnings.warn(
                    f"We detected some HTML elements in the downloaded file. "
                    f"This most likely means that the download triggered an unhandled API response by GDrive. "
                    f"Please report this to torchvision at https://github.com/pytorch/vision/issues including "
                    f"the response:\n\n{text}"
                )

    if md5 and not check_md5(fpath, md5):
        raise RuntimeError(
            f"The MD5 checksum of the download file {fpath} does not match the one on record."
            f"Please delete the file and try again. "
            f"If the issue persists, please report this to torchvision at https://github.com/pytorch/vision/issues."
        )
def _extract_gdrive_api_response(response, chunk_size: int = 32 * 1024) -> Tuple[bytes, Iterator[bytes]]:
    content = response.iter_content(chunk_size)
    first_chunk = None
    # filter out keep-alive new chunks
    while not first_chunk:
        first_chunk = next(content)
    content = itertools.chain([first_chunk], content)

    try:
        match = re.search("<title>Google Drive - (?P<api_response>.+?)</title>", first_chunk.decode())
        api_response = match["api_response"] if match is not None else None
    except UnicodeDecodeError:
        api_response = None
    return api_response, content    
def _save_response_content(
    content: Iterator[bytes],
    destination: str,
    length: Optional[int] = None,
) -> None:
    with open(destination, "wb") as fh, tqdm(total=length) as pbar:
        for chunk in content:
            # filter out keep-alive new chunks
            if not chunk:
                continue

            fh.write(chunk)
            pbar.update(len(chunk))
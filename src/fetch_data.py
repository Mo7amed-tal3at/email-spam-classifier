import urllib.request
import tarfile
from pathlib import Path

def fetch_data():
    data_url = "https://spamassassin.apache.org/old/publiccorpus/"
    ham_url = data_url+"20030228_easy_ham.tar.bz2"
    spam_url = data_url + "20030228_spam.tar.bz2"
    data_path = Path() / "data" / "raw"
    data_path.mkdir(parents=True , exist_ok=True)
    for dir_name , tar_name , url in (("easy_ham","ham",ham_url),("spam","spam",spam_url)):
        if not (data_path / dir_name).is_dir():
            path = (data_path/tar_name).with_suffix(".tar.bz2")
            urllib.request.urlretrieve(url , path)
            tar_file = tarfile.open(path)
            tar_file.extractall(data_path)
            tar_file.close()
    
if __name__ == "__main__":
    fetch_data()
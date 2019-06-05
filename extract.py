import tarfile


def extract(filename):
    print('Extracting {}...'.format(filename))
    tar = tarfile.open(filename, 'r')
    tar.extractall('data')
    tar.close()


if __name__ == "__main__":
    extract('data/train-clean-100.tar.gz')
    extract('data/train-clean-360.tar.gz')
    extract('data/train-other-500.tar.gz')
    extract('data/dev-clean.tar.gz')
    extract('data/dev-other.tar.gz')
    extract('data/test-clean.tar.gz')
    extract('data/test-other.tar.gz')

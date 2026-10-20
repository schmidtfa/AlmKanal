"""Check that an invalid step sequence is rejected without processing data."""

from almkanal import AlmKanal, Epochs, Resample


def main() -> None:
    try:
        AlmKanal([Resample(sfreq=200), Epochs()])
    except ValueError as error:
        if 'Dependency Error' not in str(error):
            raise AssertionError('A different error occurred.') from error
        print(f'Expected ordering error:\n{error}')
    else:
        raise AssertionError('Expected construction to reject this order.')


if __name__ == '__main__':
    main()

from quelfilm.nlp.load import MdLoader
from quelfilm.nlp.preprocessing import Preprocessing
from quelfilm.nlp.representation import MergedMatrixRepresentation


def main():
    loader = MdLoader('./training.md')
    preprocessor = Preprocessing(loader)

    all_data = preprocessor.data

    test_data = None
    train_data = None

    data_representation = MergedMatrixRepresentation()


if __name__ == '__main__':
    main()

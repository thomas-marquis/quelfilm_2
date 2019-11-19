from quelfilm import utils
from quelfilm.nlp.load import MdLoader
from quelfilm.nlp.preprocessing import Preprocessing
from quelfilm.nlp.representation import MergedMatrixRepresentation
from quelfilm.nlp.classifier import ClassificationProcessor, NaiveBayseTfIdfClassifier


_processor: Preprocessing = None
_data_repr: MergedMatrixRepresentation = None
_classifier_processor: ClassificationProcessor = None


def _start_app():
    loader = MdLoader('./training.md')
    _processor = Preprocessing(loader)
    _data_repr = MergedMatrixRepresentation(_processor.data)
    classifier = NaiveBayseTfIdfClassifier()
    _classifier_processor = ClassificationProcessor(classifier, _data_repr)
    _classifier_processor.train()


def classify(message: str) -> str:
    msg = _data_repr.process_new_data(
        _processor.process_sentence(message))
    intent, score = _classifier_processor.predict(msg)
    print(score)

    return intent
